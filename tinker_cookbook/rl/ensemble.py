"""
LoRA Ensemble Manager — wraps mLoRA for multi-adapter lifecycle.

Handles:
    - Initializing K LoRA adapters with different random seeds
    - Batched forward passes for ensemble scoring (single pass, all K adapters)
    - Per-adapter logprob extraction
    - Autoregressive text generation from a single adapter
    - Optimizer stepping for all K adapters
    - Checkpoint save/load

This module is the bridge between mLoRA and the RL training loop.
The training loop (mlora_train.py) should never call mLoRA directly.
"""

import logging
import math
import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from mlora.model.llm import LLMModel
from mlora.model.args import (
    LinearInfo,
    MLoRAData,
    MLoRADataConfig,
    Tokens,
    Masks,
)
from mlora.config import LoRAConfig
from mlora.executor.context.lora import TrainLoRAContext

logger = logging.getLogger(__name__)


class LoRAEnsemble:
    """
    Manages K LoRA adapters on a shared base model via mLoRA.

    Each adapter is initialized with a different random seed (Kaiming normal),
    creating a "chorus of experts" that represents different hypotheses about
    optimal parameters — the basis for Bayesian epistemic uncertainty estimation.
    """

    def __init__(
        self,
        model: LLMModel,
        num_members: int = 5,
        lora_rank: int = 32,
        lora_alpha: int = 64,
        lora_dropout: float = 0.05,
        target_modules: Optional[Dict[str, bool]] = None,
        learning_rate: float = 3e-4,
        optimizer: str = "adamw",
    ):
        self.model = model
        self.K = num_members
        self.contexts: List[TrainLoRAContext] = []

        if target_modules is None:
            target_modules = {
                "q_proj": True, "o_proj": True,
            }

        self._init_adapters(
            lora_rank, lora_alpha, lora_dropout,
            target_modules, learning_rate, optimizer,
        )

    def _init_adapters(
        self,
        rank: int,
        alpha: int,
        dropout: float,
        target_modules: Dict[str, bool],
        lr: float,
        optimizer: str,
    ):
        """Create K LoRA adapters with different random seeds."""
        linears_info = self.model.linears_info()

        for k in range(self.K):
            # Different seed → different Kaiming initialization → different hypothesis
            torch.manual_seed(42 + k * 1000)

            lora_config = LoRAConfig({
                "type": "lora",
                "name": f"ensemble_{k}",
                "path": f"./adapters/ensemble_{k}",
                "r": rank,
                "alpha": alpha,
                "dropout": dropout,
                "target_modules": target_modules,
                "optimizer": optimizer,
                "lr": lr,
                # Match original train.py: AdamParams(beta1=0.9, beta2=0.95, eps=1e-8)
                "beta1": 0.9,
                "beta2": 0.95,
                "weight_decay": 0.0,
            })

            context = TrainLoRAContext(lora_config, linears_info)
            context.switch_device(self.model.device_)
            self.contexts.append(context)
            self.model.load_adapter(context.adapter_model())

        logger.info(f"Initialized {self.K} LoRA ensemble members (rank={rank})")

    # ── LM head access ───────────────────────────────────────────────────

    def _get_lm_head(self) -> torch.nn.Linear:
        """Extract LM head from the model's sequential modules."""
        for module in self.model.sequential():
            if hasattr(module, 'wrapper_module_') and hasattr(module.wrapper_module_, 'lm_head_'):
                return module.wrapper_module_.lm_head_
        raise RuntimeError("Could not find LM head in model")

    # ── Ensemble scoring ──────────────────────────────────────────────────

    def _build_mlora_data(
        self, token_ids: List[int], task_prefix: str
    ) -> MLoRAData:
        """Build MLoRAData for K adapters scoring the same sequence."""
        K = self.K
        T = len(token_ids)
        data_configs = [
            MLoRADataConfig(
                adapter_name=f"ensemble_{k}",
                adapter_type="lora",
                start_idx=k,
                end_idx=k + 1,
                expand_fn=self._expand_fn,
                loss_fn=self._noop_loss,
                task_name=f"{task_prefix}_{k}",
            )
            for k in range(K)
        ]
        mlora_data = MLoRAData(
            batch_tokens=[list(token_ids)] * K,
            batch_mask=[[True] * T] * K,
            data_config=data_configs,
        )
        mlora_data.use_flash_causal_ = True
        return mlora_data

    def _chunked_logprobs(
        self,
        hidden: torch.Tensor,
        token_ids: List[int],
        chunk_size: int = 256,
    ) -> torch.Tensor:
        """
        Extract per-token logprobs from hidden states without materializing
        the full (K, T, V) logits tensor.

        Applies the LM head in chunks over the sequence dimension.
        Peak logit memory: K × chunk_size × V × 4 bytes.
        """
        K = hidden.shape[0]
        T = len(token_ids)
        lm_head = self._get_lm_head()
        target = torch.tensor(token_ids[1:], dtype=torch.long, device=hidden.device)

        results = []
        for start in range(0, T - 1, chunk_size):
            end = min(start + chunk_size, T - 1)
            # hidden[:, start:end] predicts tokens[start+1:end+1]
            chunk_logits = lm_head(hidden[:, start:end, :]).float()  # (K, chunk, V)
            chunk_lp = F.log_softmax(chunk_logits, dim=-1)
            chunk_target = target[start:end].unsqueeze(0).unsqueeze(-1).expand(K, -1, 1)
            results.append(chunk_lp.gather(2, chunk_target).squeeze(-1))
            del chunk_logits, chunk_lp

        return torch.cat(results, dim=1)  # (K, T-1)

    def compute_ensemble_logprobs(
        self,
        token_ids: List[int],
        chunk_size: int = 256,
    ) -> torch.Tensor:
        """
        Score a token sequence through all K adapters in a single forward pass.

        Uses flash causal attention (O(n) memory) and chunked logit extraction
        to avoid materializing the full (K, T, V) logits tensor.

        Args:
            token_ids: Complete token sequence (prompt + generated), length T.
            chunk_size: Tokens per chunk for LM head. Lower = less memory.

        Returns:
            (K, T-1) tensor of log p_k(y_t | y_{<t}) for each adapter k,
            shifted so position t predicts token t+1.
        """
        mlora_data = self._build_mlora_data(token_ids, "score")
        mlora_data.return_hidden_states_ = True

        with torch.no_grad():
            hidden = self.model.forward(mlora_data.model_data())  # (K, T, D)

        return self._chunked_logprobs(hidden, token_ids, chunk_size)

    def compute_training_logprobs(
        self,
        token_ids: List[int],
        chunk_size: int = 256,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Like compute_ensemble_logprobs but WITH gradient tracking.
        Used during the RL training step.

        Returns:
            hidden: (K, T, dim) hidden states with grad attached
            per_token_logprobs: (K, T-1)
        """
        mlora_data = self._build_mlora_data(token_ids, "train")
        mlora_data.return_hidden_states_ = True

        hidden = self.model.forward(mlora_data.model_data())  # (K, T, D)
        per_token_logprobs = self._chunked_logprobs(hidden, token_ids, chunk_size)

        return hidden, per_token_logprobs

    # ── Text generation ───────────────────────────────────────────────────

    def _get_n_layers(self) -> int:
        """Count decoder layers in the model."""
        return sum(1 for m in self.model.sequential() if m.name() == "Decoder")

    def _make_gen_data(
        self,
        token_ids: List[int],
        adapter_name: str,
        kv_cache: Optional[List] = None,
        cache_position: int = 0,
    ) -> MLoRAData:
        """Build MLoRAData for single-adapter generation."""
        data_config = MLoRADataConfig(
            adapter_name=adapter_name,
            adapter_type="lora",
            start_idx=0,
            end_idx=1,
            expand_fn=self._expand_fn,
            loss_fn=self._noop_loss,
            task_name="generation",
        )
        mlora_data = MLoRAData(
            batch_tokens=[token_ids],
            batch_mask=[[True] * len(token_ids)],
            data_config=[data_config],
        )
        mlora_data.use_flash_causal_ = True
        mlora_data.kv_cache_ = kv_cache
        mlora_data.cache_position_ = cache_position
        return mlora_data

    @torch.no_grad()
    def generate(
        self,
        prompt_tokens: List[int],
        max_tokens: int,
        temperature: float = 1.0,
        eos_token_id: int = 2,
        adapter_idx: int = 0,
    ) -> Tuple[List[int], List[float]]:
        """
        Autoregressive generation with KV caching.

        Prefills the KV cache with the prompt in one pass, then decodes
        one token at a time reusing cached K/V — O(P + M) instead of O((P+M)²).

        Args:
            prompt_tokens: Tokenized prompt.
            max_tokens: Maximum new tokens to generate.
            temperature: Sampling temperature.
            eos_token_id: End-of-sequence token ID.
            adapter_idx: Which ensemble member to use for generation.

        Returns:
            generated_tokens: Full sequence (prompt + generated).
            logprobs: Log-probabilities of each generated token.
        """
        adapter_name = f"ensemble_{adapter_idx}"
        n_layers = self._get_n_layers()
        kv_cache: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * n_layers

        # ── Prefill: process entire prompt, populate KV cache ──
        mlora_data = self._make_gen_data(
            list(prompt_tokens), adapter_name, kv_cache=kv_cache, cache_position=0,
        )
        logits = self.model.forward(mlora_data.model_data())  # (1, P, V)
        next_logits = logits[0, -1, :]  # (V,)

        tokens = list(prompt_tokens)
        gen_logprobs: List[float] = []
        cur_pos = len(prompt_tokens)

        # ── Decode: one token at a time, reusing KV cache ──
        for _ in range(max_tokens):
            if temperature > 0:
                probs = F.softmax(next_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
            else:
                next_token = next_logits.argmax().item()

            log_prob = F.log_softmax(next_logits, dim=-1)[next_token].item()
            gen_logprobs.append(log_prob)
            tokens.append(next_token)

            if next_token == eos_token_id:
                break

            # Forward pass with just the new token + KV cache
            mlora_data = self._make_gen_data(
                [next_token], adapter_name, kv_cache=kv_cache, cache_position=cur_pos,
            )
            logits = self.model.forward(mlora_data.model_data())  # (1, 1, V)
            next_logits = logits[0, -1, :]
            cur_pos += 1

        return tokens, gen_logprobs

    # ── Optimizer management ──────────────────────────────────────────────

    def step_all_optimizers(self):
        """Step all K adapter optimizers and zero gradients."""
        for ctx in self.contexts:
            ctx.step()

    # ── Checkpointing ─────────────────────────────────────────────────────

    def save(self, path: str, step: int):
        """Save all adapter weights."""
        for k, ctx in enumerate(self.contexts):
            save_dir = os.path.join(path, f"ensemble_{k}", f"step_{step}")
            os.makedirs(save_dir, exist_ok=True)
            torch.save(ctx.weight_dict(), os.path.join(save_dir, "adapter.pt"))

    def load(self, path: str, step: int):
        """Load adapter weights from checkpoint."""
        for k, ctx in enumerate(self.contexts):
            fpath = os.path.join(path, f"ensemble_{k}", f"step_{step}", "adapter.pt")
            if os.path.exists(fpath):
                ctx.recover_weight(torch.load(fpath, weights_only=True))
                logger.info(f"Loaded ensemble member {k} from {fpath}")

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _expand_fn(
        batch_tokens: List[Tokens], align_len: Optional[int] = None
    ) -> Tuple[List[Tokens], List[Masks]]:
        """Pad tokens to uniform length."""
        if align_len is None:
            align_len = max(len(t) for t in batch_tokens)
        padded_tokens, padded_masks = [], []
        for tokens in batch_tokens:
            pad_len = align_len - len(tokens)
            padded_tokens.append(tokens + [0] * pad_len)
            padded_masks.append([True] * len(tokens) + [False] * pad_len)
        return padded_tokens, padded_masks

    @staticmethod
    def _noop_loss(
        input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> Optional[torch.Tensor]:
        return None
