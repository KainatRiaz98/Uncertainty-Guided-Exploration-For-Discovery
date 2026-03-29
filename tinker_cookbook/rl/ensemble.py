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
        learning_rate: float = 4e-5,
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
        kv_quantize: bool = False,
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
        mlora_data.kv_cache_quantize_ = kv_quantize
        return mlora_data

    def _make_gen_data_batch(
        self,
        token_ids_batch: List[List[int]],
        adapter_name: str,
        kv_cache: Optional[List] = None,
        cache_position: int = 0,
        kv_quantize: bool = False,
    ) -> MLoRAData:
        """Build MLoRAData for batched generation (N sequences, same adapter)."""
        N = len(token_ids_batch)
        data_config = MLoRADataConfig(
            adapter_name=adapter_name,
            adapter_type="lora",
            start_idx=0,
            end_idx=N,
            expand_fn=self._expand_fn,
            loss_fn=self._noop_loss,
            task_name="generation_batch",
        )
        mlora_data = MLoRAData(
            batch_tokens=token_ids_batch,
            batch_mask=[[True] * len(toks) for toks in token_ids_batch],
            data_config=[data_config],
        )
        mlora_data.use_flash_causal_ = True
        mlora_data.kv_cache_ = kv_cache
        mlora_data.cache_position_ = cache_position
        mlora_data.kv_cache_quantize_ = kv_quantize
        return mlora_data

    def _make_gen_data_multi_adapter(
        self,
        token_ids_batch: List[List[int]],
        adapter_configs: List[Tuple[str, int, int]],
        kv_cache: Optional[List] = None,
        cache_position: int = 0,
        kv_quantize: bool = False,
    ) -> MLoRAData:
        """Build MLoRAData for multi-adapter generation (N sequences, multiple adapters).

        Each adapter_config is (adapter_name, start_idx, end_idx) specifying which
        contiguous batch range uses which adapter.
        """
        data_configs = [
            MLoRADataConfig(
                adapter_name=name,
                adapter_type="lora",
                start_idx=start,
                end_idx=end,
                expand_fn=self._expand_fn,
                loss_fn=self._noop_loss,
                task_name=f"gen_multi_{name}",
            )
            for name, start, end in adapter_configs
        ]
        mlora_data = MLoRAData(
            batch_tokens=token_ids_batch,
            batch_mask=[[True] * len(toks) for toks in token_ids_batch],
            data_config=data_configs,
        )
        mlora_data.use_flash_causal_ = True
        mlora_data.kv_cache_ = kv_cache
        mlora_data.cache_position_ = cache_position
        mlora_data.kv_cache_quantize_ = kv_quantize
        return mlora_data

    @torch.no_grad()
    def generate(
        self,
        prompt_tokens: List[int],
        max_tokens: int,
        temperature: float = 1.0,
        eos_token_id: int = 2,
        adapter_idx: int = 0,
        kv_quantize: bool = False,
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
            kv_quantize: If True, store KV cache in int8 (halves memory bandwidth).

        Returns:
            generated_tokens: Full sequence (prompt + generated).
            logprobs: Log-probabilities of each generated token.
        """
        adapter_name = f"ensemble_{adapter_idx}"
        n_layers = self._get_n_layers()
        kv_cache: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * n_layers

        # ── Prefill: process entire prompt, populate KV cache ──
        mlora_data = self._make_gen_data(
            list(prompt_tokens), adapter_name,
            kv_cache=kv_cache, cache_position=0, kv_quantize=kv_quantize,
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
                [next_token], adapter_name,
                kv_cache=kv_cache, cache_position=cur_pos, kv_quantize=kv_quantize,
            )
            logits = self.model.forward(mlora_data.model_data())  # (1, 1, V)
            next_logits = logits[0, -1, :]
            cur_pos += 1

        return tokens, gen_logprobs

    @torch.no_grad()
    def generate_batch(
        self,
        prompt_tokens: List[int],
        num_sequences: int,
        max_tokens: int,
        temperature: float = 1.0,
        eos_token_id: int = 2,
        adapter_idx: int = 0,
        kv_quantize: bool = False,
    ) -> List[Tuple[List[int], List[float]]]:
        """
        Batched autoregressive generation: N sequences from the same prompt
        in parallel via batched forward passes with shared KV cache prefix.

        ~N× faster than calling generate() N times sequentially because
        the GPU processes all N next-tokens in a single forward pass.

        Args:
            prompt_tokens: Tokenized prompt (shared by all sequences).
            num_sequences: Number of sequences to generate (N).
            max_tokens: Maximum new tokens per sequence.
            temperature: Sampling temperature.
            eos_token_id: End-of-sequence token ID.
            adapter_idx: Which ensemble member to use.
            kv_quantize: If True, store KV cache in int8 (halves bandwidth).

        Returns:
            List of (full_tokens, logprobs) tuples, one per sequence.
        """
        if num_sequences == 1:
            result = self.generate(
                prompt_tokens, max_tokens, temperature, eos_token_id, adapter_idx,
                kv_quantize=kv_quantize,
            )
            return [result]

        adapter_name = f"ensemble_{adapter_idx}"
        n_layers = self._get_n_layers()
        N = num_sequences

        # ── Prefill: process prompt once, then expand KV cache to batch N ──
        # Prefill without quantization for accuracy, quantize after expansion
        kv_cache_single: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * n_layers
        mlora_data = self._make_gen_data(
            list(prompt_tokens), adapter_name,
            kv_cache=kv_cache_single, cache_position=0,
        )
        logits = self.model.forward(mlora_data.model_data())  # (1, P, V)
        next_logits = logits[0, -1, :]  # (V,)

        # Expand KV cache from batch=1 to batch=N by repeating
        kv_cache_batch: List = []
        for layer_cache in kv_cache_single:
            if layer_cache is not None:
                k, v = layer_cache  # (1, n_kv_head, seq_len, head_dim)
                k_expanded = k.expand(N, -1, -1, -1).contiguous()
                v_expanded = v.expand(N, -1, -1, -1).contiguous()
                if kv_quantize:
                    # Quantize to int8 with per-channel scales
                    k_absmax = k_expanded.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
                    v_absmax = v_expanded.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
                    k_scale = k_absmax / 127.0
                    v_scale = v_absmax / 127.0
                    kv_cache_batch.append((
                        (k_expanded / k_scale).round().to(torch.int8),
                        (v_expanded / v_scale).round().to(torch.int8),
                        k_scale, v_scale,
                    ))
                else:
                    kv_cache_batch.append((k_expanded, v_expanded))
            else:
                kv_cache_batch.append(None)

        # Initialize per-sequence state
        all_tokens: List[List[int]] = [list(prompt_tokens) for _ in range(N)]
        all_logprobs: List[List[float]] = [[] for _ in range(N)]
        finished = [False] * N
        cur_pos = len(prompt_tokens)

        # Sample first token for all N sequences from shared prefill logits
        log_probs_all = F.log_softmax(next_logits, dim=-1)  # (V,)
        if temperature > 0:
            probs = F.softmax(next_logits / temperature, dim=-1)
            first_tokens = torch.multinomial(probs.unsqueeze(0).expand(N, -1), num_samples=1).squeeze(-1)  # (N,)
        else:
            first_tokens = next_logits.argmax().unsqueeze(0).expand(N)

        next_token_ids: List[List[int]] = []
        for i in range(N):
            tok = first_tokens[i].item()
            lp = log_probs_all[tok].item()
            all_tokens[i].append(tok)
            all_logprobs[i].append(lp)
            if tok == eos_token_id:
                finished[i] = True
            next_token_ids.append([tok])

        # ── Batched decode loop ──
        for step in range(1, max_tokens):
            if all(finished):
                break

            # Build batch of next tokens (finished sequences get a dummy token)
            batch_tokens = []
            for i in range(N):
                if finished[i]:
                    batch_tokens.append([eos_token_id])  # dummy, output ignored
                else:
                    batch_tokens.append([all_tokens[i][-1]])

            mlora_data = self._make_gen_data_batch(
                batch_tokens, adapter_name,
                kv_cache=kv_cache_batch, cache_position=cur_pos,
                kv_quantize=kv_quantize,
            )
            logits = self.model.forward(mlora_data.model_data())  # (N, 1, V)
            cur_pos += 1

            # Sample next token for each active sequence
            for i in range(N):
                if finished[i]:
                    continue

                seq_logits = logits[i, -1, :]  # (V,)
                log_probs_i = F.log_softmax(seq_logits, dim=-1)

                if temperature > 0:
                    probs_i = F.softmax(seq_logits / temperature, dim=-1)
                    next_tok = torch.multinomial(probs_i, num_samples=1).item()
                else:
                    next_tok = seq_logits.argmax().item()

                all_tokens[i].append(next_tok)
                all_logprobs[i].append(log_probs_i[next_tok].item())

                if next_tok == eos_token_id:
                    finished[i] = True

        return list(zip(all_tokens, all_logprobs))

    # ── Multi-adapter batched generation ─────────────────────────────────

    @torch.no_grad()
    def generate_batch_multi_adapter(
        self,
        prompt_tokens: List[int],
        adapter_assignments: List[int],
        max_tokens: int,
        temperature: float = 1.0,
        eos_token_id: int = 2,
        kv_quantize: bool = False,
    ) -> List[Tuple[List[int], List[float]]]:
        """
        Batched autoregressive generation where different sequences use
        different adapters, all in a single forward pass per decode step.

        Leverages mLoRA's native multi-adapter routing: each batch element
        is assigned to an adapter via MLoRADataConfig, and the LoRA forward
        applies the correct weights per batch slice.

        ~K× faster than calling generate_batch() once per adapter because:
          - One prefill pass (K adapters batched) instead of K separate prefills
          - One decode loop with N elements instead of K loops with N/K elements

        Args:
            prompt_tokens: Tokenized prompt (shared by all sequences).
            adapter_assignments: adapter_idx for each of N sequences.
                E.g. [0, 1, 2, 3, 4, 0, 1, 2] for group_size=8, K=5.
            max_tokens: Maximum new tokens per sequence.
            temperature: Sampling temperature.
            eos_token_id: End-of-sequence token ID.
            kv_quantize: If True, store KV cache in int8.

        Returns:
            List of (full_tokens, logprobs) tuples in the ORIGINAL order
            of adapter_assignments (not grouped by adapter).
        """
        N = len(adapter_assignments)

        # ── Fast path: single adapter → delegate to existing method ──
        unique_adapters = set(adapter_assignments)
        if len(unique_adapters) == 1:
            return self.generate_batch(
                prompt_tokens, N, max_tokens, temperature,
                eos_token_id, adapter_idx=adapter_assignments[0],
                kv_quantize=kv_quantize,
            )
        if N == 1:
            result = self.generate(
                prompt_tokens, max_tokens, temperature,
                eos_token_id, adapter_idx=adapter_assignments[0],
                kv_quantize=kv_quantize,
            )
            return [result]

        n_layers = self._get_n_layers()

        # ── Phase 1: Group sequences by adapter for contiguous batch ranges ──
        # mLoRA requires data[start_idx:end_idx] to be contiguous per adapter.
        adapter_groups: Dict[int, List[int]] = {}
        for i, a in enumerate(adapter_assignments):
            adapter_groups.setdefault(a, []).append(i)
        sorted_adapters = sorted(adapter_groups.keys())

        batch_to_orig: List[int] = []     # batch_pos → original_pos
        adapter_configs: List[Tuple[str, int, int]] = []  # (name, start, end)
        batch_pos = 0
        for k in sorted_adapters:
            positions = adapter_groups[k]
            n_seqs = len(positions)
            batch_to_orig.extend(positions)
            adapter_configs.append((f"ensemble_{k}", batch_pos, batch_pos + n_seqs))
            batch_pos += n_seqs

        # ── Phase 2: Batched multi-adapter prefill ──
        # One forward pass with num_unique batch elements, each routing to its adapter.
        num_unique = len(sorted_adapters)
        prefill_kv_cache: List = [None] * n_layers
        prefill_configs = [
            (f"ensemble_{k}", idx, idx + 1)
            for idx, k in enumerate(sorted_adapters)
        ]
        prefill_data = self._make_gen_data_multi_adapter(
            token_ids_batch=[list(prompt_tokens)] * num_unique,
            adapter_configs=prefill_configs,
            kv_cache=prefill_kv_cache,
            cache_position=0,
            kv_quantize=False,  # don't quantize yet; quantize after expansion
        )
        logits = self.model.forward(prefill_data.model_data())  # (num_unique, P, V)

        # Extract per-adapter last-token logits
        adapter_prefill_logits: Dict[int, torch.Tensor] = {}
        for idx, k in enumerate(sorted_adapters):
            adapter_prefill_logits[k] = logits[idx, -1, :]  # (V,)
        del logits

        # ── Phase 3: Expand KV cache from num_unique → N sequences ──
        # When quantizing, quantize the K unique caches FIRST (K × fp16 → K × int8),
        # then expand the int8+scale tensors to N. Peak memory: K × fp16 + N × int8
        # instead of N × fp16 + N × int8.
        kv_cache_batch: List = []
        for layer_cache in prefill_kv_cache:
            if layer_cache is None:
                kv_cache_batch.append(None)
                continue
            cached_k, cached_v = layer_cache  # (num_unique, n_kv_heads, P, head_dim)

            if kv_quantize:
                # Quantize K unique rows in fp16 first
                k_absmax = cached_k.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
                v_absmax = cached_v.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
                k_scale = k_absmax / 127.0
                v_scale = v_absmax / 127.0
                k_q = (cached_k / k_scale).round().to(torch.int8)  # (num_unique, ...)
                v_q = (cached_v / v_scale).round().to(torch.int8)
                del cached_k, cached_v  # free fp16 immediately

                # Expand int8 + scales from num_unique → N
                expanded_kq, expanded_vq = [], []
                expanded_ks, expanded_vs = [], []
                for idx, k in enumerate(sorted_adapters):
                    n_seqs = len(adapter_groups[k])
                    expanded_kq.append(k_q[idx:idx + 1].expand(n_seqs, -1, -1, -1).contiguous())
                    expanded_vq.append(v_q[idx:idx + 1].expand(n_seqs, -1, -1, -1).contiguous())
                    expanded_ks.append(k_scale[idx:idx + 1].expand(n_seqs, -1, -1, -1).contiguous())
                    expanded_vs.append(v_scale[idx:idx + 1].expand(n_seqs, -1, -1, -1).contiguous())

                kv_cache_batch.append((
                    torch.cat(expanded_kq, dim=0),
                    torch.cat(expanded_vq, dim=0),
                    torch.cat(expanded_ks, dim=0),
                    torch.cat(expanded_vs, dim=0),
                ))
            else:
                expanded_k_parts = []
                expanded_v_parts = []
                for idx, k in enumerate(sorted_adapters):
                    n_seqs = len(adapter_groups[k])
                    expanded_k_parts.append(
                        cached_k[idx:idx + 1].expand(n_seqs, -1, -1, -1).contiguous()
                    )
                    expanded_v_parts.append(
                        cached_v[idx:idx + 1].expand(n_seqs, -1, -1, -1).contiguous()
                    )
                kv_cache_batch.append((
                    torch.cat(expanded_k_parts, dim=0),
                    torch.cat(expanded_v_parts, dim=0),
                ))
        del prefill_kv_cache

        # ── Phase 4: Sample first token for all N sequences ──
        all_tokens: List[List[int]] = [list(prompt_tokens) for _ in range(N)]
        all_logprobs: List[List[float]] = [[] for _ in range(N)]
        finished = [False] * N
        cur_pos = len(prompt_tokens)

        for name, start, end in adapter_configs:
            k = int(name.split("_")[1])
            next_logits_k = adapter_prefill_logits[k]
            log_probs_k = F.log_softmax(next_logits_k, dim=-1)
            n_seqs = end - start

            if temperature > 0:
                probs_k = F.softmax(next_logits_k / temperature, dim=-1)
                first_tokens_k = torch.multinomial(
                    probs_k.unsqueeze(0).expand(n_seqs, -1), num_samples=1
                ).squeeze(-1)
            else:
                first_tokens_k = next_logits_k.argmax().unsqueeze(0).expand(n_seqs)

            for j in range(n_seqs):
                batch_pos = start + j
                tok = first_tokens_k[j].item()
                lp = log_probs_k[tok].item()
                all_tokens[batch_pos].append(tok)
                all_logprobs[batch_pos].append(lp)
                if tok == eos_token_id:
                    finished[batch_pos] = True
        del adapter_prefill_logits

        # ── Phase 5: Decode loop (multi-adapter forward per step) ──
        for step in range(1, max_tokens):
            if all(finished):
                break

            batch_tokens = []
            for i in range(N):
                if finished[i]:
                    batch_tokens.append([eos_token_id])
                else:
                    batch_tokens.append([all_tokens[i][-1]])

            mlora_data = self._make_gen_data_multi_adapter(
                token_ids_batch=batch_tokens,
                adapter_configs=adapter_configs,
                kv_cache=kv_cache_batch,
                cache_position=cur_pos,
                kv_quantize=kv_quantize,
            )
            logits = self.model.forward(mlora_data.model_data())  # (N, 1, V)
            cur_pos += 1

            for i in range(N):
                if finished[i]:
                    continue
                seq_logits = logits[i, -1, :]
                log_probs_i = F.log_softmax(seq_logits, dim=-1)
                if temperature > 0:
                    probs_i = F.softmax(seq_logits / temperature, dim=-1)
                    next_tok = torch.multinomial(probs_i, num_samples=1).item()
                else:
                    next_tok = seq_logits.argmax().item()
                all_tokens[i].append(next_tok)
                all_logprobs[i].append(log_probs_i[next_tok].item())
                if next_tok == eos_token_id:
                    finished[i] = True

        # ── Phase 6: Reorder results from batch order → original order ──
        results_batch_order = list(zip(all_tokens, all_logprobs))
        results: List[Optional[Tuple[List[int], List[float]]]] = [None] * N
        for batch_pos, orig_pos in enumerate(batch_to_orig):
            results[orig_pos] = results_batch_order[batch_pos]
        return results

    # ── Batched ensemble scoring ─────────────────────────────────────────

    def compute_ensemble_logprobs_batch(
        self,
        token_ids_list: List[List[int]],
        chunk_size: int = 256,
    ) -> List[torch.Tensor]:
        """
        Score multiple sequences through all K adapters, batching sequences
        that share the same length for efficient GPU utilization.

        Args:
            token_ids_list: List of token sequences to score.
            chunk_size: Tokens per chunk for LM head.

        Returns:
            List of (K, T_i-1) tensors, one per input sequence.
        """
        if len(token_ids_list) == 1:
            return [self.compute_ensemble_logprobs(token_ids_list[0], chunk_size)]

        # Group by length for efficient batching (avoids excessive padding)
        length_groups: Dict[int, List[int]] = {}
        for idx, tids in enumerate(token_ids_list):
            length_groups.setdefault(len(tids), []).append(idx)

        results: List[Optional[torch.Tensor]] = [None] * len(token_ids_list)

        for length, indices in length_groups.items():
            # Build a combined batch: len(indices) sequences × K adapters
            # = len(indices)*K batch elements, all same length
            batch_tokens = []
            batch_masks = []
            data_configs = []

            for seq_pos, orig_idx in enumerate(indices):
                tids = token_ids_list[orig_idx]
                for k in range(self.K):
                    batch_idx = seq_pos * self.K + k
                    batch_tokens.append(list(tids))
                    batch_masks.append([True] * length)
                    data_configs.append(MLoRADataConfig(
                        adapter_name=f"ensemble_{k}",
                        adapter_type="lora",
                        start_idx=batch_idx,
                        end_idx=batch_idx + 1,
                        expand_fn=self._expand_fn,
                        loss_fn=self._noop_loss,
                        task_name=f"batch_score_{orig_idx}_{k}",
                    ))

            mlora_data = MLoRAData(
                batch_tokens=batch_tokens,
                batch_mask=batch_masks,
                data_config=data_configs,
            )
            mlora_data.use_flash_causal_ = True
            mlora_data.return_hidden_states_ = True

            with torch.no_grad():
                hidden = self.model.forward(mlora_data.model_data())
                # hidden: (len(indices)*K, T, D)

            # Extract per-sequence, per-adapter logprobs
            lm_head = self._get_lm_head()
            for seq_pos, orig_idx in enumerate(indices):
                tids = token_ids_list[orig_idx]
                T = len(tids)
                target = torch.tensor(tids[1:], dtype=torch.long, device=hidden.device)

                # Gather hidden states for this sequence's K adapters
                adapter_hidden = hidden[seq_pos * self.K:(seq_pos + 1) * self.K]  # (K, T, D)

                # Chunked logprob extraction (same as _chunked_logprobs)
                chunks = []
                for start in range(0, T - 1, chunk_size):
                    end = min(start + chunk_size, T - 1)
                    chunk_logits = lm_head(adapter_hidden[:, start:end, :]).float()
                    chunk_lp = F.log_softmax(chunk_logits, dim=-1)
                    chunk_target = target[start:end].unsqueeze(0).unsqueeze(-1).expand(self.K, -1, 1)
                    chunks.append(chunk_lp.gather(2, chunk_target).squeeze(-1))
                    del chunk_logits, chunk_lp

                results[orig_idx] = torch.cat(chunks, dim=1)  # (K, T-1)

        return results

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
