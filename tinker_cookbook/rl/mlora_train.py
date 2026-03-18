"""
Multi-LoRA Ensemble RL Training Loop for TTT-Discover.

Drop-in replacement for tinker_cookbook/rl/train.py that uses mLoRA instead
of Tinker for model serving, enabling concurrent multi-LoRA training for
Bayesian epistemic uncertainty (RMI exploration bonus).

Mirrors the structure of the original train.py:
    - Config class with ensemble/uncertainty fields
    - compute_advantages() reused from original
    - Rollout → reward → advantages → train step
    - W&B logging, checkpointing, evaluation
    - Sync training loop (async variant can be added later)

New additions over original:
    - LoRA ensemble initialization (K adapters, different seeds)
    - Post-rollout ensemble scoring → RMI computation
    - Reward augmentation: R_total = R_exec + γ * RMI
    - Training all K adapters per step
"""

import argparse
import asyncio
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

# ── mLoRA ──────────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "mLoRA"))
from mlora.model import load_model as mlora_load_model
from mlora.model.tokenizer import Tokenizer as MLoRATokenizer

# ── Our new modules ────────────────────────────────────────────────────────────
from tinker_cookbook.rl.ensemble import LoRAEnsemble
from tinker_cookbook.rl.uncertainty import UNCERTAINTY_FNS

# ── Existing TTT-Discover modules (no Tinker dependency) ──────────────────────
from tinker_cookbook.recipes.ttt.state import State
from tinker_cookbook.recipes.ttt.sampler import StateSampler
from tinker_cookbook.utils.ml_log import setup_logging, Logger as MLLogger

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Config — mirrors tinker_cookbook/rl/train.py:Config with ensemble additions
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    """
    Training configuration.

    Mirrors the original Config in train.py, replacing Tinker-specific
    fields with mLoRA equivalents and adding ensemble/uncertainty fields.
    """

    # ── Model (replaces model_name + Tinker service) ──────────────────────
    base_model: str = "meta-llama/Llama-3.1-8B"
    precision: str = "nf4"
    device: str = "cuda"

    # ── Environment ───────────────────────────────────────────────────────
    env: str = "ac1"
    problem_idx: str = "improvement"
    budget_s: int = 1000

    # ── RL hyperparameters (matching original CLIConfig defaults) ─────────
    group_size: int = 8
    groups_per_batch: int = 64
    learning_rate: float = 4e-5
    num_epochs: int = 50
    max_tokens: int = 26000
    temperature: float = 1.0
    adv_estimator: str = "entropic"
    adv_estimator_beta: float = 2.0
    loss_fn: Literal["importance_sampling", "ppo"] = "importance_sampling"
    ppo_clip: float = 0.2
    num_substeps: int = 1
    remove_constant_reward_groups: bool = True

    # ── KL penalty (matching original Config defaults) ─────────────────
    kl_penalty_coef: float = 0.0
    kl_discount_factor: float = 0.0
    compute_post_kl: bool = False

    # ── Additional config fields from original ─────────────────────────
    dynamic_max_tokens: bool = True
    two_phase_sampling: bool = False
    phase1_max_tokens: int = 26000
    eval_timeout: int = 1000
    dataset_timeout: int = 1000
    num_cpus_per_task: int = 1
    gpu_mode_score_scale: float = 3000.0

    # ── Ensemble (NEW) ────────────────────────────────────────────────────
    num_ensemble_members: int = 5
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    target_modules: Dict[str, bool] = field(default_factory=lambda: {
        "q_proj": True, "o_proj": True,
    })

    # ── Uncertainty (NEW) ─────────────────────────────────────────────────
    rmi_coef: float = 0.1           # α: base exploration strength (γ_eff adapts via β-coupling)
    uncertainty_metric: str = "rmi"  # "rmi", "variance", "predictive_entropy"
    gamma_max_ratio: float = 10.0   # clip γ_eff/rmi_coef to prevent explosion

    # ── Sampler ───────────────────────────────────────────────────────────
    sampler_type: str = "greedy"
    initial_exp_type: str = "random"

    # ── Logging (same as original) ────────────────────────────────────────
    log_path: str = "./logs/mlora_train"
    wandb_project: Optional[str] = None
    wandb_name: Optional[str] = None

    # ── Eval / checkpointing (same as original) ──────────────────────────
    eval_every: int = 3
    save_every: int = 5

    # ── Dataset / task wiring ─────────────────────────────────────────────
    dataset_builder: Optional[Callable] = None  # set by cli_main()


# ═══════════════════════════════════════════════════════════════════════════════
# Advantage computation — reuses logic from train.py:compute_advantages()
# ═══════════════════════════════════════════════════════════════════════════════

def compute_advantages(
    rewards: List[float],
    adv_estimator: str,
    adv_estimator_beta: float = 2.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute advantages from a group of rewards.
    Direct port from tinker_cookbook/rl/train.py:compute_advantages(),
    adapted to work on a flat reward list instead of TrajectoryGroup.

    Returns:
        (advantages, metadata) where metadata includes "beta" — the solved
        (or fixed) entropic temperature. Used by train_step for γ-coupling.
    """
    rewards_G = torch.tensor(rewards, dtype=torch.float32)
    k = rewards_G.shape[0]

    if adv_estimator == "mean_baseline":
        return rewards_G - rewards_G.mean(), {"beta": 0.0}

    elif adv_estimator == "entropic":
        s_safe = rewards_G - rewards_G.max()
        e = torch.exp(adv_estimator_beta * s_safe)
        Z = (e.sum() - e) / (k - 1) if k > 1 else e
        return e / (Z + 1e-12) - 1.0, {"beta": float(adv_estimator_beta)}

    elif adv_estimator == "entropic_adaptive_beta":
        delta = np.log(2)
        beta_max = 1e6
        r = rewards_G.float()

        if k < 2:
            beta = r.new_tensor(0.0)
        else:
            def kl_hat(b: float) -> float:
                logits = r.new_tensor(b) * (r - r.max())
                logq = logits - torch.logsumexp(logits, dim=0, keepdim=True)
                q = torch.exp(logq)
                return float((q * (logq + math.log(k))).sum().item())

            lo, hi = 0.0, 1.0
            if kl_hat(hi) < delta:
                while hi < beta_max and kl_hat(hi) < delta:
                    hi *= 2.0
                if kl_hat(hi) < delta:
                    beta = r.new_tensor(hi)
                else:
                    beta = None
            else:
                beta = None

            if beta is None:
                for _ in range(60):
                    mid = 0.5 * (lo + hi)
                    if kl_hat(mid) < delta:
                        lo = mid
                    else:
                        hi = mid
                beta = r.new_tensor(hi)

        e = torch.exp(beta * (r - r.max()))
        Z = (e.sum() - e) / (k - 1) if k > 1 else e
        return e / (Z + 1e-12) - 1.0, {"beta": float(beta.item())}

    else:
        raise ValueError(f"Unknown advantage estimator: {adv_estimator}")


# ═══════════════════════════════════════════════════════════════════════════════
# RL Loss — replaces Tinker's loss_fn (importance_sampling / ppo)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_rl_loss(
    current_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantage: "Union[float, torch.Tensor]",
    action_mask: torch.Tensor,
    loss_fn: str = "importance_sampling",
    ppo_clip: float = 0.2,
) -> torch.Tensor:
    """
    Compute RL policy gradient loss for a single trajectory.

    Replaces Tinker's forward_backward() + loss_fn mechanism.

    Args:
        advantage: Either a scalar (trajectory-level) or a (T-1,) per-token
            tensor. When incorporate_kl_penalty is active, per-token advantages
            are passed so that each token sees its own KL adjustment — matching
            the original train.py behaviour where Tinker receives per-token
            advantages via datum.loss_fn_inputs["advantages"].
    """
    if isinstance(advantage, torch.Tensor):
        adv = advantage.to(current_logprobs.device)
    else:
        adv = torch.tensor(advantage, device=current_logprobs.device)
    log_ratio = current_logprobs - old_logprobs
    ratio = log_ratio.exp()

    if loss_fn == "ppo":
        clipped = ratio.clamp(1.0 - ppo_clip, 1.0 + ppo_clip)
        surrogate = torch.min(ratio * adv, clipped * adv)
        masked = -(surrogate * action_mask)
    else:
        masked = -(ratio * adv * action_mask)

    return masked.sum() / action_mask.sum().clamp(min=1)


# ═══════════════════════════════════════════════════════════════════════════════
# KL penalty — port of metrics.py:incorporate_kl_penalty()
# ═══════════════════════════════════════════════════════════════════════════════

def discounted_future_sum_vectorized(x: np.ndarray, gamma: float) -> np.ndarray:
    """
    Discounted sum of future values for each position.
    Port of metrics.py:discounted_future_sum_vectorized().
    """
    import scipy.signal
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1].astype(x.dtype)


def incorporate_kl_penalty(
    rollouts: List["Rollout"],
    advantages: torch.Tensor,
    ensemble: "LoRAEnsemble",
    kl_penalty_coef: float,
    kl_discount_factor: float,
    device: str = "cuda",
) -> Tuple[List[torch.Tensor], Dict[str, float]]:
    """
    Adjust advantages with per-token KL penalty against the base model.

    Matches metrics.py:incorporate_kl_penalty() exactly: returns per-token
    advantage tensors so that each action token sees its own KL adjustment.

    For each rollout:
        1. Compute base model logprobs (no LoRA) for the generated sequence
        2. kl_diff_t = sampling_logprobs_t - base_logprobs_t
        3. avg_kl_diff = mean(kl_diff) across all rollouts (action tokens only)
        4. kl_advantage_t = kl_coef * mask_t * (avg_kl_diff - kl_diff_t)
        5. Optionally apply temporal discounting
        6. per_token_advantage_t = broadcast(scalar_advantage) + kl_advantage_t

    Args:
        rollouts: List of Rollout objects with sampling_logprobs.
        advantages: (N,) tensor of per-rollout scalar advantages.
        ensemble: LoRAEnsemble (used to get base model for logprob computation).
        kl_penalty_coef: Coefficient for KL penalty.
        kl_discount_factor: Temporal discount factor (0 = disabled).
        device: Device for tensors.

    Returns:
        per_token_advantages: List of (T_i-1,) tensors, one per rollout.
            Each token's advantage = scalar_advantage + per-token KL adjustment.
            Matches how train.py's assemble_training_data broadcasts the scalar
            advantage to all action tokens, then incorporate_kl_penalty adds
            per-token KL adjustments in-place.
        metrics: Dict with 'kl_policy_base' metric.
    """
    # Compute per-token KL diffs for each rollout
    logprob_diffs = []
    float_masks = []

    for rollout in rollouts:
        base_logprobs = _compute_base_logprobs(ensemble, rollout.full_tokens, device)

        sampling_lp = torch.tensor(rollout.sampling_logprobs, device=device)
        mask = rollout.action_mask.to(device)

        # kl_diff = sampling_logprobs - base_logprobs, masked
        diff = (sampling_lp - base_logprobs) * mask
        logprob_diffs.append(diff)
        float_masks.append(mask)

    # Average KL diff across all action tokens in the batch
    total_diff = sum(d.sum() for d in logprob_diffs)
    total_mask = sum(m.sum() for m in float_masks)
    avg_logp_diff = total_diff / total_mask.clamp(min=1)

    # Build per-token advantages: broadcast scalar advantage + per-token KL
    # This matches the original: assemble_training_data broadcasts the scalar
    # advantage to all action tokens, then incorporate_kl_penalty modifies in-place.
    per_token_advantages = []
    for i, rollout in enumerate(rollouts):
        kl_advantages = kl_penalty_coef * float_masks[i] * (avg_logp_diff - logprob_diffs[i])
        if kl_discount_factor > 0:
            kl_advantages = torch.tensor(
                discounted_future_sum_vectorized(kl_advantages.cpu().numpy(), kl_discount_factor),
                device=device,
            )
        # Broadcast scalar advantage to all tokens, add per-token KL adjustment
        token_adv = advantages[i] * float_masks[i] + kl_advantages
        per_token_advantages.append(token_adv)

    return per_token_advantages, {"kl_policy_base": float(avg_logp_diff)}


@torch.no_grad()
def _compute_base_logprobs(
    ensemble: "LoRAEnsemble",
    token_ids: List[int],
    device: str,
) -> torch.Tensor:
    """
    Compute logprobs from the base model (no LoRA adapters).

    This gives us p_base(y_t | y_{<t}) for KL penalty computation.
    We temporarily disable all LoRA adapters and do a forward pass.

    Returns:
        (T-1,) tensor of base model log-probabilities.
    """
    T = len(token_ids)

    # Build data config with no adapter (base model only)
    from mlora.model.args import MLoRAData, MLoRADataConfig, Tokens, Masks

    data_config = MLoRADataConfig(
        adapter_name="__base__",
        adapter_type="",
        start_idx=0,
        end_idx=1,
        expand_fn=ensemble._expand_fn,
        loss_fn=ensemble._noop_loss,
        task_name="base_scoring",
    )

    mlora_data = MLoRAData(
        batch_tokens=[list(token_ids)],
        batch_mask=[[True] * T],
        data_config=[data_config],
    )

    logits = ensemble.model.forward(mlora_data.model_data())  # (1, T, V)

    # Extract per-token logprobs for actual tokens
    log_probs = F.log_softmax(logits[0, :-1, :], dim=-1)  # (T-1, V)
    target = torch.tensor(token_ids[1:], dtype=torch.long, device=logits.device)
    per_token = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)  # (T-1,)

    return per_token


# ═══════════════════════════════════════════════════════════════════════════════
# KL tracking — port of metrics.py:compute_kl_sample_train()
# ═══════════════════════════════════════════════════════════════════════════════

def compute_kl_sample_train(
    rollouts: List["Rollout"],
    training_logprobs: List[torch.Tensor],
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Compute KL divergence between sampling and training logprobs.

    Port of metrics.py:compute_kl_sample_train(). This is the primary
    signal for detecting training instability — if KL is large, the
    policy has shifted significantly from the sampling policy.

    Args:
        rollouts: List of Rollout objects with sampling_logprobs.
        training_logprobs: List of (T-1,) tensors from the training forward pass.
        device: Device for tensors.

    Returns:
        Dict with 'optim/kl_sample_train_v1', 'optim/kl_sample_train_v2',
        'optim/entropy' metrics.
    """
    all_diffs: List[torch.Tensor] = []
    all_sampling_logprobs: List[torch.Tensor] = []

    for rollout, train_lp in zip(rollouts, training_logprobs):
        sampling_lp = torch.tensor(rollout.sampling_logprobs, device=device)
        mask = rollout.action_mask.to(device) > 0

        sampling_actions = sampling_lp[mask]
        training_actions = train_lp[mask]

        if len(sampling_actions) > 0:
            logprob_diff = sampling_actions - training_actions
            all_diffs.append(logprob_diff)
            all_sampling_logprobs.append(sampling_actions)

    if not all_diffs:
        return {
            "optim/kl_sample_train_v1": 0.0,
            "optim/kl_sample_train_v2": 0.0,
            "optim/entropy": 0.0,
        }

    flat_diffs = torch.cat(all_diffs)
    flat_sampling = torch.cat(all_sampling_logprobs)

    return {
        "optim/kl_sample_train_v1": flat_diffs.mean().item(),
        "optim/kl_sample_train_v2": 0.5 * (flat_diffs ** 2).mean().item(),
        "optim/entropy": -flat_sampling.mean().item(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Rollout data structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Rollout:
    """Single trajectory result. Analogous to original Trajectory + final reward."""
    prompt_tokens: List[int]
    full_tokens: List[int]           # prompt + generated
    sampling_logprobs: List[float]   # logprobs from sampling adapter
    reward_exec: float               # R_exec from task verifier
    reward_rmi: float = 0.0          # uncertainty bonus
    reward_total: float = 0.0        # R_exec + γ * RMI
    adapter_idx: int = 0             # which member generated this
    state: Optional[Any] = None      # parent state
    child_state: Optional[Any] = None  # child state for sampler update
    metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_tokens)

    @property
    def action_mask(self) -> torch.Tensor:
        """1 for generated tokens, 0 for prompt. Length = len(full_tokens) - 1."""
        total = len(self.full_tokens) - 1
        mask = torch.zeros(total)
        mask[self.prompt_len - 1:] = 1.0
        return mask


@dataclass
class RolloutGroup:
    """Group of rollouts from the same parent state. Analogous to TrajectoryGroup."""
    rollouts: List[Rollout]
    state: Optional[Any] = None

    def get_total_rewards(self) -> List[float]:
        """Matches TrajectoryGroup.get_total_rewards() interface."""
        return [r.reward_total for r in self.rollouts]

    def get_exec_rewards(self) -> List[float]:
        return [r.reward_exec for r in self.rollouts]


# ═══════════════════════════════════════════════════════════════════════════════
# Training loop — mirrors train.py:do_sync_training() structure
# ═══════════════════════════════════════════════════════════════════════════════

def do_rollout(
    ensemble: LoRAEnsemble,
    tokenizer: MLoRATokenizer,
    prompt_tokens: List[int],
    adapter_idx: int,
    parent_state: Optional[Any],
    step: int,
    process_fn: Callable[[str, Any, int], Dict[str, Any]],
    uncertainty_fn: Callable,
    rmi_coef: float,
    max_tokens: int,
    temperature: float,
) -> Rollout:
    """
    Generate a single rollout, compute R_exec and RMI.

    Args:
        process_fn: (generated_text, parent_state, step) → dict with keys:
            reward (float), child_state (Optional[State]), correctness (float), metrics (dict)
    """
    # ── 1. Generate (one adapter) ─────────────────────────────────────────
    full_tokens, sampling_logprobs = ensemble.generate(
        prompt_tokens=prompt_tokens,
        max_tokens=max_tokens,
        temperature=temperature,
        eos_token_id=tokenizer.eos_id_,
        adapter_idx=adapter_idx,
    )

    # ── 2. Execute → R_exec (via task verifier) ──────────────────────────
    generated_text = tokenizer.decode(full_tokens)
    result = process_fn(generated_text, parent_state, step)
    reward_exec = result["reward"]
    child_state = result.get("child_state")

    # ── 3. Ensemble scoring → RMI (all K adapters, single forward pass) ──
    ensemble_logprobs = ensemble.compute_ensemble_logprobs(full_tokens)  # (K, T-1)
    reward_rmi = float(uncertainty_fn(ensemble_logprobs))

    # ── 4. Augmented reward ──────────────────────────────────────────────
    reward_total = reward_exec + rmi_coef * reward_rmi

    return Rollout(
        prompt_tokens=prompt_tokens,
        full_tokens=full_tokens,
        sampling_logprobs=sampling_logprobs,
        reward_exec=reward_exec,
        reward_rmi=reward_rmi,
        reward_total=reward_total,
        adapter_idx=adapter_idx,
        child_state=child_state,
        metrics={
            "reward/exec": reward_exec,
            "reward/rmi": reward_rmi,
            "reward/total": reward_total,
            "gen/num_tokens": len(full_tokens) - len(prompt_tokens),
            "gen/adapter_idx": adapter_idx,
            "correctness": result.get("correctness", 0.0),
        },
    )


def _split_list(lst: list, n: int) -> List[list]:
    """Split list into n roughly equal chunks. Port of train.py:split_list()."""
    if n <= 0:
        return []
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


def train_step(
    ensemble: LoRAEnsemble,
    rollout_groups: List[RolloutGroup],
    cfg: Config,
) -> Dict[str, float]:
    """
    Train all K adapters on a batch of rollout groups.

    Mirrors the original train_step() → forward_backward() → optim_step() flow,
    but uses mLoRA's batched forward and trains all K adapters per step.

    Flow:
        1. For each group: compute advantages from augmented rewards
        2. Apply KL penalty if kl_penalty_coef > 0
        3. Split into num_substeps mini-batches
        4. For each substep:
           a. Forward-backward on mini-batch
           b. Optimizer step
        5. Compute KL tracking metrics
    """
    metrics_accum: Dict[str, List[float]] = {
        "reward/exec": [], "reward/rmi": [], "reward/total": [],
        "advantage/mean": [], "advantage/std": [],
        "advantage/min": [], "advantage/max": [],
        "beta": [], "gamma_eff": [],
    }

    # ── Phase 1: Compute advantages for all groups, collect trainable rollouts ──
    # When kl_penalty_coef > 0, advantages are per-token tensors (matching train.py).
    # When kl_penalty_coef == 0, advantages are scalars (float).
    trainable_rollouts: List[Tuple["Rollout", "Union[float, torch.Tensor]"]] = []

    for group in rollout_groups:
        # β sees R_exec ONLY — preserves max-reward-seeking gradient
        scalar_advantages, adv_meta = compute_advantages(
            group.get_exec_rewards(),
            adv_estimator=cfg.adv_estimator,
            adv_estimator_beta=cfg.adv_estimator_beta,
        )

        # ── RMI advantage-space bonus (after β, before KL) ──
        # γ_eff is coupled to β: exploration pressure increases as policy converges
        # β_ref = adv_estimator_beta so γ_eff = rmi_coef when β equals the fixed default
        beta = adv_meta.get("beta", cfg.adv_estimator_beta)
        if cfg.rmi_coef > 0:
            rmi_values = torch.tensor(
                [r.reward_rmi for r in group.rollouts], dtype=torch.float32
            )
            rmi_centered = rmi_values - rmi_values.mean()
            gamma_eff = cfg.rmi_coef * min(beta / cfg.adv_estimator_beta, cfg.gamma_max_ratio)
            scalar_advantages = scalar_advantages + gamma_eff * rmi_centered
        else:
            gamma_eff = 0.0

        metrics_accum["beta"].append(float(beta))
        metrics_accum["gamma_eff"].append(float(gamma_eff))

        # ── KL penalty adjustment → promotes to per-token advantages ──
        kl_metrics = {}
        if cfg.kl_penalty_coef > 0:
            per_token_advs, kl_metrics = incorporate_kl_penalty(
                rollouts=group.rollouts,
                advantages=scalar_advantages,
                ensemble=ensemble,
                kl_penalty_coef=cfg.kl_penalty_coef,
                kl_discount_factor=cfg.kl_discount_factor,
                device=cfg.device,
            )
            metrics_accum["advantage/mean"].append(scalar_advantages.mean().item())
            metrics_accum["advantage/std"].append(scalar_advantages.std().item())
            metrics_accum["advantage/min"].append(scalar_advantages.min().item())
            metrics_accum["advantage/max"].append(scalar_advantages.max().item())

            for rollout, scalar_adv, token_adv in zip(
                group.rollouts, scalar_advantages, per_token_advs
            ):
                metrics_accum["reward/exec"].append(rollout.reward_exec)
                metrics_accum["reward/rmi"].append(rollout.reward_rmi)
                metrics_accum["reward/total"].append(rollout.reward_total)

                if abs(scalar_adv.item()) < 1e-8:
                    continue

                trainable_rollouts.append((rollout, token_adv))
        else:
            metrics_accum["advantage/mean"].append(scalar_advantages.mean().item())
            metrics_accum["advantage/std"].append(scalar_advantages.std().item())
            metrics_accum["advantage/min"].append(scalar_advantages.min().item())
            metrics_accum["advantage/max"].append(scalar_advantages.max().item())

            for rollout, adv in zip(group.rollouts, scalar_advantages):
                metrics_accum["reward/exec"].append(rollout.reward_exec)
                metrics_accum["reward/rmi"].append(rollout.reward_rmi)
                metrics_accum["reward/total"].append(rollout.reward_total)

                if abs(adv.item()) < 1e-8:
                    continue

                trainable_rollouts.append((rollout, adv.item()))

    # ── Phase 2: Substep training (matching original train_step batch splitting) ──
    num_substeps = min(cfg.num_substeps, len(trainable_rollouts)) if trainable_rollouts else 0
    substep_batches = _split_list(trainable_rollouts, num_substeps) if num_substeps > 0 else []

    total_loss_val = 0.0
    all_training_logprobs: List[torch.Tensor] = []
    all_trained_rollouts: List[Rollout] = []

    for substep_batch in substep_batches:
        substep_loss = torch.tensor(0.0, device=cfg.device, requires_grad=True)

        for rollout, adv in substep_batch:
            # Forward through all K adapters WITH gradients
            _hidden, current_logprobs = ensemble.compute_training_logprobs(
                rollout.full_tokens
            )  # current_logprobs: (K, T-1)

            action_mask = rollout.action_mask.to(cfg.device)
            old_logprobs = torch.tensor(
                rollout.sampling_logprobs, device=cfg.device
            )

            # RL loss per adapter, averaged over K
            for k in range(ensemble.K):
                loss_k = compute_rl_loss(
                    current_logprobs=current_logprobs[k],
                    old_logprobs=old_logprobs,
                    advantage=adv,
                    action_mask=action_mask,
                    loss_fn=cfg.loss_fn,
                    ppo_clip=cfg.ppo_clip,
                )
                substep_loss = substep_loss + loss_k / ensemble.K

            # Store first adapter's training logprobs for KL tracking
            all_training_logprobs.append(current_logprobs[0].detach())
            all_trained_rollouts.append(rollout)

        if len(substep_batch) > 0:
            avg_loss = substep_loss / len(substep_batch)
            avg_loss.backward()
            ensemble.step_all_optimizers()
            total_loss_val += avg_loss.item()

    # ── Phase 3: KL tracking metrics ──
    kl_track_metrics = {}
    if all_trained_rollouts:
        kl_track_metrics = compute_kl_sample_train(
            all_trained_rollouts, all_training_logprobs, cfg.device
        )

    # ── Aggregate metrics ──
    metrics: Dict[str, float] = {}
    for key, vals in metrics_accum.items():
        if vals:
            metrics[f"train/{key}/mean"] = float(np.mean(vals))
            if "reward" in key:
                metrics[f"train/{key}/max"] = float(np.max(vals))
            if "advantage" in key and key.endswith("/mean"):
                # Already covered by the mean aggregation above
                pass
    # Ensure advantage min/max across all groups are tracked
    if metrics_accum["advantage/min"]:
        metrics["train/advantage/min"] = float(np.min(metrics_accum["advantage/min"]))
    if metrics_accum["advantage/max"]:
        metrics["train/advantage/max"] = float(np.max(metrics_accum["advantage/max"]))

    metrics["train/loss"] = total_loss_val
    metrics["train/num_rollouts"] = len(trainable_rollouts)
    metrics.update(kl_track_metrics)
    if kl_metrics:
        metrics.update(kl_metrics)

    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# main() — mirrors train.py:main()
# ═══════════════════════════════════════════════════════════════════════════════

def main(
    cfg: Config,
    prompt_fn: Callable[[Optional[Any]], str],
    process_fn: Callable[[str, Any, int], Dict[str, Any]],
    sampler: StateSampler,
):
    """
    Main training entry point.

    Args:
        cfg: Training configuration.
        prompt_fn: (state) → prompt text string.
        process_fn: (generated_text, parent_state, step) → dict with
            reward, child_state, correctness, metrics.
        sampler: State sampler (greedy, PUCT, etc.).
    """
    # ── 1. Setup logging (reuses existing ml_log) ─────────────────────────
    ml_logger = setup_logging(
        log_dir=cfg.log_path,
        wandb_project=cfg.wandb_project,
        config=cfg,
        wandb_name=cfg.wandb_name,
    )

    # ── 2. Load model + ensemble ──────────────────────────────────────────
    logger.info(f"Loading base model: {cfg.base_model} ({cfg.precision})")
    args = argparse.Namespace(
        base_model=cfg.base_model,
        device=cfg.device,
        precision=cfg.precision,
        model_type="llama",
        pipeline=False,
        balance=None,
        rank=None,
    )
    tokenizer, model = mlora_load_model(args)

    ensemble = LoRAEnsemble(
        model=model,
        num_members=cfg.num_ensemble_members,
        lora_rank=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        learning_rate=cfg.learning_rate,
        optimizer="adamw",
    )

    uncertainty_fn = UNCERTAINTY_FNS[cfg.uncertainty_metric]

    # ── 3. Resume ─────────────────────────────────────────────────────────
    start_epoch = 0
    checkpoint_file = os.path.join(cfg.log_path, "last_epoch.txt")
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file) as f:
            start_epoch = int(f.read().strip()) + 1
        ensemble.load(cfg.log_path, start_epoch - 1)
        logger.info(f"Resumed from epoch {start_epoch}")

    os.makedirs(cfg.log_path, exist_ok=True)

    # ── 4. Training loop ─────────────────────────────────────────────────
    num_batches_per_epoch = cfg.groups_per_batch
    num_batches_total = num_batches_per_epoch * cfg.num_epochs
    logger.info(
        f"Training: {cfg.num_epochs} epochs, "
        f"{cfg.groups_per_batch} groups/batch × {cfg.group_size} rollouts/group, "
        f"K={cfg.num_ensemble_members} adapters, γ={cfg.rmi_coef}"
    )

    for epoch in range(start_epoch, cfg.num_epochs):
        t_start = time.time()
        metrics: Dict[str, Any] = {
            "progress/epoch": epoch,
            "progress/done_frac": (epoch + 1) / cfg.num_epochs,
        }

        # ── Rollout phase ─────────────────────────────────────────────
        rollout_groups: List[RolloutGroup] = []
        # Sample parent states for this epoch
        parent_states = sampler.sample_states(cfg.groups_per_batch)

        for g, parent_state in enumerate(parent_states):
            group_rollouts: List[Rollout] = []

            # Build prompt text once per parent state, tokenize
            prompt_text = prompt_fn(parent_state)
            prompt_tokens = tokenizer.encode(prompt_text, bos=True, eos=False)

            for i in range(cfg.group_size):
                # Rotate adapter for generation diversity
                adapter_idx = (epoch * cfg.group_size + i) % ensemble.K
                step_idx = epoch * cfg.groups_per_batch * cfg.group_size + g * cfg.group_size + i

                rollout = do_rollout(
                    ensemble=ensemble,
                    tokenizer=tokenizer,
                    prompt_tokens=prompt_tokens,
                    adapter_idx=adapter_idx,
                    parent_state=parent_state,
                    step=step_idx,
                    process_fn=process_fn,
                    uncertainty_fn=uncertainty_fn,
                    rmi_coef=cfg.rmi_coef,
                    max_tokens=cfg.max_tokens,
                    temperature=cfg.temperature,
                )
                rollout.state = parent_state
                group_rollouts.append(rollout)

            # Filter constant-reward groups (same as original)
            rewards = [r.reward_total for r in group_rollouts]
            if cfg.remove_constant_reward_groups and len(set(rewards)) <= 1:
                continue

            rollout_groups.append(
                RolloutGroup(rollouts=group_rollouts, state=parent_state)
            )

        metrics["time/rollout"] = time.time() - t_start

        # ── Train step ────────────────────────────────────────────────
        t_train = time.time()
        train_metrics = train_step(ensemble, rollout_groups, cfg)
        metrics.update(train_metrics)
        metrics["time/train"] = time.time() - t_train

        # ── Update sampler buffer with child states ──────────────────
        for group in rollout_groups:
            child_states = []
            parent_states_for_update = []
            for rollout in group.rollouts:
                if rollout.child_state is not None:
                    child_states.append(rollout.child_state)
                    parent_states_for_update.append(rollout.state)
                elif hasattr(sampler, 'record_failed_rollout'):
                    sampler.record_failed_rollout(rollout.state)
            if child_states:
                sampler.update_states(child_states, parent_states_for_update, save=False)
        sampler.flush(step=epoch)

        # ── Logging ───────────────────────────────────────────────────
        metrics["time/total"] = time.time() - t_start
        ml_logger.log_metrics(metrics, step=epoch)

        r_exec = metrics.get("train/reward/exec/mean", 0)
        r_max = metrics.get("train/reward/exec/max", 0)
        r_rmi = metrics.get("train/reward/rmi/mean", 0)
        loss = metrics.get("train/loss", 0)
        logger.info(
            f"[Epoch {epoch}] R_exec={r_exec:.4f} (max={r_max:.4f}) "
            f"RMI={r_rmi:.4f} loss={loss:.4f} "
            f"time={metrics['time/total']:.1f}s"
        )

        # ── Checkpointing ─────────────────────────────────────────────
        if cfg.save_every > 0 and (epoch + 1) % cfg.save_every == 0:
            ensemble.save(cfg.log_path, epoch)
            with open(checkpoint_file, "w") as f:
                f.write(str(epoch))
            logger.info(f"Checkpoint saved at epoch {epoch}")

    # ── 5. Final checkpoint ───────────────────────────────────────────────
    ensemble.save(cfg.log_path, cfg.num_epochs - 1)
    with open(checkpoint_file, "w") as f:
        f.write(str(cfg.num_epochs - 1))

    ml_logger.close()
    logger.info("Training complete")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI — mirrors tinker_cookbook/recipes/ttt/train.py:cli_main()
# ═══════════════════════════════════════════════════════════════════════════════

def cli_main():
    """
    CLI entry point. Parses args → Config, then calls main().

    Task/prompt/sampler wiring should happen here, matching the pattern
    in tinker_cookbook/recipes/ttt/train.py:cli_main().
    """
    parser = argparse.ArgumentParser(description="Multi-LoRA Ensemble RL Training")

    # Model
    parser.add_argument("--base_model", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--precision", default="nf4",
                        choices=["nf4", "fp4", "int8", "bf16", "fp16", "fp32"])
    parser.add_argument("--device", default="cuda")

    # Environment
    parser.add_argument("--env", default="ac1")
    parser.add_argument("--problem_idx", default="improvement")
    parser.add_argument("--budget_s", type=int, default=1000)

    # RL
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--groups_per_batch", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--max_tokens", type=int, default=26000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--adv_estimator", default="entropic")
    parser.add_argument("--loss_fn", default="importance_sampling")

    # Ensemble
    parser.add_argument("--num_ensemble_members", type=int, default=5)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=4e-5)

    # Uncertainty
    parser.add_argument("--rmi_coef", type=float, default=0.1)
    parser.add_argument("--uncertainty_metric", default="rmi",
                        choices=["rmi", "variance", "predictive_entropy"])

    # Sampler
    # KL penalty
    parser.add_argument("--kl_penalty_coef", type=float, default=0.0)
    parser.add_argument("--kl_discount_factor", type=float, default=0.0)
    parser.add_argument("--compute_post_kl", action="store_true", default=False)

    # Additional config fields from original
    parser.add_argument("--num_substeps", type=int, default=1)
    parser.add_argument("--dynamic_max_tokens", action="store_true", default=True)
    parser.add_argument("--two_phase_sampling", action="store_true", default=False)
    parser.add_argument("--phase1_max_tokens", type=int, default=26000)
    parser.add_argument("--eval_timeout", type=int, default=1000)
    parser.add_argument("--dataset_timeout", type=int, default=1000)
    parser.add_argument("--num_cpus_per_task", type=int, default=1)
    parser.add_argument("--gpu_mode_score_scale", type=float, default=3000.0)
    parser.add_argument("--remove_constant_reward_groups", action="store_true", default=True)

    parser.add_argument("--sampler_type", default="greedy")
    parser.add_argument("--initial_exp_type", default="random")

    # Logging
    parser.add_argument("--log_path", default="./logs/mlora_train")
    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--wandb_name", default=None)
    parser.add_argument("--eval_every", type=int, default=3)
    parser.add_argument("--save_every", type=int, default=5)

    args = parser.parse_args()

    cfg = Config(
        base_model=args.base_model,
        precision=args.precision,
        device=args.device,
        env=args.env,
        problem_idx=args.problem_idx,
        budget_s=args.budget_s,
        group_size=args.group_size,
        groups_per_batch=args.groups_per_batch,
        num_epochs=args.num_epochs,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        adv_estimator=args.adv_estimator,
        loss_fn=args.loss_fn,
        num_substeps=args.num_substeps,
        remove_constant_reward_groups=args.remove_constant_reward_groups,
        kl_penalty_coef=args.kl_penalty_coef,
        kl_discount_factor=args.kl_discount_factor,
        compute_post_kl=args.compute_post_kl,
        dynamic_max_tokens=args.dynamic_max_tokens,
        two_phase_sampling=args.two_phase_sampling,
        phase1_max_tokens=args.phase1_max_tokens,
        eval_timeout=args.eval_timeout,
        dataset_timeout=args.dataset_timeout,
        num_cpus_per_task=args.num_cpus_per_task,
        gpu_mode_score_scale=args.gpu_mode_score_scale,
        num_ensemble_members=args.num_ensemble_members,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        learning_rate=args.learning_rate,
        rmi_coef=args.rmi_coef,
        uncertainty_metric=args.uncertainty_metric,
        sampler_type=args.sampler_type,
        initial_exp_type=args.initial_exp_type,
        log_path=args.log_path,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        eval_every=args.eval_every,
        save_every=args.save_every,
    )

    # ── Wire task/prompt/sampler from existing TTT-Discover code ──────────
    # Same pattern as tinker_cookbook/recipes/ttt/train.py:cli_main()
    # but with direct function wiring instead of Env classes.

    from tinker_cookbook.recipes.ttt.sampler import create_sampler
    from tinker_cookbook.recipes.ttt.env_ttt import last_codeblock_postprocess
    from tinker_cookbook.recipes.ttt.state import InequalitiesState, CirclePackingState

    os.makedirs(cfg.log_path, exist_ok=True)

    # ── Build prompt_fn and process_fn based on env type ──────────────────
    if cfg.env in ("ac1", "ac2"):
        from tinker_cookbook.recipes.ttt.env_ac import build_ac_prompt, verify_ac

        def prompt_fn(state: InequalitiesState) -> str:
            return build_ac_prompt(
                state, env_name=cfg.env,
                problem_idx=cfg.problem_idx,
                budget_s=cfg.budget_s,
                num_cpus_per_task=cfg.num_cpus_per_task,
            )

        def process_fn(generated_text: str, parent_state: InequalitiesState, step: int) -> Dict[str, Any]:
            parsed_code = last_codeblock_postprocess(generated_text, ["python"], keep_separators=True)
            if not parsed_code or not parsed_code.strip():
                return {"reward": 0.0, "child_state": None, "correctness": 0.0, "metrics": {}}

            outs = verify_ac(
                parsed_code, step,
                num_cpus_per_task=cfg.num_cpus_per_task,
                env_name=cfg.env,
                eval_timeout=cfg.eval_timeout,
                log_path=cfg.log_path,
                state=parent_state,
            )
            correctness = outs.get("correctness", 0.0)

            # Reward — same as AutoCorrInequalityEnv._compute_reward()
            if correctness > 0:
                if cfg.env == "ac1":
                    reward = 1.0 / (-outs["performance"]) if outs["performance"] != 0 else 0.0
                else:
                    reward = outs["performance"]
            else:
                reward = 0.0

            # Child state — same as AutoCorrInequalityEnv._create_next_state()
            child_state = None
            result_construction = outs.get("result_construction")
            if correctness > 0 and result_construction is not None:
                child_state = InequalitiesState(
                    timestep=step,
                    construction=result_construction,
                    code=parsed_code,
                    value=outs["performance"],
                    observation=outs.get("stdout", ""),
                )

            return {
                "reward": reward,
                "child_state": child_state,
                "correctness": correctness,
                "metrics": {
                    "score": outs.get("score", 0.0),
                    "correctness": correctness,
                    "performance": outs.get("performance", 0.0),
                },
            }

    elif cfg.env == "cp":
        from tinker_cookbook.recipes.ttt.env_cp import build_cp_prompt, verify_cp, parse_cp_problem_idx

        n, _ = parse_cp_problem_idx(cfg.problem_idx)

        def prompt_fn(state: CirclePackingState) -> str:
            return build_cp_prompt(state, n=n)

        def process_fn(generated_text: str, parent_state: CirclePackingState, step: int) -> Dict[str, Any]:
            parsed_code = last_codeblock_postprocess(generated_text, ["python"], keep_separators=True)
            if not parsed_code or not parsed_code.strip():
                return {"reward": 0.0, "child_state": None, "correctness": 0.0, "metrics": {}}

            outs = verify_cp(
                parsed_code, step,
                num_cpus_per_task=cfg.num_cpus_per_task,
                problem_idx=n,
                eval_timeout=cfg.eval_timeout,
                log_path=cfg.log_path,
                state=parent_state,
            )
            correctness = outs.get("correctness", 0.0)

            # Reward — same as CirclePackingEnv._compute_reward()
            sum_radii = outs.get("sum_radii", outs.get("score", 0.0))
            reward = sum_radii if correctness > 0 else 0.0

            # Child state — same as CirclePackingEnv._create_next_state()
            child_state = None
            if correctness > 0:
                parent_values = ([parent_state.value] + parent_state.parent_values
                                 if parent_state.value is not None else [])
                child_state = CirclePackingState(
                    timestep=step,
                    construction=outs.get("result_construction"),
                    code=parsed_code,
                    value=sum_radii,
                    parent_values=parent_values,
                    observation=outs.get("stdout", ""),
                )

            return {
                "reward": reward,
                "child_state": child_state,
                "correctness": correctness,
                "metrics": {
                    "score": outs.get("score", 0.0),
                    "correctness": correctness,
                    "sum_radii": sum_radii,
                },
            }

    else:
        raise ValueError(f"Unsupported env: {cfg.env}. Supported: ac1, ac2, cp")

    # ── Create sampler ────────────────────────────────────────────────────
    sampler = create_sampler(
        sampler_type=cfg.sampler_type,
        log_path=cfg.log_path,
        env_type=cfg.env,
        initial_exp_type=cfg.initial_exp_type,
        batch_size=cfg.groups_per_batch,
    )

    main(
        cfg=cfg,
        prompt_fn=prompt_fn,
        process_fn=process_fn,
        sampler=sampler,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    cli_main()
