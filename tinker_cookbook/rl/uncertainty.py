"""
Uncertainty metrics for epistemic exploration in TTT-Discover.

Primary metric: True MI computed from full vocabulary distributions during
ensemble scoring (H[mixture] - E_k[H[member]]). This captures distributional
disagreement across the entire vocabulary, not just at generated tokens.

Ablation baselines: Jensen-gap MI, variance, and token-level predictive entropy
computed from per-token logprobs at generated tokens only.

Reference: He et al., "Survey of uncertainty estimation in LLMs" (2026), Eq. 6
           Malinin & Gales, "Uncertainty estimation in autoregressive structured
           prediction" (ICLR 2020)
"""

import math

import torch


# ── Primary metric ───────────────────────────────────────────────────────────


def compute_true_mi(per_token_mi: torch.Tensor) -> torch.Tensor:
    """
    Aggregate pre-computed per-token true mutual information.

    True MI is the Bayesian decomposition of predictive uncertainty:

        MI(y; Θ|x) = H[p̄(·|x)] - E_k[H[p_k(·|x)]]

    where H is entropy over the full vocabulary V at each position,
    p̄(v) = (1/K) Σ_k p_k(v) is the arithmetic mixture, and the
    expectation is over K ensemble members.

    Per-position true MI is computed during the chunked LM head pass in
    LoRAEnsemble._chunked_logprobs(compute_mi=True), which has access to
    the full (K, chunk, V) logit tensor before it is discarded. This
    function simply averages the per-position values.

    Args:
        per_token_mi: (T,) tensor of per-position true MI values,
            pre-computed from full vocabulary distributions.

    Returns:
        Scalar: mean true MI across positions (non-negative by construction).
    """
    return per_token_mi.mean()


# ── Ablation baselines (computed from per-token logprobs at generated tokens) ─


def compute_rmi(ensemble_logprobs: torch.Tensor) -> torch.Tensor:
    """
    Jensen-gap MI estimate from ensemble logprobs (ablation baseline).

    Approximation that only considers the probability of the generated token
    at each position, NOT the full vocabulary distribution:

        JG(t) = log p̄(y_t) - E_k[log p_k(y_t)]

    where p̄(y_t) = (1/K) Σ_k p_k(y_t) is the arithmetic mean probability.
    Non-negative by Jensen's inequality (log is concave).

    This misses distributional disagreement at non-generated tokens. Use
    compute_true_mi for full-distribution epistemic uncertainty.

    Args:
        ensemble_logprobs: (K, T) tensor. Entry [k, t] is
            log p_k(y_t | y_{<t}, x) for ensemble member k.

    Returns:
        Scalar tensor: mean per-token Jensen-gap MI (non-negative).
    """
    K = ensemble_logprobs.shape[0]
    # Numerically stable: log((1/K) Σ_k p_k) = logsumexp(log_p_k, dim=0) - log(K)
    log_avg_probs = torch.logsumexp(ensemble_logprobs, dim=0) - math.log(K)
    mean_member_logprobs = ensemble_logprobs.mean(dim=0)
    mi_per_token = log_avg_probs - mean_member_logprobs
    return mi_per_token.mean()


def compute_variance(ensemble_logprobs: torch.Tensor) -> torch.Tensor:
    """
    Variance of per-token logprobs across ensemble members.
    Ablation baseline: simpler proxy for ensemble disagreement.

    Args:
        ensemble_logprobs: (K, T) tensor.

    Returns:
        Scalar: mean variance across tokens.
    """
    return ensemble_logprobs.var(dim=0).mean()


def compute_predictive_entropy(ensemble_logprobs: torch.Tensor) -> torch.Tensor:
    """
    Entropy of the mixture predictive at the generated token (approximate).
    Ablation baseline: captures total uncertainty (epistemic + aleatoric)
    but only at the generated token, not the full vocabulary.

    H_approx(t) = -p̄(y_t) log p̄(y_t)

    Args:
        ensemble_logprobs: (K, T) tensor.

    Returns:
        Scalar: mean token-level predictive entropy across positions.
    """
    K = ensemble_logprobs.shape[0]
    log_avg_probs = torch.logsumexp(ensemble_logprobs, dim=0) - math.log(K)
    avg_probs = log_avg_probs.exp()
    entropy = -(avg_probs * log_avg_probs)
    return entropy.mean()


# Logprob-based metrics (take (K, T) ensemble_logprobs).
# compute_true_mi has a different signature (takes pre-computed (T,) MI)
# and is handled separately by callers.
UNCERTAINTY_FNS = {
    "rmi": compute_rmi,
    "variance": compute_variance,
    "predictive_entropy": compute_predictive_entropy,
}
