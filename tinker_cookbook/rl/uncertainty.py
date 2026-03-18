"""
Uncertainty metrics for epistemic exploration in TTT-Discover.

Computes Reverse Mutual Information (RMI) and ablation alternatives
from ensemble logprobs. Pure math — no framework dependencies.

Given K LoRA ensemble members and a generated token sequence of length T,
each function takes a (K, T) tensor of per-token log-probabilities and
returns a scalar uncertainty score.

Reference: Malinin & Gales, "Reverse KL-Divergence Training of Prior Networks"
"""

import torch


def compute_rmi(ensemble_logprobs: torch.Tensor) -> torch.Tensor:
    """
    Compute Reverse Mutual Information from ensemble logprobs.

    RMI measures ensemble disagreement by evaluating the likelihood of model
    parameters given the average predictive distribution. High RMI indicates
    OOD territory where ensemble members structurally disagree.

    For token-level logprobs of the actual generated sequence:

        MI(y; Θ|x) = log p̄(y) - E_k[log p_k(y)]

    where p̄(y_t) = (1/K) Σ_k p_k(y_t) is the geometric-mean approximation
    to the mixture predictive. MI >= 0 by Jensen's inequality. Higher MI
    means more ensemble disagreement → more epistemic uncertainty.

    Args:
        ensemble_logprobs: (K, T) tensor. Entry [k, t] is
            log p_k(y_t | y_{<t}, x) for ensemble member k.

    Returns:
        Scalar tensor: mean per-token mutual information (non-negative).
    """
    # p_k(y_t) for each member and token
    ensemble_probs = ensemble_logprobs.exp()  # (K, T)

    # Average probability: p̄(y_t) = (1/K) Σ_k p_k(y_t)
    log_avg_probs = ensemble_probs.mean(dim=0).log()  # (T,)

    # E_k[log p_k(y_t)]
    mean_member_logprobs = ensemble_logprobs.mean(dim=0)  # (T,)

    # MI per token = log p̄(y_t) - E_k[log p_k(y_t)]
    # This is >= 0 by Jensen's inequality
    mi_per_token = log_avg_probs - mean_member_logprobs  # (T,)

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
    Entropy of the average predictive distribution.
    Ablation baseline: captures total uncertainty (epistemic + aleatoric).

    Note: This is an approximation — we only have logprobs for the generated
    token at each position, not the full distribution. We compute the entropy
    of the mixture probability for the selected token.

    Args:
        ensemble_logprobs: (K, T) tensor.

    Returns:
        Scalar: mean predictive entropy across tokens.
    """
    avg_probs = ensemble_logprobs.exp().mean(dim=0)  # (T,)
    entropy = -(avg_probs * avg_probs.log())
    return entropy.mean()


UNCERTAINTY_FNS = {
    "rmi": compute_rmi,
    "variance": compute_variance,
    "predictive_entropy": compute_predictive_entropy,
}
