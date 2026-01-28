"""Polarization and phase metrics."""

import numpy as np
import torch


def magnitude_variance(Z: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Variance of magnitudes across tokens.

    Args:
        Z: Complex tensor (..., n_tokens, d_complex)
        dim: Dimension to compute variance over (token dimension)

    Returns:
        Variance per sample/dimension
    """
    mags = torch.abs(Z)
    return mags.var(dim=dim)


def max_mean_ratio(Z: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    """Ratio of max to mean magnitude.

    Higher values indicate stronger polarization (one token dominates).
    """
    mags = torch.abs(Z)
    max_mag = mags.max(dim=dim).values
    mean_mag = mags.mean(dim=dim)
    return max_mag / (mean_mag + eps)


def gini_coefficient(Z: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Gini coefficient of magnitudes.

    0 = perfect equality (all same magnitude)
    1 = perfect inequality (one token has all magnitude)
    """
    mags = torch.abs(Z)

    # Sort magnitudes
    sorted_mags, _ = torch.sort(mags, dim=dim)
    n = mags.shape[dim]

    # Gini formula: G = (2 * sum(i * x_i)) / (n * sum(x_i)) - (n+1)/n
    indices = torch.arange(1, n + 1, device=Z.device, dtype=Z.dtype)

    # Reshape indices for broadcasting
    shape = [1] * mags.ndim
    shape[dim] = n
    indices = indices.view(shape)

    numerator = 2 * (indices * sorted_mags).sum(dim=dim)
    denominator = n * sorted_mags.sum(dim=dim)

    return numerator / (denominator + 1e-8) - (n + 1) / n


def magnitude_entropy(Z: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    """Entropy of normalized magnitude distribution.

    Lower entropy = more concentrated (stronger polarization).
    """
    mags = torch.abs(Z)
    # Normalize to probability distribution
    probs = mags / (mags.sum(dim=dim, keepdim=True) + eps)
    # Entropy
    log_probs = torch.log(probs + eps)
    entropy = -(probs * log_probs).sum(dim=dim)
    return entropy


def phase_coherence(Z: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Mean resultant length - measures phase concentration.

    1 = all phases identical
    0 = phases uniformly distributed
    """
    # Normalize to unit magnitude
    unit_Z = Z / (torch.abs(Z) + 1e-8)
    # Mean of unit vectors
    mean_unit = unit_Z.mean(dim=dim)
    # Length of mean
    return torch.abs(mean_unit)


def compute_all_metrics(Z: torch.Tensor, dim: int = 1) -> dict:
    """Compute all polarization metrics.

    Args:
        Z: Complex tensor
        dim: Token dimension

    Returns:
        Dict of metric_name -> tensor
    """
    return {
        "magnitude_variance": magnitude_variance(Z, dim),
        "max_mean_ratio": max_mean_ratio(Z, dim),
        "gini": gini_coefficient(Z, dim),
        "entropy": magnitude_entropy(Z, dim),
        "phase_coherence": phase_coherence(Z, dim),
    }


def summarize_metrics(metrics: dict) -> dict:
    """Reduce metric tensors to scalar summaries."""
    summary = {}
    for name, tensor in metrics.items():
        summary[f"{name}_mean"] = tensor.mean().item()
        summary[f"{name}_std"] = tensor.std().item()
    return summary


# NumPy versions for analysis
def circular_variance(phases: np.ndarray) -> float:
    """Circular variance of phase angles."""
    mean_vec = np.mean(np.exp(1j * phases))
    return 1 - np.abs(mean_vec)


def circular_mean(phases: np.ndarray) -> float:
    """Circular mean of phase angles."""
    return np.angle(np.mean(np.exp(1j * phases)))
