"""
Regularization losses for CV-KAN.

Includes:
- Diversity loss: Encourages different dimensions to learn different patterns
- Phase anchor loss: Soft attraction to discrete phase anchors
"""

import torch


def diversity_loss(Z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Penalize correlation between dimensions' phase patterns.

    Encourages different complex dimensions to learn different
    phase relationships, creating implicit "heads" in Approach A.

    Args:
        Z: Complex tensor of shape (batch, n_tokens, d_complex)
        eps: Small constant for numerical stability

    Returns:
        Scalar loss (higher = more correlated = bad)
    """
    # Extract phases: (batch, n_tokens, d_complex)
    phases = torch.angle(Z)

    # Flatten batch and tokens: (batch * n_tokens, d_complex)
    phases_flat = phases.reshape(-1, phases.shape[-1])

    # Compute correlation matrix between dimensions
    # First center the data
    phases_centered = phases_flat - phases_flat.mean(dim=0, keepdim=True)

    # Covariance
    cov = torch.mm(phases_centered.T, phases_centered) / (phases_flat.shape[0] - 1 + eps)

    # Correlation (normalize by std)
    std = phases_flat.std(dim=0) + eps
    corr = cov / (std.unsqueeze(0) * std.unsqueeze(1))

    # Penalize off-diagonal correlations (upper triangle to avoid double counting)
    d = corr.shape[0]
    mask = torch.triu(torch.ones(d, d, device=corr.device, dtype=torch.bool), diagonal=1)
    off_diag_corr = corr[mask]

    # L2 penalty on correlations
    return (off_diag_corr**2).mean()


def phase_anchor_loss(
    Z: torch.Tensor,
    n_anchors: int = 4,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Encourage phases to cluster near canonical angles.

    Creates an attractor landscape without hardcoding specific phases.
    Helps prevent phase drift/chaos in deep networks.

    Args:
        Z: Complex tensor of shape (batch, n_tokens, d_complex)
        n_anchors: Number of anchor points (evenly spaced on unit circle)
                   Default 4 = 0, π/2, π, 3π/2
        eps: Small constant for numerical stability

    Returns:
        Scalar loss (higher = further from anchors)
    """
    # Extract phases: (batch, n_tokens, d_complex)
    phases = torch.angle(Z)

    # Create anchor points: 0, 2π/n, 4π/n, ...
    anchors = torch.arange(n_anchors, device=Z.device, dtype=phases.dtype) * (
        2 * torch.pi / n_anchors
    )

    # Compute distance to nearest anchor for each phase
    # phases: (...) -> (..., 1)
    # anchors: (n_anchors,) -> (1, ..., 1, n_anchors)
    phases_expanded = phases.unsqueeze(-1)

    # Angular distance (accounting for wrap-around)
    diff = phases_expanded - anchors
    # Map to [-π, π]
    diff = torch.atan2(torch.sin(diff), torch.cos(diff))
    angular_dist = torch.abs(diff)

    # Distance to nearest anchor
    min_dist = angular_dist.min(dim=-1).values

    return min_dist.mean()


def magnitude_sparsity_loss(Z: torch.Tensor, target_sparsity: float = 0.5) -> torch.Tensor:
    """
    Encourage sparse magnitude activations.

    Helps with interpretability - only some dimensions should be "active".

    Args:
        Z: Complex tensor of shape (batch, n_tokens, d_complex)
        target_sparsity: Desired fraction of "inactive" dimensions

    Returns:
        Scalar loss (deviation from target sparsity)
    """
    mags = torch.abs(Z)

    # Normalize magnitudes per sample
    mags_norm = mags / (mags.mean() + 1e-6)

    # Compute "activation" (sigmoid of normalized magnitude)
    activation = torch.sigmoid(mags_norm - 1)  # centered at mean

    # Current sparsity (fraction below threshold)
    current_sparsity = (activation < 0.5).float().mean()

    # Penalize deviation from target
    return (current_sparsity - target_sparsity) ** 2


def coherence_loss(Z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Encourage tokens of the same class to have coherent phases.

    Useful for the signal/noise task where signal tokens should align.

    Args:
        Z: Complex tensor of shape (batch, n_tokens, d_complex)
        labels: Binary tensor of shape (batch, n_tokens) indicating signal tokens

    Returns:
        Scalar loss (higher = less coherent within class)
    """
    batch, n_tokens, d = Z.shape

    losses = []
    for b in range(batch):
        # Get signal and noise tokens
        signal_mask = labels[b].bool()

        if signal_mask.sum() > 1:
            # Phase coherence for signal tokens
            signal_phases = torch.angle(Z[b, signal_mask])  # (n_signal, d)

            # Compute circular variance per dimension
            # Low variance = coherent
            cos_sum = torch.cos(signal_phases).sum(dim=0)
            sin_sum = torch.sin(signal_phases).sum(dim=0)
            n = signal_mask.sum().float()
            R = torch.sqrt(cos_sum**2 + sin_sum**2) / n  # [0, 1], 1 = perfect coherence

            # Loss: 1 - R (want to maximize R)
            losses.append((1 - R).mean())

    if losses:
        return torch.stack(losses).mean()
    else:
        return torch.tensor(0.0, device=Z.device)
