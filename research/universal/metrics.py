import numpy as np
import torch


def compute_gini_coefficient(x: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """Compute Gini coefficient of the last dimension of x.

    Args:
        x: Input tensor (..., dim) representing magnitudes.
        epsilon: Small constant to avoid division by zero.

    Returns:
        Scalar tensor with mean Gini coefficient.
    """
    # Ensure positive
    x = torch.abs(x) + epsilon

    # Flatten all batch dims to (N, dim) if needed, but here we likely want mean over batch
    # Let's assume x is (batch, dim) or (batch, tokens, dim)
    # We compute Gini per sample/token, then average.

    dim = x.size(-1)

    # Sort along last dim
    x_sorted, _ = torch.sort(x, dim=-1)

    # Gini index formula:
    # G = (2 * sum(i * x_i) / (n * sum(x_i))) - (n + 1) / n
    # where i is 1-based index

    index = torch.arange(1, dim + 1, device=x.device).float()

    numerator = (x_sorted * index).sum(dim=-1)
    denominator = x_sorted.sum(dim=-1)

    gini = (2.0 * numerator) / (dim * denominator) - (dim + 1.0) / dim

    return gini.mean()


def compute_phase_consistency(
    pooled_features: torch.Tensor, labels: torch.Tensor, n_classes: int = 2
) -> float:
    """Compute phase consistency (Rayleigh test statistic R) for each class.

    R = |(1/N) * sum(exp(i * theta))|

    Args:
        pooled_features: Complex tensor (batch, dim).
        labels: Long tensor (batch,).
        n_classes: Number of classes.

    Returns:
        Mean R value across all classes and dimensions.
    """
    phase = torch.angle(pooled_features)
    coherences = []

    for c in range(n_classes):
        mask = labels == c
        if mask.sum() > 0:
            # Phase for class c: (N_c, dim)
            p_c = phase[mask]

            # Vector strength R
            # exp(i * theta)
            z = torch.exp(1j * p_c)

            # Mean vector
            # mean over samples (dim 0) -> (dim,)
            mean_z = torch.mean(z, dim=0)

            # Magnitude R
            R = torch.abs(mean_z)
            coherences.append(R.mean().item())

    if not coherences:
        return 0.0

    return np.mean(coherences)


def compute_gradient_norms(model) -> dict:
    """Compute gradient norms for monitoring stability.

    Args:
        model: PyTorch model.

    Returns:
        Dictionary of layer names and their gradient norms.
    """
    norms = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            norm = p.grad.norm().item()
            # Group by layer prefix (e.g., "layers.0")
            prefix = name.split(".")[0]
            # Actually, "layers.0" is better
            if "layers" in name:
                parts = name.split(".")
                prefix = f"{parts[0]}.{parts[1]}"
            else:
                prefix = "other"

            if prefix not in norms:
                norms[prefix] = []
            norms[prefix].append(norm)

    # Average within groups
    avg_norms = {k: np.mean(v) for k, v in norms.items()}
    return avg_norms
