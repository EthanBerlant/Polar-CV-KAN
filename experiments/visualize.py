"""
Visualization utilities for CV-KAN analysis.

Visualizes:
- Phase distributions across layers
- Magnitude distributions
- Token attention patterns (via phase coherence)
- Head specialization (for multi-head approaches)
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import create_signal_noise_dataloader
from src.models.cv_kan import CVKANTokenClassifier


def plot_phase_distribution(phases: np.ndarray, title: str, ax=None):
    """
    Plot phase distribution on polar histogram.

    Args:
        phases: Array of phases in radians
        title: Plot title
        ax: Matplotlib axis (creates new if None)
    """
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})

    # Histogram bins
    n_bins = 36
    bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    hist, _ = np.histogram(phases.flatten(), bins=bins)

    # Plot as polar bars
    theta = (bins[:-1] + bins[1:]) / 2
    width = 2 * np.pi / n_bins
    ax.bar(theta, hist, width=width, alpha=0.7)
    ax.set_title(title)

    return ax


def plot_magnitude_distribution(mags: np.ndarray, title: str, ax=None):
    """Plot magnitude distribution as histogram."""
    if ax is None:
        fig, ax = plt.subplots()

    ax.hist(mags.flatten(), bins=50, alpha=0.7, density=True)
    ax.axvline(mags.mean(), color="r", linestyle="--", label=f"Mean: {mags.mean():.2f}")
    ax.set_xlabel("Magnitude")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()

    return ax


def plot_layer_phases(intermediates: list, save_path: str = None):
    """
    Plot phase distributions at each layer.

    Args:
        intermediates: List of complex tensors from each layer
        save_path: Path to save figure (shows if None)
    """
    n_layers = len(intermediates)
    fig, axes = plt.subplots(
        2,
        (n_layers + 1) // 2,
        figsize=(4 * ((n_layers + 1) // 2), 8),
        subplot_kw={"projection": "polar"},
    )
    axes = axes.flatten()

    for i, Z in enumerate(intermediates):
        phases = torch.angle(Z).cpu().numpy()
        plot_phase_distribution(phases, f"Layer {i}", axes[i])

    # Hide unused axes
    for i in range(n_layers, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_layer_magnitudes(intermediates: list, save_path: str = None):
    """Plot magnitude distributions at each layer."""
    n_layers = len(intermediates)
    fig, axes = plt.subplots(2, (n_layers + 1) // 2, figsize=(4 * ((n_layers + 1) // 2), 6))
    axes = axes.flatten()

    for i, Z in enumerate(intermediates):
        mags = torch.abs(Z).cpu().numpy()
        plot_magnitude_distribution(mags, f"Layer {i}", axes[i])

    for i in range(n_layers, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_signal_noise_separation(
    Z: torch.Tensor,
    labels: torch.Tensor,
    save_path: str = None,
):
    """
    Visualize separation between signal and noise tokens.

    Args:
        Z: Complex tensor (batch, n_tokens, d)
        labels: Token labels (batch, n_tokens) - 1 for signal
        save_path: Path to save figure
    """
    # Flatten batch dimension
    Z_flat = Z.reshape(-1, Z.shape[-1])
    labels_flat = labels.reshape(-1)

    # Get magnitudes and phases
    mags = torch.abs(Z_flat).cpu().numpy()
    phases = torch.angle(Z_flat).cpu().numpy()
    labels_np = labels_flat.cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Magnitude comparison
    signal_mags = mags[labels_np == 1].flatten()
    noise_mags = mags[labels_np == 0].flatten()

    axes[0].hist(signal_mags, bins=50, alpha=0.6, label="Signal", density=True)
    axes[0].hist(noise_mags, bins=50, alpha=0.6, label="Noise", density=True)
    axes[0].set_xlabel("Magnitude")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Magnitude Distribution")
    axes[0].legend()

    # 2. Phase comparison
    signal_phases = phases[labels_np == 1].flatten()
    noise_phases = phases[labels_np == 0].flatten()

    axes[1].hist(signal_phases, bins=50, alpha=0.6, label="Signal", density=True)
    axes[1].hist(noise_phases, bins=50, alpha=0.6, label="Noise", density=True)
    axes[1].set_xlabel("Phase (radians)")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Phase Distribution")
    axes[1].legend()

    # 3. Phase coherence per dimension (signal vs noise)
    # Compute circular variance
    def circular_variance(phases_arr):
        cos_sum = np.cos(phases_arr).sum(axis=0)
        sin_sum = np.sin(phases_arr).sum(axis=0)
        n = phases_arr.shape[0]
        R = np.sqrt(cos_sum**2 + sin_sum**2) / n
        return 1 - R  # Variance = 1 - coherence

    signal_phase_var = circular_variance(phases[labels_np == 1])
    noise_phase_var = circular_variance(phases[labels_np == 0])

    dims = np.arange(len(signal_phase_var))
    width = 0.35
    axes[2].bar(dims - width / 2, 1 - signal_phase_var, width, label="Signal Coherence", alpha=0.7)
    axes[2].bar(dims + width / 2, 1 - noise_phase_var, width, label="Noise Coherence", alpha=0.7)
    axes[2].set_xlabel("Dimension")
    axes[2].set_ylabel("Phase Coherence")
    axes[2].set_title("Per-Dimension Phase Coherence")
    axes[2].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_dimension_similarity(Z: torch.Tensor, save_path: str = None):
    """
    Plot correlation matrix between dimensions' phase patterns.

    Lower correlation = more diversity = better for Approach A.

    Args:
        Z: Complex tensor (batch, n_tokens, d)
        save_path: Path to save figure
    """
    phases = torch.angle(Z).reshape(-1, Z.shape[-1]).cpu().numpy()

    # Compute correlation matrix
    corr = np.corrcoef(phases.T)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Dimension")
    ax.set_title("Phase Pattern Correlation Between Dimensions")
    plt.colorbar(im, ax=ax)

    # Annotate mean off-diagonal correlation
    mask = ~np.eye(corr.shape[0], dtype=bool)
    mean_corr = np.abs(corr[mask]).mean()
    ax.text(
        0.02,
        0.98,
        f"Mean |corr|: {mean_corr:.3f}",
        transform=ax.transAxes,
        va="top",
        fontsize=12,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_from_checkpoint(checkpoint_path: str, output_dir: str = "visualizations"):
    """
    Load a checkpoint and create all visualizations.

    Args:
        checkpoint_path: Path to .pt checkpoint
        output_dir: Directory to save visualizations
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    args_dict = checkpoint["args"]

    # Recreate model
    model = CVKANTokenClassifier(
        d_input=args_dict["d_complex"],
        d_complex=args_dict["d_complex"],
        n_layers=args_dict["n_layers"],
        n_classes=2,
        kan_hidden=args_dict["kan_hidden"],
        head_approach=args_dict["head_approach"],
        n_heads=args_dict["n_heads"],
        input_type="complex",
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Create data
    dataloader = create_signal_noise_dataloader(
        n_samples=1000,
        n_tokens=args_dict["n_tokens"],
        k_signal=args_dict["k_signal"],
        d_complex=args_dict["d_complex"],
        batch_size=64,
        shuffle=False,
        seed=999,
    )

    # Get a batch
    batch = next(iter(dataloader))
    sequences = batch["sequence"]
    labels = batch["token_labels"]

    # Forward pass
    with torch.no_grad():
        outputs = model(sequences, return_intermediates=True)

    # Create visualizations
    print("Creating visualizations...")

    # 1. Phase distributions across layers
    plot_layer_phases(outputs["intermediates"], os.path.join(output_dir, "layer_phases.png"))

    # 2. Magnitude distributions across layers
    plot_layer_magnitudes(
        outputs["intermediates"], os.path.join(output_dir, "layer_magnitudes.png")
    )

    # 3. Signal vs noise separation (final layer)
    plot_signal_noise_separation(
        outputs["Z"], labels, os.path.join(output_dir, "signal_noise_separation.png")
    )

    # 4. Dimension similarity
    plot_dimension_similarity(outputs["Z"], os.path.join(output_dir, "dimension_similarity.png"))

    print(f"All visualizations saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Visualize CV-KAN representations")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--output_dir", type=str, default="visualizations", help="Output directory")
    args = parser.parse_args()

    visualize_from_checkpoint(args.checkpoint, args.output_dir)


if __name__ == "__main__":
    main()
