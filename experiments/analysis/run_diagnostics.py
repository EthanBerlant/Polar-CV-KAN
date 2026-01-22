"""
Run diagnostics and visualizations on trained CV-KAN models.

Generates:
- Phase distribution plots across layers
- Magnitude polarization analysis
- Per-word magnitude heatmaps for sentiment analysis
- Dimension correlation analysis

Usage:
    python experiments/run_diagnostics.py --checkpoint outputs/sst2/best.pt
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.text import load_sst2, pad_collate
from src.models import CVKAN


class TextClassifier(torch.nn.Module):
    """Wrapper for loading SST-2 trained model."""

    def __init__(self, vocab_size, embed_dim, cvkan_args):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.model = CVKAN(d_input=embed_dim, **cvkan_args)

    def forward(self, indices, mask=None, return_intermediates=False):
        x = self.embedding(indices)
        return self.model(x, mask=mask, return_intermediates=return_intermediates)


def plot_layer_phases(intermediates, save_path):
    """Plot phase distributions across all layers."""
    n_layers = len(intermediates)
    cols = min(4, n_layers)
    rows = (n_layers + cols - 1) // cols

    fig, axes = plt.subplots(
        rows, cols, figsize=(4 * cols, 4 * rows), subplot_kw={"projection": "polar"}
    )
    if rows == 1:
        axes = [axes] if cols == 1 else list(axes)
    else:
        axes = axes.flatten()

    for i, Z in enumerate(intermediates):
        phases = torch.angle(Z).cpu().numpy().flatten()
        n_bins = 36
        bins = np.linspace(-np.pi, np.pi, n_bins + 1)
        hist, _ = np.histogram(phases, bins=bins)
        theta = (bins[:-1] + bins[1:]) / 2
        width = 2 * np.pi / n_bins
        axes[i].bar(theta, hist, width=width, alpha=0.7, color="steelblue")
        axes[i].set_title(f"Layer {i}")

    for i in range(n_layers, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle("Phase Distributions Across Layers", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def plot_layer_magnitudes(intermediates, save_path):
    """Plot magnitude distributions across all layers."""
    n_layers = len(intermediates)
    cols = min(4, n_layers)
    rows = (n_layers + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    if rows == 1:
        axes = [axes] if cols == 1 else list(axes)
    else:
        axes = axes.flatten()

    for i, Z in enumerate(intermediates):
        mags = torch.abs(Z).cpu().numpy().flatten()
        axes[i].hist(mags, bins=50, alpha=0.7, density=True, color="coral")
        axes[i].axvline(mags.mean(), color="red", linestyle="--", label=f"μ={mags.mean():.2f}")
        axes[i].set_xlabel("Magnitude")
        axes[i].set_ylabel("Density")
        axes[i].set_title(f"Layer {i}")
        axes[i].legend(fontsize=8)

    for i in range(n_layers, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle("Magnitude Distributions Across Layers", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def plot_dimension_correlation(Z, save_path):
    """Plot correlation matrix between dimensions' phase patterns."""
    phases = torch.angle(Z).reshape(-1, Z.shape[-1]).cpu().numpy()
    corr = np.corrcoef(phases.T)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Dimension")
    ax.set_title("Phase Pattern Correlation Between Dimensions")
    plt.colorbar(im, ax=ax)

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

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def plot_word_magnitudes(words, magnitudes, label, pred, save_path):
    """Plot per-word magnitude heatmap for a single example."""
    fig, ax = plt.subplots(figsize=(max(10, len(words) * 0.5), 3))

    mags = magnitudes.mean(dim=-1).cpu().numpy()  # Average across dimensions
    mags_norm = (mags - mags.min()) / (mags.max() - mags.min() + 1e-8)

    # Create heatmap
    colors = plt.cm.YlOrRd(mags_norm)
    for i, (word, mag, color) in enumerate(zip(words, mags, colors, strict=False)):
        ax.bar(i, 1, color=color, edgecolor="gray", linewidth=0.5)
        ax.text(i, 0.5, word, ha="center", va="center", rotation=45, fontsize=9)
        ax.text(i, -0.1, f"{mag:.2f}", ha="center", va="top", fontsize=7)

    ax.set_xlim(-0.5, len(words) - 0.5)
    ax.set_ylim(-0.3, 1.2)
    ax.set_yticks([])
    ax.set_xticks([])

    label_str = "Positive" if label == 1 else "Negative"
    pred_str = "Positive" if pred == 1 else "Negative"
    ax.set_title(f"Word Magnitudes | True: {label_str} | Pred: {pred_str}")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def analyze_sentiment_examples(model, dataloader, vocab, device, output_dir, n_examples=5):
    """Analyze specific examples and show word-level magnitudes."""
    model.eval()
    idx_to_word = vocab.itos

    examples_analyzed = 0
    for batch in dataloader:
        if examples_analyzed >= n_examples:
            break

        indices = batch["indices"].to(device)
        mask = batch["mask"].to(device)
        labels = batch["label"].to(device)

        with torch.no_grad():
            outputs = model(indices, mask=mask, return_intermediates=True)

        logits = outputs["logits"]
        preds = logits.argmax(dim=-1)
        Z_final = outputs["intermediates"][-1]
        magnitudes = torch.abs(Z_final)

        for i in range(min(indices.shape[0], n_examples - examples_analyzed)):
            # Get words
            seq_len = mask[i].sum().item()
            word_indices = indices[i, : int(seq_len)].cpu().tolist()
            words = [idx_to_word.get(idx, "<unk>") for idx in word_indices]

            # Get magnitudes for this example
            mags = magnitudes[i, : int(seq_len)]

            save_path = os.path.join(output_dir, f"word_magnitude_{examples_analyzed}.png")
            plot_word_magnitudes(words, mags, labels[i].item(), preds[i].item(), save_path)
            print(f"Saved: {save_path}")

            examples_analyzed += 1
            if examples_analyzed >= n_examples:
                break


def compute_aggregate_stats(model, dataloader, device):
    """Compute aggregate statistics about magnitude distribution by word position."""
    model.eval()

    all_mags = []
    all_labels = []

    for batch in dataloader:
        indices = batch["indices"].to(device)
        mask = batch["mask"].to(device)
        labels = batch["label"].to(device)

        with torch.no_grad():
            outputs = model(indices, mask=mask, return_intermediates=True)

        Z_final = outputs["intermediates"][-1]
        magnitudes = torch.abs(Z_final).mean(dim=-1)  # (batch, seq_len)

        # Pad to max_len (64) if needed for aggregation
        if magnitudes.shape[1] < 64:
            pad_len = 64 - magnitudes.shape[1]
            magnitudes = F.pad(magnitudes, (0, pad_len))
        elif magnitudes.shape[1] > 64:
            magnitudes = magnitudes[:, :64]

        all_mags.append(magnitudes.cpu())
        all_labels.append(labels.cpu())

    all_mags = torch.cat(all_mags, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    pos_mags = all_mags[all_labels == 1].mean(dim=0)
    neg_mags = all_mags[all_labels == 0].mean(dim=0)

    return {
        "positive_mean_mags": pos_mags,
        "negative_mean_mags": neg_mags,
        "overall_mean": all_mags.mean().item(),
        "overall_std": all_mags.std().item(),
    }


def main():
    parser = argparse.ArgumentParser(description="Run CV-KAN diagnostics")
    parser.add_argument(
        "--checkpoint", type=str, default="outputs/sst2/best.pt", help="Path to checkpoint"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="visualizations/diagnostics",
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--n_examples", type=int, default=10, help="Number of examples to visualize"
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print("Loading SST-2 data...")
    _, val_ds, vocab = load_sst2(max_len=64)
    from torch.utils.data import DataLoader

    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=pad_collate)

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Reconstruct model (using default SST-2 config)
    model = TextClassifier(
        vocab_size=len(vocab),
        embed_dim=64,
        cvkan_args={
            "d_complex": 64,
            "n_layers": 2,
            "n_classes": 2,
            "kan_hidden": 32,
            "head_approach": "emergent",
            "pooling": "mean",
            "input_type": "real",
        },
    ).to(device)

    model.load_state_dict(checkpoint)
    model.eval()
    print("Model loaded successfully!")

    # Get sample batch for layer visualizations
    print("\nGenerating visualizations...")
    batch = next(iter(val_loader))
    indices = batch["indices"].to(device)
    mask = batch["mask"].to(device)

    with torch.no_grad():
        outputs = model(indices, mask=mask, return_intermediates=True)

    # 1. Phase distributions
    plot_layer_phases(
        outputs["intermediates"], os.path.join(args.output_dir, "phase_distributions.png")
    )

    # 2. Magnitude distributions
    plot_layer_magnitudes(
        outputs["intermediates"], os.path.join(args.output_dir, "magnitude_distributions.png")
    )

    # 3. Dimension correlation
    Z_final = outputs["intermediates"][-1]
    plot_dimension_correlation(Z_final, os.path.join(args.output_dir, "dimension_correlation.png"))

    # 4. Word-level analysis
    print("\nAnalyzing individual examples...")
    analyze_sentiment_examples(model, val_loader, vocab, device, args.output_dir, args.n_examples)

    # 5. Aggregate statistics
    print("\nComputing aggregate statistics...")
    stats = compute_aggregate_stats(model, val_loader, device)
    print(f"  Overall mean magnitude: {stats['overall_mean']:.4f}")
    print(f"  Overall std magnitude: {stats['overall_std']:.4f}")

    print(f"\nAll visualizations saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
