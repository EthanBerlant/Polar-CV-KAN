"""Visualization utilities for research experiments."""

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.style.use("seaborn-v0_8-whitegrid")
COLORS = plt.cm.tab10.colors


def save_figure(fig: plt.Figure, path: Path, dpi: int = 150):
    """Save figure and close."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_polarization_trajectory(
    layer_metrics: dict[int, dict[str, float]],
    metric_name: str,
    output_path: Path,
    title: str | None = None,
):
    """Plot how a polarization metric evolves across layers.

    Args:
        layer_metrics: {layer_idx: {metric_name: value}}
        metric_name: Which metric to plot
        output_path: Where to save
        title: Optional title
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    layers = sorted(layer_metrics.keys())
    values = [layer_metrics[l][metric_name] for l in layers]

    ax.plot(layers, values, "o-", linewidth=2, markersize=8, color=COLORS[0])
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel(metric_name.replace("_", " ").title(), fontsize=12)
    ax.set_title(title or f"{metric_name} Across Layers", fontsize=14)

    # Add trend line
    z = np.polyfit(layers, values, 1)
    p = np.poly1d(z)
    ax.plot(layers, p(layers), "--", alpha=0.5, color=COLORS[1], label=f"Trend (slope={z[0]:.4f})")
    ax.legend()

    save_figure(fig, output_path)


def plot_multi_metric_trajectory(
    layer_metrics: dict[int, dict[str, float]],
    metric_names: list[str],
    output_path: Path,
    title: str | None = None,
):
    """Plot multiple metrics on same axes (normalized)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    layers = sorted(layer_metrics.keys())

    for i, metric_name in enumerate(metric_names):
        values = np.array([layer_metrics[l][metric_name] for l in layers])
        # Normalize to [0, 1] for comparison
        values_norm = (values - values.min()) / (values.max() - values.min() + 1e-8)
        ax.plot(
            layers, values_norm, "o-", linewidth=2, markersize=6, color=COLORS[i], label=metric_name
        )

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Normalized Value", fontsize=12)
    ax.set_title(title or "Polarization Metrics Across Layers", fontsize=14)
    ax.legend(loc="best")

    save_figure(fig, output_path)


def plot_phase_histogram(
    phases: np.ndarray,
    output_path: Path,
    n_bins: int = 36,
    title: str | None = None,
    labels: np.ndarray | None = None,
):
    """Polar histogram of phase angles.

    Args:
        phases: Array of phase angles in radians
        output_path: Where to save
        n_bins: Number of angular bins
        title: Optional title
        labels: Optional labels for coloring
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="polar")

    if labels is None:
        # Single histogram
        counts, bins = np.histogram(phases, bins=n_bins, range=(-np.pi, np.pi))
        width = 2 * np.pi / n_bins
        bars = ax.bar(bins[:-1], counts, width=width, bottom=0, alpha=0.7)
    else:
        # Stacked by label
        unique_labels = np.unique(labels)
        width = 2 * np.pi / n_bins
        bins = np.linspace(-np.pi, np.pi, n_bins + 1)
        bottom = np.zeros(n_bins)

        for i, label in enumerate(unique_labels):
            mask = labels == label
            counts, _ = np.histogram(phases[mask], bins=bins)
            ax.bar(
                bins[:-1],
                counts,
                width=width,
                bottom=bottom,
                alpha=0.7,
                label=str(label),
                color=COLORS[i % len(COLORS)],
            )
            bottom += counts
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1))

    ax.set_title(title or "Phase Distribution", fontsize=14, pad=20)

    save_figure(fig, output_path)


def plot_magnitude_phase_scatter(
    magnitudes: np.ndarray,
    phases: np.ndarray,
    output_path: Path,
    labels: np.ndarray | None = None,
    title: str | None = None,
    max_points: int = 5000,
):
    """Scatter plot in polar coordinates.

    Args:
        magnitudes: Array of magnitudes (will be radial coordinate)
        phases: Array of phases (angular coordinate)
        output_path: Where to save
        labels: Optional labels for coloring
        title: Optional title
        max_points: Subsample if more points
    """
    # Subsample if needed
    if len(magnitudes) > max_points:
        idx = np.random.choice(len(magnitudes), max_points, replace=False)
        magnitudes = magnitudes[idx]
        phases = phases[idx]
        if labels is not None:
            labels = labels[idx]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="polar")

    if labels is None:
        ax.scatter(phases, magnitudes, alpha=0.3, s=10)
    else:
        unique_labels = np.unique(labels)
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(
                phases[mask],
                magnitudes[mask],
                alpha=0.3,
                s=10,
                color=COLORS[i % len(COLORS)],
                label=str(label),
            )
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1))

    ax.set_title(title or "Magnitude-Phase Distribution", fontsize=14, pad=20)

    save_figure(fig, output_path)


def plot_transform_function(
    input_values: np.ndarray,
    output_values: np.ndarray,
    output_path: Path,
    title: str = "Learned Transform",
    xlabel: str = "Input",
    ylabel: str = "Output",
):
    """Plot a learned 1D transform function."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Sort by input for clean line
    sort_idx = np.argsort(input_values)
    x = input_values[sort_idx]
    y = output_values[sort_idx]

    ax.plot(x, y, linewidth=2, color=COLORS[0])
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)

    # Identity line for reference
    ax.plot(
        [x.min(), x.max()], [x.min(), x.max()], "--", color=COLORS[1], alpha=0.5, label="Identity"
    )

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()

    save_figure(fig, output_path)


def plot_comparison_bars(
    results: dict[str, float],
    output_path: Path,
    title: str = "Comparison",
    ylabel: str = "Value",
    error_bars: dict[str, float] | None = None,
):
    """Bar chart comparing methods/conditions."""
    fig, ax = plt.subplots(figsize=(10, 6))

    names = list(results.keys())
    values = list(results.values())
    x = np.arange(len(names))

    bars = ax.bar(x, values, color=COLORS[: len(names)], alpha=0.8)

    if error_bars:
        errors = [error_bars.get(n, 0) for n in names]
        ax.errorbar(x, values, yerr=errors, fmt="none", color="black", capsize=5)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)

    # Add value labels on bars
    for bar, val in zip(bars, values, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    save_figure(fig, output_path)


def plot_training_curves(
    histories: dict[str, dict[str, list[float]]],
    metric: str,
    output_path: Path,
    title: str | None = None,
):
    """Plot training curves for multiple runs.

    Args:
        histories: {run_name: {"train_loss": [...], "val_acc": [...], ...}}
        metric: Which metric to plot
        output_path: Where to save
        title: Optional title
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (name, history) in enumerate(histories.items()):
        if metric in history:
            values = history[metric]
            ax.plot(values, label=name, color=COLORS[i % len(COLORS)], linewidth=2)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
    ax.set_title(title or f"Training Curves: {metric}", fontsize=14)
    ax.legend(loc="best")

    save_figure(fig, output_path)


def save_results_json(results: dict[str, Any], output_path: Path):
    """Save results dict to JSON."""

    # Convert numpy types to Python types
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(output_path, "w") as f:
        json.dump(convert(results), f, indent=2)
    print(f"Saved: {output_path}")
