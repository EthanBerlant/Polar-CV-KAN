"""Research utilities."""

from .checkpoint_utils import (
    evaluate_transform,
    extract_kan_transforms,
    extract_layer_outputs,
    find_checkpoints,
    get_device,
    get_model_statistics,
    get_polarizing_blocks,
    load_checkpoint,
    load_model_from_checkpoint,
)
from .metrics import (
    circular_mean,
    circular_variance,
    compute_all_metrics,
    gini_coefficient,
    magnitude_entropy,
    magnitude_variance,
    max_mean_ratio,
    phase_coherence,
    summarize_metrics,
)
from .visualization import (
    plot_comparison_bars,
    plot_magnitude_phase_scatter,
    plot_multi_metric_trajectory,
    plot_phase_histogram,
    plot_polarization_trajectory,
    plot_training_curves,
    plot_transform_function,
    save_figure,
    save_results_json,
)

__all__ = [
    # Metrics
    "magnitude_variance",
    "max_mean_ratio",
    "gini_coefficient",
    "magnitude_entropy",
    "phase_coherence",
    "compute_all_metrics",
    "summarize_metrics",
    "circular_variance",
    "circular_mean",
    # Visualization
    "plot_polarization_trajectory",
    "plot_multi_metric_trajectory",
    "plot_phase_histogram",
    "plot_magnitude_phase_scatter",
    "plot_transform_function",
    "plot_comparison_bars",
    "plot_training_curves",
    "save_results_json",
    "save_figure",
    # Checkpoint utils
    "get_device",
    "load_checkpoint",
    "load_model_from_checkpoint",
    "extract_layer_outputs",
    "get_polarizing_blocks",
    "extract_kan_transforms",
    "evaluate_transform",
    "get_model_statistics",
    "find_checkpoints",
]
