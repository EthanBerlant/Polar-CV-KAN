"""Shared configuration for research experiments."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ResearchConfig:
    """Central configuration for all research experiments."""

    # Paths
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    output_dir: Path = field(default_factory=lambda: Path(__file__).parent / "outputs")
    checkpoint_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "outputs")

    # Default checkpoints to analyze (update these after training)
    checkpoints: dict = field(
        default_factory=lambda: {
            "sst2": "outputs/sst2_best.pt",
            "cifar10": "outputs/cifar10_best.pt",
            "synthetic": "outputs/synthetic_best.pt",
        }
    )

    # Experiment settings
    seed: int = 42
    device: str = "cuda"  # Will fall back to CPU if unavailable

    # Phase 1: Polarization
    polarization_measures: list = field(
        default_factory=lambda: [
            "variance",  # Var_i(|z_i|)
            "max_mean",  # max/mean ratio
            "gini",  # Gini coefficient of magnitudes
            "entropy",  # Entropy of normalized magnitudes
        ]
    )

    # Phase 2: Semantics
    n_phase_clusters: int = 8
    sentiment_lexicon: str = "vader"  # or "custom"

    # Phase 3: Ablations
    ablation_epochs: int = 10
    ablation_trials: int = 3

    # Phase 5: Stability
    gradient_sample_freq: int = 50  # Log every N batches

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for i in range(1, 6):
            (self.output_dir / f"phase{i}").mkdir(exist_ok=True)


# Singleton instance
CONFIG = ResearchConfig()
