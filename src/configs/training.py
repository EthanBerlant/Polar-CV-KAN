from dataclasses import dataclass
from typing import Literal


@dataclass
class TrainingConfig:
    """Configuration for training loop."""

    epochs: int = 50
    batch_size: int = 64
    lr: float = 0.001
    optimizer: Literal["adam", "adamw", "sgd"] = "adam"
    weight_decay: float = 1e-4
    patience: int = 10
    metric_name: str = "accuracy"
    metric_mode: Literal["min", "max"] = "max"
    amp: bool = False
    gradient_clip: float = 1.0
    seed: int = 42
    output_dir: str = "outputs"
    run_name: str | None = None
    resume_from_checkpoint: str | None = None
    save_top_k: int = 1
    subset_size: int | None = None
