from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Configuration for the Polar-CV-KAN model architecture."""

    # General dimensions
    d_complex: int = 64
    n_layers: int = 4
    kan_hidden: int = 32

    # Components
    embedding_type: str = "linear"  # linear, conv, identity
    normalization: str = "none"  # none, batch, layer
    aggregation: str = "polar"  # polar, mean, max
    block_type: str = "polarizing"  # polarizing, hierarchical

    # Specifics
    dropout: float = 0.0
    input_type: str = "real"  # real, complex

    # Hierarchical specifics (if block_type is hierarchical)
    hierarchical_levels: int = 3
    hierarchical_sharing: str = "none"
    hierarchical_top_down: str = "none"


@dataclass
class DataConfig:
    """Configuration for data loading."""

    dataset_name: str = "cifar10"
    data_dir: str = "./data"
    batch_size: int = 32
    num_workers: int = 4
    subset_size: int | None = None

    # Specifics
    img_size: int = 32  # Default for CIFAR, easy to override
    patch_size: int = 4  # Default for CIFAR
    in_channels: int = 3


@dataclass
class TrainerConfig:
    """Configuration for the training loop."""

    output_dir: str = "./outputs"
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 10
    metric_name: str = "accuracy"
    metric_mode: str = "max"  # max or min
    seed: int = 42
    use_amp: bool = False
    debug: bool = False

    # Logging
    project_name: str = "cvkan_project"
    run_name: str | None = None


@dataclass
class ExperimentConfig:
    """Root configuration object."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(
        default_factory=DataConfig
    )  # Will need to instantiate manually usually
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
