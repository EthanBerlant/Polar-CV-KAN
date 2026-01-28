from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ModelConfig:
    """Base configuration for all models."""

    d_complex: int = 64
    n_layers: int = 4
    kan_hidden: int = 32
    embedding_type: str = "linear"
    aggregation: str = "magnitude_weighted"
    block_type: str = "polarizing"
    pooling: Literal["mean", "max", "attention"] = "mean"
    center_magnitudes: bool = True
    dropout: float = 0.0
    input_type: Literal["real", "complex"] = "real"
    normalization: Literal["none", "layer", "batch", "rms"] = "none"


@dataclass
class CVKANConfig(ModelConfig):
    """Configuration for standard CVKAN models."""

    head_approach: Literal["emergent", "offset", "factored"] = "emergent"
    aggregation_type: Literal["mean", "causal", "window", "neighborhood", "magnitude_weighted"] = (
        "magnitude_weighted"
    )
    n_heads: int = 8
    n_classes: int = 2
    skip_connections: bool = False

    # Hierarchical Config
    hierarchical_levels: int | None = None
    hierarchical_sharing: str = "per_level"  # shared, per_level, per_direction
    hierarchical_top_down: str = "none"  # none, mirror, learned
    hierarchical_split_idx: int = 2  # For hybrid sharing
    hierarchical_phase_shifting: bool = False  # For phase alignment
    hierarchical_interaction: str = "broadcast"  # broadcast or pointwise
    mag_init_scale: float = 0.1


@dataclass
class ImageConfig(CVKANConfig):
    """Image-specific configuration."""

    dataset_name: str = "cifar10"
    img_size: int = 32
    patch_size: int = 4
    in_channels: int = 3
    embedding_type: Literal["linear", "conv"] = "conv"
    pos_encoding: bool = True
    input_type: Literal["real", "complex"] = "real"


@dataclass
class AudioConfig(CVKANConfig):
    """Audio-specific configuration."""

    dataset_name: str = "speech_commands"
    sample_rate: int = 16000
    n_fft: int = 1024
    hop_length: int = 512
    n_mels: int = 64
    use_stft_frontend: bool = True
    input_type: Literal["real", "complex"] = "real"


@dataclass
class TimeSeriesConfig(CVKANConfig):
    """Time series-specific configuration."""

    dataset_name: str = "etth1"
    seq_len: int = 96
    pred_len: int = 96
    features: Literal["M", "S", "MS"] = "M"
    d_input: int = 7  # Number of input features
    output_mode: Literal["magnitude", "real", "complex", "both"] = "magnitude"
    input_type: Literal["real", "complex"] = "real"


@dataclass
class NLPConfig(CVKANConfig):
    """NLP-specific configuration."""

    dataset_name: str = "sst2"
    vocab_size: int = 20000
    max_seq_len: int = 256
    d_model: int = 64  # Usually matches d_complex but can be distinct for embedding
    input_type: Literal["real", "complex"] = "real"


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
    data: DataConfig = field(default_factory=DataConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
