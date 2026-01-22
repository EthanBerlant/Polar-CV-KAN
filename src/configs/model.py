from dataclasses import dataclass
from typing import Literal


@dataclass
class ModelConfig:
    """Base configuration for all models."""

    d_complex: int = 64
    n_layers: int = 4
    kan_hidden: int = 32
    pooling: Literal["mean", "max", "attention"] = "mean"
    center_magnitudes: bool = True
    dropout: float = 0.0
    input_type: Literal["real", "complex"] = "complex"


@dataclass
class CVKANConfig(ModelConfig):
    """Configuration for standard CVKAN models."""

    head_approach: Literal["emergent", "offset", "factored"] = "emergent"
    block_type: Literal["polarizing", "attention"] = "polarizing"
    aggregation_type: Literal["mean", "causal", "window", "neighborhood"] = "mean"
    n_heads: int = 8
    n_classes: int = 2
    skip_connections: bool = False


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
