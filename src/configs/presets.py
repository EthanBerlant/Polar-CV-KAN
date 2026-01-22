"""Domain presets with optimized defaults."""

from .model import AudioConfig, ImageConfig, NLPConfig, TimeSeriesConfig

PRESETS = {
    # Image
    "cifar10": ImageConfig(
        dataset_name="cifar10",
        d_complex=256,
        n_layers=6,
        pooling="attention",
        embedding_type="conv",
        img_size=32,
        n_classes=10,
        dropout=0.1,
    ),
    # Audio
    "speech_commands": AudioConfig(
        dataset_name="speech_commands",
        d_complex=128,
        n_layers=4,
        pooling="attention",
        n_classes=35,
        dropout=0.1,
    ),
    # Audio (fast) - smaller FFT for faster training
    "speech_commands_fast": AudioConfig(
        dataset_name="speech_commands",
        d_complex=128,
        n_layers=4,
        n_fft=512,
        hop_length=128,
        pooling="attention",
        n_classes=35,
        dropout=0.1,
    ),
    # Time Series
    "etth1": TimeSeriesConfig(
        dataset_name="etth1",
        d_complex=64,
        n_layers=4,
        seq_len=96,
        pred_len=96,
        d_input=7,  # Default for ETTh1 multivariate
        pooling="mean",
        dropout=0.1,
    ),
    # NLP
    "sst2": NLPConfig(
        dataset_name="sst2",
        d_complex=64,
        n_layers=2,
        vocab_size=20000,
        n_classes=2,
        max_seq_len=64,
        pooling="mean",
    ),
    "imdb": NLPConfig(
        dataset_name="imdb",
        d_complex=128,
        n_layers=3,
        vocab_size=30000,
        n_classes=2,
        max_seq_len=256,
        pooling="attention",
    ),
}


def get_preset(name: str):
    """Get a preset configuration by name."""
    if name not in PRESETS:
        raise ValueError(f"Unknown preset: {name}. Available: {list(PRESETS.keys())}")
    return PRESETS[name]
