"""
Audio classification domain (Speech Commands).

Supports two modes:
1. Raw waveform mode (default): STFT computed on every forward pass
2. Precomputed mode: Uses cached spectrograms for ~2-3x faster training
   Run `python experiments/precompute_audio.py` first to generate cache.
"""

import os

import torch
import torch.nn.functional as F

from src.configs.model import AudioConfig
from src.configs.training import TrainingConfig
from src.data import create_audio_dataloader  # SpeechCommands
from src.data import (
    TORCHAUDIO_AVAILABLE,
    create_esc50_dataloader,
    create_precomputed_audio_dataloader,
    create_urbansound8k_dataloader,
)
from src.models.cv_kan_audio import CVKANAudio
from src.trainer import BaseTrainer


def create_model(config: AudioConfig, use_precomputed: bool = False):
    """Create audio classification model.

    Args:
        config: Audio configuration
        use_precomputed: If True, model expects precomputed spectrograms (2, time, freq)
                         If False, model expects raw waveforms
    """
    return CVKANAudio(
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        d_complex=config.d_complex,
        n_layers=config.n_layers,
        n_classes=config.n_classes,
        kan_hidden=config.kan_hidden,
        task="classification",
        pooling=config.pooling,
        use_stft_frontend=not use_precomputed,  # Skip STFT if precomputed
        dropout=config.dropout,
        center_magnitudes=config.center_magnitudes,
    )


def create_dataloaders(model_config: AudioConfig, train_config: TrainingConfig):
    """Create audio dataloaders based on config.dataset_name.

    Automatically uses precomputed spectrograms if available.
    """
    dataset = model_config.dataset_name.lower()
    data_root = "./data"

    # Check for precomputed spectrograms
    precomputed_root = f"{data_root}/speech_commands_stft"
    use_precomputed = (
        dataset == "speech_commands"
        and os.path.exists(precomputed_root)
        and os.path.exists(os.path.join(precomputed_root, "training"))
    )

    if use_precomputed:
        print("Using precomputed spectrograms for faster training!")
        train, val, test, n_classes = create_precomputed_audio_dataloader(
            root=precomputed_root,
            batch_size=train_config.batch_size,
        )
        return train, val, test, {"n_classes": n_classes, "use_precomputed": True}

    # Fall back to raw audio
    if not TORCHAUDIO_AVAILABLE:
        raise RuntimeError("torchaudio is required for audio domain")

    kwargs = {
        "batch_size": train_config.batch_size,
        "subset_size": train_config.subset_size,
    }

    if dataset == "speech_commands":
        root = f"{data_root}/speech_commands"
        train, val, test, n_classes = create_audio_dataloader(root=root, download=True, **kwargs)
    elif dataset == "esc50":
        root = f"{data_root}/esc50"
        train, val, test, n_classes = create_esc50_dataloader(root=root, download=True, **kwargs)
    elif dataset == "urbansound8k":
        root = f"{data_root}/urbansound8k"
        train, val, test, n_classes = create_urbansound8k_dataloader(
            root=root, download=False, **kwargs
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return train, val, test, {"n_classes": n_classes, "use_precomputed": False}


class AudioTrainer(BaseTrainer):
    """Trainer for audio classification.

    Handles both raw waveforms and precomputed spectrograms.
    """

    def __init__(self, *args, use_precomputed: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_precomputed = use_precomputed

    def _process_input(self, batch):
        """Process batch based on input mode."""
        inputs, labels = batch
        labels = labels.to(self.device)

        if self.use_precomputed:
            # Precomputed: (batch, 2, time, freq) -> complex (batch, time, freq)
            inputs = inputs.to(self.device)
            inputs = torch.complex(inputs[:, 0], inputs[:, 1])
        else:
            # Raw waveforms: squeeze channel dim
            inputs = inputs.squeeze(1).to(self.device)

        return inputs, labels

    def train_step(self, batch):
        inputs, labels = self._process_input(batch)

        outputs = self.model(inputs)
        logits = outputs["logits"]

        loss = F.cross_entropy(logits, labels)

        _, predicted = logits.max(1)
        total = labels.size(0)
        correct = predicted.eq(labels).sum().item()
        accuracy = 100.0 * correct / total

        return {"loss": loss, "accuracy": accuracy}

    def validate_step(self, batch):
        inputs, labels = self._process_input(batch)

        outputs = self.model(inputs)
        logits = outputs["logits"]

        loss = F.cross_entropy(logits, labels)

        _, predicted = logits.max(1)
        total = labels.size(0)
        correct = predicted.eq(labels).sum().item()
        accuracy = 100.0 * correct / total

        return {"loss": loss, "accuracy": accuracy}


Trainer = AudioTrainer
