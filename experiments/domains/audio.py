"""
Audio classification domain (Speech Commands).
"""

import torch.nn.functional as F

from src.configs.model import AudioConfig
from src.configs.training import TrainingConfig
from src.data import (
    TORCHAUDIO_AVAILABLE,
    create_audio_dataloader,  # SpeechCommands
    create_esc50_dataloader,
    create_urbansound8k_dataloader,
)
from src.models.cv_kan_audio import CVKANAudio
from src.trainer import BaseTrainer


def create_model(config: AudioConfig):
    """Create audio classification model."""
    return CVKANAudio(
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        d_complex=config.d_complex,
        n_layers=config.n_layers,
        n_classes=config.n_classes,
        kan_hidden=config.kan_hidden,
        task="classification",  # Currently only supporting classification in trainer
        pooling=config.pooling,
        use_stft_frontend=True,
        dropout=config.dropout,
        center_magnitudes=config.center_magnitudes,
    )


def create_dataloaders(model_config: AudioConfig, train_config: TrainingConfig):
    """Create audio dataloaders based on config.dataset_name."""
    if not TORCHAUDIO_AVAILABLE:
        raise RuntimeError("torchaudio is required for audio domain")

    kwargs = {
        "batch_size": train_config.batch_size,
        "subset_size": train_config.subset_size,
    }

    dataset = model_config.dataset_name.lower()
    data_root = "./data"  # Constant or derived

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

    return train, val, test, {"n_classes": n_classes}


class AudioTrainer(BaseTrainer):
    """Trainer for audio classification."""

    def train_step(self, batch):
        waveforms, labels = batch
        waveforms = waveforms.squeeze(1).to(self.device)
        labels = labels.to(self.device)

        outputs = self.model(waveforms)
        logits = outputs["logits"]

        loss = F.cross_entropy(logits, labels)

        _, predicted = logits.max(1)
        total = labels.size(0)
        correct = predicted.eq(labels).sum().item()
        accuracy = 100.0 * correct / total

        return {"loss": loss, "accuracy": accuracy}

    def validate_step(self, batch):
        waveforms, labels = batch
        waveforms = waveforms.squeeze(1).to(self.device)
        labels = labels.to(self.device)

        outputs = self.model(waveforms)
        logits = outputs["logits"]

        loss = F.cross_entropy(logits, labels)

        _, predicted = logits.max(1)
        total = labels.size(0)
        correct = predicted.eq(labels).sum().item()
        accuracy = 100.0 * correct / total

        return {"loss": loss, "accuracy": accuracy}


Trainer = AudioTrainer
