"""
LSTM Baseline for Audio Classification.

Bidirectional LSTM on mel-spectrograms.
Sized to match CV-KAN parameter count (~200-500k params).
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.audio_data import TORCHAUDIO_AVAILABLE, create_audio_dataloader

if TORCHAUDIO_AVAILABLE:
    import torchaudio

from .base_trainer import run_baseline


class AudioLSTM(nn.Module):
    """Bidirectional LSTM for audio classification on spectrograms."""

    def __init__(
        self,
        n_mels=64,
        num_classes=35,
        hidden_dim=128,
        num_layers=2,
        dropout=0.1,
        bidirectional=True,
    ):
        super().__init__()

        # Spectrogram frontend
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=1024, hop_length=512, n_mels=n_mels
        )

        # LSTM
        self.lstm = nn.LSTM(
            input_size=n_mels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # Classification head
        lstm_out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_out_dim),
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim, num_classes),
        )

    def forward(self, waveform):
        # Compute mel spectrogram
        with torch.no_grad():
            spec = self.mel_spec(waveform.squeeze(1))
            spec = torch.log(spec + 1e-9)

        x = spec.transpose(1, 2)
        x, _ = self.lstm(x)
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x


def create_dataloaders(args):
    """Create Speech Commands dataloaders."""
    if not TORCHAUDIO_AVAILABLE:
        raise RuntimeError("torchaudio is required for audio baseline")

    train_loader, val_loader, test_loader, n_classes = create_audio_dataloader(
        root=getattr(args, "data_root", "./data/speech_commands"),
        batch_size=args.batch_size,
        subset_size=getattr(args, "subset_size", None),
    )
    return train_loader, val_loader, test_loader, {"n_classes": n_classes}


def create_model(args, metadata):
    """Create AudioLSTM model."""
    return AudioLSTM(
        n_mels=64,
        num_classes=metadata["n_classes"],
        hidden_dim=args.d_complex,
        num_layers=args.n_layers,
    )


def classification_accuracy(outputs, targets):
    """Compute classification accuracy."""
    _, predicted = outputs.max(1)
    return 100.0 * predicted.eq(targets).sum().item() / targets.size(0)


if __name__ == "__main__":
    run_baseline(
        model_class=AudioLSTM,
        domain="audio",
        create_dataloaders=create_dataloaders,
        create_model=create_model,
        loss_fn=F.cross_entropy,
        metric_fn=classification_accuracy,
        metric_name="accuracy",
        metric_mode="max",
        default_args={
            "d_complex": 128,
            "n_layers": 2,
            "epochs": 50,
            "output_dir": "outputs/baselines/audio",
        },
    )
