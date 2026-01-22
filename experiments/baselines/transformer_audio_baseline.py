"""
Transformer Baseline for Audio Classification.

A standard Transformer encoder on mel-spectrograms.
Sized to match CV-KAN parameter count (~200-500k params).
"""

import math
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


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class AudioTransformer(nn.Module):
    """Transformer encoder for audio classification on spectrograms."""

    def __init__(
        self,
        n_mels=64,
        num_classes=35,
        d_model=128,
        nhead=4,
        num_layers=4,
        dim_feedforward=256,
        dropout=0.1,
    ):
        super().__init__()

        # Spectrogram frontend
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=1024, hop_length=512, n_mels=n_mels
        )

        self.input_proj = nn.Linear(n_mels, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, waveform):
        with torch.no_grad():
            spec = self.mel_spec(waveform.squeeze(1))
            spec = torch.log(spec + 1e-9)

        x = spec.transpose(1, 2)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
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
    """Create AudioTransformer model."""
    return AudioTransformer(
        n_mels=64,
        num_classes=metadata["n_classes"],
        d_model=args.d_complex,
        nhead=4,
        num_layers=args.n_layers,
        dim_feedforward=args.d_complex * 2,
    )


def classification_accuracy(outputs, targets):
    """Compute classification accuracy."""
    _, predicted = outputs.max(1)
    return 100.0 * predicted.eq(targets).sum().item() / targets.size(0)


if __name__ == "__main__":
    run_baseline(
        model_class=AudioTransformer,
        domain="audio",
        create_dataloaders=create_dataloaders,
        create_model=create_model,
        loss_fn=F.cross_entropy,
        metric_fn=classification_accuracy,
        metric_name="accuracy",
        metric_mode="max",
        default_args={
            "d_complex": 128,
            "n_layers": 4,
            "epochs": 50,
            "output_dir": "outputs/baselines/audio",
        },
    )
