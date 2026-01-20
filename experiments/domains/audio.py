"""
Audio classification domain (Speech Commands).
"""

import torch.nn.functional as F

from src.data import TORCHAUDIO_AVAILABLE, create_audio_dataloader
from src.models.cv_kan_audio import CVKANAudio
from src.trainer import BaseTrainer

DEFAULTS = {
    "batch_size": 256,
    "d_complex": 128,
    "n_layers": 4,
    "epochs": 30,
    "metric_name": "accuracy",
    "metric_mode": "max",
}


def add_args(parser):
    """Add audio-specific arguments."""
    parser.add_argument("--data_root", type=str, default="./data/speech_commands")
    parser.add_argument("--n_fft", type=int, default=512)
    parser.add_argument("--hop_length", type=int, default=128)
    return parser


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


def create_model(args):
    """Create audio classification model."""
    return CVKANAudio(
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        d_complex=args.d_complex,
        n_layers=args.n_layers,
        n_classes=args.n_classes,
        kan_hidden=args.kan_hidden,
        task="classification",
        pooling=args.pooling,
        use_stft_frontend=True,
    )


def create_dataloaders(args):
    """Create Speech Commands dataloaders."""
    if not TORCHAUDIO_AVAILABLE:
        raise RuntimeError("torchaudio is required for audio domain")

    train_loader, val_loader, test_loader, n_classes = create_audio_dataloader(
        root=args.data_root,
        batch_size=args.batch_size,
        subset_size=args.subset_size,
        download=True,
    )
    return train_loader, val_loader, test_loader, {"n_classes": n_classes}
