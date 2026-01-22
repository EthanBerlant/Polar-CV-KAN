"""
M5 CNN Baseline for Speech Commands Audio Classification.
Based on "Very Deep Convolutional Neural Networks for Raw Waveforms" (Dai et al., 2017).
"""

import sys
from pathlib import Path

import torch.nn.functional as F
from torch import nn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data import TORCHAUDIO_AVAILABLE
from src.data.audio_data import create_audio_dataloader

from .base_trainer import run_baseline


class M5(nn.Module):
    """M5 CNN architecture for audio classification from raw waveforms."""

    def __init__(self, n_input=1, n_output=35, stride=16, n_channel=32):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)

        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)

        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)

        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)

        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        # x: (batch, input_len) -> (batch, 1, input_len)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)

        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)

        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)

        # Global Average Pooling
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2).squeeze(1)


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
    """Create M5 model."""
    return M5(
        n_input=1,
        n_output=metadata["n_classes"],
        n_channel=args.d_complex,
    )


def nll_loss(outputs, targets):
    """NLL loss for audio classification."""
    return F.nll_loss(outputs, targets)


def classification_accuracy(outputs, targets):
    """Compute classification accuracy."""
    _, predicted = outputs.max(1)
    return 100.0 * predicted.eq(targets).sum().item() / targets.size(0)


if __name__ == "__main__":
    run_baseline(
        model_class=M5,
        domain="audio",
        create_dataloaders=create_dataloaders,
        create_model=create_model,
        loss_fn=nll_loss,
        metric_fn=classification_accuracy,
        metric_name="accuracy",
        metric_mode="max",
        default_args={
            "d_complex": 32,
            "n_layers": 4,
            "epochs": 30,
            "lr": 1e-2,
            "weight_decay": 0.0001,
            "output_dir": "outputs/baselines/audio",
        },
    )
