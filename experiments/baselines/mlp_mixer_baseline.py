"""
MLP-Mixer Baseline for Image Classification.

An attention-free vision architecture that uses only MLPs.
Sized to match CV-KAN parameter count (~200-500k params).

Reference: "MLP-Mixer: An all-MLP Architecture for Vision"
https://arxiv.org/abs/2105.01601
"""

import sys
from pathlib import Path

import torch.nn.functional as F
from torch import nn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data import create_cifar10_dataloader

from .base_trainer import run_baseline


class MlpBlock(nn.Module):
    """A simple MLP block with GELU activation."""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class MixerBlock(nn.Module):
    """A single Mixer block with token-mixing and channel-mixing MLPs."""

    def __init__(self, num_patches, hidden_dim, tokens_mlp_dim, channels_mlp_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.token_mixing = MlpBlock(num_patches, tokens_mlp_dim, dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.channel_mixing = MlpBlock(hidden_dim, channels_mlp_dim, dropout)

    def forward(self, x):
        # Token mixing
        y = self.norm1(x)
        y = y.transpose(1, 2)
        y = self.token_mixing(y)
        y = y.transpose(1, 2)
        x = x + y

        # Channel mixing
        y = self.norm2(x)
        y = self.channel_mixing(y)
        x = x + y

        return x


class MLPMixer(nn.Module):
    """MLP-Mixer model for image classification."""

    def __init__(
        self,
        image_size=32,
        patch_size=4,
        num_classes=10,
        hidden_dim=128,
        depth=4,
        tokens_mlp_dim=64,
        channels_mlp_dim=256,
        dropout=0.0,
    ):
        super().__init__()

        assert image_size % patch_size == 0
        self.num_patches = (image_size // patch_size) ** 2

        # Patch embedding
        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, hidden_dim, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(2),
        )

        # Mixer blocks
        self.mixer_blocks = nn.Sequential(
            *[
                MixerBlock(self.num_patches, hidden_dim, tokens_mlp_dim, channels_mlp_dim, dropout)
                for _ in range(depth)
            ]
        )

        # Classification head
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.transpose(1, 2)
        x = self.mixer_blocks(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        x = self.head(x)
        return x


def create_dataloaders(args):
    """Create CIFAR-10 dataloaders."""
    train_loader, val_loader, test_loader, n_classes = create_cifar10_dataloader(
        batch_size=args.batch_size,
        subset_size=getattr(args, "subset_size", None),
    )
    return train_loader, val_loader, test_loader, {"n_classes": n_classes}


def create_model(args, metadata):
    """Create MLP-Mixer model."""
    return MLPMixer(
        image_size=32,
        patch_size=4,
        num_classes=metadata["n_classes"],
        hidden_dim=args.d_complex,
        depth=args.n_layers,
        tokens_mlp_dim=args.d_complex // 2,
        channels_mlp_dim=args.d_complex * 2,
    )


def classification_accuracy(outputs, targets):
    """Compute classification accuracy."""
    _, predicted = outputs.max(1)
    return 100.0 * predicted.eq(targets).sum().item() / targets.size(0)


if __name__ == "__main__":
    run_baseline(
        model_class=MLPMixer,
        domain="image",
        create_dataloaders=create_dataloaders,
        create_model=create_model,
        loss_fn=F.cross_entropy,
        metric_fn=classification_accuracy,
        metric_name="accuracy",
        metric_mode="max",
        default_args={"d_complex": 128, "n_layers": 4, "output_dir": "outputs/baselines/image"},
    )
