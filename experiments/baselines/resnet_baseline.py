"""
ResNet Baseline for Image Classification.

A small ResNet variant sized to match CV-KAN parameter count (~200-500k params).
"""

import sys
from pathlib import Path

import torch.nn.functional as F
from torch import nn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data import create_cifar10_dataloader

from .base_trainer import run_baseline


class BasicBlock(nn.Module):
    """Basic ResNet block with skip connection."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetSmall(nn.Module):
    """Small ResNet for CIFAR-sized images (32x32)."""

    def __init__(self, num_classes=10, base_channels=32, num_blocks=None):
        super().__init__()
        if num_blocks is None:
            num_blocks = [2, 2, 2]
        self.in_channels = base_channels

        # Initial convolution
        self.conv1 = nn.Conv2d(3, base_channels, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)

        # Residual layers
        self.layer1 = self._make_layer(base_channels, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(base_channels * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(base_channels * 4, num_blocks[2], stride=2)

        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_channels * 4, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def create_dataloaders(args):
    """Create CIFAR-10 dataloaders."""
    train_loader, val_loader, test_loader, n_classes = create_cifar10_dataloader(
        batch_size=args.batch_size,
        subset_size=getattr(args, "subset_size", None),
    )
    return train_loader, val_loader, test_loader, {"n_classes": n_classes}


def create_model(args, metadata):
    """Create ResNet model."""
    return ResNetSmall(
        num_classes=metadata["n_classes"],
        base_channels=32,  # Fixed for parameter matching
    )


def classification_accuracy(outputs, targets):
    """Compute classification accuracy."""
    _, predicted = outputs.max(1)
    return 100.0 * predicted.eq(targets).sum().item() / targets.size(0)


if __name__ == "__main__":
    run_baseline(
        model_class=ResNetSmall,
        domain="image",
        create_dataloaders=create_dataloaders,
        create_model=create_model,
        loss_fn=F.cross_entropy,
        metric_fn=classification_accuracy,
        metric_name="accuracy",
        metric_mode="max",
        default_args={"d_complex": 128, "n_layers": 6, "output_dir": "outputs/baselines/image"},
    )
