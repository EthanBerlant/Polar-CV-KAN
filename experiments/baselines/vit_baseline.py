"""
ViT-Tiny Baseline for CIFAR-10 Image Classification.

Designed to match parameter count of CV-KAN models (~50k-300k parameters).
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data import create_cifar10_dataloader

from .base_trainer import run_baseline


class ViT(nn.Module):
    """Vision Transformer with configurable dimensions."""

    def __init__(
        self,
        image_size=32,
        patch_size=4,
        num_classes=10,
        dim=64,
        depth=6,
        heads=4,
        mlp_dim=128,
        dropout=0.1,
    ):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size**2
        self.patch_size = patch_size

        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.to_cls_token = nn.Identity()

        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, img):
        p = self.patch_size
        x = (
            img.unfold(2, p, p)
            .unfold(3, p, p)
            .permute(0, 2, 3, 1, 4, 5)
            .contiguous()
            .view(img.shape[0], -1, p * p * 3)
        )
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape

        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)


def create_dataloaders(args):
    """Create CIFAR-10 dataloaders."""
    train_loader, val_loader, test_loader, n_classes = create_cifar10_dataloader(
        root=getattr(args, "data_root", "./data/cifar10"),
        batch_size=args.batch_size,
        image_size=32,
        subset_size=getattr(args, "subset_size", None),
    )
    return train_loader, val_loader, test_loader, {"n_classes": n_classes}


def create_model(args, metadata):
    """Create ViT model."""
    return ViT(
        image_size=32,
        patch_size=4,
        num_classes=metadata["n_classes"],
        dim=args.d_complex,
        depth=args.n_layers,
        heads=4,
        mlp_dim=args.d_complex * 2,
    )


def classification_accuracy(outputs, targets):
    """Compute classification accuracy."""
    _, predicted = outputs.max(1)
    return 100.0 * predicted.eq(targets).sum().item() / targets.size(0)


if __name__ == "__main__":
    run_baseline(
        model_class=ViT,
        domain="image",
        create_dataloaders=create_dataloaders,
        create_model=create_model,
        loss_fn=F.cross_entropy,
        metric_fn=classification_accuracy,
        metric_name="accuracy",
        metric_mode="max",
        default_args={"d_complex": 64, "n_layers": 6, "output_dir": "outputs/baselines/image"},
    )
