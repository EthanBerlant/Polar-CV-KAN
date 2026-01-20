"""
Image classification domain (CIFAR-10).
"""

import torch.nn.functional as F

from src.data import create_cifar10_dataloader
from src.models.cv_kan_image import CVKANImageClassifier
from src.trainer import BaseTrainer

DEFAULTS = {
    "batch_size": 128,
    "d_complex": 256,
    "n_layers": 6,
    "epochs": 100,
    "weight_decay": 0.05,
    "metric_name": "accuracy",
    "metric_mode": "max",
}


def add_args(parser):
    """Add image-specific arguments."""
    parser.add_argument("--data_root", type=str, default="./data/cifar10")
    parser.add_argument("--img_size", type=int, default=32)
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--embedding_type", type=str, default="linear", choices=["linear", "conv"])
    return parser


class ImageTrainer(BaseTrainer):
    """Trainer for image classification."""

    def train_step(self, batch):
        images, labels = batch
        images, labels = images.to(self.device), labels.to(self.device)

        outputs = self.model(images)
        logits = outputs["logits"]

        loss = F.cross_entropy(logits, labels)

        _, predicted = logits.max(1)
        total = labels.size(0)
        correct = predicted.eq(labels).sum().item()
        accuracy = 100.0 * correct / total

        return {"loss": loss, "accuracy": accuracy}

    def validate_step(self, batch):
        images, labels = batch
        images, labels = images.to(self.device), labels.to(self.device)

        outputs = self.model(images)
        logits = outputs["logits"]

        loss = F.cross_entropy(logits, labels)

        _, predicted = logits.max(1)
        total = labels.size(0)
        correct = predicted.eq(labels).sum().item()
        accuracy = 100.0 * correct / total

        return {"loss": loss, "accuracy": accuracy}


def create_model(args):
    """Create image classification model."""
    return CVKANImageClassifier(
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_channels=3,
        d_complex=args.d_complex,
        n_layers=args.n_layers,
        n_classes=10,  # CIFAR-10
        kan_hidden=args.kan_hidden,
        pos_encoding=args.pos_encoding,
        pooling=args.pooling,
        embedding_type=args.embedding_type,
    )


def create_dataloaders(args):
    """Create CIFAR-10 dataloaders."""
    train_loader, val_loader, test_loader, n_classes = create_cifar10_dataloader(
        root=args.data_root,
        batch_size=args.batch_size,
        image_size=args.img_size,
        subset_size=args.subset_size,
    )
    return train_loader, val_loader, test_loader, {"n_classes": n_classes}
