"""
Image classification domain (CIFAR-10).
"""

import torch.nn.functional as F

from src.configs.model import ImageConfig
from src.configs.training import TrainingConfig
from src.data import (
    create_cifar10_dataloader,
    create_cifar100_dataloader,
    create_fashionmnist_dataloader,
    create_tinyimagenet_dataloader,
)
from src.models.cv_kan_image import CVKANImageClassifier
from src.trainer import BaseTrainer


def create_model(config: ImageConfig):
    """Create image classification model."""
    return CVKANImageClassifier(
        img_size=config.img_size,
        patch_size=config.patch_size,
        in_channels=config.in_channels,
        d_complex=config.d_complex,
        n_layers=config.n_layers,
        n_classes=config.n_classes,
        kan_hidden=config.kan_hidden,
        pos_encoding="sinusoidal" if config.pos_encoding else None,
        pooling=config.pooling,
        embedding_type=config.embedding_type,
        center_magnitudes=config.center_magnitudes,
        dropout=config.dropout,
    )


def create_dataloaders(model_config: ImageConfig, train_config: TrainingConfig):
    """Create image dataloaders based on config.dataset_name."""
    kwargs = {
        "batch_size": train_config.batch_size,
        "image_size": model_config.img_size,
        "subset_size": train_config.subset_size,
    }

    # Determine root based on dataset
    data_root = "./data"  # Constant for now

    dataset = model_config.dataset_name.upper()

    if dataset == "CIFAR10":
        root = f"{data_root}/cifar10"
        train, val, test, n_classes = create_cifar10_dataloader(root=root, **kwargs)
    elif dataset == "CIFAR100":
        root = f"{data_root}/cifar100"
        train, val, test, n_classes = create_cifar100_dataloader(root=root, **kwargs)
    elif dataset == "FASHIONMNIST":
        root = f"{data_root}/fashionmnist"
        train, val, test, n_classes = create_fashionmnist_dataloader(root=root, **kwargs)
    elif dataset == "TINYIMAGENET":
        root = f"{data_root}/tinyimagenet"
        train, val, test, n_classes = create_tinyimagenet_dataloader(root=root, **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return train, val, test, {"n_classes": n_classes}


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


Trainer = ImageTrainer
