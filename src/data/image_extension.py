from __future__ import annotations

import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def create_tinyimagenet_dataloader(
    root: str = "./data/tinyimagenet",
    batch_size: int = 64,
    image_size: int = 64,
    num_workers: int = 2,
    subset_size: int | None = None,
    download: bool = False,
):
    """Create TinyImageNet dataloaders from an ImageFolder-style directory."""
    train_dir = os.path.join(root, "train")
    val_dir = os.path.join(root, "val")

    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        raise FileNotFoundError(
            "TinyImageNet dataset not found. Expected train/ and val/ folders under "
            f"{root}."
        )

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform)

    if subset_size:
        train_dataset = torch.utils.data.Subset(train_dataset, range(subset_size))
        val_dataset = torch.utils.data.Subset(val_dataset, range(max(1, subset_size // 10)))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = val_loader
    return train_loader, val_loader, test_loader, len(train_dataset.classes)
