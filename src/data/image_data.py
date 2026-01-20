import os

import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def create_cifar10_dataloader(
    root: str = "./data/cifar10",
    batch_size: int = 64,
    image_size: int = 32,
    num_workers: int = 2,
    subset_size: int = None,
    download: bool = True,
):
    """
    Create CIFAR-10 dataloaders.

    Args:
        root: Data directory
        batch_size: Batch size
        image_size: Resize images to this size (CIFAR is naturally 32x32)
        num_workers: DataLoader workers
        subset_size: If set, use only N samples (for debugging/piloting)
        download: Whether to download dataset

    Returns:
        train_loader, val_loader, test_loader, num_classes
    """

    transform_train = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    os.makedirs(root, exist_ok=True)

    train_dataset = datasets.CIFAR10(
        root=root, train=True, download=download, transform=transform_train
    )
    test_dataset = datasets.CIFAR10(
        root=root, train=False, download=download, transform=transform_test
    )

    # Split train into train/val (90/10)
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(0.1 * num_train))

    np.random.shuffle(indices)

    if subset_size:
        train_idx, val_idx = (
            indices[split : split + subset_size],
            indices[: min(split, subset_size // 10)],
        )
        test_indices = list(range(len(test_dataset)))[: subset_size // 5]
        test_dataset = Subset(test_dataset, test_indices)
    else:
        train_idx, val_idx = indices[split:], indices[:split]

    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(train_dataset, val_idx)

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader, 10


def create_cifar100_dataloader(
    root: str = "./data/cifar100",
    batch_size: int = 64,
    image_size: int = 32,
    num_workers: int = 2,
    subset_size: int = None,
    download: bool = True,
):
    """
    Create CIFAR-100 dataloaders (100 fine-grained classes).

    Args:
        root: Data directory
        batch_size: Batch size
        image_size: Resize images to this size (CIFAR is naturally 32x32)
        num_workers: DataLoader workers
        subset_size: If set, use only N samples (for debugging/piloting)
        download: Whether to download dataset

    Returns:
        train_loader, val_loader, test_loader, num_classes
    """

    transform_train = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )

    os.makedirs(root, exist_ok=True)

    train_dataset = datasets.CIFAR100(
        root=root, train=True, download=download, transform=transform_train
    )
    test_dataset = datasets.CIFAR100(
        root=root, train=False, download=download, transform=transform_test
    )

    # Split train into train/val (90/10)
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(0.1 * num_train))

    np.random.shuffle(indices)

    if subset_size:
        train_idx, val_idx = (
            indices[split : split + subset_size],
            indices[: min(split, subset_size // 10)],
        )
        test_indices = list(range(len(test_dataset)))[: subset_size // 5]
        test_dataset = Subset(test_dataset, test_indices)
    else:
        train_idx, val_idx = indices[split:], indices[:split]

    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(train_dataset, val_idx)

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader, 100


def create_fashionmnist_dataloader(
    root: str = "./data/fashionmnist",
    batch_size: int = 64,
    image_size: int = 28,
    num_workers: int = 2,
    subset_size: int = None,
    download: bool = True,
):
    """
    Create FashionMNIST dataloaders (10 grayscale clothing classes).

    Args:
        root: Data directory
        batch_size: Batch size
        image_size: Resize images to this size (FashionMNIST is naturally 28x28)
        num_workers: DataLoader workers
        subset_size: If set, use only N samples (for debugging/piloting)
        download: Whether to download dataset

    Returns:
        train_loader, val_loader, test_loader, num_classes
    """

    # Convert grayscale to 3 channels for consistency with other image datasets
    transform_train = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Grayscale to RGB
            transforms.Normalize((0.2860,) * 3, (0.3530,) * 3),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Grayscale to RGB
            transforms.Normalize((0.2860,) * 3, (0.3530,) * 3),
        ]
    )

    os.makedirs(root, exist_ok=True)

    train_dataset = datasets.FashionMNIST(
        root=root, train=True, download=download, transform=transform_train
    )
    test_dataset = datasets.FashionMNIST(
        root=root, train=False, download=download, transform=transform_test
    )

    # Split train into train/val (90/10)
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(0.1 * num_train))

    np.random.shuffle(indices)

    if subset_size:
        train_idx, val_idx = (
            indices[split : split + subset_size],
            indices[: min(split, subset_size // 10)],
        )
        test_indices = list(range(len(test_dataset)))[: subset_size // 5]
        test_dataset = Subset(test_dataset, test_indices)
    else:
        train_idx, val_idx = indices[split:], indices[:split]

    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(train_dataset, val_idx)

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader, 10
