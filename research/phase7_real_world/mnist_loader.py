from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def get_mnist_loaders(root="./data", batch_size=64, subset_size=None, rotation_range=None):
    """Get MNIST loaders.

    Args:
        root: Data directory
        batch_size: Batch size
        subset_size: If not None, use only N samples (for debugging)
        rotation_range: Tuple (min_deg, max_deg) for random rotation augmentation.
                       If None, no rotation is applied.
    """
    # Transforms
    trans_list = [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]

    if rotation_range:
        # Add random rotation
        trans_list.insert(0, transforms.RandomRotation(degrees=rotation_range))

    transform = transforms.Compose(trans_list)

    # Download and load
    # Use standard train set for training
    train_dataset = datasets.MNIST(root, train=True, download=True, transform=transform)

    # Use standard test set for testing
    # Note: If we want to test robustness, we usually want to Apply rotation to Test set too?
    # Or train on Upright, Test on Rotated?
    # This function applies transform to the dataset it creates.
    # So if we call this for training, we get rotated training data?
    # We should probably separate train/test transform logic if we want "Train Upright / Test Rotated".

    # Let's simplify: This function returns ONE pair of loaders with THIS transform.
    # So to do the experiment, we call it twice with different transforms? No, dataset is cached.

    # Better: return TRAIN and TEST datasets with potentially DIFFERENT transforms.


def create_robustness_loaders(root="./data", batch_size=64, train_subset=None):
    """Create specific loaders for the robustness experiment:
    1. Train: Upright (Standard MNIST)
    2. Test A: Upright (Standard MNIST Test)
    3. Test B: Rotated (Standard MNIST Test + Random Rotation)
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    # 1. Train Transform (Upright + slight augmentation maybe? No, let's keep strict for baseline)
    train_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    # 2. Test A Transform (Upright)
    test_a_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    # 3. Test B Transform (Rotated 0-360)
    test_b_transform = transforms.Compose(
        [
            transforms.RandomRotation(degrees=(0, 360)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    # Datasets
    train_ds = datasets.MNIST(root, train=True, download=True, transform=train_transform)
    test_ds_ups = datasets.MNIST(root, train=False, download=True, transform=test_a_transform)
    test_ds_rot = datasets.MNIST(root, train=False, download=True, transform=test_b_transform)

    if train_subset:
        indices = torch.randperm(len(train_ds))[:train_subset]
        train_ds = Subset(train_ds, indices)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader_up = DataLoader(test_ds_ups, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader_rot = DataLoader(test_ds_rot, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader_up, test_loader_rot
