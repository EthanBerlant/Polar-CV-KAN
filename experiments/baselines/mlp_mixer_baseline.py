"""
MLP-Mixer Baseline for Image Classification.

An attention-free vision architecture that uses only MLPs.
Sized to match CV-KAN parameter count (~200-500k params).

Reference: "MLP-Mixer: An all-MLP Architecture for Vision"
https://arxiv.org/abs/2105.01601
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.image_data import (
    create_cifar10_dataloader,
    create_cifar100_dataloader,
    create_fashionmnist_dataloader,
)


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
    """
    A single Mixer block with token-mixing and channel-mixing MLPs.
    """

    def __init__(self, num_patches, hidden_dim, tokens_mlp_dim, channels_mlp_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.token_mixing = MlpBlock(num_patches, tokens_mlp_dim, dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.channel_mixing = MlpBlock(hidden_dim, channels_mlp_dim, dropout)

    def forward(self, x):
        # Token mixing: transpose to (batch, hidden_dim, num_patches), apply MLP, transpose back
        y = self.norm1(x)
        y = y.transpose(1, 2)  # (B, hidden_dim, num_patches)
        y = self.token_mixing(y)
        y = y.transpose(1, 2)  # (B, num_patches, hidden_dim)
        x = x + y

        # Channel mixing
        y = self.norm2(x)
        y = self.channel_mixing(y)
        x = x + y

        return x


class MLPMixer(nn.Module):
    """
    MLP-Mixer model for image classification.

    Sized to have ~200-500k parameters to match CV-KAN.
    """

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
        # Patch embedding: (B, C, H, W) -> (B, num_patches, hidden_dim)
        x = self.patch_embed(x)  # (B, hidden_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, hidden_dim)

        # Mixer blocks
        x = self.mixer_blocks(x)

        # Global average pooling and classification
        x = self.norm(x)
        x = x.mean(dim=1)  # (B, hidden_dim)
        x = self.head(x)

        return x


def parse_args():
    parser = argparse.ArgumentParser(description="MLP-Mixer Baseline for Image Classification")
    parser.add_argument(
        "--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100", "fashionmnist"]
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--d_complex", type=int, default=128, help="Ignored, for compatibility")
    parser.add_argument("--n_layers", type=int, default=4, help="Ignored, for compatibility")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subset_size", type=int, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/baselines/mlpmixer")
    parser.add_argument("--amp", action="store_true", help="Use automatic mixed precision")
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch, scaler=None):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f"Train Epoch {epoch}")
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss = F.cross_entropy(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({"loss": total_loss / (batch_idx + 1), "acc": 100.0 * correct / total})

    scheduler.step()
    return total_loss / len(dataloader), 100.0 * correct / total


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return total_loss / len(dataloader), 100.0 * correct / total


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Determine image size based on dataset
    if args.dataset == "fashionmnist":
        image_size = 28
    else:
        image_size = 32

    # Create dataloaders
    if args.dataset == "cifar10":
        train_loader, val_loader, test_loader, num_classes = create_cifar10_dataloader(
            batch_size=args.batch_size, subset_size=args.subset_size
        )
    elif args.dataset == "cifar100":
        train_loader, val_loader, test_loader, num_classes = create_cifar100_dataloader(
            batch_size=args.batch_size, subset_size=args.subset_size
        )
    else:  # fashionmnist
        train_loader, val_loader, test_loader, num_classes = create_fashionmnist_dataloader(
            batch_size=args.batch_size, subset_size=args.subset_size
        )

    # Create model
    model = MLPMixer(
        image_size=image_size,
        patch_size=args.patch_size,
        num_classes=num_classes,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        tokens_mlp_dim=args.hidden_dim // 2,
        channels_mlp_dim=args.hidden_dim * 2,
        dropout=args.dropout,
    ).to(device)

    n_params = count_parameters(model)
    print(f"Model parameters: {n_params:,}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    scaler = torch.amp.GradScaler("cuda") if args.amp and device.type == "cuda" else None

    # Training
    best_val_acc = 0
    patience_counter = 0

    run_name = args.run_name or f"mlpmixer_{args.dataset}_s{args.seed}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, device, epoch, scaler
        )
        val_loss, val_acc = evaluate(model, val_loader, device)

        print(f"Epoch {epoch}: Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    train_time = time.time() - start_time

    # Test evaluation
    model.load_state_dict(torch.load(output_dir / "best_model.pt", weights_only=True))
    test_loss, test_acc = evaluate(model, test_loader, device)

    print(f"\nTest Accuracy: {test_acc:.2f}%")

    # Save results
    results = {
        "dataset": args.dataset,
        "model": "MLP-Mixer",
        "n_params": n_params,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "train_time_seconds": train_time,
        "epochs_trained": epoch,
        "args": vars(args),
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
