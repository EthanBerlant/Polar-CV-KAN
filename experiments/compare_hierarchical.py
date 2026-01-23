"""
Compare flat vs hierarchical polarization architectures.

This experiment tests whether hierarchical/recursive polarization
(inspired by polar error correction) can match or beat flat architectures,
potentially without needing attention-based aggregation.

Usage:
    python experiments/compare_hierarchical.py --epochs 3
    python experiments/compare_hierarchical.py --domain text --epochs 5
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import SignalNoiseDataset
from src.modules import HierarchicalPolarization, PolarizingBlock


class FlatModel(nn.Module):
    """Standard flat CV-KAN: stack of PolarizingBlocks with global aggregation."""

    def __init__(self, d_complex: int, n_layers: int, n_classes: int):
        super().__init__()
        self.layers = nn.ModuleList([PolarizingBlock(d_complex) for _ in range(n_layers)])
        self.classifier = nn.Linear(d_complex * 2, n_classes)  # Real+Imag

    def forward(self, Z):
        for layer in self.layers:
            Z = layer(Z)
        # Pool and classify
        pooled = Z.mean(dim=1)  # (batch, d_complex)
        features = torch.cat([pooled.real, pooled.imag], dim=-1)
        return self.classifier(features)


class HierarchicalModel(nn.Module):
    """Hierarchical CV-KAN: single HierarchicalPolarization block."""

    def __init__(
        self,
        d_complex: int,
        n_classes: int,
        weight_sharing: str = "per_level",
        aggregation: str = "mean",
        top_down: str = "none",
    ):
        super().__init__()
        self.hierarchical = HierarchicalPolarization(
            d_complex=d_complex,
            weight_sharing=weight_sharing,
            aggregation=aggregation,
            top_down=top_down,
        )
        self.classifier = nn.Linear(d_complex * 2, n_classes)

    def forward(self, Z):
        Z = self.hierarchical(Z)
        pooled = Z.mean(dim=1)
        features = torch.cat([pooled.real, pooled.imag], dim=-1)
        return self.classifier(features)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in dataloader:
        Z = batch["sequence"].to(device)
        labels = batch["sequence_label"].to(device).long()

        optimizer.zero_grad()
        logits = model(Z)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(dataloader), correct / total


def evaluate(model, dataloader, device):
    """Evaluate model."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            Z = batch["sequence"].to(device)
            labels = batch["sequence_label"].to(device).long()

            logits = model(Z)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


def run_experiment(
    model_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    device: torch.device,
    lr: float = 1e-3,
):
    """Run training experiment for a single model."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print(f"\n{'=' * 60}")
    print(f"Model: {model_name}")
    print(f"Parameters: {count_parameters(model):,}")
    print(f"{'=' * 60}")

    results = []
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_acc = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch}/{epochs}: "
            f"Loss={train_loss:.4f}, Train={train_acc:.1%}, Val={val_acc:.1%}"
        )

        results.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
            }
        )

    elapsed = time.time() - start_time
    print(f"Time: {elapsed:.1f}s")

    return results


def main():
    parser = argparse.ArgumentParser(description="Compare flat vs hierarchical CV-KAN")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--d-complex", type=int, default=32, help="Complex dimension")
    parser.add_argument("--n-layers", type=int, default=3, help="Layers for flat model")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=16, help="Sequence length")
    parser.add_argument("--n-samples", type=int, default=5000, help="Training samples")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    args = parser.parse_args()

    # Device setup
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Create synthetic dataset with balanced classes
    print("\nCreating balanced signal/noise dataset...")

    # Positive samples (containing signal)
    train_pos = SignalNoiseDataset(
        n_samples=args.n_samples // 2,
        n_tokens=args.seq_len,
        k_signal=4,
        d_complex=args.d_complex,
    )
    # Negative samples (pure noise)
    train_neg = SignalNoiseDataset(
        n_samples=args.n_samples // 2,
        n_tokens=args.seq_len,
        k_signal=0,  # No signal
        d_complex=args.d_complex,
    )
    train_dataset = torch.utils.data.ConcatDataset([train_pos, train_neg])

    # Validation set
    val_pos = SignalNoiseDataset(
        n_samples=args.n_samples // 10,
        n_tokens=args.seq_len,
        k_signal=4,
        d_complex=args.d_complex,
    )
    val_neg = SignalNoiseDataset(
        n_samples=args.n_samples // 10,
        n_tokens=args.seq_len,
        k_signal=0,
        d_complex=args.d_complex,
    )
    val_dataset = torch.utils.data.ConcatDataset([val_pos, val_neg])

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Models to compare
    models = {
        "Flat (mean agg)": FlatModel(
            d_complex=args.d_complex,
            n_layers=args.n_layers,
            n_classes=2,
        ),
        "Hierarchical (mean, shared)": HierarchicalModel(
            d_complex=args.d_complex,
            n_classes=2,
            weight_sharing="shared",
            aggregation="mean",
        ),
        "Hierarchical (mean, per-level)": HierarchicalModel(
            d_complex=args.d_complex,
            n_classes=2,
            weight_sharing="per_level",
            aggregation="mean",
        ),
        "Hierarchical (mag-weighted, per-level)": HierarchicalModel(
            d_complex=args.d_complex,
            n_classes=2,
            weight_sharing="per_level",
            aggregation="magnitude_weighted",
        ),
        "Hierarchical (mean, top-down)": HierarchicalModel(
            d_complex=args.d_complex,
            n_classes=2,
            weight_sharing="per_level",
            aggregation="mean",
            top_down="mirror",
        ),
    }

    # Run experiments
    all_results = {}
    for name, model in models.items():
        results = run_experiment(name, model, train_loader, val_loader, args.epochs, device)
        all_results[name] = results

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Model':<40} {'Params':>10} {'Val Acc':>10}")
    print("-" * 60)

    for name, model in models.items():
        params = count_parameters(model)
        final_acc = all_results[name][-1]["val_acc"]
        print(f"{name:<40} {params:>10,} {final_acc:>10.1%}")


if __name__ == "__main__":
    main()
