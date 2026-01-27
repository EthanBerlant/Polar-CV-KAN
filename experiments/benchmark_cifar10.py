"""
Benchmark Best-Arch on CIFAR-10.
Architecture: Hybrid Sharing + Magnitude-Weighted Aggregation.
"""

import argparse
import os
import sys
import time

import torch
from torch import nn

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.image_data import create_cifar10_dataloader
from src.models.cv_kan_image import CVKANImageClassifier


def train_eval(epochs=100, device="cuda"):
    print("\n=== CIFAR-10 Deep Residual CV-KAN (12 Layers) ===")

    # CIFAR-10 Data
    train_loader, val_loader, test_loader, n_classes = create_cifar10_dataloader(
        batch_size=128, num_workers=2
    )

    model = CVKANImageClassifier(
        img_size=32,
        patch_size=4,
        d_complex=64,  # Lean but deep
        n_layers=12,  # DEEP STACK
        n_classes=10,
        kan_hidden=64,
        embedding_type="conv",
        dropout=0.1,
        # Backbone Config
        block_type="polarizing",  # Standard stack for depth
        skip_connections=True,  # Residuals for 90%
        normalization="layer",  # More stable for depth
        aggregation_type="magnitude_weighted",
        center_magnitudes=False,
        mag_init_scale=0.01,  # Start gently
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
    # Scheduler: Linear warmup followed by Cosine Annealing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    criterion = nn.CrossEntropyLoss()

    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        start = time.time()

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out["logits"], y)
            loss.backward()
            # Gradient clipping for stability in deep KAN
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            correct += (out["logits"].argmax(dim=-1) == y).sum().item()
            total += y.size(0)

        # Step scheduler
        scheduler.step()

        # Val
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                val_correct += (out["logits"].argmax(dim=-1) == y).sum().item()
                val_total += y.size(0)

        val_acc = val_correct / val_total
        print(
            f"Ep {epoch+1}/{epochs}: Loss {total_loss/len(train_loader):.3f}, Acc {correct/total:.1%}, Val {val_acc:.1%}, LR {optimizer.param_groups[0]['lr']:.6f}, Time {time.time()-start:.1f}s"
        )
        best_acc = max(best_acc, val_acc)

    print(f"\nFinal Best Val Acc: {best_acc:.1%}")
    return best_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_eval(epochs=args.epochs, device=device)
