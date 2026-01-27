"""Verify Vision Baseline Improvements."""

import os
import sys
import time

import torch
from torch import nn

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.image_data import create_cifar10_dataloader
from src.models.cv_kan_image import CVKANImageClassifier


def run_verification():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Data (with new augmentation)
    print("Loading CIFAR-10...")
    train_loader, val_loader, test_loader, num_classes = create_cifar10_dataloader(
        batch_size=128, num_workers=0
    )

    # Create Model (with new default conv embedding)
    print("Creating Model (Conv Embedding + Augmentation)...")
    model = CVKANImageClassifier(
        img_size=32,
        patch_size=4,  # 4x4 patches for CIFAR
        d_complex=64,
        n_layers=4,
        n_classes=10,
        kan_hidden=64,
        embedding_type="conv",  # Explicitly ensuring it is conv
        dropout=0.1,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()

    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    # Train for 5 epochs
    for epoch in range(5):
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
            optimizer.step()

            total_loss += loss.item()
            correct += (out["logits"].argmax(dim=-1) == y).sum().item()
            total += y.size(0)

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
            f"Ep {epoch+1}: Loss {total_loss/len(train_loader):.3f}, Train Acc {correct/total:.1%}, Val Acc {val_acc:.1%}, Time {time.time()-start:.1f}s"
        )


if __name__ == "__main__":
    run_verification()
