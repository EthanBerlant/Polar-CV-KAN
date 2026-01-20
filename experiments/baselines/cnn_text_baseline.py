"""
CNN Baseline for Text Classification.

TextCNN with multiple filter sizes for n-gram feature extraction.
Reference: "Convolutional Neural Networks for Sentence Classification" (Kim, 2014)

Sized to match CV-KAN parameter count (~200-500k params).
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
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.text import (
    load_agnews,
    load_imdb,
    load_sst2,
    pad_collate,
)


class TextCNN(nn.Module):
    """
    TextCNN for text classification.

    Uses multiple convolutional filter sizes to capture different n-gram patterns.
    """

    def __init__(
        self,
        vocab_size,
        embed_dim=128,
        num_classes=2,
        filter_sizes=None,
        num_filters=100,
        dropout=0.5,
    ):
        super().__init__()
        if filter_sizes is None:
            filter_sizes = [3, 4, 5]

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Multiple filter sizes for different n-grams
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(embed_dim, num_filters, kernel_size=fs, padding=fs // 2)
                for fs in filter_sizes
            ]
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x, mask=None):
        # x: (batch, seq_len)
        x = self.embedding(x)  # (batch, seq_len, embed_dim)
        x = x.transpose(1, 2)  # (batch, embed_dim, seq_len)

        # Apply convolutions with ReLU and max pooling
        pooled = []
        for conv in self.convs:
            h = F.relu(conv(x))  # (batch, num_filters, seq_len)
            h = F.max_pool1d(h, h.size(2)).squeeze(2)  # (batch, num_filters)
            pooled.append(h)

        # Concatenate all pooled features
        cat = torch.cat(pooled, dim=1)  # (batch, num_filters * len(filter_sizes))
        cat = self.dropout(cat)

        return self.fc(cat)


def parse_args():
    parser = argparse.ArgumentParser(description="TextCNN Baseline for Text Classification")
    parser.add_argument("--dataset", type=str, default="sst2", choices=["sst2", "imdb", "agnews"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--num_filters", type=int, default=100)
    parser.add_argument("--d_complex", type=int, default=128, help="Ignored, for compatibility")
    parser.add_argument("--n_layers", type=int, default=4, help="Ignored, for compatibility")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subset_size", type=int, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/baselines/cnn_text")
    parser.add_argument("--amp", action="store_true")
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
    for batch_idx, batch in enumerate(pbar):
        indices = batch["indices"].to(device)
        labels = batch["label"].to(device)
        mask = batch.get("mask")

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                outputs = model(indices, mask)
                loss = F.cross_entropy(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(indices, mask)
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
        for batch in dataloader:
            indices = batch["indices"].to(device)
            labels = batch["label"].to(device)
            mask = batch.get("mask")

            outputs = model(indices, mask)
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

    # Load dataset
    if args.dataset == "sst2":
        train_dataset, val_dataset, vocab = load_sst2()
        num_classes = 2
    elif args.dataset == "imdb":
        train_dataset, test_dataset, vocab = load_imdb()
        # Split train into train/val
        val_size = int(0.1 * len(train_dataset))
        train_dataset, val_dataset = random_split(
            train_dataset, [len(train_dataset) - val_size, val_size]
        )
        num_classes = 2
    else:  # agnews
        train_dataset, test_dataset, vocab = load_agnews()
        val_size = int(0.1 * len(train_dataset))
        train_dataset, val_dataset = random_split(
            train_dataset, [len(train_dataset) - val_size, val_size]
        )
        num_classes = 4

    # Subset if requested
    if args.subset_size and hasattr(train_dataset, "texts"):
        train_dataset.texts = train_dataset.texts[: args.subset_size]
        train_dataset.labels = train_dataset.labels[: args.subset_size]

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=pad_collate
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate
    )

    # For IMDB/AG News, we need to get test set separately
    if args.dataset in ["imdb", "agnews"]:
        if args.dataset == "imdb":
            _, test_dataset, _ = load_imdb(vocab=vocab)
        else:
            _, test_dataset, _ = load_agnews(vocab=vocab)
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate
        )
    else:
        test_loader = val_loader  # SST-2 uses dev as test

    # Create model
    model = TextCNN(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        num_classes=num_classes,
        num_filters=args.num_filters,
    ).to(device)

    n_params = count_parameters(model)
    print(f"Model parameters: {n_params:,}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    scaler = torch.amp.GradScaler("cuda") if args.amp and device.type == "cuda" else None

    # Training
    best_val_acc = 0
    patience_counter = 0

    run_name = args.run_name or f"cnn_text_{args.dataset}_s{args.seed}"
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
        "model": "TextCNN",
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
