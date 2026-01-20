"""
LSTM Baseline for Text Classification.

Bidirectional LSTM for sequence classification.

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


class TextLSTM(nn.Module):
    """
    Bidirectional LSTM for text classification.
    """

    def __init__(
        self,
        vocab_size,
        embed_dim=128,
        hidden_dim=128,
        num_layers=2,
        num_classes=2,
        dropout=0.3,
        bidirectional=True,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        lstm_out_dim = hidden_dim * 2 if bidirectional else hidden_dim

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_out_dim, num_classes)

    def forward(self, x, mask=None):
        # x: (batch, seq_len)
        x = self.embedding(x)  # (batch, seq_len, embed_dim)

        # Pack padded sequences if mask provided
        if mask is not None:
            lengths = mask.sum(dim=1).cpu()
            # Clamp to at least 1 to avoid empty sequences
            lengths = lengths.clamp(min=1)
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            output, (hidden, _) = self.lstm(packed)
            # Unpack
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        else:
            output, (hidden, _) = self.lstm(x)

        # Use mean pooling over all timesteps
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            output = (output * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            output = output.mean(dim=1)

        output = self.dropout(output)
        return self.fc(output)


def parse_args():
    parser = argparse.ArgumentParser(description="LSTM Baseline for Text Classification")
    parser.add_argument("--dataset", type=str, default="sst2", choices=["sst2", "imdb", "agnews"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--d_complex", type=int, default=128, help="Ignored, for compatibility")
    parser.add_argument("--n_layers", type=int, default=4, help="Ignored, for compatibility")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subset_size", type=int, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/baselines/lstm_text")
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
        if mask is not None:
            mask = mask.to(device)

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
            if mask is not None:
                mask = mask.to(device)

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

    # For IMDB/AG News, get test set separately
    if args.dataset in ["imdb", "agnews"]:
        if args.dataset == "imdb":
            _, test_dataset, _ = load_imdb(vocab=vocab)
        else:
            _, test_dataset, _ = load_agnews(vocab=vocab)
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate
        )
    else:
        test_loader = val_loader

    # Create model
    model = TextLSTM(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=num_classes,
    ).to(device)

    n_params = count_parameters(model)
    print(f"Model parameters: {n_params:,}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    scaler = torch.amp.GradScaler("cuda") if args.amp and device.type == "cuda" else None

    # Training
    best_val_acc = 0
    patience_counter = 0

    run_name = args.run_name or f"lstm_text_{args.dataset}_s{args.seed}"
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
        "model": "TextLSTM",
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
