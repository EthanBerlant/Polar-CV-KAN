"""
LSTM Baseline for ETTh1 Time Series Forecasting.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data import create_timeseries_dataloader


class LSTMForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, pred_len, dropout=0.1):
        super().__init__()
        self.pred_len = pred_len
        self.output_dim = output_dim

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.projection = nn.Linear(hidden_dim, output_dim * pred_len)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        batch_size = x.size(0)

        # LSTM output: (batch, seq_len, hidden_dim)
        lstm_out, _ = self.lstm(x)

        # Take last time step: (batch, hidden_dim)
        last_out = lstm_out[:, -1, :]

        # Project to full prediction window: (batch, pred_len * output_dim)
        out = self.projection(last_out)

        # Reshape: (batch, pred_len, output_dim)
        return out.view(batch_size, self.pred_len, self.output_dim)


def parse_args():
    parser = argparse.ArgumentParser(description="Train LSTM on ETTh1")

    # Data args
    parser.add_argument("--data_root", type=str, default="./data/ETT")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument(
        "--subset_size", type=int, default=None
    )  # Ignored but kept for interface compatibility

    # Model args
    parser.add_argument("--d_complex", type=int, default=64, help="Used as hidden dim")
    parser.add_argument("--n_layers", type=int, default=2, help="LSTM layers")

    # Training args
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience (for compatibility)"
    )
    parser.add_argument("--amp", action="store_true", help="Use Automatic Mixed Precision")

    # Output
    parser.add_argument("--output_dir", type=str, default="outputs/baselines/timeseries")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(model, dataloader, optimizer, device, epoch, pred_len):
    model.train()
    total_loss = 0
    total_mse = 0
    total_mae = 0
    n_batches = 0

    pbar = tqdm(dataloader, desc=f"Train Epoch {epoch}")
    for seq_x, seq_y in pbar:
        seq_x = seq_x.to(device)
        seq_y = seq_y.to(device)

        target = seq_y[:, -pred_len:, :]

        optimizer.zero_grad()
        predictions = model(seq_x)

        loss = F.mse_loss(predictions, target)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_mse += F.mse_loss(predictions, target, reduction="sum").item()
        total_mae += F.l1_loss(predictions, target, reduction="sum").item()
        n_batches += 1

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    n_samples = n_batches * dataloader.batch_size * pred_len * target.shape[-1]

    return {
        "loss": total_loss / n_batches,
        "mse": total_mse / n_samples,
        "mae": total_mae / n_samples,
    }


@torch.no_grad()
def evaluate(model, dataloader, device, pred_len):
    model.eval()
    total_loss = 0
    total_mse = 0
    total_mae = 0
    n_batches = 0

    for seq_x, seq_y in dataloader:
        seq_x = seq_x.to(device)
        seq_y = seq_y.to(device)

        target = seq_y[:, -pred_len:, :]

        predictions = model(seq_x)

        loss = F.mse_loss(predictions, target)
        total_loss += loss.item()
        total_mse += F.mse_loss(predictions, target, reduction="sum").item()
        total_mae += F.l1_loss(predictions, target, reduction="sum").item()
        n_batches += 1

    n_samples = n_batches * dataloader.batch_size * pred_len * target.shape[-1]

    return {
        "loss": total_loss / n_batches,
        "mse": total_mse / n_samples,
        "mae": total_mae / n_samples,
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    run_name = args.run_name or f"lstm_ts_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print("Loading ETTh1...")
    train_loader, val_loader, test_loader, input_dim = create_timeseries_dataloader(
        root=args.data_root,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
    )

    print(f"Creating LSTM model (hidden={args.d_complex}, layers={args.n_layers})...")
    model = LSTMForecaster(
        input_dim=input_dim,
        hidden_dim=args.d_complex,
        num_layers=args.n_layers,
        output_dim=input_dim,
        pred_len=args.pred_len,
    ).to(device)

    n_params = count_parameters(model)
    print(f"Model parameters: {n_params:,}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    print("Starting training...")
    history = []
    best_val_mse = float("inf")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_results = train_epoch(model, train_loader, optimizer, device, epoch, args.pred_len)
        val_results = evaluate(model, val_loader, device, args.pred_len)
        scheduler.step()

        epoch_time = time.time() - epoch_start
        print(
            f"Epoch {epoch}/{args.epochs} ({epoch_time:.1f}s) - "
            f"Train MSE: {train_results['mse']:.6f} - "
            f"Val MSE: {val_results['mse']:.6f}"
        )

        history.append(
            {
                "epoch": epoch,
                "train": train_results,
                "val": val_results,
                "epoch_time": epoch_time,
            }
        )

        if val_results["mse"] < best_val_mse:
            best_val_mse = val_results["mse"]
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "val_mse": best_val_mse,
                },
                output_dir / "best.pt",
            )

    print("\nEvaluating on test set...")
    test_results = evaluate(model, test_loader, device, args.pred_len)
    print(f"Test MSE: {test_results['mse']:.6f}")

    results = {
        "model": "LSTM",
        "dataset": "ETTh1",
        "n_params": n_params,
        "best_val_mse": best_val_mse,
        "test_mse": test_results["mse"],
        "test_mae": test_results["mae"],
        "total_time_seconds": time.time() - start_time,
        "epochs": args.epochs,
        "history": history,
        "config": vars(args),
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    main()
