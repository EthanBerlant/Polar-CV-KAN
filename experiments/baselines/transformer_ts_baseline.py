"""
Transformer Baseline for Time Series Forecasting.

A standard Transformer encoder-decoder for multivariate forecasting.
Sized to match CV-KAN parameter count (~200-500k params).
"""

import argparse
import json
import math
import sys
import time
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

from src.data.timeseries_data import (
    create_ettm1_dataloader,
    create_timeseries_dataloader,
    create_weather_dataloader,
)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerForecaster(nn.Module):
    """
    Transformer for time series forecasting using encoder-only architecture.
    Predicts future values from the last hidden state.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        pred_len,
        d_model=128,
        nhead=4,
        num_layers=4,
        dim_feedforward=256,
        dropout=0.1,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.output_dim = output_dim

        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection: from d_model to pred_len * output_dim
        self.output_proj = nn.Linear(d_model, pred_len * output_dim)

    def forward(self, x):
        # x: (B, seq_len, input_dim)
        x = self.input_proj(x)
        x = self.pos_encoder(x)

        # Transformer
        x = self.transformer(x)

        # Use last timestep to predict future
        x = x[:, -1, :]  # (B, d_model)

        # Project to output
        x = self.output_proj(x)  # (B, pred_len * output_dim)
        x = x.view(-1, self.pred_len, self.output_dim)

        return x


def parse_args():
    parser = argparse.ArgumentParser(description="Transformer Baseline for Time Series Forecasting")
    parser.add_argument(
        "--dataset", type=str, default="etth1", choices=["etth1", "ettm1", "weather"]
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--d_complex", type=int, default=128, help="Ignored, for compatibility")
    parser.add_argument("--n_layers", type=int, default=4, help="Ignored, for compatibility")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/baselines/transformer_ts")
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch, pred_len, scaler=None):
    model.train()
    total_loss = 0

    pbar = tqdm(dataloader, desc=f"Train Epoch {epoch}")
    for batch_idx, (seq_x, seq_y) in enumerate(pbar):
        seq_x = seq_x.to(device)
        target = seq_y[:, -pred_len:, :].to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                pred = model(seq_x)
                loss = F.mse_loss(pred, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(seq_x)
            loss = F.mse_loss(pred, target)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({"mse": total_loss / (batch_idx + 1)})

    scheduler.step()
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device, pred_len):
    model.eval()
    total_mse = 0
    total_mae = 0
    n_samples = 0

    with torch.no_grad():
        for seq_x, seq_y in dataloader:
            seq_x = seq_x.to(device)
            target = seq_y[:, -pred_len:, :].to(device)

            pred = model(seq_x)

            mse = F.mse_loss(pred, target, reduction="sum")
            mae = F.l1_loss(pred, target, reduction="sum")

            total_mse += mse.item()
            total_mae += mae.item()
            n_samples += target.numel()

    return total_mse / n_samples, total_mae / n_samples


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create dataloaders
    if args.dataset == "etth1":
        train_loader, val_loader, test_loader, input_dim = create_timeseries_dataloader(
            batch_size=args.batch_size, seq_len=args.seq_len, pred_len=args.pred_len
        )
    elif args.dataset == "ettm1":
        train_loader, val_loader, test_loader, input_dim = create_ettm1_dataloader(
            batch_size=args.batch_size, seq_len=args.seq_len, pred_len=args.pred_len
        )
    else:  # weather
        train_loader, val_loader, test_loader, input_dim = create_weather_dataloader(
            batch_size=args.batch_size, seq_len=args.seq_len, pred_len=args.pred_len
        )

    # Create model
    model = TransformerForecaster(
        input_dim=input_dim,
        output_dim=input_dim,
        pred_len=args.pred_len,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
    ).to(device)

    n_params = count_parameters(model)
    print(f"Model parameters: {n_params:,}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    scaler = torch.amp.GradScaler("cuda") if args.amp and device.type == "cuda" else None

    # Training
    best_val_mse = float("inf")
    patience_counter = 0

    run_name = args.run_name or f"transformer_ts_{args.dataset}_s{args.seed}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device, epoch, args.pred_len, scaler
        )
        val_mse, val_mae = evaluate(model, val_loader, device, args.pred_len)

        print(
            f"Epoch {epoch}: Train MSE={train_loss:.4f}, Val MSE={val_mse:.4f}, Val MAE={val_mae:.4f}"
        )

        if val_mse < best_val_mse:
            best_val_mse = val_mse
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
    test_mse, test_mae = evaluate(model, test_loader, device, args.pred_len)

    print(f"\nTest MSE: {test_mse:.4f}, Test MAE: {test_mae:.4f}")

    # Save results
    results = {
        "dataset": args.dataset,
        "model": "Transformer-TS",
        "n_params": n_params,
        "best_val_mse": best_val_mse,
        "test_mse": test_mse,
        "test_mae": test_mae,
        "train_time_seconds": train_time,
        "epochs_trained": epoch,
        "args": vars(args),
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
