"""
Transformer Baseline for Time Series Forecasting.

A standard Transformer encoder for multivariate forecasting.
Sized to match CV-KAN parameter count (~200-500k params).
"""

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data import create_timeseries_dataloader

from .base_trainer import run_baseline


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
    """Transformer for time series forecasting using encoder-only architecture."""

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


# Store pred_len in closure for metric function
_pred_len = 96


def create_dataloaders(args):
    """Create ETTh1 dataloaders."""
    global _pred_len
    _pred_len = getattr(args, "pred_len", 96)

    train_loader, val_loader, test_loader, input_dim = create_timeseries_dataloader(
        root=getattr(args, "data_root", "./data/ETT"),
        batch_size=args.batch_size,
        seq_len=getattr(args, "seq_len", 96),
        pred_len=_pred_len,
    )
    return train_loader, val_loader, test_loader, {"input_dim": input_dim, "pred_len": _pred_len}


def create_model(args, metadata):
    """Create Transformer model."""
    return TransformerForecaster(
        input_dim=metadata["input_dim"],
        output_dim=metadata["input_dim"],
        pred_len=metadata["pred_len"],
        d_model=args.d_complex,
        nhead=4,
        num_layers=args.n_layers,
        dim_feedforward=args.d_complex * 2,
    )


def timeseries_loss(outputs, targets):
    """MSE loss for timeseries."""
    target = targets[:, -_pred_len:, :]
    return F.mse_loss(outputs, target)


def timeseries_mse(outputs, targets):
    """Compute MSE metric."""
    target = targets[:, -_pred_len:, :]
    return F.mse_loss(outputs, target).item()


if __name__ == "__main__":
    run_baseline(
        model_class=TransformerForecaster,
        domain="timeseries",
        create_dataloaders=create_dataloaders,
        create_model=create_model,
        loss_fn=timeseries_loss,
        metric_fn=timeseries_mse,
        metric_name="mse",
        metric_mode="min",
        default_args={
            "d_complex": 128,
            "n_layers": 4,
            "epochs": 50,
            "output_dir": "outputs/baselines/timeseries",
        },
    )
