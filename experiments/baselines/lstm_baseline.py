"""
LSTM Baseline for ETTh1 Time Series Forecasting.
"""

import sys
from pathlib import Path

import torch.nn.functional as F
from torch import nn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data import create_timeseries_dataloader

from .base_trainer import run_baseline


class LSTMForecaster(nn.Module):
    """LSTM model for multivariate time series forecasting."""

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
    """Create LSTM model."""
    return LSTMForecaster(
        input_dim=metadata["input_dim"],
        hidden_dim=args.d_complex,
        num_layers=args.n_layers,
        output_dim=metadata["input_dim"],
        pred_len=metadata["pred_len"],
    )


def timeseries_loss(outputs, targets):
    """MSE loss for timeseries, taking last pred_len timesteps from target."""
    target = targets[:, -_pred_len:, :]
    return F.mse_loss(outputs, target)


def timeseries_mse(outputs, targets):
    """Compute MSE metric."""
    target = targets[:, -_pred_len:, :]
    return F.mse_loss(outputs, target).item()


if __name__ == "__main__":
    run_baseline(
        model_class=LSTMForecaster,
        domain="timeseries",
        create_dataloaders=create_dataloaders,
        create_model=create_model,
        loss_fn=timeseries_loss,
        metric_fn=timeseries_mse,
        metric_name="mse",
        metric_mode="min",
        default_args={
            "d_complex": 64,
            "n_layers": 2,
            "epochs": 50,
            "output_dir": "outputs/baselines/timeseries",
        },
    )
