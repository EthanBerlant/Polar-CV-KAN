"""
TCN (Temporal Convolutional Network) Baseline for Time Series Forecasting.

A dilated causal convolution network for sequence modeling.
Reference: "An Empirical Evaluation of Generic Convolutional and Recurrent
Networks for Sequence Modeling" (Bai et al., 2018)
"""

import sys
from pathlib import Path

import torch.nn.functional as F
from torch import nn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data import create_timeseries_dataloader

from .base_trainer import run_baseline


class CausalConv1d(nn.Module):
    """Causal convolution with proper padding."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, dilation=dilation, padding=self.padding
        )

    def forward(self, x):
        x = self.conv(x)
        return x[:, :, : -self.padding] if self.padding > 0 else x


class TCNBlock(nn.Module):
    """A single TCN residual block with dilated causal convolutions."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.1):
        super().__init__()

        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)

        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)

        self.dropout = nn.Dropout(dropout)

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = F.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = F.relu(out)
        out = self.dropout(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        return F.relu(out + residual)


class TCN(nn.Module):
    """Temporal Convolutional Network for time series forecasting."""

    def __init__(
        self,
        input_dim,
        output_dim,
        pred_len,
        hidden_dim=64,
        num_layers=4,
        kernel_size=3,
        dropout=0.1,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.output_dim = output_dim

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        layers = []
        for i in range(num_layers):
            dilation = 2**i
            layers.append(TCNBlock(hidden_dim, hidden_dim, kernel_size, dilation, dropout))

        self.tcn = nn.Sequential(*layers)
        self.output_proj = nn.Linear(hidden_dim, pred_len * output_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x = self.input_proj(x)
        x = x.transpose(1, 2)
        x = self.tcn(x)
        x = x[:, :, -1]
        x = self.output_proj(x)
        x = x.view(batch_size, self.pred_len, self.output_dim)
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
    """Create TCN model."""
    return TCN(
        input_dim=metadata["input_dim"],
        output_dim=metadata["input_dim"],
        pred_len=metadata["pred_len"],
        hidden_dim=args.d_complex,
        num_layers=args.n_layers,
        kernel_size=3,
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
        model_class=TCN,
        domain="timeseries",
        create_dataloaders=create_dataloaders,
        create_model=create_model,
        loss_fn=timeseries_loss,
        metric_fn=timeseries_mse,
        metric_name="mse",
        metric_mode="min",
        default_args={
            "d_complex": 64,
            "n_layers": 4,
            "epochs": 50,
            "output_dir": "outputs/baselines/timeseries",
        },
    )
