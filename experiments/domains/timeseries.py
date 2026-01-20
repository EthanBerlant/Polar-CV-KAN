"""
Time series forecasting domain (ETTh1).
"""

import torch.nn.functional as F

from src.data import create_timeseries_dataloader
from src.models.cv_kan_timeseries import CVKANTimeSeries
from src.trainer import BaseTrainer

DEFAULTS = {
    "batch_size": 32,
    "d_complex": 64,
    "n_layers": 4,
    "epochs": 50,
    "metric_name": "mse",
    "metric_mode": "min",
}


def add_args(parser):
    """Add timeseries-specific arguments."""
    parser.add_argument("--data_root", type=str, default="./data/ETT")
    parser.add_argument("--seq_len", type=int, default=96, help="Lookback window")
    parser.add_argument("--pred_len", type=int, default=96, help="Prediction horizon")
    parser.add_argument(
        "--output_mode", type=str, default="real", choices=["magnitude", "real", "phase", "both"]
    )
    return parser


class TimeSeriesTrainer(BaseTrainer):
    """Trainer for time series forecasting."""

    def train_step(self, batch):
        seq_x, seq_y = batch
        seq_x = seq_x.to(self.device)
        seq_y = seq_y.to(self.device)

        pred_len = self.args.pred_len
        target = seq_y[:, -pred_len:, :]

        outputs = self.model(seq_x, return_sequence=False)
        predictions = outputs["predictions"]

        loss = F.mse_loss(predictions, target)
        mae = F.l1_loss(predictions, target)

        return {"loss": loss, "mse": loss, "mae": mae}

    def validate_step(self, batch):
        seq_x, seq_y = batch
        seq_x = seq_x.to(self.device)
        seq_y = seq_y.to(self.device)

        pred_len = self.args.pred_len
        target = seq_y[:, -pred_len:, :]

        outputs = self.model(seq_x, return_sequence=False)
        predictions = outputs["predictions"]

        loss = F.mse_loss(predictions, target)
        mae = F.l1_loss(predictions, target)

        return {"loss": loss, "mse": loss, "mae": mae}


def create_model(args):
    """Create time series forecasting model."""
    pos_enc = args.pos_encoding if args.pos_encoding != "none" else None

    return CVKANTimeSeries(
        input_dim=args.input_dim,
        d_complex=args.d_complex,
        n_layers=args.n_layers,
        output_dim=args.input_dim,
        kan_hidden=args.kan_hidden,
        output_mode=args.output_mode,
        forecast_horizon=args.pred_len,
        pos_encoding=pos_enc,
        dropout=args.dropout,
    )


def create_dataloaders(args):
    """Create ETTh1 dataloaders."""
    train_loader, val_loader, test_loader, input_dim = create_timeseries_dataloader(
        root=args.data_root,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
    )
    return train_loader, val_loader, test_loader, {"input_dim": input_dim}
