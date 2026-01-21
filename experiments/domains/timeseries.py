"""
Time series forecasting domain (ETTh1, ETTm1, Exchange).
"""

import torch.nn.functional as F

from src.configs.model import TimeSeriesConfig
from src.configs.training import TrainingConfig
from src.data import (
    create_ettm1_dataloader,
    create_timeseries_dataloader,  # ETTh1
)
from src.data.timeseries_extension import create_exchange_dataloader
from src.models.cv_kan_timeseries import CVKANTimeSeries
from src.trainer import BaseTrainer


def create_model(config: TimeSeriesConfig):
    """Create time series forecasting model."""
    return CVKANTimeSeries(
        input_dim=config.d_input,
        d_complex=config.d_complex,
        n_layers=config.n_layers,
        output_dim=config.d_input,
        kan_hidden=config.kan_hidden,
        output_mode=config.output_mode,
        forecast_horizon=config.pred_len,
        pos_encoding="sinusoidal",  # or config if we add it
        dropout=config.dropout,
        center_magnitudes=config.center_magnitudes,
    )


def create_dataloaders(model_config: TimeSeriesConfig, train_config: TrainingConfig):
    """Create dataloaders based on config.dataset_name."""
    kwargs = {
        "batch_size": train_config.batch_size,
        "seq_len": model_config.seq_len,
        "pred_len": model_config.pred_len,
        "subset_size": train_config.subset_size,
    }

    dataset = model_config.dataset_name.lower()
    data_root = "./data"

    if dataset == "etth1":
        root = f"{data_root}/ETT"
        train, val, test, dim = create_timeseries_dataloader(root=root, **kwargs)
    elif dataset == "ettm1":
        root = f"{data_root}/ETT"
        train, val, test, dim = create_ettm1_dataloader(root=root, **kwargs)
    elif dataset == "exchange":
        root = f"{data_root}/exchange"
        train, val, test, dim = create_exchange_dataloader(root=root, **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return train, val, test, {"input_dim": dim}


class TimeSeriesTrainer(BaseTrainer):
    """Trainer for time series forecasting."""

    def train_step(self, batch):
        seq_x, seq_y = batch
        seq_x = seq_x.to(self.device)
        seq_y = seq_y.to(self.device)

        # Access prediction length from model
        pred_len = self.model.forecast_horizon
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

        pred_len = self.model.forecast_horizon
        target = seq_y[:, -pred_len:, :]

        outputs = self.model(seq_x, return_sequence=False)
        predictions = outputs["predictions"]

        loss = F.mse_loss(predictions, target)
        mae = F.l1_loss(predictions, target)

        return {"loss": loss, "mse": loss, "mae": mae}


Trainer = TimeSeriesTrainer
