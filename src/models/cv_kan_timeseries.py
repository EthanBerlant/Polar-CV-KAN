"""
CV-KAN Time Series Model.

A CV-KAN variant optimized for time series forecasting using:
- Causal aggregation (cumulative mean)
- Multiple output decodings (magnitude, real, phase)
- Support for multi-step forecasting

The causal aggregation ensures that predictions at time t only depend
on information from times 1 to t, enabling autoregressive processing.

Inherits from BaseCVKAN for log-magnitude centering.
"""

from typing import Literal

import torch
import torch.nn as nn

from ..configs.model import CVKANConfig
from ..modules.positional_encoding import (
    ComplexPositionalEncoding,
    LearnableComplexPositionalEncoding,
)
from .base import ComplexEmbedding
from .cv_kan import CVKAN, CVKANBackbone


class TimeSeriesEmbedding(nn.Module):
    """
    Composite embedding for TimeSeries: Complex Projection + Positional Encoding.
    """

    def __init__(self, input_dim, d_complex, pos_encoding="sinusoidal"):
        super().__init__()
        self.embedding = ComplexEmbedding(input_dim, d_complex)

        if pos_encoding == "sinusoidal":
            self.pos_encoding = ComplexPositionalEncoding(d_complex)
        elif pos_encoding == "learnable":
            self.pos_encoding = LearnableComplexPositionalEncoding(d_complex)
        else:
            self.pos_encoding = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.embedding(x)
        if self.pos_encoding is not None:
            z = self.pos_encoding(z)
        return z


class TimeSeriesHead(nn.Module):
    """
    Forecasting head for Time Series.
    Supports multiple decoding modes and horizons.
    """

    def __init__(
        self, output_dim, d_complex, kan_hidden, output_mode="real", forecast_horizon=None
    ):
        super().__init__()
        self.output_mode = output_mode
        self.output_dim = output_dim
        self.forecast_horizon = forecast_horizon

        # Output projection input size
        if output_mode == "both":
            proj_input = d_complex * 2
        else:
            proj_input = d_complex

        if forecast_horizon is not None:
            # Multi-step forecast from final state
            self.output_proj = nn.Sequential(
                nn.Linear(proj_input, kan_hidden * 2),
                nn.GELU(),
                nn.Linear(kan_hidden * 2, forecast_horizon * output_dim),
            )
        else:
            # Per-timestep output
            self.output_proj = nn.Sequential(
                nn.Linear(proj_input, kan_hidden),
                nn.GELU(),
                nn.Linear(kan_hidden, output_dim),
            )

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        if self.output_mode == "magnitude":
            return torch.abs(z)
        elif self.output_mode == "real":
            return z.real
        elif self.output_mode == "phase":
            return torch.angle(z)
        elif self.output_mode == "both":
            return torch.cat([z.real, z.imag], dim=-1)
        else:
            raise ValueError(f"Unknown output_mode: {self.output_mode}")

    def forward(self, z: torch.Tensor, return_sequence: bool = True, **kwargs) -> dict:
        features = self._decode(z)
        output = {"z_final": z}

        if self.forecast_horizon is not None:
            # Multi-step from final position
            # z shape: (batch, seq, d)
            final_features = features[:, -1]  # (batch, d_proj)
            forecast = self.output_proj(final_features)
            forecast = forecast.view(-1, self.forecast_horizon, self.output_dim)
            output["predictions"] = forecast
        else:
            # Per-timestep
            predictions = self.output_proj(features)
            output["predictions"] = predictions

        if return_sequence:
            if self.forecast_horizon is None:
                output["sequence_output"] = output["predictions"]
            else:
                # Compute per-step outputs for sequence return if needed
                seq_output = self.output_proj(features)
                output["sequence_output"] = seq_output

        return output


class CVKANTimeSeries(CVKAN):
    """
    CV-KAN model for time series forecasting (Composition-based).
    """

    def __init__(
        self,
        input_dim: int = 1,
        d_complex: int = 64,
        n_layers: int = 4,
        output_dim: int = 1,
        kan_hidden: int = 32,
        output_mode: Literal["magnitude", "real", "phase", "both"] = "real",
        forecast_horizon: int | None = None,
        pos_encoding: Literal["sinusoidal", "learnable"] | None = "sinusoidal",
        dropout: float = 0.0,
        center_magnitudes: bool = True,
        **kwargs,
    ):
        # 1. Config
        # Force aggregation_type="causal" for time series
        config = CVKANConfig(
            d_complex=d_complex,
            n_layers=n_layers,
            kan_hidden=kan_hidden,
            center_magnitudes=center_magnitudes,
            dropout=dropout,
            input_type="real",
            aggregation_type="causal",
            block_type="polarizing",  # Or use default
            head_approach="emergent",  # Default
        )

        # 2. Components
        embedding = TimeSeriesEmbedding(
            input_dim=input_dim, d_complex=d_complex, pos_encoding=pos_encoding
        )

        backbone = CVKANBackbone(config)

        head = TimeSeriesHead(
            output_dim=output_dim,
            d_complex=d_complex,
            kan_hidden=kan_hidden,
            output_mode=output_mode,
            forecast_horizon=forecast_horizon,
        )

        super().__init__(embedding, backbone, head)

        self.forecast_horizon = forecast_horizon
        self.output_dim = output_dim

    def generate(
        self,
        x: torch.Tensor,
        n_steps: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Autoregressive generation for time series.
        """
        generated = []
        current = x

        for _ in range(n_steps):
            # Forward pass
            out = self.forward(current, return_sequence=False)

            # Get next prediction
            if self.forecast_horizon is not None:
                next_pred = out["predictions"][:, 0:1]  # First forecasted step
            else:
                next_pred = out["predictions"][:, -1:]  # Last seq position

            generated.append(next_pred)

            # Append to context
            current = torch.cat([current, next_pred], dim=1)

        return torch.cat(generated, dim=1)
