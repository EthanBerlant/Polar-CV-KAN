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

import torch
import torch.nn as nn
from typing import Literal, Optional

from .base import BaseCVKAN, ComplexEmbedding
from ..modules.polarizing_block import PolarizingBlock
from ..modules.aggregation import CausalAggregation
from ..modules.positional_encoding import (
    ComplexPositionalEncoding,
    LearnableComplexPositionalEncoding,
)


class CVKANTimeSeries(BaseCVKAN):
    """
    CV-KAN model for time series forecasting.
    
    Uses causal aggregation for autoregressive processing, with multiple
    output decoding strategies optimized for different forecasting tasks.
    
    Inherits from BaseCVKAN for log-magnitude centering.
    
    Args:
        input_dim: Input feature dimension per timestep
        d_complex: Complex representation dimension
        n_layers: Number of causal polarizing layers
        output_dim: Output dimension per timestep
        kan_hidden: Hidden size for KAN MLPs
        output_mode: Decoding strategy:
            - 'magnitude': Predict positive values (prices, counts)
            - 'real': Predict signed values
            - 'phase': Predict periodic signals (seasonal data)
            - 'both': Output both real and imaginary parts
        forecast_horizon: Number of future steps to predict (None = per-step)
        pos_encoding: 'sinusoidal', 'learnable', or None
        center_magnitudes: Whether to center log-magnitudes (recommended)
    """
    
    def __init__(
        self,
        input_dim: int = 1,
        d_complex: int = 64,
        n_layers: int = 4,
        output_dim: int = 1,
        kan_hidden: int = 32,
        output_mode: Literal['magnitude', 'real', 'phase', 'both'] = 'real',
        forecast_horizon: Optional[int] = None,
        pos_encoding: Optional[Literal['sinusoidal', 'learnable']] = 'sinusoidal',
        center_magnitudes: bool = True,
    ):
        # Initialize base class (pooling not used for timeseries)
        super().__init__(
            d_complex=d_complex,
            n_layers=n_layers,
            kan_hidden=kan_hidden,
            pooling='mean',  # Not used, but required by base
            center_magnitudes=center_magnitudes,
        )
        
        self.output_mode = output_mode
        self.forecast_horizon = forecast_horizon
        self.output_dim = output_dim
        
        # Override layers to use causal aggregation
        # per_dim=False because CausalAggregation returns per-position aggregates (batch, seq, d)
        # not global aggregates (batch, 1, d)
        causal_agg = CausalAggregation()
        self.layers = nn.ModuleList([
            PolarizingBlock(d_complex, kan_hidden, aggregation=causal_agg, per_dim=False)
            for _ in range(n_layers)
        ])
        
        # Embedding
        self.embedding = ComplexEmbedding(input_dim, d_complex)
        
        # Positional encoding
        if pos_encoding == 'sinusoidal':
            self.pos_encoding = ComplexPositionalEncoding(d_complex)
        elif pos_encoding == 'learnable':
            self.pos_encoding = LearnableComplexPositionalEncoding(d_complex)
        else:
            self.pos_encoding = None
        
        # Output projection
        if output_mode == 'both':
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
        """
        Decode complex representation based on output_mode.
        
        Args:
            z: Complex tensor
        
        Returns:
            Real tensor for prediction
        """
        if self.output_mode == 'magnitude':
            return torch.abs(z)
        elif self.output_mode == 'real':
            return z.real
        elif self.output_mode == 'phase':
            # Map phase from [-pi, pi] to output range
            return torch.angle(z)
        elif self.output_mode == 'both':
            return torch.cat([z.real, z.imag], dim=-1)
        else:
            raise ValueError(f"Unknown output_mode: {self.output_mode}")
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_sequence: bool = True,
    ) -> dict:
        """
        Forward pass for time series forecasting.
        
        Args:
            x: Input tensor (batch, seq_len, input_dim)
            mask: Optional mask (batch, seq_len)
            return_sequence: Whether to return per-step predictions
        
        Returns:
            Dictionary with:
                - predictions: Forecasted values
                - sequence_output: Per-timestep outputs (if return_sequence=True)
                - z_final: Final complex representation
        """
        batch, seq_len, _ = x.shape
        
        # Embed
        z = self.embedding(x)
        
        # Position encoding
        if self.pos_encoding is not None:
            z = self.pos_encoding(z)
        
        # Apply causal layers with magnitude centering (from base class)
        z = self._apply_layers(z, mask)
        
        # Decode
        features = self._decode(z)
        
        output = {'z_final': z}
        
        if self.forecast_horizon is not None:
            # Multi-step forecast from final position
            final_features = features[:, -1]  # (batch, proj_input)
            forecast = self.output_proj(final_features)  # (batch, horizon * output_dim)
            forecast = forecast.view(batch, self.forecast_horizon, self.output_dim)
            output['predictions'] = forecast
        else:
            # Per-timestep predictions
            predictions = self.output_proj(features)  # (batch, seq_len, output_dim)
            output['predictions'] = predictions
        
        if return_sequence:
            if self.forecast_horizon is None:
                output['sequence_output'] = output['predictions']
            else:
                # Also compute per-step outputs
                seq_output = self.output_proj(features)
                output['sequence_output'] = seq_output
        
        return output
    
    def generate(
        self,
        x: torch.Tensor,
        n_steps: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Autoregressive generation for time series.
        
        Args:
            x: Initial context (batch, context_len, input_dim)
            n_steps: Number of steps to generate
            temperature: Sampling temperature (for stochastic generation)
        
        Returns:
            Generated sequence (batch, n_steps, output_dim)
        """
        batch = x.shape[0]
        generated = []
        
        current = x
        
        for _ in range(n_steps):
            # Forward pass
            out = self.forward(current, return_sequence=False)
            
            # Get next prediction
            if self.forecast_horizon is not None:
                next_pred = out['predictions'][:, 0:1]  # First forecasted step
            else:
                next_pred = out['predictions'][:, -1:]  # Last seq position
            
            generated.append(next_pred)
            
            # Append to context (rolling window could be used for efficiency)
            current = torch.cat([current, next_pred], dim=1)
        
        return torch.cat(generated, dim=1)
