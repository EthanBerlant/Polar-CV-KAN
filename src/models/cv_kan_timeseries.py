"""
CV-KAN Time Series Model.

A CV-KAN variant optimized for time series forecasting using:
- Causal aggregation (cumulative mean)
- Multiple output decodings (magnitude, real, phase)
- Support for multi-step forecasting

The causal aggregation ensures that predictions at time t only depend
on information from times 1 to t, enabling autoregressive processing.
"""

import torch
import torch.nn as nn
from typing import Literal, Optional

from ..modules.polarizing_block import PolarizingBlock
from ..modules.aggregation import CausalAggregation
from ..modules.positional_encoding import (
    ComplexPositionalEncoding,
    LearnableComplexPositionalEncoding,
)


class TimeSeriesEmbedding(nn.Module):
    """
    Embed time series features to complex space.
    
    Args:
        input_dim: Input feature dimension per timestep
        d_complex: Output complex dimension
    """
    
    def __init__(self, input_dim: int, d_complex: int):
        super().__init__()
        self.proj_real = nn.Linear(input_dim, d_complex)
        self.proj_imag = nn.Linear(input_dim, d_complex)
        
        nn.init.xavier_uniform_(self.proj_real.weight, gain=0.5)
        nn.init.xavier_uniform_(self.proj_imag.weight, gain=0.5)
        nn.init.zeros_(self.proj_real.bias)
        nn.init.zeros_(self.proj_imag.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embed to complex space.
        
        Args:
            x: Real tensor (batch, seq_len, input_dim)
        
        Returns:
            Complex tensor (batch, seq_len, d_complex)
        """
        return torch.complex(self.proj_real(x), self.proj_imag(x))


class CausalPolarizingBlock(nn.Module):
    """
    Polarizing block with causal aggregation.
    
    Uses cumulative mean aggregation so that position t only depends
    on positions 1 through t.
    
    Args:
        d_complex: Complex dimension
        kan_hidden: Hidden size for KAN MLPs
        mag_init_scale: Initial magnitude scale
    """
    
    def __init__(
        self,
        d_complex: int,
        kan_hidden: int = 32,
        mag_init_scale: float = 0.1,
    ):
        super().__init__()
        self.d_complex = d_complex
        self.aggregation = CausalAggregation()
        
        # KAN approximation networks
        self.psi_mag = nn.Sequential(
            nn.Linear(1, kan_hidden),
            nn.GELU(),
            nn.Linear(kan_hidden, 1),
        )
        self.psi_phase = nn.Sequential(
            nn.Linear(2, kan_hidden),
            nn.GELU(),
            nn.Linear(kan_hidden, 2),
        )
        
        self.mag_scale = nn.Parameter(torch.tensor(mag_init_scale))
        self._init_weights()
    
    def _init_weights(self):
        for module in [self.psi_mag, self.psi_phase]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=0.1)
                    nn.init.zeros_(layer.bias)
    
    def forward(
        self, 
        Z: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Causal polarizing forward pass.
        
        Args:
            Z: Complex tensor (batch, seq_len, d_complex)
            mask: Optional mask (batch, seq_len)
        
        Returns:
            Transformed complex tensor
        """
        # Causal aggregate: A_t = mean(Z_1, ..., Z_t)
        A = self.aggregation(Z, mask)  # (batch, seq_len, d_complex)
        
        # Polar decomposition
        mag = torch.abs(A)
        log_mag = torch.log(mag + 1e-6)
        
        phase_vec = torch.stack([A.real, A.imag], dim=-1)
        phase_vec = phase_vec / (mag.unsqueeze(-1) + 1e-6)
        
        # Transform
        mag_delta = self.psi_mag(log_mag.unsqueeze(-1)).squeeze(-1)
        log_mag_out = log_mag + self.mag_scale * mag_delta
        
        phase_out_vec = self.psi_phase(phase_vec)
        phase_out_vec = torch.nn.functional.normalize(phase_out_vec, dim=-1)
        
        # Recompose
        r_out = torch.exp(log_mag_out)
        A_new = r_out * torch.complex(phase_out_vec[..., 0], phase_out_vec[..., 1])
        
        return Z + A_new


class CVKANTimeSeries(nn.Module):
    """
    CV-KAN model for time series forecasting.
    
    Uses causal aggregation for autoregressive processing, with multiple
    output decoding strategies optimized for different forecasting tasks.
    
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
    ):
        super().__init__()
        self.d_complex = d_complex
        self.output_mode = output_mode
        self.forecast_horizon = forecast_horizon
        
        # Embedding
        self.embedding = TimeSeriesEmbedding(input_dim, d_complex)
        
        # Positional encoding
        if pos_encoding == 'sinusoidal':
            self.pos_encoding = ComplexPositionalEncoding(d_complex)
        elif pos_encoding == 'learnable':
            self.pos_encoding = LearnableComplexPositionalEncoding(d_complex)
        else:
            self.pos_encoding = None
        
        # Causal polarizing layers
        self.layers = nn.ModuleList([
            CausalPolarizingBlock(d_complex, kan_hidden)
            for _ in range(n_layers)
        ])
        
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
            self.output_dim = output_dim
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
        
        # Apply causal layers
        for layer in self.layers:
            z = layer(z, mask)
        
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
