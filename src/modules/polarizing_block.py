"""
PolarizingBlock: Core primitive for CV-KAN.

This module implements the fundamental operation:
1. Aggregate tokens via mean (bounded, stable)
2. Decompose to polar coordinates (log-magnitude + phase as sin/cos)
3. Transform with learnable 1D functions (KAN approximation)
4. Recompose and broadcast back to tokens
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aggregation import GlobalMeanAggregation


class PolarizingBlock(nn.Module):
    """
    Core polarizing block that enables token interaction through
    phase alignment and magnitude polarization.

    Enhanced with configurable capacity for fair comparison with transformers:
    - MLP expansion factor for wider hidden layers
    - Per-dimension magnitude/phase transforms (vs shared 1D functions)
    - Deeper MLP option
    - Pre/post projection layers

    Args:
        d_complex: Number of complex dimensions
        kan_hidden: Base hidden size for KAN MLPs
        mag_init_scale: Initial scale for magnitude transform (small = stable)
        aggregation: Custom aggregation strategy (default: GlobalMeanAggregation)
        mlp_expansion: Expansion factor for MLP width (default 4)
        per_dim: If True, use per-dimension transforms (more params)
        deep_mlp: If True, use 3-layer MLPs instead of 2-layer
    """

    def __init__(
        self,
        d_complex: int,
        kan_hidden: int = 32,
        mag_init_scale: float = 0.1,
        aggregation: nn.Module | None = None,
        mlp_expansion: int = 4,
        per_dim: bool = True,
        deep_mlp: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_complex = d_complex
        self.per_dim = per_dim
        self.dropout_rate = dropout

        # Aggregation strategy (default: global mean)
        self.aggregation = aggregation if aggregation is not None else GlobalMeanAggregation()

        # Expanded hidden dimension
        hidden_dim = kan_hidden * mlp_expansion

        if per_dim:
            # Per-dimension transforms: input/output is full d_complex
            # This is the main source of parameter increase

            if deep_mlp:
                # 3-layer MLP for magnitude
                self.psi_mag = nn.Sequential(
                    nn.Linear(d_complex, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, d_complex),
                )

                # 3-layer MLP for phase
                self.psi_phase = nn.Sequential(
                    nn.Linear(2 * d_complex, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 2 * d_complex),
                )
            else:
                # 2-layer MLP
                self.psi_mag = nn.Sequential(
                    nn.Linear(d_complex, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, d_complex),
                )
                self.psi_phase = nn.Sequential(
                    nn.Linear(2 * d_complex, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 2 * d_complex),
                )
        else:
            # Original shared 1D transforms (low param count)
            if deep_mlp:
                self.psi_mag = nn.Sequential(
                    nn.Linear(1, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 1),
                )
                self.psi_phase = nn.Sequential(
                    nn.Linear(2, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 2),
                )
            else:
                self.psi_mag = nn.Sequential(
                    nn.Linear(1, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 1),
                )
                self.psi_phase = nn.Sequential(
                    nn.Linear(2, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 2),
                )

        # Small initial scale for stability
        self.mag_scale = nn.Parameter(torch.tensor(mag_init_scale))

        # Initialize to near-identity
        self._init_weights()

    def _init_weights(self):
        """Initialize to approximate identity transform."""
        # Small weights for residual-like behavior
        for module in [self.psi_mag, self.psi_phase]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=0.1)
                    nn.init.zeros_(layer.bias)

    def forward(self, Z: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the polarizing block.

        Args:
            Z: Complex tensor of shape (batch, n_tokens, d_complex)
            mask: Optional binary mask (batch, n_tokens) where 1=valid, 0=pad

        Returns:
            Complex tensor of same shape with polarizing interaction applied
        """
        # Use the aggregation strategy
        A = self.aggregation(Z, mask)

        # Decompose to polar coordinates
        mag = torch.abs(A)
        log_mag = torch.log(mag + 1e-6)  # Log-space for multiplicative dynamics

        # Phase as unit vector (more stable than angle)
        phase_vec = torch.stack([A.real, A.imag], dim=-1)  # (batch, 1, d, 2)
        phase_vec = phase_vec / (mag.unsqueeze(-1) + 1e-6)

        if self.per_dim:
            # Per-dimension transform: flatten to (batch, d_complex) or (batch, 2*d_complex)
            # Handle both (batch, d) and (batch, 1, d) shapes from aggregation
            orig_shape = log_mag.shape
            if log_mag.dim() == 3:
                # (batch, 1, d) -> (batch, d)
                log_mag_flat = log_mag.squeeze(1)
                phase_flat = phase_vec.squeeze(1).reshape(phase_vec.shape[0], -1)  # (batch, 2*d)
            else:
                log_mag_flat = log_mag
                phase_flat = phase_vec.reshape(phase_vec.shape[0], -1)

            # Transform
            mag_delta = self.psi_mag(log_mag_flat)
            log_mag_out = log_mag_flat + self.mag_scale * mag_delta

            phase_out_flat = self.psi_phase(phase_flat)
            phase_out_vec = phase_out_flat.reshape(-1, self.d_complex, 2)
            phase_out_vec = F.normalize(phase_out_vec, dim=-1)

            # Restore original shape if needed
            if len(orig_shape) == 3:
                log_mag_out = log_mag_out.unsqueeze(1)
                phase_out_vec = phase_out_vec.unsqueeze(1)
        else:
            # Original shared transform
            mag_delta = self.psi_mag(log_mag.unsqueeze(-1)).squeeze(-1)
            log_mag_out = log_mag + self.mag_scale * mag_delta

            phase_out_vec = self.psi_phase(phase_vec)
            phase_out_vec = F.normalize(phase_out_vec, dim=-1)

        # Recompose to complex
        r_out = torch.exp(log_mag_out)
        A_new = r_out * torch.complex(phase_out_vec[..., 0], phase_out_vec[..., 1])

        # Broadcast interaction back to all tokens (residual connection)
        return Z + A_new

    def get_aggregate(self, Z: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Get the aggregate representation (useful for diagnostics)."""
        A = self.aggregation(Z, mask)
        # If returned as (batch, 1, d), squeeze to (batch, d)
        if A.dim() == 3 and A.shape[1] == 1:
            return A.squeeze(1)
        return A
