"""PolarizingBlock: Core primitive for CV-KAN.

This module implements the fundamental operation:
1. Aggregate tokens via mean (bounded, stable)
2. Decompose to polar coordinates (log-magnitude + phase as sin/cos)
3. Transform with learnable 1D functions (KAN approximation)
4. Recompose and broadcast back to tokens
"""

import torch
import torch.nn.functional as F
from torch import nn

from .aggregation import GlobalMeanAggregation


class PolarizingBlock(nn.Module):
    """Core polarizing block that enables token interaction through
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
        interaction: "broadcast" (add f(A) to Z) or "pointwise" (Z + f(Z, A))
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
        interaction: str = "broadcast",
    ):
        super().__init__()
        self.d_complex = d_complex
        self.per_dim = per_dim
        self.dropout_rate = dropout
        self.interaction = interaction

        # Aggregation strategy (default: global mean)
        self.aggregation = aggregation if aggregation is not None else GlobalMeanAggregation()

        # Expanded hidden dimension
        hidden_dim = kan_hidden * mlp_expansion

        # Pointwise interaction doubles input dimension (Token + Aggregate)
        in_scale = 2 if interaction == "pointwise" else 1

        if per_dim:
            # Per-dimension transforms: input/output is full d_complex
            # This is the main source of parameter increase

            if deep_mlp:
                # 3-layer MLP for magnitude
                self.psi_mag = nn.Sequential(
                    nn.Linear(in_scale * d_complex, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, d_complex),
                )

                # 3-layer MLP for phase
                self.psi_phase = nn.Sequential(
                    nn.Linear(2 * in_scale * d_complex, hidden_dim),
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
                    nn.Linear(in_scale * d_complex, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, d_complex),
                )
                self.psi_phase = nn.Sequential(
                    nn.Linear(2 * in_scale * d_complex, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 2 * d_complex),
                )
        # Original shared 1D transforms (low param count)
        elif deep_mlp:
            self.psi_mag = nn.Sequential(
                nn.Linear(in_scale, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
            self.psi_phase = nn.Sequential(
                nn.Linear(2 * in_scale, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 2),
            )
        else:
            self.psi_mag = nn.Sequential(
                nn.Linear(in_scale, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
            self.psi_phase = nn.Sequential(
                nn.Linear(2 * in_scale, hidden_dim),
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
        """Forward pass of the polarizing block.

        Args:
            Z: Complex tensor of shape (batch, n_tokens, d_complex)
            mask: Optional binary mask (batch, n_tokens) where 1=valid, 0=pad

        Returns:
            Complex tensor of same shape with polarizing interaction applied
        """
        batch, n, d = Z.shape

        # 1. Aggregate context
        A = self.aggregation(Z, mask)  # (batch, 1, d) or (batch, d)
        if A.dim() == 2:
            A = A.unsqueeze(1)  # (batch, 1, d)

        # 2. Decompose context to polar
        mag_A = torch.abs(A)
        log_mag_A = torch.log(mag_A + 1e-6)
        phase_vec_A = torch.stack([A.real, A.imag], dim=-1) / (mag_A.unsqueeze(-1) + 1e-6)

        if self.interaction == "pointwise":
            # 3. Decompose tokens to polar
            mag_Z = torch.abs(Z)
            log_mag_Z = torch.log(mag_Z + 1e-6)
            phase_vec_Z = torch.stack([Z.real, Z.imag], dim=-1) / (mag_Z.unsqueeze(-1) + 1e-6)

            # Broadcast A to match Z
            log_mag_A_br = log_mag_A.expand_as(log_mag_Z)
            phase_vec_A_br = phase_vec_A.expand_as(phase_vec_Z)

            # Combine
            mag_in = torch.cat([log_mag_Z, log_mag_A_br], dim=-1)  # (batch, n, 2*d)
            phase_in = torch.cat([phase_vec_Z, phase_vec_A_br], dim=-1)  # (batch, n, d, 4)

            # Flatten for MLP
            if self.per_dim:
                mag_in = mag_in.reshape(batch * n, 2 * d)
                phase_in = phase_in.reshape(batch * n, 4 * d)

                mag_out_flat = self.psi_mag(mag_in).reshape(batch, n, d)
                phase_out_flat = self.psi_phase(phase_in).reshape(batch, n, d, 2)
            else:
                # Shared 1D transform
                mag_out_flat = self.psi_mag(mag_in.unsqueeze(-1)).squeeze(-1)
                phase_out_flat = self.psi_phase(phase_in)

            # Recompose update U with Tanh-Gating for absolute stability
            # Maps MLP output to [-5, 5] range, ensuring r_U in [0.006, 148.4]
            mag_out_gate = torch.tanh(mag_out_flat) * 5.0
            r_U = torch.exp(mag_out_gate)
            p_U = F.normalize(phase_out_flat, dim=-1, eps=1e-6)
            U = r_U * torch.complex(p_U[..., 0], p_U[..., 1])

            # Additive Residual
            return Z + self.mag_scale * U

        # Original Broadcast mode: only uses A to compute update
        if self.per_dim:
            log_mag_A_flat = log_mag_A.squeeze(1)
            phase_A_flat = phase_vec_A.squeeze(1).reshape(batch, -1)

            mag_out_flat = self.psi_mag(log_mag_A_flat).unsqueeze(1)  # (batch, 1, d)
            phase_out_flat = self.psi_phase(phase_A_flat).reshape(batch, 1, d, 2)
        else:
            mag_out_flat = self.psi_mag(log_mag_A.unsqueeze(-1)).squeeze(-1)
            phase_out_flat = self.psi_phase(phase_vec_A)

        # Recompose update U (broadcasted) with Tanh-Gating
        mag_out_gate = torch.tanh(mag_out_flat) * 5.0
        r_U = torch.exp(mag_out_gate)
        p_U = F.normalize(phase_out_flat, dim=-1, eps=1e-6)
        U = r_U * torch.complex(p_U[..., 0], p_U[..., 1])

        # Broadcast interaction back to all tokens (residual connection)
        return Z + self.mag_scale * U

    def get_aggregate(self, Z: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Get the aggregate representation (useful for diagnostics)."""
        A = self.aggregation(Z, mask)
        # If returned as (batch, 1, d), squeeze to (batch, d)
        if A.dim() == 3 and A.shape[1] == 1:
            return A.squeeze(1)
        return A
