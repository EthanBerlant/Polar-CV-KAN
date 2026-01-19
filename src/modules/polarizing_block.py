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


class PolarizingBlock(nn.Module):
    """
    Core polarizing block that enables token interaction through 
    phase alignment and magnitude polarization.
    
    Args:
        d_complex: Number of complex dimensions
        kan_hidden: Hidden size for the KAN approximation MLPs
        mag_init_scale: Initial scale for magnitude transform (small = stable)
    """
    
    def __init__(
        self,
        d_complex: int,
        kan_hidden: int = 32,
        mag_init_scale: float = 0.1,
    ):
        super().__init__()
        self.d_complex = d_complex
        
        # Learnable 1D functions (small MLPs as KAN approximation)
        # Magnitude transform: log(|z|) -> log(|z'|)
        self.psi_mag = nn.Sequential(
            nn.Linear(1, kan_hidden),
            nn.GELU(),
            nn.Linear(kan_hidden, 1),
        )
        
        # Phase transform: (cos θ, sin θ) -> (cos θ', sin θ')
        self.psi_phase = nn.Sequential(
            nn.Linear(2, kan_hidden),
            nn.GELU(),
            nn.Linear(kan_hidden, 2),
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
        # Aggregate: mean is bounded regardless of sequence length
        if mask is not None:
            # Masked mean
            mask_expanded = mask.unsqueeze(-1).float()  # (batch, n_tokens, 1)
            sum_Z = (Z * mask_expanded).sum(dim=1, keepdim=True)
            count = mask_expanded.sum(dim=1, keepdim=True).clamp(min=1.0)
            A = sum_Z / count
        else:
            A = Z.mean(dim=1, keepdim=True)  # (batch, 1, d_complex)
        
        # Decompose to polar coordinates

        mag = torch.abs(A)
        log_mag = torch.log(mag + 1e-6)  # Log-space for multiplicative dynamics
        
        # Phase as unit vector (more stable than angle)
        phase_vec = torch.stack([A.real, A.imag], dim=-1)  # (batch, 1, d, 2)
        phase_vec = phase_vec / (mag.unsqueeze(-1) + 1e-6)
        
        # Transform magnitude (with residual for stability)
        mag_delta = self.psi_mag(log_mag.unsqueeze(-1)).squeeze(-1)
        log_mag_out = log_mag + self.mag_scale * mag_delta
        
        # Transform phase (output normalized to stay on unit circle)
        phase_out_vec = self.psi_phase(phase_vec)
        phase_out_vec = F.normalize(phase_out_vec, dim=-1)
        
        # Recompose to complex
        r_out = torch.exp(log_mag_out)
        A_new = r_out * torch.complex(
            phase_out_vec[..., 0],
            phase_out_vec[..., 1]
        )
        
        # Broadcast interaction back to all tokens (residual connection)
        return Z + A_new
    
    def get_aggregate(self, Z: torch.Tensor) -> torch.Tensor:
        """Get the aggregate representation (useful for diagnostics)."""
        return Z.mean(dim=1)
