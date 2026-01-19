"""
Gated polarization module.

Allows the network to learn how aggressive to be with polarization,
starting near identity and becoming more nonlinear if helpful.
"""

import torch
import torch.nn as nn


class GatedPolarization(nn.Module):
    """
    Soft polarization with learnable strength.
    
    Interpolates between identity (no transform) and full polarization.
    Network learns on its own how much polarization to apply.
    
    Args:
        d_complex: Number of complex dimensions
        hidden_dim: Hidden size for the polarization transform
    """
    
    def __init__(self, d_complex: int, hidden_dim: int = 32):
        super().__init__()
        self.d_complex = d_complex
        
        # Polarization strength starts at 0 (identity)
        self.polarization_strength = nn.Parameter(torch.tensor(0.0))
        
        # The "aggressive" polarization function
        self.polarize_mag = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        
        # Initialize small
        self._init_weights()
    
    def _init_weights(self):
        for layer in self.polarize_mag:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.1)
                nn.init.zeros_(layer.bias)
    
    def forward(self, mag: torch.Tensor) -> torch.Tensor:
        """
        Apply gated polarization to magnitudes.
        
        Args:
            mag: Magnitude tensor of shape (..., d_complex)
        
        Returns:
            Polarized magnitudes of same shape
        """
        # How much polarization to apply (0 = identity, 1 = full)
        alpha = torch.sigmoid(self.polarization_strength)
        
        # Compute polarized version
        log_mag = torch.log(mag + 1e-6)
        mag_polarized = torch.exp(
            self.polarize_mag(log_mag.unsqueeze(-1)).squeeze(-1)
        )
        
        # Interpolate
        return (1 - alpha) * mag + alpha * mag_polarized
    
    def get_strength(self) -> float:
        """Get current polarization strength as a float."""
        with torch.no_grad():
            return torch.sigmoid(self.polarization_strength).item()


class GatedPolarizingBlock(nn.Module):
    """
    A PolarizingBlock with gated polarization strength.
    
    Combines the core polarizing operation with learnable control
    over how aggressive the transformation is.
    """
    
    def __init__(
        self,
        d_complex: int,
        kan_hidden: int = 32,
    ):
        super().__init__()
        self.d_complex = d_complex
        
        # Gated magnitude transform
        self.gated_mag = GatedPolarization(d_complex, kan_hidden)
        
        # Phase transform (same as basic block)
        self.psi_phase = nn.Sequential(
            nn.Linear(2, kan_hidden),
            nn.GELU(),
            nn.Linear(kan_hidden, 2),
        )
        
        # Phase gate (separate from magnitude)
        self.phase_strength = nn.Parameter(torch.tensor(0.0))
        
        self._init_weights()
    
    def _init_weights(self):
        for layer in self.psi_phase:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.1)
                nn.init.zeros_(layer.bias)
    
    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with gated polarization.
        
        Args:
            Z: Complex tensor of shape (batch, n_tokens, d_complex)
        
        Returns:
            Complex tensor with gated polarizing interaction
        """
        import torch.nn.functional as F
        
        # Aggregate
        A = Z.mean(dim=1, keepdim=True)
        
        # Decompose
        mag = torch.abs(A)
        phase_vec = torch.stack([A.real, A.imag], dim=-1)
        phase_vec = phase_vec / (mag.unsqueeze(-1) + 1e-6)
        
        # Gated magnitude transform
        mag_out = self.gated_mag(mag)
        
        # Gated phase transform
        alpha_phase = torch.sigmoid(self.phase_strength)
        phase_transformed = self.psi_phase(phase_vec)
        phase_transformed = F.normalize(phase_transformed, dim=-1)
        phase_out_vec = (1 - alpha_phase) * phase_vec + alpha_phase * phase_transformed
        phase_out_vec = F.normalize(phase_out_vec, dim=-1)  # Ensure unit circle
        
        # Recompose
        A_new = mag_out * torch.complex(phase_out_vec[..., 0], phase_out_vec[..., 1])
        
        return Z + A_new
