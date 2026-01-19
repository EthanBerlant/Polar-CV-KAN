"""
Complex-valued normalization layers.

These preserve the phase structure while normalizing magnitudes,
preventing explosion/collapse in deep stacks.
"""

import torch
import torch.nn as nn


class ComplexLayerNorm(nn.Module):
    """
    Layer normalization for complex tensors.
    
    Normalizes log-magnitudes to zero mean, unit variance while
    preserving phase information. This is crucial for stability
    in deep stacks of polarizing blocks.
    
    Args:
        d_complex: Number of complex dimensions
        eps: Small constant for numerical stability
    """
    
    def __init__(self, d_complex: int, eps: float = 1e-6):
        super().__init__()
        self.d_complex = d_complex
        self.eps = eps
        
        # Learnable affine transform (applied to log-magnitude)
        self.gamma = nn.Parameter(torch.ones(d_complex))
        self.beta = nn.Parameter(torch.zeros(d_complex))
    
    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Normalize complex tensor.
        
        Args:
            Z: Complex tensor of shape (..., d_complex)
        
        Returns:
            Normalized complex tensor of same shape
        """
        # Extract magnitude and phase
        mag = torch.abs(Z)
        phase = torch.angle(Z)
        
        # Normalize log-magnitudes
        log_mag = torch.log(mag + self.eps)
        mean = log_mag.mean(dim=-1, keepdim=True)
        std = log_mag.std(dim=-1, keepdim=True)
        log_mag_norm = (log_mag - mean) / (std + self.eps)
        
        # Apply affine transform
        log_mag_norm = self.gamma * log_mag_norm + self.beta
        
        # Reconstruct with normalized magnitude, original phase
        mag_norm = torch.exp(log_mag_norm)
        return mag_norm * torch.exp(1j * phase)


class ComplexRMSNorm(nn.Module):
    """
    RMS normalization for complex tensors.
    
    Simpler than LayerNorm - just normalizes by RMS of magnitudes.
    Often works just as well with fewer parameters.
    
    Args:
        d_complex: Number of complex dimensions
        eps: Small constant for numerical stability
    """
    
    def __init__(self, d_complex: int, eps: float = 1e-6):
        super().__init__()
        self.d_complex = d_complex
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_complex))
    
    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Normalize complex tensor by RMS.
        
        Args:
            Z: Complex tensor of shape (..., d_complex)
        
        Returns:
            Normalized complex tensor of same shape
        """
        # RMS of magnitudes
        mag_sq = torch.abs(Z) ** 2
        rms = torch.sqrt(mag_sq.mean(dim=-1, keepdim=True) + self.eps)
        
        # Normalize and scale
        return self.scale * Z / rms
