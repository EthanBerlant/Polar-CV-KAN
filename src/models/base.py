"""
BaseCVKAN: Abstract base class for all CV-KAN models.

Provides shared functionality:
- Log-magnitude centering to prevent drift
- Layer stacking with configurable block types
- Pooling strategies (mean, max, attention)
- Magnitude-based feature extraction

All domain-specific models (image, audio, timeseries) should inherit
from this class to ensure consistent behavior.
"""

from abc import ABC, abstractmethod
from typing import Literal, Optional, List

import torch
import torch.nn as nn

from ..modules.polarizing_block import PolarizingBlock
from ..modules.multi_head import EmergentHeadsPolarizing


class BaseCVKAN(nn.Module, ABC):
    """
    Abstract base class for CV-KAN models.
    
    Subclasses must implement:
    - _embed(): Convert input to complex representation
    - forward(): Full forward pass with domain-specific logic
    
    Provides:
    - Log-magnitude centering (prevents exponential drift)
    - Layer construction helpers
    - Pooling strategies
    - Magnitude feature extraction
    
    Args:
        d_complex: Complex representation dimension
        n_layers: Number of CV-KAN layers
        kan_hidden: Hidden size for KAN MLPs
        pooling: Pooling strategy ('mean', 'max', 'attention')
        center_magnitudes: Whether to center log-magnitudes after layers
        use_multi_head: Use EmergentHeadsPolarizing instead of basic PolarizingBlock
        skip_connections: If True, add residual skip connections between layers
    """
    
    def __init__(
        self,
        d_complex: int,
        n_layers: int,
        kan_hidden: int = 32,
        pooling: Literal['mean', 'max', 'attention'] = 'mean',
        center_magnitudes: bool = True,
        use_multi_head: bool = False,
        skip_connections: bool = False,
    ):
        super().__init__()
        self.d_complex = d_complex
        self.n_layers = n_layers
        self.kan_hidden = kan_hidden
        self.pooling_strategy = pooling
        self.center_magnitudes = center_magnitudes
        self.use_multi_head = use_multi_head
        self.skip_connections = skip_connections
        
        # Build layers
        self.layers = self._build_layers()
        
        # Attention pooling query (if needed)
        if pooling == 'attention':
            self.pool_query = nn.Parameter(
                torch.randn(1, 1, d_complex, dtype=torch.cfloat) * 0.02
            )
    
    def _build_layers(self, aggregation: Optional[nn.Module] = None) -> nn.ModuleList:
        """
        Build the stack of CV-KAN layers.
        
        Args:
            aggregation: Optional aggregation module to use in each layer
        
        Returns:
            ModuleList of CV-KAN layers
        """
        layers = nn.ModuleList()
        for _ in range(self.n_layers):
            if self.use_multi_head:
                layer = EmergentHeadsPolarizing(self.d_complex, self.kan_hidden)
            else:
                if aggregation is not None:
                    layer = PolarizingBlock(self.d_complex, self.kan_hidden, aggregation=aggregation)
                else:
                    layer = PolarizingBlock(self.d_complex, self.kan_hidden)
            layers.append(layer)
        return layers
    
    def _center_log_magnitudes(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Center log-magnitudes across tokens to prevent drift.
        
        This is the CV-KAN alternative to normalization. It preserves:
        - Relative magnitudes between tokens (the "attention" signal)
        - Per-dimension magnitude specialization
        
        While preventing:
        - Exponential magnitude growth/collapse over many layers
        
        Args:
            Z: Complex tensor of shape (batch, n_tokens, d_complex)
        
        Returns:
            Complex tensor with centered log-magnitudes
        """
        log_mags = torch.log(torch.abs(Z) + 1e-6)
        # Center across tokens (dim=1), not dimensions
        log_mags_centered = log_mags - log_mags.mean(dim=1, keepdim=True)
        # Reconstruct: exp(centered log-mag) * unit phase
        return torch.exp(log_mags_centered) * torch.exp(1j * torch.angle(Z))
    
    def _apply_layers(
        self, 
        z: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply CV-KAN layers with optional magnitude centering.
        
        Args:
            z: Complex tensor of shape (batch, n_tokens, d_complex)
            mask: Optional mask (batch, n_tokens)
        
        Returns:
            Processed complex tensor
        """
        for layer in self.layers:
            if self.skip_connections:
                z = z + layer(z, mask=mask)
            else:
                z = layer(z, mask=mask)
        
        if self.center_magnitudes:
            z = self._center_log_magnitudes(z)
        
        return z
    
    def _pool(
        self, 
        z: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Pool complex tensor across the token/sequence dimension.
        
        Args:
            z: Complex tensor of shape (batch, n_tokens, d_complex)
            mask: Optional mask (batch, n_tokens)
        
        Returns:
            Pooled tensor of shape (batch, d_complex)
        """
        if self.pooling_strategy == 'mean':
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1).float()
                sum_z = (z * mask_expanded).sum(dim=1)
                count = mask_expanded.sum(dim=1).clamp(min=1.0)
                return sum_z / count
            return z.mean(dim=1)
        
        elif self.pooling_strategy == 'max':
            mags = torch.abs(z)
            if mask is not None:
                mags = mags.masked_fill(~mask.bool().unsqueeze(-1), -1e9)
            # Max by sum of magnitudes across dimensions
            max_idx = mags.sum(dim=-1).argmax(dim=1, keepdim=True)
            return torch.gather(
                z, 1, 
                max_idx.unsqueeze(-1).expand(-1, -1, self.d_complex)
            ).squeeze(1)
        
        elif self.pooling_strategy == 'attention':
            # Attention-based pooling using complex inner product
            query = self.pool_query.expand(z.shape[0], -1, -1)
            attn = torch.einsum('btd,bqd->btq', z, query.conj())
            attn = torch.softmax(attn.abs(), dim=1)
            if mask is not None:
                attn = attn * mask.unsqueeze(-1)
                attn = attn / attn.sum(dim=1, keepdim=True).clamp(min=1e-6)
            return torch.einsum('btq,btd->bqd', attn, z).squeeze(1)
        
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
    
    def _extract_features(self, pooled: torch.Tensor) -> torch.Tensor:
        """
        Extract real-valued features from pooled complex representation.
        
        Uses magnitude (phase-invariant) for classification.
        
        Args:
            pooled: Complex tensor of shape (batch, d_complex)
        
        Returns:
            Real tensor of shape (batch, d_complex)
        """
        return torch.abs(pooled)
    
    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> dict:
        """
        Forward pass. Must be implemented by subclasses.
        
        Args:
            x: Input tensor (domain-specific shape)
            **kwargs: Additional arguments
        
        Returns:
            Dictionary with model outputs
        """
        pass


class ComplexEmbedding(nn.Module):
    """
    Project real-valued input to complex space.
    
    Used by models that receive real-valued inputs (text, tabular).
    
    Args:
        input_dim: Input feature dimension
        d_complex: Output complex dimension
    """
    
    def __init__(self, input_dim: int, d_complex: int):
        super().__init__()
        self.d_complex = d_complex
        
        # Project to real and imaginary parts
        self.proj_real = nn.Linear(input_dim, d_complex)
        self.proj_imag = nn.Linear(input_dim, d_complex)
        
        # Initialize to small values for stability
        nn.init.xavier_uniform_(self.proj_real.weight, gain=0.5)
        nn.init.xavier_uniform_(self.proj_imag.weight, gain=0.5)
        nn.init.zeros_(self.proj_real.bias)
        nn.init.zeros_(self.proj_imag.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project to complex space.
        
        Args:
            x: Real tensor of shape (..., input_dim)
        
        Returns:
            Complex tensor of shape (..., d_complex)
        """
        real = self.proj_real(x)
        imag = self.proj_imag(x)
        return torch.complex(real, imag)


def build_classifier_head(
    d_complex: int, 
    n_classes: int, 
    hidden_dim: Optional[int] = None,
    dropout: float = 0.1,
) -> nn.Module:
    """
    Build a standard classification head.
    
    Args:
        d_complex: Input dimension (from magnitude features)
        n_classes: Number of output classes
        hidden_dim: Hidden layer dimension (default: 2 * d_complex)
        dropout: Dropout probability
    
    Returns:
        Classification head module
    """
    hidden_dim = hidden_dim or d_complex * 2
    return nn.Sequential(
        nn.Linear(d_complex, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, n_classes),
    )
