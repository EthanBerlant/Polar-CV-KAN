"""
CV-KAN: Full model for sequence classification.

Architecture:
- Input embedding → complex projection
- Stack of PolarizingBlocks with ComplexLayerNorm
- Global pooling → classifier head
"""

import torch
import torch.nn as nn
from typing import Literal, Optional

from ..modules import PolarizingBlock, ComplexLayerNorm, ComplexRMSNorm
from ..modules.multi_head import (
    EmergentHeadsPolarizing,
    PhaseOffsetPolarizing,
    FactoredHeadsPolarizing,
)
from ..modules.phase_attention import PhaseAttentionBlock


class ComplexEmbedding(nn.Module):
    """
    Project real-valued input to complex space.
    
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
        
        # Initialize to small values
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


class ComplexInputEmbedding(nn.Module):
    """
    Handle complex input directly (for the synthetic task).
    
    Optionally applies a learnable linear transform in complex space.
    
    Args:
        d_in: Input complex dimension
        d_out: Output complex dimension
        use_transform: Whether to apply a transform (if d_in == d_out, can skip)
    """
    
    def __init__(self, d_in: int, d_out: int, use_transform: bool = True):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.use_transform = use_transform and (d_in != d_out)
        
        if self.use_transform:
            # Complex linear transform
            self.weight = nn.Parameter(
                torch.randn(d_in, d_out, dtype=torch.cfloat) * 0.02
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass through or transform complex input."""
        if self.use_transform:
            return torch.einsum('...i,io->...o', x, self.weight)
        return x


class CVKAN(nn.Module):
    """
    Complete CV-KAN model for sequence classification.
    
    Args:
        d_input: Input dimension (real if input_type='real', complex if 'complex')
        d_complex: Complex representation dimension
        n_layers: Number of layers
        n_classes: Number of output classes
        kan_hidden: Hidden dimension for KAN MLPs
        head_approach: 'emergent', 'offset', or 'factored' (for polarizing block)
        n_heads: Number of heads (for offset/factored/attention)
        input_type: 'real' or 'complex'
        pooling: 'mean', 'max', or 'first'
        block_type: 'polarizing' or 'attention'
        norm_type: 'layer' or 'rms'
    """
    
    def __init__(
        self,
        d_input: int = 32,
        d_complex: int = 64,
        n_layers: int = 4,
        n_classes: int = 2,
        kan_hidden: int = 32,
        head_approach: Literal['emergent', 'offset', 'factored'] = 'emergent',
        n_heads: int = 8,
        input_type: Literal['real', 'complex'] = 'complex',
        pooling: Literal['mean', 'max', 'first'] = 'mean',
        block_type: Literal['polarizing', 'attention'] = 'polarizing',
        norm_type: Literal['layer', 'rms', 'none'] = 'none',
    ):
        super().__init__()
        self.d_complex = d_complex
        self.n_layers = n_layers
        self.pooling = pooling
        self.head_approach = head_approach
        self.block_type = block_type
        
        # Input embedding
        if input_type == 'real':
            self.embedding = ComplexEmbedding(d_input, d_complex)
        else:
            self.embedding = ComplexInputEmbedding(d_input, d_complex)
        
        # Build layers based on approach
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for _ in range(n_layers):
            # Select Block Type
            if block_type == 'polarizing':
                if head_approach == 'emergent':
                    layer = EmergentHeadsPolarizing(d_complex, kan_hidden)
                elif head_approach == 'offset':
                    d_per_head = d_complex // n_heads
                    layer = PhaseOffsetPolarizing(n_heads, d_per_head, kan_hidden)
                elif head_approach == 'factored':
                    d_per_head = d_complex // n_heads
                    layer = FactoredHeadsPolarizing(n_heads, d_complex, d_per_head, kan_hidden)
                else:
                    raise ValueError(f"Unknown head approach: {head_approach}")
            elif block_type == 'attention':
                layer = PhaseAttentionBlock(d_complex, n_heads=n_heads)
            else:
                 raise ValueError(f"Unknown block type: {block_type}")
            
            self.layers.append(layer)
            
            # Select Norm Type
            if norm_type == 'layer':
                self.norms.append(ComplexLayerNorm(d_complex))
            elif norm_type == 'rms':
                self.norms.append(ComplexRMSNorm(d_complex))
            elif norm_type == 'none':
                self.norms.append(nn.Identity())
            else:
                raise ValueError(f"Unknown norm type: {norm_type}")
        
        # Classification head (operates on magnitude)
        self.classifier = nn.Sequential(
            nn.Linear(d_complex, kan_hidden),
            nn.GELU(),
            nn.Linear(kan_hidden, n_classes),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        return_intermediates: bool = False,
    ) -> dict:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, n_tokens, d_input)
            mask: Optional binary mask (batch, n_tokens)
            return_intermediates: If True, return intermediate representations
        
        Returns:
            Dictionary with:
                - logits: Classification logits (batch, n_classes)
                - pooled: Pooled representation (batch, d_complex) - complex
                - intermediates: List of intermediate Z values (if requested)
        """
        intermediates = []
        
        # Embed to complex space
        Z = self.embedding(x)
        if return_intermediates:
            intermediates.append(Z.clone())
        
        # Apply polarizing layers
        for layer, norm in zip(self.layers, self.norms):
            Z = layer(Z, mask=mask)
            Z = norm(Z)
            if return_intermediates:
                intermediates.append(Z.clone())
        
        # Pool across tokens
        if self.pooling == 'mean':
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1).float()
                sum_Z = (Z * mask_expanded).sum(dim=1)
                count = mask_expanded.sum(dim=1).clamp(min=1.0)
                pooled = sum_Z / count
            else:
                pooled = Z.mean(dim=1)
        elif self.pooling == 'max':
            # Max by magnitude (mask out padding by setting mag to -inf)
            mags = torch.abs(Z)
            if mask is not None:
                mags = mags.masked_fill(~mask.bool().unsqueeze(-1), -1e9)
            
            max_idx = mags.argmax(dim=1, keepdim=True)
            pooled = torch.gather(Z, 1, max_idx.expand(-1, -1, Z.shape[-1])).squeeze(1)
        elif self.pooling == 'first':
            pooled = Z[:, 0]
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        
        # Classify based on magnitude (phase-invariant)
        features = torch.abs(pooled)
        logits = self.classifier(features)
        
        output = {
            'logits': logits,
            'pooled': pooled,
        }
        
        if return_intermediates:
            output['intermediates'] = intermediates
        
        return output
    
    def get_phase_coherence(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Compute phase coherence across tokens.
        
        High coherence = tokens have similar phases = attention-like alignment.
        
        Args:
            Z: Complex tensor of shape (batch, n_tokens, d_complex)
        
        Returns:
            Coherence per dimension of shape (batch, d_complex)
        """
        phases = torch.angle(Z)  # (batch, n_tokens, d)
        
        # Circular mean resultant length
        cos_sum = torch.cos(phases).sum(dim=1)
        sin_sum = torch.sin(phases).sum(dim=1)
        n = phases.shape[1]
        R = torch.sqrt(cos_sum ** 2 + sin_sum ** 2) / n
        
        return R  # [0, 1] per dimension, 1 = perfect coherence


class CVKANTokenClassifier(CVKAN):
    """
    CV-KAN variant for per-token classification (e.g., signal/noise detection).
    
    Instead of pooling, outputs logits per token.
    """
    
    def __init__(self, **kwargs):
        # Remove n_classes from kwargs temporarily
        n_classes = kwargs.pop('n_classes', 2)
        super().__init__(n_classes=n_classes, **kwargs)
        
        # Replace classifier with per-token version
        self.classifier = nn.Sequential(
            nn.Linear(kwargs.get('d_complex', 64), kwargs.get('kan_hidden', 32)),
            nn.GELU(),
            nn.Linear(kwargs.get('kan_hidden', 32), n_classes),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        return_intermediates: bool = False,
    ) -> dict:
        """
        Forward pass for token classification.
        
        Returns:
            Dictionary with:
                - token_logits: Per-token logits (batch, n_tokens, n_classes)
                - sequence_logits: Sequence-level logits (batch, n_classes)
                - Z: Final complex representation
        """
        intermediates = []
        
        # Embed
        Z = self.embedding(x)
        if return_intermediates:
            intermediates.append(Z.clone())
        
        # Apply layers
        for layer, norm in zip(self.layers, self.norms):
            Z = layer(Z, mask=mask)
            Z = norm(Z)
            if return_intermediates:
                intermediates.append(Z.clone())
        
        # Per-token classification (on magnitudes)
        token_features = torch.abs(Z)  # (batch, n_tokens, d)
        token_logits = self.classifier(token_features)  # (batch, n_tokens, n_classes)
        
        # Sequence-level: pool token logits
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            sum_logits = (token_logits * mask_expanded).sum(dim=1)
            count = mask_expanded.sum(dim=1).clamp(min=1.0)
            sequence_logits = sum_logits / count
        else:
            sequence_logits = token_logits.mean(dim=1)
        
        output = {
            'token_logits': token_logits,
            'sequence_logits': sequence_logits,
            'Z': Z,
        }
        
        if return_intermediates:
            output['intermediates'] = intermediates
        
        return output
