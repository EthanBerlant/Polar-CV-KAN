"""
Complex-Valued Positional Encodings for CV-KAN.

These positional encodings naturally integrate with CV-KAN's phase mechanics,
encoding position information as phase shifts while maintaining unit magnitude.

Classes:
- ComplexPositionalEncoding: 1D sinusoidal encoding (phase encodes position)
- Complex2DPositionalEncoding: 2D encoding for images/grids
- LearnableComplexPositionalEncoding: Learnable embeddings in complex space
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


class ComplexPositionalEncoding(nn.Module):
    """
    1D complex-valued sinusoidal positional encoding.
    
    Unlike traditional sinusoidal encodings that concatenate sin/cos pairs,
    this encodes position as the phase of a complex number on the unit circle.
    This naturally integrates with CV-KAN's phase-based operations.
    
    Args:
        d_complex: Complex dimension size
        max_len: Maximum sequence length
        base: Base for the geometric progression of frequencies (default: 10000)
    """
    
    def __init__(
        self, 
        d_complex: int, 
        max_len: int = 5000,
        base: float = 10000.0,
    ):
        super().__init__()
        self.d_complex = d_complex
        
        # Compute position encoding as phases
        position = torch.arange(max_len).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_complex, 1) * (-math.log(base) / d_complex)
        )  # (d_complex,)
        
        # Phase = position * frequency
        phases = position * div_term  # (max_len, d_complex)
        
        # Complex encoding: e^(i * phase) = cos(phase) + i*sin(phase)
        # Magnitude = 1, phase encodes position
        pe = torch.exp(1j * phases)  # (max_len, d_complex)
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_complex)
    
    def forward(
        self, 
        x: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply positional encoding.
        
        Args:
            x: Complex tensor of shape (batch, seq_len, d_complex)
            positions: Optional custom positions (batch, seq_len) of integers.
                       If None, uses sequential positions 0, 1, 2, ...
        
        Returns:
            Complex tensor with positional encoding applied (multiplicative)
        """
        if positions is not None:
            # Gather encodings for custom positions
            batch, seq_len = positions.shape
            idx = positions.view(-1)  # (batch * seq_len,)
            pe_gathered = self.pe[0, idx]  # (batch * seq_len, d_complex)
            pe_gathered = pe_gathered.view(batch, seq_len, -1)
            return x * pe_gathered
        else:
            seq_len = x.shape[1]
            return x * self.pe[:, :seq_len]


class Complex2DPositionalEncoding(nn.Module):
    """
    2D complex-valued positional encoding for images/grids.
    
    Encodes both row and column positions using separate frequency bands,
    combining them multiplicatively in complex space.
    
    Args:
        d_complex: Complex dimension size
        max_height: Maximum height of the grid
        max_width: Maximum width of the grid
        base: Base for frequency computation
    """
    
    def __init__(
        self,
        d_complex: int,
        max_height: int = 224,
        max_width: int = 224,
        base: float = 10000.0,
    ):
        super().__init__()
        self.d_complex = d_complex
        self.max_height = max_height
        self.max_width = max_width
        
        # Split dimensions between height and width
        d_half = d_complex // 2
        d_h = d_half
        d_w = d_complex - d_half  # Handle odd d_complex
        
        # Height positional encoding
        h_position = torch.arange(max_height).unsqueeze(1)
        h_div_term = torch.exp(
            torch.arange(0, d_h, 1) * (-math.log(base) / d_h)
        )
        h_phases = h_position * h_div_term
        pe_h = torch.exp(1j * h_phases)  # (max_height, d_h)
        
        # Width positional encoding
        w_position = torch.arange(max_width).unsqueeze(1)
        w_div_term = torch.exp(
            torch.arange(0, d_w, 1) * (-math.log(base) / d_w)
        )
        w_phases = w_position * w_div_term
        pe_w = torch.exp(1j * w_phases)  # (max_width, d_w)
        
        self.register_buffer('pe_h', pe_h)
        self.register_buffer('pe_w', pe_w)
        self.d_h = d_h
        self.d_w = d_w
    
    def forward(
        self,
        x: torch.Tensor,
        spatial_shape: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Apply 2D positional encoding.
        
        Args:
            x: Complex tensor of shape:
               - (batch, H, W, d_complex) for 2D format
               - (batch, H*W, d_complex) for flattened format (requires spatial_shape)
            spatial_shape: (H, W) tuple if x is flattened
        
        Returns:
            Complex tensor with 2D positional encoding applied
        """
        original_shape = x.shape
        
        # Handle flattened input
        if len(x.shape) == 3:
            batch, n_tokens, d_complex = x.shape
            if spatial_shape is None:
                # Assume square
                H = W = int(n_tokens ** 0.5)
                assert H * W == n_tokens, f"Cannot infer spatial dims from {n_tokens} tokens"
            else:
                H, W = spatial_shape
            x = x.view(batch, H, W, d_complex)
        else:
            batch, H, W, d_complex = x.shape
        
        # Get encodings for H and W
        pe_h = self.pe_h[:H]  # (H, d_h)
        pe_w = self.pe_w[:W]  # (W, d_w)
        
        # Build 2D encoding grid
        # Height encoding: (H, 1, d_h) broadcast over width
        pe_h_expanded = pe_h.unsqueeze(1)  # (H, 1, d_h)
        # Width encoding: (1, W, d_w) broadcast over height
        pe_w_expanded = pe_w.unsqueeze(0)  # (1, W, d_w)
        
        # Concatenate along dimension
        pe_2d = torch.cat([
            pe_h_expanded.expand(H, W, -1),  # (H, W, d_h)
            pe_w_expanded.expand(H, W, -1),  # (H, W, d_w)
        ], dim=-1)  # (H, W, d_complex)
        
        # Apply multiplicatively
        x = x * pe_2d.unsqueeze(0)
        
        # Restore original shape
        if len(original_shape) == 3:
            x = x.view(batch, -1, d_complex)
        
        return x


class LearnableComplexPositionalEncoding(nn.Module):
    """
    Learnable complex-valued positional embeddings.
    
    Instead of fixed sinusoidal patterns, learns position-specific
    complex embeddings during training. Applied multiplicatively to
    preserve the structure of CV-KAN representations.
    
    Args:
        d_complex: Complex dimension size
        max_len: Maximum sequence length
        init_scale: Initialization scale for magnitudes (default: 1.0)
    """
    
    def __init__(
        self,
        d_complex: int,
        max_len: int = 5000,
        init_scale: float = 1.0,
    ):
        super().__init__()
        self.d_complex = d_complex
        self.max_len = max_len
        
        # Learn magnitude and phase separately for better optimization
        # Initialize magnitudes to 1 (log-scale for positivity)
        self.log_magnitudes = nn.Parameter(
            torch.zeros(max_len, d_complex)
        )
        # Phases initialized uniformly in [-pi, pi]
        self.phases = nn.Parameter(
            torch.randn(max_len, d_complex) * 0.1  # Small initial phases
        )
        
        self.init_scale = init_scale
    
    def forward(
        self, 
        x: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply learnable positional encoding.
        
        Args:
            x: Complex tensor of shape (batch, seq_len, d_complex)
            positions: Optional custom positions (batch, seq_len)
        
        Returns:
            Complex tensor with positional encoding applied
        """
        # Construct complex embedding from magnitude and phase
        magnitudes = torch.exp(self.log_magnitudes * self.init_scale)
        pe = magnitudes * torch.exp(1j * self.phases)  # (max_len, d_complex)
        
        if positions is not None:
            batch, seq_len = positions.shape
            idx = positions.view(-1)
            pe_gathered = pe[idx].view(batch, seq_len, -1)
            return x * pe_gathered
        else:
            seq_len = x.shape[1]
            return x * pe[:seq_len].unsqueeze(0)


class Learnable2DComplexPositionalEncoding(nn.Module):
    """
    Learnable 2D complex positional encoding.
    
    Learns separate row and column embeddings that are composed
    to form 2D position encodings. More parameter-efficient than
    learning full H*W embeddings.
    
    Args:
        d_complex: Complex dimension size
        max_height: Maximum grid height
        max_width: Maximum grid width
    """
    
    def __init__(
        self,
        d_complex: int,
        max_height: int = 224,
        max_width: int = 224,
    ):
        super().__init__()
        self.d_complex = d_complex
        
        d_half = d_complex // 2
        d_h = d_half
        d_w = d_complex - d_half
        
        # Learnable row and column embeddings
        self.row_log_mag = nn.Parameter(torch.zeros(max_height, d_h))
        self.row_phase = nn.Parameter(torch.randn(max_height, d_h) * 0.1)
        
        self.col_log_mag = nn.Parameter(torch.zeros(max_width, d_w))
        self.col_phase = nn.Parameter(torch.randn(max_width, d_w) * 0.1)
        
        self.d_h = d_h
        self.d_w = d_w
    
    def forward(
        self,
        x: torch.Tensor,
        spatial_shape: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Apply learnable 2D positional encoding.
        
        Args:
            x: Complex tensor, (batch, H, W, d) or (batch, H*W, d)
            spatial_shape: (H, W) tuple if flattened
        
        Returns:
            Encoded complex tensor
        """
        original_shape = x.shape
        
        if len(x.shape) == 3:
            batch, n_tokens, d_complex = x.shape
            if spatial_shape is None:
                H = W = int(n_tokens ** 0.5)
                assert H * W == n_tokens
            else:
                H, W = spatial_shape
            x = x.view(batch, H, W, d_complex)
        else:
            batch, H, W, d_complex = x.shape
        
        # Build embeddings
        row_mag = torch.exp(self.row_log_mag[:H])
        row_pe = row_mag * torch.exp(1j * self.row_phase[:H])  # (H, d_h)
        
        col_mag = torch.exp(self.col_log_mag[:W])
        col_pe = col_mag * torch.exp(1j * self.col_phase[:W])  # (W, d_w)
        
        # Combine
        pe_2d = torch.cat([
            row_pe.unsqueeze(1).expand(H, W, -1),
            col_pe.unsqueeze(0).expand(H, W, -1),
        ], dim=-1)
        
        x = x * pe_2d.unsqueeze(0)
        
        if len(original_shape) == 3:
            x = x.view(batch, -1, d_complex)
        
        return x
