"""
Phase-based Attention Mechanism.

Implements attention where scores are determined by phase alignment
in the complex plane. This makes "polarization" explicit:
tokens with similar phases attend to each other.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhaseAttentionBlock(nn.Module):
    """
    Attention block based on phase alignment.

    Attention scores are computed as the real part of the Hermitian dot product
    between Query and Key. This corresponds to:
    Score = |Q||K| cos(theta_Q - theta_K)

    If magnitudes are normalized, this is purely phase coherence.

    Args:
        d_complex: Complex dimension
        n_heads: Number of attention heads
        dropout: Dropout probability
    """

    def __init__(self, d_complex: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_complex = d_complex
        self.n_heads = n_heads
        self.head_dim = d_complex // n_heads

        assert self.head_dim * n_heads == d_complex, "d_complex must be divisible by n_heads"

        # Complex Linear projections
        # We implement complex linear as two real linears for flexibility
        # Or use the complex data type directly

        # To keep it simple and consistent with PyTorch complex:
        self.wk = nn.Linear(d_complex, d_complex, dtype=torch.cfloat)
        self.wq = nn.Linear(d_complex, d_complex, dtype=torch.cfloat)
        self.wv = nn.Linear(d_complex, d_complex, dtype=torch.cfloat)

        self.wo = nn.Linear(d_complex, d_complex, dtype=torch.cfloat)

        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, Z: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            Z: Input complex tensor (batch, seq_len, d_complex)
            mask: Binary mask (batch, seq_len)
        """
        B, N, D = Z.shape

        # Project
        # (B, N, H, D_head)
        q = self.wq(Z).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(Z).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(Z).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        # Compute scores: Real(Q @ K*)
        # (B, H, N, N)
        # torch.matmul with complex handles complex multiplication
        # We take real part for the score to use with standard softmax
        attn_scores = torch.matmul(q, k.transpose(-2, -1))
        attn_scores = attn_scores.real * self.scale

        # Masking
        if mask is not None:
            # mask: (B, N) -> (B, 1, 1, N) for broadcasting
            # We want to mask out invalid keys (columns)
            mask_expanded = mask.view(B, 1, 1, N).expand(-1, self.n_heads, N, -1)
            # Fill with -inf where mask is 0
            attn_scores = attn_scores.masked_fill(mask_expanded == 0, -1e9)

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Aggregate: Weights (real) * Values (complex)
        # PyTorch handles float * complex broadcasting
        # (B, H, N, N) * (B, H, N, D_head) -> (B, H, N, D_head)
        # cast attn_weights to complex for matmul
        out = torch.matmul(attn_weights.to(torch.cfloat), v)

        # Recombine heads
        out = out.transpose(1, 2).contiguous().view(B, N, D)

        # Output projection
        out = self.wo(out)

        return Z + out  # Residual connection
