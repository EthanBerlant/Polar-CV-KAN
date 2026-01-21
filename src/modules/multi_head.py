"""
Multi-head approaches for CV-KAN.

Three approaches as described in the design:
- Approach A: Emergent heads via channel diversity
- Approach B: Explicit phase offsets
- Approach C: Factored magnitude-phase heads
"""

import math

import torch
import torch.nn as nn

from .polarizing_block import PolarizingBlock


class EmergentHeadsPolarizing(nn.Module):
    """
    Approach A: Emergent Heads via Channel Diversity.

    Uses multiple complex dimensions, each evolving independently.
    Heads are implicit - diversity emerges from different dimensions
    learning different patterns.

    Add diversity_loss (from losses.regularizers) to training loss
    to encourage different dimensions to specialize.

    Args:
        d_complex: Number of complex dimensions (implicit heads)
        kan_hidden: Hidden size for KAN MLPs
    """

    def __init__(self, d_complex: int, kan_hidden: int = 32, aggregation=None, **kwargs):
        super().__init__()
        self.d_complex = d_complex
        self.block = PolarizingBlock(d_complex, kan_hidden, aggregation=aggregation, **kwargs)

    def forward(self, Z: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass - just applies the block.

        Args:
            Z: Complex tensor of shape (batch, n_tokens, d_complex)
            mask: Optional mask

        Returns:
            Complex tensor of same shape
        """
        return self.block(Z, mask=mask)

    def get_phase_matrix(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Get phase matrix for diversity analysis.

        Returns:
            Phase tensor of shape (batch, n_tokens, d_complex)
        """
        return torch.angle(Z)


class PhaseOffsetPolarizing(nn.Module):
    """
    Approach B: Explicit Phase Offsets.

    Each head has a fixed reference phase offset (0, 2π/H, 4π/H, ...).
    Tokens aligning with a head's reference constructively interfere.
    Same underlying dynamics, different frame of reference.

    Pros:
        - Explicit head structure
        - Shared parameters across heads (efficient)
        - Guaranteed diversity of reference frames

    Args:
        n_heads: Number of explicit heads
        d_per_head: Complex dimensions per head
        kan_hidden: Hidden size for KAN MLPs
    """

    def __init__(
        self,
        n_heads: int,
        d_per_head: int,
        kan_hidden: int = 32,
        aggregation=None,  # Shared aggregation
        **kwargs,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.d_per_head = d_per_head

        # Fixed phase offsets: 0, 2π/H, 4π/H, ...
        offsets = torch.arange(n_heads) * (2 * math.pi / n_heads)
        self.register_buffer("phase_offsets", offsets)

        # Shared polarizing transform (parameter efficient)
        self.polarizer = PolarizingBlock(d_per_head, kan_hidden, aggregation=aggregation, **kwargs)

    def forward(self, Z: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass with phase offset heads.

        Args:
            Z: Complex tensor of shape (batch, n_tokens, n_heads, d_per_head)
               OR (batch, n_tokens, n_heads * d_per_head) which will be reshaped
            mask: Optional mask

        Returns:
            Complex tensor of same shape
        """
        # Handle both input shapes
        original_shape = Z.shape
        if len(Z.shape) == 3:
            batch, n_tokens, d = Z.shape
            Z = Z.view(batch, n_tokens, self.n_heads, self.d_per_head)

        batch, n_tokens, n_heads, d_per_head = Z.shape

        # Rotate each head by its offset
        # phase_offsets: (n_heads,) -> (1, 1, n_heads, 1)
        rotation = torch.exp(1j * self.phase_offsets[None, None, :, None])
        Z_rotated = Z * rotation

        # Apply shared polarizer per head (flatten heads into batch)
        Z_flat = Z_rotated.permute(0, 2, 1, 3).reshape(batch * n_heads, n_tokens, d_per_head)

        # Expand mask if provided
        mask_flat = None
        if mask is not None:
            # mask: (batch, n_tokens) -> (batch, n_heads, n_tokens) -> (batch*n_heads, n_tokens)
            mask_flat = mask.unsqueeze(1).expand(-1, n_heads, -1).reshape(batch * n_heads, n_tokens)

        out_flat = self.polarizer(Z_flat, mask=mask_flat)
        out = out_flat.reshape(batch, n_heads, n_tokens, d_per_head).permute(0, 2, 1, 3)

        # Rotate back
        out = out * torch.exp(-1j * self.phase_offsets[None, None, :, None])

        # Return in original shape
        if len(original_shape) == 3:
            out = out.reshape(batch, n_tokens, -1)

        return out


class FactoredHeadsPolarizing(nn.Module):
    """
    Approach C: Factored Magnitude-Phase Heads.

    Separates "what to select" (phase) from "how much" (magnitude).
    Each head has its own phase projection, but shares magnitude processing.

    Pros:
        - Clean factorization
        - Heads differ only in what phase relationships they detect

    Cons:
        - More parameters than Approach B

    Args:
        n_heads: Number of heads
        d_model: Input dimension
        d_per_head: Dimension per head (output will be n_heads * d_per_head)
        kan_hidden: Hidden size for KAN MLPs
    """

    def __init__(
        self,
        n_heads: int,
        d_model: int,
        d_per_head: int,
        kan_hidden: int = 32,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_per_head = d_per_head

        # Each head gets its own phase projection
        self.phase_projections = nn.Parameter(
            torch.randn(n_heads, d_model, d_per_head, dtype=torch.cfloat) * 0.02
        )

        # Shared magnitude transform
        self.shared_mag_transform = nn.Sequential(
            nn.Linear(1, kan_hidden),
            nn.GELU(),
            nn.Linear(kan_hidden, 1),
        )

        # Scale for residual
        self.mag_scale = nn.Parameter(torch.tensor(0.1))

        self._init_weights()

    def _init_weights(self):
        for layer in self.shared_mag_transform:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.1)
                nn.init.zeros_(layer.bias)

    def forward(self, Z: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass with factored heads.

        Args:
            Z: Complex tensor of shape (batch, n_tokens, d_model)
            mask: Optional mask

        Returns:
            Complex tensor of shape (batch, n_tokens, n_heads * d_per_head)
        """
        batch, n_tokens, d_model = Z.shape

        # Project to head-specific phase spaces
        # Z: (batch, n_tokens, d_model)
        # phase_projections: (n_heads, d_model, d_per_head)
        # Result: (batch, n_tokens, n_heads, d_per_head)
        Z_heads = torch.einsum("btd,hde->bthe", Z, self.phase_projections)

        # Aggregate per head
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).unsqueeze(-1).float()  # (batch, n_tokens, 1, 1)
            sum_Z = (Z_heads * mask_expanded).sum(dim=1, keepdim=True)
            count = mask_expanded.sum(dim=1, keepdim=True).clamp(min=1.0)
            A = sum_Z / count
        else:
            A = Z_heads.mean(dim=1, keepdim=True)  # (batch, 1, n_heads, d_per_head)

        # Shared magnitude processing
        mags = torch.abs(A)
        log_mags = torch.log(mags + 1e-6)
        mag_delta = self.shared_mag_transform(log_mags.unsqueeze(-1)).squeeze(-1)
        mags_out = torch.exp(log_mags + self.mag_scale * mag_delta)

        # Phases from aggregate
        phases = torch.angle(A)

        # Recompose and add residual
        A_new = mags_out * torch.exp(1j * phases)

        # Broadcast back
        out = Z_heads + A_new

        # Flatten heads
        return out.reshape(batch, n_tokens, -1)


class MultiHeadPolarizingMixer(nn.Module):
    """
    Wrapper that applies different multi-head approaches and mixes.

    Useful for ablation studies comparing approaches.

    Args:
        approach: One of 'emergent', 'offset', or 'factored'
        **kwargs: Arguments passed to the specific approach
    """

    def __init__(self, approach: str, **kwargs):
        super().__init__()
        self.approach = approach

        if approach == "emergent":
            self.block = EmergentHeadsPolarizing(
                d_complex=kwargs.get("d_complex", 64),
                kan_hidden=kwargs.get("kan_hidden", 32),
            )
        elif approach == "offset":
            self.block = PhaseOffsetPolarizing(
                n_heads=kwargs.get("n_heads", 8),
                d_per_head=kwargs.get("d_per_head", 8),
                kan_hidden=kwargs.get("kan_hidden", 32),
            )
        elif approach == "factored":
            self.block = FactoredHeadsPolarizing(
                n_heads=kwargs.get("n_heads", 8),
                d_model=kwargs.get("d_model", 64),
                d_per_head=kwargs.get("d_per_head", 8),
                kan_hidden=kwargs.get("kan_hidden", 32),
            )
        else:
            raise ValueError(f"Unknown approach: {approach}")

    def forward(self, Z: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        return self.block(Z, mask=mask)
