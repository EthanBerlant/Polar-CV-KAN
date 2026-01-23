"""Hierarchical Polarization: Recursive multi-scale polarization.

Inspired by polar error correction, this module applies polarization
recursively at multiple scales: local groups first, then progressively
larger groups, until reaching a global aggregate.

This addresses the "mean aggregation noise" problem by:
1. Operating locally first (higher SNR)
2. Allowing signal to compound at each level
3. By global level, noise has been suppressed through multiple stages
"""

import math

import torch
from torch import nn

from .aggregation import GlobalMeanAggregation, MagnitudeWeightedAggregation
from .polarizing_block import PolarizingBlock


class HierarchicalPolarization(nn.Module):
    """Recursive multi-scale polarization inspired by polar error correction.

    Bottom-up: local groups → global
    Optional top-down: global → local (reconstruction/refinement)

    Args:
        d_complex: Complex dimension
        max_levels: Maximum recursion depth (auto-computed from seq_len if None)
        weight_sharing: How to share transform weights:
            - 'shared': Single transform for all levels
            - 'per_level': One transform per level (default)
            - 'per_direction': Separate up/down transforms
        aggregation: Aggregation strategy within groups:
            - 'mean': Simple mean (default)
            - 'magnitude_weighted': Use magnitude as weights
        top_down: Whether to include top-down pass:
            - 'none': Bottom-up only (default)
            - 'mirror': Reuse bottom-up transforms in reverse
            - 'learned': Separate learned transforms for top-down
        kan_hidden: Hidden dimension for polarizing blocks
        dropout: Dropout rate
    """

    def __init__(
        self,
        d_complex: int,
        max_levels: int | None = None,
        weight_sharing: str = "per_level",
        aggregation: str = "mean",
        top_down: str = "none",
        kan_hidden: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_complex = d_complex
        self.max_levels = max_levels
        self.weight_sharing = weight_sharing
        self.aggregation_type = aggregation
        self.top_down_mode = top_down
        self.kan_hidden = kan_hidden
        self.dropout_rate = dropout

        # Create aggregation module
        if aggregation == "mean":
            self.aggregator = GlobalMeanAggregation()
        elif aggregation == "magnitude_weighted":
            self.aggregator = MagnitudeWeightedAggregation()
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

        # Create transforms based on weight sharing strategy
        # We'll create up to 10 levels (supports seq_len up to 1024)
        n_transforms = 10 if max_levels is None else max_levels

        if weight_sharing == "shared":
            self.up_transforms = nn.ModuleList(
                [PolarizingBlock(d_complex, kan_hidden=kan_hidden, dropout=dropout)]
            )
        elif weight_sharing in ["per_level", "per_direction"]:
            self.up_transforms = nn.ModuleList(
                [
                    PolarizingBlock(d_complex, kan_hidden=kan_hidden, dropout=dropout)
                    for _ in range(n_transforms)
                ]
            )
        else:
            raise ValueError(f"Unknown weight_sharing: {weight_sharing}")

        # Top-down transforms (if needed)
        if top_down == "learned":
            self.down_transforms = nn.ModuleList(
                [
                    PolarizingBlock(d_complex, kan_hidden=kan_hidden, dropout=dropout)
                    for _ in range(n_transforms)
                ]
            )
        else:
            self.down_transforms = None

    def _get_transform(self, level: int, direction: str = "up") -> PolarizingBlock:
        """Get the appropriate transform for a given level and direction."""
        if direction == "up":
            if self.weight_sharing == "shared":
                return self.up_transforms[0]
            return self.up_transforms[min(level, len(self.up_transforms) - 1)]
        if self.top_down_mode == "mirror":
            if self.weight_sharing == "shared":
                return self.up_transforms[0]
            return self.up_transforms[min(level, len(self.up_transforms) - 1)]
        if self.top_down_mode == "learned":
            return self.down_transforms[min(level, len(self.down_transforms) - 1)]
        raise ValueError(f"No down transform for top_down={self.top_down_mode}")

    def _pad_to_power_of_2(self, Z: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Pad sequence length to next power of 2."""
        batch, seq_len, d = Z.shape
        if seq_len & (seq_len - 1) == 0:  # Already power of 2
            return Z, seq_len

        next_pow2 = 2 ** math.ceil(math.log2(seq_len))
        pad_len = next_pow2 - seq_len

        # Pad with zeros (complex)
        padding = torch.zeros(batch, pad_len, d, dtype=Z.dtype, device=Z.device)
        Z_padded = torch.cat([Z, padding], dim=1)

        return Z_padded, seq_len

    def _aggregate_groups(
        self, Z: torch.Tensor, group_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Aggregate within non-overlapping groups.

        Args:
            Z: (batch, seq_len, d_complex)
            group_size: Size of each group

        Returns:
            aggregates: (batch, n_groups, d_complex) - one aggregate per group
            Z_grouped: (batch, n_groups, group_size, d_complex) - reshaped input
        """
        batch, seq_len, d = Z.shape
        n_groups = seq_len // group_size

        # Reshape into groups
        Z_grouped = Z.view(batch, n_groups, group_size, d)

        # Aggregate within each group
        # Reshape for aggregator: (batch * n_groups, group_size, d)
        Z_flat = Z_grouped.view(batch * n_groups, group_size, d)

        if self.aggregation_type == "mean":
            A_flat = Z_flat.mean(dim=1)  # (batch * n_groups, d)
        else:  # magnitude_weighted
            mag = torch.abs(Z_flat).mean(dim=-1, keepdim=True)
            weights = mag / (mag.sum(dim=1, keepdim=True) + 1e-6)
            A_flat = (Z_flat * weights).sum(dim=1)

        aggregates = A_flat.view(batch, n_groups, d)
        return aggregates, Z_grouped

    def forward(self, Z: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Apply hierarchical polarization.

        Args:
            Z: Complex tensor (batch, seq_len, d_complex)
            mask: Optional mask (batch, seq_len) - currently ignored for simplicity

        Returns:
            Polarized tensor (batch, seq_len, d_complex)
        """
        batch, orig_seq_len, d = Z.shape

        # Pad to power of 2
        Z, _ = self._pad_to_power_of_2(Z)
        seq_len = Z.size(1)

        # Compute number of levels
        n_levels = int(math.log2(seq_len))
        if self.max_levels is not None:
            n_levels = min(n_levels, self.max_levels)

        # ========== Bottom-up pass ==========
        # At each level, aggregate groups of size 2^(level+1)
        for level in range(n_levels):
            group_size = 2 ** (level + 1)

            if group_size > seq_len:
                break

            # Aggregate within groups
            aggregates, Z_grouped = self._aggregate_groups(Z, group_size)

            # Transform aggregates
            # aggregates: (batch, n_groups, d) -> need (batch, n_groups, d) for PolarizingBlock
            # But PolarizingBlock expects (batch, n_tokens, d), treating n_groups as tokens
            transform = self._get_transform(level, "up")
            A_transformed = transform(aggregates)

            # Broadcast back to group members
            # A_transformed: (batch, n_groups, d) -> expand to (batch, n_groups, group_size, d)
            A_broadcast = A_transformed.unsqueeze(2).expand_as(Z_grouped)

            # Residual connection: add transformed aggregate to each group member
            Z_grouped = Z_grouped + A_broadcast

            # Reshape back
            Z = Z_grouped.view(batch, seq_len, d)

        # ========== Top-down pass (optional) ==========
        if self.top_down_mode != "none":
            # Reverse: start from largest groups, go to smallest
            for level in range(n_levels - 1, -1, -1):
                group_size = 2 ** (level + 1)

                if group_size > seq_len:
                    continue

                # Aggregate within groups (again, with updated Z)
                aggregates, Z_grouped = self._aggregate_groups(Z, group_size)

                # Transform with down transform
                transform = self._get_transform(level, "down")
                A_transformed = transform(aggregates)

                # Broadcast and add
                A_broadcast = A_transformed.unsqueeze(2).expand_as(Z_grouped)
                Z_grouped = Z_grouped + A_broadcast

                Z = Z_grouped.view(batch, seq_len, d)

        # Unpad to original length
        return Z[:, :orig_seq_len, :]

    def get_config(self) -> dict:
        """Return configuration for serialization."""
        return {
            "d_complex": self.d_complex,
            "max_levels": self.max_levels,
            "weight_sharing": self.weight_sharing,
            "aggregation": self.aggregation_type,
            "top_down": self.top_down_mode,
            "kan_hidden": self.kan_hidden,
            "dropout": self.dropout_rate,
        }
