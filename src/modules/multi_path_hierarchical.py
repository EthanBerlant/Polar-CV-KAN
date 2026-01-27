"""Multi-Path Hierarchical Polarization.

Runs multiple HierarchicalPolarization blocks in parallel with different
grouping offsets (phases) to avoid local optima in the aggregation process.

In a single hierarchy, if tokens (i, i+1) are grouped, their relationship is
fixed early. If the "true" relationship is between (i-1, i), the single
hierarchy might miss it.

Multi-path runs:
- Path 0: Standard grouping [0,1], [2,3]...
- Path 1: Offset grouping [1,2], [3,4]... (via cyclic shift)
- ...

The results are then aggregated (mean or parameterized mixture).
"""

import torch
from torch import nn

from .hierarchical import HierarchicalPolarization
from .polarizing_block import PolarizingBlock


class MultiPathHierarchicalPolarization(nn.Module):
    """Parallel hierarchical blocks with offset diversity.

    Args:
        d_complex: Complex dimension
        n_paths: Number of parallel paths (default: 2)
        combine_method: How to combine paths:
            - 'mean': Simple average (parameter free)
            - 'learned': Learned weighted sum
            - 'kan': PolarizingBlock transforms the aggregate (Final Boss)
        share_horizontal_weights: If True, all paths use the same HierarchicalPolarization instance
        share_vertical_weights: If True, all levels in hierarchy use same weights ('shared' mode).
                               If False, use independent weights per level ('per_level' mode).
        combine_kan_hidden: Hidden size for the combiner KAN (if combine_method='kan')
        **kwargs: Arguments passed to underlying HierarchicalPolarization
                 (kan_hidden, dropout, weight_sharing, etc.)
    """

    def __init__(
        self,
        d_complex: int,
        n_paths: int = 2,
        combine_method: str = "mean",
        share_horizontal_weights: bool = False,
        share_vertical_weights: bool = False,
        combine_kan_hidden: int = 32,
        **kwargs,
    ):
        super().__init__()
        self.d_complex = d_complex
        self.n_paths = n_paths
        self.combine_method = combine_method
        self.share_horizontal_weights = share_horizontal_weights
        self.share_vertical_weights = share_vertical_weights

        # Map boolean vertical sharing to HierarchicalPolarization string config
        # If user explicitly passed weight_sharing in kwargs, that takes precedence (advanced use)
        if "weight_sharing" not in kwargs:
            kwargs["weight_sharing"] = "shared" if share_vertical_weights else "per_level"

        # Create parallel paths
        if share_horizontal_weights:
            # Shared: Single instance used for all paths
            self.paths = HierarchicalPolarization(d_complex, **kwargs)
        else:
            # Independent: List of instances
            self.paths = nn.ModuleList(
                [HierarchicalPolarization(d_complex, **kwargs) for _ in range(n_paths)]
            )

        if combine_method == "learned":
            self.path_weights = nn.Parameter(torch.ones(n_paths) / n_paths)
        elif combine_method == "kan":
            # "Final Boss" KAN to process the aggregate
            self.combiner_block = PolarizingBlock(d_complex, kan_hidden=combine_kan_hidden)

    def forward(self, Z: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Apply multi-path polarization.

        Args:
            Z: (batch, seq_len, d_complex)
            mask: (batch, seq_len)

        Returns:
            (batch, seq_len, d_complex)
        """
        results = []

        for i in range(self.n_paths):
            # Calculate shift: Path i shifts by i positions
            shift = i

            if shift == 0:
                Z_input = Z
            else:
                Z_input = torch.roll(Z, shifts=-shift, dims=1)

            # Apply hierarchical block
            if self.share_horizontal_weights:
                Z_out = self.paths(Z_input, mask)
            else:
                Z_out = self.paths[i](Z_input, mask)

            # Shift back
            if shift != 0:
                Z_out = torch.roll(Z_out, shifts=shift, dims=1)

            results.append(Z_out)

        # Stack: (batch, seq_len, d_complex, n_paths)
        Z_stack = torch.stack(results, dim=-1)

        # Combine
        if self.combine_method == "mean":
            return Z_stack.mean(dim=-1)

        if self.combine_method == "learned":
            weights = torch.softmax(self.path_weights, dim=0)
            return (Z_stack * weights).sum(dim=-1)

        if self.combine_method == "kan":
            # Initial aggregate (mean)
            Z_agg = Z_stack.mean(dim=-1)
            # Refine with KAN ("Final Boss")
            return self.combiner_block(Z_agg, mask)

        raise ValueError(f"Unknown combine_method: {self.combine_method}")

    def get_config(self) -> dict:
        """Return config."""
        base_config = (
            self.paths.get_config() if self.share_horizontal_weights else self.paths[0].get_config()
        )
        return {
            "n_paths": self.n_paths,
            "combine_method": self.combine_method,
            "share_horizontal_weights": self.share_horizontal_weights,
            "share_vertical_weights": self.share_vertical_weights,
            **base_config,
        }
