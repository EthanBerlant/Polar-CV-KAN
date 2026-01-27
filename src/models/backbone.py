import torch
from torch import nn

from src.configs.model import CVKANConfig
from src.modules.hierarchical import HierarchicalPolarization
from src.modules.polarizing_block import PolarizingBlock
from src.registry import AGGREGATION_REGISTRY, NORMALIZATION_REGISTRY


class Residual(nn.Module):
    """Wrapper for residual connections."""

    def __init__(self, fn: nn.Module) -> None:
        """Initialize Residual."""
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs: dict) -> torch.Tensor:
        """Forward pass."""
        return x + self.fn(x, **kwargs)


class NormWrapper(nn.Module):
    """Wrapper for complex-aware normalization to handle dimension ordering."""

    def __init__(self, dim: int, normalization: str = "batch") -> None:
        """Initialize NormWrapper."""
        super().__init__()
        try:
            norm_cls = NORMALIZATION_REGISTRY.get(normalization)
            self.norm = norm_cls(dim)
        except KeyError as err:
            if normalization == "none":
                self.norm = nn.Identity()
            else:
                raise ValueError(
                    f"Normalization type '{normalization}' not found in registry."
                ) from err

        self.normalization_type = normalization

    def forward(self, z: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass."""
        if isinstance(self.norm, nn.Identity):
            return z

        # Check if we need to transpose for Batch Norm (expects B, C, H, W or B, D, N)
        # Our z is (B, N, D)

        if self.normalization_type == "batch":
            # BN expects (B, C, N) or (B, C, H, W)
            # z is (B, N, D) -> (B, D, N) -> Norm -> (B, N, D)
            z_perm = z.transpose(1, 2)
            # Handle 3D tensor vs 4D tensor if Norm expects 4D (ComplexBatchNorm2d usually expects 4D)
            # If ComplexBatchNorm2d is strictly 2D spatial, we need (B, D, N, 1) or similar.
            # Looking at normalization.py, it treated input as (N, C, H, W) and did mean([0, 2, 3]).
            # So it expects 4D.
            z_perm = z_perm.unsqueeze(-1)  # (B, D, N, 1)
            z_norm = self.norm(z_perm)
            return z_norm.squeeze(-1).transpose(1, 2)

        return self.norm(z)


class CVKANBackbone(nn.Module):
    """Standard CV-KAN backbone consisting of a stack of PolarizingBlocks (or variants)."""

    def __init__(self, config: CVKANConfig) -> None:
        """Initialize CVKANBackbone."""
        super().__init__()
        self.config = config
        self.layers = self._build_layers()

    def _get_aggregation(self) -> tuple[nn.Module, bool]:
        """Get aggregation strategy and per_dim setting."""
        # Use aggregation from config, default to mean
        agg_name = getattr(self.config, "aggregation", "mean")
        # Legacy support: config checks 'aggregation_type' sometimes
        if hasattr(self.config, "aggregation_type") and self.config.aggregation_type:
            agg_name = self.config.aggregation_type

        try:
            agg_cls = AGGREGATION_REGISTRY.get(agg_name)
            return agg_cls(), True
        except KeyError:
            # Fallback to mean if not found? Or raise?
            print(f"Warning: Aggregation '{agg_name}' not found, defaulting to mean.")
            from src.modules.aggregation import GlobalMeanAggregation

            return GlobalMeanAggregation(), True

    def _build_layers(self) -> nn.ModuleList:
        layers = nn.ModuleList()
        aggregation, per_dim = self._get_aggregation()

        # Determine normalization strategy name
        norm_name = getattr(self.config, "normalization", "none")

        for _ in range(self.config.n_layers):
            if self.config.block_type == "hierarchical":
                layer = HierarchicalPolarization(
                    d_complex=self.config.d_complex,
                    max_levels=self.config.hierarchical_levels,
                    weight_sharing=self.config.hierarchical_sharing,
                    aggregation=getattr(
                        self.config, "aggregation", "mean"
                    ),  # Pass string or object? Hierarchical expects string usually?
                    # Keep legacy interface for Hierarchical for now or refactor it too.
                    # Looking at hierarchical.py (not read yet), assuming it takes string.
                    top_down=self.config.hierarchical_top_down,
                    kan_hidden=self.config.kan_hidden,
                    dropout=self.config.dropout,
                    atomic=True,
                    hybrid_split_idx=getattr(self.config, "hierarchical_split_idx", 2),
                    phase_shifting=getattr(self.config, "hierarchical_phase_shifting", False),
                    interaction=getattr(self.config, "hierarchical_interaction", "broadcast"),
                )
            elif self.config.block_type == "polarizing":
                layer = PolarizingBlock(
                    self.config.d_complex,
                    kan_hidden=self.config.kan_hidden,
                    aggregation=aggregation,
                    per_dim=per_dim,
                    dropout=self.config.dropout,
                    interaction=getattr(self.config, "hierarchical_interaction", "broadcast"),
                    mag_init_scale=getattr(self.config, "mag_init_scale", 0.1),
                )
            else:
                raise ValueError(f"Unknown block type: {self.config.block_type}")

            # 1. Apply Transformation (with optional residual)
            if getattr(self.config, "skip_connections", False):
                layers.append(Residual(layer))
            else:
                layers.append(layer)

            # 2. Apply Normalization
            if norm_name != "none":
                layers.append(NormWrapper(self.config.d_complex, norm_name))

        return layers

    def _center_log_magnitudes(self, Z: torch.Tensor) -> torch.Tensor:
        """Center log-magnitudes across tokens to prevent drift."""
        mag = torch.abs(Z)
        log_mag = torch.log(mag + 1e-6)
        mean_log_mag = log_mag.mean(dim=1, keepdim=True)
        centered_log_mag = log_mag - mean_log_mag
        phase = torch.angle(Z)
        return torch.exp(centered_log_mag) * torch.exp(1j * phase)

    def forward(self, z: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass of the backbone."""
        for layer in self.layers:
            z = layer(z, mask=mask)

            if getattr(self.config, "center_magnitudes", True):
                z = self._center_log_magnitudes(z)

        return z
