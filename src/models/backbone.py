import torch
from torch import nn

from ..configs.model import CVKANConfig
from ..modules.aggregation import GlobalMeanAggregation, MagnitudeWeightedAggregation
from ..modules.hierarchical import HierarchicalPolarization
from ..modules.polarizing_block import PolarizingBlock


class Residual(nn.Module):
    """Wrapper for residual connections."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return x + self.fn(x, **kwargs)


class NormWrapper(nn.Module):
    """Wrapper for complex-aware normalization."""

    def __init__(self, dim, normalization="batch"):
        super().__init__()
        if normalization == "batch":
            from .normalization import ComplexBatchNorm2d

            self.norm = ComplexBatchNorm2d(dim)
        elif normalization == "layer":
            from .normalization import ComplexLayerNorm

            self.norm = ComplexLayerNorm(dim)
        else:
            self.norm = nn.Identity()

    def forward(self, z, mask=None):
        if isinstance(self.norm, nn.Identity):
            return z
        if hasattr(self.norm, "forward") and not isinstance(self.norm, nn.Identity):
            # ComplexLayerNorm works on (B, N, D)
            if hasattr(self.norm, "gamma") and not hasattr(self.norm, "gamma_rr"):
                return self.norm(z)
            # ComplexBatchNorm2d works on (B, D, N, 1)
            z_perm = z.transpose(1, 2).unsqueeze(-1)
            z_norm = self.norm(z_perm)
            return z_norm.squeeze(-1).transpose(1, 2)
        return z


class CVKANBackbone(nn.Module):
    """Standard CV-KAN backbone consisting of a stack of PolarizingBlocks (or variants)."""

    def __init__(self, config: CVKANConfig):
        super().__init__()
        self.config = config
        self.layers = self._build_layers()

    def _get_aggregation(self):
        """Get aggregation strategy and per_dim setting."""
        t = getattr(self.config, "aggregation_type", "mean")

        if t == "mean":
            return GlobalMeanAggregation(), True
        if t == "magnitude_weighted":
            return MagnitudeWeightedAggregation(), True
        # ... other types if needed ...
        return GlobalMeanAggregation(), True

    def _build_layers(self) -> nn.ModuleList:
        layers = nn.ModuleList()
        aggregation, per_dim = self._get_aggregation()

        for _ in range(self.config.n_layers):
            if self.config.block_type == "hierarchical":
                layer = HierarchicalPolarization(
                    d_complex=self.config.d_complex,
                    max_levels=self.config.hierarchical_levels,
                    weight_sharing=self.config.hierarchical_sharing,
                    aggregation=self.config.aggregation_type,
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
            if self.config.normalization != "none":
                layers.append(NormWrapper(self.config.d_complex, self.config.normalization))

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
        for layer in self.layers:
            z = layer(z, mask=mask)

            if self.config.center_magnitudes:
                z = self._center_log_magnitudes(z)

        return z
