import torch
import torch.nn as nn

from ..configs.model import CVKANConfig
from ..modules.aggregation import (
    CausalAggregation,
    GlobalMeanAggregation,
    LocalWindowAggregation,
    NeighborhoodAggregation,
)
from ..modules.multi_head import (
    EmergentHeadsPolarizing,
    FactoredHeadsPolarizing,
    PhaseOffsetPolarizing,
)
from ..modules.phase_attention import PhaseAttentionBlock
from ..modules.polarizing_block import PolarizingBlock


class CVKANBackbone(nn.Module):
    """
    Standard CV-KAN backbone consisting of a stack of PolarizingBlocks (or variants).
    Responsible for:
    - Stacking layers
    - Applying layers sequentially
    - Memory management (centering magnitudes)
    """

    def __init__(self, config: CVKANConfig):
        super().__init__()
        self.config = config
        self.layers = self._build_layers()

    def _get_aggregation(self):
        """Get aggregation strategy and per_dim setting."""
        t = getattr(self.config, "aggregation_type", "mean")

        if t == "mean":
            return GlobalMeanAggregation(), True
        elif t == "causal":
            return CausalAggregation(), False
        elif t == "window":
            return LocalWindowAggregation(window_size=3), True  # Default window
        elif t == "neighborhood":
            return NeighborhoodAggregation(), True
        else:
            return GlobalMeanAggregation(), True

    def _build_layers(self) -> nn.ModuleList:
        layers = nn.ModuleList()
        aggregation, per_dim = self._get_aggregation()

        for _ in range(self.config.n_layers):
            if self.config.block_type == "polarizing":
                if self.config.head_approach == "emergent":
                    layer = EmergentHeadsPolarizing(
                        self.config.d_complex,
                        self.config.kan_hidden,
                        aggregation=aggregation,
                        per_dim=per_dim,
                        dropout=self.config.dropout,
                    )
                elif self.config.head_approach == "offset":
                    d_per_head = self.config.d_complex // self.config.n_heads
                    layer = PhaseOffsetPolarizing(
                        self.config.n_heads,
                        d_per_head,
                        self.config.kan_hidden,
                        aggregation=aggregation,
                        per_dim=per_dim,
                        dropout=self.config.dropout,
                    )
                elif self.config.head_approach == "factored":
                    d_per_head = self.config.d_complex // self.config.n_heads
                    # Factored heads has custom internal aggregation currently
                    layer = FactoredHeadsPolarizing(
                        self.config.n_heads,
                        self.config.d_complex,
                        d_per_head,
                        self.config.kan_hidden,
                    )
                else:
                    # Default/Fallback
                    layer = PolarizingBlock(
                        self.config.d_complex,
                        kan_hidden=self.config.kan_hidden,
                        aggregation=aggregation,
                        per_dim=per_dim,
                        dropout=self.config.dropout,
                    )
            elif self.config.block_type == "attention":
                layer = PhaseAttentionBlock(self.config.d_complex, n_heads=self.config.n_heads)
            else:
                raise ValueError(f"Unknown block type: {self.config.block_type}")

            layers.append(layer)

        return layers

    def _center_log_magnitudes(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Center log-magnitudes across tokens to prevent drift.
        Preserves relative magnitudes (attention) but keeps absolute scale bounded.
        """
        mag = torch.abs(Z)
        log_mag = torch.log(mag + 1e-6)

        # Mean across tokens: (batch, 1, d_complex)
        mean_log_mag = log_mag.mean(dim=1, keepdim=True)

        # Center
        centered_log_mag = log_mag - mean_log_mag

        # Reconstruct
        # Phase is unchanged
        phase = torch.angle(Z)
        return torch.exp(centered_log_mag) * torch.exp(1j * phase)

    def forward(self, z: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        for layer in self.layers:
            z = layer(z, mask=mask)

            if self.config.center_magnitudes:
                z = self._center_log_magnitudes(z)

        return z
