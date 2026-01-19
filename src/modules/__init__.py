# Core modules for CV-KAN
#
# Note: ComplexLayerNorm and ComplexRMSNorm are intentionally NOT exported.
# CV-KAN uses magnitudes to encode attention-like information, so traditional
# normalization would destroy this signal. Use log-magnitude centering instead
# (built into the CVKAN model).

from .polarizing_block import PolarizingBlock
from .gated_polarization import GatedPolarization
from .phase_attention import PhaseAttentionBlock
from .multi_head import (
    EmergentHeadsPolarizing,
    PhaseOffsetPolarizing,
    FactoredHeadsPolarizing,
)

__all__ = [
    "PolarizingBlock",
    "GatedPolarization",
    "PhaseAttentionBlock",
    "EmergentHeadsPolarizing",
    "PhaseOffsetPolarizing",
    "FactoredHeadsPolarizing",
]

