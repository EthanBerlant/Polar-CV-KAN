# Core modules for CV-KAN

from .polarizing_block import PolarizingBlock
from .complex_norm import ComplexLayerNorm, ComplexRMSNorm
from .gated_polarization import GatedPolarization
from .phase_attention import PhaseAttentionBlock
from .multi_head import (
    EmergentHeadsPolarizing,
    PhaseOffsetPolarizing,
    FactoredHeadsPolarizing,
)

__all__ = [
    "PolarizingBlock",
    "ComplexLayerNorm",
    "ComplexRMSNorm",
    "GatedPolarization",
    "PhaseAttentionBlock",
    "EmergentHeadsPolarizing",
    "PhaseOffsetPolarizing",
    "FactoredHeadsPolarizing",
]
