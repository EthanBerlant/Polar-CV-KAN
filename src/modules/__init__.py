# Core modules for CV-KAN
#
# Note: ComplexLayerNorm and ComplexRMSNorm are intentionally NOT exported.
# CV-KAN uses magnitudes to encode attention-like information, so traditional
# normalization would destroy this signal. Use log-magnitude centering instead
# (built into the CVKAN model).

from .aggregation import (
    CausalAggregation,
    GlobalMeanAggregation,
    LocalWindowAggregation,
    MagnitudeWeightedAggregation,
    NeighborhoodAggregation,
)
from .gated_polarization import GatedPolarization
from .hierarchical import HierarchicalPolarization
from .multi_head import (
    EmergentHeadsPolarizing,
    FactoredHeadsPolarizing,
    PhaseOffsetPolarizing,
)
from .phase_attention import PhaseAttentionBlock
from .polarizing_block import PolarizingBlock
from .positional_encoding import (
    Complex2DPositionalEncoding,
    ComplexPositionalEncoding,
    Learnable2DComplexPositionalEncoding,
    LearnableComplexPositionalEncoding,
)

__all__ = [
    # Core blocks
    "PolarizingBlock",
    "GatedPolarization",
    "PhaseAttentionBlock",
    "HierarchicalPolarization",
    # Multi-head variants
    "EmergentHeadsPolarizing",
    "PhaseOffsetPolarizing",
    "FactoredHeadsPolarizing",
    # Aggregation strategies
    "GlobalMeanAggregation",
    "MagnitudeWeightedAggregation",
    "LocalWindowAggregation",
    "CausalAggregation",
    "NeighborhoodAggregation",
    # Positional encodings
    "ComplexPositionalEncoding",
    "Complex2DPositionalEncoding",
    "LearnableComplexPositionalEncoding",
    "Learnable2DComplexPositionalEncoding",
]
