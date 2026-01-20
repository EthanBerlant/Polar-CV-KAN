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
    NeighborhoodAggregation,
)
from .gated_polarization import GatedPolarization
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
    # Multi-head variants
    "EmergentHeadsPolarizing",
    "PhaseOffsetPolarizing",
    "FactoredHeadsPolarizing",
    # Aggregation strategies
    "GlobalMeanAggregation",
    "LocalWindowAggregation",
    "CausalAggregation",
    "NeighborhoodAggregation",
    # Positional encodings
    "ComplexPositionalEncoding",
    "Complex2DPositionalEncoding",
    "LearnableComplexPositionalEncoding",
    "Learnable2DComplexPositionalEncoding",
]
