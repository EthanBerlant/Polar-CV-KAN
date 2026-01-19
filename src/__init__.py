# Polar CV-KAN: Complex-Valued Kolmogorov-Arnold Networks with Polar Coordinates
"""
A novel attention mechanism using complex-valued representations with polar
coordinate transformations. Tokens interact through phase alignment and
magnitude polarization.
"""

from .modules import PolarizingBlock, ComplexLayerNorm, GatedPolarization
from .models import CVKAN

__version__ = "0.1.0"
__all__ = ["PolarizingBlock", "ComplexLayerNorm", "GatedPolarization", "CVKAN"]
