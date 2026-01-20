# Polar CV-KAN: Complex-Valued Kolmogorov-Arnold Networks with Polar Coordinates
"""
A novel attention mechanism using complex-valued representations with polar
coordinate transformations. Tokens interact through phase alignment and
magnitude polarization.

Note: Traditional normalization is NOT used because magnitudes encode
attention-like information. Log-magnitude centering is used instead.
"""

from .models import CVKAN
from .modules import GatedPolarization, PolarizingBlock

__version__ = "0.1.0"
__all__ = ["PolarizingBlock", "GatedPolarization", "CVKAN"]
