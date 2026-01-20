# Models

from .base import BaseCVKAN, ComplexEmbedding, build_classifier_head
from .cv_kan import CVKAN, CVKANTokenClassifier
from .baseline_transformer import BaselineTransformer
from .cv_kan_image import CVKANImageClassifier
from .cv_kan_timeseries import CVKANTimeSeries
from .cv_kan_audio import CVKANAudio

__all__ = [
    # Base class
    "BaseCVKAN",
    "ComplexEmbedding",
    "build_classifier_head",
    # Sequence classification
    "CVKAN",
    "CVKANTokenClassifier",
    # Domain-specific models
    "CVKANImageClassifier",
    "CVKANTimeSeries",
    "CVKANAudio",
    # Baselines
    "BaselineTransformer",
]
