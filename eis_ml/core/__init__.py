"""
Core functionality for retrospective analysis.

This package contains preprocessing, embeddings, and similarity calculations.
"""

from . import preprocessing
from . import embeddings
from . import similarity

__all__ = ["preprocessing", "embeddings", "similarity"]