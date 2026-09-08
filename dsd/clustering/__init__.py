"""Clustering backends. Importing this package registers them by name."""

from . import constrained_ahc  # noqa: F401  (import registers)
from .base import Clusterer

__all__ = ["Clusterer"]
