"""Gender-classifier backends. Importing this package registers them by name."""

from . import http_ecapa  # noqa: F401  (import registers)
from .base import GenderClassifier

__all__ = ["GenderClassifier"]
