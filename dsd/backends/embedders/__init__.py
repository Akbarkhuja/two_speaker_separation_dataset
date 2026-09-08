"""Speaker-embedder backends. Importing this package registers them by name."""

from . import titanet  # noqa: F401  (import registers)
from .base import Embedder

__all__ = ["Embedder"]
