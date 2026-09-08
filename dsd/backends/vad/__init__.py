"""VAD backends. Importing this package registers them by name."""

from . import silero  # noqa: F401  (import registers)
from .base import VAD

__all__ = ["VAD"]
