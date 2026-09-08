"""Diarizer backends.

Importing this package registers every backend by name. Adding one is a new
module here plus a line below -- no stage code changes.
"""

from . import from_dir, moss_http, sortformer  # noqa: F401  (import registers)
from .base import Diarizer, per_channel

__all__ = ["Diarizer", "per_channel"]
