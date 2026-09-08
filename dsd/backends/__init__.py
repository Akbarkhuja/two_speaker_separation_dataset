"""Pluggable model backends, one subpackage per family.

Each subpackage's `__init__` imports its modules, and importing a module runs
its `@REGISTRY.register(...)` decorator. Nothing here loads weights -- factories
are lazy, so listing backends or printing `--help` never touches CUDA.
"""

from . import (  # noqa: F401  (import registers)
    diarizers,
    embedders,
    gender,
    speech_enhancement,
    vad,
)

__all__ = ["diarizers", "embedders", "gender", "speech_enhancement", "vad"]
