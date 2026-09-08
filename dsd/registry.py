"""Name -> factory registries for the pluggable halves of the pipeline.

Every stage that touches a model does it through one of these. Adding a new
diarizer, VAD, embedder, gender classifier or clustering algorithm is therefore
a new file plus a decorator: no stage code changes, and the only place that has
to learn the new name is its family's `__init__.py`.

Factories take the backend's own options dict (`cfg.<stage>.options[<name>]`
from the YAML) and return the object. They are called lazily, at stage run time
-- registering a backend must never import torch or load weights, or `--help`
would spend a minute on CUDA init.
"""

from __future__ import annotations

from typing import Any, Callable


class Registry:
    """A tiny name -> factory map with a decorator interface."""

    def __init__(self, kind: str):
        self.kind = kind
        self._factories: dict[str, Callable[..., Any]] = {}

    def register(self, name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(factory: Callable[..., Any]) -> Callable[..., Any]:
            if name in self._factories:
                raise ValueError(f"{self.kind} backend {name!r} is already registered")
            self._factories[name] = factory
            return factory

        return decorator

    def create(self, name: str, options: dict | None = None, **kwargs: Any) -> Any:
        if name not in self._factories:
            raise KeyError(
                f"unknown {self.kind} backend {name!r}; available: {', '.join(self.names())}"
            )
        return self._factories[name](dict(options or {}), **kwargs)

    def names(self) -> list[str]:
        return sorted(self._factories)

    def __contains__(self, name: object) -> bool:
        return name in self._factories


DIARIZERS = Registry("diarizer")
VADS = Registry("vad")
EMBEDDERS = Registry("embedder")
GENDER = Registry("gender")
CLUSTERERS = Registry("clusterer")
ENHANCERS = Registry("enhancer")
