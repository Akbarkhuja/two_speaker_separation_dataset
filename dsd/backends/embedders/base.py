"""The speaker-embedder contract.

`sample_rate` is the rate the model actually wants. Stages do not resample --
the embedder does, because whether resampling is needed is a property of the
model, and getting it wrong is silent (see `titanet.py`).

`embed_batch` returns L2-normalized rows, so a stack of embeddings has the
useful property that `E @ E.T` is exactly the cosine similarity matrix.
"""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np


class Embedder(Protocol):
    sample_rate: int
    dim: int

    def embed_batch(self, segments: Sequence[np.ndarray], sr: int) -> np.ndarray:
        """Embed mono segments recorded at `sr`; returns (n, dim), L2-normalized."""
