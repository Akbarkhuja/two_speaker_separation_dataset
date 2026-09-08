"""The clustering contract.

A clusterer receives the keys, their pairwise similarity matrix, and the call
each key came from. The call ids are the cannot-link information: two sides of
one call are different people by construction, so no cluster may contain two
keys from the same call. Passing groups rather than a precomputed constraint
matrix lets a backend enforce that however it likes.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Clusterer(Protocol):
    def fit(self, keys: list[str], similarity: np.ndarray, groups: list[str]) -> list[list[str]]:
        """Return clusters as lists of keys. Every key appears exactly once."""
