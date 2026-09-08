"""The gender-classifier contract.

Used only for balancing the dataset, so what matters is that the label is
stable for one voice across calls -- not that any individual call is perfect.
Backends return `(label, confidence)` with the label lowercased to
"male"/"female", and the stage aggregates several segments per speaker.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class GenderClassifier(Protocol):
    def predict(self, samples: np.ndarray, sr: int) -> tuple[str, float]:
        """Classify one mono segment; returns (label, confidence in [0, 1])."""
