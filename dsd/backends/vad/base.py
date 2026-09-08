"""The VAD contract.

A VAD takes one mono channel and returns speech intervals in seconds. Stages
never pass it a stereo array -- the per-channel loop lives in the stage, because
"which channel" is meaningful to the pipeline and not to the model.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class VAD(Protocol):
    def speech(self, samples: np.ndarray, sr: int) -> list[tuple[float, float]]:
        """Return (start, end) second-pairs covering speech, sorted, disjoint."""
