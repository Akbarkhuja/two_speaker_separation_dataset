"""The speech-enhancement contract.

An enhancer takes one mono span and returns it denoised, **at the same rate and
with the same number of samples**. That is not a stylistic preference: the
enhanced audio is spliced back into the channel at an exact sample offset, and
every VAD mask downstream is applied sample-for-sample. A backend that returns a
different length would silently shift every subsequent label.

The contract is cheap to enforce and expensive to get wrong, so backends assert
it rather than trusting the service.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class SpeechEnhancer(Protocol):
    def enhance(self, samples: np.ndarray, sr: int) -> np.ndarray:
        """Denoise a mono span; returns float32 of identical length at `sr`."""
