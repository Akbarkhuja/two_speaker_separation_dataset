"""The speech-enhancement contract.

An enhancer takes one mono span at `sr` and returns it cleaned at
`output_rate(sr)`, covering **exactly the same duration**:
`len(out) == round(len(span) * output_rate(sr) / sr)`. That is not a stylistic
preference: the enhanced audio is spliced back into the channel at an exact
time offset, and every VAD mask downstream is applied sample-for-sample at the
target's rate. A backend that returned a sample more or less would silently
shift every subsequent label.

The rate is allowed to differ from the input's because a *generative* restorer
(Sidon) synthesises band that was never recorded, and asking it for the input
rate back throws that band away. A *masking* enhancer cannot create band, so
for one of those `output_rate(sr)` is `sr` and asking for more would only
interpolate. Backends without an `output_rate` method are taken to return the
input rate.

The contract is cheap to enforce and expensive to get wrong, so backends assert
it rather than trusting the service -- and pad or trim nothing.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class SpeechEnhancer(Protocol):
    def output_rate(self, sr: int) -> int:
        """The rate `enhance` returns for a span at `sr`."""

    def enhance(self, samples: np.ndarray, sr: int) -> np.ndarray:
        """Clean a mono span; float32 at `output_rate(sr)`, same duration as the input."""


def output_rate(enhancer, sr: int) -> int:
    """`enhancer.output_rate(sr)`, or `sr` for a backend that predates the method."""
    method = getattr(enhancer, "output_rate", None)
    return int(method(sr)) if method is not None else int(sr)


def expected_length(n_samples: int, sr: int, out_sr: int) -> int:
    """Samples a span of `n_samples` at `sr` must have at `out_sr`.

    `round`, because that is what the Sidon service pins its output to; for the
    integer ratios the pipeline accepts it is exact anyway.
    """
    return n_samples if out_sr == sr else int(round(n_samples * out_sr / sr))
