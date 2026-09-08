"""Silence everything a channel records outside its own speaker's speech.

The two channels are already physically isolated -- measured cross-channel
|corr| over 158 calls is mean 0.0006, max 0.005 -- so a channel is very nearly a
clean single-speaker recording already. Zerofying removes what is left: line
noise, breath, keyboard, hold music, and the handful of dB of crosstalk that do
leak through. What remains is a true source signal, and `mix = s1 + s2` is an
exact separation target rather than an approximation.

The fade is not cosmetic. See `dsd/core/audio.py:fade_envelope`.
"""

from __future__ import annotations

import numpy as np

from ..core.audio import fade_envelope, speech_mask


def zerofy(
    samples: np.ndarray,
    intervals,
    sr: int,
    fade_ms: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (masked_samples, boolean_speech_mask)."""
    mask = speech_mask(intervals, len(samples), sr)
    envelope = fade_envelope(mask, sr, fade_ms)
    return np.asarray(samples, dtype=np.float32) * envelope, mask
