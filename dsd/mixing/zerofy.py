"""Silence everything a channel records outside its own speaker's speech.

This is applied to the **targets only**. `s1` and `s2` are what the model is
asked to produce, so they hold that speaker's speech and nothing else: line
noise, breath, keyboard, hold music and the few dB of crosstalk that leak
through are all faded out. The mixture is built from the channels as recorded
(see `dsd/stages/build.py`), so that material is still in the model's input --
which is the point. A model trained on a mixture that is silent between the
turns has never seen the noise floor it will meet in production, and learns to
treat "not silence" as "speech".

The two channels are physically isolated -- measured cross-channel |corr| over
158 calls is mean 0.0006, max 0.005 -- so a channel really is a single-speaker
recording, and masking it gives a true source signal for that speaker.

The fade is not cosmetic. See `dsd/core/audio.py:fade_envelope`.
"""

from __future__ import annotations

import numpy as np

from ..core.audio import fade_envelope, speech_mask


def apply_mask(
    samples: np.ndarray,
    mask: np.ndarray,
    sr: int,
    fade_ms: float = 10.0,
) -> np.ndarray:
    """Keep `samples` where `mask` is True, faded to zero elsewhere."""
    return np.asarray(samples, dtype=np.float32) * fade_envelope(mask, sr, fade_ms)


def mute(
    samples: np.ndarray,
    intervals,
    sr: int,
    fade_ms: float = 10.0,
) -> np.ndarray:
    """The complement of `zerofy`: silence *inside* the given intervals.

    Used for the spans the filter attributed to a minor diarizer label. Those
    have to leave the mixture as well as the targets -- an unlabelled third
    voice in the model's input with no target to match is worse than the brief
    gap that removing it leaves.
    """
    samples = np.asarray(samples, dtype=np.float32)
    if not intervals:
        return samples
    keep = ~speech_mask(intervals, len(samples), sr)
    return apply_mask(samples, keep, sr, fade_ms)


def zerofy(
    samples: np.ndarray,
    intervals,
    sr: int,
    fade_ms: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (masked_samples, boolean_speech_mask)."""
    mask = speech_mask(intervals, len(samples), sr)
    return apply_mask(samples, mask, sr, fade_ms), mask