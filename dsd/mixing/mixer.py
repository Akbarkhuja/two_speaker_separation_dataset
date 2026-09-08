"""Level the two sources to a target SIR and sum them into the mixture."""

from __future__ import annotations

import numpy as np

from ..core.audio import active_rms


def scale_to_sir(
    s1: np.ndarray,
    s2: np.ndarray,
    mask1: np.ndarray,
    mask2: np.ndarray,
    sir_db: float,
) -> tuple[np.ndarray, float]:
    """Scale s2 so that s1 sits `sir_db` decibels above it. Returns (s2, gain).

    Levels are measured over each source's *speech* only. A whole-signal RMS
    would read a mostly-silent side as quiet and then boost it far past its real
    loudness, because zerofying already set the rest of it to zero.
    """
    rms1 = active_rms(s1, mask1)
    rms2 = active_rms(s2, mask2)
    if rms1 <= 0.0 or rms2 <= 0.0:
        return s2, 1.0

    gain = float(rms1 / (rms2 * (10.0 ** (sir_db / 20.0))))
    return (s2 * gain).astype(np.float32), gain


def mix(
    s1: np.ndarray,
    s2: np.ndarray,
    ceiling: float = 0.99,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Sum the sources, guarding against clipping. Returns (mix, s1, s2, gain).

    When anything runs hot, *all three* signals are scaled by the same factor.
    Scaling only the mixture would break `mix == s1 + s2`, and that identity is
    the entire premise of the dataset -- every separation loss compares the
    model's outputs against these sources assuming they add up.

    The peak is taken over all three, not just the mixture. A source can exceed
    full scale while the sum does not: `scale_to_sir` can push s2 well above 1.0
    at a negative SIR, and wherever the two sources partially cancel the mixture
    still fits under the ceiling. The PCM_16 write then clips s2 alone and the
    identity is silently broken -- measured on a real build, 22 of 2513 calls,
    with residuals to 1.06e-01 against a 1e-4 tolerance, every one of them a
    negative-SIR call whose s2 was pinned at full scale.
    """
    mixture = (s1 + s2).astype(np.float32)
    peak = 0.0
    for signal in (mixture, s1, s2):
        if signal.size:
            peak = max(peak, float(np.abs(signal).max()))
    if peak <= ceiling or peak == 0.0:
        return mixture, s1, s2, 1.0

    gain = ceiling / peak
    return (
        (mixture * gain).astype(np.float32),
        (s1 * gain).astype(np.float32),
        (s2 * gain).astype(np.float32),
        gain,
    )
