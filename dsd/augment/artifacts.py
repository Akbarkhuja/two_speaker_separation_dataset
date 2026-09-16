"""Three degradations that need nothing but the signal itself.

Band limitation, clipping and packet loss, in the forms DialogueSidon's
Appendix A specifies -- with one adaptation, spelled out in `band_limit`.

Every function here is length-preserving and takes a float32 signal in, float32
out. That matters: `build` indexes the VAD masks against these arrays, and a
degradation that changed the length would shift every label in the call.
"""

from __future__ import annotations

import numpy as np


def band_limit(x: np.ndarray, sr: int, cutoff_hz: float) -> np.ndarray:
    """Low-pass at `cutoff_hz`, an 8th-order Butterworth applied zero-phase.

    The paper resamples to one of {8, 16, 22.05, 24, 44.1, 48} kHz and back.
    Every one of those is at or above this corpus's 8 kHz, so the round trip is
    an identity here and the step would do nothing at all -- measured, 99.9% of
    the energy in these calls is already below 3.8 kHz. Drawing a cutoff below
    Nyquist instead reproduces what the step is *for*: a narrower channel than
    the one the model was trained on.

    Zero-phase (`sosfiltfilt`) rather than a causal filter, so the step removes
    band without also shifting the signal against its labels.
    """
    from scipy.signal import sosfiltfilt, butter

    nyquist = sr / 2.0
    # A cutoff at or above Nyquist has nothing to remove; one at zero would
    # remove everything. Both are draws the config could legitimately produce.
    if not 0.0 < cutoff_hz < nyquist:
        return np.asarray(x, dtype=np.float32)

    sos = butter(8, cutoff_hz / nyquist, btype="low", output="sos")
    # filtfilt needs more samples than its padding length; a very short signal
    # is left alone rather than raising.
    if x.size <= 3 * (sos.shape[0] * 2 + 1):
        return np.asarray(x, dtype=np.float32)
    return np.ascontiguousarray(sosfiltfilt(sos, x), dtype=np.float32)


def clip(x: np.ndarray, low_pct: float, high_pct: float) -> np.ndarray:
    """Clamp to the amplitude at the given percentiles of this signal.

    Straight from the paper: the new minimum is the value at a percentile drawn
    from U(0, 10) and the new maximum the value at one drawn from U(90, 100).
    Percentiles of the signal rather than fixed thresholds, so the amount of
    clipping does not depend on how loud the call happened to be recorded.

    `low_pct=0, high_pct=100` is the identity, which is what makes the draw
    continuous down to "no clipping at all".
    """
    if x.size == 0:
        return np.asarray(x, dtype=np.float32)
    low = float(np.percentile(x, low_pct))
    high = float(np.percentile(x, high_pct))
    if low >= high:
        return np.asarray(x, dtype=np.float32)
    return np.clip(x, low, high).astype(np.float32)


def packet_loss(
    x: np.ndarray,
    sr: int,
    rng,
    frac: float = 0.09,
    segment_ms: tuple[float, float] = (20.0, 200.0),
) -> tuple[np.ndarray, int]:
    """Zero a random `frac` of consecutive segments. Returns (signal, n dropped).

    The signal is tiled into back-to-back segments whose individual lengths are
    drawn from U(*segment_ms), then `frac` of those segments are selected and
    replaced with zeros -- the paper's formulation exactly.

    The zeros are hard, with no taper. A tapered dropout would be gentler on the
    model but it is not what a lost packet sounds like: the discontinuity at each
    edge is the artifact, and softening it would train the model against a
    degradation that does not occur.
    """
    x = np.asarray(x, dtype=np.float32)
    n = x.size
    if n == 0 or frac <= 0.0:
        return x.copy(), 0

    low, high = float(segment_ms[0]), float(segment_ms[1])
    bounds: list[tuple[int, int]] = []
    start = 0
    while start < n:
        length = max(1, int(round(rng.uniform(low, high) * sr / 1000.0)))
        end = min(n, start + length)
        bounds.append((start, end))
        start = end

    n_drop = int(round(frac * len(bounds)))
    if n_drop <= 0:
        return x.copy(), 0

    out = x.copy()
    for index in rng.sample(range(len(bounds)), min(n_drop, len(bounds))):
        a, b = bounds[index]
        out[a:b] = 0.0
    return out, n_drop
