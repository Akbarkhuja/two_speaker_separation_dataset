"""Measuring and synthesizing speech overlap between the two sources.

Natural overlap in this corpus is tiny -- measured with Silero VAD over 25
calls: mean 3.2%, p90 7.0%, max 19%. That is honest telephony (people take
turns), but a separation model trained only on it barely has to separate
anything; most of the time one source is silent and copying the mixture is
already the right answer.

Because the channels are isolated recordings, the fix is free and exact: shift
one source in time and the mixture is still perfectly labelled. A circular roll
is used so length and total speech are preserved -- nothing is truncated, and
the speech that wraps past the end simply lands in silence at the start.

`shift_for_target` finds the shift by cross-correlating the two speech masks,
which gives the overlap for *every* possible shift in one FFT instead of
sampling shifts and hoping. Masks are reduced to 100 Hz frames first: 10 ms is
far finer than the overlap ratio needs, and it keeps the transform small even
for a 30-minute call.
"""

from __future__ import annotations

import numpy as np

FRAME_RATE = 100  # frames per second for mask arithmetic


def to_frames(mask: np.ndarray, sr: int) -> np.ndarray:
    """Downsample a sample-rate boolean mask to a 100 Hz boolean mask."""
    step = max(1, int(round(sr / FRAME_RATE)))
    n_frames = int(np.ceil(len(mask) / step))
    padded = np.zeros(n_frames * step, dtype=bool)
    padded[: len(mask)] = mask
    # A frame counts as speech if any sample in it is speech.
    return padded.reshape(n_frames, step).any(axis=1)


def overlap_ratio(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over union of two boolean masks; 0.0 when both are silent."""
    union = int(np.count_nonzero(a | b))
    if union == 0:
        return 0.0
    return float(np.count_nonzero(a & b)) / union


def _dilate(frames: np.ndarray, guard: int) -> np.ndarray:
    """Widen a frame mask by `guard` frames each way."""
    if guard <= 0:
        return frames
    widened = frames.copy()
    for step in range(1, guard + 1):
        widened |= np.roll(frames, step) | np.roll(frames, -step)
    return widened


def safe_shifts(frames_b: np.ndarray, guard: int) -> np.ndarray:
    """Which shifts wrap source B without cutting an utterance in half.

    `np.roll` is circular, and that creates two new adjacencies in B:

      - the **split seam**: the original is cut at frame `n - k`, everything
        before it landing at the end of the file and everything after it at the
        start. An utterance spanning that point is chopped mid-word and its two
        halves teleported to opposite ends -- which is what "the chunks are not
        in their original place" sounds like.
      - the **boundary seam**: original frame `n - 1` is joined to frame `0`.
        Harmless when the recording starts and ends in silence, a click when it
        does not, and independent of `k` -- a property of the call, not of the
        shift.

    Both must land in silence. Measured over 983 shifted calls, leaving this
    unconstrained put the split seam inside an utterance 22.4% of the time; the
    B channel is 71.4% silent on average, so a safe offset is nearly always
    available and constraining costs almost no overlap coverage.
    """
    n = len(frames_b)
    if n == 0:
        return np.zeros(0, dtype=bool)

    widened = _dilate(frames_b, guard)
    if widened[0] or widened[-1]:
        # No value of k moves the boundary seam, so the whole call is unshiftable.
        safe = np.zeros(n, dtype=bool)
        safe[0] = True  # not shifting at all is always safe
        return safe

    # For shift k, the original frame that lands at output index 0.
    split = (-np.arange(n)) % n
    safe = ~widened[split]
    safe[0] = True
    return safe


def shift_for_target(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    sr: int,
    target: tuple[float, float],
    rng,
    guard_sec: float = 0.2,
) -> tuple[int, float, str]:
    """Pick a circular shift of source B landing the overlap ratio inside `target`.

    Returns (shift_in_samples, achieved_ratio, reason). Only shifts whose wrap
    seams fall in silence are considered -- see `safe_shifts`. If no shift can
    reach the band -- which happens when one side barely speaks, so even total
    alignment leaves the ratio low -- the safe shift that gets closest is
    returned instead, and the caller can see the achieved value in the metadata.

    `reason` records why the answer is what it is, so a call that came back
    unshifted can be explained later without re-deriving it.
    """
    frames_a = to_frames(mask_a, sr)
    frames_b = to_frames(mask_b, sr)
    n = len(frames_a)
    count_a = int(frames_a.sum())
    count_b = int(frames_b.sum())
    if n == 0 or count_a == 0 or count_b == 0:
        return 0, 0.0, "silent_source"

    # Circular cross-correlation. intersections[k] must count the frames where
    # A and `np.roll(B, k)` are both speech, i.e. sum_i a[i] * b[i - k], which
    # is irfft(rfft(a) * conj(rfft(b))) -- conjugating the *product* instead
    # would give sum_i a[i] * b[i + k] and roll the source the wrong way, a
    # mistake that stays invisible because the ratios still look plausible.
    spectrum = np.fft.rfft(frames_a.astype(np.float64)) * np.conj(
        np.fft.rfft(frames_b.astype(np.float64))
    )
    intersections = np.fft.irfft(spectrum, n=n)
    intersections = np.clip(np.rint(intersections), 0, min(count_a, count_b))

    # |A or B| = |A| + |B| - |A and B|, exactly, for every shift at once.
    unions = count_a + count_b - intersections
    ratios = np.divide(
        intersections, unions, out=np.zeros_like(intersections), where=unions > 0
    )

    # Sample the ratio first, then find a shift that achieves it -- rather than
    # sampling uniformly among the shifts that land in the band. Those are not
    # the same thing: far more shifts produce low overlap than high, so picking
    # a shift uniformly piles the results against the bottom edge of the band
    # (measured: 0.152-0.180 for a band of 0.15-0.60). Choosing the target
    # first spreads them across the band the way the config implies.
    low, high = target
    want = rng.uniform(low, high)
    distances = np.abs(ratios - want)

    # Restrict the search to shifts that do not chop an utterance, rather than
    # picking first and checking after: the nearest achievable ratio among the
    # safe shifts is what we want, not the nearest overall.
    guard = int(round(guard_sec * FRAME_RATE))
    allowed = np.flatnonzero(safe_shifts(frames_b, guard))
    step = max(1, int(round(sr / FRAME_RATE)))

    reason = "ok"
    if allowed.size <= 1:
        # Only shift 0 survived. Either the call starts or ends mid-utterance,
        # or B speaks nearly throughout and every wrap point is inside speech.
        widened = _dilate(frames_b, guard)
        reason = "boundary_not_silent" if (widened[0] or widened[-1]) else "no_safe_shift"
        return 0, float(ratios[0]), reason

    # Break ties at random. The near-ties are usually many, and taking argmin
    # would always pick the smallest shift among them.
    allowed_distances = distances[allowed]
    best = float(allowed_distances.min())
    candidates = allowed[np.flatnonzero(allowed_distances <= best + 1e-9)]
    frame_shift = int(rng.choice(candidates.tolist()))

    return frame_shift * step, float(ratios[frame_shift]), reason


def roll(x: np.ndarray, shift: int) -> np.ndarray:
    return x if shift == 0 else np.roll(x, shift)
