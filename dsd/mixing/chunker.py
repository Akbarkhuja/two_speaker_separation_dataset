"""Cut a built call into fixed-length training windows.

A window is kept only when *both* sources carry at least `min_active` seconds of
speech in it. Without that rule most windows would be one person talking into
silence, where the correct output is "copy the mixture to one output, emit zeros
on the other" -- a shortcut that trains fast, scores well, and does not separate
anything.
"""

from __future__ import annotations

import numpy as np


def windows(n_samples: int, sr: int, chunk_sec: float, hop_sec: float):
    """Yield (start, end) sample indices for complete windows only.

    A trailing partial window is dropped rather than zero-padded: padding would
    put a block of digital silence into the training set that never occurs in
    the audio itself.
    """
    size = int(round(chunk_sec * sr))
    hop = max(1, int(round(hop_sec * sr)))
    if size <= 0 or n_samples < size:
        return
    for start in range(0, n_samples - size + 1, hop):
        yield start, start + size


def is_usable(
    mask1: np.ndarray,
    mask2: np.ndarray,
    start: int,
    end: int,
    sr: int,
    min_active: float,
) -> bool:
    need = min_active * sr
    return (
        np.count_nonzero(mask1[start:end]) >= need
        and np.count_nonzero(mask2[start:end]) >= need
    )
