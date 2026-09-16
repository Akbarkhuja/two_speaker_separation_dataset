"""Background noise: one recording, looped to length, added at a target SNR.

The paper draws from AudioSet, the Free Music Archive, WHAM!, FSD50K and a wind
generator. Only WHAM! is on this machine -- 28,000 clips, 81.7 h of real recorded
ambience at 16 kHz -- and it is the one of the five whose recordings are actual
background rather than foreground sound, so it is what the pool is built from.
The `augment` stage screens it for clips that contain intelligible speech, which
would otherwise put an unlabelled third voice into the mixture.
"""

from __future__ import annotations

import numpy as np
import soundfile as sf

from ..core.audio import active_rms, resample


class NoiseBank:
    """Reads, resamples and caches noise clips; builds a bed of any length."""

    def __init__(self, files, sr: int, crossfade_sec: float = 0.25, cache_size: int = 48):
        self.files = list(files)
        self.sr = int(sr)
        self.crossfade_sec = float(crossfade_sec)
        self.cache_size = int(cache_size)
        self._cache: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.files)

    def load(self, path: str) -> np.ndarray:
        """Mono, at our rate. WHAM! is 16 kHz stereo, so both conversions apply."""
        if path not in self._cache:
            data, file_sr = sf.read(path, dtype="float32", always_2d=True)
            # Channel 0 rather than the mean of the two: WHAM!'s pair is a
            # spaced microphone array, and averaging them comb-filters the result.
            mono = resample(data[:, 0], file_sr, self.sr)
            if len(self._cache) >= self.cache_size:
                self._cache.pop(next(iter(self._cache)))
            self._cache[path] = mono
        return self._cache[path]

    def build(self, n_samples: int, rng) -> tuple[np.ndarray, str]:
        """One clip, looped past `n_samples` and truncated. Returns (bed, path).

        The paper's wording exactly -- "sampled a noise recording, looped it to
        exceed the duration, and truncated it". The one addition is an
        equal-power crossfade at each loop seam: a hard splice between the end
        and the start of a clip is a click at a perfectly periodic offset, which
        a model will find long before it finds the voices.
        """
        if not self.files or n_samples <= 0:
            return np.zeros(max(n_samples, 0), dtype=np.float32), ""

        path = rng.choice(self.files)
        clip = self.load(path)
        if clip.size == 0:
            return np.zeros(n_samples, dtype=np.float32), path
        if clip.size >= n_samples:
            # Long enough already: take a random excerpt rather than always the head.
            start = rng.randrange(0, clip.size - n_samples + 1)
            return np.array(clip[start : start + n_samples], dtype=np.float32), path

        fade = min(int(self.crossfade_sec * self.sr), clip.size // 2)
        out = np.zeros(n_samples, dtype=np.float32)
        position = 0
        while position < n_samples:
            piece = clip if position == 0 or fade <= 0 else clip[fade:]
            end = min(n_samples, position + piece.size)
            out[position:end] = piece[: end - position]
            if end >= n_samples:
                break
            # Overlap the next repeat onto the tail we just wrote.
            seam = min(fade, n_samples - end)
            if seam > 0:
                ramp = np.linspace(0.0, 1.0, seam, dtype=np.float32)
                out[end - seam : end] = (
                    out[end - seam : end] * np.sqrt(1.0 - ramp)
                    + clip[:seam] * np.sqrt(ramp)
                )
            position = end
        return out, path


def add_noise(
    x: np.ndarray,
    bed: np.ndarray,
    mask: np.ndarray | None,
    snr_db: float,
) -> np.ndarray:
    """Add `bed` to `x` so the speech in `x` sits `snr_db` above it.

    The speech level is measured over `mask` -- this channel's VAD speech -- not
    over the whole signal. A telephone leg is mostly silence, and a whole-signal
    RMS would read a quiet side as quieter than it is and then under-add noise
    by however much silence the call happened to contain.
    """
    x = np.asarray(x, dtype=np.float32)
    bed = np.asarray(bed, dtype=np.float32)
    if bed.size == 0 or x.size == 0:
        return x.copy()
    if bed.size < x.size:
        bed = np.concatenate([bed, np.zeros(x.size - bed.size, dtype=np.float32)])
    bed = bed[: x.size]

    speech = active_rms(x, mask) if mask is not None else float(
        np.sqrt(np.mean(np.square(x, dtype=np.float64)))
    )
    noise = float(np.sqrt(np.mean(np.square(bed, dtype=np.float64))))
    if speech <= 0.0 or noise <= 0.0:
        return x.copy()

    want = speech / (10.0 ** (snr_db / 20.0))
    return (x + bed * (want / noise)).astype(np.float32)
