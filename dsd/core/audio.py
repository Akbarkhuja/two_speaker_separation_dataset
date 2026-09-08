"""Audio I/O and the sample-domain helpers every stage shares.

All disk I/O goes through soundfile. Sources are 8 kHz 2-channel OGG/OPUS and
stay at 8 kHz end to end -- the only resampling in the pipeline is the one
TitaNet forces (see `dsd/backends/embedders/titanet.py`).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import soundfile as sf


def read_audio(path: str | Path, dtype: str = "float32") -> tuple[np.ndarray, int]:
    """Read a file as (samples, sample_rate), always shaped (n, channels)."""
    data, sr = sf.read(str(path), dtype=dtype, always_2d=True)
    return data, int(sr)


def probe(path: str | Path) -> tuple[int, int, float]:
    """Return (channels, sample_rate, duration_sec) without decoding the file."""
    info = sf.info(str(path))
    return info.channels, int(info.samplerate), float(info.duration)


def write_wav(
    path: str | Path,
    data: np.ndarray,
    sr: int,
    subtype: str = "PCM_16",
    fmt: str = "WAV",
) -> None:
    """Write an audio file atomically, so a kill mid-write cannot leave a half file.

    Stages skip work whose output already exists; a truncated file left by an
    interrupted run would be treated as done and silently poison the dataset.

    `fmt` is normally "WAV"; the enhancement cache passes "FLAC", which is
    lossless at PCM_16 and roughly half the size over a corpus of speech.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    # `format` is explicit because the temp name ends in `.tmp`, and soundfile
    # infers the container from the extension -- without this it raises
    # "unable to get format from file extension" on every single write.
    sf.write(str(tmp), data, sr, subtype=subtype, format=fmt)
    os.replace(tmp, path)


def demux(data: np.ndarray, channel: int) -> np.ndarray:
    """Extract one channel as a contiguous float32 mono array.

    `np.ascontiguousarray` is not decoration: slicing column `c` out of an
    (n, 2) array gives a strided view, and several model preprocessors read it
    as a raw buffer.
    """
    return np.ascontiguousarray(data[:, channel], dtype=np.float32)


def resample(x: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample a 1-D signal. librosa is imported lazily -- it is slow to load."""
    if orig_sr == target_sr:
        return np.ascontiguousarray(x, dtype=np.float32)

    import librosa

    return np.ascontiguousarray(
        librosa.resample(np.asarray(x, dtype=np.float32), orig_sr=orig_sr, target_sr=target_sr),
        dtype=np.float32,
    )


# --------------------------------------------------------------------------- #
# masks
# --------------------------------------------------------------------------- #
def speech_mask(intervals, n_samples: int, sr: int) -> np.ndarray:
    """Boolean sample mask that is True inside any (start, end) second-pair."""
    mask = np.zeros(n_samples, dtype=bool)
    for start, end in intervals:
        a = max(0, int(round(start * sr)))
        b = min(n_samples, int(round(end * sr)))
        if b > a:
            mask[a:b] = True
    return mask


def fade_envelope(mask: np.ndarray, sr: int, fade_ms: float) -> np.ndarray:
    """Turn a boolean mask into a float gain envelope with raised-cosine edges.

    Multiplying by the bare mask zeroes non-speech instantly, and every one of
    those step discontinuities is a broadband click sitting exactly on a label
    boundary. A separation model happily keys on that instead of on the voices,
    then falls apart on real audio. A few milliseconds of cosine taper removes
    the step without eating any measurable speech.
    """
    envelope = mask.astype(np.float32)
    fade = int(round(sr * fade_ms / 1000.0))
    if fade < 2 or not mask.any():
        return envelope

    # Ramp shape, and the sample indices where the mask turns on and off.
    ramp = (0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, fade + 2)[1:-1])).astype(np.float32)
    edges = np.flatnonzero(np.diff(mask.astype(np.int8)))

    for edge in edges:
        if mask[edge + 1]:  # silence -> speech: fade in, ending at the edge
            start = edge + 1
            stop = min(len(envelope), start + fade)
            envelope[start:stop] = np.minimum(envelope[start:stop], ramp[: stop - start])
        else:  # speech -> silence: fade out, starting at the edge
            stop = edge + 1
            start = max(0, stop - fade)
            envelope[start:stop] = np.minimum(envelope[start:stop], ramp[: stop - start][::-1])
    return envelope


def active_rms(x: np.ndarray, mask: np.ndarray | None = None) -> float:
    """RMS over the masked (speech) samples only.

    Measuring over the whole signal would make a mostly-silent side look quiet
    and get it boosted far above its true level when the mixer hits a target
    SIR.
    """
    values = x if mask is None else x[mask]
    if values.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))
