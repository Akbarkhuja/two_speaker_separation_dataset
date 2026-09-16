"""Encode and decode the signal through a real codec, via ffmpeg.

The paper compresses with MP3 at 65-245 kbps. On this corpus that step barely
does anything: measured on a real call, an MP3 round trip at 128 kbps leaves
25 dB SNR, and the source files are `.opus` already -- the audio has been
through a codec before we ever see it. What a call-centre recording actually
survives is a narrowband telephony codec, so that is what we draw from:

    G.711 mu-law   37.3 dB     the PSTN default, near-transparent
    G.711 A-law    37.5 dB     the same, European variant
    GSM 06.10      10.7 dB     a mobile leg
    Opus 6-24 kbps  3.9-26 dB  VoIP, and the severe end of the range

(Measured on ten seconds of a real call from `Datasets/audios`.)

Everything runs through pipes rather than temp files -- two ffmpeg processes
chained, encoder into decoder -- so a build that degrades 2 channels x 4
variants per call does not write 8 files per call to do it.
"""

from __future__ import annotations

import subprocess

import numpy as np

# name -> (encoder args, the container/raw format to carry it in)
#
# The raw formats (mulaw/alaw/gsm) are used in preference to a container
# because they cannot carry a sample rate, which means ffmpeg cannot silently
# resample on the way back in.
CODECS: dict[str, tuple[list[str], str]] = {
    "mulaw": (["-acodec", "pcm_mulaw"], "mulaw"),
    "alaw": (["-acodec", "pcm_alaw"], "alaw"),
    "gsm": (["-acodec", "libgsm"], "gsm"),
    "opus": (["-acodec", "libopus"], "ogg"),
    "mp3": (["-acodec", "libmp3lame"], "mp3"),
}


class CodecError(RuntimeError):
    pass


def available() -> list[str]:
    """Which of `CODECS` this machine's ffmpeg can actually encode."""
    try:
        listing = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, timeout=30
        ).stdout.decode("utf-8", "replace")
    except (OSError, subprocess.SubprocessError):
        return []
    return [
        name
        for name, (args, _) in CODECS.items()
        if f" {args[args.index('-acodec') + 1]} " in listing
    ]


# Longest encoder delay we will look for, in seconds. MP3's is 1105 samples
# (138 ms at 8 kHz); nothing here is near a quarter second.
MAX_DELAY_SEC = 0.25

# (rate, codec) -> measured encoder delay in samples. Filled on first use and
# kept for the process: it is a property of the codec, not of the audio.
_DELAY: dict[tuple[int, str], int] = {}


def _measure_delay(sr: int, kind: str) -> int:
    """How many samples this codec shifts the signal.

    Measured, not assumed. The block-based codecs prepend encoder delay, and
    whether the decoder removes it again depends on the container: Opus in Ogg
    carries a pre-skip field and comes back aligned, while MP3 through a raw
    pipe has no gapless metadata to read and arrives 1105 samples late. Left
    uncompensated that is a 138 ms shift between a degraded mixture and its
    targets -- a dataset that looks fine and trains a model to output late.

    Two choices make the measurement trustworthy. It runs on a chirp rather than
    on the caller's audio, because the correlation peak has to be unambiguous
    after a codec that mangles the signal. And it runs at the codec's **default**
    bitrate, not the caller's: delay comes from the frame size and is the same at
    every bitrate (measured: MP3 gives 1105 at both 65 and 245 kbps), but at
    6 kbps Opus destroys the chirp badly enough that the peak wanders by a few
    samples -- and "correcting" by that much is worse than not correcting at all.
    """
    key = (sr, kind)
    if key in _DELAY:
        return _DELAY[key]

    from scipy.signal import correlate

    n = int(2 * sr)
    span = n / sr
    t = np.arange(n) / sr
    # A linear sweep from 200 Hz to just under Nyquist.
    probe = (0.5 * np.sin(2 * np.pi * (200.0 + (sr / 2 - 400.0) * t / (2 * span)) * t)).astype(
        np.float32
    )
    max_lag = int(MAX_DELAY_SEC * sr)
    try:
        got = _decode(probe, sr, kind, None)
    except CodecError:
        _DELAY[key] = 0
        return 0

    if got.size < n + max_lag:
        got = np.concatenate([got, np.zeros(n + max_lag - got.size, dtype=np.float32)])
    scores = correlate(got[: n + max_lag], probe, mode="valid")
    _DELAY[key] = int(np.argmax(scores))
    return _DELAY[key]


def round_trip(x: np.ndarray, sr: int, kind: str, kbps: int | None = None) -> np.ndarray:
    """Encode `x` with `kind` and decode it straight back. Length is preserved.

    Two things have to be undone for the result to still line up with the VAD
    masks `build` holds for this channel. Codecs pad -- they work in frames and
    hand back a whole number of them -- so the tail is trimmed, and short-padded
    if a codec ever returns less. And some of them delay, so the front is
    dropped by the amount `_measure_delay` found for this configuration.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x.copy()

    delay = _measure_delay(sr, kind)
    y = _decode(x, sr, kind, kbps)[delay:]
    if y.size >= x.size:
        return np.ascontiguousarray(y[: x.size])
    return np.concatenate([y, np.zeros(x.size - y.size, dtype=np.float32)])


def _decode(x: np.ndarray, sr: int, kind: str, kbps: int | None) -> np.ndarray:
    """One encode/decode round trip, with neither delay nor length corrected."""
    if kind not in CODECS:
        raise CodecError(f"unknown codec {kind!r}; known: {', '.join(sorted(CODECS))}")

    encode_args, container = CODECS[kind]
    encode = [
        "ffmpeg", "-hide_banner", "-v", "error",
        "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "pipe:0",
        *encode_args,
        *(["-b:a", f"{kbps}k"] if kbps else []),
        "-ar", str(sr), "-ac", "1", "-f", container, "pipe:1",
    ]
    decode = [
        "ffmpeg", "-hide_banner", "-v", "error",
        # The raw formats carry no header, so the rate has to be declared again
        # on the way in. It is harmless for the container formats.
        *(["-ar", str(sr)] if container in ("mulaw", "alaw") else []),
        "-f", container, "-i", "pipe:0",
        "-f", "f32le", "-ar", str(sr), "-ac", "1", "pipe:1",
    ]

    # Two sequential runs rather than one chained pipeline. Chaining the
    # encoder's stdout straight into the decoder deadlocks: the parent writes
    # megabytes into the encoder's stdin while nothing is draining the decoder's
    # stdout, both 64 KB pipe buffers fill, and every process blocks forever.
    # `subprocess.run` pumps each pipe for us, and the intermediate is the
    # compressed stream, which is small.
    try:
        encoded = subprocess.run(encode, input=x.tobytes(), capture_output=True, timeout=300)
        if encoded.returncode != 0 or not encoded.stdout:
            raise CodecError(
                f"{kind}: encoding failed "
                f"({encoded.stderr.decode('utf-8', 'replace').strip()[:200]})"
            )
        decoded = subprocess.run(decode, input=encoded.stdout, capture_output=True, timeout=300)
    except (OSError, subprocess.SubprocessError) as exc:
        raise CodecError(f"{kind}: ffmpeg failed ({exc!r})") from exc

    if decoded.returncode != 0 or not decoded.stdout:
        detail = decoded.stderr.decode("utf-8", "replace").strip()[:200]
        raise CodecError(f"{kind}: decoding returned nothing ({detail})")

    return np.frombuffer(decoded.stdout, dtype=np.float32)
