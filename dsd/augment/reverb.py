"""Reverberation, from simulated shoebox rooms and from real measured rooms.

The paper simulates with pyroomacoustics: RT60 ~ U(0.1, 1.0) s, each room
dimension ~ U(2, 20) m, wall absorption and image-source order from Sabine's
equation. That is `simulate()` below, and it gives unlimited distinct rooms.

Alongside it the bank carries 270 real impulse responses from the MIT
Acoustical Reverberation Scene Statistics Survey, already peak-aligned on disk
in the sibling project. A shoebox model cannot produce a measured room's
irregular early reflections, and a pool of 270 is small enough to memorise, so
neither source is used alone -- `RIRBank.draw` picks between them.

Every impulse response in the bank is **peak-aligned and unit-peak**: the direct
path sits at sample 0 and has amplitude 1. That is what lets `apply()` truncate
the convolution back to the input length without shifting the signal against its
VAD labels -- the single most important property here, since `build` indexes the
speech masks into these arrays.
"""

from __future__ import annotations

import numpy as np
import soundfile as sf

MEASURED = "measured"
SIMULATED = "simulated"


def align(ir: np.ndarray, sr: int, max_sec: float = 1.0) -> np.ndarray:
    """Trim to the direct-path peak, cap the tail, normalise to unit peak.

    Same three operations as `prepare_irs()` in the sibling project's
    `build_synth_diar_v4.py`, and for the same reason: convolution truncated
    from index 0 has no propagation-delay compensation, so any leading silence
    in the impulse response delays the whole signal while the labels stay put.
    """
    ir = np.asarray(ir, dtype=np.float32)
    if ir.size < 2:
        return np.array([1.0], dtype=np.float32)
    ir = ir[int(np.argmax(np.abs(ir))) :][: max(1, int(max_sec * sr))]
    peak = float(np.abs(ir).max())
    if peak <= 0.0:
        return np.array([1.0], dtype=np.float32)
    return np.ascontiguousarray(ir / peak, dtype=np.float32)


def simulate(rng, sr: int, rt60_range=(0.1, 1.0), dim_range=(2.0, 20.0),
             max_sec: float = 1.0) -> tuple[np.ndarray, dict]:
    """One shoebox room impulse response. Returns (ir, the draws that made it).

    Sabine's formula turns the target RT60 and the room volume into a wall
    absorption and an image-source order; pyroomacoustics' `inverse_sabine` does
    exactly that and also tells us when a room is too small or too absorptive
    for the target to be physically reachable, which it raises on. Those draws
    are re-rolled rather than clamped -- a clamped RT60 would put a spike in the
    distribution at whichever bound was hit.
    """
    import pyroomacoustics as pra

    for _ in range(50):
        rt60 = rng.uniform(*rt60_range)
        dims = [rng.uniform(*dim_range) for _ in range(3)]
        try:
            absorption, max_order = pra.inverse_sabine(rt60, dims)
        except ValueError:
            continue
        # A 20 m room at a long RT60 asks for an image-source order up to ~77,
        # which is 77^3 image sources and minutes of compute for one response.
        # Capped at 12, measured over eight rooms: requested and realised RT60
        # (Schroeder T30) correlate at 0.89 with a mean ratio of 0.955, at 2-4 ms
        # per room. The cap shortens the tail slightly; it does not flatten the
        # distribution, which is what would have made the step useless.
        max_order = int(min(max_order, 12))
        room = pra.ShoeBox(
            dims, fs=sr, materials=pra.Material(absorption), max_order=max_order
        )
        # Source and microphone placed independently, each kept half a metre off
        # every wall so neither sits in a corner null.
        margin = 0.5
        place = lambda: [rng.uniform(margin, d - margin) for d in dims]  # noqa: E731
        room.add_source(place())
        room.add_microphone(np.array([place()]).T)
        room.compute_rir()
        ir = align(np.asarray(room.rir[0][0], dtype=np.float32), sr, max_sec)
        if ir.size > 1:
            return ir, {
                "rt60": round(rt60, 4),
                "room_dim": [round(d, 3) for d in dims],
                "absorption": round(float(absorption), 4),
                "max_order": max_order,
            }
    raise RuntimeError("could not simulate a usable room in 50 attempts")


def apply(x: np.ndarray, ir: np.ndarray) -> np.ndarray:
    """Convolve, truncate back to length, and restore the dry RMS.

    The level is put back deliberately. `build` measured and recorded this
    channel's SIR before the degradation chain ran, and a reverberant tail adds
    energy -- letting it through would make the `sir_db` in `meta.json` a
    description of a signal that no longer exists. Distance cues are lost with
    it, which is the right trade here: the mixture's level relationship to its
    targets is something a separation model depends on.
    """
    from scipy.signal import fftconvolve

    x = np.asarray(x, dtype=np.float32)
    ir = np.asarray(ir, dtype=np.float32)
    if x.size == 0 or ir.size == 0:
        return x.copy()

    wet = np.asarray(fftconvolve(x, ir)[: x.size], dtype=np.float32)
    dry_rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
    wet_rms = float(np.sqrt(np.mean(np.square(wet, dtype=np.float64))))
    if dry_rms <= 0.0 or wet_rms <= 0.0:
        return wet
    return (wet * (dry_rms / wet_rms)).astype(np.float32)


class RIRBank:
    """The two impulse-response pools, drawn from by `simulated_frac`."""

    def __init__(self, simulated_dir, measured_dir, sr: int, cache_size: int = 256):
        self.sr = int(sr)
        self.cache_size = int(cache_size)
        self.simulated = sorted(str(p) for p in _wavs(simulated_dir))
        self.measured = sorted(str(p) for p in _wavs(measured_dir))
        self._cache: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.simulated) + len(self.measured)

    def available(self) -> bool:
        return bool(self.simulated or self.measured)

    def load(self, path: str) -> np.ndarray:
        if path not in self._cache:
            data, file_sr = sf.read(path, dtype="float32", always_2d=True)
            ir = data[:, 0]
            if file_sr != self.sr:
                # The bank is written at our rate by the `augment` stage, so this
                # is a fallback for a hand-placed file rather than the normal path.
                from ..core.audio import resample

                ir = align(resample(ir, file_sr, self.sr), self.sr)
            if len(self._cache) >= self.cache_size:
                self._cache.pop(next(iter(self._cache)))
            self._cache[path] = np.ascontiguousarray(ir, dtype=np.float32)
        return self._cache[path]

    def draw(self, rng, simulated_frac: float = 0.5) -> tuple[str, str] | None:
        """Pick one impulse response. Returns (kind, path), or None if empty.

        Falls back to whichever pool exists when the other is empty, so a run
        with no pyroomacoustics installed still gets measured reverb rather than
        silently getting no reverb at all.
        """
        pools = []
        if self.simulated:
            pools.append((SIMULATED, self.simulated))
        if self.measured:
            pools.append((MEASURED, self.measured))
        if not pools:
            return None
        if len(pools) == 1:
            kind, pool = pools[0]
        else:
            kind, pool = pools[0] if rng.random() < simulated_frac else pools[1]
        return kind, rng.choice(pool)


def _wavs(directory) -> list:
    from pathlib import Path

    if directory is None:
        return []
    path = Path(directory)
    return sorted(path.glob("*.wav")) if path.is_dir() else []
