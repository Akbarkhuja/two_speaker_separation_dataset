"""Silero VAD, per channel.

Defaults are the notebook's (cell 14) and are tuned for telephone speech:
250 ms minimum speech, 200 ms minimum silence, 20 ms padding. The 200 ms silence
rule also matches the DIHARD-III / VoxConverse convention the sibling projects
use, so segment boundaries stay comparable across repos.

Silero only accepts 8 kHz and 16 kHz. Our sources are 8 kHz, so nothing is
resampled here -- which matters, because the VAD masks produced by this stage
are applied sample-for-sample to the original audio when the sources are
zerofied.
"""

from __future__ import annotations

import numpy as np

from ...registry import VADS


class SileroVAD:
    def __init__(self, options: dict):
        self.min_speech_duration_ms = int(options.get("min_speech_duration_ms", 250))
        self.min_silence_duration_ms = int(options.get("min_silence_duration_ms", 200))
        self.speech_pad_ms = int(options.get("speech_pad_ms", 20))
        self.threshold = float(options.get("threshold", 0.5))
        self._model = None

    def _load(self):
        if self._model is None:
            from silero_vad import load_silero_vad

            print("[vad] loading silero")
            self._model = load_silero_vad()
        return self._model

    def speech(self, samples: np.ndarray, sr: int) -> list[tuple[float, float]]:
        import torch
        from silero_vad import get_speech_timestamps

        if sr not in (8000, 16000):
            raise ValueError(f"silero supports 8 kHz and 16 kHz only, got {sr}")

        timestamps = get_speech_timestamps(
            torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32)),
            self._load(),
            threshold=self.threshold,
            min_speech_duration_ms=self.min_speech_duration_ms,
            min_silence_duration_ms=self.min_silence_duration_ms,
            speech_pad_ms=self.speech_pad_ms,
            sampling_rate=sr,
            return_seconds=True,
        )
        return [(float(t["start"]), float(t["end"])) for t in timestamps]


@VADS.register("silero")
def _build(options: dict) -> SileroVAD:
    return SileroVAD(options)
