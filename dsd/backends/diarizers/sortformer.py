"""NVIDIA NeMo streaming-Sortformer diarizer, run per channel.

Offline (full-context) inference is the default, and that is a deliberate
departure from the notebook. The notebook copied the 1.04 s-latency streaming
preset out of the NVIDIA streaming tutorial, which is the right setting for a
live app and the wrong one for batch dataset building. Measured here on one
389 s call, both channels, on an RTX 4060:

    offline    3.0 s   259x realtime   ch1 -> 1 speaker label
    streaming 36.5 s    21x realtime   ch1 -> 2 speaker labels

Twelve times slower *and* noisier: as the streaming speaker cache rotates it
re-identifies the same voice and splits it into a second label, which then makes
the one-speaker-per-channel filter reject a call that is perfectly good. Over
the corpus this is the difference between ~2 h and ~23 h. Full context also
holds up on the worst case -- the longest call (1807 s, both channels) runs in
12.7 s at 2.5 GiB peak VRAM.

`streaming: true` restores the preset for anyone who wants to reproduce the
notebook's numbers.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

import numpy as np

from ...core.rttm import Segment
from ...registry import DIARIZERS
from .base import per_channel

DEFAULT_MODEL_PATH = os.getenv("STREAMING_SORTFORMER_PATH", None) 
# (
#     "/home/akbar/craft/prod/steno2/core/ml/model_weights/"
#     "diar_streaming_sortformer_4spk-v2.1.nemo"
# )


class SortformerDiarizer:
    needs_audio = True

    def __init__(self, options: dict):
        self.model_path = options.get("model_path", DEFAULT_MODEL_PATH)
        self.device = options.get("device", "auto")
        self.streaming = bool(options.get("streaming", False))
        self.batch_size = int(options.get("batch_size", 2))
        self._model = None

    # ------------------------------------------------------------------ #
    def _load(self):
        """Restore the model on first use, so `--help` never touches CUDA."""
        if self._model is not None:
            return self._model

        import torch
        from nemo.collections.asr.models import SortformerEncLabelModel

        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if not self.model_path:
            # Without this, DEFAULT_MODEL_PATH being None (the env var unset and
            # no model_path in the config) surfaces as `Path(None)` raising an
            # opaque TypeError instead of saying what is missing.
            raise FileNotFoundError(
                "no Sortformer weights configured; set "
                "diarize.options.sortformer.model_path in the config, or the "
                "STREAMING_SORTFORMER_PATH environment variable"
            )
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Sortformer weights not found: {self.model_path}")

        print(f"[diarize] loading sortformer on {device}: {self.model_path}")
        # NeMo prints a full config dump to stdout on restore; keep it out of
        # the progress bar's way.
        with contextlib.redirect_stdout(open(os.devnull, "w")):
            model = SortformerEncLabelModel.restore_from(
                restore_path=self.model_path, map_location=device, strict=False
            )
            model.to(device)
        model.eval()

        if self.streaming:
            # 1.04 s-latency preset from the NVIDIA-NeMo/Speech streaming
            # diarization tutorial (all units are 80 ms frames).
            sm = model.sortformer_modules
            sm.chunk_len = 6
            sm.chunk_right_context = 7
            sm.fifo_len = 188
            sm.spkcache_len = 188
            sm.spkcache_update_period = 144
            sm.log = False
            sm._check_streaming_parameters()

        self._model = model
        return model

    # ------------------------------------------------------------------ #
    def diarize(
        self,
        call: str,
        audio_path: Path,
        data: np.ndarray | None,
        sr: int | None,
    ) -> list[Segment]:
        model = self._load()

        def diarize_mono(channels: list[np.ndarray], sample_rate: int):
            # One call for every channel: NeMo batches them together, so both
            # sides of a conversation cost roughly what one side would.
            with contextlib.redirect_stdout(open(os.devnull, "w")):
                predictions = model.diarize(
                    audio=channels,
                    sample_rate=sample_rate,
                    batch_size=max(self.batch_size, len(channels)),
                    verbose=False,
                )
            return [_parse(channel) for channel in predictions]

        return per_channel(data, sr, diarize_mono)


def _parse(lines) -> list[Segment]:
    """NeMo returns 'start end speaker_N' strings; channel is filled in later."""
    segments = [
        Segment(start=float(start), end=float(end), speaker=speaker)
        for start, end, speaker in (line.split() for line in lines)
    ]
    segments.sort(key=lambda s: s.start)
    return segments


@DIARIZERS.register("sortformer")
def _build(options: dict) -> SortformerDiarizer:
    return SortformerDiarizer(options)
