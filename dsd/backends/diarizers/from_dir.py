"""Adopt RTTMs that already exist on disk instead of running a model.

This is how `Datasets/rttms/` -- 5602 files produced elsewhere by
`sandbox/moss_serving/diarize_batch.py --mode mono` and copied over -- stays
usable without re-diarizing anything.

Those RTTMs are mono-mode output: every line says channel 1 and the speakers are
S01/S02/S03, so the channel field is pure filler. This backend copies them
through verbatim rather than guessing; recovering which channel each speaker is
on is `filter_channels`' job, and it does it from audio energy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ...core.rttm import Segment, read_rttm
from ...registry import DIARIZERS


class FromDirDiarizer:
    # No model and no audio decode: this stage becomes a file copy.
    needs_audio = False

    def __init__(self, options: dict):
        rttm_dir = options.get("rttm_dir")
        if not rttm_dir:
            raise ValueError("the from_dir diarizer needs options.rttm_dir")
        self.rttm_dir = Path(rttm_dir).expanduser()

    def diarize(
        self,
        call: str,
        audio_path: Path,
        data: np.ndarray | None,
        sr: int | None,
    ) -> list[Segment]:
        source = self.rttm_dir / f"{call}.rttm"
        if not source.exists():
            raise FileNotFoundError(f"no RTTM for {call} under {self.rttm_dir}")
        return read_rttm(source)


@DIARIZERS.register("from_dir")
def _build(options: dict) -> FromDirDiarizer:
    return FromDirDiarizer(options)
