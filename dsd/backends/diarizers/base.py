"""The diarizer contract, plus the per-channel wrapper they all share.

A backend returns a flat `list[Segment]` with the channel field already filled
in, so the stage only has to write the RTTM. Backends that work per channel get
`per_channel()` to do the demux, the label namespacing and the merge.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Protocol

import numpy as np

from ...core.audio import demux
from ...core.rttm import Segment


class Diarizer(Protocol):
    # False lets a backend skip the audio decode entirely (`from_dir` does).
    needs_audio: bool

    def diarize(
        self,
        call: str,
        audio_path: Path,
        data: np.ndarray | None,
        sr: int | None,
    ) -> list[Segment]:
        """Return every segment of the call, channel field set, 1-based."""


def per_channel(
    data: np.ndarray,
    sr: int,
    diarize_mono: Callable[[list[np.ndarray], int], list[list[Segment]]],
) -> list[Segment]:
    """Diarize each channel alone and merge, namespacing the per-channel labels.

    `diarize_mono` takes every channel at once so a backend can batch them.

    The namespacing is load-bearing. A diarizer run on one channel restarts its
    labels at S01/speaker_0 for that channel, so writing them raw gives both
    parties the same name and collapses them into one identity downstream.
    Overwriting the label with the channel index instead -- which is what
    `sortformer_diarize_dual_channel_audio` originally did -- is worse: it makes
    the one-speaker-per-channel filter compare a value against itself, so every
    call passes and nothing is ever filtered. Keep the model's own label and
    prefix it, so a channel carrying two real speakers still shows up as two
    distinct labels.
    """
    channels = [demux(data, c) for c in range(data.shape[1])]
    per_channel_segments = diarize_mono(channels, sr)

    segments: list[Segment] = []
    for index, channel_segments in enumerate(per_channel_segments):
        segments.extend(
            Segment(
                start=s.start,
                end=s.end,
                speaker=f"ch{index}_{s.speaker}",
                channel=index + 1,  # RTTM channels are 1-based
            )
            for s in channel_segments
        )
    segments.sort(key=lambda s: (s.start, s.channel))
    return segments
