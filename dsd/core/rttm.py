"""RTTM reading and writing.

Field layout, 1-indexed as the format numbers them:

    SPEAKER <uri> <channel> <start> <duration> <NA> <NA> <speaker> <NA> <NA>
       1      2       3        4         5                   8

Repo invariant, inherited from the sibling diarization projects: field 2 (the
URI) always equals the file stem, because every downstream join -- audio to
RTTM, hypothesis to reference -- is done on that stem.

A warning about field 3 that this pipeline exists partly to work around: it is
*not* reliably a channel index. `sandbox/moss_serving/diarize_batch.py` writes a
constant there (`--rttm-channel`, default "1"), so every line of the 5602 RTTMs
in `Datasets/rttms/` says channel 1 regardless of who is speaking. Code that
reads field 3 as the channel is correct only for RTTMs this pipeline produced;
see `dsd/stages/filter_channels.py` for the energy-based recovery used
otherwise.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Segment:
    start: float
    end: float
    speaker: str
    channel: int = 1  # as written in the RTTM, 1-based; see the module note

    @property
    def duration(self) -> float:
        return self.end - self.start


def rttm_path_for(audio_path: str | Path, rttm_dir: str | Path) -> tuple[str, Path]:
    """Return (call_id, rttm_path) for an audio file."""
    name = Path(audio_path).stem
    return name, Path(rttm_dir) / f"{name}.rttm"


def read_rttm(path: str | Path) -> list[Segment]:
    segments = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        start, duration = float(parts[3]), float(parts[4])
        segments.append(
            Segment(
                start=start,
                end=start + duration,
                speaker=parts[7],
                channel=int(parts[2]),
            )
        )
    segments.sort(key=lambda s: (s.start, s.channel))
    return segments


def write_rttm(path: str | Path, uri: str, segments) -> None:
    """Write an RTTM atomically -- a kill mid-write must not leave a partial file.

    Every per-file stage decides it has nothing to do by checking that the
    output exists, so a truncated RTTM would be permanently mistaken for a
    finished one.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        for s in segments:
            handle.write(
                f"SPEAKER {uri} {s.channel} {s.start:.3f} {s.duration:.3f} "
                f"<NA> <NA> {s.speaker} <NA> <NA>\n"
            )
    os.replace(tmp, path)


def speakers_of(segments) -> set[str]:
    return {s.speaker for s in segments}


def intervals_of(segments, speaker: str | None = None) -> list[tuple[float, float]]:
    """(start, end) pairs, optionally for one speaker only."""
    return [(s.start, s.end) for s in segments if speaker is None or s.speaker == speaker]


def total_speech(segments, speaker: str | None = None) -> float:
    """Seconds of speech, counting overlapping segments of one speaker once."""
    merged = merge_intervals(intervals_of(segments, speaker))
    return sum(end - start for start, end in merged)


def merge_intervals(
    intervals: list[tuple[float, float]], gap: float = 0.0
) -> list[tuple[float, float]]:
    """Union of intervals, joining any pair separated by at most `gap` seconds."""
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged = [list(ordered[0])]
    for start, end in ordered[1:]:
        if start - merged[-1][1] <= gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]


def subtract_intervals(
    intervals: list[tuple[float, float]],
    excluded: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """`intervals` minus `excluded`, both as (start, end) second-pairs.

    Used to cut a rejected minor speaker's time out of a channel's speech mask,
    so their voice never reaches the source signal.
    """
    remaining = merge_intervals(intervals)
    if not excluded:
        return remaining

    result = []
    for start, end in remaining:
        pieces = [(start, end)]
        for cut_start, cut_end in merge_intervals(excluded):
            survivors = []
            for piece_start, piece_end in pieces:
                if cut_end <= piece_start or cut_start >= piece_end:
                    survivors.append((piece_start, piece_end))
                    continue
                if piece_start < cut_start:
                    survivors.append((piece_start, cut_start))
                if piece_end > cut_end:
                    survivors.append((cut_end, piece_end))
            pieces = survivors
            if not pieces:
                break
        result.extend(pieces)
    return [(s, e) for s, e in result if e > s]


def split_long(start: float, end: float, longest: float) -> list[tuple[float, float]]:
    """Cut an interval into equal pieces of at most `longest` seconds."""
    from math import ceil

    pieces = max(1, ceil((end - start) / longest))
    step = (end - start) / pieces
    return [(start + i * step, start + (i + 1) * step) for i in range(pieces)]
