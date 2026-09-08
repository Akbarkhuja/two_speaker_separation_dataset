"""Finding source audio and pairing it with its RTTM.

Calls are identified by the file stem (a UUID in this corpus), which is also the
RTTM URI field and the key in every artifact under `work/`.
"""

from __future__ import annotations

import os
import sys
from glob import glob
from pathlib import Path


def collect_audio(dirs, pattern: str) -> list[str]:
    """All files matching `pattern` under each directory, deduplicated and sorted."""
    files: list[str] = []
    for directory in dirs:
        found = glob(os.path.join(str(directory), pattern), recursive=True)
        if not found:
            print(f"[warn] no files matched {pattern!r} under {directory}", file=sys.stderr)
        files.extend(found)
    return sorted(set(files))


def call_id(path: str | Path) -> str:
    return Path(path).stem


def index_by_call(paths) -> dict[str, str]:
    """Map call id -> path, warning about stems that appear more than once.

    Every artifact in `work/` is keyed by call id, so two audio files sharing a
    stem would quietly overwrite each other's RTTM, VAD and embeddings.
    """
    index: dict[str, list[str]] = {}
    for path in paths:
        index.setdefault(call_id(path), []).append(str(path))

    duplicates = {k: v for k, v in index.items() if len(v) > 1}
    if duplicates:
        example = next(iter(duplicates))
        print(
            f"[warn] {len(duplicates)} duplicate basenames; only the first of each is used "
            f"(e.g. {example}: {len(duplicates[example])} files)",
            file=sys.stderr,
        )
    return {k: v[0] for k, v in index.items()}


def pending(items, output_for, overwrite: bool = False, limit: int | None = None) -> list:
    """Filter a work list down to items whose output is missing.

    This is the whole resume mechanism for the per-file stages: nothing is
    tracked in a database, an output on disk simply means done.
    """
    if not overwrite:
        items = [item for item in items if not Path(output_for(item)).exists()]
    if limit:
        items = items[:limit]
    return list(items)
