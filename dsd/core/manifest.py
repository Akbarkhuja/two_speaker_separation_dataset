"""Atomic JSON / JSONL helpers for the artifacts stages hand to each other.

Everything is written through a temp file and `os.replace`. These files are the
resume points: a stage that finds `work/speakers.json` on disk assumes it is
complete, so a partial write from an interrupted run would be indistinguishable
from a finished one.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator


def write_json(path: str | Path, payload: Any, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=indent, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    count = 0
    with open(tmp, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    os.replace(tmp, path)
    return count


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_failures(path: str | Path, failures: list[tuple[str, str]]) -> None:
    """Record `(item, error)` pairs a stage could not process, as TSV."""
    path = Path(path)
    if not failures:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        for item, error in failures:
            handle.write(f"{item}\t{error}\n")
    os.replace(tmp, path)
