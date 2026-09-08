"""What every stage looks like, and the few helpers they all use.

A stage is a module exposing four names:

    NAME      the CLI subcommand
    REQUIRES  stage names whose artifacts must already exist
    add_args  extra argparse flags for this stage
    run       run(cfg, args) -> None

That is the whole contract. `dsd/stages/__init__.py` holds the ordered list, and
`dsd/cli.py` builds a subparser per stage from it.

Every stage is idempotent. Per-file stages skip an item whose output is already
on disk; whole-artifact stages skip entirely unless `--overwrite`. Items that
raise are collected and written to `work/<stage>_failed.tsv` rather than killing
the run -- with thousands of calls, one unreadable file should not cost the
other 5600.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

from tqdm import tqdm


def banner(name: str, message: str) -> None:
    print(f"[{name}] {message}")


def progress(items: Iterable, desc: str, total: int | None = None):
    return tqdm(items, desc=desc, total=total, unit="call", dynamic_ncols=True)


def require(paths: dict[str, Path], stage: str) -> None:
    """Fail early and legibly when a prerequisite artifact is missing.

    The alternative is a FileNotFoundError three frames deep, which does not
    tell the user which earlier stage they skipped.
    """
    missing = [(what, path) for what, path in paths.items() if not Path(path).exists()]
    if not missing:
        return
    lines = [f"stage {stage!r} is missing input:"]
    lines += [f"  {what}: {path}" for what, path in missing]
    lines.append("run the earlier stages first, or check paths.work_dir")
    sys.exit("\n".join(lines))


def skip_if_done(path: Path, overwrite: bool, stage: str) -> bool:
    if path.exists() and not overwrite:
        banner(stage, f"{path.name} exists, nothing to do (use --overwrite to redo)")
        return True
    return False


def summarize_counts(stage: str, counts: dict[str, int], total: int) -> None:
    """Print a reason -> count table, widest label first."""
    if not counts:
        return
    width = max(len(k) for k in counts)
    for reason, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        share = (count / total * 100) if total else 0.0
        print(f"[{stage}]   {reason:<{width}}  {count:>6}  ({share:5.1f}%)")
