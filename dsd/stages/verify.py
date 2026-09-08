"""A separate pass that re-reads the finished dataset and checks its invariants.

Written as a stage rather than a test because it checks the *artifact*, not the
code: it re-opens the wavs from disk and re-derives everything, so it catches a
truncated write or a half-finished rerun that a unit test never would.

The mixture tolerance is 1e-4 rather than 0. Each of the three wavs is
independently quantized to 16-bit PCM, so `mix - (s1 + s2)` carries up to about
1.5 LSB of rounding, roughly 4.6e-5. Anything materially larger means the peak
guard scaled the mixture without scaling the sources, or the files came from
different runs.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from ..core.audio import read_audio
from ..core.manifest import read_json, read_jsonl
from .base import banner, progress, require

NAME = "verify"
REQUIRES = ("build",)

MIX_TOLERANCE = 1e-4


def _check_split_manifests(directory: Path, combined, cfg, what: str) -> list[str]:
    """The per-split manifests must partition the combined one.

    They are derived data, so the failure to guard against is drift: a stale
    `train.jsonl` left by an earlier build with different splits would keep
    training on rows the current dataset no longer contains, and nothing else
    here would notice.
    """
    problems: list[str] = []
    seen: list[dict] = []

    for split in cfg.select.splits:
        path = directory / f"{split}.jsonl"
        if not path.exists():
            problems.append(f"{path.name} is missing from {directory.name}/")
            continue
        rows = list(read_jsonl(path))
        wrong = [r for r in rows if r.get("split") != split]
        if wrong:
            problems.append(f"{path.name}: {len(wrong)} row(s) carry another split")
        seen.extend(rows)

    if len(seen) != len(combined):
        problems.append(
            f"{what}: split manifests hold {len(seen)} rows, combined holds {len(combined)}"
        )

    # Compare identities, not just counts -- equal totals with the wrong members
    # is exactly what a stale file looks like.
    key = (lambda r: r["mix"]) if what == "chunks" else (lambda r: r["call"])
    missing = {key(r) for r in combined} - {key(r) for r in seen}
    if missing:
        problems.append(
            f"{what}: {len(missing)} in the combined manifest are in no split file "
            f"(e.g. {sorted(missing)[0]})"
        )
    return problems


def add_args(parser) -> None:
    parser.add_argument("--limit", type=int, help="check only the first N calls")
    parser.add_argument(
        "--sample", type=int, help="check a random N calls instead of all of them"
    )


def run(cfg, args) -> None:
    dataset_dir = cfg.paths.dataset_dir
    manifest_path = dataset_dir / "manifest.jsonl"
    require({"manifest.jsonl": manifest_path}, NAME)

    rows = list(read_jsonl(manifest_path))
    if args.sample:
        import random

        rows = random.Random(cfg.seed).sample(rows, min(args.sample, len(rows)))
    elif args.limit:
        rows = rows[: args.limit]

    banner(NAME, f"checking {len(rows)} calls under {dataset_dir}")
    problems: list[str] = []

    # ---- per-call signal checks -------------------------------------- #
    worst_residual = 0.0
    for row in progress(rows, "verify"):
        call = row["call"]
        try:
            mixture, sr = read_audio(dataset_dir / row["mix"])
            s1, sr1 = read_audio(dataset_dir / row["s1"])
            s2, sr2 = read_audio(dataset_dir / row["s2"])
        except Exception as exc:
            problems.append(f"{call}: unreadable ({exc})")
            continue

        if not (sr == sr1 == sr2 == cfg.sample_rate):
            problems.append(f"{call}: sample rates {sr}/{sr1}/{sr2}, expected {cfg.sample_rate}")
        if not (len(mixture) == len(s1) == len(s2)):
            problems.append(
                f"{call}: length mismatch mix={len(mixture)} s1={len(s1)} s2={len(s2)}"
            )
            continue

        residual = float(np.abs(mixture[:, 0] - (s1[:, 0] + s2[:, 0])).max())
        worst_residual = max(worst_residual, residual)
        if residual > MIX_TOLERANCE:
            problems.append(f"{call}: mix != s1 + s2 (max residual {residual:.2e})")

        # Zerofying must have silenced something on both sides; a source with
        # no exact zeros means the mask was never applied.
        for name, source in (("s1", s1[:, 0]), ("s2", s2[:, 0])):
            if not np.any(source == 0.0):
                problems.append(f"{call}: {name} has no silence -- was it zerofied?")
            if not np.any(source != 0.0):
                problems.append(f"{call}: {name} is entirely silent")

    # ---- dataset-level checks ---------------------------------------- #
    all_rows = list(read_jsonl(manifest_path))
    splits_by_speaker: dict[str, set] = defaultdict(set)
    calls_by_speaker: dict[str, int] = defaultdict(int)
    gender_counts: dict[str, int] = defaultdict(int)
    for row in all_rows:
        for speaker in row["speakers"]:
            splits_by_speaker[speaker].add(row["split"])
            calls_by_speaker[speaker] += 1
        for gender in row["genders"]:
            gender_counts[str(gender)] += 1

    leaked = {s: sorted(v) for s, v in splits_by_speaker.items() if len(v) > 1}
    if leaked:
        example = next(iter(leaked.items()))
        problems.append(
            f"{len(leaked)} speakers appear in more than one split (e.g. {example[0]} in "
            f"{example[1]})"
        )

    cap = cfg.select.max_calls_per_speaker
    over = {s: n for s, n in calls_by_speaker.items() if n > cap}
    if over:
        worst = max(over.items(), key=lambda kv: kv[1])
        problems.append(
            f"{len(over)} speakers exceed the {cap}-call cap (worst: {worst[0]} with {worst[1]})"
        )

    missing = [row["mix"] for row in all_rows if not (dataset_dir / row["mix"]).exists()]
    if missing:
        problems.append(f"{len(missing)} manifest paths do not resolve (e.g. {missing[0]})")

    male, female = gender_counts.get("male", 0), gender_counts.get("female", 0)
    if male + female:
        share = male / (male + female)
        banner(NAME, f"gender sides: male={male} female={female} (male share {share:.3f})")
        if cfg.select.balance_gender and abs(share - 0.5) > 0.10:
            problems.append(f"gender share {share:.3f} is more than 10 points off balance")

    # ---- per-split manifests partition the combined one --------------- #
    problems += _check_split_manifests(dataset_dir, all_rows, cfg, "calls")

    # ---- chunks, if present ------------------------------------------ #
    chunk_manifest = dataset_dir / "chunks" / "manifest.jsonl"
    if chunk_manifest.exists():
        chunk_rows = list(read_jsonl(chunk_manifest))
        by_call = {row["call"]: row["split"] for row in all_rows}
        stray = [r["call"] for r in chunk_rows if by_call.get(r["call"]) != r["split"]]
        if stray:
            problems.append(f"{len(stray)} chunks disagree with their call's split")
        problems += _check_split_manifests(
            dataset_dir / "chunks", chunk_rows, cfg, "chunks"
        )
        banner(NAME, f"chunks: {len(chunk_rows)} rows")

    # ---- verdict ------------------------------------------------------ #
    banner(NAME, f"calls in manifest: {len(all_rows)}, unique speakers: {len(calls_by_speaker)}")
    banner(NAME, f"worst mix residual: {worst_residual:.2e} (tolerance {MIX_TOLERANCE:.0e})")
    if problems:
        banner(NAME, f"FAILED with {len(problems)} problem(s):")
        for problem in problems[:40]:
            print(f"[{NAME}]   - {problem}")
        if len(problems) > 40:
            print(f"[{NAME}]   ... and {len(problems) - 40} more")
        raise SystemExit(1)
    banner(NAME, "OK -- every check passed")
