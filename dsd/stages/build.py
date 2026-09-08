"""Stage 8 -- write the mixtures and their two ground-truth sources.

Per selected call, at the source's native 8 kHz:

    zerofy each channel against its own VAD mask   -> s1, s2
    optionally roll s2 to synthesize speech overlap
    scale s2 to a sampled SIR
    mix = s1 + s2, with a shared peak guard

The output for one call is `<split>/<call>/{mix,s1,s2}.wav` plus a `meta.json`
recording the speaker ids, genders, realized overlap, SIR and shift -- enough to
reconstruct exactly what was done without re-reading the source.

`mix == s1 + s2` holds sample-for-sample, and `dsd verify` checks it. Every
separation loss compares model outputs to these sources assuming they add up,
so anything that breaks the identity silently biases training.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ..core.audio import read_audio, write_wav
from ..core.manifest import read_json, write_failures, write_json, write_jsonl
from ..core.rttm import read_rttm, subtract_intervals
from ..mixing.chunker import is_usable, windows
from ..mixing.mixer import mix as mix_sources
from ..mixing.mixer import scale_to_sir
from ..mixing.overlap import overlap_ratio, roll, shift_for_target
from ..mixing.zerofy import zerofy
from .base import banner, progress, require
from .enhance import enhanced_path

NAME = "build"
REQUIRES = ("select", "vad")


def add_args(parser) -> None:
    parser.add_argument("--chunks", action="store_true", help="also write fixed-length windows")
    parser.add_argument("--overwrite", action="store_true", help="rebuild calls already written")
    parser.add_argument("--limit", type=int, help="build only the first N calls")


def run(cfg, args) -> None:
    require(
        {"selection.json": cfg.paths.selection_json, "VAD directory": cfg.paths.vad_dir},
        NAME,
    )
    selection = read_json(cfg.paths.selection_json)
    calls = sorted(selection["calls"].items())
    if args.limit:
        calls = calls[: args.limit]

    want_chunks = args.chunks or cfg.build.chunks
    dataset_dir = cfg.paths.dataset_dir
    banner(NAME, f"{len(calls)} calls -> {dataset_dir}  (chunks={want_chunks})")

    # Checked before any work, not per call: the per-call handler turns an
    # exception into a `build_failed.tsv` line and carries on, which for this
    # particular guard would mean exiting 0 with an empty dataset. The whole
    # point of `always` is to be loud about a cache that is not ready.
    if cfg.build.use_enhanced == "always":
        absent = [call for call, _ in calls if not enhanced_path(cfg, call).exists()]
        if absent:
            raise SystemExit(
                f"[{NAME}] build.use_enhanced='always' but {len(absent)} of {len(calls)} "
                f"calls have no enhanced audio (e.g. {absent[0]}).\n"
                f"[{NAME}] run `python -m dsd enhance` to finish the cache, or set "
                "build.use_enhanced=auto to build from raw audio where it is missing."
            )

    rows: list[dict] = []
    chunk_rows: list[dict] = []
    failed: list[tuple[str, str]] = []
    stats: dict[str, list] = defaultdict(list)

    for call, record in progress(calls, "build"):
        out_dir = dataset_dir / record["split"] / call
        meta_path = out_dir / "meta.json"
        if meta_path.exists() and not args.overwrite:
            meta = read_json(meta_path)
        else:
            try:
                # Seeded per call, so rebuilding one call reproduces exactly what
                # the full run made -- a global rng would not survive --limit.
                rng = random.Random(f"{cfg.seed}:{call}")
                meta = _build_call(call, record, cfg, out_dir, rng)
            except Exception as exc:
                failed.append((call, repr(exc)))
                continue

        rows.append(_manifest_row(meta, dataset_dir))
        stats["overlap"].append(meta["overlap"])
        stats["sir_db"].append(meta["sir_db"])
        stats["duration"].append(meta["duration"])
        stats["enhanced"].append(bool(meta.get("enhanced", False)))
        stats["shift_reason"].append(meta.get("shift_reason", "unknown"))
        stats[f"hours_{record['split']}"].append(meta["duration"] / 3600.0)

        if want_chunks:
            try:
                chunk_rows.extend(_build_chunks(call, record, meta, cfg, dataset_dir))
            except Exception as exc:
                failed.append((f"{call}:chunks", repr(exc)))

    write_jsonl(dataset_dir / "manifest.jsonl", rows)
    call_counts = write_split_manifests(dataset_dir, rows, cfg.select.splits)
    chunk_counts = {}
    if want_chunks:
        write_jsonl(dataset_dir / "chunks" / "manifest.jsonl", chunk_rows)
        chunk_counts = write_split_manifests(
            dataset_dir / "chunks", chunk_rows, cfg.select.splits
        )
    write_failures(cfg.paths.work_dir / "build_failed.tsv", failed)
    _write_stats(cfg, dataset_dir, selection, rows, chunk_rows, stats, chunk_counts)

    banner(NAME, "per-split manifests:")
    for split in sorted(call_counts):
        chunks_here = f"  {chunk_counts[split]:>7,} chunks" if chunk_counts else ""
        print(f"[{NAME}]   {split}.jsonl  {call_counts[split]:>6,} calls{chunks_here}")

    banner(NAME, f"wrote {len(rows)} calls" + (f", {len(chunk_rows)} chunks" if want_chunks else ""))
    if failed:
        banner(NAME, f"{len(failed)} failed; see work/build_failed.tsv")


# --------------------------------------------------------------------------- #
def write_split_manifests(directory: Path, rows: list[dict], splits) -> dict[str, int]:
    """One manifest per split, written beside the combined one.

    A file is emitted for **every configured split**, not only for the splits
    that happen to have rows. An empty `test.jsonl` is a far better answer to a
    training command than a missing path, and empty splits do occur -- on the
    current corpus `test` holds one call and no chunks at all.

    Splits the rows carry but the config does not name are written too, so a
    split renamed in config cannot make finished data silently disappear.

    Returns the row count per split.
    """
    by_split: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_split[row["split"]].append(row)

    names = list(splits) + [name for name in sorted(by_split) if name not in splits]
    if "manifest" in names:
        raise ValueError(
            "a split named 'manifest' would overwrite the combined manifest.jsonl; "
            "rename it in select.splits"
        )

    return {
        name: write_jsonl(directory / f"{name}.jsonl", by_split.get(name, []))
        for name in names
    }


def _source_audio(call: str, record: dict, cfg) -> tuple[str, bool]:
    """Pick the enhanced cache or the original recording, per `build.use_enhanced`.

    The enhanced file has the same rate and the same sample count as the
    original -- the enhance stage refuses anything else -- so every VAD offset
    below is valid either way.
    """
    mode = cfg.build.use_enhanced
    if mode == "never":
        return record["audio"], False

    cached = enhanced_path(cfg, call)
    if cached.exists():
        return str(cached), True
    if mode == "always":
        raise FileNotFoundError(
            f"build.use_enhanced='always' but {cached} is missing; run `python -m dsd enhance`"
        )
    return record["audio"], False


def _build_call(call: str, record: dict, cfg, out_dir: Path, rng: random.Random) -> dict:
    audio_path, enhanced = _source_audio(call, record, cfg)
    data, sr = read_audio(audio_path)
    segments = read_rttm(cfg.paths.vad_dir / f"{call}.rttm")

    intervals: dict[int, list] = defaultdict(list)
    for segment in segments:
        intervals[segment.channel - 1].append((segment.start, segment.end))

    # Cut out any time the filter attributed to a minor diarizer label. That
    # label was judged too small to reject the call over, but if it really was
    # a brief second voice this is what keeps it out of the source signal.
    for channel, spans in record.get("excluded", {}).items():
        channel = int(channel)
        intervals[channel] = subtract_intervals(
            intervals[channel], [tuple(span) for span in spans]
        )

    channels = record["channels"]
    s1, mask1 = zerofy(data[:, channels[0]], intervals[channels[0]], sr, cfg.build.fade_ms)
    s2, mask2 = zerofy(data[:, channels[1]], intervals[channels[1]], sr, cfg.build.fade_ms)

    natural = overlap_ratio(mask1, mask2)

    shift, shift_reason = 0, "natural"
    if rng.random() >= cfg.build.natural_frac:
        shift, _predicted, shift_reason = shift_for_target(
            mask1,
            mask2,
            sr,
            tuple(cfg.build.target_overlap),
            rng,
            guard_sec=cfg.build.seam_guard_ms / 1000.0,
        )
        s2 = roll(s2, shift)
        mask2 = roll(mask2, shift)

    sir_db = rng.uniform(*cfg.build.sir_db)
    s2, sir_gain = scale_to_sir(s1, s2, mask1, mask2, sir_db)
    mixture, s1, s2, peak_gain = mix_sources(s1, s2, cfg.build.peak_ceiling)

    write_wav(out_dir / "mix.wav", mixture, sr)
    write_wav(out_dir / "s1.wav", s1, sr)
    write_wav(out_dir / "s2.wav", s2, sr)

    meta = {
        "call": call,
        "split": record["split"],
        "source": audio_path,
        "enhanced": enhanced,
        "enhance_backend": cfg.enhance.backend if enhanced else None,
        "sample_rate": sr,
        "samples": int(len(mixture)),
        "duration": round(len(mixture) / sr, 3),
        "speakers": record["speakers"],
        "genders": record["genders"],
        "channels": channels,
        "speech_sec": [
            round(float(np.count_nonzero(mask1)) / sr, 3),
            round(float(np.count_nonzero(mask2)) / sr, 3),
        ],
        "natural_overlap": round(natural, 5),
        "overlap": round(overlap_ratio(mask1, mask2), 5),
        "shift_samples": int(shift),
        "shifted": bool(shift),
        # Why this call is (or is not) shifted -- see mixing/overlap.py:safe_shifts.
        "shift_reason": shift_reason,
        "sir_db": round(sir_db, 3),
        "sir_gain": round(sir_gain, 5),
        "peak_gain": round(peak_gain, 5),
    }
    write_json(out_dir / "meta.json", meta)
    return meta


def _build_chunks(call: str, record: dict, meta: dict, cfg, dataset_dir: Path) -> list[dict]:
    """Cut a built call into fixed-length windows where both speakers are active.

    Windows are cut from the *built* wavs rather than recomputed from the
    source, so a chunk is always a literal slice of the call it names and
    inherits its shift, SIR and peak gain unchanged.
    """
    split = record["split"]
    call_dir = dataset_dir / split / call
    mixture, sr = read_audio(call_dir / "mix.wav")
    s1, _ = read_audio(call_dir / "s1.wav")
    s2, _ = read_audio(call_dir / "s2.wav")
    mixture, s1, s2 = mixture[:, 0], s1[:, 0], s2[:, 0]

    # Recover activity from the written sources: no VAD needed, and it reflects
    # exactly what is in these files.
    mask1, mask2 = s1 != 0.0, s2 != 0.0

    out_root = dataset_dir / "chunks" / split
    rows = []
    index = 0
    for start, end in windows(len(mixture), sr, cfg.build.chunk_sec, cfg.build.chunk_hop):
        if not is_usable(mask1, mask2, start, end, sr, cfg.build.min_active_per_src):
            continue
        name = f"{call}_{index:04d}.wav"
        write_wav(out_root / "mix" / name, mixture[start:end], sr)
        write_wav(out_root / "s1" / name, s1[start:end], sr)
        write_wav(out_root / "s2" / name, s2[start:end], sr)
        rows.append(
            {
                "mix": str((out_root / "mix" / name).relative_to(dataset_dir)),
                "s1": str((out_root / "s1" / name).relative_to(dataset_dir)),
                "s2": str((out_root / "s2" / name).relative_to(dataset_dir)),
                "call": call,
                "split": split,
                "offset": round(start / sr, 3),
                "duration": round((end - start) / sr, 3),
                "sample_rate": sr,
                "speakers": meta["speakers"],
                "genders": meta["genders"],
                "overlap": round(
                    float(np.count_nonzero(mask1[start:end] & mask2[start:end]))
                    / max(int(np.count_nonzero(mask1[start:end] | mask2[start:end])), 1),
                    5,
                ),
            }
        )
        index += 1
    return rows


def _manifest_row(meta: dict, dataset_dir: Path) -> dict:
    base = Path(meta["split"]) / meta["call"]
    return {
        "mix": str(base / "mix.wav"),
        "s1": str(base / "s1.wav"),
        "s2": str(base / "s2.wav"),
        "call": meta["call"],
        "split": meta["split"],
        "duration": meta["duration"],
        "sample_rate": meta["sample_rate"],
        "speakers": meta["speakers"],
        "genders": meta["genders"],
        "overlap": meta["overlap"],
        "natural_overlap": meta["natural_overlap"],
        "sir_db": meta["sir_db"],
        "shifted": meta["shifted"],
        "enhanced": bool(meta.get("enhanced", False)),
    }


def _write_stats(cfg, dataset_dir: Path, selection, rows, chunk_rows, stats,
                 chunk_counts: dict[str, int] | None = None) -> None:
    """Realized numbers next to the targets that produced them."""
    def describe(values):
        if not values:
            return None
        array = np.asarray(values, dtype=np.float64)
        return {
            "n": int(array.size),
            "mean": round(float(array.mean()), 4),
            "p50": round(float(np.median(array)), 4),
            "p90": round(float(np.percentile(array, 90)), 4),
            "max": round(float(array.max()), 4),
        }

    per_split: dict[str, dict] = defaultdict(lambda: {"calls": 0, "hours": 0.0})
    speakers_per_split: dict[str, set] = defaultdict(set)
    gender_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        bucket = per_split[row["split"]]
        bucket["calls"] += 1
        bucket["hours"] += row["duration"] / 3600.0
        speakers_per_split[row["split"]].update(row["speakers"])
        for gender in row["genders"]:
            gender_counts[str(gender)] += 1

    for split, bucket in per_split.items():
        bucket["hours"] = round(bucket["hours"], 3)
        bucket["speakers"] = len(speakers_per_split[split])
        # Chunks per split, so `test: 1 call, 0 chunks` is visible here rather
        # than only discoverable by opening the manifest.
        bucket["chunks"] = int((chunk_counts or {}).get(split, 0))

    payload = {
        "config": cfg.to_dict(),
        "calls": len(rows),
        "chunks": len(chunk_rows),
        "splits": dict(per_split),
        "gender_sides": dict(gender_counts),
        "overlap": {
            "target": list(cfg.build.target_overlap),
            "natural_frac_target": cfg.build.natural_frac,
            "shifted_frac_realized": round(
                sum(1 for r in rows if r["shifted"]) / max(len(rows), 1), 4
            ),
            "realized": describe(stats["overlap"]),
        },
        "sir_db": {"target": list(cfg.build.sir_db), "realized": describe(stats["sir_db"])},
        # Why each call is or is not shifted. `boundary_not_silent` and
        # `no_safe_shift` are calls left unshifted because no wrap point fell in
        # silence -- shifting them anyway would cut an utterance in half.
        "shift_reason": dict(Counter(stats["shift_reason"])),
        "duration_sec": describe(stats["duration"]),
        "enhanced": {
            "mode": cfg.build.use_enhanced,
            "backend": cfg.enhance.backend,
            "calls": int(sum(stats["enhanced"])),
            "of": len(rows),
        },
        "selection_dropped": selection.get("dropped", {}),
    }
    write_json(dataset_dir / "stats.json", payload)

    print(f"[{NAME}] ===== dataset =====")
    for split, bucket in sorted(per_split.items()):
        print(
            f"[{NAME}] {split:<6} {bucket['calls']:>6} calls  {bucket['hours']:>7.2f} h  "
            f"{bucket['speakers']:>5} speakers  {bucket.get('chunks', 0):>7,} chunks"
        )
    if payload["overlap"]["realized"]:
        realized = payload["overlap"]["realized"]
        print(
            f"[{NAME}] overlap  mean={realized['mean']:.3f} p50={realized['p50']:.3f} "
            f"p90={realized['p90']:.3f}   [target {cfg.build.target_overlap} on the shifted share]"
        )
    print(f"[{NAME}] gender sides: {dict(gender_counts)}")

    reasons = Counter(stats["shift_reason"])
    declined = reasons.get("boundary_not_silent", 0) + reasons.get("no_safe_shift", 0)
    print(f"[{NAME}] shift: {dict(reasons)}")
    if declined:
        print(
            f"[{NAME}] {declined:,} call(s) left unshifted because no wrap point fell in "
            "silence; shifting them would have cut an utterance in half"
        )

    n_enhanced = int(sum(stats["enhanced"]))
    print(f"[{NAME}] enhanced audio used for {n_enhanced:,} / {len(rows):,} calls")
    if rows and 0 < n_enhanced < len(rows):
        # Half the dataset denoised and half not is a domain split the model will
        # happily learn instead of learning to separate. Say so loudly.
        print(
            f"[{NAME}] WARNING: mixed dataset -- {len(rows) - n_enhanced:,} calls were built "
            "from raw audio. Finish `python -m dsd enhance`, then rebuild with --overwrite, "
            "or set build.use_enhanced=never for a uniformly raw dataset."
        )
