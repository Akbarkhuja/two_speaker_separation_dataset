"""Stage 6 -- label each speaker male or female, for balancing.

The dataset should not be 70% men just because the call centre is, so `select`
needs a gender per speaker. Two differences from the notebook:

  - It classified the single longest VAD segment per side. That is a thin basis
    for a label the balancing then acts on, so this sends the K longest
    segments (within a total audio budget) and combines them by
    confidence-weighted vote.

  - The label is resolved **per global speaker**, not per call-side. One agent
    appears in many calls; a majority vote across all of them is far steadier
    than any single call, and per-speaker is the granularity `select` balances
    at anyway. The raw per-side answers are kept as well.

A cluster whose sides disagree on gender is a useful signal that stage 5 merged
two different people, so the disagreement count is reported.
"""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..backends import gender as gender_backends  # noqa: F401  (registers backends)
from ..core.audio import read_audio
from ..core.manifest import read_json, write_failures, write_json
from ..core.rttm import read_rttm
from ..registry import GENDER
from .base import banner, progress, require, skip_if_done

NAME = "gender"
REQUIRES = ("filter", "vad")


def add_args(parser) -> None:
    parser.add_argument("--backend", help=f"one of: {', '.join(GENDER.names())}")
    parser.add_argument("--base-url", help="override the service base URL")
    parser.add_argument("--workers", type=int, help="concurrent requests")
    parser.add_argument("--overwrite", action="store_true", help="re-label everything")
    parser.add_argument("--limit", type=int, help="process only the first N calls")


def run(cfg, args) -> None:
    require({"suitable.json": cfg.paths.suitable_json, "VAD directory": cfg.paths.vad_dir}, NAME)
    if skip_if_done(cfg.paths.gender_json, args.overwrite, NAME):
        return

    backend = args.backend or cfg.gender.backend
    options = dict(cfg.gender.options.get(backend, {}))
    if args.base_url:
        options["base_url"] = args.base_url
    classifier = GENDER.create(backend, options)

    suitable = read_json(cfg.paths.suitable_json)
    calls = sorted(suitable.items())
    if args.limit:
        calls = calls[: args.limit]

    workers = args.workers or cfg.gender.workers
    banner(NAME, f"backend={backend}  {len(calls)} calls, {workers} workers")

    per_side: dict[str, dict] = {}
    failed: list[tuple[str, str]] = []

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(_classify_call, classifier, call, entry, cfg): call
                   for call, entry in calls}
        for future in progress(as_completed(futures), "gender", total=len(futures)):
            call = futures[future]
            try:
                per_side.update(future.result())
            except Exception as exc:
                failed.append((call, repr(exc)))

    aggregate, disagreements = _aggregate_by_speaker(cfg, per_side)

    write_json(
        cfg.paths.gender_json,
        {"per_side": dict(sorted(per_side.items())), "per_speaker": aggregate},
    )
    write_failures(cfg.paths.work_dir / "gender_failed.tsv", failed)

    counts: dict[str, int] = defaultdict(int)
    for record in aggregate.values():
        counts[record["label"]] += 1
    banner(NAME, f"sides labelled: {len(per_side)}  speakers labelled: {len(aggregate)}")
    banner(NAME, f"per speaker: {dict(counts)}")
    if disagreements:
        banner(
            NAME,
            f"{disagreements} clusters mix genders -- possible over-merges in the cluster stage",
        )
    if failed:
        banner(NAME, f"{len(failed)} calls failed; see work/gender_failed.tsv")


# --------------------------------------------------------------------------- #
def _classify_call(classifier, call: str, entry: dict, cfg) -> dict[str, dict]:
    vad_path = cfg.paths.vad_dir / f"{call}.rttm"
    if not vad_path.exists():
        return {}

    segments = read_rttm(vad_path)
    data, sr = read_audio(entry["audio"])

    by_speaker: dict[str, list] = defaultdict(list)
    for segment in segments:
        if segment.duration >= cfg.gender.min_segment_sec:
            by_speaker[segment.speaker].append(segment)

    results: dict[str, dict] = {}
    for speaker, speaker_segments in by_speaker.items():
        channel = int(speaker.rsplit("_", 1)[1]) - 1
        if channel < 0 or channel >= data.shape[1]:
            continue

        # Longest first, then take what fits in the audio budget.
        speaker_segments.sort(key=lambda s: s.duration, reverse=True)
        chosen, budget = [], cfg.gender.max_total_sec
        for segment in speaker_segments[: cfg.gender.segments_per_speaker]:
            if budget <= 0:
                break
            chosen.append(segment)
            budget -= segment.duration
        if not chosen:
            continue

        votes: dict[str, float] = defaultdict(float)
        details = []
        for segment in chosen:
            start, end = int(segment.start * sr), int(segment.end * sr)
            label, confidence = classifier.predict(data[start:end, channel], sr)
            # Weight by confidence AND duration: a confident read of 8 s of
            # speech should outrank a confident read of 1 s.
            votes[label] += confidence * segment.duration
            details.append(
                {"start": segment.start, "dur": round(segment.duration, 3),
                 "label": label, "confidence": confidence}
            )

        label = max(votes, key=votes.__getitem__)
        total = sum(votes.values())
        results[f"{call}_{speaker}"] = {
            "call": call,
            "speaker": speaker,
            "channel": channel,
            "label": label,
            "confidence": round(votes[label] / total, 4) if total else 0.0,
            "segments": details,
        }
    return results


def _aggregate_by_speaker(cfg, per_side: dict[str, dict]) -> tuple[dict, int]:
    """Resolve one label per global speaker id, if the cluster stage has run."""
    if not cfg.paths.speakers_json.exists():
        banner(NAME, "speakers.json not found -- per-side labels only, no aggregate")
        return {}, 0

    by_key = read_json(cfg.paths.speakers_json)["by_key"]

    votes: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    sides: dict[str, int] = defaultdict(int)
    for key, record in per_side.items():
        speaker_id = by_key.get(key)
        if speaker_id:
            votes[speaker_id][record["label"]] += record["confidence"]
            sides[speaker_id] += 1

    aggregate, disagreements = {}, 0
    for speaker_id, tally in votes.items():
        label = max(tally, key=tally.__getitem__)
        total = sum(tally.values())
        if len(tally) > 1:
            disagreements += 1
        aggregate[speaker_id] = {
            "label": label,
            "confidence": round(tally[label] / total, 4) if total else 0.0,
            "sides": sides[speaker_id],
            "mixed": len(tally) > 1,
        }
    return dict(sorted(aggregate.items())), disagreements
