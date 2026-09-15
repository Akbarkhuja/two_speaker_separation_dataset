"""Stage 7 -- decide which calls get built, and into which split.

Three constraints, applied in order, each recorded in `work/selection.json` so
every drop can be explained afterwards.

1. **Cap per speaker.** A call-centre corpus has a handful of agents opposite
   thousands of one-off customers, so without a cap a few voices would carry
   most of the training set and the model would learn them instead of learning
   to separate. A call draws on the budget of *both* its speakers, so the cap is
   applied greedily, best calls first.

   The budget is **speech time**, not call count. What a model hears of a voice
   is seconds, and per-side speech in one call ranges from 5 s to ~14 min, so
   "3 calls" can be fifteen seconds of one person or forty minutes of another.
   Compared at matched dataset size on the real corpus (~40 h), a 20-minute
   duration cap kept 23% more calls and 15% more unique speakers than the
   equivalent call cap, held the loudest voice at exactly 20.0 min instead of
   31.5, and cut the top-10 speakers' share from 10.8% to 7.5%.

   Seconds are counted as the speaker's own VAD speech minus the `excluded`
   spans -- the audio of that voice that actually lands in s1/s2. Call
   wall-clock ran 2.6x-8.3x a side's real speech, and the embedding stage's
   `speech_sec` undercounts it by 15-39%, so either would make the configured
   number mean something other than what it says. A call-count cap remains
   available as a secondary guard.

2. **Gender balance.** Only same-gender calls can move the ratio -- a
   male/female call contributes one side to each -- so the balancing drops the
   weakest same-gender calls of the majority class. When it runs out of those it
   stops and reports what it actually achieved rather than deleting good data to
   chase a number.

3. **Speaker-disjoint splits.** Calls are edges between two speakers; whole
   connected components of that graph go to one split. Splitting call-by-call
   would put the same voice in train and test, which is the standard way a
   separation benchmark ends up flattering itself.
"""

from __future__ import annotations

import random
from collections import defaultdict

from ..core.manifest import read_json, write_json
from ..core.rttm import read_rttm, subtract_intervals
from .base import banner, require, skip_if_done

NAME = "select"
REQUIRES = ("filter", "vad", "cluster")


def add_args(parser) -> None:
    parser.add_argument(
        "--max-duration-per-speaker",
        type=float,
        metavar="SECONDS",
        help="cap on one speaker's own speech across the selection; 0 turns it off",
    )
    parser.add_argument(
        "--max-calls-per-speaker",
        type=int,
        help="cap on one speaker's appearances; 0 turns it off",
    )
    parser.add_argument("--no-balance-gender", action="store_true", help="skip gender balancing")
    parser.add_argument("--overwrite", action="store_true", help="reselect")


def _enabled(value):
    """A cap of None or <= 0 is off."""
    return value if value is not None and value > 0 else None


def run(cfg, args) -> None:
    require(
        {
            "suitable.json": cfg.paths.suitable_json,
            "speakers.json": cfg.paths.speakers_json,
            "VAD directory": cfg.paths.vad_dir,
        },
        NAME,
    )
    if skip_if_done(cfg.paths.selection_json, args.overwrite, NAME):
        return

    rng = random.Random(cfg.seed)
    suitable = read_json(cfg.paths.suitable_json)
    by_key = read_json(cfg.paths.speakers_json)["by_key"]

    speech_by_key: dict[str, float] = {}
    if cfg.paths.embed_index_json.exists():
        speech_by_key = {
            key: record["speech_sec"]
            for key, record in read_json(cfg.paths.embed_index_json)["index"].items()
        }

    gender_by_speaker: dict[str, str] = {}
    if cfg.paths.gender_json.exists():
        gender_by_speaker = {
            sid: record["label"]
            for sid, record in read_json(cfg.paths.gender_json)["per_speaker"].items()
        }
    elif cfg.select.balance_gender and not args.no_balance_gender:
        banner(NAME, "gender.json not found -- balancing skipped")

    # `is not None`, not `or`: with `or`, passing 0 on the command line falls
    # through to the config value, so a cap could never be switched off.
    max_duration = _enabled(
        args.max_duration_per_speaker
        if args.max_duration_per_speaker is not None
        else cfg.select.max_duration_per_speaker
    )
    max_calls = _enabled(
        args.max_calls_per_speaker
        if args.max_calls_per_speaker is not None
        else cfg.select.max_calls_per_speaker
    )
    dropped: dict[str, int] = defaultdict(int)

    # ---------------------------------------------------------------- #
    # 0. build one record per call
    # ---------------------------------------------------------------- #
    candidates = []
    for call, entry in sorted(suitable.items()):
        channels = sorted(int(c) for c in entry["channels"])
        # The VAD stage names a channel's speaker `speaker_{channel + 1}`, and
        # every downstream key is built from that, so it is derivable here.
        keys = [f"{call}_speaker_{c + 1}" for c in channels]
        speaker_ids = [by_key.get(key) for key in keys]
        if any(sid is None for sid in speaker_ids):
            dropped["no_speaker_id"] += 1
            continue
        if len(set(speaker_ids)) != len(speaker_ids):
            # Both sides clustered to one identity: either the same person on
            # both lines, or a cluster-stage merge that violated cannot-link.
            dropped["same_speaker_both_sides"] += 1
            continue

        # Prefer the VAD-derived seconds the embed stage measured. Falling back
        # to the filter's numbers means changing key space: the filter counts
        # speech per *diarizer* label (`ch0_speaker_0`) while `keys` are VAD
        # labels, so the lookup has to go through `channels`, which is the
        # channel -> diarizer-label map the filter writes for exactly this.
        # Indexing `speech_sec` with a VAD key can never hit, and silently
        # yields 0.0 -- which then reads as `too_little_speech` rather than as
        # the missing embedding it actually is.
        speech = []
        for channel, key in zip(channels, keys):
            if key in speech_by_key:
                speech.append(speech_by_key[key])
            else:
                label = entry["channels"][str(channel)]
                speech.append(entry["speech_sec"].get(label, 0.0))

        quality = min(speech) if speech else 0.0
        if quality < cfg.select.min_speech_sec:
            dropped["too_little_speech"] += 1
            continue

        # What the duration cap charges: the speech of each voice that will
        # actually reach s1/s2. A missing VAD file is a drop, not zero seconds --
        # zero would let the call slip past any cap for free.
        source_speech = _source_speech(cfg.paths.vad_dir / f"{call}.rttm", channels, entry)
        if source_speech is None:
            dropped["no_vad_rttm"] += 1
            continue

        candidates.append(
            {
                "call": call,
                "audio": entry["audio"],
                "channels": channels,
                "keys": keys,
                # Carried through to `build`, which cuts this time out of the
                # sources so a minor label's voice never reaches a wav.
                "excluded": entry.get("excluded", {}),
                "speakers": speaker_ids,
                "genders": [gender_by_speaker.get(sid) for sid in speaker_ids],
                "speech_sec": [round(s, 3) for s in speech],
                # Per side, in `channels` order; this is what the duration cap
                # and `verify` count against.
                "source_speech_sec": [round(s, 3) for s in source_speech],
                "quality": round(quality, 3),
                "duration": entry["duration"],
            }
        )

    banner(NAME, f"{len(suitable)} suitable -> {len(candidates)} candidates")

    # ---------------------------------------------------------------- #
    # 1. per-speaker caps, best calls first
    # ---------------------------------------------------------------- #
    rng.shuffle(candidates)  # break quality ties without a positional bias
    candidates.sort(key=lambda c: -c["quality"])

    capped, cap_drops, _calls, _seconds = apply_caps(candidates, max_calls, max_duration)
    for reason, count in cap_drops.items():
        dropped[reason] += count

    limits = []
    if max_duration is not None:
        limits.append(f"{max_duration / 60:g} min of speech")
    if max_calls is not None:
        limits.append(f"{max_calls} calls")
    banner(
        NAME,
        f"after cap of {' and '.join(limits) or 'nothing (caps off)'} per speaker: "
        f"{len(capped)} calls",
    )

    # ---------------------------------------------------------------- #
    # 2. gender balance
    # ---------------------------------------------------------------- #
    balance = cfg.select.balance_gender and not args.no_balance_gender and bool(gender_by_speaker)
    if balance:
        capped, gender_drops, ratio = _balance_gender(capped, cfg.select.gender_tolerance)
        dropped["gender_balance"] += gender_drops
        banner(NAME, f"after gender balance: {len(capped)} calls, male side share {ratio:.3f}")

    if not capped:
        raise SystemExit("nothing selected; loosen the caps or check the earlier stages")

    # ---------------------------------------------------------------- #
    # 3. speaker-disjoint splits
    # ---------------------------------------------------------------- #
    assignment = _split_by_component(capped, cfg.select.splits, rng)
    for record in capped:
        record["split"] = assignment[record["call"]]

    selection = {record["call"]: record for record in sorted(capped, key=lambda r: r["call"])}
    per_split: dict[str, int] = defaultdict(int)
    hours: dict[str, float] = defaultdict(float)
    for record in capped:
        per_split[record["split"]] += 1
        hours[record["split"]] += record["duration"] / 3600.0

    # Exposure is summarised from the final selection, after gender balancing has
    # dropped its calls -- the numbers apply_caps returned are from before that.
    exposure = _exposure(capped)

    write_json(
        cfg.paths.selection_json,
        {
            "seed": cfg.seed,
            # The caps actually used, which may differ from config when set on the
            # command line. `verify` checks against these, not against config.
            "max_duration_per_speaker": max_duration,
            "max_calls_per_speaker": max_calls,
            "exposure": exposure,
            "dropped": dict(dropped),
            "counts": dict(per_split),
            "hours": {k: round(v, 2) for k, v in hours.items()},
            "calls": selection,
        },
    )

    banner(NAME, "dropped:")
    for reason, count in sorted(dropped.items(), key=lambda kv: -kv[1]):
        print(f"[{NAME}]   {reason:<24} {count:>6}")
    banner(NAME, "splits:")
    for split in cfg.select.splits:
        print(
            f"[{NAME}]   {split:<6} {per_split.get(split, 0):>6} calls  "
            f"{hours.get(split, 0.0):>7.2f} h"
        )
    banner(
        NAME,
        f"unique speakers used: {exposure['speakers']}   loudest voice "
        f"{exposure['max_sec'] / 60:.1f} min   p99 {exposure['p99_sec'] / 60:.1f} min   "
        f"top-10 share {exposure['top10_share']:.1%}",
    )


# --------------------------------------------------------------------------- #
def _source_speech(vad_path, channels: list[int], entry: dict) -> list[float] | None:
    """Seconds of each side's own speech that will reach s1/s2.

    VAD speech with the `excluded` minor-label spans subtracted, because `build`
    removes those spans too. `subtract_intervals` merges as it goes, so
    overlapping VAD segments are not double-counted. Returns None when the VAD
    RTTM is missing.
    """
    if not vad_path.exists():
        return None

    spans: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for segment in read_rttm(vad_path):
        spans[segment.channel - 1].append((segment.start, segment.end))

    excluded = entry.get("excluded", {})
    seconds = []
    for channel in channels:
        cut = [tuple(span) for span in excluded.get(str(channel), [])]
        kept = subtract_intervals(spans.get(channel, []), cut)
        seconds.append(sum(end - start for start, end in kept))
    return seconds


def apply_caps(
    candidates: list[dict],
    max_calls: int | None,
    max_duration: float | None,
) -> tuple[list[dict], dict[str, int], dict[str, int], dict[str, float]]:
    """Admit calls in the given order while every speaker stays within budget.

    Returns (kept, dropped_reasons, calls_by_speaker, seconds_by_speaker).
    A cap of None is off. Each candidate needs `speakers` and, when the duration
    cap is on, `source_speech_sec` in the same order.

    Admission is **strict**: a call is kept only if it leaves both speakers
    within the duration budget. Admitting while a speaker is merely *under*
    budget lets the last call overshoot by its full length -- simulated on the
    real corpus, a 5-minute cap let the loudest voice reach 10.5 minutes.

    Three drop reasons, deliberately distinct so a selection explains itself:

      longer_than_duration_cap  one side is longer than the cap on its own, so
                                no budget could ever admit it
      speaker_cap               a speaker has used up their calls
      speaker_duration_cap      admitting would push a speaker past their time
    """
    kept: list[dict] = []
    dropped: dict[str, int] = defaultdict(int)
    calls: dict[str, int] = {}
    seconds: dict[str, float] = {}

    for record in candidates:
        sides = record.get("source_speech_sec") or [0.0] * len(record["speakers"])
        reason = None
        for speaker, side in zip(record["speakers"], sides):
            if max_duration is not None and side > max_duration:
                reason = "longer_than_duration_cap"
                break
            # `.get`, never `defaultdict[...]`: a lookup here must not register a
            # speaker who ends up with nothing admitted.
            if max_calls is not None and calls.get(speaker, 0) >= max_calls:
                reason = "speaker_cap"
                break
            if max_duration is not None and seconds.get(speaker, 0.0) + side > max_duration:
                reason = "speaker_duration_cap"
                break

        if reason is not None:
            dropped[reason] += 1
            continue

        for speaker, side in zip(record["speakers"], sides):
            calls[speaker] = calls.get(speaker, 0) + 1
            seconds[speaker] = seconds.get(speaker, 0.0) + side
        kept.append(record)

    return kept, dict(dropped), calls, seconds


def _exposure(records: list[dict]) -> dict:
    """How concentrated the selected speech is on its loudest voices."""
    seconds: dict[str, float] = defaultdict(float)
    for record in records:
        sides = record.get("source_speech_sec") or record.get("speech_sec") or []
        for speaker, side in zip(record["speakers"], sides):
            seconds[speaker] += side

    values = sorted(seconds.values(), reverse=True)
    total = sum(values)
    if not values:
        return {"speakers": 0, "max_sec": 0.0, "p99_sec": 0.0, "top10_share": 0.0, "hours": 0.0}

    p99_index = min(len(values) - 1, int(round(0.01 * (len(values) - 1))))
    return {
        "speakers": len(values),
        "max_sec": round(values[0], 3),
        "p99_sec": round(values[p99_index], 3),
        "top10_share": round(sum(values[:10]) / total, 4) if total else 0.0,
        "hours": round(total / 3600.0, 3),
    }


# --------------------------------------------------------------------------- #
def _balance_gender(records: list[dict], tolerance: float) -> tuple[list[dict], int, float]:
    """Drop weakest same-gender calls of the majority class until balanced."""
    def sides(rows):
        counts = defaultdict(int)
        for row in rows:
            for gender in row["genders"]:
                if gender:
                    counts[gender] += 1
        return counts

    kept = list(records)
    drops = 0
    while True:
        counts = sides(kept)
        male, female = counts.get("male", 0), counts.get("female", 0)
        total = male + female
        if total == 0:
            return kept, drops, 0.0
        share = male / total
        if abs(share - 0.5) <= tolerance:
            return kept, drops, share

        majority = "male" if share > 0.5 else "female"
        # Only a call whose *both* sides are the majority gender moves the
        # ratio; dropping a mixed call changes nothing.
        movable = [r for r in kept if all(g == majority for g in r["genders"])]
        if not movable:
            return kept, drops, share

        worst = min(movable, key=lambda r: r["quality"])
        kept.remove(worst)
        drops += 1


def _split_by_component(records: list[dict], targets: dict[str, float], rng) -> dict[str, str]:
    """Assign whole connected components of the speaker graph to splits."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for record in records:
        first, *rest = record["speakers"]
        for other in rest:
            union(first, other)

    components: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        components[find(record["speakers"][0])].append(record)

    # Largest components first: placing the big ones while every split is still
    # empty is what keeps the final shares near target.
    ordered = sorted(components.values(), key=lambda rows: -len(rows))
    # Jitter the long tail of equal-sized (mostly singleton) components so the
    # splits are not decided by call-id order. Shuffling a slice in place would
    # be a no-op -- a slice is a copy -- so it is reassigned.
    cut = len(ordered) // 2
    tail = ordered[cut:]
    rng.shuffle(tail)
    ordered = ordered[:cut] + tail

    total = len(records)
    want = {split: share * total for split, share in targets.items()}
    have = {split: 0 for split in targets}

    assignment: dict[str, str] = {}
    for rows in ordered:
        # Whichever split is furthest below its target, in absolute calls.
        split = max(targets, key=lambda s: want[s] - have[s])
        have[split] += len(rows)
        for record in rows:
            assignment[record["call"]] = split
    return assignment
