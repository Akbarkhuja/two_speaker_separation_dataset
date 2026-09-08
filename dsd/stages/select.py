"""Stage 7 -- decide which calls get built, and into which split.

Three constraints, applied in order, each recorded in `work/selection.json` so
every drop can be explained afterwards.

1. **Cap per speaker.** A call-centre corpus has a handful of agents opposite
   thousands of one-off customers, so without a cap a few voices would carry
   most of the training set and the model would learn them instead of learning
   to separate. A call consumes a slot from *both* its speakers, so the cap is
   applied greedily, best calls first.

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
from .base import banner, require, skip_if_done

NAME = "select"
REQUIRES = ("filter", "cluster")


def add_args(parser) -> None:
    parser.add_argument("--max-calls-per-speaker", type=int)
    parser.add_argument("--no-balance-gender", action="store_true", help="skip gender balancing")
    parser.add_argument("--overwrite", action="store_true", help="reselect")


def run(cfg, args) -> None:
    require(
        {"suitable.json": cfg.paths.suitable_json, "speakers.json": cfg.paths.speakers_json},
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

    max_per_speaker = args.max_calls_per_speaker or cfg.select.max_calls_per_speaker
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
                "quality": round(quality, 3),
                "duration": entry["duration"],
            }
        )

    banner(NAME, f"{len(suitable)} suitable -> {len(candidates)} candidates")

    # ---------------------------------------------------------------- #
    # 1. cap calls per speaker, best calls first
    # ---------------------------------------------------------------- #
    rng.shuffle(candidates)  # break quality ties without a positional bias
    candidates.sort(key=lambda c: -c["quality"])

    used: dict[str, int] = defaultdict(int)
    capped = []
    for record in candidates:
        if any(used[sid] >= max_per_speaker for sid in record["speakers"]):
            dropped["speaker_cap"] += 1
            continue
        for sid in record["speakers"]:
            used[sid] += 1
        capped.append(record)

    banner(NAME, f"after cap of {max_per_speaker}/speaker: {len(capped)} calls")

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

    write_json(
        cfg.paths.selection_json,
        {
            "seed": cfg.seed,
            "max_calls_per_speaker": max_per_speaker,
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
    banner(NAME, f"unique speakers used: {len(used)}")


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
