"""Stage 8 -- write the mixtures and their two ground-truth sources.

Per selected call, at the source's native 8 kHz:

    cut the excluded (possible third voice) spans from both channels
    optionally roll channel 2 to synthesize speech overlap
    scale channel 2 to a sampled SIR
    mix = channel 1 + channel 2, as recorded          -> mix.wav
    zerofy each channel against its own VAD mask      -> s1.wav, s2.wav
    one shared peak guard over all three

The order matters and it changed: the mixture is formed **before** zerofying,
not after. The model's input therefore keeps the noise floor, the room and the
breath that the two microphones actually picked up between the turns, while its
targets hold only that speaker's speech. Summing two already-zerofied channels
gave a mixture that was digitally silent whenever nobody was talking -- a signal
that does not occur in any real call, and one a model quickly learns to key on.

The cost is the exact identity. `mix - (s1 + s2)` is now precisely that
background, so it is not zero; `build.zerofy_mix=true` restores the old
behaviour for anyone who needs `mix == s1 + s2` sample-for-sample. `dsd verify`
knows which mode a dataset was built in and checks the matching invariant:
under `zerofy_mix` the full identity, otherwise equality wherever both speakers
are at full gain plus a bound on how loud the background is allowed to be.

The output for one call is `<split>/<call>/{mix,s1,s2}.wav` plus a `meta.json`
recording the speaker ids, genders, realized overlap, SIR, shift and background
level -- enough to reconstruct exactly what was done without re-reading the
source.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ..augment import chain as augment_chain
from ..core.audio import active_rms, demux, read_audio, speech_mask, write_wav
from ..core.manifest import read_json, write_failures, write_json, write_jsonl
from ..core.rttm import read_rttm, subtract_intervals
from ..mixing.chunker import is_usable, windows
from ..mixing.mixer import guard_peaks, scale_to_sir
from ..mixing.overlap import overlap_ratio, roll, shift_for_target
from ..mixing.zerofy import apply_mask, mute
from .augment import load_banks
from .base import banner, progress, require
from .enhance import enhanced_path

NAME = "build"
REQUIRES = ("select", "vad")


def add_args(parser) -> None:
    parser.add_argument("--chunks", action="store_true", help="also write fixed-length windows")
    parser.add_argument("--overwrite", action="store_true", help="rebuild calls already written")
    parser.add_argument("--limit", type=int, help="build only the first N calls")
    parser.add_argument(
        "--variants",
        type=int,
        help="degraded copies of the mixture to write per call (0 for none)",
    )
    parser.add_argument(
        "--no-prune",
        action="store_true",
        help="keep call directories the selection no longer holds",
    )
    shuffle = parser.add_mutually_exclusive_group()
    shuffle.add_argument(
        "--shuffle",
        dest="shuffle",
        action="store_const",
        const=True,
        help="roll s2 in time to synthesize speech overlap for this run",
    )
    shuffle.add_argument(
        "--no-shuffle",
        dest="shuffle",
        action="store_const",
        const=False,
        help="keep every call's recorded timing for this run",
    )


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
    # None means "not given on the command line", so config decides. With `or`,
    # a config value of true could never be switched off for a single run.
    flag = getattr(args, "shuffle", None)
    want_shuffle = cfg.build.shuffle if flag is None else flag
    # Same rule, and for the same reason: `or` would swallow --variants 0.
    chosen = getattr(args, "variants", None)
    variants = cfg.build.augment.variants if chosen is None else chosen
    variants = max(0, int(variants))
    dataset_dir = cfg.paths.dataset_dir
    banner(
        NAME,
        f"{len(calls)} calls -> {dataset_dir}  "
        f"(chunks={want_chunks}, shuffle={want_shuffle}, variants={variants})",
    )
    if want_shuffle:
        banner(
            NAME,
            f"overlap boosting ON: {1.0 - cfg.build.natural_frac:.0%} of calls get s2 rolled "
            f"into {tuple(cfg.build.target_overlap)}",
        )
    else:
        banner(NAME, "overlap boosting OFF: every call keeps its recorded timing (--shuffle to enable)")
    if cfg.build.zerofy_mix:
        banner(NAME, "mixture = zerofied s1 + s2 exactly (build.zerofy_mix=true)")
    else:
        banner(NAME, "mixture = the two channels as recorded; only s1/s2 are zerofied")

    # The degradation banks are opened once for the whole run: they cache the
    # impulse responses and noise clips they have read, and rebuilding them per
    # call would re-read the same few hundred files thousands of times.
    banks, augment_fingerprint = None, None
    if variants:
        banks = load_banks(cfg)
        augment_fingerprint = _augment_fingerprint(cfg)
        banner(
            NAME,
            f"augmentation ON: {variants} degraded mixture(s) per call from "
            f"{len(banks.rirs) if banks.rirs else 0} rooms, "
            f"{len(banks.noise) if banks.noise else 0} noise clips, "
            f"codecs {banks.codecs or 'none'}; each degradation fires at "
            f"p={cfg.build.augment.prob} per track",
        )

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
        meta = read_json(meta_path) if meta_path.exists() and not args.overwrite else None
        # A call written under a different --shuffle or build.zerofy_mix setting
        # is not reusable: keeping it would leave one dataset holding calls
        # built two different ways, which is a domain split a model will learn
        # instead of learning to separate. meta.json files written before these
        # keys existed carry neither and are rebuilt once, in whichever mode is
        # asked for.
        if meta is not None and (
            meta.get("shuffle") != want_shuffle
            or meta.get("zerofy_mix") != cfg.build.zerofy_mix
            # A call built when the mixture came from the enhanced audio, or with
            # raw targets while this run wants denoised ones, is a different
            # dataset. `targets_enhanced` is absent from every meta written
            # before the mixture was split off the targets, so those rebuild too.
            or "targets_enhanced" not in meta
            or meta["targets_enhanced"] != _target_audio(call, record, cfg)[1]
            # A call carrying a different number of degraded variants, or ones
            # drawn under different settings, would leave the dataset holding two
            # augmentation regimes at once. The fingerprint covers every knob
            # under build.augment, so changing an SNR range rebuilds too.
            or meta.get("augment", {}).get("variants", 0) != variants
            or (variants and meta.get("augment", {}).get("fingerprint") != augment_fingerprint)
        ):
            meta = None
        if meta is None:
            try:
                # Seeded per call, so rebuilding one call reproduces exactly what
                # the full run made -- a global rng would not survive --limit.
                rng = random.Random(f"{cfg.seed}:{call}")
                meta = _build_call(
                    call, record, cfg, out_dir, rng, want_shuffle,
                    variants=variants, banks=banks, fingerprint=augment_fingerprint,
                )
            except Exception as exc:
                failed.append((call, repr(exc)))
                continue

        rows.extend(_manifest_rows(meta))
        stats["overlap"].append(meta["overlap"])
        stats["sir_db"].append(meta["sir_db"])
        stats["duration"].append(meta["duration"])
        stats["enhanced"].append(bool(meta.get("enhanced", False)))
        stats["shift_reason"].append(meta.get("shift_reason", "unknown"))
        if meta.get("background_snr_db") is not None:
            stats["background_snr_db"].append(meta["background_snr_db"])
        stats[f"hours_{record['split']}"].append(meta["duration"] / 3600.0)

        # What the degradation chain actually drew, per track. Collected here so
        # `stats.json` can show the realized distributions next to the configured
        # ranges, the same way it already does for overlap and SIR.
        for item in meta.get("augment", {}).get("items", []):
            for recipe in item["recipes"]:
                stats["tracks"].append(1)
                for step in augment_chain.STEPS:
                    if recipe.get(step) is not None:
                        stats["degradation"].append(step)
                if recipe.get("noise"):
                    stats["noise_snr_db"].append(recipe["noise"]["snr_db"])
                if recipe.get("codec"):
                    stats["codec"].append(recipe["codec"]["kind"])
                if recipe.get("band_limit"):
                    stats["cutoff_hz"].append(recipe["band_limit"]["cutoff_hz"])
                if recipe.get("reverb"):
                    stats["reverb"].append(recipe["reverb"]["kind"])

        if want_chunks:
            try:
                chunk_rows.extend(_build_chunks(call, record, meta, cfg, dataset_dir))
            except Exception as exc:
                failed.append((f"{call}:chunks", repr(exc)))

    pruned = 0
    if not getattr(args, "no_prune", False):
        pruned = prune_stale_calls(dataset_dir, set(selection["calls"]))

    write_jsonl(dataset_dir / "manifest.jsonl", rows)
    call_counts = write_split_manifests(dataset_dir, rows, cfg.select.splits)
    chunk_counts = {}
    if want_chunks:
        write_jsonl(dataset_dir / "chunks" / "manifest.jsonl", chunk_rows)
        chunk_counts = write_split_manifests(
            dataset_dir / "chunks", chunk_rows, cfg.select.splits
        )
    write_failures(cfg.paths.work_dir / "build_failed.tsv", failed)
    _write_stats(
        cfg, dataset_dir, selection, rows, chunk_rows, stats, chunk_counts,
        want_shuffle, variants,
    )

    banner(NAME, "per-split manifests:")
    for split in sorted(call_counts):
        chunks_here = f"  {chunk_counts[split]:>7,} chunks" if chunk_counts else ""
        print(f"[{NAME}]   {split}.jsonl  {call_counts[split]:>6,} rows{chunks_here}")

    if pruned:
        banner(NAME, f"pruned {pruned:,} call directories the selection no longer holds")

    built = len({row["call"] for row in rows})
    detail = f"wrote {built} calls as {len(rows)} rows" if variants else f"wrote {built} calls"
    banner(NAME, detail + (f", {len(chunk_rows)} chunks" if want_chunks else ""))
    if failed:
        banner(NAME, f"{len(failed)} failed; see work/build_failed.tsv")


# --------------------------------------------------------------------------- #
def prune_stale_calls(dataset_dir: Path, selected: set[str]) -> int:
    """Delete call directories the current selection no longer holds.

    Keyed on the **selection**, never on the rows this run happened to build, so
    a `--limit` run cannot delete the calls it simply skipped.

    Left alone, the tree and the manifest drift apart: a `--limit 30` build over
    a 2513-call selection leaves 2483 directories that no manifest mentions, and
    anything globbing `dataset/train/*/mix.wav` rather than reading the manifest
    picks all of them up -- including calls written under an older mixture
    definition entirely.
    """
    import shutil

    removed = 0
    for split_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        if split_dir.name == "chunks":
            continue
        for call_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            if call_dir.name not in selected:
                shutil.rmtree(call_dir, ignore_errors=True)
                removed += 1
    return removed


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


def _target_audio(call: str, record: dict, cfg) -> tuple[str, bool]:
    """Where the *targets* come from: the enhanced cache, or the original.

    The mixture is never taken from here -- it is always the recording as it was
    made (see `_build_call`). Only `s1`/`s2` may be denoised, which is what asks
    the model to separate *and* clean rather than to separate something already
    clean.

    The enhanced file has the same rate and the same sample count as the
    original -- the enhance stage refuses anything else -- so the two can be read
    side by side and every VAD offset stays valid for both.
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


def _taper_seam(signal: np.ndarray, shift: int, sr: int, fade_ms: float) -> np.ndarray:
    """Fade the one join a circular roll creates inside the file.

    `np.roll(x, k)` puts original sample `x[n-1]` next to `x[0]` at output index
    `k`. When the channel was zerofied first both sides of that join were exact
    zeros and it did not matter. On a raw channel they are two unrelated
    samples of noise floor, so the join is a step -- small, but a step at a
    fixed offset is exactly the kind of artifact a model finds before it finds
    the voices. `shift_for_target` already guarantees the seam sits inside
    `seam_guard_ms` of silence, so a few milliseconds of taper there costs no
    speech at all.
    """
    n = len(signal)
    if not shift or n == 0:
        return signal
    fade = max(2, int(round(sr * fade_ms / 1000.0)))
    keep = np.ones(n, dtype=bool)
    keep[np.arange(shift % n - fade, shift % n + fade) % n] = False
    return apply_mask(signal, keep, sr, fade_ms)


def _background(
    mixture: np.ndarray,
    s1: np.ndarray,
    s2: np.ndarray,
    active: np.ndarray,
) -> tuple[float, float | None]:
    """(rms of mix - (s1 + s2), how far the two voices sit above it in dB).

    Returns `(0.0, None)` when the mixture is the exact sum of the sources,
    which is what `zerofy_mix` produces.
    """
    residual = np.asarray(mixture, dtype=np.float64) - (
        np.asarray(s1, dtype=np.float64) + np.asarray(s2, dtype=np.float64)
    )
    if residual.size == 0:
        return 0.0, None
    rms = float(np.sqrt(np.mean(np.square(residual))))
    if rms <= 0.0:
        return 0.0, None
    voices = active_rms(s1 + s2, active)
    if voices <= 0.0:
        return rms, None
    return rms, float(20.0 * np.log10(voices / rms))


def _augment_fingerprint(cfg) -> str:
    """A short hash of every `build.augment` knob except the variant count.

    The count is compared on its own, so this answers the other half of "is a
    cached call still valid": a change to an SNR range or a codec list has to
    rebuild, and a change to nothing at all must not.
    """
    import dataclasses

    settings = dataclasses.asdict(cfg.build.augment)
    settings.pop("variants", None)
    payload = json.dumps(settings, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _degrade_variant(
    mix1: np.ndarray,
    mix2: np.ndarray,
    mask1: np.ndarray,
    mask2: np.ndarray,
    cfg,
    banks,
    rng: random.Random,
) -> tuple[np.ndarray, list[dict], float]:
    """One degraded mixture: degrade each channel on its own, then sum.

    The two channels get **independent** draws. That is the part of the method
    that matters: degrading the finished mixture would give both speakers one
    shared room, one shared noise recording and one shared codec, which is a
    much weaker task and not what either the paper or this corpus describes --
    our two channels are two telephone legs, each with its own line and its own
    handset.

    The speech mask goes in with each channel so the noise step can measure that
    speaker's level over their speech rather than over a mostly-silent track.
    """
    acfg = cfg.build.augment
    recipes, degraded = [], []
    for channel, mask in ((mix1, mask1), (mix2, mask2)):
        recipe = augment_chain.sample(rng, acfg, banks)
        degraded.append(augment_chain.apply(channel, recipe, cfg.sample_rate, banks, mask))
        recipes.append(recipe)

    mixture = (degraded[0] + degraded[1]).astype(np.float32)

    # The degraded mixture is peak-guarded alone. It cannot share a gain with
    # s1/s2 the way the clean mixture does: the targets are shared by every
    # variant, so scaling them here would change them for all the others. That
    # is only sound because a degraded mixture is not claimed to be the sum of
    # its targets anyway -- the gain is recorded so a trainer can undo it.
    peak = float(np.abs(mixture).max())
    gain = 1.0
    if peak > cfg.build.peak_ceiling:
        gain = cfg.build.peak_ceiling / peak
        mixture = (mixture * gain).astype(np.float32)

    return mixture, [r.to_dict() for r in recipes], gain


def _build_call(
    call: str,
    record: dict,
    cfg,
    out_dir: Path,
    rng: random.Random,
    shuffle: bool = False,
    variants: int = 0,
    banks=None,
    fingerprint: str | None = None,
) -> dict:
    mix_path = record["audio"]
    target_path, targets_enhanced = _target_audio(call, record, cfg)

    # The mixture always comes from the recording as it was made. The targets may
    # be the denoised copy. Reading both is the whole point: a single file would
    # force the model's input and its answer to share a provenance, and enhancing
    # the input throws away the noise the model exists to cope with.
    mix_data, sr = read_audio(mix_path)
    if targets_enhanced:
        target_data, target_sr = read_audio(target_path)
        if target_sr != sr or target_data.shape != mix_data.shape:
            # Every VAD offset below indexes both arrays. A quiet mismatch here
            # would shift one of them against the labels for the whole call.
            raise ValueError(
                f"enhanced audio does not match the original: "
                f"{target_data.shape} @ {target_sr} Hz vs {mix_data.shape} @ {sr} Hz"
            )
    else:
        target_data = mix_data

    segments = read_rttm(cfg.paths.vad_dir / f"{call}.rttm")
    fade_ms = cfg.build.fade_ms

    intervals: dict[int, list] = defaultdict(list)
    for segment in segments:
        intervals[segment.channel - 1].append((segment.start, segment.end))

    # Time the filter attributed to a minor diarizer label. That label was
    # judged too small to reject the call over, but if it really was a brief
    # third voice this is what keeps it out. It is cut from the *channel*, not
    # just from the VAD mask, so it leaves the mixture as well as the targets:
    # an unlabelled voice in the model's input with no target to match is worse
    # than the short gap that removing it leaves behind.
    excluded = {
        int(channel): [tuple(span) for span in spans]
        for channel, spans in record.get("excluded", {}).items()
    }
    for channel, spans in excluded.items():
        intervals[channel] = subtract_intervals(intervals[channel], spans)

    channels = record["channels"]
    n_samples = mix_data.shape[0]

    # Two copies of each channel: `mix*` as recorded, `tgt*` possibly denoised.
    # When enhancement is off they are the same array, and everything below
    # collapses to the single-signal case.
    mix1 = mute(demux(mix_data, channels[0]), excluded.get(channels[0], []), sr, fade_ms)
    mix2 = mute(demux(mix_data, channels[1]), excluded.get(channels[1], []), sr, fade_ms)
    if targets_enhanced:
        tgt1 = mute(demux(target_data, channels[0]), excluded.get(channels[0], []), sr, fade_ms)
        tgt2 = mute(demux(target_data, channels[1]), excluded.get(channels[1], []), sr, fade_ms)
    else:
        tgt1, tgt2 = mix1, mix2

    mask1 = speech_mask(intervals[channels[0]], n_samples, sr)
    mask2 = speech_mask(intervals[channels[1]], n_samples, sr)

    natural = overlap_ratio(mask1, mask2)

    # Rolling s2 is what manufactures overlap, and it is also what makes a call
    # stop sounding like a recorded conversation: the roll moves speech across
    # the file, so turns land in places the two people never actually spoke.
    # Opt in with `--shuffle` (or build.shuffle) when a model needs more than
    # the corpus's ~3% natural overlap; otherwise the timing is left alone.
    shift, shift_reason = 0, ("natural" if shuffle else "shuffle_off")
    if shuffle and rng.random() >= cfg.build.natural_frac:
        shift, _predicted, shift_reason = shift_for_target(
            mask1,
            mask2,
            sr,
            tuple(cfg.build.target_overlap),
            rng,
            guard_sec=cfg.build.seam_guard_ms / 1000.0,
        )
        # The whole channel rolls, background included -- and *both* copies roll
        # by the same amount, or the mixture and the target stop describing the
        # same moment of the same call.
        mix2 = _taper_seam(roll(mix2, shift), shift, sr, fade_ms)
        if targets_enhanced:
            tgt2 = _taper_seam(roll(tgt2, shift), shift, sr, fade_ms)
        else:
            tgt2 = mix2
        mask2 = roll(mask2, shift)

    # Level the whole channel, not only its speech: these samples go into the
    # mixture and into s2, so one gain has to cover both. The gain is measured on
    # the targets -- that is the speech level the model has to reproduce -- and
    # then multiplied into the mixture copy unchanged.
    sir_db = rng.uniform(*cfg.build.sir_db)
    tgt2, sir_gain = scale_to_sir(tgt1, tgt2, mask1, mask2, sir_db)
    mix2 = tgt2 if not targets_enhanced else (mix2 * sir_gain).astype(np.float32)

    # Targets: each speaker's own channel with everything outside their speech
    # faded away. The mixture keeps that material, which is what asks the model
    # to pull two clean voices out of a real recording rather than out of a
    # sum of two already-clean ones.
    s1 = apply_mask(tgt1, mask1, sr, fade_ms)
    s2 = apply_mask(tgt2, mask2, sr, fade_ms)

    # `zerofy_mix` buys the exact identity, and the only way to have it is for the
    # mixture to be the sum of the targets -- so that mode alone does not use the
    # original channels.
    mixture = (s1 + s2) if cfg.build.zerofy_mix else (mix1 + mix2)
    mixture, s1, s2, peak_gain = guard_peaks(mixture, s1, s2, cfg.build.peak_ceiling)

    # What the mixture carries that no target accounts for: zero under
    # `zerofy_mix`, the recorded background otherwise. Measured here rather
    # than left for `verify` to discover, so a call whose noise floor swamps
    # its speech is visible in the manifest.
    background_rms, background_snr = _background(mixture, s1, s2, mask1 | mask2)

    write_wav(out_dir / "mix.wav", mixture, sr)
    write_wav(out_dir / "s1.wav", s1, sr)
    write_wav(out_dir / "s2.wav", s2, sr)

    # Degraded copies of the mixture. They are built from `mix1`/`mix2` -- the
    # two channels after the roll and the SIR gain but before the sum -- which is
    # the only point in this function where the two speakers still exist as
    # separate signals. s1/s2 are untouched and shared by every variant: the pair
    # the model trains on is (degraded mixture, clean targets).
    augment_items = []
    for index in range(variants):
        # Per call *and* per variant, so variant 2 is reproducible on its own and
        # the four variants of a call never draw the same room or the same clip.
        variant_rng = random.Random(f"{cfg.seed}:{call}:aug{index}")
        degraded, recipes, aug_gain = _degrade_variant(
            mix1, mix2, mask1, mask2, cfg, banks, variant_rng
        )
        name = f"mix_aug{index}.wav"
        write_wav(out_dir / name, degraded, sr)
        augment_items.append(
            {
                "index": index,
                "file": name,
                "recipes": recipes,
                "peak_gain": round(aug_gain, 5),
                "rms": float(
                    f"{np.sqrt(np.mean(np.square(degraded, dtype=np.float64))):.9g}"
                ),
                # How much of the clean mixture survives. `verify` uses it as the
                # floor for its own measurement, so a variant that is silence, or
                # a file from another call, is visible without re-deriving what
                # the recipe should have done.
                "correlation": round(_correlation(mixture, degraded), 4),
            }
        )

    # Variant files a previous build wrote and this one did not. Dropping from
    # four variants to two, or to none, leaves mix_aug2/3.wav behind otherwise:
    # the call directory survives (the selection still holds the call) so
    # `prune_stale_calls` never looks at it, and anything globbing
    # `dataset/train/*/mix_aug*.wav` instead of reading the manifest would train
    # on mixtures degraded under settings this dataset no longer uses.
    keep = {item["file"] for item in augment_items}
    for stale in out_dir.glob("mix_aug*.wav"):
        if stale.name not in keep:
            stale.unlink()

    meta = {
        "call": call,
        "split": record["split"],
        # Provenance, separately for the two halves of the pair: the mixture is
        # the recording, the targets may be the denoised copy.
        "mix_source": mix_path,
        "target_source": target_path,
        "targets_enhanced": targets_enhanced,
        # Kept so older readers (and `stats.json`) keep working.
        "source": mix_path,
        "enhanced": targets_enhanced,
        "enhance_backend": cfg.enhance.backend if targets_enhanced else None,
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
        # Whether the mixture was formed from the zerofied channels (so that
        # mix == s1 + s2) or from the channels as recorded.
        "zerofy_mix": bool(cfg.build.zerofy_mix),
        # RMS of mix - (s1 + s2): the background the mixture carries and the
        # targets do not, and how far it sits below the two voices.
        "background_rms": round(background_rms, 6),
        "background_snr_db": (
            None if background_snr is None else round(background_snr, 2)
        ),
        # The mode this call was built in, so a rebuild can tell a cached call
        # apart from one that has to be redone.
        "shuffle": bool(shuffle),
        "shift_samples": int(shift),
        "shifted": bool(shift),
        # Why this call is (or is not) shifted -- see mixing/overlap.py:safe_shifts.
        "shift_reason": shift_reason,
        "sir_db": round(sir_db, 3),
        "sir_gain": round(sir_gain, 5),
        "peak_gain": round(peak_gain, 5),
        # A fingerprint of the three files as written, for `verify` to compare
        # against. Once the targets are denoised no sample relationship between
        # the mixture and them is exact, and correlation cannot see a level
        # error -- a mixture rescaled by 0.8, or swapped for another call with
        # the same voices at a different SIR, correlates just as well. The RMS
        # of each file can: PCM_16 rounding moves it only to second order.
        "rms": {
            name: float(f"{np.sqrt(np.mean(np.square(signal, dtype=np.float64))):.9g}")
            for name, signal in (("mix", mixture), ("s1", s1), ("s2", s2))
        },
        # Every degraded copy of the mixture, and the exact draws that made it.
        # `fingerprint` is the hash of the settings they were drawn under, which
        # is what lets `run()` tell a reusable cached call from a stale one.
        "augment": {
            "variants": int(variants),
            "fingerprint": fingerprint,
            "items": augment_items,
        },
    }
    write_json(out_dir / "meta.json", meta)
    return meta


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    """|cos| between two signals. 1.0 is identical up to scale, 0.0 unrelated."""
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    n = min(x.size, y.size)
    if n == 0:
        return 0.0
    scale = float(np.linalg.norm(x[:n]) * np.linalg.norm(y[:n]))
    if scale <= 0.0:
        return 0.0
    return abs(float(x[:n] @ y[:n])) / scale


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

    # Every degraded mixture is windowed the same way, at the same offsets. The
    # target slices are written once and shared, so four variants cost four
    # mixture chunks rather than four complete triples.
    variants = []
    for item in meta.get("augment", {}).get("items", []):
        data, variant_sr = read_audio(call_dir / item["file"])
        if variant_sr != sr or len(data) != len(mixture):
            raise ValueError(
                f"{item['file']} is {len(data)} samples @ {variant_sr} Hz, "
                f"but mix.wav is {len(mixture)} @ {sr}"
            )
        variants.append((item["index"], data[:, 0]))

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

        common = {
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
        rows.append(
            {
                "mix": str((out_root / "mix" / name).relative_to(dataset_dir)),
                "variant": None,
                **common,
            }
        )
        for variant_index, data in variants:
            folder = out_root / f"mix_aug{variant_index}"
            write_wav(folder / name, data[start:end], sr)
            rows.append(
                {
                    "mix": str((folder / name).relative_to(dataset_dir)),
                    "variant": variant_index,
                    **common,
                }
            )
        index += 1
    return rows


def _manifest_rows(meta: dict) -> list[dict]:
    """One row for the clean mixture, then one per degraded variant.

    Every row names the *same* `s1`/`s2`: the targets are clean and shared, and
    only the model's input differs. `variant` is null on the clean row and the
    variant index on the others, so a trainer can take the whole file, or filter
    to `variant is not None` for the augmented material alone.

    Rows share a `call`, which means the manifest no longer has one row per
    call. Anything counting speakers or calls has to fold on `call` first --
    `verify` does, and `_write_stats` below does.
    """
    base = Path(meta["split"]) / meta["call"]
    common = {
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

    rows = [{"mix": str(base / "mix.wav"), "variant": None, **common}]
    for item in meta.get("augment", {}).get("items", []):
        rows.append(
            {
                "mix": str(base / item["file"]),
                "variant": item["index"],
                # The steps that actually fired on each track, flattened so a
                # manifest reader can filter on them without opening meta.json.
                "degradations": sorted(
                    {step for recipe in item["recipes"] for step in recipe}
                ),
                **common,
            }
        )
    return rows


def _write_stats(cfg, dataset_dir: Path, selection, rows, chunk_rows, stats,
                 chunk_counts: dict[str, int] | None = None,
                 shuffle: bool = False, variants: int = 0) -> None:
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

    # Rows and calls are no longer the same thing: with augmentation on, one
    # call contributes the clean mixture plus one row per degraded variant.
    # `calls` counts distinct calls, `hours` the training material all the rows
    # add up to -- which is the number the 4x expansion is about.
    per_split: dict[str, dict] = defaultdict(
        lambda: {"calls": 0, "rows": 0, "hours": 0.0, "clean_hours": 0.0}
    )
    speakers_per_split: dict[str, set] = defaultdict(set)
    gender_counts: dict[str, int] = defaultdict(int)
    seen: set[str] = set()
    for row in rows:
        bucket = per_split[row["split"]]
        bucket["rows"] += 1
        bucket["hours"] += row["duration"] / 3600.0
        speakers_per_split[row["split"]].update(row["speakers"])
        # Per call, not per row: five rows of one call are still two speakers
        # and one pair of genders.
        if row["call"] not in seen:
            seen.add(row["call"])
            bucket["calls"] += 1
            bucket["clean_hours"] += row["duration"] / 3600.0
            for gender in row["genders"]:
                gender_counts[str(gender)] += 1

    for split, bucket in per_split.items():
        bucket["hours"] = round(bucket["hours"], 3)
        bucket["clean_hours"] = round(bucket["clean_hours"], 3)
        bucket["speakers"] = len(speakers_per_split[split])
        # Chunks per split, so `test: 1 call, 0 chunks` is visible here rather
        # than only discoverable by opening the manifest.
        bucket["chunks"] = int((chunk_counts or {}).get(split, 0))

    payload = {
        "config": cfg.to_dict(),
        "calls": len(seen),
        "rows": len(rows),
        "chunks": len(chunk_rows),
        "augment": _augment_stats(cfg, stats, variants, describe),
        "splits": dict(per_split),
        "gender_sides": dict(gender_counts),
        "overlap": {
            # The setting actually used, which `config` below cannot show:
            # --shuffle can turn it on for a run without touching the config.
            "shuffle": bool(shuffle),
            "target": list(cfg.build.target_overlap) if shuffle else None,
            "natural_frac_target": cfg.build.natural_frac if shuffle else None,
            "shifted_frac_realized": round(
                sum(1 for r in rows if r["shifted"]) / max(len(rows), 1), 4
            ),
            "realized": describe(stats["overlap"]),
        },
        "sir_db": {"target": list(cfg.build.sir_db), "realized": describe(stats["sir_db"])},
        "mix": {
            # False means the mixture carries the recorded background, so
            # mix - (s1 + s2) is that background rather than rounding noise.
            # `verify` reads this to pick which invariant to check.
            "zerofy_mix": bool(cfg.build.zerofy_mix),
            "sums_to_sources": bool(cfg.build.zerofy_mix),
            "background_snr_db": describe(stats["background_snr_db"]),
            # describe() has no min, and for an SNR the minimum is the
            # interesting end: it is the noisiest call in the dataset.
            "background_snr_db_min": (
                round(min(stats["background_snr_db"]), 2)
                if stats["background_snr_db"]
                else None
            ),
        },
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
    _print_augment(cfg, payload["augment"], per_split)
    if payload["overlap"]["realized"]:
        realized = payload["overlap"]["realized"]
        note = (
            f"[target {tuple(cfg.build.target_overlap)} on the shifted share]"
            if shuffle
            else "[as recorded; --shuffle to boost it]"
        )
        print(
            f"[{NAME}] overlap  mean={realized['mean']:.3f} p50={realized['p50']:.3f} "
            f"p90={realized['p90']:.3f}   {note}"
        )
    print(f"[{NAME}] gender sides: {dict(gender_counts)}")

    if cfg.build.zerofy_mix:
        print(f"[{NAME}] mixture: zerofied sources summed -- mix == s1 + s2 exactly")
    else:
        snr = payload["mix"]["background_snr_db"]
        detail = (
            f"voices sit {snr['p50']:.1f} dB above the background (p50), "
            f"noisiest call {payload['mix']['background_snr_db_min']:.1f} dB"
            if snr
            else "no measurable background"
        )
        print(f"[{NAME}] mixture: channels as recorded -- {detail}")

    reasons = Counter(stats["shift_reason"])
    declined = reasons.get("boundary_not_silent", 0) + reasons.get("no_safe_shift", 0)
    print(f"[{NAME}] shift: {dict(reasons)}")
    if declined:
        print(
            f"[{NAME}] {declined:,} call(s) left unshifted because no wrap point fell in "
            "silence; shifting them would have cut an utterance in half"
        )

    n_enhanced = int(sum(stats["enhanced"]))
    print(f"[{NAME}] enhanced audio used for {n_enhanced:,} / {len(seen):,} calls")
    if seen and 0 < n_enhanced < len(seen):
        # Half the dataset denoised and half not is a domain split the model will
        # happily learn instead of learning to separate. Say so loudly.
        print(
            f"[{NAME}] WARNING: mixed dataset -- {len(seen) - n_enhanced:,} calls were built "
            "from raw audio. Finish `python -m dsd enhance`, then rebuild with --overwrite, "
            "or set build.use_enhanced=never for a uniformly raw dataset."
        )


def _augment_stats(cfg, stats, variants: int, describe) -> dict:
    """What the degradation chain drew, next to what it was asked to draw.

    Every rate is per *track*, not per variant: each degradation is sampled once
    for each of the two channels, so the denominator is twice the number of
    degraded mixtures. A rate materially below `build.augment.prob` means the
    step's bank was empty rather than that the coin came up tails.
    """
    acfg = cfg.build.augment
    tracks = len(stats["tracks"])
    fired = Counter(stats["degradation"])
    return {
        "variants": variants,
        "prob_target": acfg.prob,
        "tracks": tracks,
        "fired": {
            step: {
                "n": fired.get(step, 0),
                "rate": round(fired.get(step, 0) / tracks, 4) if tracks else None,
            }
            for step in augment_chain.STEPS
        },
        "noise_snr_db": {
            "target": list(acfg.noise.snr_db),
            "realized": describe(stats["noise_snr_db"]),
        },
        "band_limit_hz": {
            "target": list(acfg.band_limit.cutoff_hz),
            "realized": describe(stats["cutoff_hz"]),
        },
        "codec": dict(Counter(stats["codec"])),
        "reverb": dict(Counter(stats["reverb"])),
    }


def _print_augment(cfg, payload: dict, per_split: dict) -> None:
    if not payload["variants"]:
        return
    tracks = payload["tracks"]
    print(
        f"[{NAME}] augmentation: {payload['variants']} degraded mixture(s) per call, "
        f"{tracks:,} degraded tracks"
    )
    rates = ", ".join(
        f"{step}={payload['fired'][step]['rate']:.2f}"
        for step in augment_chain.STEPS
        if payload["fired"][step]["rate"] is not None
    )
    print(f"[{NAME}] fired (per track, target {payload['prob_target']}): {rates}")
    if payload["codec"]:
        print(f"[{NAME}] codecs: {payload['codec']}   rooms: {payload['reverb']}")
    snr = payload["noise_snr_db"]["realized"]
    if snr:
        print(
            f"[{NAME}] noise SNR: mean={snr['mean']:.1f} p50={snr['p50']:.1f} dB "
            f"[target {tuple(cfg.build.augment.noise.snr_db)}]"
        )
    clean = sum(b["clean_hours"] for b in per_split.values())
    total = sum(b["hours"] for b in per_split.values())
    if clean:
        print(
            f"[{NAME}] training material: {clean:.2f} h of calls -> {total:.2f} h of "
            f"(mixture, targets) pairs ({total / clean:.1f}x)"
        )
