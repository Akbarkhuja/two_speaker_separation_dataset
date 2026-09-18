"""A separate pass that re-reads the finished dataset and checks its invariants.

Written as a stage rather than a test because it checks the *artifact*, not the
code: it re-opens the wavs from disk and re-derives everything, so it catches a
truncated write or a half-finished rerun that a unit test never would.

The mixture tolerance is 1e-4 rather than 0. Each of the three wavs is
independently quantized to 16-bit PCM, so `mix - (s1 + s2)` carries up to about
1.5 LSB of rounding, roughly 4.6e-5. Anything materially larger means the peak
guard scaled the mixture without scaling the sources, or the files came from
different runs.

Which form of that check applies depends on how the dataset was built, and
`stats.json` says which. Under `build.zerofy_mix` the mixture is the sum of the
two zerofied sources and the identity holds everywhere. By default it is the two
channels as recorded, so `mix - (s1 + s2)` is the background between the turns:
the identity then holds only where both speakers are at full gain, and what is
checked elsewhere is that the background stays below the voices rather than
above them. A dataset with no `stats.json`, or one written before this option
existed, is checked the strict way.

The three files need not share a rate. The mixture is at the source's 8 kHz;
the targets are at the row's `target_sample_rate` -- 24 kHz when Sidon's
restored band was kept. What must hold is the same duration, to the sample at
an integer ratio. The mixture invariants compare samples, so they run on the
targets brought down to the mixture's rate, for the check only.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from ..core.audio import read_audio, resample
from ..core.manifest import read_json, read_jsonl
from .base import banner, progress, require

NAME = "verify"
REQUIRES = ("build",)

MIX_TOLERANCE = 1e-4
# A mixture whose background is louder than the two voices is not a mixture of
# those two voices. Deliberately loose -- this is a wrong-file check, not a
# quality bar; real calls in this corpus measure 20-40 dB.
MIN_BACKGROUND_SNR_DB = 0.0
# How much of the sources may show up in what the mixture holds beyond them.
# Measured on built calls with the background 24 dB down: a correct one sits at
# 0.015, a mixture swapped for another call's at 0.66, one scaled 20% against
# its sources at 0.97. The limit is deliberately nearer the broken end -- this
# is a wrong-file test, and the exact-sum check above is what catches small
# drift wherever the two speakers overlap.
MAX_RESIDUAL_CORRELATION = 0.5

# Where only one speaker is active, the mixture is that speaker's channel as
# recorded and the target is the same channel restored: the same voice at the
# same instant. Compared as short-time *magnitude* spectra, not waveforms, at
# the mixture's rate. A masking enhancer keeps the input's phase and its output
# correlates with the recording at 0.997 as a waveform, but Sidon resynthesises
# through a vocoder and is not phase-locked to its input at all: measured on
# real calls, waveform |corr| 0.00-0.04 for correctly aligned files. Magnitudes
# do not care about phase. Pearson correlation of 32 ms magnitude spectra,
# measured on real calls:
#   correct, Sidon targets       0.82 - 0.89
#   correct, masker targets      0.98 - 0.999
#   target 50 ms out of step     0.30 - 0.47
#   target 250 ms out of step    0.08 - 0.23
#   target from another call     0.03 - 0.10
# 0.65 sits between the worst correct file and the best broken one.
MIN_SOLO_CORRELATION = 0.65
SPECTRAL_FFT = 256   # 32 ms at 8 kHz
SPECTRAL_HOP = 80    # 10 ms

# How far a file's RMS may drift from the value build recorded before writing it.
# PCM_16 rounding adds noise ~9e-6 rms, which moves a signal's RMS only to second
# order (sqrt(r^2 + q^2) - r), so this is loose by orders of magnitude and still
# a factor of 200 tighter than a 20% rescale.
RMS_REL_TOLERANCE = 1e-3
RMS_ABS_TOLERANCE = 1e-6

# How far a degraded variant may fall below the correlation `build` measured
# between it and the clean mixture when it wrote it. The absolute value is not
# checkable here -- a variant that drew noise at -5 dB SNR legitimately sits near
# 0.3, one that drew nothing at all sits at 1.0 -- so what is checked is that the
# file on disk still matches the number recorded for it. A tolerance rather than
# equality because both files are quantized to PCM_16 independently.
VARIANT_CORRELATION_TOLERANCE = 0.02


def _erode(mask: np.ndarray, fade: int) -> np.ndarray:
    """Drop the `fade` samples at each edge of every True run.

    Those are the raised-cosine ramps `build` puts at each mask boundary, where
    a source sits at partial gain. They have to come out of both checks below:
    inside a ramp the mixture legitimately differs from the source, and the
    difference there is a scaled copy of the source itself, which is exactly the
    thing the correlation test treats as evidence of a broken file.

    A cumulative sum rather than repeated shifts -- a 10 ms fade at 8 kHz is 80
    samples each way and a call runs to minutes.
    """
    if fade <= 0:
        return mask
    width = 2 * fade + 1
    if mask.size < width:
        return np.zeros_like(mask)
    counts = np.concatenate(([0], np.cumsum(mask, dtype=np.int64)))
    eroded = np.zeros_like(mask)
    eroded[fade : mask.size - fade] = (counts[width:] - counts[:-width]) == width
    return eroded


def _both_at_full_gain(s1: np.ndarray, s2: np.ndarray, fade: int) -> np.ndarray:
    """Samples where both sources are speech and neither is inside a fade.

    Only there is `mix == s1 + s2` exact when the mixture was built from the raw
    channels; elsewhere the mixture carries background the targets faded out.
    """
    return _erode((s1 != 0.0) & (s2 != 0.0), fade)


def _active(target: np.ndarray, k: int) -> np.ndarray:
    """Where a target is non-zero, at the mixture's rate (`k` target samples each).

    A mixture sample counts as active if any of its `k` target samples is; so
    silence means all of them are exactly zero, which is what zerofying leaves.
    """
    on = target != 0.0
    return on if k == 1 else on.reshape(-1, k).any(axis=1)


def _spectral_alignment(mixture: np.ndarray, target: np.ndarray, solo: np.ndarray) -> float | None:
    """Correlation of the two magnitude spectrograms, over frames wholly in `solo`.

    Phase-free on purpose -- see MIN_SOLO_CORRELATION. None when there are too
    few such frames (under ~0.2 s) to say anything.
    """
    n = (len(mixture) - SPECTRAL_FFT) // SPECTRAL_HOP + 1
    if n <= 0:
        return None
    starts = SPECTRAL_HOP * np.arange(n)
    keep = starts[_window_inside(solo, starts)]
    if keep.size < 20:
        return None
    index = keep[:, None] + np.arange(SPECTRAL_FFT)[None, :]
    window = np.hanning(SPECTRAL_FFT)
    a = np.abs(np.fft.rfft(mixture[index].astype(np.float64) * window))
    b = np.abs(np.fft.rfft(target[index].astype(np.float64) * window))
    # Mean removed: magnitudes are all positive, so a plain cosine gives two
    # unrelated dense signals ~0.6 before they share anything at all.
    a -= a.mean()
    b -= b.mean()
    scale = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / scale) if scale > 0.0 else None


def _window_inside(mask: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """Whether each window `[s, s + SPECTRAL_FFT)` lies entirely inside `mask`."""
    counts = np.concatenate(([0], np.cumsum(mask, dtype=np.int64)))
    return (counts[starts + SPECTRAL_FFT] - counts[starts]) == SPECTRAL_FFT


def _residual_correlation(voices: np.ndarray, residual: np.ndarray) -> float:
    """|cos| between the sources and what the mixture holds beyond them.

    In a correct raw-channel build the leftover is the *other* channel's
    background, recorded on a physically isolated line, so it carries no trace
    of these sources: measured on real calls this sits at a few hundredths.

    Every way the three files can stop describing the same call moves it. If
    the mixture was scaled without the sources, the leftover is a scaled copy of
    them and this goes to 1. If the mixture belongs to a different call, or is
    time-shifted, the leftover contains -(s1 + s2) and it climbs just as far.
    That makes this the check that survives a call where the two speakers never
    overlap, which is common at the ~3% natural overlap of this corpus and
    leaves nothing for the exact-sum test to stand on.
    """
    v = voices.astype(np.float64)
    r = residual.astype(np.float64)
    scale = float(np.linalg.norm(v) * np.linalg.norm(r))
    if scale <= 0.0:
        return 0.0
    return abs(float(v @ r)) / scale


def _check_variants(
    dataset_dir: Path,
    call: str,
    clean: np.ndarray,
    variant_rows: list[dict],
    meta: dict,
    sample_rate: int,
    ceiling: float,
) -> list[str]:
    """Check the degraded copies of one call's mixture.

    None of the three mixture invariants above can apply here. A degraded
    variant has been through a room, a noise recording at down to -5 dB SNR, a
    codec and a packet dropper, so it is neither the sum of its targets nor
    strongly correlated with them -- by design. Enforcing a correlation floor
    against s1/s2 would fail precisely the variants that degraded the most,
    which is the opposite of what a check should do.

    What is still true, and is what gets checked:

      - it is the same length, at the same rate, as the clean mixture;
      - its RMS is the one `build` recorded before writing it;
      - its correlation with the clean mixture is the one `build` measured.
        That single number is what makes this a real check: it is the only
        quantity that moves if the file is from another call, is time-shifted,
        was rescaled, or was written from a different recipe than the one
        meta.json claims -- and unlike an absolute floor it stays valid whether
        the recipe degraded the signal heavily or not at all;
      - nothing in it is non-finite, and nothing exceeds the peak ceiling.
    """
    problems: list[str] = []
    items = {
        item["index"]: item for item in meta.get("augment", {}).get("items", [])
    }

    for row in variant_rows:
        index = row["variant"]
        name = Path(row["mix"]).name
        try:
            data, sr = read_audio(dataset_dir / row["mix"])
        except Exception as exc:
            problems.append(f"{call}: {name} unreadable ({exc})")
            continue

        signal = data[:, 0]
        if sr != sample_rate or signal.size != clean.size:
            problems.append(
                f"{call}: {name} is {signal.size} samples @ {sr} Hz, but the clean "
                f"mixture is {clean.size} @ {sample_rate}"
            )
            continue

        if not np.all(np.isfinite(signal)):
            problems.append(f"{call}: {name} contains non-finite samples")
            continue
        peak = float(np.abs(signal).max())
        if peak > ceiling + 1e-3:
            problems.append(f"{call}: {name} peaks at {peak:.4f}, above the {ceiling} ceiling")
        if peak == 0.0:
            problems.append(f"{call}: {name} is entirely silent")
            continue

        item = items.get(index)
        if item is None:
            problems.append(
                f"{call}: the manifest lists variant {index} but meta.json does not "
                "describe it -- the manifest and the call directory disagree"
            )
            continue

        got = float(np.sqrt(np.mean(np.square(signal, dtype=np.float64))))
        want = float(item.get("rms", got))
        if abs(got - want) > max(RMS_ABS_TOLERANCE, RMS_REL_TOLERANCE * want):
            problems.append(
                f"{call}: {name} has rms {got:.6f} where build wrote {want:.6f} "
                "-- it is not the file build produced for this variant"
            )

        if "correlation" in item:
            measured = _residual_correlation(clean, signal)
            expected = float(item["correlation"])
            if abs(measured - expected) > VARIANT_CORRELATION_TOLERANCE:
                problems.append(
                    f"{call}: {name} correlates with mix.wav at {measured:.3f} where "
                    f"build measured {expected:.3f} -- it is not the variant build "
                    "wrote for this call"
                )

    listed = {row["variant"] for row in variant_rows}
    for index in sorted(set(items) - listed):
        problems.append(
            f"{call}: meta.json describes variant {index} but no manifest row names it"
        )
    return problems


def _check_speaker_caps(cfg, rows, calls_by_speaker) -> list[str]:
    """Hold the dataset to the caps its selection was made with.

    The caps come from `work/selection.json`, not from config: `select` accepts
    them on the command line, and a selection made with `--max-calls-per-speaker
    50` used to be judged here against the config's 3 -- reporting 182 speakers
    "over the cap" in a selection that was exactly as asked. Config is only the
    fallback when no selection file is around.

    The duration check sums the `source_speech_sec` that `select` itself admitted
    against, so the two can never disagree about how speech is measured.
    """
    problems: list[str] = []
    selection = read_json(cfg.paths.selection_json) if cfg.paths.selection_json.exists() else None

    def configured(key):
        value = selection.get(key) if selection is not None and key in selection else getattr(
            cfg.select, key
        )
        return value if value is not None and value > 0 else None

    max_calls = configured("max_calls_per_speaker")
    if max_calls is not None:
        over = {s: n for s, n in calls_by_speaker.items() if n > max_calls}
        if over:
            worst = max(over.items(), key=lambda kv: kv[1])
            problems.append(
                f"{len(over)} speakers exceed the {max_calls}-call cap "
                f"(worst: {worst[0]} with {worst[1]})"
            )

    max_duration = configured("max_duration_per_speaker")
    if max_duration is None:
        return problems
    if selection is None:
        banner(NAME, "duration cap not checkable: no work/selection.json to read speech from")
        return problems

    records = selection.get("calls", {})
    seconds: dict[str, float] = defaultdict(float)
    unmeasured = 0
    # Distinct calls, not rows. With augmentation on a call appears once per
    # degraded variant plus once clean, and counting rows would multiply every
    # speaker's speech by that factor -- reporting a selection that is exactly
    # as `select` admitted it as five times over its own cap.
    for name in dict.fromkeys(row["call"] for row in rows):
        record = records.get(name)
        if record is None or "source_speech_sec" not in record:
            unmeasured += 1
            continue
        for speaker, side in zip(record["speakers"], record["source_speech_sec"]):
            seconds[speaker] += side

    if unmeasured:
        # A selection made before the field existed is not wrong, just not
        # checkable -- say so rather than fail every call.
        banner(
            NAME,
            f"duration cap not checkable for {unmeasured} call(s): their selection "
            "records carry no source_speech_sec (re-run `select --overwrite`)",
        )

    # `select` admits against these same stored (rounded) values, so the totals
    # agree exactly; the slack only absorbs float addition in a different order.
    over = {s: t for s, t in seconds.items() if t > max_duration + 1e-6}
    if over:
        worst = max(over.items(), key=lambda kv: kv[1])
        problems.append(
            f"{len(over)} speakers exceed the {max_duration / 60:g}-minute speech cap "
            f"(worst: {worst[0]} with {worst[1] / 60:.1f} min)"
        )
    return problems


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
    # is exactly what a stale file looks like. Keyed on the mixture path rather
    # than the call: a call now contributes one row per degraded variant, and
    # only the mixture path tells those rows apart.
    missing = {r["mix"] for r in combined} - {r["mix"] for r in seen}
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

    all_rows = list(read_jsonl(manifest_path))

    # Rows are grouped by call before anything else. With augmentation on, a
    # call is a clean mixture plus one row per degraded variant, all sharing the
    # same s1/s2. Sampling then picks whole calls rather than rows, so a variant
    # is never checked without the clean mixture it is measured against, and the
    # shared targets are read once instead of once per variant.
    by_call: dict[str, list[dict]] = defaultdict(list)
    for row in all_rows:
        by_call[row["call"]].append(row)

    names = list(by_call)
    if args.sample:
        import random

        names = random.Random(cfg.seed).sample(names, min(args.sample, len(names)))
    elif args.limit:
        names = names[: args.limit]

    # How the dataset was built decides which mixture invariant holds. Absent
    # or old stats.json -> assume the strict one, which can only be too strict.
    stats_path = dataset_dir / "stats.json"
    zerofy_mix = True
    if stats_path.exists():
        zerofy_mix = bool(read_json(stats_path).get("mix", {}).get("zerofy_mix", True))
    # Whether the targets were denoised while the mixture stayed the recording.
    # Taken from the rows themselves, not config, so a dataset is judged by how
    # it was built. Absent on anything written before the two were split apart.
    # The whole manifest, not the sample: `--sample`/`--limit` may have narrowed it.
    enhanced_targets = (not zerofy_mix) and any(row.get("enhanced") for row in all_rows)
    n_variants = sum(1 for row in all_rows if row.get("variant") is not None)
    fade = int(round(cfg.sample_rate * cfg.build.fade_ms / 1000.0))

    banner(NAME, f"checking {len(names)} calls under {dataset_dir}")
    if n_variants:
        banner(
            NAME,
            f"{n_variants:,} degraded variant rows: checked against the clean mixture "
            "they were derived from, not against s1 + s2",
        )
    banner(
        NAME,
        "mixture is the sum of the two sources"
        if zerofy_mix
        else (
            "mixture is the two channels as recorded; targets are the denoised copy"
            if enhanced_targets
            else "mixture is the two channels as recorded; sources are zerofied"
        ),
    )
    problems: list[str] = []

    # ---- per-call signal checks -------------------------------------- #
    worst_residual = 0.0
    worst_snr, worst_correlation, unchecked = None, 0.0, 0
    worst_solo = None
    for call in progress(names, "verify"):
        call_rows = by_call[call]
        # The clean mixture is the row every other row for this call is judged
        # against. A dataset built with `variants` but no clean row is not one
        # this pipeline produces, so fall back rather than skip the call.
        row = next((r for r in call_rows if r.get("variant") is None), call_rows[0])
        variant_rows = [r for r in call_rows if r.get("variant") is not None]
        try:
            mixture, sr = read_audio(dataset_dir / row["mix"])
            s1, sr1 = read_audio(dataset_dir / row["s1"])
            s2, sr2 = read_audio(dataset_dir / row["s2"])
        except Exception as exc:
            problems.append(f"{call}: unreadable ({exc})")
            continue

        # The mixture is at the source rate; the targets at whatever rate the
        # manifest says they were written at (24 kHz with Sidon's band kept).
        # Rows from before the key existed had one rate for all three.
        tsr = int(row.get("target_sample_rate", sr))
        if sr != cfg.sample_rate or sr1 != tsr or sr2 != tsr:
            problems.append(
                f"{call}: sample rates mix={sr} s1={sr1} s2={sr2}, expected mix={cfg.sample_rate} "
                f"and targets={tsr}"
            )
            continue
        if tsr % sr:
            problems.append(f"{call}: target rate {tsr} is not a whole multiple of {sr}")
            continue
        # Same duration, to the sample: with an integer ratio that is exactly
        # `k` target samples per mixture sample, and anything else means one
        # track is shifted against the labels.
        k = tsr // sr
        if not (len(mixture) * k == len(s1) == len(s2)):
            problems.append(
                f"{call}: duration mismatch mix={len(mixture)} @ {sr} s1={len(s1)} s2={len(s2)} "
                f"@ {tsr} (expected {len(mixture) * k})"
            )
            continue

        native_left, native_right, mono = s1[:, 0], s2[:, 0], mixture[:, 0]
        # Every mixture invariant below compares the mixture with the targets
        # sample by sample, so it needs one rate. It is the mixture's: the
        # mixture never had the band above 4 kHz, so comparing at 24 kHz would
        # mean inventing one for it, while bringing the targets down discards
        # only what the mixture cannot hold. For this check only -- the files
        # are untouched. Where a target is speech and where it is exactly zero
        # is read off the native samples: resampling rings across a zerofied
        # edge, so a resampled target has no exact zeros left to find.
        left_on = _active(native_left, k)
        right_on = _active(native_right, k)
        if k > 1:
            left, right = resample(native_left, tsr, sr), resample(native_right, tsr, sr)
        else:
            left, right = native_left, native_right

        # The three files must be the ones build wrote for this call. Checked by
        # level against the fingerprint in meta.json, because in the
        # enhanced-target mode nothing else here can see a level error: a
        # mixture rescaled by 0.8, or swapped for another call carrying the same
        # voices at a different SIR, still correlates perfectly with its targets.
        meta_path = dataset_dir / row["split"] / call / "meta.json"
        meta = read_json(meta_path) if meta_path.exists() else {}
        recorded = meta.get("rms")
        if recorded:
            for name, signal in (("mix", mono), ("s1", native_left), ("s2", native_right)):
                got = float(np.sqrt(np.mean(np.square(signal, dtype=np.float64))))
                want = float(recorded.get(name, got))
                if abs(got - want) > max(RMS_ABS_TOLERANCE, RMS_REL_TOLERANCE * want):
                    problems.append(
                        f"{call}: {name}.wav has rms {got:.6f} where build wrote {want:.6f} "
                        "-- it is not the file build produced for this call"
                    )
        difference = mono - (left + right)

        if zerofy_mix:
            residual = float(np.abs(difference).max())
            worst_residual = max(worst_residual, residual)
            if residual > MIX_TOLERANCE:
                problems.append(f"{call}: mix != s1 + s2 (max residual {residual:.2e})")
        elif enhanced_targets:
            # The mixture is the recording and the targets are its denoised copy,
            # so no sample of the mixture is the exact sum of the targets, and
            # what it holds beyond them is background *plus* what the enhancer
            # removed -- which is correlated with the voices by construction
            # (measured |corr| mean 0.42, max 0.69). Neither the exact-sum test
            # nor the leftover-correlation ceiling can apply.
            #
            # What still has to be true: wherever exactly one speaker is at full
            # gain, the mixture is that speaker's channel as recorded and the
            # target is the same channel restored -- the same voice at the same
            # instant. That collapses if the files come from different calls or
            # are shifted against each other. (A level error is the RMS
            # fingerprint's job above; both measures here are scale-free.)
            for name, target, on, other_on in (
                ("s1", left, left_on, right_on),
                ("s2", right, right_on, left_on),
            ):
                solo = _erode(on, fade) & ~other_on
                aligned = _spectral_alignment(mono, target, solo)
                if aligned is None:
                    continue
                worst_solo = aligned if worst_solo is None else min(worst_solo, aligned)
                if aligned < MIN_SOLO_CORRELATION:
                    problems.append(
                        f"{call}: where only {name} speaks, the mixture and {name} correlate "
                        f"at {aligned:.2f} (floor {MIN_SOLO_CORRELATION}); they are not "
                        "the same call, or are shifted against each other"
                    )

            # Reported, deliberately not enforced. How loud the leftover is here
            # depends on how much the enhancer changed the speech level, which
            # is a property of the enhancer rather than of the files: one that
            # halves its output leaves a leftover exactly as loud as the voices
            # on a perfectly matched call. The alignment check above is what
            # catches mismatched files, and it does not care about level.
            support = _erode(left_on, fade) | _erode(right_on, fade)
            if support.any():
                voices = (left + right)[support]
                noise = float(np.sqrt(np.mean(np.square(difference[support], dtype=np.float64))))
                level = float(np.sqrt(np.mean(np.square(voices, dtype=np.float64))))
                if noise > 0.0 and level > 0.0:
                    snr = float(20.0 * np.log10(level / noise))
                    worst_snr = snr if worst_snr is None else min(worst_snr, snr)
        else:
            # Where both speakers are at full gain the mixture is still exactly
            # their sum, and that is the check that catches a mismatched or
            # half-written trio of files. It needs the two to actually overlap
            # somewhere; calls where they never do are counted and reported
            # rather than passed over in silence.
            exact = _erode(left_on & right_on, fade)
            if exact.any():
                residual = float(np.abs(difference[exact]).max())
                worst_residual = max(worst_residual, residual)
                if residual > MIX_TOLERANCE:
                    problems.append(
                        f"{call}: mix != s1 + s2 where both speak "
                        f"(max residual {residual:.2e})"
                    )
            else:
                unchecked += 1

            # Wherever at least one source is at full gain, the rest of the
            # difference is the other channel's background. Two things have to
            # be true of it: it sits under the voices, and it looks nothing
            # like them. The second is what still has teeth on a call where the
            # speakers never overlap and the exact test above never ran.
            support = _erode(left_on, fade) | _erode(right_on, fade)
            if support.any():
                voices = (left + right)[support]
                leftover = difference[support]
                noise = float(np.sqrt(np.mean(np.square(leftover, dtype=np.float64))))
                level = float(np.sqrt(np.mean(np.square(voices, dtype=np.float64))))

                correlation = _residual_correlation(voices, leftover)
                worst_correlation = max(worst_correlation, correlation)
                if correlation > MAX_RESIDUAL_CORRELATION:
                    problems.append(
                        f"{call}: what the mixture holds beyond s1 + s2 looks like the "
                        f"sources themselves (|corr| {correlation:.2f}); mix, s1 and s2 "
                        "are not the same call, or the mixture was scaled alone"
                    )

                if noise > 0.0 and level > 0.0:
                    snr = float(20.0 * np.log10(level / noise))
                    worst_snr = snr if worst_snr is None else min(worst_snr, snr)
                    if snr < MIN_BACKGROUND_SNR_DB:
                        problems.append(
                            f"{call}: background is louder than the voices "
                            f"({snr:.1f} dB); mix and sources may not match"
                        )

        # ---- degraded variants ---------------------------------------- #
        if variant_rows:
            problems += _check_variants(
                dataset_dir, call, mono, variant_rows, meta, cfg.sample_rate,
                cfg.build.peak_ceiling,
            )

        # The other direction, and the reason `build` deletes these itself: a
        # rebuild with fewer variants leaves the extra files in a directory the
        # selection still holds, so nothing prunes them. They are invisible to a
        # manifest reader and picked up by anything that globs.
        named = {Path(r["mix"]).name for r in call_rows}
        extra = [
            path.name
            for path in sorted((dataset_dir / row["split"] / call).glob("mix_aug*.wav"))
            if path.name not in named
        ]
        if extra:
            problems.append(
                f"{call}: {len(extra)} degraded mixture(s) on disk that no manifest row "
                f"names (e.g. {extra[0]}) -- left by a build with more variants"
            )

        # Zerofying must have silenced something on both sides; a source with
        # no exact zeros means the mask was never applied.
        for name, source in (("s1", s1[:, 0]), ("s2", s2[:, 0])):
            if not np.any(source == 0.0):
                problems.append(f"{call}: {name} has no silence -- was it zerofied?")
            if not np.any(source != 0.0):
                problems.append(f"{call}: {name} is entirely silent")

    # ---- dataset-level checks ---------------------------------------- #
    # Folded to one entry per call first. A speaker's call count and the gender
    # balance are properties of the calls, not of how many degraded copies of
    # each mixture happen to be on disk; counting rows would inflate both by the
    # variant count and make the cap check fire on a correct dataset.
    unique_rows = list({row["call"]: row for row in all_rows}.values())
    splits_by_speaker: dict[str, set] = defaultdict(set)
    calls_by_speaker: dict[str, int] = defaultdict(int)
    gender_counts: dict[str, int] = defaultdict(int)
    for row in unique_rows:
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

    problems += _check_speaker_caps(cfg, all_rows, calls_by_speaker)

    missing = [row["mix"] for row in all_rows if not (dataset_dir / row["mix"]).exists()]
    if missing:
        problems.append(f"{len(missing)} manifest paths do not resolve (e.g. {missing[0]})")

    # The other direction: call directories on disk that no manifest lists.
    # Anything globbing the tree instead of reading the manifest trains on them.
    listed = {row["call"] for row in all_rows}
    orphans = [
        call_dir
        for split_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir() and p.name != "chunks")
        for call_dir in sorted(p for p in split_dir.iterdir() if p.is_dir())
        if call_dir.name not in listed
    ]
    if orphans:
        problems.append(
            f"{len(orphans)} call directories are on disk but in no manifest "
            f"(e.g. {orphans[0].relative_to(dataset_dir)}) -- usually a `build --limit` "
            "over a larger earlier build; `build` prunes calls the selection dropped"
        )

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
    banner(
        NAME,
        f"calls in manifest: {len(unique_rows)}"
        + (f" across {len(all_rows)} rows" if len(all_rows) != len(unique_rows) else "")
        + f", unique speakers: {len(calls_by_speaker)}",
    )
    banner(NAME, f"worst mix residual: {worst_residual:.2e} (tolerance {MIX_TOLERANCE:.0e})")
    if not zerofy_mix:
        margin = "n/a" if worst_snr is None else f"{worst_snr:.1f} dB"
        banner(NAME, f"noisiest call: voices {margin} above the background")
    if enhanced_targets:
        solo = "n/a" if worst_solo is None else f"{worst_solo:.3f}"
        banner(
            NAME,
            f"weakest mixture/target alignment where one speaker talks alone: {solo} "
            f"(floor {MIN_SOLO_CORRELATION})",
        )
    elif not zerofy_mix:
        banner(
            NAME,
            f"worst source/leftover correlation: {worst_correlation:.3f} "
            f"(limit {MAX_RESIDUAL_CORRELATION})",
        )
        if unchecked:
            banner(
                NAME,
                f"{unchecked} call(s) never have both speakers active at once, so the "
                "exact sum could not be checked on them; the correlation test above "
                "covers them instead",
            )
    
    if problems:
        banner(NAME, f"FAILED with {len(problems)} problem(s):")
        for problem in problems[:40]:
            print(f"[{NAME}]   - {problem}")
        if len(problems) > 40:
            print(f"[{NAME}]   ... and {len(problems) - 40} more")
        raise SystemExit(1)
    banner(NAME, "OK -- every check passed")
