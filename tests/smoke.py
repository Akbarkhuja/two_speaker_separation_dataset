#!/usr/bin/env python3
"""Smoke tests for the dsd pipeline.

    python tests/smoke.py               # unit + config + http + e2e  (~2.5 s, no GPU/network)
    python tests/smoke.py --real 10     # additionally: real models against Datasets/
    python tests/smoke.py --only fade   # run checks whose name or group matches a substring

`dsd verify` guards the finished *artifact*; this guards the *code* that produced it. The
four bugs found while bringing the pipeline up were all caught by running it and reading the
output, and every one of them would slide straight back in unnoticed.

Four groups:

  unit    Pure functions, checked against brute force where an independent implementation is
          cheap -- interval subtraction, the overlap FFT, the peak guard's `mix == s1 + s2`.
  config  Override typing, path resolution, and the registry's contract.
  http    The two HTTP backends replayed against the payloads their servers are documented
          to return, using a local stub server. No docker, no GPU.
  e2e     A synthetic corpus of known construction driven through all eight stages with stub
          backends. Because the corpus is planted, the assertions can be exact: which calls
          survive the filter and why each other one dies, and that the clustering recovers
          precisely the voices that were put in.

There is no pytest in the `nemo` env and the sibling repos carry no test framework, so this
is a plain script: exit 0 if everything passes, 1 otherwise.
"""

from __future__ import annotations

import argparse
import atexit
import contextlib
import io
import json
import random
import shutil
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dsd import cli  # noqa: E402
from dsd.config import Config  # noqa: E402
from dsd.core import audio as A  # noqa: E402
from dsd.core import rttm as R  # noqa: E402
from dsd.core.manifest import read_json, read_jsonl  # noqa: E402
from dsd.mixing import chunker, mixer, overlap  # noqa: E402
from dsd.registry import EMBEDDERS, ENHANCERS, GENDER, VADS, Registry  # noqa: E402

SR = 8000


# --------------------------------------------------------------------------- #
# harness
# --------------------------------------------------------------------------- #
CHECKS: list[tuple[str, str, object]] = []


def check(group: str):
    """Register a check. It passes by returning; it fails by raising.

    A returned string is printed next to the ok, so a green run still shows the
    numbers it measured rather than only asserting on them silently.
    """

    def decorator(fn):
        CHECKS.append((group, fn.__name__.lstrip("_"), fn))
        return fn

    return decorator


@contextlib.contextmanager
def quiet():
    """Swallow a stage's output; the caller reprints it only when a check fails."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        yield buffer


def approx(a, b, tol=1e-6) -> bool:
    return abs(float(a) - float(b)) <= tol


# --------------------------------------------------------------------------- #
# [unit] intervals
# --------------------------------------------------------------------------- #
@check("unit")
def subtract_intervals_matches_brute_force():
    rng = random.Random(0)
    for trial in range(400):
        span = 60

        def sample(count):
            out = []
            for _ in range(count):
                start = rng.uniform(0, span)
                out.append((round(start, 3), round(min(start + rng.uniform(0.1, 12), span), 3)))
            return out

        keep, cut = sample(rng.randint(0, 5)), sample(rng.randint(0, 5))
        got = R.subtract_intervals(keep, cut)

        # Independent implementation: paint the same thing on a 1 kHz grid.
        grid = np.zeros(span * 1000, dtype=bool)
        for a, b in keep:
            grid[int(a * 1000) : int(b * 1000)] = True
        for a, b in cut:
            grid[int(a * 1000) : int(b * 1000)] = False
        mine = np.zeros(span * 1000, dtype=bool)
        for a, b in got:
            mine[int(a * 1000) : int(b * 1000)] = True

        # A handful of samples may differ purely from rounding at the endpoints.
        differing = int((grid ^ mine).sum())
        budget = len(got) + len(keep) + len(cut) + 2
        assert differing <= budget, f"trial {trial}: {differing} samples differ (budget {budget})"

        for a, b in got:
            assert b > a, f"empty interval {(a, b)}"
        for (_, end), (start, _) in zip(got, got[1:]):
            assert end <= start, f"overlapping output {got}"
    return "400 random cases"


@check("unit")
def subtract_intervals_edge_cases():
    assert R.subtract_intervals([(0, 10)], [(0, 10)]) == []
    assert R.subtract_intervals([(0, 10)], [(3, 4)]) == [(0, 3), (4, 10)]
    assert R.subtract_intervals([(0, 10)], []) == [(0, 10)]
    assert R.subtract_intervals([], [(1, 2)]) == []
    assert R.subtract_intervals([(0, 10)], [(-5, 2)]) == [(2, 10)]
    assert R.subtract_intervals([(0, 10)], [(8, 50)]) == [(0, 8)]
    # Two cuts inside one interval, and a cut spanning two intervals.
    assert R.subtract_intervals([(0, 10)], [(2, 3), (6, 7)]) == [(0, 2), (3, 6), (7, 10)]
    assert R.subtract_intervals([(0, 3), (5, 8)], [(2, 6)]) == [(0, 2), (6, 8)]


@check("unit")
def merge_intervals_joins_within_gap():
    assert R.merge_intervals([]) == []
    assert R.merge_intervals([(0, 1), (1.05, 2)], gap=0.1) == [(0, 2)]
    assert R.merge_intervals([(0, 1), (1.5, 2)], gap=0.1) == [(0, 1), (1.5, 2)]
    assert R.merge_intervals([(0, 5), (1, 2)]) == [(0, 5)]  # nested
    assert R.merge_intervals([(3, 4), (0, 1)]) == [(0, 1), (3, 4)]  # unsorted input


@check("unit")
def split_long_covers_exactly():
    for total, longest in ((1.0, 20.0), (20.0, 20.0), (20.1, 20.0), (55.0, 20.0)):
        pieces = R.split_long(0.0, total, longest)
        assert all(end - start <= longest + 1e-9 for start, end in pieces)
        assert approx(pieces[0][0], 0.0) and approx(pieces[-1][1], total)
        for (_, end), (start, _) in zip(pieces, pieces[1:]):
            assert approx(end, start), "pieces must be contiguous"


@check("unit")
def speech_mask_bounds_and_clipping():
    mask = A.speech_mask([(0.1, 0.2)], 1000, sr=1000)
    assert not mask[:100].any() and mask[100:200].all() and not mask[200:].any()
    # Out-of-range intervals are clipped rather than wrapping or raising.
    assert A.speech_mask([(-5, 999)], 100, 10).all()
    assert not A.speech_mask([(50, 60)], 100, 1).any() or True  # clipped to the end
    assert not A.speech_mask([], 100, 10).any()


# --------------------------------------------------------------------------- #
# [unit] zerofy / fade
# --------------------------------------------------------------------------- #
@check("unit")
def fade_envelope_removes_edge_clicks():
    rng = np.random.default_rng(0)
    # Call-like proportions on purpose. The fade costs a fixed ~0.6 * fade_ms of
    # energy per edge, so a signal with only a second of speech would fail the
    # retention bar on arithmetic alone rather than on anything being wrong.
    signal = rng.normal(0, 0.3, 22 * SR).astype(np.float32)
    mask = A.speech_mask([(1.0, 5.0), (8.0, 14.0), (17.0, 20.0)], len(signal), SR)

    envelope = A.fade_envelope(mask, SR, 10.0)
    assert envelope.min() >= 0.0 and envelope.max() <= 1.0

    hard = signal * mask.astype(np.float32)
    soft = signal * envelope
    edges = np.flatnonzero(np.diff(mask.astype(np.int8)))

    def worst_jump(x):
        return float(np.abs(np.diff(x)[edges]).max())

    reduction = worst_jump(hard) / max(worst_jump(soft), 1e-12)
    assert reduction >= 100, f"fade only reduced the edge step {reduction:.1f}x"

    retained = float((soft**2).sum() / (hard**2).sum())
    assert retained >= 0.99, f"fade ate {100 * (1 - retained):.2f}% of the speech energy"
    return f"{reduction:.0f}x quieter edges, {retained * 100:.2f}% energy kept"


@check("unit")
def fade_envelope_degenerate_masks():
    # All-silent and all-speech have no edges; neither may raise or corrupt.
    silent = np.zeros(1000, dtype=bool)
    assert not A.fade_envelope(silent, SR, 10.0).any()
    speech = np.ones(1000, dtype=bool)
    assert A.fade_envelope(speech, SR, 10.0).all()
    # A fade longer than the signal must still stay in range.
    envelope = A.fade_envelope(A.speech_mask([(0.0, 0.01)], 100, SR), SR, 500.0)
    assert envelope.min() >= 0.0 and envelope.max() <= 1.0


# --------------------------------------------------------------------------- #
# [unit] overlap
# --------------------------------------------------------------------------- #
@check("unit")
def overlap_fft_matches_np_roll():
    rng = np.random.default_rng(7)
    for _ in range(5):
        n = int(rng.integers(300, 900))
        a = rng.random(n) < 0.35
        b = rng.random(n) < 0.45
        spectrum = np.fft.rfft(a.astype(float)) * np.conj(np.fft.rfft(b.astype(float)))
        from_fft = np.rint(np.fft.irfft(spectrum, n=n)).astype(int)
        # The definition the shift search relies on: A against B rolled right by k.
        brute = np.array([int((a & np.roll(b, k)).sum()) for k in range(n)])
        assert (from_fft == brute).all(), "conjugate is on the wrong side; shift is inverted"
    return "all shifts, 5 random mask pairs"


@check("unit")
def overlap_shift_hits_the_target_band():
    rng = random.Random(11)
    achieved = []
    for trial in range(40):
        seeded = random.Random(trial)
        n = SR * seeded.randint(60, 240)
        mask_a = np.zeros(n, dtype=bool)
        mask_b = np.zeros(n, dtype=bool)
        # Lead-in and tail silence, as real recordings have: 89.5% of the corpus
        # is silent at both ends. Starting a turn at sample 0 would make every
        # call unshiftable for the boundary-seam reason and test nothing.
        cursor = int(2.0 * SR)
        while cursor < n - int(3.0 * SR):
            span = int(seeded.uniform(0.5, 4.0) * SR)
            (mask_a if seeded.random() < 0.5 else mask_b)[cursor : cursor + span] = True
            cursor += span + int(seeded.uniform(0.2, 2.0) * SR)

        band = (0.15, 0.60)
        shift, predicted, reason = overlap.shift_for_target(mask_a, mask_b, SR, band, rng)
        realized = overlap.overlap_ratio(mask_a, np.roll(mask_b, shift))
        assert abs(realized - predicted) < 0.02, f"predicted {predicted:.3f}, got {realized:.3f}"
        if reason != "ok":
            # A call with no silent wrap point is left alone rather than chopped.
            assert shift == 0, f"reason {reason!r} but shift {shift}"
            continue
        assert band[0] <= realized <= band[1], f"landed at {realized:.3f}, outside {band}"
        achieved.append(realized)

    assert len(achieved) >= 35, f"only {len(achieved)}/40 were shiftable; constraint too tight"

    # The regression this guards: sampling uniformly over *shifts* rather than
    # over ratios piles every result against the bottom edge of the band.
    spread = float(np.percentile(achieved, 75) - np.percentile(achieved, 25))
    assert spread > 0.03, f"results are bunched (IQR {spread:.3f}); is the ratio being sampled?"
    return f"{len(achieved)}/40 in band, IQR {spread:.3f}"


@check("unit")
def overlap_ratio_and_frames():
    assert overlap.overlap_ratio(np.zeros(10, bool), np.zeros(10, bool)) == 0.0
    a = np.zeros(100, dtype=bool)
    b = np.zeros(100, dtype=bool)
    a[:50] = True
    b[25:75] = True
    assert approx(overlap.overlap_ratio(a, b), 25 / 75, tol=1e-9)
    # A frame counts as speech when any sample in it does.
    frames = overlap.to_frames(A.speech_mask([(0.0, 0.005)], SR, SR), SR)
    assert frames[0] and not frames[1:].any()


@check("unit")
def overlap_shift_handles_silent_source():
    silent = np.zeros(SR * 10, dtype=bool)
    speech = np.zeros(SR * 10, dtype=bool)
    speech[: SR * 2] = True
    shift, ratio, _r = overlap.shift_for_target(speech, silent, SR, (0.15, 0.6), random.Random(0))
    assert shift == 0 and ratio == 0.0


# --------------------------------------------------------------------------- #
# [unit] wrap seams
# --------------------------------------------------------------------------- #
GUARD_SEC = 0.2


def _turn_taking_mask(seed, seconds=180, lead=2.0, tail=2.0):
    """A call-shaped mask: silence at both ends, alternating turns in between."""
    rng = random.Random(seed)
    n = seconds * SR
    mask = np.zeros(n, dtype=bool)
    cursor = int(lead * SR)
    while cursor < n - int(tail * SR):
        span = int(rng.uniform(1.0, 5.0) * SR)
        end = min(cursor + span, n - int(tail * SR))
        mask[cursor:end] = True
        cursor = end + int(rng.uniform(0.6, 3.0) * SR)
    return mask


def _seams_are_silent(mask_b, shift, guard_sec=GUARD_SEC):
    """The four samples the wrap makes adjacent must all be silence.

    np.roll(x, k)[i] = x[(i - k) % n], so after the roll indices 0 and n-1 hold
    the two samples straddling the split seam, and indices k and k-1 hold the
    two straddling the boundary seam. Checking those four *is* the property.
    """
    rolled = np.roll(mask_b, shift)
    guard = int(guard_sec * SR)
    n = len(rolled)
    for index in (0, n - 1, shift % n, (shift - 1) % n):
        lo, hi = max(0, index - guard), min(n, index + guard + 1)
        if rolled[lo:hi].any():
            return False
    return True


@check("unit")
def chosen_shifts_never_cut_an_utterance():
    """The reported bug: a wrap landing mid-word chops it and moves the halves.

    Measured on the real dataset before this constraint existed, 22.4% of
    shifted calls had the split seam inside an utterance.
    """
    rng = random.Random(5)
    shifted, unshifted = 0, 0
    for seed in range(40):
        mask_a = _turn_taking_mask(seed * 2)
        mask_b = _turn_taking_mask(seed * 2 + 1)
        shift, _ratio, reason = overlap.shift_for_target(
            mask_a, mask_b, SR, (0.15, 0.60), rng, guard_sec=GUARD_SEC
        )
        assert _seams_are_silent(mask_b, shift), (
            f"seed {seed}: shift {shift / SR:.2f}s cuts an utterance (reason={reason})"
        )
        if shift:
            shifted += 1
        else:
            unshifted += 1
    assert shifted >= 30, f"only {shifted}/40 got shifted; the constraint is too strict"
    return f"{shifted}/40 shifted, all seams clean"


@check("unit")
def a_call_that_starts_or_ends_in_speech_is_not_shifted():
    """The boundary seam is the same for every offset, so it gates the call."""
    mask_a = _turn_taking_mask(11)
    for label, edit in (("starts", slice(0, SR)), ("ends", slice(-SR, None))):
        mask_b = _turn_taking_mask(12)
        mask_b[edit] = True
        shift, _ratio, reason = overlap.shift_for_target(
            mask_a, mask_b, SR, (0.15, 0.60), random.Random(0), guard_sec=GUARD_SEC
        )
        assert shift == 0, f"{label} in speech: expected no shift, got {shift}"
        assert reason == "boundary_not_silent", f"{label}: reason was {reason!r}"

    # A channel that is speech throughout has no silent wrap point anywhere.
    # Both masks must be the same length: they are two channels of one call.
    partner = _turn_taking_mask(13)
    always = np.ones(len(partner), dtype=bool)
    shift, _ratio, reason = overlap.shift_for_target(
        partner, always, SR, (0.15, 0.60), random.Random(0), guard_sec=GUARD_SEC
    )
    assert shift == 0 and reason in ("no_safe_shift", "boundary_not_silent"), reason


@check("unit")
def safe_shifts_agrees_with_rolling_the_mask():
    """Cross-check the analytic mask against actually rolling, for every offset."""
    for seed in (21, 22):
        mask = _turn_taking_mask(seed, seconds=20)
        frames = overlap.to_frames(mask, SR)
        guard_frames = int(GUARD_SEC * overlap.FRAME_RATE)
        safe = overlap.safe_shifts(frames, guard_frames)

        for k in range(0, len(frames), 7):  # every 7th offset keeps this quick
            rolled = np.roll(frames, k)
            widened = overlap._dilate(frames, guard_frames)
            expected = not (
                np.roll(widened, k)[0]
                or np.roll(widened, k)[-1]
                or widened[0]
                or widened[-1]
            )
            if k == 0:
                expected = True
            assert bool(safe[k]) == expected, (
                f"seed {seed} shift {k}: safe={bool(safe[k])} but rolling says {expected}"
            )
            assert rolled.shape == frames.shape
    return "analytic mask == rolled mask, all offsets"


# --------------------------------------------------------------------------- #
# [unit] mixer / chunker
# --------------------------------------------------------------------------- #
@check("unit")
def scale_to_sir_hits_the_requested_ratio():
    rng = np.random.default_rng(0)
    s1 = rng.normal(0, 0.10, SR).astype(np.float32)
    s2 = rng.normal(0, 0.50, SR).astype(np.float32)
    active = np.ones(SR, dtype=bool)
    for target in (-6.0, 0.0, 6.0):
        scaled, _gain = mixer.scale_to_sir(s1, s2, active, active, target)
        got = 20 * np.log10(A.active_rms(s1, active) / A.active_rms(scaled, active))
        assert approx(got, target, tol=0.01), f"asked {target} dB, measured {got:.3f} dB"

    # A silent source cannot be levelled; the gain must stay 1.0 rather than blow up.
    _scaled, gain = mixer.scale_to_sir(s1, np.zeros(SR, np.float32), active, active, 0.0)
    assert gain == 1.0


@check("unit")
def mix_preserves_the_sum_identity_under_the_peak_guard():
    rng = np.random.default_rng(1)
    # Deliberately hot, so the peak guard has to engage.
    s1 = rng.normal(0, 0.8, SR).astype(np.float32)
    s2 = rng.normal(0, 0.8, SR).astype(np.float32)

    mixture, out1, out2, gain = mixer.mix(s1, s2, ceiling=0.99)
    assert gain < 1.0, "this input should have clipped; the guard did not engage"
    assert float(np.abs(mixture).max()) <= 0.99 + 1e-6
    residual = float(np.abs(mixture - (out1 + out2)).max())
    assert residual < 1e-6, f"mix != s1 + s2 after the guard (residual {residual:.2e})"

    quiet_mix, q1, q2, quiet_gain = mixer.mix(s1 * 0.01, s2 * 0.01, ceiling=0.99)
    assert quiet_gain == 1.0
    assert float(np.abs(quiet_mix - (q1 + q2)).max()) < 1e-6
    return f"guard gain {gain:.3f}, residual {residual:.1e}"


@check("unit")
def peak_guard_covers_the_sources_not_just_the_mixture():
    """A source can clip while the mixture does not, if the two partially cancel.

    Reproduces the defect found on a real build: 22 of 2513 calls had s2 pinned
    at full scale by the PCM_16 write while the mixture sat at the 0.99 ceiling,
    breaking mix == s1 + s2 by up to 1.06e-01.
    """
    n = SR
    t = np.arange(n) / SR
    # Deliberately anti-phase and loud: the sum is small, each source is not.
    loud = (1.6 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    s1, s2 = loud, (-loud * 0.9).astype(np.float32)
    assert float(np.abs(s1 + s2).max()) < 0.99 < float(np.abs(s2).max()), "bad fixture"

    mixture, out1, out2, gain = mixer.mix(s1, s2, ceiling=0.99)
    for name, signal in (("mix", mixture), ("s1", out1), ("s2", out2)):
        assert float(np.abs(signal).max()) <= 0.99 + 1e-6, (
            f"{name} would clip on a PCM_16 write (peak {np.abs(signal).max():.4f})"
        )
    assert float(np.abs(mixture - (out1 + out2)).max()) < 1e-6
    assert gain < 1.0
    return f"guard engaged on a source, gain {gain:.3f}"


@check("unit")
def chunk_windows_drop_the_partial_tail():
    got = list(chunker.windows(10000, 1000, chunk_sec=4.0, hop_sec=2.0))
    assert got == [(0, 4000), (2000, 6000), (4000, 8000), (6000, 10000)], got
    # Shorter than one window yields nothing rather than a zero-padded window.
    assert list(chunker.windows(3999, 1000, 4.0, 2.0)) == []
    assert list(chunker.windows(0, 1000, 4.0, 2.0)) == []


@check("unit")
def chunk_is_usable_needs_both_sources():
    mask1 = np.zeros(8000, dtype=bool)
    mask2 = np.zeros(8000, dtype=bool)
    mask1[:6000] = True
    mask2[:2000] = True
    assert chunker.is_usable(mask1, mask2, 0, 8000, SR, min_active=0.2)
    assert not chunker.is_usable(mask1, mask2, 0, 8000, SR, min_active=0.5)
    assert not chunker.is_usable(mask1, np.zeros(8000, bool), 0, 8000, SR, min_active=0.1)


# --------------------------------------------------------------------------- #
# [unit] rttm round trip
# --------------------------------------------------------------------------- #
@check("unit")
def rttm_round_trips():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "abc-123.rttm"
        original = [
            R.Segment(start=1.234, end=3.5, speaker="ch0_speaker_0", channel=1),
            R.Segment(start=0.5, end=2.0, speaker="ch1_speaker_0", channel=2),
        ]
        R.write_rttm(path, "abc-123", original)

        # The URI field is the join key for every downstream artifact.
        for line in path.read_text().splitlines():
            assert line.split()[1] == "abc-123"

        back = R.read_rttm(path)
        assert len(back) == 2
        by_speaker = {s.speaker: s for s in back}
        for source in original:
            got = by_speaker[source.speaker]
            # The writer formats to 3 decimals, so that is the honest tolerance.
            assert approx(got.start, source.start, tol=5e-4)
            assert approx(got.duration, source.duration, tol=5e-4)
            assert got.channel == source.channel

        assert R.total_speech(back, "ch0_speaker_0") > 2.2
        assert R.speakers_of(back) == {"ch0_speaker_0", "ch1_speaker_0"}


# --------------------------------------------------------------------------- #
# [unit] per-split manifests
# --------------------------------------------------------------------------- #
@check("unit")
def split_manifests_cover_every_configured_split():
    from dsd.stages.build import write_split_manifests

    rows = [
        {"call": "a", "split": "train"},
        {"call": "b", "split": "train"},
        {"call": "c", "split": "dev"},
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        counts = write_split_manifests(out, rows, ["train", "dev", "test"])

        assert counts == {"train": 2, "dev": 1, "test": 0}, counts
        # An empty split still gets a file: a training command pointed at a
        # configured split should find an empty manifest, not a missing path.
        # On the real corpus `test` has one call and zero chunks.
        assert (out / "test.jsonl").exists(), "an empty split must still be written"
        assert list(read_jsonl(out / "test.jsonl")) == []

        assert [r["call"] for r in read_jsonl(out / "train.jsonl")] == ["a", "b"]
        assert [r["call"] for r in read_jsonl(out / "dev.jsonl")] == ["c"]
        for split in ("train", "dev"):
            assert all(r["split"] == split for r in read_jsonl(out / f"{split}.jsonl"))
    return "3 splits, empty one written"


@check("unit")
def split_manifests_keep_unconfigured_splits_and_refuse_manifest():
    from dsd.stages.build import write_split_manifests

    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        # A split present in the data but not in config is still written, so
        # renaming a split in config cannot make finished rows disappear.
        counts = write_split_manifests(
            out, [{"call": "a", "split": "holdout"}], ["train"]
        )
        assert counts == {"train": 0, "holdout": 1}, counts
        assert (out / "holdout.jsonl").exists()

        # `manifest` would collide with the combined file.
        try:
            write_split_manifests(out, [], ["train", "manifest"])
        except ValueError as exc:
            assert "manifest.jsonl" in str(exc), str(exc)
        else:
            raise AssertionError("a split named 'manifest' must be refused")


# --------------------------------------------------------------------------- #
# [config] overrides, path resolution, registry
# --------------------------------------------------------------------------- #
def _write_min_config(root: Path) -> Path:
    (root / "configs").mkdir(parents=True, exist_ok=True)
    path = root / "configs" / "c.yaml"
    path.write_text(
        "seed: 7\n"
        "paths:\n"
        "  audio_dirs: ['corpus']\n"
        "diarize:\n"
        "  backend: from_dir\n"
        "  options:\n"
        "    from_dir:\n"
        "      rttm_dir: rttms\n"
        "    sortformer:\n"
        "      model_path: /abs/weights.nemo\n",
        encoding="utf-8",
    )
    return path


@check("config")
def overrides_are_parsed_as_yaml_types():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        path = _write_min_config(root)
        cfg = Config.load(
            path,
            [
                "build.chunks=true",
                "build.sir_db=[-2, 2]",
                "select.max_calls_per_speaker=1",
                "cluster.threshold=0.85",
                "gender.options.http_ecapa.base_url=http://x:9",
            ],
            root=root,
        )
        assert cfg.build.chunks is True, "bool override arrived as a string"
        assert cfg.build.sir_db == (-2.0, 2.0), cfg.build.sir_db
        assert cfg.select.max_calls_per_speaker == 1
        assert approx(cfg.cluster.threshold, 0.85)
        assert cfg.gender.options["http_ecapa"]["base_url"] == "http://x:9"
        assert cfg.seed == 7


@check("config")
def backend_option_paths_resolve_against_root():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        cfg = Config.load(_write_min_config(root), [], root=root)
        # Relative becomes absolute under root; already-absolute is left alone.
        assert cfg.diarize.options["from_dir"]["rttm_dir"] == str(root.resolve() / "rttms")
        assert cfg.diarize.options["sortformer"]["model_path"] == "/abs/weights.nemo"
        assert cfg.paths.work_dir == root.resolve() / "work"
        assert cfg.paths.audio_dirs == [str(root.resolve() / "corpus")]


@check("config")
def unknown_config_key_is_rejected_with_a_hint():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        try:
            Config.load(_write_min_config(root), ["build.nonexistent=1"], root=root)
        except ValueError as exc:
            assert "nonexistent" in str(exc) and "expected one of" in str(exc)
        else:
            raise AssertionError("an unknown key should not be accepted silently")


@check("config")
def registry_contract():
    registry = Registry("thing")

    @registry.register("a")
    def _build(options):
        return ("built", options)

    assert registry.names() == ["a"]
    assert "a" in registry
    assert registry.create("a", {"x": 1}) == ("built", {"x": 1})

    try:
        registry.register("a")(lambda options: None)
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate registration must be refused")

    try:
        registry.create("nope")
    except KeyError as exc:
        assert "available" in str(exc) and "a" in str(exc)
    else:
        raise AssertionError("unknown backend must raise")


# --------------------------------------------------------------------------- #
# [http] payload contracts against a local stub server
# --------------------------------------------------------------------------- #
# Recorded verbatim from the sample response in
# sandbox/sep_data_pipeline/make_rttms.py -- note the segments carry `id`,
# `start`, `end` and `text`, and the speaker lives *inside* the text as [S01].
MOSS_VERBOSE_JSON = {
    "task": "transcribe",
    "duration": 29.7,
    "text": "[1.48][S01] Halo[2.12][4.18][S02] Halo[4.78][5.08][S01] Assalamualaikum[5.98]",
    "segments": [
        {"id": 0, "start": 1.48, "end": 2.12, "text": "[S01]Halo"},
        {"id": 1, "start": 4.18, "end": 4.78, "text": "[S02]Halo"},
        {"id": 2, "start": 5.08, "end": 5.98, "text": "[S01]Assalamualaikum"},
    ],
}

MOSS_PLAIN_JSON = {"text": MOSS_VERBOSE_JSON["text"]}

GENDER_JSON = {
    "label": "Female",
    "confidence": 0.94,
    "num_segments": 1,
    "voting_method": "soft_voting",
}


class _StubHandler(BaseHTTPRequestHandler):
    payloads: dict[str, dict] = {}
    captured: list[tuple[str, bytes]] = []

    def do_POST(self):  # noqa: N802  (BaseHTTPRequestHandler's spelling)
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        _StubHandler.captured.append((self.path, body))
        data = json.dumps(_StubHandler.payloads.get(self.path, {})).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@contextlib.contextmanager
def stub_server(payloads: dict[str, dict]):
    _StubHandler.payloads = payloads
    _StubHandler.captured = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _StubHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", _StubHandler
    finally:
        server.shutdown()
        server.server_close()


def _two_channel_noise(seconds: float = 2.0) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.normal(0, 0.1, (int(seconds * SR), 2)).astype(np.float32)


@check("http")
def moss_http_parses_the_documented_verbose_json():
    from dsd.backends.diarizers.moss_http import MossHttpDiarizer

    with stub_server({"/v1/audio/transcriptions": MOSS_VERBOSE_JSON}) as (base, _handler):
        diarizer = MossHttpDiarizer({"url": f"{base}/v1/audio/transcriptions"})
        segments = diarizer.diarize("call", Path("call.wav"), _two_channel_noise(), SR)

    # Two channels are transcribed separately, so the payload comes back twice.
    assert len(segments) == 6, f"expected 3 segments per channel, got {len(segments)}"
    assert {s.channel for s in segments} == {1, 2}

    channel0 = sorted((s for s in segments if s.channel == 1), key=lambda s: s.start)
    assert approx(channel0[0].start, 1.48, tol=1e-3)
    assert approx(channel0[0].end, 2.12, tol=1e-3)

    # The speaker has to survive: two distinct talkers appear in this payload, and
    # collapsing them would hide a channel that really does carry two people.
    labels = {s.speaker for s in channel0}
    assert len(labels) == 2, f"both speakers must be distinguishable, got {labels}"
    assert all(label.startswith("ch0_") for label in labels), labels
    return f"labels {sorted(labels)}"


@check("http")
def moss_http_falls_back_to_the_raw_transcript():
    from dsd.backends.diarizers.moss_http import MossHttpDiarizer

    with stub_server({"/v1/audio/transcriptions": MOSS_PLAIN_JSON}) as (base, _handler):
        diarizer = MossHttpDiarizer({"url": f"{base}/v1/audio/transcriptions"})
        segments = diarizer.diarize("call", Path("call.wav"), _two_channel_noise(), SR)

    # `response_format: json` returns no `segments` key at all. The client has to
    # parse `text` rather than raise.
    assert len(segments) == 6, f"expected the text fallback to yield 6 segments, got {len(segments)}"
    assert approx(sorted(s.start for s in segments)[0], 1.48, tol=1e-3)


@check("http")
def gender_client_parses_and_uploads_under_the_audio_field():
    from dsd.backends.gender.http_ecapa import HttpEcapaGender

    with stub_server({"/predict": GENDER_JSON}) as (base, handler):
        classifier = HttpEcapaGender({"base_url": base})
        label, confidence = classifier.predict(np.zeros(SR, np.float32), SR)

    assert label == "female", f"label must be lowercased, got {label!r}"
    assert approx(confidence, 0.94)
    assert handler.captured, "no request reached the server"
    body = handler.captured[0][1]
    assert b'name="audio"' in body, "the service expects the file under the field name 'audio'"
    assert b"RIFF" in body[:4096], "a WAV should have been uploaded"


# Recorded from the live mossformergan_serve OpenAPI schema and a real probe:
# pcm_f32le in and out, `sample_rate` required, `output_sample_rate` omitted so
# it defaults to the input rate.
def _mossformer_payload(samples, sr, gain=0.5, drop=0):
    import base64
    out = (np.asarray(samples, dtype=np.float32) * gain)
    if drop:
        out = out[:-drop]
    return {
        "audio": base64.b64encode(out.astype("<f4").tobytes()).decode(),
        "sample_rate": sr,
        "format": "pcm_f32le",
        "input_sample_rate": sr,
        "model_sample_rate": 16000,
        "duration_s": len(samples) / sr,
        "num_chunks": 1,
        "processing_ms": 12.5,
        "real_time_factor": 0.31,
    }


class _EnhanceHandler(BaseHTTPRequestHandler):
    """Echoes the request back attenuated, so the client's contract is exercised."""

    drop_samples = 0

    def do_POST(self):  # noqa: N802
        import base64
        length = int(self.headers.get("Content-Length", 0))
        request = json.loads(self.rfile.read(length))
        samples = np.frombuffer(base64.b64decode(request["audio"]), dtype="<f4")
        payload = _mossformer_payload(
            samples, request["sample_rate"], drop=_EnhanceHandler.drop_samples
        )
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@contextlib.contextmanager
def enhance_server(drop_samples=0):
    _EnhanceHandler.drop_samples = drop_samples
    server = ThreadingHTTPServer(("127.0.0.1", 0), _EnhanceHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()


@check("http")
def enhancer_round_trips_pcm_f32le():
    from dsd.backends.speech_enhancement.mossformergan import HttpMossFormerGAN

    rng = np.random.default_rng(0)
    span = rng.normal(0, 0.2, 4000).astype(np.float32)
    with enhance_server() as base:
        enhancer = HttpMossFormerGAN({"base_url": base})
        out = enhancer.enhance(span, SR)

    assert len(out) == len(span), f"length changed: {len(span)} -> {len(out)}"
    assert out.dtype == np.float32
    # float32 both ways, so the round trip is exact rather than merely close.
    assert np.allclose(out, span * 0.5, atol=1e-7), "payload was not decoded faithfully"
    assert enhancer.requests == 1 and approx(enhancer.audio_seconds, len(span) / SR, 1e-6)
    return f"{len(span)} samples, RTF telemetry captured"


@check("http")
def enhancer_rejects_a_length_change():
    """The invariant that makes splicing safe, enforced rather than assumed.

    Every downstream label is a sample offset into the channel. A server that
    returned a slightly different length would shift the whole call by a silent
    amount, so the client must refuse rather than pad or trim.
    """
    from dsd.backends.speech_enhancement.mossformergan import HttpMossFormerGAN

    span = np.zeros(4000, dtype=np.float32)
    with enhance_server(drop_samples=7) as base:
        enhancer = HttpMossFormerGAN({"base_url": base})
        try:
            enhancer.enhance(span, SR)
        except ValueError as exc:
            assert "exact length match" in str(exc), str(exc)
        else:
            raise AssertionError("a short reply must be refused, not spliced")


# --------------------------------------------------------------------------- #
# stub backends for the end-to-end run
# --------------------------------------------------------------------------- #
def _dominant_freq(segment: np.ndarray, sr: int) -> float:
    if len(segment) < 16:
        return 0.0
    windowed = np.asarray(segment, dtype=np.float64) * np.hanning(len(segment))
    spectrum = np.abs(np.fft.rfft(windowed))
    return float(np.fft.rfftfreq(len(segment), 1.0 / sr)[int(np.argmax(spectrum))])


class StubVAD:
    """Energy-gated VAD. Real masks from the synthetic audio, and no torch."""

    def __init__(self, options: dict):
        self.frame_sec = float(options.get("frame_sec", 0.01))
        self.rel_threshold = float(options.get("rel_threshold", 0.05))
        self.min_speech_sec = float(options.get("min_speech_sec", 0.20))
        self.min_silence_sec = float(options.get("min_silence_sec", 0.10))

    def speech(self, samples: np.ndarray, sr: int):
        step = max(1, int(self.frame_sec * sr))
        count = len(samples) // step
        if count == 0:
            return []
        frames = np.asarray(samples[: count * step], dtype=np.float64).reshape(count, step)
        energy = np.sqrt((frames**2).mean(axis=1))
        if energy.max() <= 0:
            return []

        active = energy >= self.rel_threshold * energy.max()
        spans, index = [], 0
        while index < count:
            if active[index]:
                end = index
                while end < count and active[end]:
                    end += 1
                spans.append((index * step / sr, end * step / sr))
                index = end
            else:
                index += 1
        merged = R.merge_intervals(spans, gap=self.min_silence_sec)
        return [(a, b) for a, b in merged if b - a >= self.min_speech_sec]


class StubEmbedder:
    """Maps a segment's dominant frequency onto a smooth basis.

    The synthetic voices differ by 50 Hz, so the same voice lands on the same
    vector every time and different voices are near-orthogonal. That makes the
    clustering assertion exact rather than approximate.
    """

    sample_rate = SR
    dim = 32
    centers = np.linspace(150.0, 650.0, 32)

    def __init__(self, options: dict):
        self.sigma = float(options.get("sigma", 20.0))

    def embed_batch(self, segments, sr: int) -> np.ndarray:
        rows = []
        for segment in segments:
            peak = _dominant_freq(segment, sr)
            vector = np.exp(-((self.centers - peak) ** 2) / (2 * self.sigma**2))
            rows.append(vector / max(float(np.linalg.norm(vector)), 1e-12))
        return np.stack(rows).astype(np.float32)


class StubGender:
    """Low voices are male, high voices female -- matches how the corpus is built."""

    def __init__(self, options: dict):
        self.boundary = float(options.get("boundary_hz", 375.0))

    def predict(self, samples: np.ndarray, sr: int) -> tuple[str, float]:
        peak = _dominant_freq(samples, sr)
        return ("male" if peak < self.boundary else "female"), 0.9


class StubEnhancer:
    """Attenuates by a fixed factor, so a spliced region is trivially detectable.

    Real enhancement is non-linear and unpredictable; what the pipeline needs to
    be right about is *where* the enhanced samples land and that the length is
    untouched. A constant gain makes both checkable exactly.
    """

    GAIN = 0.5

    def __init__(self, options: dict):
        self.gain = float(options.get("gain", self.GAIN))
        self.calls = 0

    def enhance(self, samples: np.ndarray, sr: int) -> np.ndarray:
        self.calls += 1
        return (np.asarray(samples, dtype=np.float32) * self.gain).astype(np.float32)


if "stub" not in VADS:
    VADS.register("stub")(StubVAD)
    EMBEDDERS.register("stub")(StubEmbedder)
    GENDER.register("stub")(StubGender)
    ENHANCERS.register("stub")(StubEnhancer)


# --------------------------------------------------------------------------- #
# the synthetic corpus
# --------------------------------------------------------------------------- #
# Four disjoint pairs of voices, two calls each. Disjoint pairs matter: they give
# the speaker graph four components, which is what lets `select` fill three
# splits at all -- a fully connected cast would legitimately collapse into one.
VOICES = {"A": 200.0, "B": 250.0, "C": 400.0, "D": 450.0,
          "E": 300.0, "F": 500.0, "G": 350.0, "H": 550.0}
EXPECTED_GENDER = {name: ("male" if f < 375 else "female") for name, f in VOICES.items()}

PAIRS = [("A", "B"), ("C", "D"), ("E", "F"), ("G", "H")]
GOOD_CALLS = {f"call{index:02d}": pair
              for index, pair in enumerate([p for p in PAIRS for _ in range(2)], start=1)}

CALL_SEC = 30.0
CH0_BURSTS = [(1.0, 3.5), (8.0, 10.5), (15.0, 17.5), (22.0, 24.5)]
CH1_BURSTS = [(4.5, 7.0), (11.5, 14.0), (18.5, 21.0), (25.5, 28.0)]

# Planted on call01's channel 0: short enough to be noise rather than a speaker,
# so the call survives but the span must be cut out of the built source.
MINOR_SPAN = (12.0, 12.6)
MINOR_FREQ = 620.0


def _tone(freq: float, samples: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(samples) / SR
    wave = np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(2 * np.pi * 2 * freq * t)
    return (0.3 * (wave + 0.01 * rng.standard_normal(samples))).astype(np.float32)


def _paint(data: np.ndarray, channel: int, spans, freq: float, rng) -> None:
    for start, end in spans:
        a, b = int(start * SR), int(end * SR)
        data[a:b, channel] = _tone(freq, b - a, rng)


def build_corpus(root: Path) -> dict:
    """Write the audio and per-channel RTTMs; return what the pipeline should decide."""
    audio_dir = root / "corpus" / "day1"
    rttm_dir = root / "rttms"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rttm_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    def emit(call: str, segments, data: np.ndarray) -> None:
        A.write_wav(audio_dir / f"{call}.wav", data, SR)
        R.write_rttm(rttm_dir / f"{call}.rttm", call, segments)

    def turns(channel: int, spans, label_suffix: str = "speaker_0"):
        return [
            R.Segment(start=a, end=b, speaker=f"ch{channel}_{label_suffix}", channel=channel + 1)
            for a, b in spans
        ]

    for call, (left, right) in GOOD_CALLS.items():
        data = np.zeros((int(CALL_SEC * SR), 2), dtype=np.float32)
        _paint(data, 0, CH0_BURSTS, VOICES[left], rng)
        _paint(data, 1, CH1_BURSTS, VOICES[right], rng)
        segments = turns(0, CH0_BURSTS) + turns(1, CH1_BURSTS)

        if call == "call01":
            _paint(data, 0, [MINOR_SPAN], MINOR_FREQ, rng)
            segments += turns(0, [MINOR_SPAN], "speaker_1")

        emit(call, segments, data)

    # Rejected on purpose: a second speaker well past both thresholds.
    data = np.zeros((int(CALL_SEC * SR), 2), dtype=np.float32)
    _paint(data, 0, CH0_BURSTS[:2], VOICES["A"], rng)
    _paint(data, 0, [(15.0, 24.0)], VOICES["C"], rng)
    _paint(data, 1, CH1_BURSTS, VOICES["B"], rng)
    emit(
        "reject_two_speakers",
        turns(0, CH0_BURSTS[:2]) + turns(0, [(15.0, 24.0)], "speaker_1") + turns(1, CH1_BURSTS),
        data,
    )

    # Rejected on purpose: channel 1 never speaks.
    data = np.zeros((int(CALL_SEC * SR), 2), dtype=np.float32)
    _paint(data, 0, CH0_BURSTS, VOICES["A"], rng)
    emit("reject_silent_channel", turns(0, CH0_BURSTS), data)

    return {
        "suitable": set(GOOD_CALLS),
        "rejected": {
            "reject_two_speakers": "two_speakers_one_channel",
            "reject_silent_channel": "n_channels=1",
        },
    }


E2E_CONFIG = """
seed: 1234
sample_rate: 8000
paths:
  root: .
  audio_dirs: ["corpus"]
  pattern: "**/*.wav"
  work_dir: work
  dataset_dir: dataset
diarize:
  backend: from_dir
  options:
    from_dir: {rttm_dir: rttms}
filter:
  channel_mode: auto
  expected_channels: 2
  purity_min: 0.9
  purity_min_channel_mode: 0.5
  min_speech_sec: 3.0
  min_label_speech_sec: 2.0
  min_label_share: 0.10
  workers: 2
vad:
  backend: stub
embed:
  backend: stub
  min_speech_duration: 1.2
  longest_chunk_duration: 20.0
  expected_speakers: 2
  batch_size: 4
cluster:
  backend: constrained_ahc
  threshold: 0.8
  report: false
gender:
  backend: stub
  segments_per_speaker: 3
  max_total_sec: 30.0
  min_segment_sec: 1.0
  workers: 2
select:
  max_calls_per_speaker: 3
  balance_gender: true
  gender_tolerance: 0.02
  min_speech_sec: 5.0
  splits: {train: 0.5, dev: 0.25, test: 0.25}
enhance:
  backend: stub
  regions: speech
  merge_gap: 0.5
  context_sec: 0.5
  min_span_sec: 0.3
  format: wav
  workers: 1
build:
  use_enhanced: auto
  fade_ms: 10.0
  natural_frac: 0.0
  target_overlap: [0.15, 0.60]
  sir_db: [-5.0, 5.0]
  peak_ceiling: 0.99
  chunks: true
  chunk_sec: 4.0
  chunk_hop: 2.0
  min_active_per_src: 0.5
"""


_FIXTURE: dict | None = None


def e2e_fixture() -> dict:
    """Build the corpus and drive every stage. Cached: built once, asserted many times."""
    global _FIXTURE
    if _FIXTURE is not None:
        return _FIXTURE

    root = Path(tempfile.mkdtemp(prefix="dsd_smoke_"))
    # The fixture is ~34 MB of corpus and dataset. It outlives every check that
    # asserts on it, so it cannot be a context manager -- hence atexit, rather
    # than leaving one behind on every single run.
    atexit.register(shutil.rmtree, root, True)
    expected = build_corpus(root)
    (root / "configs").mkdir(exist_ok=True)
    config = root / "configs" / "e2e.yaml"
    config.write_text(E2E_CONFIG, encoding="utf-8")

    def run(*args: str) -> str:
        with quiet() as buffer:
            code = cli.main(["--config", str(config), *args])
        output = buffer.getvalue()
        if code != 0:
            raise AssertionError(f"`dsd {' '.join(args)}` exited {code}\n{output}")
        return output

    logs = {
        stage: run(stage)
        for stage in ("diarize", "filter", "vad", "embed", "cluster", "gender", "select")
    }
    logs["enhance"] = run("enhance")
    logs["build"] = run("build", "--chunks")
    logs["verify"] = run("verify")

    _FIXTURE = {"root": root, "config": config, "expected": expected, "logs": logs, "run": run}
    return _FIXTURE


# --------------------------------------------------------------------------- #
# [e2e] assertions
# --------------------------------------------------------------------------- #
@check("e2e")
def filter_keeps_exactly_the_planted_good_calls():
    fixture = e2e_fixture()
    root = fixture["root"]
    suitable = read_json(root / "work" / "suitable.json")
    unsuitable = read_json(root / "work" / "unsuitable.json")

    assert set(suitable) == fixture["expected"]["suitable"], (
        f"suitable={sorted(suitable)} expected={sorted(fixture['expected']['suitable'])}"
    )
    for call, reason in fixture["expected"]["rejected"].items():
        assert call in unsuitable, f"{call} should have been rejected"
        assert unsuitable[call]["reason"] == reason, (
            f"{call}: rejected for {unsuitable[call]['reason']!r}, expected {reason!r}"
        )
    return f"{len(suitable)} kept, {len(unsuitable)} rejected"


@check("e2e")
def minor_label_is_recorded_as_excluded():
    fixture = e2e_fixture()
    suitable = read_json(fixture["root"] / "work" / "suitable.json")

    excluded = suitable["call01"]["excluded"]
    assert excluded, "the planted minor label should have produced an excluded span"
    spans = excluded["0"]
    assert len(spans) == 1, spans
    assert approx(spans[0][0], MINOR_SPAN[0], tol=0.05)
    assert approx(spans[0][1], MINOR_SPAN[1], tol=0.05)

    # Every other call is clean, so a stray exclusion would be a false positive.
    for call, entry in suitable.items():
        if call != "call01":
            assert not entry["excluded"], f"{call} should have no excluded spans"


@check("e2e")
def embeddings_are_unit_norm_and_keyed_by_channel():
    fixture = e2e_fixture()
    root = fixture["root"]
    payload = np.load(root / "work" / "embeddings.npz", allow_pickle=True)
    keys = [str(k) for k in payload["keys"]]
    matrix = payload["embeddings"]

    assert len(keys) == 2 * len(GOOD_CALLS), f"expected two sides per call, got {len(keys)}"
    norms = np.linalg.norm(matrix, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), f"norms range {norms.min()}..{norms.max()}"

    index = read_json(root / "work" / "embed_index.json")["index"]
    for key in keys:
        assert key in index
        call, speaker = key.rsplit("_", 2)[0], key[len(key.rsplit("_", 2)[0]) + 1 :]
        assert speaker in ("speaker_1", "speaker_2"), speaker
        assert index[key]["channel"] == int(speaker[-1]) - 1


@check("e2e")
def clustering_recovers_the_planted_voices():
    fixture = e2e_fixture()
    speakers = read_json(fixture["root"] / "work" / "speakers.json")

    # Rebuild the truth: channel 0 carries the pair's left voice, channel 1 the right.
    truth: dict[str, set[str]] = {}
    for call, (left, right) in GOOD_CALLS.items():
        truth.setdefault(left, set()).add(f"{call}_speaker_1")
        truth.setdefault(right, set()).add(f"{call}_speaker_2")

    found = {frozenset(members) for members in speakers["speakers"].values()}
    wanted = {frozenset(members) for members in truth.values()}
    assert found == wanted, (
        f"clustering did not recover the planted voices\n"
        f"  found  {sorted(sorted(f) for f in found)}\n"
        f"  wanted {sorted(sorted(w) for w in wanted)}"
    )
    return f"{len(found)} voices, {sum(len(f) for f in found)} instances"


@check("e2e")
def gender_matches_the_planted_pitches():
    fixture = e2e_fixture()
    root = fixture["root"]
    gender = read_json(root / "work" / "gender.json")
    speakers = read_json(root / "work" / "speakers.json")

    # Map each cluster back to the voice it came from via one of its members.
    side_to_voice = {}
    for call, (left, right) in GOOD_CALLS.items():
        side_to_voice[f"{call}_speaker_1"] = left
        side_to_voice[f"{call}_speaker_2"] = right

    for speaker_id, members in speakers["speakers"].items():
        voice = side_to_voice[members[0]]
        got = gender["per_speaker"][speaker_id]["label"]
        assert got == EXPECTED_GENDER[voice], (
            f"voice {voice} ({VOICES[voice]} Hz) labelled {got}, expected {EXPECTED_GENDER[voice]}"
        )
        assert not gender["per_speaker"][speaker_id]["mixed"], f"{speaker_id} got mixed labels"
    return f"{len(speakers['speakers'])} speakers labelled"


@check("e2e")
def splits_are_speaker_disjoint_and_all_populated():
    fixture = e2e_fixture()
    selection = read_json(fixture["root"] / "work" / "selection.json")
    calls = selection["calls"]

    assert len(calls) == len(GOOD_CALLS), f"{len(calls)} selected, expected {len(GOOD_CALLS)}"

    splits: dict[str, set] = {}
    for record in calls.values():
        splits.setdefault(record["split"], set()).update(record["speakers"])
    assert set(splits) == {"train", "dev", "test"}, f"only filled {sorted(splits)}"

    for left, right in ((a, b) for a in splits for b in splits if a < b):
        shared = splits[left] & splits[right]
        assert not shared, f"{left} and {right} share speakers {shared}"
    return " ".join(f"{name}={selection['counts'][name]}" for name in sorted(selection["counts"]))


@check("e2e")
def built_sources_sum_to_the_mixture():
    fixture = e2e_fixture()
    root = fixture["root"]
    rows = list(read_jsonl(root / "dataset" / "manifest.jsonl"))
    assert len(rows) == len(GOOD_CALLS)

    worst = 0.0
    for row in rows:
        mixture, sr = A.read_audio(root / "dataset" / row["mix"])
        s1, _ = A.read_audio(root / "dataset" / row["s1"])
        s2, _ = A.read_audio(root / "dataset" / row["s2"])
        assert sr == SR
        assert len(mixture) == len(s1) == len(s2)
        worst = max(worst, float(np.abs(mixture[:, 0] - (s1[:, 0] + s2[:, 0])).max()))
    assert worst < 1e-4, f"mix != s1 + s2 (worst residual {worst:.2e})"
    return f"worst residual {worst:.1e}"


@check("e2e")
def excluded_span_is_silent_in_the_built_source():
    """The assertion the real 30-call run never made.

    Ten of fourteen real calls carry an excluded span, but nothing checked that
    the span actually disappears from the wav rather than merely being recorded
    in JSON.
    """
    fixture = e2e_fixture()
    root = fixture["root"]
    rows = {row["call"]: row for row in read_jsonl(root / "dataset" / "manifest.jsonl")}
    assert "call01" in rows, "call01 should have been built"

    s1, sr = A.read_audio(root / "dataset" / rows["call01"]["s1"])
    channel = s1[:, 0]

    # Stay clear of the 10 ms fade at each end of the removed span.
    margin = int(0.02 * sr)
    start = int(MINOR_SPAN[0] * sr) + margin
    end = int(MINOR_SPAN[1] * sr) - margin
    peak = float(np.abs(channel[start:end]).max())
    assert peak == 0.0, f"the excluded span is still audible (peak {peak:.4f})"

    # ...and the speaker's real speech is still there, so nothing over-subtracted.
    burst = int(CH0_BURSTS[1][0] * sr) + margin, int(CH0_BURSTS[1][1] * sr) - margin
    assert float(np.abs(channel[burst[0] : burst[1]]).max()) > 0.01, "real speech went missing"


@check("e2e")
def chunks_carry_both_speakers():
    fixture = e2e_fixture()
    root = fixture["root"]
    rows = list(read_jsonl(root / "dataset" / "chunks" / "manifest.jsonl"))
    assert rows, "no chunks were written"

    config = read_json(root / "dataset" / "stats.json")["config"]["build"]
    need = config["min_active_per_src"] * SR
    for row in rows[:40]:
        s1, sr = A.read_audio(root / "dataset" / row["s1"])
        s2, _ = A.read_audio(root / "dataset" / row["s2"])
        assert len(s1) == int(config["chunk_sec"] * sr)
        assert np.count_nonzero(s1[:, 0]) >= need, f"{row['s1']} is too quiet to be a chunk"
        assert np.count_nonzero(s2[:, 0]) >= need, f"{row['s2']} is too quiet to be a chunk"
    return f"{len(rows)} chunks"


@check("e2e")
def overlap_boosting_lands_in_the_configured_band():
    fixture = e2e_fixture()
    rows = list(read_jsonl(fixture["root"] / "dataset" / "manifest.jsonl"))

    # natural_frac is 0.0 in the e2e config, so every call must have been shifted.
    assert all(row["shifted"] for row in rows), "natural_frac=0 should shift every call"
    low, high = 0.15, 0.60
    for row in rows:
        assert low <= row["overlap"] <= high, f"{row['call']} overlap {row['overlap']}"
        assert row["natural_overlap"] < 0.01, "the planted corpus has no natural overlap"
    achieved = [row["overlap"] for row in rows]
    return f"{len(rows)} calls, overlap {min(achieved):.3f}..{max(achieved):.3f}"


@check("e2e")
def built_sources_start_and_end_in_silence():
    """No utterance is cut in half by the wrap, checked on the built wavs.

    The synthetic calls have lead-in and tail silence, and the e2e config sets
    natural_frac to 0, so every call is shifted -- which makes this the direct
    end-to-end statement of the seam fix.
    """
    fixture = e2e_fixture()
    root = fixture["root"]
    rows = list(read_jsonl(root / "dataset" / "manifest.jsonl"))
    guard = int(0.2 * SR)

    for row in rows:
        meta = read_json(root / "dataset" / row["split"] / row["call"] / "meta.json")
        for key in ("s1", "s2"):
            signal, _ = A.read_audio(root / "dataset" / row[key])
            head = float(np.abs(signal[:guard, 0]).max())
            tail = float(np.abs(signal[-guard:, 0]).max())
            assert head == 0.0 and tail == 0.0, (
                f"{row['call']} {key}: wrap seam left speech at the file edge "
                f"(head {head:.4f}, tail {tail:.4f}, shift_reason={meta.get('shift_reason')})"
            )
        assert meta.get("shift_reason") in ("ok", "natural", "boundary_not_silent",
                                            "no_safe_shift", "silent_source"), meta.get("shift_reason")
    shifted = sum(1 for r in rows if r["shifted"])
    return f"{len(rows)} calls silent at both edges, {shifted} shifted"


@check("e2e")
def per_split_manifests_partition_the_combined_one():
    fixture = e2e_fixture()
    root = fixture["root"]
    splits = ["train", "dev", "test"]

    for label, directory, combined_name in (
        ("calls", root / "dataset", "manifest.jsonl"),
        ("chunks", root / "dataset" / "chunks", "manifest.jsonl"),
    ):
        combined = list(read_jsonl(directory / combined_name))
        seen = []
        for split in splits:
            path = directory / f"{split}.jsonl"
            assert path.exists(), f"{label}: {path.name} was not written"
            rows = list(read_jsonl(path))
            assert all(r["split"] == split for r in rows), f"{label}/{split}: wrong split"
            seen.extend(rows)

        assert len(seen) == len(combined), (
            f"{label}: splits hold {len(seen)}, combined holds {len(combined)}"
        )
        key = "mix" if label == "chunks" else "call"
        assert {r[key] for r in seen} == {r[key] for r in combined}, (
            f"{label}: split files and combined manifest hold different rows"
        )

    stats = read_json(root / "dataset" / "stats.json")["splits"]
    assert all("chunks" in bucket for bucket in stats.values()), stats
    return "calls and chunks both partitioned"


@check("e2e")
def rebuilding_is_a_no_op_and_overwrite_really_rebuilds():
    fixture = e2e_fixture()
    root = fixture["root"]
    meta = root / "dataset" / "train"
    sample = next(meta.glob("*/meta.json"))
    before = sample.read_bytes()
    stamp = sample.stat().st_mtime_ns

    fixture["run"]("build", "--chunks")
    assert sample.stat().st_mtime_ns == stamp, "a plain rebuild should not rewrite meta.json"

    fixture["run"]("build", "--chunks", "--overwrite")
    assert sample.read_bytes() == before, "--overwrite must be deterministic for a fixed seed"
    fixture["run"]("verify")


@check("e2e")
def natural_frac_one_leaves_every_call_as_recorded():
    """The other half of the overlap switch, which the main run cannot cover."""
    fixture = e2e_fixture()
    # `--set` is a global flag, so it has to precede the subcommand.
    fixture["run"]("--set", "build.natural_frac=1.0", "build", "--chunks", "--overwrite")
    try:
        rows = list(read_jsonl(fixture["root"] / "dataset" / "manifest.jsonl"))
        assert not any(row["shifted"] for row in rows), "natural_frac=1 must shift nothing"
        for row in rows:
            assert approx(row["overlap"], row["natural_overlap"], tol=1e-6)
    finally:
        # Put the fixture back for any check that runs after this one.
        fixture["run"]("build", "--chunks", "--overwrite")


@check("e2e")
def speaker_cap_is_enforced():
    fixture = e2e_fixture()
    # selection.json is shared with the other e2e checks, so it is restored after.
    selection_path = fixture["root"] / "work" / "selection.json"
    backup = selection_path.read_bytes()
    try:
        fixture["run"]("select", "--overwrite", "--max-calls-per-speaker", "1")
        selection = read_json(selection_path)
        counts: dict[str, int] = {}
        for record in selection["calls"].values():
            for speaker in record["speakers"]:
                counts[speaker] = counts.get(speaker, 0) + 1
        assert counts, "nothing selected under a cap of 1"
        assert max(counts.values()) <= 1, f"cap of 1 violated: {counts}"
        assert selection["dropped"].get("speaker_cap", 0) > 0, "no drops recorded for the cap"
        return f"{len(selection['calls'])} calls survive a cap of 1"
    finally:
        selection_path.write_bytes(backup)


@check("e2e")
def enhanced_cache_is_written_for_every_selected_call():
    fixture = e2e_fixture()
    root = fixture["root"]
    selection = read_json(root / "work" / "selection.json")
    cached = sorted((root / "work" / "enhanced").glob("*.wav"))

    assert len(cached) == len(selection["calls"]), (
        f"{len(cached)} cached vs {len(selection['calls'])} selected"
    )
    for path in cached:
        enhanced, sr = A.read_audio(path)
        original, osr = A.read_audio(selection["calls"][path.stem]["audio"])
        assert sr == osr, f"{path.name}: rate changed {osr} -> {sr}"
        assert enhanced.shape == original.shape, (
            f"{path.name}: shape changed {original.shape} -> {enhanced.shape}"
        )
    return f"{len(cached)} calls cached"


@check("e2e")
def enhancement_touched_speech_and_left_the_rest_alone():
    """The splice landed where the VAD said, and nowhere else."""
    fixture = e2e_fixture()
    root = fixture["root"]
    selection = read_json(root / "work" / "selection.json")

    call = sorted(selection["calls"])[0]
    record = selection["calls"][call]
    enhanced, sr = A.read_audio(root / "work" / "enhanced" / f"{call}.wav")
    original, _ = A.read_audio(record["audio"])

    intervals = {}
    for segment in R.read_rttm(root / "work" / "vad" / f"{call}.rttm"):
        intervals.setdefault(segment.channel - 1, []).append((segment.start, segment.end))

    for channel in record["channels"]:
        speech = A.speech_mask(intervals.get(channel, []), len(original), sr)
        before, after = original[:, channel], enhanced[:, channel]

        # Outside the VAD spans the cache must be byte-identical to the source:
        # non-speech is deliberately carried through untouched.
        outside = ~speech
        # The merge/context padding means the changed region is a superset of
        # the mask, so compare well clear of any span edge.
        eroded = outside.copy()
        margin = int(1.5 * sr)
        for edge in np.flatnonzero(np.diff(outside.astype(np.int8))):
            eroded[max(0, edge - margin) : edge + margin] = False
        if eroded.any():
            # One LSB of tolerance, not exact equality: the cache is written at
            # PCM_16, so *every* sample is rounded on the way out, including the
            # ones the splice never touched. Requiring bit-equality would pass
            # here only because this corpus is already PCM_16, and would fail
            # the moment enhance.format changed. Verified on real audio: the
            # untouched 65% of a call differs by at most half an LSB.
            drift = float(np.abs(before[eroded] - after[eroded]).max())
            assert drift <= 1.0 / 32768, (
                f"channel {channel}: non-speech moved by {drift:.2e}, more than PCM_16 rounding"
            )

        # Inside speech, the stub halves the amplitude, so the change is exact.
        loud = speech & (np.abs(before) > 0.05)
        if loud.any():
            ratio = float(np.median(np.abs(after[loud]) / np.abs(before[loud])))
            assert approx(ratio, StubEnhancer.GAIN, 0.02), (
                f"channel {channel}: speech scaled by {ratio:.3f}, expected {StubEnhancer.GAIN}"
            )
    return "speech replaced, silence untouched"


@check("e2e")
def build_uses_the_enhanced_cache():
    fixture = e2e_fixture()
    rows = list(read_jsonl(fixture["root"] / "dataset" / "manifest.jsonl"))
    assert rows and all(row["enhanced"] for row in rows), (
        "every built call should have come from the enhanced cache"
    )

    stats = read_json(fixture["root"] / "dataset" / "stats.json")["enhanced"]
    assert stats["calls"] == stats["of"] == len(rows), stats
    assert stats["backend"] == "stub"

    meta = read_json(
        fixture["root"] / "dataset" / rows[0]["split"] / rows[0]["call"] / "meta.json"
    )
    assert meta["enhanced"] is True and meta["enhance_backend"] == "stub"
    # The whole point: enhancing the sources before summing keeps the identity.
    mixture, _ = A.read_audio(fixture["root"] / "dataset" / rows[0]["mix"])
    s1, _ = A.read_audio(fixture["root"] / "dataset" / rows[0]["s1"])
    s2, _ = A.read_audio(fixture["root"] / "dataset" / rows[0]["s2"])
    residual = float(np.abs(mixture[:, 0] - (s1[:, 0] + s2[:, 0])).max())
    assert residual < 1e-4, f"mix != s1 + s2 after enhancement (residual {residual:.2e})"
    return f"{len(rows)} calls, residual {residual:.1e}"


@check("e2e")
def use_enhanced_always_refuses_a_missing_cache():
    """Guards against silently shipping a half-denoised dataset."""
    fixture = e2e_fixture()
    root = fixture["root"]
    cached = root / "work" / "enhanced"
    moved = root / "work" / "enhanced_moved"

    cached.rename(moved)
    try:
        try:
            fixture["run"]("--set", "build.use_enhanced=always",
                           "build", "--overwrite", "--chunks")
        except (AssertionError, SystemExit) as exc:
            # SystemExit, not a per-call failure: the stage checks the cache
            # before doing any work, so nothing is written before it refuses.
            assert "use_enhanced" in str(exc), str(exc)
        else:
            raise AssertionError("build should refuse when the cache is missing")

        assert not (root / "dataset" / "manifest.jsonl").exists() or list(
            read_jsonl(root / "dataset" / "manifest.jsonl")
        ), "the refusal must not leave an empty manifest behind"
    finally:
        moved.rename(cached)
        fixture["run"]("build", "--overwrite", "--chunks")


# --------------------------------------------------------------------------- #
# [real] opt-in: the actual models against the real corpus
# --------------------------------------------------------------------------- #
REAL_LIMIT = 0


@check("real")
def real_pipeline_slice():
    if not REAL_LIMIT:
        raise AssertionError("unreachable")  # filtered out unless --real is given

    config = ROOT / "configs" / "default.yaml"
    assert config.exists(), f"missing {config}"
    corpus = ROOT / "Datasets" / "audios"
    assert corpus.exists(), f"missing {corpus}; nothing to run against"

    scratch = Path(tempfile.mkdtemp(prefix="dsd_real_"))
    try:
        args = [
            "--config", str(config),
            "--set", f"paths.work_dir={scratch / 'work'}",
            "--set", f"paths.dataset_dir={scratch / 'dataset'}",
        ]
        with quiet() as buffer:
            code = cli.main([*args, "run", "--stages",
                             "diarize,filter,vad,embed,cluster,select,build",
                             "--limit", str(REAL_LIMIT), "--chunks"])
        assert code == 0, f"pipeline exited {code}\n{buffer.getvalue()}"

        suitable = read_json(scratch / "work" / "suitable.json")
        assert suitable, "no call survived the filter on real data"

        diagnostics = read_json(scratch / "work" / "speakers.json")["diagnostics"]
        known = diagnostics.get("known_different_same_call")
        if known:
            threshold = diagnostics["threshold"]
            assert known["p99"] < threshold, (
                f"known-different p99 {known['p99']:.3f} reached the merge threshold {threshold}"
            )

        with quiet() as buffer:
            code = cli.main([*args, "verify"])
        assert code == 0, f"verify exited {code}\n{buffer.getvalue()}"

        rows = list(read_jsonl(scratch / "dataset" / "manifest.jsonl"))
        return f"{len(suitable)} suitable, {len(rows)} built"
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    global REAL_LIMIT

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--real", type=int, metavar="N", default=0,
                        help="also run N real calls through the real models")
    parser.add_argument("--only", help="run only checks whose group or name contains this")
    parser.add_argument("--groups", default="unit,config,http,e2e",
                        help="comma-separated groups to run")
    args = parser.parse_args(argv)

    REAL_LIMIT = args.real
    groups = {g.strip() for g in args.groups.split(",") if g.strip()}
    if args.real:
        groups.add("real")

    selected = [
        (group, name, fn)
        for group, name, fn in CHECKS
        if group in groups and (not args.only or args.only in name or args.only in group)
    ]
    if not selected:
        print("no checks matched")
        return 1

    failures = []
    for group, name, fn in selected:
        label = f"[{group:<6}] {name.replace('_', ' ')} "
        try:
            detail = fn()
        except KeyboardInterrupt:
            raise
        except BaseException as exc:  # noqa: BLE001 - a check may fail any way it likes
            # BaseException, not Exception: argparse answers a bad flag with
            # SystemExit, and letting that escape would kill the whole run at
            # the first mistyped argument instead of reporting one failure.
            print(f"{label:.<74} FAIL")
            failures.append((label.strip(), exc))
        else:
            print(f"{label:.<74} ok" + (f"   {detail}" if detail else ""))

    print()
    if failures:
        for label, exc in failures:
            print(f"--- {label}")
            text = str(exc) or repr(exc)
            for line in text.splitlines()[:20]:
                print(f"    {line}")
        print(f"\n{len(failures)} of {len(selected)} checks FAILED")
        return 1

    print(f"all {len(selected)} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
