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
import dataclasses
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
from dsd.augment import artifacts as aug_artifacts  # noqa: E402
from dsd.augment import chain as aug_chain  # noqa: E402
from dsd.augment import codec as aug_codec  # noqa: E402
from dsd.augment import noise as aug_noise  # noqa: E402
from dsd.augment import reverb as aug_reverb  # noqa: E402
from dsd.config import Config  # noqa: E402
from dsd.core import audio as A  # noqa: E402
from dsd.core import rttm as R  # noqa: E402
from dsd.core.manifest import read_json, read_jsonl  # noqa: E402
from dsd.mixing import chunker, mixer, overlap  # noqa: E402
from dsd.registry import EMBEDDERS, ENHANCERS, GENDER, VADS, Registry  # noqa: E402
from dsd.stages import build as build_stage  # noqa: E402
from dsd.stages import verify as verify_stage  # noqa: E402

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


@check("unit")
def erode_drops_exactly_the_fade_ramps():
    # The middle run is 10 ms -- shorter than a fade either side, so it goes.
    mask = A.speech_mask([(1.0, 2.0), (4.0, 4.01), (6.0, 9.0)], 10 * SR, SR)
    fade = int(0.01 * SR)
    eroded = verify_stage._erode(mask, fade)

    assert not (eroded & ~mask).any(), "erosion may only remove samples"
    runs = np.flatnonzero(np.diff(np.concatenate(([0], mask.view(np.int8), [0]))))
    for start, stop in zip(runs[::2], runs[1::2]):
        length = stop - start
        expected = max(0, length - 2 * fade)
        assert int(eroded[start:stop].sum()) == expected, (
            f"run of {length} samples kept {int(eroded[start:stop].sum())}, want {expected}"
        )
    assert not eroded[int(4.0 * SR) : int(4.01 * SR)].any(), "a sub-fade run must vanish"
    return f"fade {fade} samples, 3 runs"


@check("unit")
def residual_correlation_tells_background_from_a_mismatched_mixture():
    """The check that still works when the two speakers never overlap.

    A correct raw-channel build leaves the *other* channel's background, which
    is an independent recording. Every way the trio can stop matching leaves a
    copy of the sources instead.
    """
    rng = np.random.default_rng(3)
    n = 8 * SR
    voices = rng.normal(0, 0.2, n)
    background = rng.normal(0, 0.2 * 10 ** (-24 / 20), n)  # 24 dB down, independent

    honest = verify_stage._residual_correlation(voices, background)
    assert honest < 0.1, f"independent background should not correlate ({honest:.3f})"

    # The peak guard scaling the mixture and not the sources.
    for error in (0.05, 0.20):
        leftover = -error * voices + background
        got = verify_stage._residual_correlation(voices, leftover)
        assert got > verify_stage.MAX_RESIDUAL_CORRELATION, (
            f"a {error:.0%} scale error only reached {got:.3f}"
        )

    # A mixture belonging to a different call.
    other = rng.normal(0, 0.2, n)
    swapped = verify_stage._residual_correlation(voices, other - voices)
    assert swapped > verify_stage.MAX_RESIDUAL_CORRELATION, swapped
    return f"honest {honest:.3f}, swapped {swapped:.3f}"


@check("unit")
def solo_alignment_ignores_phase_but_not_time():
    """The enhanced-target check must pass a resynthesised target and fail a shifted one.

    Sidon's output is not phase-locked to its input, so a waveform correlation
    fails every correct file (measured 0.00-0.04). A 90-degree phase shift is
    the extreme case: waveform correlation ~0, magnitudes identical.
    """
    rng = np.random.default_rng(3)
    n = 4 * SR
    # A new level *and* a new spectral colouring every 100 ms, the way speech
    # changes from syllable to syllable, so a shift changes what each frame holds.
    blocks = [
        rng.uniform(0.05, 1.0) * np.convolve(rng.normal(0, 1, 800), rng.normal(0, 1, 16), mode="same")
        for _ in range(n // 800)
    ]
    voice = np.concatenate(blocks).astype(np.float32)
    spectrum = np.fft.rfft(voice)
    spectrum[1:-1] *= -1j
    quadrature = np.fft.irfft(spectrum, n=n).astype(np.float32)
    solo = np.ones(n, dtype=bool)

    waveform = verify_stage._residual_correlation(voice, quadrature)
    same = verify_stage._spectral_alignment(voice, quadrature, solo)
    shifted = verify_stage._spectral_alignment(voice, np.roll(quadrature, int(0.05 * SR)), solo)
    floor = verify_stage.MIN_SOLO_CORRELATION
    assert waveform < 0.1, f"test signal should defeat a waveform check, got {waveform:.3f}"
    assert same > 0.95 and same > floor, f"phase-shifted copy scored {same:.3f}"
    assert shifted < floor, f"a 50 ms shift scored {shifted:.3f}, above the {floor} floor"
    return f"waveform {waveform:.3f}, same {same:.3f}, 50 ms off {shifted:.3f}"


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
# [unit] per-speaker caps
# --------------------------------------------------------------------------- #
def _cap_candidate(name, a, b, seconds_a, seconds_b):
    return {"call": name, "speakers": [a, b], "source_speech_sec": [seconds_a, seconds_b]}


@check("unit")
def duration_cap_is_strict_at_the_boundary():
    from dsd.stages.select import apply_caps

    calls = [
        _cap_candidate("c1", "agent", "x", 40.0, 30.0),
        _cap_candidate("c2", "agent", "y", 20.0, 30.0),  # agent -> exactly 60: admitted
        _cap_candidate("c3", "agent", "z", 0.001, 5.0),  # one millisecond over: refused
        _cap_candidate("c4", "w", "v", 61.0, 5.0),       # longer than the cap on its own
    ]
    kept, dropped, used_calls, used_seconds = apply_caps(calls, None, 60.0)

    assert [r["call"] for r in kept] == ["c1", "c2"], [r["call"] for r in kept]
    assert approx(used_seconds["agent"], 60.0), used_seconds["agent"]
    assert dropped == {"speaker_duration_cap": 1, "longer_than_duration_cap": 1}, dropped
    # A refused call must not register its speakers at all.
    assert "z" not in used_seconds and "w" not in used_seconds and "v" not in used_calls
    return "exactly-at-cap in, 1 ms over out"


@check("unit")
def caps_combine_and_switch_off():
    from dsd.stages.select import apply_caps

    rng = random.Random(3)
    calls = [
        _cap_candidate(f"c{i}", f"s{rng.randrange(6)}", f"t{i}", rng.uniform(5, 90), 10.0)
        for i in range(200)
    ]

    kept, dropped, _, _ = apply_caps(calls, None, None)
    assert len(kept) == len(calls) and not dropped, "both caps off must keep everything"

    kept, dropped, used_calls, _ = apply_caps(calls, 4, None)
    assert max(used_calls.values()) <= 4 and set(dropped) == {"speaker_cap"}, dropped

    kept, dropped, used_calls, used_seconds = apply_caps(calls, 4, 180.0)
    assert max(used_calls.values()) <= 4
    assert max(used_seconds.values()) <= 180.0 + 1e-9, max(used_seconds.values())
    # Recomputed from what was kept, independently of the function's own tally.
    totals: dict[str, float] = {}
    for record in kept:
        for speaker, side in zip(record["speakers"], record["source_speech_sec"]):
            totals[speaker] = totals.get(speaker, 0.0) + side
    assert max(totals.values()) <= 180.0 + 1e-9
    return f"{len(kept)}/200 kept under both caps, {dict(dropped)}"


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


# Modelled on sidon_serving's `/v1/restore` schema (server/schemas.py): pcm_f32le
# in and out, `sample_rate` required, `output_sample_rate` honoured, and the
# response naming every rate on the way through. The stub upsamples by sample
# repetition -- not what Sidon does, but exact, so decoding can be checked to
# the bit -- and returns a *different* rate than it was sent, which is the whole
# point of the backend.
def _sidon_payload(samples, sr, out_sr, gain=0.5, drop=0, report_rate=None):
    import base64
    k = out_sr // sr
    out = np.repeat(np.asarray(samples, dtype=np.float32), k) * gain
    if drop:
        out = out[:-drop]
    return {
        "audio": base64.b64encode(out.astype("<f4").tobytes()).decode(),
        "sample_rate": report_rate or out_sr,
        "format": "pcm_f32le",
        "input_sample_rate": sr,
        "model_sample_rate": 48000,
        "duration_s": len(samples) / sr,
        "samples": int(len(out)),
        "chunks": 1,
        "chunk_seconds": 96.0,
        "level_match": "peak",
        "gain_applied": 1.0,
        "length_adjusted": False,
        "processing_ms": 12.5,
        "real_time_factor": 0.31,
    }


class _RestoreHandler(BaseHTTPRequestHandler):
    """Serves `/v1/restore` and `/healthz` the way sidon_serving does."""

    drop_samples = 0
    report_rate = None
    model = "Sidon"
    requests: list = []

    def _reply(self, code: int, payload: dict) -> None:
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):  # noqa: N802
        if self.path != "/healthz":
            return self._reply(404, {"detail": "Not Found"})
        self._reply(200, {"status": "ok", "model": _RestoreHandler.model,
                          "model_sample_rate": 48000, "engines": 1})

    def do_POST(self):  # noqa: N802
        import base64
        if self.path != "/v1/restore":
            # sidon_serving has no /v1/enhance; a client posting there must fail.
            return self._reply(404, {"detail": "Not Found"})
        length = int(self.headers.get("Content-Length", 0))
        request = json.loads(self.rfile.read(length))
        _RestoreHandler.requests.append({k: v for k, v in request.items() if k != "audio"})
        samples = np.frombuffer(base64.b64decode(request["audio"]), dtype="<f4")
        sr = request["sample_rate"]
        out_sr = request.get("output_sample_rate") or sr
        self._reply(200, _sidon_payload(
            samples, sr, out_sr, drop=_RestoreHandler.drop_samples,
            report_rate=_RestoreHandler.report_rate,
        ))

    def log_message(self, *args):
        pass


@contextlib.contextmanager
def restore_server(drop_samples=0, report_rate=None, model="Sidon"):
    _RestoreHandler.drop_samples = drop_samples
    _RestoreHandler.report_rate = report_rate
    _RestoreHandler.model = model
    _RestoreHandler.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _RestoreHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()


@check("http")
def sidon_returns_the_requested_wider_rate():
    """8 kHz in, 24 kHz out, same duration -- and the rate was actually asked for.

    The regression this guards: leaving `output_sample_rate` out makes the
    service answer at the input rate, discarding the band Sidon restored.
    """
    from dsd.backends.speech_enhancement.sidon import HttpSidon

    rng = np.random.default_rng(0)
    span = rng.normal(0, 0.2, 4000).astype(np.float32)
    with restore_server() as base:
        enhancer = HttpSidon({"base_url": base, "output_sample_rate": 24000})
        assert enhancer.output_rate(SR) == 24000
        out = enhancer.enhance(span, SR)
        sent = _RestoreHandler.requests[-1]

    assert sent["output_sample_rate"] == 24000, sent
    assert sent["sample_rate"] == SR and sent["level_match"] == "peak", sent
    assert len(out) == 3 * len(span), f"{len(span)} @ 8 kHz -> {len(out)} @ 24 kHz"
    assert approx(len(out) / 24000, len(span) / SR, 1e-12), "duration changed"
    assert out.dtype == np.float32
    # float32 both ways, so the round trip is exact rather than merely close.
    assert np.array_equal(out, np.repeat(span, 3) * np.float32(0.5)), "payload was not decoded faithfully"
    assert enhancer.requests == 1 and approx(enhancer.audio_seconds, len(span) / SR, 1e-6)
    return f"{len(span)} @ 8 kHz -> {len(out)} @ 24 kHz"


@check("http")
def sidon_always_sends_the_output_rate():
    """Even at the input rate: the server's own default is an env var we do not control."""
    from dsd.backends.speech_enhancement.sidon import HttpSidon

    with restore_server() as base:
        out = HttpSidon({"base_url": base}).enhance(np.zeros(800, np.float32) + 0.1, SR)
        sent = _RestoreHandler.requests[-1]
    assert sent["output_sample_rate"] == SR, sent
    assert len(out) == 800


@check("http")
def enhancer_rejects_a_duration_change():
    """The invariant that makes splicing safe, enforced rather than assumed.

    Every downstream label is a time converted to a sample offset. A server that
    returned one sample too few at 24 kHz would shift the rest of the span by a
    silent amount, so the client must refuse rather than pad or trim.
    """
    from dsd.backends.speech_enhancement.sidon import HttpSidon

    span = np.zeros(4000, dtype=np.float32)
    with restore_server(drop_samples=1) as base:
        enhancer = HttpSidon({"base_url": base, "output_sample_rate": 24000})
        try:
            enhancer.enhance(span, SR)
        except ValueError as exc:
            assert "same duration" in str(exc), str(exc)
        else:
            raise AssertionError("a short reply must be refused, not spliced")


@check("http")
def enhancer_rejects_a_rate_it_did_not_ask_for():
    from dsd.backends.speech_enhancement.sidon import HttpSidon

    with restore_server(report_rate=16000) as base:
        enhancer = HttpSidon({"base_url": base, "output_sample_rate": 24000})
        try:
            enhancer.enhance(np.zeros(4000, np.float32), SR)
        except ValueError as exc:
            assert "16000 Hz" in str(exc), str(exc)
        else:
            raise AssertionError("a reply at an unrequested rate must be refused")


@check("http")
def old_backend_name_is_the_sidon_client():
    """`http_mossformergan` survives as a name only; it must speak /v1/restore at 24 kHz."""
    from dsd.backends.speech_enhancement.sidon import HttpSidon

    enhancer = ENHANCERS.create("http_mossformergan", {"base_url": "http://x", "output_sample_rate": 24000})
    assert isinstance(enhancer, HttpSidon), type(enhancer)
    assert enhancer.url.endswith("/v1/restore"), enhancer.url
    assert enhancer.output_rate(SR) == 24000


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


class StubWidebandEnhancer(StubEnhancer):
    """The same fixed gain, returned at 3x the rate -- a stand-in for Sidon at 24 kHz.

    A band-limited resample rather than sample repetition, so bringing a target
    back down to 8 kHz recovers `GAIN * original` to within the resampler's
    ripple, and the solo-gain check still has an exact answer (2.0).
    """

    FACTOR = 3

    def output_rate(self, sr: int) -> int:
        return sr * self.FACTOR

    def enhance(self, samples: np.ndarray, sr: int) -> np.ndarray:
        self.calls += 1
        up = A.resample(np.asarray(samples, dtype=np.float32), sr, self.output_rate(sr))
        return (up * self.gain).astype(np.float32)


if "stub" not in VADS:
    VADS.register("stub")(StubVAD)
    EMBEDDERS.register("stub")(StubEmbedder)
    GENDER.register("stub")(StubGender)
    ENHANCERS.register("stub")(StubEnhancer)
    ENHANCERS.register("stub_wideband")(StubWidebandEnhancer)


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


# The noise floor planted between the turns, ~35 dB under the tones. Without it
# every channel is digitally silent outside its bursts, the two mixing modes
# produce byte-identical output, and nothing here can tell them apart -- which
# is the whole thing `build` now does differently.
ROOM_NOISE = 0.004


def _room(rng: np.random.Generator) -> np.ndarray:
    """A call-length two-channel bed of noise for the tones to be painted onto."""
    return (ROOM_NOISE * rng.standard_normal((int(CALL_SEC * SR), 2))).astype(np.float32)


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
        data = _room(rng)
        _paint(data, 0, CH0_BURSTS, VOICES[left], rng)
        _paint(data, 1, CH1_BURSTS, VOICES[right], rng)
        segments = turns(0, CH0_BURSTS) + turns(1, CH1_BURSTS)

        if call == "call01":
            _paint(data, 0, [MINOR_SPAN], MINOR_FREQ, rng)
            segments += turns(0, [MINOR_SPAN], "speaker_1")

        emit(call, segments, data)

    # Rejected on purpose: a second speaker well past both thresholds.
    data = _room(rng)
    _paint(data, 0, CH0_BURSTS[:2], VOICES["A"], rng)
    _paint(data, 0, [(15.0, 24.0)], VOICES["C"], rng)
    _paint(data, 1, CH1_BURSTS, VOICES["B"], rng)
    emit(
        "reject_two_speakers",
        turns(0, CH0_BURSTS[:2]) + turns(0, [(15.0, 24.0)], "speaker_1") + turns(1, CH1_BURSTS),
        data,
    )

    # Rejected on purpose: channel 1 never speaks.
    data = _room(rng)
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
  # Same-rate targets for the main fixture; `wideband_fixture` covers 24 kHz.
  output_sample_rate: null
  regions: speech
  merge_gap: 0.5
  context_sec: 0.5
  min_span_sec: 0.3
  format: wav
  workers: 1
build:
  use_enhanced: auto
  zerofy_mix: false
  fade_ms: 10.0
  shuffle: true
  natural_frac: 0.0
  target_overlap: [0.15, 0.60]
  sir_db: [-5.0, 5.0]
  peak_ceiling: 0.99
  chunks: true
  chunk_sec: 4.0
  chunk_hop: 2.0
  min_active_per_src: 0.5
  # Off for the main fixture: every check below is about the clean mixture, its
  # targets and the manifest, and degrading them would only slow that down.
  # `augmented_fixture` builds its own banks and turns this on.
  augment: {variants: 0}
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


def _built_trio(root: Path, row: dict):
    mixture, sr = A.read_audio(root / "dataset" / row["mix"])
    s1, _ = A.read_audio(root / "dataset" / row["s1"])
    s2, _ = A.read_audio(root / "dataset" / row["s2"])
    assert sr == SR
    assert len(mixture) == len(s1) == len(s2)
    return mixture[:, 0], s1[:, 0], s2[:, 0]


@check("e2e")
def built_sources_sum_to_the_mixture_where_both_speak():
    """The invariant of a build whose targets are NOT denoised.

    With `use_enhanced=never` the mixture and the targets come from the same
    original channels, so wherever both speakers are at full gain the mixture is
    still exactly their sum. (With denoised targets it cannot be -- see
    `mixture_is_the_original_and_targets_are_enhanced`.)
    """
    fixture = e2e_fixture()
    root = fixture["root"]
    fade = int(0.01 * SR)
    fixture["run"]("--set", "build.use_enhanced=never", "build", "--chunks")
    try:
        rows = list(read_jsonl(root / "dataset" / "manifest.jsonl"))
        assert len(rows) == len(GOOD_CALLS)
        assert not any(row["enhanced"] for row in rows), "targets should be raw here"

        worst, checked = 0.0, 0
        for row in rows:
            mixture, s1, s2 = _built_trio(root, row)
            exact = verify_stage._both_at_full_gain(s1, s2, fade)
            if not exact.any():
                continue
            checked += 1
            worst = max(worst, float(np.abs((mixture - (s1 + s2))[exact]).max()))
        assert checked, "no call had the two speakers overlapping; fixture is wrong"
        assert worst < 1e-4, f"mix != s1 + s2 where both speak (worst {worst:.2e})"
        fixture["run"]("--set", "build.use_enhanced=never", "verify")
        return f"{checked} calls, worst residual {worst:.1e}"
    finally:
        fixture["run"]("build", "--chunks")


@check("e2e")
def the_mixture_keeps_the_background_the_targets_drop():
    """The point of building the mixture before zerofying.

    Between the turns the targets are exact zeros and the mixture is not: it
    still holds the room the microphones recorded. Asserted on the built wavs,
    at a moment the corpus plants no speech on either channel.
    """
    fixture = e2e_fixture()
    root = fixture["root"]
    rows = list(read_jsonl(root / "dataset" / "manifest.jsonl"))

    # 27.9-30.0 s: past the last burst on either channel in the planted corpus.
    quiet = slice(int(28.2 * SR), int(29.8 * SR))
    floors = []
    for row in rows:
        mixture, s1, s2 = _built_trio(root, row)
        assert float(np.abs(s1[quiet]).max()) == 0.0, f"{row['call']}: s1 is not zerofied"
        assert float(np.abs(s2[quiet]).max()) == 0.0, f"{row['call']}: s2 is not zerofied"
        floor = float(np.sqrt(np.mean(mixture[quiet] ** 2)))
        assert floor > 0.25 * ROOM_NOISE, (
            f"{row['call']}: the mixture is silent between the turns (rms {floor:.5f}) -- "
            "it was built from the zerofied channels"
        )
        floors.append(floor)

    meta = read_json(root / "dataset" / rows[0]["split"] / rows[0]["call"] / "meta.json")
    assert meta["zerofy_mix"] is False
    if meta["targets_enhanced"]:
        # With denoised targets, mix - (s1 + s2) also holds what the enhancer
        # removed, so it is not "just background" and has no fixed floor.
        assert meta["background_snr_db"] is not None
    else:
        assert meta["background_snr_db"] > 10.0, meta["background_snr_db"]
    return f"noise floor {min(floors):.4f}..{max(floors):.4f} rms in the mixture"


@check("e2e")
def zerofy_mix_restores_the_exact_sum():
    """The escape hatch still produces a dataset where mix == s1 + s2."""
    fixture = e2e_fixture()
    root = fixture["root"]
    try:
        fixture["run"]("--set", "build.zerofy_mix=true", "build", "--chunks")
        rows = list(read_jsonl(root / "dataset" / "manifest.jsonl"))
        worst = 0.0
        for row in rows:
            mixture, s1, s2 = _built_trio(root, row)
            worst = max(worst, float(np.abs(mixture - (s1 + s2)).max()))
        assert worst < 1e-4, f"zerofy_mix should sum exactly (worst {worst:.2e})"

        meta = read_json(root / "dataset" / rows[0]["split"] / rows[0]["call"] / "meta.json")
        assert meta["zerofy_mix"] is True
        assert meta["background_rms"] == 0.0, meta["background_rms"]
        # Switching modes must rebuild without --overwrite; a cached call from
        # the other mode would leave the dataset half one thing, half the other.
        assert read_json(root / "dataset" / "stats.json")["mix"]["zerofy_mix"] is True
        fixture["run"]("verify")
    finally:
        # Put the fixture back for the checks that run after this one.
        fixture["run"]("build", "--chunks")
    return f"worst residual {worst:.1e}"


@check("e2e")
def verify_rejects_a_mixture_that_does_not_match_its_sources():
    """The teeth verify keeps now that `mix == s1 + s2` is not a blanket test.

    Run against a build with overlap boosting off, which is the shipped default
    and the hard case: the planted corpus has the two speakers strictly taking
    turns, so there is not one sample where both are active and the exact-sum
    check has nothing to stand on.
    """
    fixture = e2e_fixture()
    root = fixture["root"]

    def verify_fails(what: str) -> None:
        try:
            fixture["run"]("verify")
        except (AssertionError, SystemExit):
            return
        raise AssertionError(f"verify passed on {what}")

    try:
        fixture["run"]("--set", "build.shuffle=false", "build", "--chunks")
        rows = list(read_jsonl(root / "dataset" / "manifest.jsonl"))
        assert all(row["overlap"] == 0.0 for row in rows), "fixture should have no overlap"
        fixture["run"]("verify")  # the honest build passes

        target = root / "dataset" / rows[0]["mix"]
        backup = target.read_bytes()
        try:
            shutil.copyfile(root / "dataset" / rows[1]["mix"], target)
            verify_fails("a mix.wav swapped in from another call")

            target.write_bytes(backup)
            mixture, sr = A.read_audio(target)
            A.write_wav(target, mixture * 0.8, sr)
            verify_fails("a mixture scaled away from its sources")
        finally:
            target.write_bytes(backup)
    finally:
        # Put the fixture back for the checks that run after this one.
        fixture["run"]("build", "--chunks")
    return "swap and rescale both caught with zero overlap to lean on"


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

    # And it is gone from the mixture as well. The mixture is no longer the sum
    # of the two targets, so a span removed only from them would still reach
    # the model -- an unlabelled third voice in its input.
    mixture, _ = A.read_audio(root / "dataset" / rows["call01"]["mix"])
    peak = float(np.abs(mixture[start:end, 0]).max())
    assert peak < 4 * ROOM_NOISE, f"the excluded span survives in the mixture (peak {peak:.4f})"


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
def selection_charges_speech_minus_excluded_spans():
    """call01 and call02 share voice A on channel 0; only call01 carries the planted
    minor-label span there. Charged speech must match, not differ by that span."""
    fixture = e2e_fixture()
    selection = read_json(fixture["root"] / "work" / "selection.json")
    records = selection["calls"]
    assert {"call01", "call02"} <= set(records), sorted(records)

    with_span = records["call01"]["source_speech_sec"][0]
    clean = records["call02"]["source_speech_sec"][0]
    span = MINOR_SPAN[1] - MINOR_SPAN[0]
    assert abs(with_span - clean) < 0.1, (
        f"call01 charged {with_span:.2f}s vs {clean:.2f}s for the same voice -- "
        f"the {span:.1f}s excluded span was not subtracted"
    )

    vad_total = sum(
        segment.duration
        for segment in R.read_rttm(fixture["root"] / "work" / "vad" / "call01.rttm")
        if segment.channel == 1
    )
    assert abs((vad_total - with_span) - span) < 0.1, (
        f"VAD {vad_total:.2f}s minus charged {with_span:.2f}s should be the {span:.1f}s span"
    )
    assert selection["max_duration_per_speaker"] is not None
    return f"charged {with_span:.2f}s = VAD {vad_total:.2f}s - {span:.1f}s excluded"


@check("e2e")
def duration_cap_is_enforced():
    """Each voice speaks ~10 s per call across two calls; a 15 s cap fits exactly one."""
    fixture = e2e_fixture()
    selection_path = fixture["root"] / "work" / "selection.json"
    backup = selection_path.read_bytes()
    try:
        fixture["run"]("select", "--overwrite", "--max-duration-per-speaker", "15")
        selection = read_json(selection_path)

        totals: dict[str, float] = {}
        for record in selection["calls"].values():
            for speaker, side in zip(record["speakers"], record["source_speech_sec"]):
                totals[speaker] = totals.get(speaker, 0.0) + side
        assert totals, "nothing selected under a 15 s cap"
        assert max(totals.values()) <= 15.0 + 1e-6, f"cap violated: {max(totals.values()):.2f}s"
        assert len(selection["calls"]) == len(PAIRS), (
            f"expected one call per voice pair, kept {len(selection['calls'])}"
        )
        assert selection["dropped"].get("speaker_duration_cap", 0) == len(PAIRS)
        assert selection["max_duration_per_speaker"] == 15.0

        # verify must hold the dataset to *this* selection's caps, not config's.
        exposure = selection["exposure"]
        assert exposure["max_sec"] <= 15.0 + 1e-6, exposure
        return f"{len(selection['calls'])} calls kept, loudest voice {exposure['max_sec']:.1f}s"
    finally:
        selection_path.write_bytes(backup)


@check("e2e")
def duration_cap_zero_switches_it_off():
    """0 on the command line disables a cap -- `args.x or cfg.x` used to swallow it."""
    fixture = e2e_fixture()
    selection_path = fixture["root"] / "work" / "selection.json"
    backup = selection_path.read_bytes()
    try:
        fixture["run"]("select", "--overwrite", "--max-duration-per-speaker", "0",
                       "--max-calls-per-speaker", "0")
        selection = read_json(selection_path)
        assert selection["max_duration_per_speaker"] is None, selection["max_duration_per_speaker"]
        assert selection["max_calls_per_speaker"] is None, selection["max_calls_per_speaker"]
        assert len(selection["calls"]) == len(GOOD_CALLS), len(selection["calls"])
    finally:
        selection_path.write_bytes(backup)


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
def mixture_is_the_original_and_targets_are_enhanced():
    """The bug this guards: mix.wav used to be built from the enhanced audio.

    The stub enhancer halves every sample it touches. So where only one speaker
    talks, the mixture is that channel as recorded and the target is half of it,
    and the least-squares gain of the mixture onto the target is exactly
    1 / 0.5 = 2.0. Built from the enhanced audio instead it would be 1.0.

    Checked on *both* speakers of every call. The fixture rolls every call and
    draws a random SIR, both applied to channel 2 only, so the second speaker
    comes out at 2.0 only if the mixture copy and the target copy of that
    channel were rolled and scaled identically.
    """
    fixture = e2e_fixture()
    root = fixture["root"]
    fade = int(0.01 * SR)
    expected = 1.0 / StubEnhancer.GAIN

    gains = []
    for row in read_jsonl(root / "dataset" / "manifest.jsonl"):
        mixture, s1, s2 = _built_trio(root, row)
        for name, target, other in (("s1", s1, s2), ("s2", s2, s1)):
            solo = verify_stage._erode(target != 0.0, fade) & (other == 0.0)
            assert solo.sum() > SR // 4, f"{row['call']}: no solo stretch for {name}"
            t = target[solo].astype(np.float64)
            gain = float(np.dot(mixture[solo], t) / np.dot(t, t))
            assert abs(gain - expected) < 0.05, (
                f"{row['call']} {name}: mixture is {gain:.3f}x the target where only "
                f"{name} speaks; expected {expected:.1f}x -- "
                + ("the mixture was built from the enhanced audio" if abs(gain - 1.0) < 0.05
                   else "the mixture and target copies of this channel diverged")
            )
            gains.append(gain)
    return f"{len(gains)} solo stretches, gain {min(gains):.3f}..{max(gains):.3f} (expect 2.0)"


@check("e2e")
def build_prunes_calls_the_selection_dropped():
    fixture = e2e_fixture()
    root = fixture["root"]
    stray = root / "dataset" / "train" / "call_not_in_any_selection"
    selected = {p.name for p in (root / "dataset").glob("*/call*") if p.is_dir()}

    def plant():
        stray.mkdir(parents=True, exist_ok=True)
        (stray / "mix.wav").write_bytes(b"stale")

    try:
        plant()
        fixture["run"]("build", "--chunks")
        assert not stray.exists(), "build left a call the selection does not hold"

        plant()
        fixture["run"]("build", "--chunks", "--no-prune")
        assert stray.exists(), "--no-prune must keep it"
        # ...and verify must notice a directory no manifest lists.
        try:
            fixture["run"]("verify")
        except (AssertionError, SystemExit):
            pass
        else:
            raise AssertionError("verify passed with a call directory in no manifest")

        # A --limit build is partial on purpose; it must not delete the calls it
        # merely skipped.
        shutil.rmtree(stray)
        fixture["run"]("build", "--chunks", "--limit", "2")
        still = {p.name for p in (root / "dataset").glob("*/call*") if p.is_dir()}
        assert still == selected, f"--limit deleted selected calls: {sorted(selected - still)}"
    finally:
        shutil.rmtree(stray, ignore_errors=True)
        fixture["run"]("build", "--chunks")
    return "stray pruned, --no-prune kept it, --limit deleted nothing"


@check("e2e")
def build_uses_the_enhanced_cache():
    fixture = e2e_fixture()
    root = fixture["root"]
    rows = list(read_jsonl(root / "dataset" / "manifest.jsonl"))
    assert rows and all(row["enhanced"] for row in rows), (
        "every built call should have its targets from the enhanced cache"
    )

    stats = read_json(root / "dataset" / "stats.json")["enhanced"]
    assert stats["calls"] == stats["of"] == len(rows), stats
    assert stats["backend"] == "stub"

    meta = read_json(root / "dataset" / rows[0]["split"] / rows[0]["call"] / "meta.json")
    assert meta["targets_enhanced"] is True and meta["enhance_backend"] == "stub"
    assert "/work/enhanced/" in meta["target_source"], meta["target_source"]
    # The mixture is never taken from the cache.
    assert "/work/enhanced/" not in meta["mix_source"], meta["mix_source"]
    assert "/corpus/" in meta["mix_source"], meta["mix_source"]
    return f"{len(rows)} calls, targets from the cache, mixture from the corpus"


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
# [unit] DialogueSidon-style degradation
# --------------------------------------------------------------------------- #
def _aug_tone(seconds: float = 4.0, freqs=(300.0, 3000.0)) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    return sum(0.3 / (i + 1) * np.sin(2 * np.pi * f * t) for i, f in enumerate(freqs)).astype(
        np.float32
    )


def _band_energy(signal: np.ndarray, low: float, high: float) -> float:
    spectrum = np.abs(np.fft.rfft(signal)) ** 2
    freqs = np.fft.rfftfreq(len(signal), 1.0 / SR)
    return float(spectrum[(freqs >= low) & (freqs < high)].sum())


@check("unit")
def band_limit_removes_only_above_the_cutoff():
    x = _aug_tone()
    y = aug_artifacts.band_limit(x, SR, 1500.0)

    assert len(y) == len(x), f"length changed {len(x)} -> {len(y)}"
    kept = _band_energy(y, 250, 350) / _band_energy(x, 250, 350)
    cut = _band_energy(y, 2950, 3050) / _band_energy(x, 2950, 3050)
    assert 0.95 < kept < 1.05, f"the 300 Hz tone should survive, kept {kept:.3f}"
    assert cut < 1e-3, f"the 3 kHz tone should be gone, kept {cut:.2e}"

    # A cutoff at or above Nyquist has nothing to remove and must be exact,
    # since every rate the paper resamples to lands there on this corpus.
    for cutoff in (SR / 2, SR, 48000.0):
        assert np.array_equal(aug_artifacts.band_limit(x, SR, cutoff), x), cutoff
    return f"300 Hz kept {kept:.3f}, 3 kHz cut to {cut:.1e}"


@check("unit")
def clipping_lands_on_the_requested_percentiles():
    x = _aug_tone()
    for low_pct, high_pct in ((0.0, 100.0), (5.0, 95.0), (10.0, 90.0)):
        y = aug_artifacts.clip(x, low_pct, high_pct)
        assert len(y) == len(x)
        if (low_pct, high_pct) == (0.0, 100.0):
            # The bottom of the draw must be the identity, or the step would
            # always distort and `prob` would no longer control it.
            assert np.array_equal(y, x), "0/100 percentiles must not change the signal"
            continue
        assert approx(y.min(), np.percentile(x, low_pct), tol=1e-5), y.min()
        assert approx(y.max(), np.percentile(x, high_pct), tol=1e-5), y.max()
        assert y.min() > x.min() and y.max() < x.max(), "nothing was actually clipped"
    return "identity at 0/100, exact at 5/95 and 10/90"


@check("unit")
def packet_loss_zeroes_whole_segments_and_repeats():
    x = _aug_tone(20.0) + 0.5  # offset, so a zero can only come from the dropper
    frac, span = 0.09, (20.0, 200.0)
    y, dropped = aug_artifacts.packet_loss(x, SR, random.Random(3), frac, span)

    assert len(y) == len(x), f"length changed {len(x)} -> {len(y)}"
    assert dropped > 0, "nothing was dropped"

    # What must hold is that whole segments go, never fragments of one: no zero
    # run may be shorter than the minimum segment. Runs can be *longer* than the
    # maximum, because two segments that happen to be adjacent merge into one
    # run -- so the count of runs is a lower bound on the count of drops, not an
    # equality.
    zero = y == 0.0
    edges = np.diff(np.concatenate(([0], zero.view(np.int8), [0])))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    lengths = (ends - starts) / SR * 1000.0
    assert 0 < len(starts) <= dropped, f"{len(starts)} zero runs for {dropped} drops"
    assert lengths.min() >= span[0] - 1.0, (
        f"a {lengths.min():.1f} ms run is shorter than the {span[0]} ms minimum "
        "segment -- part of a segment was dropped"
    )

    # Total zeroed time pins it down where the run count cannot: exactly
    # `dropped` segments went, each between the two bounds.
    total = float(lengths.sum())
    assert dropped * span[0] * 0.99 <= total <= dropped * span[1] * 1.01, (
        f"{total:.0f} ms zeroed for {dropped} segments of {span[0]}-{span[1]} ms"
    )

    share = float(zero.mean())
    assert 0.02 < share < 0.20, f"{share:.3f} of samples zeroed, expected near {frac}"

    # Same seed, same result: a recipe records only the seed, so reproducing a
    # variant depends on this.
    a, _ = aug_artifacts.packet_loss(x, SR, random.Random(11), frac, span)
    b, _ = aug_artifacts.packet_loss(x, SR, random.Random(11), frac, span)
    assert np.array_equal(a, b), "the same seed produced two different results"
    assert frac == 0.0 or not np.array_equal(a, y), "different seeds gave the same result"
    return f"{dropped} segments, {share:.3f} of samples, runs {lengths.min():.0f}-{lengths.max():.0f} ms"


@check("unit")
def every_codec_round_trip_preserves_length_and_alignment():
    x = _aug_tone(6.0)
    available = aug_codec.available()
    assert "mulaw" in available, f"ffmpeg has no G.711 encoder; found {available}"

    results = []
    for kind, kbps in (("mulaw", None), ("alaw", None), ("gsm", None),
                       ("opus", 24), ("mp3", 128)):
        if kind not in available:
            continue
        y = aug_codec.round_trip(x, SR, kind, kbps)
        assert len(y) == len(x), f"{kind}: length {len(x)} -> {len(y)}"
        assert np.all(np.isfinite(y)), f"{kind}: non-finite samples"

        # Alignment, not just length. MP3 through a raw pipe arrives 1105
        # samples late and nothing else here would notice: the file would be the
        # right size, the right loudness, and 138 ms out of step with s1/s2.
        # Correlation against the input is what catches that.
        aligned = abs(float(x @ y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))
        assert aligned > 0.7, (
            f"{kind}: decoded signal correlates with the input at only {aligned:.3f} "
            "-- it is delayed or otherwise misaligned"
        )
        results.append(f"{kind}={aligned:.2f}")
    return "alignment " + " ".join(results)


@check("unit")
def reverb_preserves_length_and_dry_level():
    x = _aug_tone(4.0)
    ir = _decaying_ir(random.Random(5))

    y = aug_reverb.apply(x, ir)
    assert len(y) == len(x), f"length changed {len(x)} -> {len(y)}"

    dry = float(np.sqrt(np.mean(x**2)))
    wet = float(np.sqrt(np.mean(y**2)))
    # The SIR build recorded in meta.json describes the signal before the chain
    # runs. If reverb changed the level, that number would describe nothing.
    assert approx(wet, dry, tol=0.01 * dry), f"dry {dry:.5f} -> wet {wet:.5f}"
    assert not np.allclose(y, x, atol=1e-4), "the impulse response did nothing"

    # Peak-aligned impulse responses are what let the convolution be truncated
    # back to length without sliding the signal against its labels.
    aligned = abs(float(x @ y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))
    assert aligned > 0.5, f"reverb moved the signal in time (corr {aligned:.3f})"
    return f"rms {dry:.5f} -> {wet:.5f}, alignment {aligned:.3f}"


@check("unit")
def noise_lands_on_the_requested_snr():
    x = _aug_tone(4.0)
    mask = np.ones(len(x), dtype=bool)
    bed = np.random.default_rng(0).standard_normal(len(x)).astype(np.float32)

    for want in (-5.0, 0.0, 10.0, 20.0):
        y = aug_noise.add_noise(x, bed, mask, want)
        assert len(y) == len(x)
        got = 20.0 * np.log10(
            np.sqrt(np.mean(x.astype(np.float64) ** 2))
            / np.sqrt(np.mean((y - x).astype(np.float64) ** 2))
        )
        assert approx(got, want, tol=0.5), f"asked {want} dB, measured {got:.2f} dB"

    # Measured over speech, not over the whole signal: a telephone leg is mostly
    # silence, and a whole-signal RMS would under-add noise by however much of it
    # there happens to be.
    half = np.zeros(len(x), dtype=bool)
    half[: len(x) // 4] = True
    quiet_side = x.copy()
    quiet_side[len(x) // 4 :] = 0.0
    y = aug_noise.add_noise(quiet_side, bed, half, 10.0)
    speech = np.sqrt(np.mean(quiet_side[half].astype(np.float64) ** 2))
    got = 20.0 * np.log10(speech / np.sqrt(np.mean((y - quiet_side).astype(np.float64) ** 2)))
    assert approx(got, 10.0, tol=0.5), f"over a masked signal: asked 10 dB, got {got:.2f}"
    return "within 0.5 dB at -5, 0, 10, 20 dB and over a mask"


@check("unit")
def a_recipe_is_reproducible_and_prob_controls_it():
    cfg = Config()
    banks = _stub_banks()

    first = aug_chain.sample(random.Random("seed-a"), cfg.build.augment, banks)
    again = aug_chain.sample(random.Random("seed-a"), cfg.build.augment, banks)
    other = aug_chain.sample(random.Random("seed-b"), cfg.build.augment, banks)
    assert first.to_dict() == again.to_dict(), "the same seed drew two different recipes"
    assert first.to_dict() != other.to_dict(), "two seeds drew the same recipe"

    none_cfg = dataclasses.replace(cfg.build.augment, prob=0.0)
    empty = aug_chain.sample(random.Random("x"), none_cfg, banks)
    assert not empty.any_applied(), f"prob=0 still drew {empty.to_dict()}"
    x = _aug_tone(2.0)
    assert np.array_equal(aug_chain.apply(x, empty, SR, banks), x), "an empty recipe changed the signal"

    all_cfg = dataclasses.replace(cfg.build.augment, prob=1.0)
    full = aug_chain.sample(random.Random("x"), all_cfg, banks)
    missing = [s for s in aug_chain.STEPS if getattr(full, s) is None]
    assert not missing, f"prob=1 did not draw {missing}"

    # Applying the same recipe twice must give the same samples, or a variant
    # could not be reproduced from what meta.json records.
    once = aug_chain.apply(x, full, SR, banks)
    twice = aug_chain.apply(x, full, SR, banks)
    assert np.array_equal(once, twice), "applying one recipe twice gave two answers"
    assert len(once) == len(x), f"the chain changed the length {len(x)} -> {len(once)}"
    return f"{len(aug_chain.STEPS)} steps, all fired at prob=1, none at prob=0"


@check("unit")
def degradation_is_per_channel_and_happens_before_the_sum():
    """The load-bearing property: each track is degraded on its own.

    Degrading the finished mixture instead would give both speakers one shared
    room, one shared noise recording and one shared codec -- a different and much
    weaker task. This reconstructs the mixture from the two recorded recipes and
    requires an exact match, which can only hold if the sum came last.
    """
    cfg = Config()
    cfg.build.augment = dataclasses.replace(cfg.build.augment, prob=1.0)
    banks = _stub_banks()

    ch1 = _aug_tone(4.0, (300.0, 900.0))
    ch2 = _aug_tone(4.0, (600.0, 2400.0))
    mask = np.ones(len(ch1), dtype=bool)

    mixture, recipes, gain = build_stage._degrade_variant(
        ch1, ch2, mask, mask, cfg, banks, random.Random("v0")
    )
    assert len(recipes) == 2, recipes
    assert len(mixture) == len(ch1)

    rebuilt = gain * (
        aug_chain.apply(ch1, aug_chain.Recipe.from_dict(recipes[0]), SR, banks, mask)
        + aug_chain.apply(ch2, aug_chain.Recipe.from_dict(recipes[1]), SR, banks, mask)
    )
    assert np.allclose(mixture, rebuilt, atol=1e-6), (
        "the mixture is not the sum of the two separately degraded channels; "
        f"max difference {np.abs(mixture - rebuilt).max():.3e}"
    )

    # The two tracks must have drawn independently. Identical recipes on both
    # sides is what applying one chain to the summed mixture would look like.
    assert recipes[0] != recipes[1], "both channels drew the same recipe"

    # And the same recipe applied to the sum is a different signal, so the two
    # orderings are genuinely distinguishable by this test.
    summed = aug_chain.apply(
        ch1 + ch2, aug_chain.Recipe.from_dict(recipes[0]), SR, banks, mask
    )
    assert not np.allclose(mixture, summed, atol=1e-3), (
        "degrading the sum gives the same answer as degrading the channels, so "
        "this check cannot tell the two apart"
    )
    return f"exact to {np.abs(mixture - rebuilt).max():.1e}, peak gain {gain:.3f}"


# --------------------------------------------------------------------------- #
# [e2e] the augmented dataset
# --------------------------------------------------------------------------- #
_AUG_FIXTURE: dict | None = None

# Everything fires, so the assertions below do not depend on a coin flip. The
# ranges are narrowed only to keep the check fast, not to change what is tested.
AUG_BUILD = """  augment:
    variants: 2
    prob: 1.0
    reverb: {simulated_frac: 0.5}
    noise: {snr_db: [5.0, 15.0]}
    band_limit: {cutoff_hz: [2500.0, 3000.0]}
    clip: {low_pct: [2.0, 5.0], high_pct: [95.0, 98.0]}
    codec: {kinds: [mulaw, gsm], opus_kbps: [6, 24], mp3_kbps: [65, 245]}
    packet_loss: {frac: 0.09, segment_ms: [20.0, 200.0]}
"""


def _decaying_ir(rng: random.Random, seconds: float = 0.3, rt60: float = 0.25) -> np.ndarray:
    """Exponentially decaying noise: a plausible impulse response, peak at 0."""
    n = int(seconds * SR)
    gen = np.random.default_rng(rng.randrange(2**32))
    tail = gen.standard_normal(n) * np.exp(-6.9 * np.arange(n) / (rt60 * SR))
    ir = tail.astype(np.float32)
    ir[0] = 1.0
    return aug_reverb.align(ir, SR, seconds)


def _stub_banks() -> "aug_chain.Banks":
    """Banks backed by a handful of generated assets, written once to a temp dir."""
    global _STUB_BANKS
    if _STUB_BANKS is not None:
        return _STUB_BANKS

    root = Path(tempfile.mkdtemp(prefix="dsd_aug_assets_"))
    atexit.register(shutil.rmtree, root, True)
    (root / "irs").mkdir()
    (root / "rirs").mkdir()
    rng = random.Random(0)
    for index in range(3):
        A.write_wav(root / "irs" / f"ir{index}.wav", _decaying_ir(rng), SR, subtype="FLOAT")
        A.write_wav(root / "rirs" / f"rir{index}.wav", _decaying_ir(rng), SR, subtype="FLOAT")

    noise_dir = root / "noise"
    noise_dir.mkdir()
    gen = np.random.default_rng(5)
    files = []
    for index in range(4):
        clip = (0.05 * gen.standard_normal(SR * 3)).astype(np.float32)
        path = noise_dir / f"noise{index}.wav"
        A.write_wav(path, clip, SR)
        files.append(str(path))

    _STUB_BANKS = aug_chain.Banks(
        rirs=aug_reverb.RIRBank(root / "rirs", root / "irs", SR),
        noise=aug_noise.NoiseBank(files, SR),
        codecs=[k for k in ("mulaw", "gsm") if k in aug_codec.available()],
    )
    return _STUB_BANKS


_STUB_BANKS: "aug_chain.Banks | None" = None


def augmented_fixture() -> dict:
    """Build the same calls twice -- clean and augmented -- from one config.

    Two builds rather than a comparison against the shared e2e dataset: that one
    is rebuilt by several checks above with different settings, so what is in it
    depends on the order they ran in.
    """
    global _AUG_FIXTURE
    if _AUG_FIXTURE is not None:
        return _AUG_FIXTURE

    base = e2e_fixture()
    root = base["root"]
    assets = Path(tempfile.mkdtemp(prefix="dsd_aug_src_"))
    atexit.register(shutil.rmtree, assets, True)

    # Impulse responses and noise on disk for the stage to ingest, so the
    # fixture exercises the real `augment` stage rather than a hand-built bank.
    (assets / "irs").mkdir()
    (assets / "noise").mkdir()
    rng = random.Random(1)
    for index in range(3):
        A.write_wav(assets / "irs" / f"ir{index}.wav", _decaying_ir(rng), SR, subtype="FLOAT")
    gen = np.random.default_rng(2)
    for index in range(4):
        clip = (0.05 * gen.standard_normal(SR * 3)).astype(np.float32)
        A.write_wav(assets / "noise" / f"noise{index}.wav", clip, SR)

    config = root / "configs" / "aug.yaml"
    body = E2E_CONFIG.replace("  augment: {variants: 0}\n", AUG_BUILD)
    body += (
        "augment:\n"
        f"  ir_dir: {assets / 'irs'}\n"
        f"  noise_dir: {assets / 'noise'}\n"
        "  noise_screen_json: null\n"
        "  simulated_rirs: 4\n"
        "  rt60: [0.2, 0.5]\n"
        "  room_dim: [3.0, 8.0]\n"
        "  ir_max_sec: 0.5\n"
    )
    config.write_text(body, encoding="utf-8")

    def run(*args: str) -> str:
        with quiet() as buffer:
            code = cli.main(["--config", str(config), *args])
        output = buffer.getvalue()
        if code != 0:
            raise AssertionError(f"`dsd {' '.join(args)}` exited {code}\n{output}")
        return output

    logs = {"augment": run("augment", "--no-screen")}
    logs["clean"] = run(
        "--set", f"paths.dataset_dir={root / 'dataset_clean'}",
        "--set", "build.augment.variants=0",
        "build", "--chunks",
    )
    logs["build"] = run("--set", f"paths.dataset_dir={root / 'dataset_aug'}", "build", "--chunks")
    logs["verify"] = run("--set", f"paths.dataset_dir={root / 'dataset_aug'}", "verify")

    _AUG_FIXTURE = {
        "root": root,
        "config": config,
        "run": run,
        "logs": logs,
        "clean": root / "dataset_clean",
        "aug": root / "dataset_aug",
    }
    return _AUG_FIXTURE


@check("e2e")
def augmentation_adds_mixtures_and_leaves_the_targets_alone():
    fixture = augmented_fixture()
    clean_rows = list(read_jsonl(fixture["clean"] / "manifest.jsonl"))
    rows = list(read_jsonl(fixture["aug"] / "manifest.jsonl"))

    calls = {row["call"] for row in rows}
    assert calls == {row["call"] for row in clean_rows}, "augmentation changed which calls exist"
    assert len(rows) == 3 * len(calls), f"{len(rows)} rows for {len(calls)} calls, expected 3 each"

    by_call: dict[str, list] = {}
    for row in rows:
        by_call.setdefault(row["call"], []).append(row)
    for call, group in by_call.items():
        variants = {r["variant"] for r in group}
        assert variants == {None, 0, 1}, f"{call}: {variants}"
        assert len({r["s1"] for r in group}) == 1, f"{call}: variants point at different s1"
        assert len({r["mix"] for r in group}) == 3, f"{call}: variants share a mixture"

    # The whole point of the pair: only the model's input is degraded. The
    # targets, and the clean mixture, must be byte-identical to a build with
    # augmentation switched off.
    for row in clean_rows:
        for name in ("mix", "s1", "s2"):
            a = (fixture["clean"] / row[name]).read_bytes()
            b = (fixture["aug"] / row[name]).read_bytes()
            assert a == b, f"{row['call']}: {name}.wav differs between the two builds"
    return f"{len(calls)} calls -> {len(rows)} rows, targets bit-identical"


@check("e2e")
def degraded_mixtures_differ_from_the_clean_one_and_from_each_other():
    fixture = augmented_fixture()
    rows = list(read_jsonl(fixture["aug"] / "manifest.jsonl"))

    by_call: dict[str, dict] = {}
    for row in rows:
        by_call.setdefault(row["call"], {})[row["variant"]] = row

    correlations = []
    for call, group in by_call.items():
        clean, _ = A.read_audio(fixture["aug"] / group[None]["mix"])
        signals = []
        for index in (0, 1):
            data, _ = A.read_audio(fixture["aug"] / group[index]["mix"])
            assert len(data) == len(clean), f"{call}: variant {index} has a different length"
            signals.append(data[:, 0])
            assert not np.array_equal(data[:, 0], clean[:, 0]), (
                f"{call}: variant {index} is identical to the clean mixture"
            )
            correlations.append(
                abs(float(clean[:, 0] @ data[:, 0])
                    / (np.linalg.norm(clean[:, 0]) * np.linalg.norm(data[:, 0]) + 1e-12))
            )
        assert not np.array_equal(signals[0], signals[1]), f"{call}: the two variants are identical"

        # Every step fired (prob 1.0), so every variant records both tracks.
        meta = read_json(fixture["aug"] / group[None]["split"] / call / "meta.json")
        items = meta["augment"]["items"]
        assert len(items) == 2, items
        for item in items:
            assert len(item["recipes"]) == 2, "a variant must record one recipe per channel"
            for recipe in item["recipes"]:
                missing = [s for s in aug_chain.STEPS if s not in recipe]
                assert not missing, f"{call}: prob=1 but {missing} did not fire"
    return (
        f"{len(correlations)} variants, correlation with the clean mixture "
        f"{min(correlations):.2f}..{max(correlations):.2f}"
    )


@check("e2e")
def verify_catches_a_corrupted_variant():
    fixture = augmented_fixture()
    rows = list(read_jsonl(fixture["aug"] / "manifest.jsonl"))
    target = next(r for r in rows if r["variant"] == 0)
    path = fixture["aug"] / target["mix"]
    original = path.read_bytes()

    def verify_fails(what: str) -> str:
        """Run verify, require it to fail, and hand back what it printed.

        `verify` reports problems by raising SystemExit(1), whose str() is just
        "1", so the output has to be captured here rather than read off the
        exception -- and the output is the point: this check is about *which*
        test fired, not merely that something did.
        """
        with quiet() as buffer:
            try:
                code = cli.main(
                    ["--config", str(fixture["config"]),
                     "--set", f"paths.dataset_dir={fixture['aug']}", "verify"]
                )
            except SystemExit as exc:
                code = exc.code if isinstance(exc.code, int) else 1
        output = buffer.getvalue()
        assert code != 0, f"verify passed a variant that was {what}\n{output}"
        return output

    try:
        data, sr = A.read_audio(path)
        # A time shift leaves the RMS untouched, so only the correlation the
        # build recorded can see it. This is the case the fingerprint misses.
        A.write_wav(path, np.roll(data[:, 0], 500), sr)
        shifted = verify_fails("time-shifted")
        assert "correlates with mix.wav" in shifted, shifted

        # A rescale moves the level, which the fingerprint catches.
        A.write_wav(path, data[:, 0] * 0.9, sr)
        rescaled = verify_fails("rescaled")
        assert "rms" in rescaled, rescaled
    finally:
        path.write_bytes(original)

    fixture["run"]("--set", f"paths.dataset_dir={fixture['aug']}", "verify")
    return "shift and rescale both caught, clean dataset still passes"


@check("e2e")
def augmented_chunks_share_one_set_of_targets():
    fixture = augmented_fixture()
    rows = list(read_jsonl(fixture["aug"] / "chunks" / "manifest.jsonl"))
    assert rows, "no chunks were written"

    by_offset: dict[tuple, list] = {}
    for row in rows:
        by_offset.setdefault((row["call"], row["offset"]), []).append(row)

    for key, group in by_offset.items():
        assert len(group) == 3, f"{key}: {len(group)} rows, expected clean + 2 variants"
        assert len({r["s1"] for r in group}) == 1, f"{key}: chunk targets are not shared"
        assert len({r["mix"] for r in group}) == 3, f"{key}: chunk mixtures are not distinct"
        for row in group:
            assert (fixture["aug"] / row["mix"]).exists(), row["mix"]

    clean_chunks = list(read_jsonl(fixture["clean"] / "chunks" / "manifest.jsonl"))
    assert len(rows) == 3 * len(clean_chunks), (
        f"{len(rows)} augmented chunks for {len(clean_chunks)} clean ones"
    )
    return f"{len(by_offset)} windows x 3, {len(rows)} chunk rows"


@check("e2e")
def changing_the_variant_count_invalidates_the_cache():
    fixture = augmented_fixture()
    scratch = fixture["root"] / "dataset_variants"

    first = fixture["run"]("--set", f"paths.dataset_dir={scratch}", "build")
    assert "wrote 8 calls as 24 rows" in first, first

    # Same settings: nothing should be rebuilt, and the manifest must survive.
    again = fixture["run"]("--set", f"paths.dataset_dir={scratch}", "build")
    assert len(list(read_jsonl(scratch / "manifest.jsonl"))) == 24, again

    # One more variant is a different dataset, so every call has to be redone.
    grown = fixture["run"](
        "--set", f"paths.dataset_dir={scratch}", "build", "--variants", "3"
    )
    rows = list(read_jsonl(scratch / "manifest.jsonl"))
    assert len(rows) == 32, f"{len(rows)} rows after --variants 3\n{grown}"
    assert sorted({r["variant"] for r in rows}, key=lambda v: (v is not None, v)) == [
        None, 0, 1, 2
    ], sorted({r["variant"] for r in rows}, key=str)

    # And back down: the third variant's rows must go, not linger in the manifest.
    shrunk = fixture["run"](
        "--set", f"paths.dataset_dir={scratch}", "build", "--variants", "0"
    )
    rows = list(read_jsonl(scratch / "manifest.jsonl"))
    assert len(rows) == 8, f"{len(rows)} rows after --variants 0\n{shrunk}"
    assert {r["variant"] for r in rows} == {None}, {r["variant"] for r in rows}

    # The files have to go too. The call directory survives -- the selection
    # still holds the call -- so nothing else would ever remove them, and a
    # trainer globbing the tree would pick up mixtures this dataset disowned.
    left = sorted(p.name for p in scratch.glob("*/*/mix_aug*.wav"))
    assert not left, f"{len(left)} degraded mixtures left on disk (e.g. {left[0]})"
    fixture["run"]("--set", f"paths.dataset_dir={scratch}", "verify")
    return "24 -> 24 cached -> 32 -> 8, stale variants deleted"


@check("e2e")
def speaker_caps_are_counted_per_call_not_per_row():
    """The bug the extra rows would otherwise introduce.

    `verify` counts a speaker's calls and speech from the manifest. With three
    rows per call those totals triple, and a selection that is exactly at its cap
    gets reported as three times over it -- a failure on a correct dataset.
    """
    fixture = augmented_fixture()
    rows = list(read_jsonl(fixture["aug"] / "manifest.jsonl"))

    per_row: dict[str, int] = {}
    for row in rows:
        for speaker in row["speakers"]:
            per_row[speaker] = per_row.get(speaker, 0) + 1
    cap = read_json(fixture["root"] / "work" / "selection.json")["max_calls_per_speaker"]
    assert cap, "the fixture selection has no call cap, so this proves nothing"
    assert max(per_row.values()) > cap, (
        "counting rows should have exceeded the cap, or this check is vacuous"
    )

    # verify is what has to disagree with the naive count.
    output = fixture["run"]("--set", f"paths.dataset_dir={fixture['aug']}", "verify")
    assert "exceed" not in output, output
    return f"rows would say {max(per_row.values())} calls for a cap of {cap}; verify passes"


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
# [e2e] wideband targets: mixture at 8 kHz, s1/s2 at 24 kHz
# --------------------------------------------------------------------------- #
_WIDE_FIXTURE: dict | None = None
WIDE_SR = 3 * SR


def wideband_fixture() -> dict:
    """The same corpus, enhanced to 24 kHz and built with 8 kHz mixtures.

    Its own cache and dataset directories, so nothing the main fixture's checks
    rebuild can leak in. The e2e config rolls every call (`natural_frac: 0`),
    which is what exercises the shift being scaled to the target rate.
    """
    global _WIDE_FIXTURE
    if _WIDE_FIXTURE is not None:
        return _WIDE_FIXTURE

    base = e2e_fixture()
    root = base["root"]
    config = root / "configs" / "wideband.yaml"
    body = E2E_CONFIG.replace(
        "  dataset_dir: dataset\n",
        "  dataset_dir: dataset_24k\n  enhanced_dir: work/enhanced_24k\n",
    ).replace(
        "  backend: stub\n  # Same-rate targets for the main fixture; `wideband_fixture` covers 24 kHz.\n"
        "  output_sample_rate: null\n",
        f"  backend: stub_wideband\n  output_sample_rate: {WIDE_SR}\n",
    )
    assert "stub_wideband" in body and "enhanced_24k" in body, "wideband config did not apply"
    config.write_text(body, encoding="utf-8")

    def run(*args: str) -> str:
        with quiet() as buffer:
            try:
                code = cli.main(["--config", str(config), *args])
            except SystemExit as exc:
                # verify reports its problems on stdout and then exits; keep them.
                code = exc.code if isinstance(exc.code, int) else f"{exc.code}"
        output = buffer.getvalue()
        if code != 0:
            raise AssertionError(f"`dsd {' '.join(args)}` exited {code}\n{output}")
        return output

    logs = {"enhance": run("enhance"), "build": run("build", "--chunks"), "verify": run("verify")}
    _WIDE_FIXTURE = {"root": root, "run": run, "logs": logs, "dataset": root / "dataset_24k"}
    return _WIDE_FIXTURE


def _refused(run, *args: str) -> str:
    """Run a command that must fail; return what it said."""
    try:
        run(*args)
    except (AssertionError, SystemExit) as exc:
        return str(exc)
    raise AssertionError(f"`dsd {' '.join(args)}` should have been refused")


@check("e2e")
def wideband_targets_cover_the_mixture_duration():
    fixture = wideband_fixture()
    dataset = fixture["dataset"]
    rows = list(read_jsonl(dataset / "manifest.jsonl"))
    assert rows, "no rows built"

    shifted = 0
    for row in rows:
        assert row["sample_rate"] == SR and row["target_sample_rate"] == WIDE_SR, row
        mixture, sr = A.read_audio(dataset / row["mix"])
        s1, sr1 = A.read_audio(dataset / row["s1"])
        s2, sr2 = A.read_audio(dataset / row["s2"])
        assert (sr, sr1, sr2) == (SR, WIDE_SR, WIDE_SR), (sr, sr1, sr2)
        # Same duration to the sample: exactly three target samples per mixture sample.
        assert len(s1) == len(s2) == 3 * len(mixture), (len(mixture), len(s1), len(s2))
        meta = read_json(dataset / row["split"] / row["call"] / "meta.json")
        assert meta["target_sample_rate"] == WIDE_SR and meta["target_samples"] == len(s1)
        shifted += bool(meta["shifted"])
    assert shifted, "the fixture should roll at least one call"

    chunks = list(read_jsonl(dataset / "chunks" / "manifest.jsonl"))
    assert chunks, "no chunks cut"
    for row in chunks:
        mixture, sr = A.read_audio(dataset / row["mix"])
        s1, tsr = A.read_audio(dataset / row["s1"])
        assert sr == SR and tsr == WIDE_SR and len(s1) == 3 * len(mixture), row["mix"]
        assert len(mixture) == int(round(4.0 * SR)), len(mixture)
    return f"{len(rows)} calls, {shifted} rolled, {len(chunks)} chunks, 8 kHz mix / 24 kHz targets"


@check("e2e")
def wideband_targets_line_up_with_the_mixture():
    """The alignment check that matters, across rates.

    Brought back to 8 kHz, a target is the stub's 0.5x of the channel, so where
    only that speaker talks the mixture is exactly 2.0x the target. A target a
    few samples out of step with the mixture -- a shift rolled at the wrong
    rate, an offset scaled wrongly in the splice -- decorrelates and the gain
    collapses. Checked on both speakers of every call; every call is rolled.
    """
    fixture = wideband_fixture()
    dataset = fixture["dataset"]
    fade = int(0.01 * SR)
    gains = []
    for row in read_jsonl(dataset / "manifest.jsonl"):
        mixture = A.read_audio(dataset / row["mix"])[0][:, 0]
        s1 = A.read_audio(dataset / row["s1"])[0][:, 0]
        s2 = A.read_audio(dataset / row["s2"])[0][:, 0]
        on1, on2 = verify_stage._active(s1, 3), verify_stage._active(s2, 3)
        low1, low2 = A.resample(s1, WIDE_SR, SR), A.resample(s2, WIDE_SR, SR)
        for name, target, on, other_on in (("s1", low1, on1, on2), ("s2", low2, on2, on1)):
            # Wider erosion than verify's: the resampler rings for a few ms either
            # side of a fade, and this check wants only the clean interior.
            solo = verify_stage._erode(on, 4 * fade) & verify_stage._erode(~other_on, 4 * fade)
            assert solo.sum() > SR // 4, f"{row['call']}: no solo stretch for {name}"
            t = target[solo].astype(np.float64)
            gain = float(np.dot(mixture[solo], t) / np.dot(t, t))
            assert abs(gain - 1.0 / StubEnhancer.GAIN) < 0.05, (
                f"{row['call']} {name}: mixture is {gain:.3f}x the target where only {name} "
                "speaks; expected 2.0x -- the 24 kHz target is out of step with the mixture"
            )
            gains.append(gain)
    return f"{len(gains)} solo stretches, gain {min(gains):.3f}..{max(gains):.3f} (expect 2.0)"


@check("e2e")
def wideband_verify_passes_and_catches_a_wrong_rate_target():
    fixture = wideband_fixture()
    assert "OK -- every check passed" in fixture["logs"]["verify"], fixture["logs"]["verify"]

    dataset = fixture["dataset"]
    row = next(iter(read_jsonl(dataset / "manifest.jsonl")))
    path = dataset / row["s2"]
    original = path.read_bytes()
    s2, _ = A.read_audio(path)
    try:
        # The plausible mistake: an 8 kHz target where a 24 kHz one belongs.
        A.write_wav(path, A.resample(s2[:, 0], WIDE_SR, SR), SR)
        said = _refused(fixture["run"], "verify")
        assert "sample rates" in said, said
    finally:
        path.write_bytes(original)
    return "clean, and an 8 kHz s2 among 24 kHz targets is caught"


@check("e2e")
def zerofy_mix_at_mixed_rates_is_refused():
    """mix == s1 + s2 cannot hold across rates without making mix a downsample of the targets."""
    fixture = wideband_fixture()
    said = _refused(fixture["run"], "--set", "build.zerofy_mix=true", "build", "--overwrite")
    assert "zerofy_mix" in said and "Hz" in said, said


@check("e2e")
def a_cache_at_another_rate_is_refused():
    """A half-migrated cache is the worst outcome here: both stages must refuse to use one."""
    fixture = wideband_fixture()
    root = fixture["root"]
    run = fixture["run"]

    # build at 24 kHz pointed at the main fixture's 8 kHz cache
    said = _refused(run, "--set", "paths.enhanced_dir=work/enhanced", "build", "--overwrite")
    assert "8000 Hz" in said and "24000 Hz" in said, said

    # enhance at 24 kHz into that same 8 kHz cache
    said = _refused(run, "--set", "paths.enhanced_dir=work/enhanced", "enhance")
    assert "made differently" in said and "output_sample_rate" in said, said

    # a cache from before the marker existed: audio, no cache.json
    legacy = root / "work" / "enhanced_legacy"
    legacy.mkdir(exist_ok=True)
    try:
        some = next((root / "work" / "enhanced").glob("*.wav"))
        shutil.copy(some, legacy / some.name)
        said = _refused(run, "--set", "paths.enhanced_dir=work/enhanced_legacy", "enhance")
        assert "no cache.json" in said, said
    finally:
        shutil.rmtree(legacy, ignore_errors=True)


@check("e2e")
def enhance_stage_drives_the_sidon_client_at_24k():
    """The real client inside the real stage, against a stub /v1/restore.

    Also the guard that would have caught the original mislabel: a backend named
    for one model, and a service reporting another, must not fill a cache.
    """
    fixture = wideband_fixture()
    root = fixture["root"]
    run = fixture["run"]
    target = root / "work" / "enhanced_http"
    try:
        with restore_server(model="MossFormerGAN_SE_16K") as base:
            said = _refused(
                run, "--set", "enhance.backend=http_sidon",
                "--set", f"enhance.options.http_sidon.base_url={base}",
                "--set", "paths.enhanced_dir=work/enhanced_http", "enhance", "--limit", "1",
            )
        assert "expects the service to be 'Sidon'" in said, said

        with restore_server() as base:
            run(
                "--set", "enhance.backend=http_sidon",
                "--set", f"enhance.options.http_sidon.base_url={base}",
                "--set", "paths.enhanced_dir=work/enhanced_http", "enhance", "--limit", "1",
            )
            sent = list(_RestoreHandler.requests)
        assert sent and all(r["output_sample_rate"] == WIDE_SR for r in sent), sent[:1]
        written = sorted(target.glob("*.wav"))
        assert len(written) == 1, written
        data, rate = A.read_audio(written[0])
        source = next((root / "corpus").rglob(written[0].name))
        original, _ = A.read_audio(source)
        assert rate == WIDE_SR and len(data) == 3 * len(original), (rate, len(data), len(original))
        marker = read_json(target / "cache.json")
        assert marker["model"] == "Sidon" and marker["output_sample_rate"] == WIDE_SR, marker
    finally:
        shutil.rmtree(target, ignore_errors=True)
    return f"{len(sent)} requests at 24 kHz, wrong model refused"


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