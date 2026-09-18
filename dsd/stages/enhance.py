"""Stage 8 -- restore the two channels of each selected call, into `work/enhanced/`.

Why this is its own stage rather than part of `build`:

  - **Enhancement is non-linear**, so `enhance(s1 + s2) != enhance(s1) + enhance(s2)`.
    It has to be applied to the two isolated channels *before* they are summed,
    or `mix == s1 + s2` -- the invariant `verify` checks and every separation
    loss assumes -- stops holding. Enhancing each channel and then summing keeps
    it exact by construction.
  - **It is GPU time, and `build` is re-run constantly.** Sidon measured ~25x
    realtime over 62 spans (the old masking service ~3.2x, flat with
    concurrency). `build` is the stage that gets re-run most while tuning
    overlap and SIR, and re-enhancing on every `--overwrite` is not viable.
  - **Duration is preserved, not length.** Sidon synthesises at 48 kHz and the
    cache is written at `enhance.output_sample_rate` (24 kHz), while the source
    and the mixture stay at 8 kHz. The ratio must be an integer, so every time
    offset maps exactly: a cached channel is exactly `k` times the source's
    sample count, and `build` converts each VAD interval in seconds at each
    track's own rate. The cache is ~3x the size it was at 8 kHz.

Only the VAD speech spans are enhanced. Everything outside them is zerofied by
`build` regardless, so enhancing it would be spending the scarcest resource in
the pipeline on samples that are about to be multiplied by zero. Those samples
are still carried to the output rate (a plain resample), so the cache stays one
file per call at one rate.

`cache.json` beside the audio records what the cache was made with -- backend,
the model `/healthz` named, both rates, the span settings. A cache is days of
GPU time, and the one outcome worse than rebuilding it is a half-migrated one:
8 kHz files and 24 kHz files side by side, each plausible on its own. So a run
whose settings differ from the marker, or a directory holding audio but no
marker, stops rather than extending it.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from ..backends import speech_enhancement  # noqa: F401  (registers backends)
from ..backends.speech_enhancement.base import output_rate
from ..core.audio import demux, read_audio, resample, write_wav
from ..core.manifest import read_json, write_failures, write_json
from ..core.rttm import merge_intervals, read_rttm
from ..registry import ENHANCERS
from .base import banner, progress, require

NAME = "enhance"
REQUIRES = ("select", "vad")


def add_args(parser) -> None:
    parser.add_argument("--backend", help=f"one of: {', '.join(ENHANCERS.names())}")
    parser.add_argument("--base-url", help="override the service base URL")
    parser.add_argument("--regions", choices=["speech", "full"], help="what to enhance")
    parser.add_argument("--workers", type=int, help="concurrent requests")
    parser.add_argument("--overwrite", action="store_true", help="redo calls already cached")
    parser.add_argument("--limit", type=int, help="enhance only the first N calls")


MARKER = "cache.json"


def enhanced_path(cfg, call: str) -> Path:
    """Where a call's enhanced audio lives. Shared with `build`."""
    return cfg.paths.enhanced_dir / f"{call}.{cfg.enhance.format.lower()}"


def marker_path(cfg) -> Path:
    """The cache's record of how it was made. Shared with `build`."""
    return cfg.paths.enhanced_dir / MARKER


def target_rate(cfg) -> int:
    """The rate enhanced targets are cached at: `enhance.output_sample_rate` or the source's."""
    return int(cfg.enhance.output_sample_rate or cfg.sample_rate)


def run(cfg, args) -> None:
    require(
        {"selection.json": cfg.paths.selection_json, "VAD directory": cfg.paths.vad_dir},
        NAME,
    )

    backend = args.backend or cfg.enhance.backend
    options = dict(cfg.enhance.options.get(backend, {}))
    if args.base_url:
        options["base_url"] = args.base_url
    # One knob for the whole pipeline, so `build` and `verify` know the target
    # rate without asking the backend. A per-backend option still wins.
    options.setdefault("output_sample_rate", cfg.enhance.output_sample_rate)
    enhancer = ENHANCERS.create(backend, options)

    out_sr = _check_rates(enhancer, cfg)
    regions = args.regions or cfg.enhance.regions
    workers = args.workers or cfg.enhance.workers

    selection = read_json(cfg.paths.selection_json)
    calls = sorted(selection["calls"].items())

    out_dir = cfg.paths.enhanced_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.overwrite:
        calls = [item for item in calls if not enhanced_path(cfg, item[0]).exists()]
    if args.limit:
        calls = calls[: args.limit]

    banner(
        NAME,
        f"backend={backend}  regions={regions}  {cfg.sample_rate} Hz -> {out_sr} Hz  "
        f"{len(calls)} calls to enhance  (cache {out_dir})",
    )
    # Before "nothing to do": a cache already full at 8 kHz must not be reported
    # as done to a run that asked for 24 kHz.
    _claim_cache(cfg, backend, None, out_sr, regions, args.overwrite, write=False)
    if not calls:
        banner(NAME, "nothing to do")
        return

    # Fail now, with a useful message, rather than after the first request.
    model = None
    if hasattr(enhancer, "health"):
        try:
            health = enhancer.health()
        except Exception as exc:
            raise SystemExit(
                f"[{NAME}] the enhancement service is not reachable: {exc!r}\n"
                f"[{NAME}] start it with: cd /home/akbar/craft/prod && "
                "SIDON_DEVICES=cuda:0 sidon_serving/venv/bin/python -m uvicorn "
                "sidon_serving.server.app:app --port 8000"
            )
        model = health.get("model")
        detail = "  ".join(
            f"{key}={health[key]}"
            for key in ("status", "model_sample_rate", "engines", "replicas", "max_batch_size")
            if key in health
        )
        banner(NAME, f"service ok: {model}  {detail}")
        expected = getattr(enhancer, "expected_model", None)
        if expected and model != expected:
            # The check that would have caught the original mislabel: a backend
            # named for one model, silently served by another.
            raise SystemExit(
                f"[{NAME}] backend {backend!r} expects the service to be {expected!r}, but "
                f"/healthz says {model!r}. Refusing to fill the cache under the wrong name."
            )

    _claim_cache(cfg, backend, model, out_sr, regions, args.overwrite)
    # Backends state their own measured speed; 3.2x was the masking service's
    # (3.17x / 3.28x / 3.27x at 1 / 4 / 8 workers) and is the cautious default.
    _estimate(cfg, selection, calls, regions, float(getattr(enhancer, "realtime_factor", 3.2)))

    failed: list[tuple[str, str]] = []
    started = time.time()

    def process(item: tuple[str, dict]) -> None:
        call, record = item
        data, sr = read_audio(record["audio"])
        if sr != cfg.sample_rate:
            raise ValueError(f"source is {sr} Hz, but sample_rate is {cfg.sample_rate}")
        spans_by_channel = _spans(cfg, call, record, data.shape[0], sr, out_sr, regions)

        channels = []
        for channel in range(data.shape[1]):
            samples = demux(data, channel)
            # A channel the selection never named is carried through un-enhanced
            # rather than raising: the cache must stay the same shape as the source.
            ranges = spans_by_channel.get(channel, [])
            channels.append(_enhance_channel(enhancer, samples, sr, out_sr, ranges))

        write_wav(
            enhanced_path(cfg, call),
            np.stack(channels, axis=1),
            out_sr,
            subtype="PCM_16",
            fmt=cfg.enhance.format.upper(),
        )

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(process, item): item for item in calls}
            for future in progress(as_completed(futures), "enhance", total=len(futures)):
                call, _ = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    failed.append((call, repr(exc)))
    else:
        for item in progress(calls, "enhance"):
            try:
                process(item)
            except Exception as exc:
                failed.append((item[0], repr(exc)))

    write_failures(cfg.paths.work_dir / "enhance_failed.tsv", failed)

    elapsed = time.time() - started
    audio = getattr(enhancer, "audio_seconds", 0.0)
    if audio:
        banner(
            NAME,
            f"enhanced {audio / 3600:.2f} h of audio in {elapsed / 3600:.2f} h "
            f"({audio / max(elapsed, 1e-9):.2f}x realtime, "
            f"{getattr(enhancer, 'requests', 0):,} requests)",
        )
    adjusted = getattr(enhancer, "length_adjusted", 0)
    if adjusted:
        banner(NAME, f"{adjusted:,} request(s) had their length pinned by the service (a frame of drift)")
    banner(NAME, f"{len(failed)} failed; see work/enhance_failed.tsv" if failed else "done, no failures")


# --------------------------------------------------------------------------- #
def _check_rates(enhancer, cfg) -> int:
    """The output rate, refused unless it is a whole multiple of the source rate.

    With an integer ratio `k` a source sample offset `a` is exactly output
    offset `a * k`, so a span can be sent at one rate and spliced at the other
    with no fractional-sample slip. 22.05 kHz from 8 kHz would put every seam
    and every label a fraction of a sample off, differently per span.
    """
    out_sr = output_rate(enhancer, cfg.sample_rate)
    expected = target_rate(cfg)
    if out_sr != expected:
        raise SystemExit(
            f"[{NAME}] the backend returns {out_sr} Hz, but enhance.output_sample_rate "
            f"asks for {expected} Hz; build would reject the cache"
        )
    if out_sr % cfg.sample_rate:
        raise SystemExit(
            f"[{NAME}] enhance.output_sample_rate={out_sr} is not a whole multiple of "
            f"sample_rate={cfg.sample_rate}; time offsets would not map exactly between "
            "the mixture and its targets"
        )
    return out_sr


def cache_settings(cfg, backend: str, model, out_sr: int, regions: str) -> dict:
    """What a cache is made with. Two caches that differ in any of it are different data."""
    return {
        "backend": backend,
        "model": model,
        "input_sample_rate": int(cfg.sample_rate),
        "output_sample_rate": int(out_sr),
        "regions": regions,
        "merge_gap": cfg.enhance.merge_gap,
        "context_sec": cfg.enhance.context_sec,
        "min_span_sec": cfg.enhance.min_span_sec,
    }


def _claim_cache(
    cfg, backend: str, model, out_sr: int, regions: str, overwrite: bool, write: bool = True
) -> None:
    """Refuse to extend a cache made some other way; stamp it otherwise.

    Without `--overwrite` the stage only fills in calls that are missing, so
    extending a cache made with another model or at another rate is exactly how
    a half-migrated cache happens. `--overwrite` re-enhances every call it
    touches, so the marker is rewritten then -- but a `--limit`ed overwrite
    still leaves old files behind, and `build` checks every file's rate too.
    """
    settings = cache_settings(cfg, backend, model, out_sr, regions)
    path = marker_path(cfg)
    audio_suffix = f".{cfg.enhance.format.lower()}"
    has_audio = any(p.suffix == audio_suffix for p in cfg.paths.enhanced_dir.iterdir())

    if not overwrite:
        if path.exists():
            recorded = read_json(path)
            # `model` is compared only when both sides know it: a stub backend
            # has no /healthz, and that must not be read as a change of model.
            keys = [k for k in settings if k != "model" or (recorded.get(k) and settings[k])]
            changed = {k: (recorded.get(k), settings[k]) for k in keys if recorded.get(k) != settings[k]}
            if changed:
                detail = ", ".join(f"{k}: {a!r} -> {b!r}" for k, (a, b) in changed.items())
                raise SystemExit(
                    f"[{NAME}] {cfg.paths.enhanced_dir} was made differently ({detail}).\n"
                    f"[{NAME}] Extending it would mix two kinds of target in one cache. Point "
                    "paths.enhanced_dir somewhere new, move this one aside, or pass --overwrite."
                )
        elif has_audio:
            raise SystemExit(
                f"[{NAME}] {cfg.paths.enhanced_dir} holds audio but no {MARKER}: it predates "
                "the marker and was made at the source rate by a masking enhancer.\n"
                f"[{NAME}] Move it aside (e.g. work/enhanced_8k), set paths.enhanced_dir to a "
                "new directory, or pass --overwrite."
            )
    if write:
        write_json(path, settings)


def _spans(cfg, call: str, record: dict, n_samples: int, sr: int, out_sr: int, regions: str):
    """Per-channel spans: where to cut the request at `sr`, where to splice at `out_sr`.

    Each span is `(a, b, inner_a, inner_b)`: the request is `samples[a:b]` at the
    source rate, and the kept part lands at `[inner_a, inner_b)` in the *output*
    channel. The splice bounds come from the VAD times in seconds, converted at
    the output rate -- never by scaling a source offset -- so they agree with the
    masks `build` computes at that rate.
    """
    if regions == "full":
        k = out_sr // sr
        return {channel: [(0, n_samples, 0, n_samples * k)] for channel in record["channels"]}

    vad_path = cfg.paths.vad_dir / f"{call}.rttm"
    if not vad_path.exists():
        raise FileNotFoundError(f"no VAD RTTM for {call}")

    intervals: dict[int, list[tuple[float, float]]] = {}
    for segment in read_rttm(vad_path):
        intervals.setdefault(segment.channel - 1, []).append((segment.start, segment.end))

    k = out_sr // sr
    spans: dict[int, list[tuple[int, int, int, int]]] = {}
    for channel in record["channels"]:
        merged = merge_intervals(intervals.get(channel, []), gap=cfg.enhance.merge_gap)
        ranges = []
        for start, end in merged:
            if end - start < cfg.enhance.min_span_sec:
                continue
            # Pad for run-in context; the padding is trimmed off before splicing,
            # so it only ever influences the model, never the output samples.
            a = max(0, int((start - cfg.enhance.context_sec) * sr))
            b = min(n_samples, int((end + cfg.enhance.context_sec) * sr))
            inner_a = max(a * k, int(start * out_sr))
            inner_b = min(b * k, int(end * out_sr))
            if inner_b > inner_a:
                ranges.append((a, b, inner_a, inner_b))
        spans[channel] = ranges
    return spans


def _enhance_channel(enhancer, samples: np.ndarray, sr: int, out_sr: int, ranges) -> np.ndarray:
    """Splice enhanced spans into the channel, carried to the output rate.

    Non-speech keeps its **original** samples rather than becoming silence, so
    the cache stays a faithful "same audio, speech cleaned" artifact that is
    still usable if the VAD is ever recomputed. At a higher output rate those
    samples are a plain resample: nothing above the source's Nyquist, which is
    true of them -- they were never restored. The seam between restored and
    original audio is a step, but every seam sits on a VAD mask edge, which is
    exactly where `build` fades to zero, so no seam survives into the dataset.

    The request starting at source offset `a` comes back starting at output
    offset `a * k`, exactly, because the ratio is an integer and the backend
    returns the same duration it was sent.
    """
    k = out_sr // sr
    out = resample(samples, sr, out_sr)
    if len(out) != len(samples) * k:
        raise ValueError(f"resampled channel is {len(out)} samples, expected {len(samples) * k}")
    out = np.array(out, dtype=np.float32, copy=True)
    for start, stop, inner_a, inner_b in ranges:
        cleaned = enhancer.enhance(samples[start:stop], sr)
        if len(cleaned) != (stop - start) * k:
            raise ValueError(
                f"enhancer returned {len(cleaned)} samples for {stop - start} @ {sr} Hz; "
                f"expected {(stop - start) * k} @ {out_sr} Hz"
            )
        origin = start * k
        out[inner_a:inner_b] = cleaned[inner_a - origin : inner_b - origin]
    return out


def _estimate(cfg, selection, calls, regions: str, realtime: float) -> None:
    """Say up front how long this will take, at the backend's measured speed."""
    seconds = 0.0
    for call, record in calls:
        duration = record.get("duration", 0.0)
        if regions == "full":
            seconds += duration * len(record["channels"])
        else:
            # speech_sec is per side and already excludes the silence.
            seconds += sum(record.get("speech_sec", [])) or duration
    wall = seconds / realtime

    def clock(value: float) -> str:
        return f"{value / 3600:.1f} h" if value >= 3600 else f"{value / 60:.1f} min"

    banner(NAME, f"~{clock(seconds)} of audio to enhance -> roughly {clock(wall)} of wall clock")
    banner(NAME, "resumable per call: interrupt and re-run to continue")
