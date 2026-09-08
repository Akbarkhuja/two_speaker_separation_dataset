"""Stage 8 -- denoise the two channels of each selected call, into `work/enhanced/`.

Why this is its own stage rather than part of `build`, in three measurements:

  - **Enhancement is non-linear**, so `enhance(s1 + s2) != enhance(s1) + enhance(s2)`.
    It has to be applied to the two isolated channels *before* they are summed,
    or `mix == s1 + s2` -- the invariant `verify` checks and every separation
    loss assumes -- stops holding. Enhancing each channel and then summing keeps
    it exact by construction.
  - **Throughput is ~3.2x realtime and does not improve with concurrency**
    (3.17x / 3.28x / 3.27x at 1 / 4 / 8 workers: one replica, adaptive batch cap
    of 1, so extra workers only queue). `build` is the stage that gets re-run
    most while tuning overlap and SIR, and paying tens of hours on every
    `--overwrite` is not viable.
  - **Length is preserved**, so a cached enhanced channel drops straight into
    `build` with every VAD offset still valid.

Only the VAD speech spans are enhanced. Everything outside them is zerofied by
`build` regardless, so enhancing it would be spending the scarcest resource in
the pipeline on samples that are about to be multiplied by zero.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from ..backends import speech_enhancement  # noqa: F401  (registers backends)
from ..core.audio import demux, read_audio, write_wav
from ..core.manifest import read_json, write_failures
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


def enhanced_path(cfg, call: str) -> Path:
    """Where a call's enhanced audio lives. Shared with `build`."""
    return cfg.paths.enhanced_dir / f"{call}.{cfg.enhance.format.lower()}"


def run(cfg, args) -> None:
    require(
        {"selection.json": cfg.paths.selection_json, "VAD directory": cfg.paths.vad_dir},
        NAME,
    )

    backend = args.backend or cfg.enhance.backend
    options = dict(cfg.enhance.options.get(backend, {}))
    if args.base_url:
        options["base_url"] = args.base_url
    enhancer = ENHANCERS.create(backend, options)

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

    banner(NAME, f"backend={backend}  regions={regions}  {len(calls)} calls to enhance")
    if not calls:
        banner(NAME, "nothing to do")
        return

    # Fail now, with a useful message, rather than after the first request.
    if hasattr(enhancer, "health"):
        try:
            health = enhancer.health()
            banner(
                NAME,
                f"service ok: {health.get('model')} "
                f"replicas={health.get('replicas')} batch_cap={health.get('max_batch_size')}",
            )
        except Exception as exc:
            raise SystemExit(
                f"[{NAME}] the enhancement service is not reachable: {exc!r}\n"
                f"[{NAME}] start it with: "
                "cd /home/akbar/craft/prod/mossformergan_serve && docker compose up -d"
            )

    _estimate(cfg, selection, calls, regions)

    failed: list[tuple[str, str]] = []
    started = time.time()

    def process(item: tuple[str, dict]) -> None:
        call, record = item
        data, sr = read_audio(record["audio"])
        spans_by_channel = _spans(cfg, call, record, data.shape[0], sr, regions)

        channels = []
        for channel in range(data.shape[1]):
            samples = demux(data, channel)
            # A channel the selection never named is copied through untouched
            # rather than raising: the cache must stay the same shape as the source.
            ranges = spans_by_channel.get(channel, [])
            channels.append(_enhance_channel(enhancer, samples, sr, ranges))

        write_wav(
            enhanced_path(cfg, call),
            np.stack(channels, axis=1),
            sr,
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
    banner(NAME, f"{len(failed)} failed; see work/enhance_failed.tsv" if failed else "done, no failures")


# --------------------------------------------------------------------------- #
def _spans(cfg, call: str, record: dict, n_samples: int, sr: int, regions: str):
    """Per-channel (start, end) sample ranges to send to the enhancer."""
    if regions == "full":
        return {channel: [(0, n_samples)] for channel in record["channels"]}

    vad_path = cfg.paths.vad_dir / f"{call}.rttm"
    if not vad_path.exists():
        raise FileNotFoundError(f"no VAD RTTM for {call}")

    intervals: dict[int, list[tuple[float, float]]] = {}
    for segment in read_rttm(vad_path):
        intervals.setdefault(segment.channel - 1, []).append((segment.start, segment.end))

    spans: dict[int, list[tuple[int, int]]] = {}
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
            inner_a = max(a, int(start * sr))
            inner_b = min(b, int(end * sr))
            if inner_b > inner_a:
                ranges.append((a, b, inner_a, inner_b))
        spans[channel] = ranges
    return spans


def _enhance_channel(enhancer, samples: np.ndarray, sr: int, ranges) -> np.ndarray:
    """Splice enhanced spans into a copy of the channel.

    Non-speech keeps its **original** samples rather than becoming silence, so
    the cache stays a faithful "same audio, speech cleaned" artifact that is
    still usable if the VAD is ever recomputed. The seam between enhanced and
    original audio is a small step -- the enhancer is measurably quieter, about
    0.89x RMS -- but every seam sits on a VAD mask edge, which is exactly where
    `build` fades to zero, so no seam survives into the dataset.
    """
    out = np.array(samples, dtype=np.float32, copy=True)
    for entry in ranges:
        if len(entry) == 2:  # regions == "full": no padding to trim
            start, stop = entry
            out[start:stop] = enhancer.enhance(samples[start:stop], sr)
            continue

        start, stop, inner_a, inner_b = entry
        cleaned = enhancer.enhance(samples[start:stop], sr)
        out[inner_a:inner_b] = cleaned[inner_a - start : inner_b - start]
    return out


def _estimate(cfg, selection, calls, regions: str) -> None:
    """Say up front how long this will take; it is measured in hours, not minutes."""
    seconds = 0.0
    for call, record in calls:
        duration = record.get("duration", 0.0)
        if regions == "full":
            seconds += duration * len(record["channels"])
        else:
            # speech_sec is per side and already excludes the silence.
            seconds += sum(record.get("speech_sec", [])) or duration
    # 3.2x realtime, measured against the live service; concurrency does not
    # change it, so this is a straight division rather than a guess.
    wall = seconds / 3.2

    def clock(value: float) -> str:
        return f"{value / 3600:.1f} h" if value >= 3600 else f"{value / 60:.1f} min"

    banner(NAME, f"~{clock(seconds)} of audio to enhance -> roughly {clock(wall)} of wall clock")
    banner(NAME, "resumable per call: interrupt and re-run to continue")
