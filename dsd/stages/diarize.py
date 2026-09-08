"""Stage 1 -- diarize every call, one channel at a time, into `work/rttms/`.

Diarizing each channel alone is what makes the speaker identity exact: on a
telephony recording one party owns one channel, so "who is this" is answered by
the demux rather than by the model. The model is only asked the easier question
of when that person speaks, and whether more than one person ever used that
channel -- which is exactly what the next stage filters on.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ..backends import diarizers  # noqa: F401  (registers backends)
from ..core.audio import read_audio
from ..core.discovery import collect_audio, index_by_call, pending
from ..core.manifest import write_failures
from ..core.rttm import write_rttm
from ..registry import DIARIZERS
from .base import banner, progress

NAME = "diarize"
REQUIRES = ()


def add_args(parser) -> None:
    parser.add_argument("--backend", help=f"one of: {', '.join(DIARIZERS.names())}")
    parser.add_argument("--workers", type=int, help="concurrent files (network backends only)")
    parser.add_argument("--overwrite", action="store_true", help="redo calls that have an RTTM")
    parser.add_argument("--limit", type=int, help="process only the first N calls")


def run(cfg, args) -> None:
    backend = args.backend or cfg.diarize.backend
    workers = args.workers or cfg.diarize.workers
    diarizer = DIARIZERS.create(backend, cfg.diarize.options.get(backend, {}))

    audios = collect_audio(cfg.paths.audio_dirs, cfg.paths.pattern)
    if not audios:
        raise SystemExit("no audio files found; check paths.audio_dirs and paths.pattern")
    index = index_by_call(audios)

    out_dir = cfg.paths.rttm_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = pending(
        sorted(index.items()),
        lambda item: out_dir / f"{item[0]}.rttm",
        overwrite=args.overwrite,
        limit=args.limit,
    )

    banner(NAME, f"backend={backend}  {len(index)} calls found, {len(todo)} to process")
    if not todo:
        return

    def process(item: tuple[str, str]) -> None:
        call, path = item
        data, sr = (read_audio(path) if diarizer.needs_audio else (None, None))
        segments = diarizer.diarize(call, Path(path), data, sr)
        write_rttm(out_dir / f"{call}.rttm", call, segments)

    failed: list[tuple[str, str]] = []
    if workers > 1:
        # Worth it only for the HTTP backend; the local GPU ones serialize on
        # the device anyway and would just contend.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(process, item): item for item in todo}
            for future in progress(as_completed(futures), "diarize", total=len(futures)):
                call, path = futures[future]
                try:
                    future.result()
                except Exception as exc:  # one bad file must not stop the corpus
                    failed.append((path, repr(exc)))
    else:
        for item in progress(todo, "diarize"):
            try:
                process(item)
            except Exception as exc:
                failed.append((item[1], repr(exc)))

    write_failures(cfg.paths.work_dir / "diarize_failed.tsv", failed)
    if failed:
        banner(NAME, f"{len(failed)} failed; see work/diarize_failed.tsv")
    else:
        banner(NAME, "done, no failures")
