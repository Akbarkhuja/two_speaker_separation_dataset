"""Stage 3 -- per-channel speech masks into `work/vad/`.

These masks, not the diarizer's output, decide what counts as speech everywhere
downstream: which audio gets embedded, which segment goes to the gender
service, what survives zerofying, and whether a training window has both people
in it.

That split of duties is on purpose. A diarizer answers "who is talking", and its
segment boundaries are turn-level and loose -- it happily wraps a pause inside a
turn, because for diarization that costs nothing. Here the boundary is applied
to samples: everything outside it is multiplied by zero. Using turn boundaries
would leave seconds of noise inside the "source" signal.

Speaker labels are `speaker_{channel + 1}`, so a key like
`<call>_speaker_1` names a channel and therefore a person for the rest of the
pipeline.
"""

from __future__ import annotations

from ..backends import vad as vad_backends  # noqa: F401  (registers backends)
from ..core.audio import demux, read_audio
from ..core.manifest import read_json, write_failures
from ..core.rttm import Segment, write_rttm
from ..registry import VADS
from .base import banner, progress, require

NAME = "vad"
REQUIRES = ("filter",)


def add_args(parser) -> None:
    parser.add_argument("--backend", help=f"one of: {', '.join(VADS.names())}")
    parser.add_argument("--overwrite", action="store_true", help="redo calls that have a VAD RTTM")
    parser.add_argument("--limit", type=int, help="process only the first N calls")
    parser.add_argument(
        "--all-calls",
        action="store_true",
        help="run over every diarized call, not just the suitable ones",
    )


def run(cfg, args) -> None:
    backend = args.backend or cfg.vad.backend
    detector = VADS.create(backend, cfg.vad.options.get(backend, {}))

    if args.all_calls:
        from ..core.discovery import collect_audio, index_by_call

        calls = index_by_call(collect_audio(cfg.paths.audio_dirs, cfg.paths.pattern))
    else:
        require({"suitable.json": cfg.paths.suitable_json}, NAME)
        suitable = read_json(cfg.paths.suitable_json)
        calls = {call: entry["audio"] for call, entry in suitable.items()}

    out_dir = cfg.paths.vad_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    todo = sorted(calls.items())
    if not args.overwrite:
        todo = [item for item in todo if not (out_dir / f"{item[0]}.rttm").exists()]
    if args.limit:
        todo = todo[: args.limit]

    banner(NAME, f"backend={backend}  {len(calls)} calls, {len(todo)} to process")
    if not todo:
        return

    failed: list[tuple[str, str]] = []
    for call, audio_path in progress(todo, "vad"):
        try:
            data, sr = read_audio(audio_path)
            segments = []
            for channel in range(data.shape[1]):
                intervals = detector.speech(demux(data, channel), sr)
                segments.extend(
                    Segment(
                        start=start,
                        end=end,
                        speaker=f"speaker_{channel + 1}",
                        channel=channel + 1,
                    )
                    for start, end in intervals
                )
            segments.sort(key=lambda s: (s.start, s.channel))
            write_rttm(out_dir / f"{call}.rttm", call, segments)
        except Exception as exc:
            failed.append((str(audio_path), repr(exc)))

    write_failures(cfg.paths.work_dir / "vad_failed.tsv", failed)
    banner(NAME, f"{len(failed)} failed; see work/vad_failed.tsv" if failed else "done, no failures")
