"""Stage 4 -- one speaker embedding per side of each call.

The same person appears in many calls: this corpus is a call centre, so a small
pool of agents sits opposite thousands of one-off customers. Left alone that
would put one agent's voice into hundreds of training mixtures and let the model
memorize them, so stage 5 has to find those repeats. Everything it needs is one
good vector per speaker per call, which is what this produces.

Two rules from the notebook are kept exactly, because both were paid for:

  - Segments under `min_speech_duration` are dropped and anything over
    `longest_chunk_duration` is split, then the per-chunk embeddings are
    averaged **weighted by duration**. Chunks run 1.2 s to 20 s, and an
    unweighted mean gives a 1.2 s scrap the same vote as a 20 s clean utterance.

  - If either speaker contributes no chunk at all, the whole call is dropped
    rather than half-embedded. Keeping the speaker we can hear would both
    undercount speakers and, worse, throw away that call's cannot-link
    constraint, which is the only hard negative the clustering has.
"""

from __future__ import annotations

import numpy as np

from ..backends import embedders  # noqa: F401  (registers backends)
from ..core.audio import read_audio
from ..core.manifest import read_json, write_json
from ..core.rttm import read_rttm, split_long, subtract_intervals
from ..registry import EMBEDDERS
from .base import banner, progress, require, skip_if_done

NAME = "embed"
REQUIRES = ("filter", "vad")


def add_args(parser) -> None:
    parser.add_argument("--backend", help=f"one of: {', '.join(EMBEDDERS.names())}")
    parser.add_argument("--overwrite", action="store_true", help="recompute all embeddings")
    parser.add_argument("--limit", type=int, help="process only the first N calls")


def run(cfg, args) -> None:
    require(
        {"suitable.json": cfg.paths.suitable_json, "VAD directory": cfg.paths.vad_dir},
        NAME,
    )
    if skip_if_done(cfg.paths.embeddings_npz, args.overwrite, NAME):
        return

    backend = args.backend or cfg.embed.backend
    embedder = EMBEDDERS.create(backend, cfg.embed.options.get(backend, {}))

    suitable = read_json(cfg.paths.suitable_json)
    calls = sorted(suitable.items())
    if args.limit:
        calls = calls[: args.limit]

    banner(NAME, f"backend={backend}  {len(calls)} suitable calls")

    keys: list[str] = []
    vectors: list[np.ndarray] = []
    index: dict[str, dict] = {}
    incomplete: list[dict] = []

    for call, entry in progress(calls, "embed"):
        vad_path = cfg.paths.vad_dir / f"{call}.rttm"
        if not vad_path.exists():
            incomplete.append({"call": call, "reason": "no_vad_rttm"})
            continue

        # Time the filter attributed to a minor label is cut out before the
        # voice is characterized -- otherwise a stray second speaker would be
        # averaged into this speaker's identity vector.
        excluded = {int(c): [tuple(s) for s in spans] for c, spans in entry.get("excluded", {}).items()}
        chunks = _chunks_by_speaker(read_rttm(vad_path), cfg, excluded)
        missing = cfg.embed.expected_speakers - len(chunks)
        if missing > 0:
            incomplete.append({"call": call, "reason": "missing_speaker", "missing": missing})
            continue

        try:
            data, sr = read_audio(entry["audio"])
        except Exception as exc:
            incomplete.append({"call": call, "reason": "unreadable", "error": repr(exc)})
            continue

        # Which channel each VAD speaker label sits on, straight from the label.
        for speaker, intervals in sorted(chunks.items()):
            channel = int(speaker.rsplit("_", 1)[1]) - 1
            if channel < 0 or channel >= data.shape[1]:
                continue

            segments = [
                np.ascontiguousarray(
                    data[int(sr * start) : int(sr * end), channel], dtype=np.float32
                )
                for start, end in intervals
            ]
            weights = np.array([end - start for start, end in intervals], dtype=np.float64)

            embeddings = _embed_all(embedder, segments, sr, cfg.embed.batch_size)
            mean = (embeddings * weights[:, None]).sum(axis=0) / weights.sum()
            mean = mean / max(float(np.linalg.norm(mean)), 1e-12)

            key = f"{call}_{speaker}"
            keys.append(key)
            vectors.append(mean.astype(np.float32))
            index[key] = {
                "call": call,
                "speaker": speaker,
                "channel": channel,
                "chunks": len(intervals),
                "speech_sec": round(float(weights.sum()), 3),
            }

    matrix = (
        np.stack(vectors).astype(np.float32)
        if vectors
        else np.zeros((0, embedder.dim), dtype=np.float32)
    )
    cfg.paths.embeddings_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cfg.paths.embeddings_npz, keys=np.array(keys, dtype=object), embeddings=matrix)
    write_json(cfg.paths.embed_index_json, {"index": index, "incomplete": incomplete})

    slots = cfg.embed.expected_speakers * len(calls)
    banner(NAME, f"embeddings: {len(keys)} / {slots} slots")
    banner(NAME, f"calls skipped for incomplete speakers: {len(incomplete)}")


# --------------------------------------------------------------------------- #
def _chunks_by_speaker(segments, cfg, excluded=None) -> dict[str, list[tuple[float, float]]]:
    """Usable speech chunks per speaker label, short dropped and long split."""
    excluded = excluded or {}

    by_speaker: dict[str, list[tuple[float, float]]] = {}
    channel_of: dict[str, int] = {}
    for segment in segments:
        by_speaker.setdefault(segment.speaker, []).append((segment.start, segment.end))
        channel_of[segment.speaker] = segment.channel - 1

    chunks: dict[str, list[tuple[float, float]]] = {}
    for speaker, intervals in by_speaker.items():
        # Subtract first, then apply the duration rules: a segment that is only
        # long enough because it includes another speaker's time should not
        # qualify on that basis.
        kept = subtract_intervals(intervals, excluded.get(channel_of[speaker], []))
        for start, end in kept:
            if end - start < cfg.embed.min_speech_duration:
                continue
            # A single VAD segment can run past a minute. Split it so no one
            # chunk carries most of the weight and the model never sees an
            # unusually long utterance.
            for piece in split_long(start, end, cfg.embed.longest_chunk_duration):
                chunks.setdefault(speaker, []).append(piece)
    return chunks


def _embed_all(embedder, segments, sr: int, batch_size: int) -> np.ndarray:
    out = [
        embedder.embed_batch(segments[i : i + batch_size], sr)
        for i in range(0, len(segments), batch_size)
    ]
    return np.concatenate(out, axis=0)
