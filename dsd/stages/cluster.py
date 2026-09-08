"""Stage 5 -- group speaker instances into global identities.

Input is one embedding per side of each call; output is a map from a global
speaker id to every `<call>_speaker_<n>` that is the same person. `select` uses
it to cap how often one voice can appear, and to keep splits speaker-disjoint.

Every embedding is unit length, so the Gram matrix `E @ E.T` *is* the cosine
matrix -- rescaled to `(cos + 1) / 2` to match the threshold's scale. That one
matrix multiply replaces the O(n^2) Python pair loop the notebook started with.
"""

from __future__ import annotations

import numpy as np

from ..clustering import constrained_ahc  # noqa: F401  (registers backends)
from ..clustering.constrained_ahc import diagnose, print_report
from ..core.manifest import read_json, write_json
from ..registry import CLUSTERERS
from .base import banner, require, skip_if_done

NAME = "cluster"
REQUIRES = ("embed",)


def add_args(parser) -> None:
    parser.add_argument("--backend", help=f"one of: {', '.join(CLUSTERERS.names())}")
    parser.add_argument("--threshold", type=float, help="merge threshold on the (cos+1)/2 scale")
    parser.add_argument("--report", action="store_true", help="print threshold diagnostics")
    parser.add_argument("--overwrite", action="store_true", help="recluster")


def run(cfg, args) -> None:
    require({"embeddings.npz": cfg.paths.embeddings_npz}, NAME)
    if skip_if_done(cfg.paths.speakers_json, args.overwrite, NAME):
        return

    payload = np.load(cfg.paths.embeddings_npz, allow_pickle=True)
    keys = [str(k) for k in payload["keys"]]
    embeddings = payload["embeddings"].astype(np.float32)
    if not keys:
        raise SystemExit("embeddings.npz is empty; run the embed stage first")

    meta = read_json(cfg.paths.embed_index_json)["index"]
    groups = [meta[k]["call"] for k in keys]
    channels = [meta[k]["channel"] for k in keys]

    # Unit-length rows, so this is the cosine matrix; rescale [-1,1] -> [0,1].
    similarity = ((embeddings @ embeddings.T + 1.0) / 2.0).astype(np.float64)

    threshold = args.threshold if args.threshold is not None else cfg.cluster.threshold
    report = diagnose(keys, similarity, groups, channels, threshold)
    if args.report or cfg.cluster.report:
        print_report(report)

    backend = args.backend or cfg.cluster.backend
    options = dict(cfg.cluster.options.get(backend, {}))
    options.setdefault("threshold", threshold)
    clusterer = CLUSTERERS.create(backend, options)

    clusters = clusterer.fit(keys, similarity, groups)

    speakers = {f"spk{i:05d}": members for i, members in enumerate(clusters, start=1)}
    by_key = {member: sid for sid, members in speakers.items() for member in members}

    write_json(
        cfg.paths.speakers_json,
        {
            "backend": backend,
            "threshold": threshold,
            "diagnostics": report,
            "speakers": speakers,
            "by_key": by_key,
        },
    )

    sizes = sorted((len(m) for m in clusters), reverse=True)
    banner(NAME, f"{len(keys)} speaker instances -> {len(clusters)} unique speakers")
    banner(NAME, f"largest clusters: {sizes[:8]}")
    banner(NAME, f"singletons: {sum(1 for s in sizes if s == 1)}")
