"""Complete-linkage agglomerative clustering with cannot-link constraints.

Ported from the notebook and `sandbox/sep_data_pipeline/define_unique_speakers.py`.
Three things about it are deliberate:

  - **Complete linkage.** Every cross-pair between two clusters must clear the
    threshold before they merge, which resists the chaining that would
    otherwise walk a cluster from one voice to another through near-misses.

  - **Cannot-link via -inf.** The two sides of one call are different people by
    construction, so their similarity is set to -inf. Because complete linkage
    takes the element-wise *minimum* when merging, the constraint is inherited
    for free: a merged cluster is blocked from `c` exactly when either half was.

  - **In-place merges.** `M` holds current cluster-to-cluster similarity and is
    updated on each merge rather than recomputed from members, which is what
    makes this usable at a few thousand speaker instances.

THRESHOLD lives on the rescaled `(cos + 1) / 2` scale, so the default 0.8 means
a raw cosine of 0.6.
"""

from __future__ import annotations

import numpy as np

from ..registry import CLUSTERERS


class ConstrainedAHC:
    def __init__(self, options: dict):
        self.threshold = float(options.get("threshold", 0.8))
        # 'complete' -- all cross-pairs must agree; 'average' -- the mean.
        self.linkage = options.get("linkage", "complete")

    def fit(self, keys: list[str], similarity: np.ndarray, groups: list[str]) -> list[list[str]]:
        n = len(keys)
        if n == 0:
            return []

        clusters = {i: {keys[i]} for i in range(n)}

        M = similarity.astype(np.float64, copy=True)
        _, call_ids = np.unique(np.asarray(groups), return_inverse=True)
        M[call_ids[:, None] == call_ids[None, :]] = -np.inf  # covers the diagonal too

        while True:
            a, b = np.unravel_index(np.argmax(M), M.shape)
            best = M[a, b]
            if not np.isfinite(best) or best <= self.threshold:
                break

            a, b = int(a), int(b)
            M[a] = np.minimum(M[a], M[b]) if self.linkage == "complete" else (M[a] + M[b]) / 2
            M[:, a] = M[a]
            M[a, a] = -np.inf
            M[b, :] = -np.inf
            M[:, b] = -np.inf

            clusters[a] |= clusters.pop(b)

        index = {key: i for i, key in enumerate(keys)}
        result = [sorted(members) for members in clusters.values()]
        for members in result:
            calls = [groups[index[k]] for k in members]
            assert len(set(calls)) == len(calls), "cannot-link violated"
        # Largest first, so cluster ids are stable and the big ones are easy to audit.
        result.sort(key=lambda members: (-len(members), members[0]))
        return result


@CLUSTERERS.register("constrained_ahc")
def _build(options: dict) -> ConstrainedAHC:
    return ConstrainedAHC(options)


# --------------------------------------------------------------------------- #
# diagnostics
# --------------------------------------------------------------------------- #
def diagnose(
    keys: list[str],
    similarity: np.ndarray,
    groups: list[str],
    channels: list[int],
    threshold: float,
) -> dict:
    """Measure whether the threshold sits where it should.

    Two numbers matter and they say different things:

      - **same-call pairs** are known-different speakers, so their similarity
        distribution is what "different people" looks like. The threshold has to
        sit above it.

      - **cross-call, same-channel pairs** are the merges that actually happen,
        and they are the harder case. In the notebook's run 5.5% of them cleared
        0.8 versus 0.17% of cross-channel pairs, so the same-call number is a
        floor on the false-merge rate, not an estimate of it. Treat this whole
        block as a smoke test; the real check is listening to the clusters.
    """
    n = len(keys)
    report: dict = {"threshold": threshold, "instances": n}
    if n < 2:
        return report

    # Boolean matrices rather than an itertools pair loop: at corpus scale n is
    # a few thousand, so the loop would be tens of millions of Python
    # iterations while this is a handful of numpy ops.
    _, call_ids = np.unique(np.asarray(groups), return_inverse=True)
    channel_ids = np.asarray(channels)
    upper = np.triu(np.ones((n, n), dtype=bool), k=1)

    same_call_mask = (call_ids[:, None] == call_ids[None, :]) & upper
    same_channel_mask = (channel_ids[:, None] == channel_ids[None, :]) & upper & ~same_call_mask
    diff_channel_mask = upper & ~same_call_mask & ~same_channel_mask

    same_call = similarity[same_call_mask]
    cross_same_channel = similarity[same_channel_mask]
    cross_diff_channel = similarity[diff_channel_mask]

    def summarize(values: np.ndarray) -> dict | None:
        if values.size == 0:
            return None
        array = np.asarray(values)
        return {
            "n": int(array.size),
            "mean": float(array.mean()),
            "p95": float(np.percentile(array, 95)),
            "p99": float(np.percentile(array, 99)),
            "max": float(array.max()),
            "above_threshold_pct": float((array > threshold).mean() * 100),
        }

    report["known_different_same_call"] = summarize(same_call)
    report["cross_call_same_channel"] = summarize(cross_same_channel)
    report["cross_call_diff_channel"] = summarize(cross_diff_channel)
    if same_call.size:
        report["threshold_sweep"] = {
            f"{t:.2f}": float((same_call > t).mean() * 100)
            for t in (0.76, 0.78, 0.80, 0.82, 0.85, 0.88)
        }
    return report


def print_report(report: dict) -> None:
    threshold = report["threshold"]
    known = report.get("known_different_same_call")
    if known:
        print("[cluster] known-different similarity (two sides of one call):")
        print(
            f"[cluster]   mean={known['mean']:.3f} p95={known['p95']:.3f} "
            f"p99={known['p99']:.3f} max={known['max']:.3f}"
        )
        print(
            f"[cluster]   above threshold={threshold}: {known['above_threshold_pct']:.2f}% "
            "-- a few % is workable; tens of % means the embeddings are broken"
        )
    for label, key in (
        ("cross-call, same channel", "cross_call_same_channel"),
        ("cross-call, diff channel", "cross_call_diff_channel"),
    ):
        stats = report.get(key)
        if stats:
            print(
                f"[cluster] {label}: mean={stats['mean']:.4f} p99={stats['p99']:.3f} "
                f"above {threshold}: {stats['above_threshold_pct']:.2f}%"
            )
    sweep = report.get("threshold_sweep")
    if sweep:
        print("[cluster] threshold sweep (% of known-different pairs above):")
        for value, pct in sweep.items():
            print(f"[cluster]   thr={value}  {pct:5.2f}%")
