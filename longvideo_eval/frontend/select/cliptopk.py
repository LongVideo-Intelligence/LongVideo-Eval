"""CLIP-TopK keyframe selection: keep the frames most similar to the question.

The flat baseline of query-aware keyframe selection. Every frame of the decoded pool is
scored against the question with CLIP, and the ``frame_count`` highest-scoring frames are
kept. It optimises relevance only, so the kept frames can cluster in one part of the video.

COST. The selector reads the WHOLE pool, so the setup decodes the whole pool and the decoder
charges for it. The CLIP forward is timed by the orchestrator's select stage, and
``Selection.cost`` is left empty to avoid double counting.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

from ..._registry import register
from ...interfaces import Budget, FramePool, RoundContext, Selection, Selector
from .clip_scorer import default_clip_scorer


def topk_by_score(scores: Sequence[float], k: int) -> list[int]:
    """Indices of the ``k`` highest scores, returned in ascending (temporal) index order.

    Ties break by smaller index for determinism; the kept indices are then sorted ascending
    so the model sees keyframes in time order. ``k`` is clamped to ``[0, len(scores)]``.
    """
    n = len(scores)
    k = max(0, min(k, n))
    if k == 0:
        return []
    if k >= n:
        return list(range(n))
    ranked = sorted(range(n), key=lambda i: (-scores[i], i))[:k]
    return sorted(ranked)


@register("select", "cliptopk")
class ClipTopKSelector(Selector):
    """Top-``frame_count`` pool frames by CLIP query similarity, temporally sorted.

    ``scorer`` is an injectable ``Callable[[FramePool, str], Sequence[float]]`` returning one
    relevance score per pool frame; it defaults to the CLIP ViT-B/32 scorer (loaded lazily).
    """

    def __init__(
        self, scorer: Optional[Callable[[FramePool, str], Sequence[float]]] = None
    ) -> None:
        self._scorer = scorer if scorer is not None else default_clip_scorer()

    def select(self, pool: FramePool, query: str, budget: Budget,
               context: Optional[RoundContext] = None) -> Selection:
        n = len(pool.timestamps)
        k = n if budget.frame_count is None else budget.frame_count  # 0 means 0, not "unset"
        if k <= 0 or n == 0:
            idx: list[int] = []
        else:
            scores = self._scorer(pool, query)
            if len(scores) != n:
                raise ValueError(
                    f"scorer returned {len(scores)} scores for {n} pool frames; "
                    "keyframe scoring must cover the whole decoded pool"
                )
            idx = topk_by_score(scores, k)
        res = [budget.resolution] * len(idx) if budget.resolution is not None else None
        return Selection(indices=idx, per_frame_resolution=res, signal="retrieval")
