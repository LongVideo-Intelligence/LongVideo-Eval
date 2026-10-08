"""AKS: Adaptive Keyframe Sampling (Tang et al., CVPR 2025).

Query-aware keyframe selection that balances RELEVANCE (keep frames that match the
question) against COVERAGE (keep frames from across the timeline). This is an independent
implementation of the algorithm described in the paper.

THE ALGORITHM. Per-frame relevance scores are min-max normalised, and the timeline is
partitioned recursively:

  * A segment is *peaked* when the mean of its top-``num`` scores exceeds its overall mean
    by more than ``t1`` (and its standard deviation exceeds ``t2``). A peaked segment is
    kept whole: relevance decides inside it.
  * Any other segment is cut into two halves, down to ``all_depth`` levels: coverage decides.

A kept segment at depth ``d`` contributes its ``int(num / 2**d)`` highest-scoring frames, so
the budget is spread evenly over a segment's halves and the total is at most ``num``. Because
of the integer division it can be slightly below ``num``; the realized frame count is
whatever the selection returns, and that is what the harness meters.

DEFAULTS. ``t1=0.8``, ``t2=-100`` (which leaves the standard-deviation test always true) and
``all_depth=5``, the published configuration. The depth is capped at ``floor(log2(num))``:
a deeper segment would be entitled to ``int(num / 2**d) == 0`` frames, so with 16 keyframes
an uncapped depth of 5 can select nothing at all.

SCORING. The paper's default scorer is a BLIP image-text matching model. This harness uses
CLIP ViT-B/32 for every query-aware selector, so they differ only in how they turn the same
scores into a frame set. Any ``Callable[[FramePool, str], Sequence[float]]`` can be injected.

COST. As for every selector that reads the whole pool: the decoder charges for decoding it,
the orchestrator times the scoring, and ``Selection.cost`` is left empty.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

from ..._registry import register
from ...interfaces import Budget, FramePool, RoundContext, Selection, Selector
from .clip_scorer import default_clip_scorer

AKS_T1 = 0.8
AKS_T2 = -100
AKS_ALL_DEPTH = 5


def effective_all_depth(num: int, all_depth: int = AKS_ALL_DEPTH) -> int:
    """Cap the recursion depth at ``floor(log2(num))`` so the deepest segments still get at
    least one frame (see the module docstring)."""
    if num <= 1:
        return 0
    return min(all_depth, num.bit_length() - 1)


def _normalize(scores: Sequence[float]) -> list[float]:
    """Min-max normalise to [0, 1]; a flat input becomes all zeros."""
    lo, hi = min(scores), max(scores)
    if hi <= lo:
        return [0.0] * len(scores)
    span = hi - lo
    return [(s - lo) / span for s in scores]


def _top(scores: Sequence[float], k: int) -> list[int]:
    """Positions of the ``k`` largest scores, highest first, earlier position on a tie."""
    return sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k]


def _is_peaked(scores: Sequence[float], num: int, t1: float, t2: float) -> bool:
    mean = sum(scores) / len(scores)
    std = (sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5
    top = _top(scores, num)
    return (sum(scores[i] for i in top) / len(top)) - mean > t1 and std > t2


def _leaves(scores, start: int, depth: int, num: int, t1: float, t2: float, all_depth: int):
    """Yield ``(start, scores, depth)`` for every segment that is kept whole."""
    if not scores:
        return
    if _is_peaked(scores, num, t1, t2) or depth >= all_depth:
        yield start, scores, depth
        return
    half = len(scores) // 2
    yield from _leaves(scores[:half], start, depth + 1, num, t1, t2, all_depth)
    yield from _leaves(scores[half:], start + half, depth + 1, num, t1, t2, all_depth)


def adaptive_keyframe_indices(
    scores: Sequence[float], num: int,
    t1: float = AKS_T1, t2: float = AKS_T2, all_depth: int = AKS_ALL_DEPTH,
) -> list[int]:
    """AKS over a per-frame score array -> selected frame indices in temporal order.

    ``num`` is the keyframe budget. With fewer frames than ``num`` every index is returned.
    Returns at most ``num`` indices, ascending, without duplicates.
    """
    length = len(scores)
    if num <= 0 or length == 0:
        return []
    if length < num:
        return list(range(length))
    out: list[int] = []
    for start, seg, depth in _leaves(_normalize(scores), 0, 0, num, t1, t2, all_depth):
        out.extend(start + i for i in _top(seg, int(num / 2 ** depth)))
    return sorted(out)


@register("select", "aks")
class AKSSelector(Selector):
    """Adaptive Keyframe Sampling. ``budget.frame_count`` is the keyframe budget.

    ``scorer`` is an injectable ``Callable[[FramePool, str], Sequence[float]]``; it defaults
    to the CLIP ViT-B/32 scorer (loaded lazily).
    """

    def __init__(
        self,
        scorer: Optional[Callable[[FramePool, str], Sequence[float]]] = None,
        t1: float = AKS_T1, t2: float = AKS_T2, all_depth: int = AKS_ALL_DEPTH,
    ) -> None:
        self._scorer = scorer if scorer is not None else default_clip_scorer()
        self.t1 = t1
        self.t2 = t2
        self.all_depth = all_depth

    def select(self, pool: FramePool, query: str, budget: Budget,
               context: Optional[RoundContext] = None) -> Selection:
        n = len(pool.timestamps)
        num = n if budget.frame_count is None else budget.frame_count  # 0 means 0
        if num <= 0 or n == 0:
            idx: list[int] = []
        else:
            scores = self._scorer(pool, query)
            if len(scores) != n:
                raise ValueError(
                    f"scorer returned {len(scores)} scores for {n} pool frames; "
                    "keyframe scoring must cover the whole decoded pool"
                )
            idx = adaptive_keyframe_indices(
                scores, num, t1=self.t1, t2=self.t2,
                all_depth=effective_all_depth(num, self.all_depth),
            )
        res = [budget.resolution] * len(idx) if budget.resolution is not None else None
        return Selection(indices=idx, per_frame_resolution=res, signal="similarity")
