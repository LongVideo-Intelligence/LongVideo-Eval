"""Uniform frame selector — the neutral baseline sampler."""
from __future__ import annotations

from typing import Optional

from ..._registry import register
from ...interfaces import Budget, FramePool, RoundContext, Selection, Selector


@register("select", "uniform")
class UniformSelector(Selector):
    """Evenly spaced frames. This is also the sampler behind the dense low-res baseline when
    paired with a high frame count and a low per-frame resolution."""

    def select(self, pool: FramePool, query: str, budget: Budget,
               context: Optional[RoundContext] = None) -> Selection:
        n = len(pool.timestamps)
        k = n if budget.frame_count is None else budget.frame_count  # 0 means 0, not "unset"
        k = max(0, min(k, n))
        if k == 0:
            idx = []
        elif k >= n:
            idx = list(range(n))
        else:
            # CONVENTION (selector "uniform"): segment-center sampling, NOT the decoder's
            # endpoint linspace/truncation (decoder.uniform_decode_indices). For
            # pool256->keep16 style setups this is a DIFFERENT frame set than decoding 16
            # frames directly with the decoder; result claims must keep the distinction
            # explicit (pool-selection convention, not decode-16 convention).
            # segment-center (midpoint) sampling: symmetric, covers start AND end of the pool
            idx = [min(n - 1, int((i + 0.5) * n / k)) for i in range(k)]
        res = [budget.resolution] * len(idx) if budget.resolution is not None else None
        return Selection(indices=idx, per_frame_resolution=res, signal="uniform")


@register("select", "lohi-uniform")
class LoHiUniformSelector(Selector):
    """LoHi-Uniform: every pooled frame goes to the low-resolution video stream, and K of
    them, at regular intervals, additionally go through the image pathway at high resolution.

    ``Selection.indices`` is the whole Lo-V frame set (temporal coverage is never traded
    away) and ``Selection.hi_indices`` is the K-subset. Picking by index costs nothing, which
    makes this the control for ``lohi-semdiv``: it shows how much of a gain comes from *which*
    frames get the high-resolution pass and how much from having one at all.
    """

    def select(self, pool: FramePool, query: str, budget: Budget,
               context: Optional[RoundContext] = None) -> Selection:
        n = 0 if pool.frames is None else len(pool.frames)
        if budget.hi_i_count is None:
            raise ValueError(
                "lohi-uniform needs Budget.hi_i_count (K, the number of high-resolution "
                "frames); got None. Use a presentation='lohi' setup (runners/setups.py)."
            )
        if n == 0:
            return Selection(indices=[], hi_indices=[], signal="none")
        k = max(0, min(budget.hi_i_count, n))
        # Regular intervals over [0, n-1] (linspace + truncation, the decoder's own sampling
        # style), so the placement is reproducible index for index.
        hi = [int((n - 1) * i / (k - 1)) for i in range(k)] if k > 1 else ([n // 2] if k else [])
        hi = sorted(set(hi))
        res = None if budget.resolution is None else [budget.resolution] * n
        return Selection(
            indices=list(range(n)), per_frame_resolution=res, hi_indices=hi, signal="none"
        )
