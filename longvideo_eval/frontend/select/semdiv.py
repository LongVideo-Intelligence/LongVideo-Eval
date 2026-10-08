"""LoHi-SemDiv: pick the K high-resolution frames by query relevance x visual diversity.

WHAT IT SELECTS. Unlike a keyframe selector, SemDiv does NOT choose which frames the model
sees: the low-resolution video stream (Lo-V) already carries ALL ``N`` frames. It chooses the
``K << N`` frames that additionally get a HIGH-RESOLUTION pass through the image pathway
(Hi-I). Temporal coverage is already supplied by Lo-V, so the K slots should instead be
(i) mutually diverse and (ii) relevant to the question.

THE OBJECTIVE. A quality-similarity determinantal point process (DPP) over L2-normalised CLIP
frame embeddings ``e_j``, with the query similarity ``rho_j = cos(CLIP_txt(Q), e_j)`` min-max
normalised and power-sharpened into a quality score::

    rho~_j = (rho_j - min rho) / (max rho - min rho)
    q_j    = rho~_j ** alpha
    L      = diag(q) @ E @ E.T @ diag(q)

Diagonals ``q_j^2`` encode relevance; off-diagonals couple relevance with pairwise visual
similarity, so a subset of near-duplicate frames has a small ``det(L_S)``. The MAP subset is
taken by the standard greedy Cholesky update (Chen, Zhang & Zhou, NeurIPS 2018).

``alpha`` defaults to 5.0 and is a constructor argument.

COST. One CLIP ViT-B/32 forward over the pool plus one text forward. The wall time is
charged by the orchestrator's select-stage timer, so ``Selection.cost`` is left empty here.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

from ..._registry import register
from ...interfaces import Budget, FramePool, RoundContext, Selection, Selector
from .clip_scorer import default_clip_scorer

DEFAULT_ALPHA = 5.0


def quality_scores(similarities: Sequence[float], alpha: float = DEFAULT_ALPHA) -> list[float]:
    """Min-max normalise raw query-frame cosines, then power-sharpen by ``alpha``.

    Degenerate pool (all cosines equal, e.g. N==1) -> every quality is 1.0 rather than a 0/0:
    with no relevance contrast left, the kernel reduces to pure diversity.
    """
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    if not similarities:
        return []
    lo, hi = min(similarities), max(similarities)
    span = hi - lo
    if span <= 0:
        return [1.0] * len(similarities)
    return [(((s - lo) / span) ** alpha) for s in similarities]


def dpp_greedy(kernel: Sequence[Sequence[float]], k: int) -> list[int]:
    """Greedy DPP MAP via the Cholesky update, O(N*k^2), numpy-free.

    Returns SELECTION ORDER (most valuable first), not temporal order; the selector sorts.
    Stops early when the best remaining marginal gain collapses to ~0, so a pool of
    near-duplicates can yield fewer than ``k``.
    """
    n = len(kernel)
    if k <= 0 or n == 0:
        return []
    k = min(k, n)
    d2 = [float(kernel[i][i]) for i in range(n)]
    cis: list[list[float]] = []          # rows of the running Cholesky factor, one per pick
    selected: list[int] = []
    for _ in range(k):
        j = max(range(n), key=lambda i: d2[i])
        if d2[j] <= 1e-12:
            break
        selected.append(j)
        d_j = d2[j] ** 0.5
        # e = (L[j, :] - c @ c[j]) / d_j   (the c @ c[j] term is empty on the first pick)
        e = [
            (float(kernel[j][i]) - sum(ci[i] * ci[j] for ci in cis)) / d_j
            for i in range(n)
        ]
        cis.append(e)
        for i in range(n):
            d2[i] -= e[i] * e[i]
        d2[j] = float("-inf")
    return selected


def semdiv_select(
    embeddings: Sequence[Sequence[float]],
    similarities: Sequence[float],
    k: int,
    alpha: float = DEFAULT_ALPHA,
) -> list[int]:
    """The full SemDiv objective, pure: L2-normalised ``embeddings`` + query cosines -> K
    indices in TEMPORAL order. Free of torch/numpy so it unit-tests on any machine."""
    n = len(embeddings)
    if n != len(similarities):
        raise ValueError(
            f"embeddings ({n}) and similarities ({len(similarities)}) must have equal length"
        )
    q = quality_scores(similarities, alpha=alpha)
    # L = diag(q) E E^T diag(q); E rows are already L2-normalised by the scorer.
    kernel = [
        [q[i] * q[j] * sum(a * b for a, b in zip(embeddings[i], embeddings[j])) for j in range(n)]
        for i in range(n)
    ]
    return sorted(dpp_greedy(kernel, k))


@register("select", "semdiv")
class SemDivSelector(Selector):
    """SemDiv as a plain frame selector: ``budget.frame_count`` frames from the pool."""

    def __init__(
        self,
        scorer_factory: Callable[[], object] = default_clip_scorer,
        alpha: float = DEFAULT_ALPHA,
    ) -> None:
        self._scorer_factory = scorer_factory
        self._scorer: Optional[object] = None
        self.alpha = alpha

    def _get_scorer(self):
        if self._scorer is None:
            self._scorer = self._scorer_factory()
        return self._scorer

    def select(
        self,
        pool: FramePool,
        query: str,
        budget: Budget,
        context: Optional[RoundContext] = None,
    ) -> Selection:
        n = 0 if pool.frames is None else len(pool.frames)
        k = n if budget.frame_count is None else min(budget.frame_count, n)
        if k <= 0 or n == 0:
            return Selection(indices=[], signal="query_similarity")

        import numpy as np

        scorer = self._get_scorer()
        # Query-frame cosines come from the scorer's own __call__. Do NOT hand-roll the dot
        # from embed_text: its output is (1, D), and zipping that against a row silently
        # yields ARRAYS, not scalars.
        sims = [float(x) for x in scorer(pool, query)]
        emb = np.asarray(scorer.embed_frames(pool.frames), dtype=float)
        embeddings = emb.reshape(emb.shape[0], -1).tolist()    # L2-normalised rows, 2-D
        indices = semdiv_select(embeddings, sims, k, alpha=self.alpha)

        res = None if budget.resolution is None else [budget.resolution] * len(indices)
        return Selection(
            indices=indices,
            per_frame_resolution=res,
            signal="query_similarity",
        )


@register("select", "lohi-semdiv")
class LoHiSemDivSelector(Selector):
    """The LoHi dual-stream allocation: keep EVERY pooled frame for Lo-V, and mark the K
    SemDiv picks as the Hi-I frames.

    ``Selection.indices`` is the whole Lo-V frame set (the model still sees the full
    timeline) and ``Selection.hi_indices`` is the K-subset that additionally rides the image
    pathway. ``budget.hi_i_count`` is K; ``budget.frame_count`` stays the Lo-V frame count,
    as in the paper's ``(N, r_l, K, r_h)`` notation.
    """

    def __init__(
        self,
        scorer_factory: Callable[[], object] = default_clip_scorer,
        alpha: float = DEFAULT_ALPHA,
    ) -> None:
        self._inner = SemDivSelector(scorer_factory=scorer_factory, alpha=alpha)

    def select(
        self,
        pool: FramePool,
        query: str,
        budget: Budget,
        context: Optional[RoundContext] = None,
    ) -> Selection:
        n = 0 if pool.frames is None else len(pool.frames)
        if budget.hi_i_count is None:
            raise ValueError(
                "lohi-semdiv needs Budget.hi_i_count (K, the number of high-resolution "
                "frames); got None. Use a presentation='lohi' setup (runners/setups.py)."
            )
        lo_indices = list(range(n))
        if n == 0:
            return Selection(indices=[], hi_indices=[], signal="query_similarity")
        # Score the LOW-RESOLUTION frames, not the native ones: SemDiv's overhead is one CLIP
        # pass over the frames the Lo-V stream already uses. The pool arrives native (the
        # high-resolution stream needs it), and CLIP resizes every input to 224x224 anyway, so
        # scoring native frames would only add resize cost.
        scoring_pool = pool
        if budget.resolution is not None and hasattr(pool.frames, "shape"):
            from dataclasses import replace as _replace

            from ..decode.decoder import resize_target
            from ..encode.encoder import _resize_frames

            # Selectors are model-agnostic, so the default patch factor is used here. It only
            # sets the intermediate size CLIP then resizes; it is NOT the grid the encoder
            # renders (that one is computed per model).
            lo_h, lo_w = resize_target(
                int(pool.frames.shape[1]), int(pool.frames.shape[2]), budget.resolution,
            )
            scoring_pool = _replace(pool, frames=_resize_frames(pool.frames, lo_h, lo_w))
        # K is chosen over the SAME pool the Lo-V uses, so the Hi-I frames are a subset of it
        # and each frame is decoded once and reused at both scales.
        hi = self._inner.select(
            scoring_pool, query, Budget(frame_count=budget.hi_i_count), context
        )
        res = None if budget.resolution is None else [budget.resolution] * n
        return Selection(
            indices=lo_indices,
            per_frame_resolution=res,
            hi_indices=list(hi.indices),
            signal="query_similarity",
        )
