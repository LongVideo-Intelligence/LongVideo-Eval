"""Tests for the SemDiv selectors (query relevance x visual diversity).

The objective is pure Python, so it is tested without torch or CLIP. Covered: the min-max and
power-sharpened quality map, the greedy determinant maximisation (against a brute-force
search), the diversity behaviour that separates SemDiv from top-k relevance, and the two
selector classes driven by a fake scorer.
"""
from __future__ import annotations

import itertools

import pytest

from longvideo_eval.frontend.select.semdiv import (
    DEFAULT_ALPHA,
    LoHiSemDivSelector,
    SemDivSelector,
    dpp_greedy,
    quality_scores,
    semdiv_select,
)
from longvideo_eval.interfaces import Budget, FramePool
from longvideo_eval.testing import FakeClipScorer


def _det(m):
    n = len(m)
    if n == 0:
        return 1.0
    if n == 1:
        return m[0][0]
    total = 0.0
    for c in range(n):
        minor = [[row[j] for j in range(n) if j != c] for row in m[1:]]
        total += ((-1) ** c) * m[0][c] * _det(minor)
    return total


# --------------------------------------------------------------------------- #
# quality map
# --------------------------------------------------------------------------- #
def test_quality_is_minmax_then_power_sharpened():
    q = quality_scores([0.0, 0.5, 1.0], alpha=2.0)
    assert q == pytest.approx([0.0, 0.25, 1.0])          # (0, .5, 1) ** 2
    # Min-max first: shifting and scaling the raw cosines changes nothing.
    assert quality_scores([0.2, 0.3, 0.4], alpha=2.0) == pytest.approx([0.0, 0.25, 1.0])


def test_quality_default_alpha_is_five():
    # Pinned so that changing the sharpening exponent is a deliberate act.
    assert DEFAULT_ALPHA == 5.0
    assert quality_scores([0.0, 1.0])[0] == 0.0
    assert quality_scores([0.0, 0.5, 1.0])[1] == pytest.approx(0.5 ** 5.0)


def test_quality_degenerate_pool_falls_back_to_pure_diversity():
    # No relevance contrast: every quality is 1.0 and the kernel reduces to E E^T.
    assert quality_scores([0.3, 0.3, 0.3]) == [1.0, 1.0, 1.0]
    assert quality_scores([0.7]) == [1.0]
    assert quality_scores([]) == []


def test_quality_rejects_non_positive_alpha():
    with pytest.raises(ValueError, match="alpha must be positive"):
        quality_scores([0.1, 0.9], alpha=0.0)


# --------------------------------------------------------------------------- #
# greedy determinant maximisation
# --------------------------------------------------------------------------- #
def test_dpp_greedy_matches_brute_force_max_det_on_a_small_kernel():
    emb = [[1.0, 0.0], [0.98, 0.199], [0.0, 1.0], [0.7, 0.714]]
    q = [1.0, 0.95, 0.9, 0.85]
    L = [[q[i] * q[j] * sum(a * b for a, b in zip(emb[i], emb[j])) for j in range(4)]
         for i in range(4)]
    got = sorted(dpp_greedy(L, 2))
    best = max(itertools.combinations(range(4), 2),
               key=lambda s: _det([[L[i][j] for j in s] for i in s]))
    assert got == sorted(best)


def test_dpp_greedy_returns_selection_order_most_valuable_first():
    # Diagonal kernel: no interaction, so the picks come out by decreasing diagonal.
    L = [[1.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 2.0]]
    assert dpp_greedy(L, 3) == [1, 2, 0]


def test_dpp_greedy_prefers_diverse_over_merely_relevant():
    """Two near-duplicate top-relevance frames against one distinct mid-relevance frame: a
    top-k relevance selector takes both duplicates, SemDiv must take the distinct one."""
    emb = [[1.0, 0.0], [0.999, 0.0447], [0.0, 1.0], [-1.0, 0.0]]
    sims = [1.0, 0.99, 0.8, 0.0]
    picked = semdiv_select(emb, sims, 2)
    assert picked == [0, 2], "the orthogonal frame must beat the near-duplicate"


def test_minmax_floor_makes_the_least_relevant_frame_unselectable():
    """A consequence of the min-max quality map: the pool minimum normalises to 0, so its
    quality is 0, its kernel diagonal is 0, and the greedy step can never pick it."""
    emb = [[1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]
    sims = [0.9, 0.5, 0.1]                       # frame 2 is the pool minimum
    assert quality_scores(sims)[2] == 0.0
    assert 2 not in semdiv_select(emb, sims, 2)
    assert 2 not in semdiv_select(emb, sims, 3)  # even when every slot is available


def test_semdiv_select_returns_temporal_order_and_respects_k():
    emb = [[1.0, 0.0], [0.0, 1.0], [0.7, 0.714], [-1.0, 0.0]]
    sims = [0.2, 0.9, 0.5, 0.7]
    for k in (0, 1, 2, 3, 4, 99):
        picked = semdiv_select(emb, sims, k)
        assert picked == sorted(picked)                  # temporal order
        assert len(picked) <= min(max(k, 0), 4)
        assert len(set(picked)) == len(picked)           # no frame selected twice
    assert semdiv_select(emb, sims, 1) == [1]            # the most relevant frame goes first


def test_dpp_greedy_stops_when_no_marginal_gain_remains():
    """An all-identical pool has rank 1: after the first pick every marginal gain is ~0, so
    the selection is cut short instead of being padded with arbitrary frames."""
    emb = [[1.0, 0.0]] * 5
    assert len(semdiv_select(emb, [0.5] * 5, 4)) == 1


def test_semdiv_rejects_mismatched_inputs():
    with pytest.raises(ValueError, match="must have equal length"):
        semdiv_select([[1.0, 0.0], [0.0, 1.0]], [0.5], 1)


def test_dpp_greedy_edge_cases():
    assert dpp_greedy([], 3) == []
    assert dpp_greedy([[1.0]], 0) == []
    assert dpp_greedy([[1.0]], -2) == []
    assert dpp_greedy([[1.0]], 5) == [0]


# --------------------------------------------------------------------------- #
# selectors (fake scorer; these two need numpy, which the selector uses internally)
# --------------------------------------------------------------------------- #
def _list_pool(n):
    return FramePool(video_id="v", frames=list(range(n)),
                     timestamps=[i / 2 for i in range(n)], fps=2.0)


class _DistinctFramesScorer(FakeClipScorer):
    """Query cosines from the package fake, but frame embeddings that are all distinct
    directions: frame i is mostly axis i with a little of its neighbour, so nearby frames are
    similar and the pool has full rank (K picks are always available)."""

    def embed_frames(self, frames):
        n = len(frames)
        rows = []
        for i in range(n):
            row = [0.0] * n
            row[i] = 1.0
            row[(i + 1) % n] = 0.3
            norm = sum(x * x for x in row) ** 0.5
            rows.append([x / norm for x in row])
        return rows


def test_semdiv_selector_picks_frame_count_frames_in_temporal_order():
    pytest.importorskip("numpy")
    sel = SemDivSelector(scorer_factory=_DistinctFramesScorer)
    out = sel.select(_list_pool(16), "a question", Budget(frame_count=4, resolution=0.5))
    assert len(out.indices) == 4
    assert list(out.indices) == sorted(set(out.indices))
    assert all(0 <= i < 16 for i in out.indices)
    assert out.per_frame_resolution == [0.5] * 4
    assert out.signal == "query_similarity"
    assert out.hi_indices is None                       # a plain selector: single stream
    assert out.cost.wall_seconds == 0.0                 # timing is metered by the orchestrator
    # Same inputs, same picks.
    again = SemDivSelector(scorer_factory=_DistinctFramesScorer).select(
        _list_pool(16), "a question", Budget(frame_count=4, resolution=0.5))
    assert list(again.indices) == list(out.indices)


def test_semdiv_selector_edge_budgets():
    pytest.importorskip("numpy")
    calls = []

    class _CountingScorer(FakeClipScorer):
        def __init__(self):
            calls.append(1)

    sel = SemDivSelector(scorer_factory=_CountingScorer)
    assert sel.select(_list_pool(8), "q", Budget(frame_count=0)).indices == []
    assert sel.select(_list_pool(0), "q", Budget(frame_count=4)).indices == []
    assert calls == []                                  # the scorer is not built when unused
    out = sel.select(_list_pool(8), "q", Budget(frame_count=3))
    assert out.per_frame_resolution is None             # no resolution in, none out
    sel.select(_list_pool(8), "q", Budget(frame_count=3))
    assert calls == [1]                                 # built lazily, once, then cached


def test_lohi_semdiv_keeps_every_frame_for_the_video_stream():
    pytest.importorskip("numpy")
    sel = LoHiSemDivSelector(scorer_factory=_DistinctFramesScorer)
    out = sel.select(_list_pool(32), "what happens?",
                     Budget(frame_count=32, resolution=0.25, hi_i_count=4))
    assert list(out.indices) == list(range(32))         # the whole timeline stays
    assert out.per_frame_resolution == [0.25] * 32
    assert len(out.hi_indices) == 4
    assert list(out.hi_indices) == sorted(set(out.hi_indices))
    assert set(out.hi_indices) <= set(out.indices)      # high-res frames are a subset
    assert out.signal == "query_similarity"


def test_lohi_semdiv_runs_on_the_package_fake_scorer():
    # The scorer a dry run injects: the selection is valid, though it may return fewer than
    # K frames because that fake's embeddings span only two dimensions.
    pytest.importorskip("numpy")
    out = LoHiSemDivSelector(scorer_factory=FakeClipScorer).select(
        _list_pool(32), "what happens?", Budget(frame_count=32, resolution=0.25, hi_i_count=4))
    assert list(out.indices) == list(range(32))
    assert 1 <= len(out.hi_indices) <= 4
    assert list(out.hi_indices) == sorted(set(out.hi_indices))
    assert set(out.hi_indices) <= set(out.indices)


def test_lohi_semdiv_requires_hi_i_count():
    with pytest.raises(ValueError, match="hi_i_count"):
        LoHiSemDivSelector(scorer_factory=FakeClipScorer).select(
            _list_pool(8), "q", Budget(frame_count=8, resolution=0.25))


def test_lohi_semdiv_empty_pool():
    out = LoHiSemDivSelector(scorer_factory=FakeClipScorer).select(
        _list_pool(0), "q", Budget(hi_i_count=4))
    assert list(out.indices) == [] and list(out.hi_indices) == []


def test_lohi_semdiv_goes_through_the_scorer_call_for_the_query_cosines():
    """A CLIP text embedding has shape (1, D). Building the cosines by zipping that against a
    frame row would give arrays instead of numbers and break the min/max of the quality map,
    so the selector must take the cosines from the scorer's own ``__call__``."""
    np = pytest.importorskip("numpy")
    pytest.importorskip("PIL")                  # the frames are resized before scoring

    class _Scorer:
        def __call__(self, pool, query):
            return [0.1, 0.9, 0.5, 0.7]

        def embed_frames(self, frames):
            return np.eye(4, 8)                 # (N, D)

        def embed_text(self, query):
            return np.zeros((1, 8))             # (1, D)

    pool = FramePool(video_id="v", frames=np.zeros((4, 8, 8, 3), dtype=np.uint8),
                     timestamps=[0.0, 1.0, 2.0, 3.0], fps=1.0)
    sel = LoHiSemDivSelector(scorer_factory=_Scorer)
    out = sel.select(pool, "q", Budget(frame_count=4, resolution=0.25, hi_i_count=2))
    assert list(out.indices) == [0, 1, 2, 3]                 # the video stream keeps all frames
    # Orthogonal embeddings: the two most relevant frames win.
    assert list(out.hi_indices) == [1, 3]


def test_lohi_semdiv_scores_the_low_resolution_frames():
    """The pool arrives at native size. Scoring uses frames resized to the video stream's
    scale, so the selection overhead is one pass over frames that stream already needs."""
    np = pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    from longvideo_eval.frontend.decode.decoder import resize_target

    shapes = []

    class _Scorer:
        def __call__(self, pool, query):
            shapes.append(("call", tuple(np.asarray(pool.frames).shape)))
            return [0.1, 0.9, 0.5, 0.7]

        def embed_frames(self, frames):
            shapes.append(("embed", tuple(np.asarray(frames).shape)))
            return np.eye(4, 8)

    native = np.zeros((4, 256, 512, 3), dtype=np.uint8)
    pool = FramePool(video_id="v", frames=native, timestamps=[0.0, 1.0, 2.0, 3.0], fps=1.0)
    LoHiSemDivSelector(scorer_factory=_Scorer).select(
        pool, "q", Budget(frame_count=4, resolution=0.25, hi_i_count=2))
    lo_h, lo_w = resize_target(256, 512, 0.25)
    assert (lo_h, lo_w) == (64, 128)
    assert shapes == [("call", (4, 64, 128, 3)), ("embed", (4, 64, 128, 3))]
    assert pool.frames is native and native.shape == (4, 256, 512, 3)   # the pool is untouched


def test_selectors_registered_under_stable_ids():
    from longvideo_eval import _registry
    import longvideo_eval.models.build  # noqa: F401  importing it registers the components

    assert _registry.get("select", "semdiv") is SemDivSelector
    assert _registry.get("select", "lohi-semdiv") is LoHiSemDivSelector
