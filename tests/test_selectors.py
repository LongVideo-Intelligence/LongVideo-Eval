"""Tests for the frame selectors: LoHi-Uniform, CLIP-TopK and AKS.

LoHi-Uniform keeps every pooled frame for the low-resolution video stream and marks K of
them, at regular intervals, for the high-resolution image stream. It looks at no pixels and
no question.

CLIP-TopK and AKS are query-aware keyframe selectors: they score every frame of the decoded
pool against the question and keep ``frame_count`` of them. Both take an injectable scorer
(``Callable[[FramePool, str], Sequence[float]]``), so their selection logic is tested here
with a fixed score list: no torch, no CLIP weights. AKS is also checked against a reference
written separately in this file from the algorithm's description.

Everything runs on placeholder frames.
"""
from __future__ import annotations

import math
import random

import pytest

from longvideo_eval.frontend.select.aks import (
    AKS_ALL_DEPTH,
    AKS_T1,
    AKS_T2,
    AKSSelector,
    _is_peaked,
    _leaves,
    _normalize,
    _top,
    adaptive_keyframe_indices,
    effective_all_depth,
)
from longvideo_eval.frontend.select.cliptopk import ClipTopKSelector, topk_by_score
from longvideo_eval.frontend.select.uniform import LoHiUniformSelector, UniformSelector
from longvideo_eval.interfaces import Budget, FramePool


def _pool(n: int) -> FramePool:
    """A pool of n placeholder frames (the selectors never look at pixels; a scorer would)."""
    return FramePool(video_id="v", frames=list(range(n)), timestamps=[i / 2 for i in range(n)],
                     fps=2.0)


def _select(n, k, resolution=0.25, query="q"):
    return LoHiUniformSelector().select(
        _pool(n), query, Budget(frame_count=n, resolution=resolution, hi_i_count=k))


def test_lohi_uniform_keeps_every_frame_for_the_video_stream():
    out = _select(128, 8)
    assert list(out.indices) == list(range(128))
    assert out.per_frame_resolution == [0.25] * 128


def test_lohi_uniform_spaces_the_high_resolution_frames_evenly():
    out = _select(128, 8)
    # Regular intervals over [0, 127], truncated: int(127 * i / 7).
    assert list(out.hi_indices) == [0, 18, 36, 54, 72, 90, 108, 127]
    assert list(_select(128, 4).hi_indices) == [0, 42, 84, 127]
    assert list(_select(9, 3).hi_indices) == [0, 4, 8]


def test_lohi_uniform_matches_the_decoder_sampling_rule():
    # The same linspace-and-truncate rule the decoder uses to sample frames.
    from longvideo_eval.frontend.decode.decoder import uniform_decode_indices

    for n, k in ((128, 8), (128, 4), (100, 7), (64, 16), (5, 5), (17, 2)):
        assert list(_select(n, k).hi_indices) == uniform_decode_indices(n, k), (n, k)


def test_lohi_uniform_hi_indices_are_a_sorted_unique_subset():
    for n, k in ((128, 8), (10, 3), (7, 7), (33, 5)):
        out = _select(n, k)
        hi = list(out.hi_indices)
        assert hi == sorted(set(hi))
        assert set(hi) <= set(out.indices)
        assert len(hi) == k
        assert hi[0] == 0 and hi[-1] == n - 1          # both ends of the timeline are covered


def test_lohi_uniform_single_high_resolution_frame_is_the_middle_one():
    assert list(_select(128, 1).hi_indices) == [64]
    assert list(_select(1, 1).hi_indices) == [0]


def test_lohi_uniform_clamps_k_to_the_pool_and_allows_zero():
    assert list(_select(4, 99).hi_indices) == [0, 1, 2, 3]
    out = _select(16, 0)
    assert list(out.hi_indices) == []
    assert list(out.indices) == list(range(16))        # the video stream is unaffected


def test_lohi_uniform_requires_hi_i_count():
    with pytest.raises(ValueError, match="hi_i_count"):
        LoHiUniformSelector().select(_pool(8), "q", Budget(frame_count=8, resolution=0.25))


def test_lohi_uniform_empty_pool():
    out = LoHiUniformSelector().select(_pool(0), "q", Budget(hi_i_count=4))
    assert list(out.indices) == [] and list(out.hi_indices) == []


def test_lohi_uniform_without_resolution_leaves_per_frame_none():
    out = _select(8, 2, resolution=None)
    assert out.per_frame_resolution is None
    assert list(out.hi_indices) == [0, 7]


def test_lohi_uniform_ignores_the_question_and_reports_no_cost():
    a = _select(64, 8, query="what is the man doing?")
    b = _select(64, 8, query="a completely different question")
    assert list(a.hi_indices) == list(b.hi_indices)
    assert a.signal == "none"
    assert a.cost.wall_seconds == 0.0 and a.cost.frames_decoded == 0


def test_lohi_uniform_ignores_frame_count_for_the_video_stream():
    # The video stream is the whole pool; frame_count does not subsample it here.
    out = LoHiUniformSelector().select(
        _pool(32), "q", Budget(frame_count=8, resolution=0.25, hi_i_count=4))
    assert list(out.indices) == list(range(32))


def test_plain_uniform_selector_is_single_stream():
    out = UniformSelector().select(_pool(32), "q", Budget(frame_count=8, hi_i_count=4))
    assert out.hi_indices is None
    assert len(out.indices) == 8


def test_lohi_uniform_registered_and_assembled_by_build():
    from longvideo_eval import _registry
    from longvideo_eval.models.build import build_pipeline

    assert _registry.get("select", "lohi-uniform") is LoHiUniformSelector
    orch = build_pipeline("qwen3-vl-4b", "lohi-uniform", dry_run=True)
    assert isinstance(orch.selector, LoHiUniformSelector)


# =========================================================================== #
# Query-aware keyframe selection: CLIP-TopK and AKS
# =========================================================================== #
def _fixed_scorer(scores):
    """A scorer that returns a fixed score list whatever the question, and counts its calls."""
    calls = {"n": 0}

    def scorer(pool, query):
        calls["n"] += 1
        return list(scores)

    scorer.calls = calls
    return scorer


# --------------------------------------------------------------------------- #
# CLIP-TopK: the top-k helper
# --------------------------------------------------------------------------- #
def test_topk_by_score_picks_highest_and_sorts_temporally():
    scores = [0.1, 0.9, 0.3, 0.8, 0.2]
    # The two highest scores are at 1 (0.9) and 3 (0.8); returned in ascending index order.
    assert topk_by_score(scores, 2) == [1, 3]
    # The order is temporal even when the best frame comes last.
    assert topk_by_score([0.2, 0.8, 0.1, 0.9], 2) == [1, 3]


def test_topk_by_score_clamps_and_edges():
    scores = [0.5, 0.1, 0.9]
    assert topk_by_score(scores, 0) == []            # nothing asked for
    assert topk_by_score(scores, -2) == []           # a negative k is clamped to zero
    assert topk_by_score(scores, 5) == [0, 1, 2]     # more than the pool: every frame
    assert topk_by_score(scores, 3) == [0, 1, 2]     # exactly the pool: every frame
    assert topk_by_score([], 3) == []                # empty pool


def test_topk_by_score_tie_breaks_by_smaller_index_deterministically():
    assert topk_by_score([0.5, 0.5, 0.5, 0.5], 2) == [0, 1]
    assert topk_by_score([0.1, 0.7, 0.7, 0.7, 0.9], 2) == [1, 4]


def test_topk_by_score_matches_a_sort_on_random_scores():
    rng = random.Random(0)
    for _ in range(50):
        n = rng.randint(1, 40)
        k = rng.randint(0, n + 3)
        scores = [rng.random() for _ in range(n)]
        out = topk_by_score(scores, k)
        assert out == sorted(set(out)) and len(out) == min(k, n)
        # No dropped frame scores higher than a kept one.
        dropped = [scores[i] for i in range(n) if i not in set(out)]
        if out and dropped:
            assert min(scores[i] for i in out) >= max(dropped)


# --------------------------------------------------------------------------- #
# CLIP-TopK selector
# --------------------------------------------------------------------------- #
def test_cliptopk_selects_topk_frames_temporally():
    scorer = _fixed_scorer([0.1, 0.9, 0.3, 0.8, 0.2, 0.95])
    sel = ClipTopKSelector(scorer=scorer).select(_pool(6), "q", Budget(frame_count=3))
    assert sel.indices == [1, 3, 5]           # the three highest (0.95, 0.9, 0.8), in time order
    assert sel.signal == "retrieval"
    assert sel.hi_indices is None             # a single-stream selection
    assert scorer.calls["n"] == 1


def test_cliptopk_passes_the_pool_and_the_question_to_the_scorer():
    seen = {}

    def scorer(pool, query):
        seen.update(pool=pool, query=query)
        return [0.0] * len(pool.timestamps)

    pool = _pool(5)
    ClipTopKSelector(scorer=scorer).select(pool, "what is the man doing?", Budget(frame_count=2))
    assert seen == {"pool": pool, "query": "what is the man doing?"}


def test_cliptopk_carries_uniform_resolution_when_set():
    scorer = _fixed_scorer([0.1, 0.9, 0.3, 0.8])
    sel = ClipTopKSelector(scorer=scorer).select(_pool(4), "q",
                                                 Budget(frame_count=2, resolution=1.0))
    assert sel.indices == [1, 3]
    assert sel.per_frame_resolution == [1.0, 1.0]     # one scale per kept frame


def test_cliptopk_native_resolution_leaves_per_frame_none():
    scorer = _fixed_scorer([0.1, 0.9, 0.3, 0.8])
    sel = ClipTopKSelector(scorer=scorer).select(_pool(4), "q", Budget(frame_count=2))
    assert sel.per_frame_resolution is None


def test_cliptopk_k_ge_pool_keeps_all_and_never_scores_when_zero():
    scorer = _fixed_scorer([0.1, 0.2, 0.3])
    sel = ClipTopKSelector(scorer=scorer).select(_pool(3), "q", Budget(frame_count=10))
    assert sel.indices == [0, 1, 2]                   # a budget above the pool keeps the pool
    assert scorer.calls["n"] == 1
    # A zero budget selects nothing and does not score at all.
    zsel = ClipTopKSelector(scorer=scorer).select(_pool(3), "q", Budget(frame_count=0))
    assert zsel.indices == []
    assert scorer.calls["n"] == 1


def test_cliptopk_empty_pool_selects_nothing_without_scoring():
    scorer = _fixed_scorer([])
    sel = ClipTopKSelector(scorer=scorer).select(_pool(0), "q", Budget(frame_count=4))
    assert sel.indices == [] and scorer.calls["n"] == 0


def test_cliptopk_frame_count_none_selects_all():
    scorer = _fixed_scorer([0.4, 0.1, 0.9])
    sel = ClipTopKSelector(scorer=scorer).select(_pool(3), "q", Budget(frame_count=None))
    assert sel.indices == [0, 1, 2]


def test_cliptopk_rejects_scorer_pool_length_mismatch():
    scorer = _fixed_scorer([0.1, 0.2])                # 2 scores for a 4-frame pool
    with pytest.raises(ValueError, match="cover the whole decoded pool"):
        ClipTopKSelector(scorer=scorer).select(_pool(4), "q", Budget(frame_count=2))


def test_cliptopk_cost_is_empty_increment():
    scorer = _fixed_scorer([0.1, 0.9, 0.3])
    sel = ClipTopKSelector(scorer=scorer).select(_pool(3), "q", Budget(frame_count=2))
    # The selector reports no cost of its own: the select stage is timed by the orchestrator
    # and the decode is charged by the decoder, so anything written here would count twice.
    assert sel.cost.wall_seconds == 0.0 and sel.cost.frames_decoded == 0


def test_cliptopk_can_cluster_in_one_part_of_the_video():
    # Relevance only: when the best-scoring frames sit together, so does the selection.
    scores = [0.0] * 64
    for i in range(20, 28):
        scores[i] = 1.0
    sel = ClipTopKSelector(scorer=_fixed_scorer(scores)).select(
        _pool(64), "q", Budget(frame_count=8))
    assert sel.indices == list(range(20, 28))


# --------------------------------------------------------------------------- #
# AKS: helpers
# --------------------------------------------------------------------------- #
def test_aks_defaults():
    assert (AKS_T1, AKS_T2, AKS_ALL_DEPTH) == (0.8, -100, 5)
    sel = AKSSelector(scorer=_fixed_scorer([]))
    assert (sel.t1, sel.t2, sel.all_depth) == (0.8, -100, 5)


def test_normalize_min_max_and_flat_guard():
    assert _normalize([0.0, 5.0, 10.0]) == [0.0, 0.5, 1.0]
    assert _normalize([-2.0, 2.0]) == [0.0, 1.0]
    assert _normalize([3.0, 3.0, 3.0]) == [0.0, 0.0, 0.0]   # flat: all zeros, no division by 0


def test_top_returns_highest_first_with_earlier_position_on_a_tie():
    assert _top([0.1, 0.9, 0.3, 0.8], 2) == [1, 3]
    assert _top([0.5, 0.5, 0.5], 2) == [0, 1]
    assert _top([0.1, 0.9], 5) == [1, 0]                    # k above the length: everything
    assert _top([0.1, 0.9], 0) == []


def test_effective_all_depth_caps_at_log2_num():
    assert effective_all_depth(16, 5) == 4      # floor(log2(16)) == 4 < 5
    assert effective_all_depth(32, 5) == 5      # floor(log2(32)) == 5: the default is unchanged
    assert effective_all_depth(64, 5) == 5      # never above the requested depth
    assert effective_all_depth(8, 5) == 3
    assert effective_all_depth(12, 5) == 3      # not a power of two: floor
    assert effective_all_depth(16, 2) == 2      # a smaller requested depth is kept
    assert effective_all_depth(1, 5) == 0
    assert effective_all_depth(0, 5) == 0
    assert effective_all_depth(16) == 4         # the default depth is 5


def test_effective_all_depth_leaves_every_deepest_segment_one_frame():
    for num in range(1, 200):
        depth = effective_all_depth(num, 12)
        assert int(num / 2 ** depth) >= 1
        assert int(num / 2 ** (depth + 1)) == 0             # one level deeper gets nothing


def test_is_peaked_compares_the_top_scores_with_the_segment_mean():
    # Four frames at 1.0 and twelve at 0.0: the top-4 mean is 1.0, the segment mean 0.25,
    # a margin of 0.75.
    peaked = [0.0] * 12 + [1.0] * 4
    assert _is_peaked(peaked, 4, t1=0.7, t2=-100)
    assert not _is_peaked(peaked, 4, t1=0.8, t2=-100)
    # A flat segment has no margin at all.
    assert not _is_peaked([0.5] * 16, 4, t1=0.0, t2=-100)
    # The standard-deviation test can veto: this segment's deviation is about 0.43.
    assert _is_peaked(peaked, 4, t1=0.7, t2=0.4)
    assert not _is_peaked(peaked, 4, t1=0.7, t2=0.5)
    # A segment no longer than the budget has its mean as its top mean: never peaked.
    assert not _is_peaked([0.0, 1.0], 4, t1=0.0, t2=-100)


def test_leaves_keep_a_peaked_segment_whole_and_split_a_flat_one():
    peaked = [0.0] * 12 + [1.0] * 4
    # Peaked at the root: one segment, kept whole at depth 0.
    assert list(_leaves(peaked, 0, 0, 4, 0.7, -100, 5)) == [(0, peaked, 0)]
    # Flat: halved until the depth limit, with each half keeping its place on the timeline.
    flat = [0.5] * 16
    assert list(_leaves(flat, 0, 0, 4, 0.7, -100, 1)) == [(0, [0.5] * 8, 1), (8, [0.5] * 8, 1)]
    deeper = list(_leaves(flat, 0, 0, 4, 0.7, -100, 2))
    assert [(start, len(seg), depth) for start, seg, depth in deeper] == [
        (0, 4, 2), (4, 4, 2), (8, 4, 2), (12, 4, 2)]
    # Depth 0 keeps the whole timeline whatever the scores.
    assert list(_leaves(flat, 0, 0, 4, 0.7, -100, 0)) == [(0, flat, 0)]


def test_leaves_split_only_the_half_that_is_not_peaked():
    # First half flat, second half peaked (two frames at 1.0 among eight).
    scores = [0.5] * 8 + [0.0] * 6 + [1.0] * 2
    out = [(start, len(seg), depth) for start, seg, depth in
           _leaves(scores, 0, 0, 2, 0.7, -100, 3)]
    assert out == [(0, 2, 3), (2, 2, 3), (4, 2, 3), (6, 2, 3), (8, 8, 1)]


def test_leaves_partition_the_timeline():
    rng = random.Random(4)
    for _ in range(30):
        n = rng.randint(1, 70)
        scores = _normalize([rng.random() ** 6 for _ in range(n)])
        covered = []
        for start, seg, depth in _leaves(scores, 0, 0, 4, 0.5, -100, rng.randint(0, 6)):
            assert seg == scores[start:start + len(seg)]
            covered.extend(range(start, start + len(seg)))
        assert covered == list(range(n))                    # in order, no gap, no overlap


# --------------------------------------------------------------------------- #
# AKS: adaptive_keyframe_indices
# --------------------------------------------------------------------------- #
def test_aks_small_pool_returns_all_frames():
    # Fewer frames than the budget: keep everything.
    assert adaptive_keyframe_indices([0.2, 0.9, 0.1], num=8) == [0, 1, 2]
    assert adaptive_keyframe_indices([0.4], num=2) == [0]


def test_aks_zero_or_negative_budget_is_empty():
    assert adaptive_keyframe_indices([0.1, 0.2, 0.3, 0.4], num=0) == []
    assert adaptive_keyframe_indices([0.1, 0.2, 0.3, 0.4], num=-3) == []
    assert adaptive_keyframe_indices([], num=4) == []


def test_aks_strong_root_peak_behaves_like_topk():
    # A globally peaked video: four frames at 1.0 and twenty-eight at 0. The margin at the
    # root is 0.875 > 0.8, so the whole timeline is kept and the four peak frames are taken.
    n, num = 32, 4
    peaks = {5, 12, 20, 27}
    scores = [1.0 if i in peaks else 0.0 for i in range(n)]
    out = adaptive_keyframe_indices(scores, num=num, all_depth=effective_all_depth(num))
    assert out == sorted(peaks)
    assert out == topk_by_score(scores, num)


def test_aks_splits_when_not_globally_peaked_and_covers_timeline():
    # A ramp is not peaked anywhere, so the timeline is halved down to the depth limit and
    # every one of the sixteen segments gives its best frame (its last one).
    n, num = 64, 16
    scores = [i / (n - 1) for i in range(n)]
    out = adaptive_keyframe_indices(scores, num=num, all_depth=effective_all_depth(num))
    assert out == list(range(3, 64, 4))
    assert out == sorted(set(out))               # ascending, no duplicates
    assert len(out) == num                       # the whole budget is used
    assert min(out) < n // 4 and max(out) > 3 * n // 4      # early and late both covered
    # A relevance-only selection of the same scores takes the last sixteen frames.
    assert topk_by_score(scores, num) == list(range(48, 64))


def test_aks_peaked_scores_concentrate_and_flat_scores_spread():
    n, num = 64, 8
    depth = effective_all_depth(num)

    def bins(indices):
        """How many of the eight equal parts of the timeline hold a selected frame."""
        return len({i // (n // num) for i in indices})

    # Peaked: the eight relevant frames sit together and all of them are selected.
    peaked = [1.0 if 40 <= i < 48 else 0.0 for i in range(n)]
    concentrated = adaptive_keyframe_indices(peaked, num, all_depth=depth)
    assert concentrated == list(range(40, 48))
    assert bins(concentrated) == 1

    # Flat (and nearly flat): one frame from each part of the timeline.
    for scores in ([0.5] * n, [0.5 + 0.001 * (i % 7) for i in range(n)]):
        spread = adaptive_keyframe_indices(scores, num, all_depth=depth)
        assert len(spread) == num and bins(spread) == num


def test_aks_flat_scores_pick_the_first_frame_of_every_segment():
    out = adaptive_keyframe_indices([0.5] * 256, 16, all_depth=4)
    assert out == list(range(0, 256, 16))


def test_aks_uncapped_depth_selects_nothing_with_16_keyframes():
    # With 16 keyframes, a segment at depth 5 is entitled to int(16 / 32) == 0 frames. A video
    # that is peaked nowhere is split all the way down, so the default depth selects nothing.
    flat = [0.5] * 256
    assert adaptive_keyframe_indices(flat, 16, all_depth=5) == []
    assert adaptive_keyframe_indices(flat, 16) == []                    # 5 is the default
    # Capped at floor(log2(16)) == 4, the same video yields the full budget.
    capped = adaptive_keyframe_indices(flat, 16, all_depth=effective_all_depth(16))
    assert len(capped) == 16
    # With 32 keyframes the default depth needs no cap.
    assert effective_all_depth(32) == 5
    assert len(adaptive_keyframe_indices(flat, 32, all_depth=5)) == 32


def test_aks_budget_is_split_evenly_over_the_halves():
    # Not peaked at the root, peaked in neither half: each half of the timeline gets half of
    # the budget, and inside a quarter the best frames win.
    scores = [0.0, 0.3, 0.1, 0.2, 0.9, 0.5, 0.6, 1.0]
    assert adaptive_keyframe_indices(scores, 4, all_depth=1) == [1, 3, 4, 7]
    assert adaptive_keyframe_indices(scores, 4, all_depth=2) == [1, 3, 4, 7]
    assert adaptive_keyframe_indices(scores, 4, all_depth=0) == [4, 5, 6, 7]   # relevance only


def test_aks_integer_division_can_leave_the_budget_slightly_unused():
    # 12 keyframes at depth 3: eight segments of int(12 / 8) == 1 frame each.
    out = adaptive_keyframe_indices([0.5] * 64, 12, all_depth=effective_all_depth(12))
    assert out == list(range(0, 64, 8))
    assert len(out) == 8 <= 12


def test_aks_is_invariant_to_shifting_and_scaling_the_scores():
    rng = random.Random(2)
    scores = [rng.random() ** 4 for _ in range(128)]
    base = adaptive_keyframe_indices(scores, 16, t1=0.5, all_depth=4)
    moved = adaptive_keyframe_indices([2.0 * s + 1.0 for s in scores], 16, t1=0.5, all_depth=4)
    assert moved == base and base


def test_aks_tiny_pool_does_not_crash():
    # Splitting a one-frame segment leaves an empty half, which must simply be skipped.
    scores = [0.9, 0.1, 0.5, 0.2]
    for depth in range(0, 6):
        out = adaptive_keyframe_indices(scores, num=2, all_depth=depth)
        assert out == sorted(set(out)) and all(0 <= i < 4 for i in out)
    assert adaptive_keyframe_indices(scores, num=2, all_depth=effective_all_depth(2)) == [0, 2]
    assert adaptive_keyframe_indices([0.3], num=1, all_depth=effective_all_depth(1)) == [0]


# --------------------------------------------------------------------------- #
# AKS against a reference written separately from the algorithm's description
# --------------------------------------------------------------------------- #
def _aks_reference(scores, num, t1, t2, all_depth):
    """AKS with an explicit work list instead of recursion.

    Scores are min-max normalised (a flat input becomes zeros). A segment is PEAKED when the
    mean of its ``num`` highest scores exceeds its mean by more than ``t1`` and its (population)
    standard deviation exceeds ``t2``. A peaked segment, or one at depth ``all_depth``, is
    final and gives its ``int(num / 2**depth)`` highest-scoring frames; any other segment is
    cut into two halves one level deeper. With fewer frames than ``num`` every frame is kept.

    Returns ``(sorted indices, depths of the final segments)``.
    """
    n = len(scores)
    if num <= 0 or n == 0:
        return [], []
    if n < num:
        return list(range(n)), []
    lo, hi = min(scores), max(scores)
    norm = [0.0 if hi <= lo else (s - lo) / (hi - lo) for s in scores]

    picked, depths = [], []
    work = [(0, n, 0)]                                   # (first frame, end, depth)
    while work:
        a, b, depth = work.pop()
        if a == b:
            continue
        seg = norm[a:b]
        mean = math.fsum(seg) / len(seg)
        std = math.sqrt(math.fsum((s - mean) ** 2 for s in seg) / len(seg))
        best = sorted(seg, reverse=True)[:num]
        peaked = math.fsum(best) / len(best) - mean > t1 and std > t2
        if peaked or depth >= all_depth:
            share = int(num / 2 ** depth)
            ranked = sorted(range(a, b), key=lambda i: norm[i], reverse=True)   # stable
            picked.extend(ranked[:share])
            depths.append(depth)
        else:
            mid = a + (b - a) // 2
            work.append((a, mid, depth + 1))
            work.append((mid, b, depth + 1))
    return sorted(picked), depths


def _score_profiles(rng, n):
    """Score arrays with different shapes: noise, a few sharp peaks, one burst, a noisy ramp."""
    noise = [rng.random() for _ in range(n)]
    sharp = [rng.random() ** 12 for _ in range(n)]
    start = rng.randrange(0, max(1, n - n // 8))
    burst = [(0.8 if start <= i < start + n // 8 else 0.0) + 0.2 * rng.random()
             for i in range(n)]
    ramp = [i / n + 0.05 * rng.random() for i in range(n)]
    return {"noise": noise, "sharp": sharp, "burst": burst, "ramp": ramp}


@pytest.mark.parametrize("num", [1, 2, 4, 7, 8, 12, 16, 32])
@pytest.mark.parametrize("t1", [0.2, 0.5, 0.8])
def test_aks_matches_the_reference_on_random_scores(num, t1):
    rng = random.Random(1000 * num + int(t1 * 10))
    seen_depths = set()
    for n in (num, num + 1, 33, 64, 100, 256):
        if n < num:
            continue
        for name, scores in _score_profiles(rng, n).items():
            for depth in (0, 1, 3, 5, effective_all_depth(num)):
                expected, depths = _aks_reference(scores, num, t1, -100, depth)
                got = adaptive_keyframe_indices(scores, num, t1=t1, all_depth=depth)
                assert got == expected, (name, n, num, t1, depth)
                seen_depths.update(depths)
                # The properties every selection has.
                assert got == sorted(set(got))               # temporal order, no duplicates
                assert len(got) <= num                       # the budget is respected
                assert all(0 <= i < n for i in got)
    if num >= 4:
        # The cases reach both outcomes: a whole timeline kept at the root, and deep splits.
        assert 0 in seen_depths and max(seen_depths) >= 2


@pytest.mark.parametrize("t2", [-100, 0.05, 0.2, 0.4])
def test_aks_matches_the_reference_with_a_standard_deviation_threshold(t2):
    rng = random.Random(77)
    for n in (48, 128):
        for name, scores in _score_profiles(rng, n).items():
            for num in (4, 8, 16):
                depth = effective_all_depth(num)
                expected, _ = _aks_reference(scores, num, 0.3, t2, depth)
                got = adaptive_keyframe_indices(scores, num, t1=0.3, t2=t2, all_depth=depth)
                assert got == expected, (name, n, num, t2)


def test_aks_default_arguments_match_the_reference():
    rng = random.Random(5)
    for name, scores in _score_profiles(rng, 256).items():
        for num in (16, 32, 64):
            expected, _ = _aks_reference(scores, num, 0.8, -100, 5)
            assert adaptive_keyframe_indices(scores, num) == expected, (name, num)


def test_aks_selector_matches_the_reference_at_the_capped_depth():
    rng = random.Random(9)
    for name, scores in _score_profiles(rng, 256).items():
        for num in (8, 16, 32):
            sel = AKSSelector(scorer=_fixed_scorer(scores)).select(
                _pool(256), "q", Budget(frame_count=num))
            expected, _ = _aks_reference(scores, num, 0.8, -100, min(5, int(math.log2(num))))
            assert sel.indices == expected, (name, num)


# --------------------------------------------------------------------------- #
# AKS selector
# --------------------------------------------------------------------------- #
def test_aks_selector_uses_scorer_and_caps_depth_for_16f():
    # A 256-frame pool and a budget of 16 keyframes. A nearly flat score array must still give
    # the full budget, which it does only because the selector caps the depth at 4.
    n = 256
    scores = [0.5 + 0.001 * (i % 7) for i in range(n)]
    scorer = _fixed_scorer(scores)
    sel = AKSSelector(scorer=scorer).select(_pool(n), "q",
                                            Budget(frame_count=16, resolution=1.0))
    assert sel.signal == "similarity"
    assert len(sel.indices) == 16
    assert sel.indices == sorted(set(sel.indices))
    assert {i // 16 for i in sel.indices} == set(range(16))     # one frame per sixteenth
    assert sel.per_frame_resolution == [1.0] * 16
    assert sel.hi_indices is None
    assert scorer.calls["n"] == 1
    # The same scores at the uncapped default depth select nothing.
    assert adaptive_keyframe_indices(scores, 16) == []


def test_aks_selector_honours_its_thresholds():
    # Eight relevant frames among 64: a margin of 0.875 at the root.
    scores = [1.0 if 40 <= i < 48 else 0.0 for i in range(64)]
    budget = Budget(frame_count=8)
    default = AKSSelector(scorer=_fixed_scorer(scores)).select(_pool(64), "q", budget)
    assert default.indices == list(range(40, 48))               # peaked: relevance decides
    strict = AKSSelector(scorer=_fixed_scorer(scores), t1=0.9).select(_pool(64), "q", budget)
    assert {i // 8 for i in strict.indices} == set(range(8))    # not peaked: coverage decides
    shallow = AKSSelector(scorer=_fixed_scorer(scores), t1=0.9, all_depth=1).select(
        _pool(64), "q", budget)
    assert shallow.indices == [0, 1, 2, 3, 40, 41, 42, 43]      # four frames from each half


def test_aks_selector_small_pool_keeps_every_frame():
    scorer = _fixed_scorer([0.3, 0.9, 0.1])
    sel = AKSSelector(scorer=scorer).select(_pool(3), "q", Budget(frame_count=16))
    assert sel.indices == [0, 1, 2]


def test_aks_selector_native_resolution_leaves_per_frame_none():
    scorer = _fixed_scorer([0.9, 0.1, 0.5, 0.2, 0.8, 0.3, 0.7, 0.4])
    sel = AKSSelector(scorer=scorer).select(_pool(8), "q", Budget(frame_count=4))
    assert sel.per_frame_resolution is None
    assert 0 < len(sel.indices) <= 4


def test_aks_selector_zero_budget_skips_scoring():
    scorer = _fixed_scorer([0.1, 0.2, 0.3])
    sel = AKSSelector(scorer=scorer).select(_pool(3), "q", Budget(frame_count=0))
    assert sel.indices == []
    assert scorer.calls["n"] == 0                          # never scored
    empty = AKSSelector(scorer=scorer).select(_pool(0), "q", Budget(frame_count=4))
    assert empty.indices == [] and scorer.calls["n"] == 0


def test_aks_selector_rejects_scorer_length_mismatch():
    scorer = _fixed_scorer([0.1, 0.2, 0.3])                # 3 scores for a 64-frame pool
    with pytest.raises(ValueError, match="cover the whole decoded pool"):
        AKSSelector(scorer=scorer).select(_pool(64), "q", Budget(frame_count=16))


def test_aks_selector_cost_is_empty_increment():
    scorer = _fixed_scorer([0.9, 0.1, 0.5, 0.2, 0.8, 0.3, 0.7, 0.4])
    sel = AKSSelector(scorer=scorer).select(_pool(8), "q", Budget(frame_count=4))
    assert sel.cost.wall_seconds == 0.0 and sel.cost.frames_decoded == 0


# --------------------------------------------------------------------------- #
# Registration and assembly (a dry run never loads CLIP)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("method,select_cls", [("cliptopk", ClipTopKSelector), ("aks", AKSSelector)])
def test_build_pipeline_assembles_keyframe_methods(method, select_cls):
    from longvideo_eval.models.build import build_pipeline
    from longvideo_eval.testing import FakeClipScorer

    orch = build_pipeline("qwen3-vl-4b", method, dry_run=True)
    assert isinstance(orch.selector, select_cls)
    assert isinstance(orch.selector._scorer, FakeClipScorer)    # no CLIP weights in a dry run


def test_keyframe_selectors_registered_under_stable_ids():
    import longvideo_eval.models.build  # noqa: F401  importing it registers the selectors
    from longvideo_eval import _registry

    assert _registry.get("select", "cliptopk") is ClipTopKSelector
    assert _registry.get("select", "aks") is AKSSelector


# --------------------------------------------------------------------------- #
# The keyframe-selection setup
# --------------------------------------------------------------------------- #
def test_keyframe_16f_pool256_setup_fields():
    from longvideo_eval.runners.setups import get_setup

    b = get_setup("keyframe_16f_pool256")
    assert b.frame_count == 16                     # 16 keyframes kept
    assert b.resolution == 1.0                     # at native resolution
    assert b.decode_budget == 256                  # out of a 256-frame decoded pool
    assert b.decode_budget > b.frame_count         # the selector reads more than it keeps
    assert b.presentation == "video" and b.token_budget is None
    # The same token allocation as 16 native frames; only the decode is larger.
    iso = get_setup("native_16f")
    assert (b.frame_count, b.resolution) == (iso.frame_count, iso.resolution)
    assert iso.decode_budget == 16
    assert math.isclose(b.frame_count * b.resolution ** 2, 16.0)
