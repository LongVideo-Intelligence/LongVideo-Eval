"""Hardware-free tests for the MMTok pruner and its numpy reference cores.

The numpy cores (``greedy_max_coverage`` and ``mmtok_combined_np``) decide which tokens MMTok
keeps. Each is checked against an independently written reference or against hand-computed
values. The pruner-level tests pin the patch declaration and the token accounting. The torch
kernel is compared against these numpy cores in ``test_mmtok_greedy.py``.
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from longvideo_eval.backend.hf.pre_llm_compress import (  # noqa: E402
    greedy_max_coverage,
    mmtok_combined_np,
    mmtok_extract_keywords,
)
from longvideo_eval.backend.hf.prune_seam import PATCH_SPEC_KEY  # noqa: E402
from longvideo_eval.frontend.encode.encoder import QWEN_TOKENS_KIND  # noqa: E402
from longvideo_eval.frontend.prune.mmtok import MMTokPruner  # noqa: E402
from longvideo_eval.interfaces import Budget, VisualTokens  # noqa: E402


# --------------------------------------------------------------------------- #
# greedy_max_coverage: compared with an independently written greedy loop
# --------------------------------------------------------------------------- #
def _greedy_reference(combined, k_max, exclude_indices=()):
    """Plain greedy maximum coverage, written separately from the package version.

    Start with zero coverage. Each step scores every column by the coverage it would add,
    takes the best one (lowest index on a tie), masks it out and raises the running coverage.
    The chosen columns are returned sorted.
    """
    C = np.asarray(combined, dtype=np.float64)
    _, n = C.shape
    best = np.zeros(C.shape[0])
    mask = np.zeros(n)
    for i in exclude_indices:
        mask[i] = -np.inf
    sel = []
    for _ in range(min(k_max, n)):
        delta = np.where(C - best[:, None] > 0, C - best[:, None], 0.0).sum(axis=0) + mask
        idx = int(np.argmax(delta))
        sel.append(idx)
        mask[idx] = -np.inf
        best = np.maximum(best, C[:, idx])
    return tuple(sorted(sel))


def test_greedy_matches_reference_on_random_inputs():
    rng = np.random.default_rng(42)
    for trial in range(10):
        m, n = int(rng.integers(1, 5)), int(rng.integers(4, 24))
        C = rng.random((m + n, n))
        k = int(rng.integers(1, n + 1))
        assert greedy_max_coverage(C, k) == _greedy_reference(C, k), (
            f"trial {trial}: selection differs from the reference greedy loop"
        )


def test_greedy_respects_exclude_and_clamps():
    C = np.eye(4) * 10.0
    assert greedy_max_coverage(C, 2, exclude_indices=(0, 1)) == (2, 3)
    assert greedy_max_coverage(C, 99) == (0, 1, 2, 3)   # k > n clamps to n


def test_greedy_prefers_marginal_coverage_over_repeat():
    # Marginal gain, not plain top-k: column 2 only re-covers row 0 (no gain), so column 1 wins.
    C = np.array([[10.0, 0.0, 9.0], [0.0, 1.0, 0.0]])
    assert greedy_max_coverage(C, 2) == (0, 1)


def test_greedy_rejects_non_2d_input():
    with pytest.raises(ValueError, match="2-D"):
        greedy_max_coverage(np.ones(4), 2)


# --------------------------------------------------------------------------- #
# mmtok_combined_np: the text->video block P and the image->image block Q
# --------------------------------------------------------------------------- #
def test_combined_shape_and_row_sums():
    rng = np.random.default_rng(0)
    text = rng.random((2, 8))
    video = rng.random((5, 8))
    img = rng.random((5, 6))    # the pre-merger features may have a different width
    C = mmtok_combined_np(text, video, img, alpha=0.5)
    assert C.shape == (2 + 5, 5)
    assert np.allclose(C[:2].sum(axis=1), 1.0 / 2)   # P rows sum to 1/m
    assert np.allclose(C[2:].sum(axis=1), 0.5 / 5)   # Q rows sum to alpha/n


def test_combined_hand_computed_two_units():
    text = np.array([[1.0, 0.0]])
    video = np.array([[1.0, 0.0], [0.0, 1.0]])
    img = video
    C = mmtok_combined_np(text, video, img, alpha=0.5, tv_temp=0.01, vv_temp=0.2)
    assert C[0, 0] == pytest.approx(1.0, abs=1e-6)     # softmax([1/0.01, 0]) ~ [1, 0]
    assert C[0, 1] == pytest.approx(0.0, abs=1e-6)
    e5 = np.exp(5.0)
    assert C[1, 0] == pytest.approx(0.25 * e5 / (e5 + 1))   # 0.5 * softmax([5, 0]) / 2
    assert C[1, 1] == pytest.approx(0.25 * 1 / (e5 + 1))


def test_combined_rejects_1d():
    with pytest.raises(ValueError, match="2-D"):
        mmtok_combined_np(np.ones(4), np.ones((3, 4)), np.ones((3, 4)))


def test_text_covered_unit_is_selected_end_to_end():
    # The text points at video unit 2 and Q is weak, so greedy picks the text-covered column.
    text = np.zeros((1, 4))
    text[0, 2] = 1.0
    video = np.eye(4)
    img = np.eye(4)
    C = mmtok_combined_np(text, video, img, alpha=0.001, tv_temp=0.01, vv_temp=0.2)
    assert greedy_max_coverage(C, 1) == (2,)


# --------------------------------------------------------------------------- #
# mmtok_extract_keywords: stopword filtering of the question
# --------------------------------------------------------------------------- #
def test_extract_keywords_filters_stopwords_and_prompts():
    out = mmtok_extract_keywords(
        "Question: What is the man doing? Answer the question using a single word or phrase."
    )
    words = out.lower().split()
    assert "man" in words and "doing" in words
    assert "what" not in words and "the" not in words and "question" not in words


def test_extract_keywords_empty_and_all_stopwords():
    assert mmtok_extract_keywords("") == ""
    assert mmtok_extract_keywords("what is the") == "what is the"   # falls back to the input


# --------------------------------------------------------------------------- #
# MMTokPruner: patch declaration, accounting, guards
# --------------------------------------------------------------------------- #
def _qwen_pkg(pre=120):
    return {"kind": QWEN_TOKENS_KIND, "model_id": "qwen3-vl-4b",
            "num_visual_tokens": pre, "grid_thw": (4, 10, 12)}


def test_mmtok_builds_pre_llm_spec():
    pruner = MMTokPruner(keep_ratio=0.25)
    vt = VisualTokens(tokens=_qwen_pkg(pre=120), num_tokens=120)
    out = pruner.prune(vt, "which unit?", Budget())
    spec = out.tokens[PATCH_SPEC_KEY]
    assert spec.method == "mmtok" and spec.patch_point == "pre_llm"
    assert spec.compression == "mmtok"
    assert spec.pre_prune_tokens == 120 and spec.keep_count == 30   # round(120 * 0.25)
    assert spec.params["alpha"] == 0.5 and spec.params["tv_temp"] == 0.01
    assert spec.params["vv_temp"] == 0.2
    assert spec.params["greedy"] == "lazy"          # the fast maximiser is the default
    # The pruned sequence is what the language model prefills, so the count shrinks.
    assert out.num_tokens == 30
    assert PATCH_SPEC_KEY not in vt.tokens          # the encoder's dict is not mutated


def test_mmtok_greedy_mode_rides_into_the_spec():
    out = MMTokPruner(greedy="plain").prune(
        VisualTokens(tokens=_qwen_pkg(120), num_tokens=120), "q", Budget())
    assert out.tokens[PATCH_SPEC_KEY].params["greedy"] == "plain"
    with pytest.raises(ValueError, match="greedy"):
        MMTokPruner(greedy="fastest")


def test_mmtok_token_budget_overrides_ratio():
    out = MMTokPruner().prune(VisualTokens(tokens=_qwen_pkg(120), num_tokens=120),
                              "q", Budget(token_budget=60))
    assert out.tokens[PATCH_SPEC_KEY].keep_count == 60
    assert out.num_tokens == 60


def test_mmtok_no_spec_when_keeping_all():
    out = MMTokPruner(keep_ratio=1.0).prune(
        VisualTokens(tokens=_qwen_pkg(120), num_tokens=120), "q", Budget())
    assert out.num_tokens == 120 and PATCH_SPEC_KEY not in out.tokens


def test_mmtok_zero_token_edge():
    out = MMTokPruner().prune(VisualTokens(tokens=None, num_tokens=0), "q", Budget())
    assert out.num_tokens == 0


def test_mmtok_rejects_non_qwen_package():
    with pytest.raises(NotImplementedError, match="seam-only"):
        MMTokPruner().prune(VisualTokens(tokens=list(range(10)), num_tokens=10), "q", Budget())


def test_mmtok_rejects_bad_ratio():
    with pytest.raises(ValueError, match="keep_ratio"):
        MMTokPruner(keep_ratio=0.0)


def test_build_mmtok_wires_pruner():
    from longvideo_eval.models.build import build_pipeline

    orch = build_pipeline("qwen3-vl-4b", "mmtok", dry_run=True)
    assert type(orch.pruner).__name__ == "MMTokPruner"
    assert orch.pruner.alpha == 0.5   # the published default


@pytest.mark.parametrize("n,ratio", [(10, 0.25), (5762, 0.25), (9590, 1 / 16), (7, 0.5)])
def test_mmtok_pruner_target_matches_the_kernel_rounding(n, ratio):
    """The pruner's target is round(N * ratio), the count the kernel selects in the backend."""
    from longvideo_eval.frontend.encode.encoder import QWEN_TOKENS_KIND
    from longvideo_eval.frontend.prune.mmtok import MMTokPruner
    from longvideo_eval.interfaces import Budget, VisualTokens

    vt = VisualTokens(tokens={"kind": QWEN_TOKENS_KIND}, num_tokens=n)
    out = MMTokPruner(keep_ratio=ratio).prune(vt, "q", Budget())
    assert out.num_tokens == max(1, int(round(n * ratio)))
