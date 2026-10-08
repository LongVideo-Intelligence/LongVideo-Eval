"""Hardware-free tests for the FlashVID pruner and its numpy reference cores.

FlashVID here runs its vision-side compression before the language model, with no pruning
stage inside the LLM. The value-modifying merges run in torch; the two index decisions that
drive them have torch-free numpy versions, which are checked here against independently
written references:

  * ``segment_np``: where the video is cut into segments;
  * ``attn_div_v2_select_np``: which tokens the attention/diversity stage keeps.

The torch kernel is exercised at the bottom when torch is installed.
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from longvideo_eval.backend.hf.pre_llm_compress import (  # noqa: E402
    FlashVidVisionConfig,
    attn_div_v2_select_np,
    flashvid_compression,
    segment_np,
)
from longvideo_eval.backend.hf.prune_seam import PATCH_SPEC_KEY  # noqa: E402
from longvideo_eval.frontend.encode.encoder import QWEN_TOKENS_KIND  # noqa: E402
from longvideo_eval.frontend.prune.flashvid import FlashVidPruner  # noqa: E402
from longvideo_eval.interfaces import Budget, VisualTokens  # noqa: E402


# --------------------------------------------------------------------------- #
# segment_np: cut where consecutive frames stop looking alike
# --------------------------------------------------------------------------- #
def _segment_reference(frame_means, thresh, min_seg, comp=True):
    """Segment lengths, written separately from the package version (plain Python lists)."""
    fm = np.asarray(frame_means, float)
    nf = fm.shape[0]
    normed = fm / np.maximum(np.linalg.norm(fm, axis=-1, keepdims=True), 1e-12)
    trans = np.sum(normed[:-1] * normed[1:], axis=-1)
    cuts = [i for i in range(len(trans)) if trans[i] < thresh]
    if len(cuts) + 1 < min_seg and comp:
        remaining = min_seg - (len(cuts) + 1)
        masked = trans.copy()
        for i in cuts:
            masked[i] = 1.0
        order = sorted(range(len(masked)), key=lambda i: (masked[i], i))
        cuts = sorted(cuts + order[:min(remaining, len(masked))])
    bounds = [-1] + cuts + [nf - 1]
    return [bounds[i + 1] - bounds[i] for i in range(len(bounds) - 1)]


def test_segment_np_matches_reference_clear_cuts():
    # 4 frames: a sharp change between frames 1 and 2 (orthogonal), identical otherwise.
    fm = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    segs = segment_np(fm, segment_threshold=0.5, min_segment_num=1)
    assert segs.tolist() == _segment_reference(fm, 0.5, 1) == [2, 2]
    assert segs.sum() == 4


def test_segment_np_complementary_fills_to_min_segments():
    rng = np.random.default_rng(0)
    fm = rng.random((10, 4))
    # A high threshold plus a segment floor forces the extra (complementary) cuts.
    segs = segment_np(fm, segment_threshold=0.99, min_segment_num=5)
    assert segs.tolist() == _segment_reference(fm, 0.99, 5)
    assert len(segs) >= 5 and segs.sum() == 10


def test_segment_np_without_complementary_keeps_only_threshold_cuts():
    fm = np.ones((6, 3))
    segs = segment_np(fm, segment_threshold=0.5, min_segment_num=4,
                      complementary_segment=False)
    assert segs.tolist() == [6]


def test_segment_np_single_segment_when_all_similar():
    fm = np.ones((6, 3))
    segs = segment_np(fm, segment_threshold=0.5, min_segment_num=1)
    assert segs.tolist() == [6]


# --------------------------------------------------------------------------- #
# attn_div_v2_select_np: farthest-point selection over attention-weighted distance
# --------------------------------------------------------------------------- #
def _attn_div_reference(features, cls_attention, k):
    """Per-batch loop version of the selection, written separately from the package one."""
    feats = np.asarray(features, float)
    B, n, d = feats.shape
    pooled = feats.mean(axis=1)
    gca = np.asarray(cls_attention, float) * 1e6
    out = np.zeros((B, k), dtype=int)
    for b in range(B):
        f = feats[b]
        normed = f / np.maximum(np.linalg.norm(f, axis=-1, keepdims=True), 1e-12)
        dist = 1.0 - normed @ normed.T
        # Relevance of each token to the pooled frames: mean over all pooled vectors.
        local = (f @ pooled.T).mean(axis=1)
        dist = dist * gca[b][None, :] * local[None, :]
        keep = [int(np.argmax(np.sort(dist, axis=0)[1, :]))]
        for _ in range(1, k):
            sub = dist[keep, :]
            md = sub.min(axis=0)
            keep.append(int(np.argmax(md)))
        out[b] = sorted(keep)
    return out


def test_attn_div_matches_reference_on_random_inputs():
    rng = np.random.default_rng(7)
    for _ in range(6):
        B, n, d = 2, int(rng.integers(6, 16)), 5
        feats = rng.random((B, n, d))
        cls = rng.random((B, n))
        k = int(rng.integers(2, n // 2 + 1))
        got = attn_div_v2_select_np(feats, cls, k)
        ref = _attn_div_reference(feats, cls, k)
        assert got.tolist() == ref.tolist()


def test_attn_div_returns_sorted_rows():
    rng = np.random.default_rng(1)
    feats = rng.random((3, 12, 4))
    cls = rng.random((3, 12))
    keep = attn_div_v2_select_np(feats, cls, 5)
    assert keep.shape == (3, 5)
    for row in keep:
        assert row.tolist() == sorted(row.tolist())
        assert all(0 <= i < 12 for i in row.tolist())


# --------------------------------------------------------------------------- #
# FlashVidPruner: patch declaration, accounting, guards
# --------------------------------------------------------------------------- #
def _qwen_pkg(pre=100):
    return {"kind": QWEN_TOKENS_KIND, "model_id": "qwen3-vl-4b",
            "num_visual_tokens": pre, "grid_thw": (4, 10, 12)}


def test_flashvid_builds_pre_llm_vision_side_spec():
    pruner = FlashVidPruner(keep_ratio=0.25)
    vt = VisualTokens(tokens=_qwen_pkg(pre=100), num_tokens=100)
    out = pruner.prune(vt, "a query", Budget())
    spec = out.tokens[PATCH_SPEC_KEY]
    assert spec.method == "flashvid" and spec.patch_point == "pre_llm"
    assert spec.compression == "flashvid"
    assert spec.params["alpha"] == 0.7            # selection/merge split, not a prune ratio
    assert spec.params["expansion"] == 1.0        # no pruning stage inside the LLM
    assert spec.params["llm_pruning"] == "off"
    assert spec.pre_prune_tokens == 100 and spec.keep_count == 25
    assert out.num_tokens == 25                   # the LLM prefill shrinks to this
    assert PATCH_SPEC_KEY not in vt.tokens        # the encoder's dict is not mutated


def test_flashvid_defaults():
    p = FlashVidPruner()
    assert p.alpha == 0.7 and p.segment_threshold == 0.9
    assert p.min_segment_num == 8 and p.temporal_threshold == 0.8
    assert p.do_segment is True and p.complementary_segment is True


def test_flashvid_constructor_arguments_ride_into_the_spec():
    p = FlashVidPruner(alpha=0.6, segment_threshold=0.8, min_segment_num=4,
                       temporal_threshold=1.0, do_segment=False, complementary_segment=False)
    spec = p.prune(VisualTokens(tokens=_qwen_pkg(100), num_tokens=100), "q",
                   Budget()).tokens[PATCH_SPEC_KEY]
    assert spec.params["alpha"] == 0.6 and spec.params["segment_threshold"] == 0.8
    assert spec.params["min_segment_num"] == 4 and spec.params["temporal_threshold"] == 1.0
    assert spec.params["do_segment"] is False
    assert spec.params["complementary_segment"] is False


def test_flashvid_token_budget_overrides_ratio():
    out = FlashVidPruner().prune(VisualTokens(tokens=_qwen_pkg(100), num_tokens=100),
                                 "q", Budget(token_budget=10))
    assert out.tokens[PATCH_SPEC_KEY].keep_count == 10
    assert out.num_tokens == 10


def test_flashvid_no_spec_when_keep_ge_pre():
    out = FlashVidPruner(keep_ratio=1.0).prune(
        VisualTokens(tokens=_qwen_pkg(10), num_tokens=10), "q", Budget())
    assert out.num_tokens == 10 and PATCH_SPEC_KEY not in out.tokens


def test_flashvid_zero_token_edge():
    out = FlashVidPruner().prune(VisualTokens(tokens=None, num_tokens=0), "q", Budget())
    assert out.num_tokens == 0


def test_flashvid_rejects_non_qwen_package():
    with pytest.raises(NotImplementedError, match="seam-only"):
        FlashVidPruner().prune(VisualTokens(tokens=list(range(100)), num_tokens=100),
                               "q", Budget())


def test_flashvid_rejects_bad_ratio():
    with pytest.raises(ValueError, match="keep_ratio"):
        FlashVidPruner(keep_ratio=0.0)


def test_build_flashvid_wires_pruner():
    from longvideo_eval.models.build import build_pipeline

    orch = build_pipeline("qwen3-vl-4b", "flashvid", dry_run=True)
    assert type(orch.pruner).__name__ == "FlashVidPruner"
    assert orch.pruner.alpha == 0.7


def test_build_prune_kwargs_sweep_keep_ratio():
    # The retention ratio is the variable these pruners sweep, set through prune_kwargs.
    from longvideo_eval.models.build import build_pipeline

    for method in ("flashvid", "mmtok"):
        orch = build_pipeline("qwen3-vl-4b", method, dry_run=True,
                              prune_kwargs={"keep_ratio": 0.1})
        assert orch.pruner.keep_ratio == 0.1
        out = orch.pruner.prune(
            VisualTokens(tokens=_qwen_pkg(pre=100), num_tokens=100), "q", Budget())
        spec = out.tokens[PATCH_SPEC_KEY]
        assert spec.patch_point == "pre_llm" and spec.keep_count == 10


# --------------------------------------------------------------------------- #
# torch kernel (runs only where torch is installed)
# --------------------------------------------------------------------------- #
def _video(torch, nf=8, nt=16, d=12, seed=0):
    """Non-negative features, so every token has a positive relevance to the pooled frames."""
    g = torch.Generator().manual_seed(seed)
    feats = torch.rand(nf, nt, d, generator=g)
    attn = torch.rand(nf, nt, generator=g)
    return feats, attn


def test_flashvid_kernel_returns_sorted_indices_and_aligned_rows():
    torch = pytest.importorskip("torch")
    feats, attn = _video(torch)
    nf, nt, d = feats.shape
    tokens, idx = flashvid_compression(feats.clone(), attn,
                                       FlashVidVisionConfig(retention_ratio=0.25))
    idx_list = idx.tolist()
    assert tokens.shape == (len(idx_list), d)            # one row per kept index
    assert idx_list == sorted(idx_list)                  # ascending token order
    assert 0 < len(idx_list) < nf * nt                   # it actually compressed
    assert 0 <= idx_list[0] and idx_list[-1] < nf * nt


def test_flashvid_kernel_keeps_distinct_tokens_when_the_video_is_one_segment():
    torch = pytest.importorskip("torch")
    for seed in range(3):
        feats, attn = _video(torch, nf=8, nt=32, seed=seed)
        _, idx = flashvid_compression(
            feats.clone(), attn, FlashVidVisionConfig(retention_ratio=0.25, do_segment=False))
        idx_list = idx.tolist()
        assert idx_list == sorted(set(idx_list))         # no token kept twice


def test_flashvid_kernel_selection_only_path_returns_the_original_rows():
    # alpha=1.0 gives the whole per-frame budget to the selection stage: nothing is merged,
    # so every returned row is an input token and each frame keeps exactly its budget.
    torch = pytest.importorskip("torch")
    feats, attn = _video(torch, nf=6, nt=16, seed=4)
    nf, nt, d = feats.shape
    original = feats.clone()
    tokens, idx = flashvid_compression(
        feats, attn, FlashVidVisionConfig(retention_ratio=0.25, alpha=1.0))
    idx_list = idx.tolist()
    assert len(idx_list) == nf * 4                       # ceil(16 * 0.25) per frame
    assert idx_list == sorted(set(idx_list))
    assert [i // nt for i in idx_list] == [f for f in range(nf) for _ in range(4)]
    assert torch.equal(tokens, original.reshape(nf * nt, d)[idx])
    assert torch.equal(feats, original)                  # the input is left as it was


def test_flashvid_kernel_selection_matches_the_numpy_core():
    # With merging switched off, the kept indices are the numpy selection core's, per frame.
    torch = pytest.importorskip("torch")
    feats, attn = _video(torch, nf=5, nt=12, seed=6)
    nf, nt, _ = feats.shape
    _, idx = flashvid_compression(
        feats.clone(), attn,
        FlashVidVisionConfig(retention_ratio=0.25, alpha=1.0, do_segment=False))
    local = attn_div_v2_select_np(feats.numpy(), attn.numpy(), 3)     # ceil(12 * 0.25)
    expected = [f * nt + int(i) for f in range(nf) for i in local[f]]
    assert idx.tolist() == expected


def test_flashvid_kernel_segments_like_the_numpy_core():
    torch = pytest.importorskip("torch")
    from longvideo_eval.backend.hf import _flashvid_torch as fv

    for seed in range(4):
        feats, _ = _video(torch, nf=12, nt=8, seed=10 + seed)
        means = feats.mean(1)
        for threshold, floor in ((0.9, 8), (0.99, 5), (0.5, 1), (0.97, 3)):
            got = fv._segment(means, threshold, floor, True).tolist()
            assert got == segment_np(means.numpy(), threshold, floor).tolist()
            assert sum(got) == 12


def test_flashvid_kernel_is_deterministic():
    torch = pytest.importorskip("torch")
    feats, attn = _video(torch, seed=3)
    a_tok, a_idx = flashvid_compression(feats.clone(), attn, FlashVidVisionConfig())
    b_tok, b_idx = flashvid_compression(feats.clone(), attn, FlashVidVisionConfig())
    assert a_idx.tolist() == b_idx.tolist()
    assert torch.equal(a_tok, b_tok)


def test_flashvid_kernel_keeps_fewer_tokens_at_a_lower_retention():
    torch = pytest.importorskip("torch")
    feats, attn = _video(torch, nf=8, nt=32, seed=5)
    _, small = flashvid_compression(feats.clone(), attn,
                                    FlashVidVisionConfig(retention_ratio=0.125))
    _, large = flashvid_compression(feats.clone(), attn,
                                    FlashVidVisionConfig(retention_ratio=0.5))
    assert len(small) < len(large)


def test_flashvid_token_budget_reaches_the_kernel_as_a_ratio(monkeypatch):
    """An absolute token budget is turned into the retention ratio that yields it."""
    from longvideo_eval.backend.hf import pre_llm_compress as C
    from longvideo_eval.backend.hf import prune_seam

    seen = {}

    def fake(video_features, cls_attention, cfg):
        seen["ratio"] = cfg.retention_ratio
        return video_features, [0]

    monkeypatch.setattr(C, "flashvid_compression", fake)

    class _F:
        shape = (4, 64, 8)

    prune_seam._run_pre_llm_compression(
        "flashvid", _F(), {"cls_attention": None}, {"keep_ratio": 0.25, "token_budget": 16}
    )
    assert seen["ratio"] == 16 / 256
