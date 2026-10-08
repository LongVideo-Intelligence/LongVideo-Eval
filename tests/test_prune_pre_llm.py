"""Hardware-free tests for the pre-LLM pruning transform.

The prune point sits after the patch merger and after the position ids are computed, and
before the language model. Everything that does not need a real model is tested here on
numpy stand-ins:

  * the write-back of the compressed tokens and the gather of the surviving positions;
  * the lifecycle of the language-model wrapper (applied on the prefill pass only, restored
    afterwards, restored on an exception);
  * the static keep set of the random control;
  * the conversion of live tensors to numpy at the boundary of the numpy reference cores;
  * the memory guard of the attention recompute.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")

from longvideo_eval.backend.hf.pre_llm_compress import (  # noqa: E402
    attn_div_v2_select_np,
    greedy_max_coverage,
    mmtok_combined_np,
    segment_np,
    to_numpy,
)
from longvideo_eval.backend.hf.prune_seam import (  # noqa: E402
    PRE_LLM_COMPRESSIONS,
    VISION_SOFTMAX_GUARD_BYTES,
    PatchSpec,
    applied_pre_llm_patch,
    check_vision_softmax_budget,
    pre_llm_subselect,
    scatter_seq_rows,
    vision_softmax_bytes,
)


# --------------------------------------------------------------------------- #
# PatchSpec: which compressions the pre-LLM point accepts
# --------------------------------------------------------------------------- #
def test_patchspec_requires_known_compression():
    spec = PatchSpec(method="mmtok", patch_point="pre_llm", pre_prune_tokens=8,
                     keep_indices=(0, 1, 2), compression="mmtok")
    assert spec.compression == "mmtok" and spec.keep_count == 3
    with pytest.raises(ValueError, match="compression in"):
        PatchSpec(method="m", patch_point="pre_llm", pre_prune_tokens=8,
                  keep_indices=(0,), compression="bogus")
    with pytest.raises(ValueError, match="compression in"):
        PatchSpec(method="m", patch_point="pre_llm", pre_prune_tokens=8, keep_indices=(0,))


def test_patchspec_accepts_every_listed_compression():
    assert PRE_LLM_COMPRESSIONS == ("mmtok", "visionzip", "flashvid", "random")
    for comp in PRE_LLM_COMPRESSIONS:
        spec = PatchSpec(method=comp, patch_point="pre_llm", pre_prune_tokens=4,
                         keep_indices=(0, 1), compression=comp)
        assert spec.compression == comp


# --------------------------------------------------------------------------- #
# scatter_seq_rows: the write-back of (possibly merged) token values
# --------------------------------------------------------------------------- #
def test_scatter_seq_rows_writes_and_copies():
    x = np.zeros((1, 5, 2))
    vals = np.array([[1.0, 1.0], [2.0, 2.0]])
    out = scatter_seq_rows(x, [1, 3], vals)
    assert out[0, 1].tolist() == [1.0, 1.0] and out[0, 3].tolist() == [2.0, 2.0]
    assert out[0, 0].tolist() == [0.0, 0.0]
    assert x[0, 1].tolist() == [0.0, 0.0]   # the input is not mutated


def test_scatter_seq_rows_torch_matches_numpy():
    torch = pytest.importorskip("torch")
    x = torch.zeros(1, 5, 2)
    vals = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    out = scatter_seq_rows(x, [1, 3], vals)
    assert out[0, 1].tolist() == [1.0, 1.0] and out[0, 3].tolist() == [2.0, 2.0]
    assert x.abs().sum().item() == 0.0      # the input is not mutated


# --------------------------------------------------------------------------- #
# pre_llm_subselect: write-back, then gather the surviving positions
# --------------------------------------------------------------------------- #
def test_pre_llm_subselect_drops_pruned_visual_and_markers_keeps_text():
    # Sequence layout: [text0, START, v0, v1, v2, v3, END, text1]  (S = 8).
    # Visual placeholders at 2,3,4,5 (N = 4); markers at 1 and 6; text at 0 and 7.
    S, D = 8, 2
    ie = np.arange(S * D, dtype=float).reshape(1, S, D)
    pos = np.arange(S).reshape(1, S).astype(float)
    position_ids = np.stack([pos, pos + 100, pos + 200])          # [3, 1, S]
    attention_mask = np.ones((1, S))
    cache_position = np.arange(S)
    visual_pos_masks = np.array([[0, 0, 1, 1, 1, 1, 0, 0]])
    deepstack = [np.arange(4 * D, dtype=float).reshape(4, D) * 10]  # one row per VISUAL token
    visual_positions = [2, 3, 4, 5]
    marker_positions = [1, 6]
    keep_visual_local = [0, 2]                                     # keep v0, v2 (positions 2, 4)
    compressed = np.array([[-1.0, -1.0], [-2.0, -2.0]])           # merged token values

    out = pre_llm_subselect(
        inputs_embeds=ie, position_ids=position_ids, attention_mask=attention_mask,
        cache_position=cache_position, visual_pos_masks=visual_pos_masks,
        deepstack_visual_embeds=deepstack, visual_positions=visual_positions,
        marker_positions=marker_positions, keep_visual_local=keep_visual_local,
        compressed_tokens=compressed,
    )
    # Survivors: the kept visual positions (2, 4) plus the text positions (0, 7).
    assert out["keep_global"] == [0, 2, 4, 7]
    assert out["post_prune_visual"] == 2
    assert out["inputs_embeds"].shape == (1, 4, D)
    # The written-back values landed at the kept visual positions.
    assert out["inputs_embeds"][0, 1].tolist() == [-1.0, -1.0]    # sequence position 2 == v0
    assert out["inputs_embeds"][0, 2].tolist() == [-2.0, -2.0]    # sequence position 4 == v2
    # Text rows are untouched.
    assert out["inputs_embeds"][0, 0].tolist() == ie[0, 0].tolist()
    assert out["inputs_embeds"][0, 3].tolist() == ie[0, 7].tolist()
    # Position ids are gathered, not renumbered: the gaps left by dropped tokens stay.
    assert out["position_ids"].shape == (3, 1, 4)
    assert out["position_ids"][0, 0].tolist() == [0, 2, 4, 7]
    assert out["position_ids"][1, 0].tolist() == [100, 102, 104, 107]
    assert out["cache_position"].tolist() == [0, 2, 4, 7]
    assert out["attention_mask"].shape == (1, 4)
    assert out["visual_pos_masks"][0].tolist() == [0, 1, 1, 0]
    # Per-visual-token features are gathered by LOCAL visual index (0, 2), not by position.
    assert out["deepstack_visual_embeds"][0].tolist() == [[0.0, 10.0], [40.0, 50.0]]


def test_pre_llm_subselect_handles_optional_none_tensors():
    ie = np.zeros((1, 4, 2))
    out = pre_llm_subselect(
        inputs_embeds=ie, position_ids=None, attention_mask=None, cache_position=None,
        visual_pos_masks=None, deepstack_visual_embeds=None,
        visual_positions=[1, 2], marker_positions=[0], keep_visual_local=[1],
        compressed_tokens=np.array([[9.0, 9.0]]),
    )
    assert out["keep_global"] == [2, 3]  # the kept visual token (2) and the text token (3)
    assert out["position_ids"] is None and out["deepstack_visual_embeds"] is None
    assert out["attention_mask"] is None and out["cache_position"] is None
    assert out["inputs_embeds"][0, 0].tolist() == [9.0, 9.0]


# --------------------------------------------------------------------------- #
# applied_pre_llm_patch: wrapper lifecycle, with injected compression and capture
# --------------------------------------------------------------------------- #
def _fake_model(lm_forward):
    lm = SimpleNamespace(forward=lm_forward)
    return SimpleNamespace(model=SimpleNamespace(language_model=lm)), lm


def _prefill_kwargs(S=8, D=2):
    pos = np.arange(S).reshape(1, S).astype(float)
    return {
        "inputs_embeds": np.arange(S * D, dtype=float).reshape(1, S, D),
        "position_ids": np.stack([pos, pos, pos]),
        "attention_mask": np.ones((1, S)),
        "cache_position": np.arange(S),
        "visual_pos_masks": np.array([[0, 0, 1, 1, 1, 1, 0, 0]]),
        "deepstack_visual_embeds": [np.arange(4 * D, dtype=float).reshape(4, D)],
    }


def test_pre_llm_patch_applies_on_prefill_and_records_arm():
    seen = {}

    def lm_forward(**kwargs):
        seen.update(kwargs)
        return "OUT"

    model, lm = _fake_model(lm_forward)
    spec = PatchSpec(method="flashvid", patch_point="pre_llm", pre_prune_tokens=4,
                     keep_indices=(0, 1), compression="flashvid")
    vis_pos = [2, 3, 4, 5]
    marker_pos = [1, 6]
    features_seen = {}

    def compress_fn(video_features, captured):
        # Keep local tokens 0 and 2, and write recognizable merged values.
        features_seen["shape"] = video_features.shape
        d = video_features.shape[-1]
        comp = np.array([[7.0] * d, [8.0] * d])
        return comp, [0, 2]

    arm = {}
    with applied_pre_llm_patch(model, spec, vis_pos, marker_pos, prefill_len=8,
                               grid_thw=[[2]], compress_fn=compress_fn,
                               install_capture=lambda: [], arm_out=arm):
        assert lm.forward is not lm_forward         # wrapped
        result = lm.forward(**_prefill_kwargs())
    assert result == "OUT"
    assert lm.forward is lm_forward                 # restored
    assert arm == {"pre_prune_visual": 4, "post_prune_visual": 2}
    # grid_thw says 2 frames, so the 4 visual tokens are handed over as (2 frames, 2 tokens, D).
    assert features_seen["shape"] == (2, 2, 2)
    # The reduced sequence is what reached the language model: positions [0, 2, 4, 7].
    assert seen["inputs_embeds"].shape == (1, 4, 2)
    assert seen["cache_position"].tolist() == [0, 2, 4, 7]
    assert seen["inputs_embeds"][0, 1].tolist() == [7.0, 7.0]   # merged v0 at position 2
    assert seen["inputs_embeds"][0, 2].tolist() == [8.0, 8.0]   # merged v2 at position 4
    assert seen["attention_mask"].shape == (1, 4)
    assert seen["deepstack_visual_embeds"][0].shape == (2, 2)


def test_pre_llm_patch_passes_through_decode_step():
    seen = {}

    def lm_forward(**kwargs):
        seen.update(kwargs)
        return "OUT"

    model, lm = _fake_model(lm_forward)
    spec = PatchSpec(method="random", patch_point="pre_llm", pre_prune_tokens=4,
                     keep_indices=(0, 2), compression="random")
    arm = {}
    with applied_pre_llm_patch(model, spec, [2, 3, 4, 5], [1, 6], prefill_len=8,
                               grid_thw=[[1]], install_capture=lambda: [], arm_out=arm):
        # A decode step has sequence length 1 (not the prefill length 8): left untouched.
        lm.forward(inputs_embeds=np.zeros((1, 1, 2)), cache_position=np.array([8]))
    assert seen["inputs_embeds"].shape == (1, 1, 2)   # not pruned
    assert seen["cache_position"].tolist() == [8]
    assert arm == {}                                  # nothing recorded for a decode step


def test_pre_llm_patch_random_static_keep_no_capture():
    # The random control uses spec.keep_indices directly and needs no tower signal.
    seen = {}

    def lm_forward(**kwargs):
        seen.update(kwargs)
        return "OUT"

    model, lm = _fake_model(lm_forward)   # note: this fake model has NO vision tower at all
    spec = PatchSpec(method="random", patch_point="pre_llm", pre_prune_tokens=4,
                     keep_indices=(1, 3), compression="random")
    full = _prefill_kwargs()
    with applied_pre_llm_patch(model, spec, [2, 3, 4, 5], [1, 6], prefill_len=8,
                               grid_thw=None):   # no grid and no capture installer
        lm.forward(**full)
    # Local tokens 1 and 3 sit at positions 3 and 5, so the survivors are [0, 3, 5, 7].
    assert seen["cache_position"].tolist() == [0, 3, 5, 7]
    # Pure selection: the kept rows carry their original values.
    assert seen["inputs_embeds"][0].tolist() == full["inputs_embeds"][0][[0, 3, 5, 7]].tolist()


def test_pre_llm_patch_restores_on_exception():
    def boom(**kwargs):
        raise RuntimeError("boom")

    model, lm = _fake_model(boom)
    original = lm.forward
    spec = PatchSpec(method="random", patch_point="pre_llm", pre_prune_tokens=4,
                     keep_indices=(0, 1), compression="random")
    with pytest.raises(RuntimeError, match="boom"):
        with applied_pre_llm_patch(model, spec, [2, 3, 4, 5], [1, 6], prefill_len=8,
                                   grid_thw=None):
            lm.forward(**_prefill_kwargs())
    assert lm.forward is original


def test_pre_llm_patch_guards_placeholder_count():
    model, lm = _fake_model(lambda **kw: None)
    original = lm.forward
    spec = PatchSpec(method="mmtok", patch_point="pre_llm", pre_prune_tokens=4,
                     keep_indices=(0, 1), compression="mmtok")
    with pytest.raises(ValueError, match="placeholder/count drift"):
        with applied_pre_llm_patch(model, spec, [2, 3, 4], [1, 6], prefill_len=8,
                                   install_capture=lambda: []):
            pass
    assert lm.forward is original     # nothing was wrapped before the guard fired


# --------------------------------------------------------------------------- #
# to_numpy: live tensors must be converted once, at the boundary of the numpy cores
# --------------------------------------------------------------------------- #
class _StrictTensor:
    """A stand-in for a live GPU tensor (on a GPU, half precision, attached to the graph).

    Converting it to numpy raises, as a real tensor does, until the caller has run the whole
    ``.detach().float().cpu()`` chain. A numpy core that calls ``np.asarray`` on its input
    directly therefore fails on this object.
    """

    def __init__(self, data, *, detached=False, floated=False, on_cpu=False):
        self._data = data
        self._detached = detached
        self._floated = floated
        self._on_cpu = on_cpu

    def detach(self):
        return _StrictTensor(self._data, detached=True, floated=self._floated,
                             on_cpu=self._on_cpu)

    def float(self):
        return _StrictTensor(self._data, detached=self._detached, floated=True,
                             on_cpu=self._on_cpu)

    def cpu(self):
        return _StrictTensor(self._data, detached=self._detached, floated=self._floated,
                             on_cpu=True)

    def __array__(self, dtype=None, copy=None):
        if not self._detached:
            raise RuntimeError("Can't call numpy() on Tensor that requires grad.")
        if not self._on_cpu:
            raise TypeError("can't convert cuda:0 device type tensor to numpy.")
        if not self._floated:
            raise TypeError("Got unsupported ScalarType BFloat16")
        arr = np.asarray(self._data, dtype=np.float32)
        return arr.astype(dtype) if dtype is not None else arr


def test_strict_tensor_refuses_partial_conversion():
    live = _StrictTensor([1.0, 2.0])
    with pytest.raises(RuntimeError, match="requires grad"):
        np.asarray(live)
    with pytest.raises(TypeError, match="cuda:0"):
        np.asarray(live.detach())
    with pytest.raises(TypeError, match="BFloat16"):
        np.asarray(live.detach().cpu())
    assert np.asarray(live.detach().float().cpu()).tolist() == [1.0, 2.0]


def test_to_numpy_converts_live_tensor_and_passes_plain_through():
    assert to_numpy(_StrictTensor([[3.0, 4.0]])).tolist() == [[3.0, 4.0]]
    assert to_numpy([1, 2, 3]).tolist() == [1, 2, 3]                 # list passthrough
    arr = np.arange(4.0)
    assert to_numpy(arr).tolist() == arr.tolist()


def test_to_numpy_handles_a_real_half_precision_tensor_with_grad():
    torch = pytest.importorskip("torch")
    t = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16, requires_grad=True)
    out = to_numpy(t)
    assert isinstance(out, np.ndarray) and out.dtype == np.float32
    assert out.tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_mmtok_combined_np_accepts_live_tensors():
    rng = np.random.default_rng(12)
    text, video, img = rng.random((2, 6)), rng.random((5, 6)), rng.random((5, 4))
    plain = mmtok_combined_np(text, video, img)
    live = mmtok_combined_np(_StrictTensor(text.astype(np.float32)),
                             _StrictTensor(video.astype(np.float32)),
                             _StrictTensor(img.astype(np.float32)))
    assert np.allclose(plain, live, atol=1e-6)   # float32 round-trip tolerance


def test_attn_div_v2_select_np_accepts_live_tensors():
    rng = np.random.default_rng(13)
    feats = rng.random((2, 10, 5))
    cls = rng.random((2, 10))
    plain = attn_div_v2_select_np(feats, cls, 4)
    live = attn_div_v2_select_np(_StrictTensor(feats.astype(np.float32)),
                                 _StrictTensor(cls.astype(np.float32)), 4)
    assert plain.tolist() == live.tolist()


def test_segment_np_accepts_live_tensors():
    rng = np.random.default_rng(14)
    fm = rng.random((8, 4))
    plain = segment_np(fm, segment_threshold=0.99, min_segment_num=4)
    live = segment_np(_StrictTensor(fm.astype(np.float32)),
                      segment_threshold=0.99, min_segment_num=4)
    assert plain.tolist() == live.tolist()


def test_greedy_max_coverage_accepts_live_tensors():
    rng = np.random.default_rng(15)
    C = rng.random((7, 5))
    assert greedy_max_coverage(_StrictTensor(C.astype(np.float32)), 3) == \
        greedy_max_coverage(C, 3)


# --------------------------------------------------------------------------- #
# memory guard of the attention recompute: it must fail before allocating a huge softmax,
# name the fix (the pixel budget), and never silently chunk or change precision
# --------------------------------------------------------------------------- #
def test_vision_softmax_bytes_formula():
    # heads x seq^2 x 4 bytes (single precision).
    assert vision_softmax_bytes(16, 1024) == 16 * 1024 * 1024 * 4
    # A frame within a normal pixel budget sits far under the guard.
    assert vision_softmax_bytes(16, 3072) < VISION_SOFTMAX_GUARD_BYTES == 8 * 1024**3


def test_vision_softmax_guard_passes_under_limit():
    check_vision_softmax_budget(16, 3072)                    # no raise (~0.6 GiB)
    check_vision_softmax_budget(16, 11585)                   # just under 8 GiB


def test_vision_softmax_guard_fails_loud_on_an_oversized_frame():
    # ~24k patches in one frame: 16 heads x 24576^2 x 4 bytes = 36 GiB, over the 8 GiB guard.
    with pytest.raises(RuntimeError) as exc:
        check_vision_softmax_budget(16, 24576)
    msg = str(exc.value)
    assert "pixel" in msg and "budget" in msg               # names the actual fix
    assert "fp32" in msg and "GiB" in msg
    assert "chunk" in msg                                    # forbids the silent workaround


def test_vision_softmax_guard_custom_limit():
    with pytest.raises(RuntimeError):
        check_vision_softmax_budget(2, 1024, limit_bytes=1024)   # tiny limit trips
    check_vision_softmax_budget(2, 1024, limit_bytes=2**40)      # huge limit passes
