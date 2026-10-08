"""Hardware-free tests for the pruning seam and the random-pruning control.

A pruner does not drop tokens itself: it declares a ``PatchSpec`` and the backend applies it
around ``generate``. Covered here:

  * ``PatchSpec`` validation and the small tensor helpers (``take_dim``, ``subselect_rows``,
    ``resolve_module``);
  * which vision-tower signals each compression asks for, and that only the matching hooks
    are installed (and removed again);
  * the random control: seeding, keep-count arithmetic, patch declaration, accounting;
  * that the backend's run record reaches the result rows.

The tests that run a real compression kernel through the seam need torch and are skipped
without it.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")

from longvideo_eval.backend.hf import prune_seam  # noqa: E402
from longvideo_eval.backend.hf.pre_llm_compress import (  # noqa: E402
    FlashVidVisionConfig,
    flashvid_compression,
    visionzip_compression,
    greedy_max_coverage,
    mmtok_combined_np,
    mmtok_extract_keywords,
)
from longvideo_eval.backend.hf.prune_seam import (  # noqa: E402
    PATCH_POINTS,
    PATCH_SPEC_KEY,
    PRE_LLM_COMPRESSIONS,
    PRE_LLM_SIGNALS,
    PRE_LLM_STATIC,
    PatchSpec,
    _install_tower_capture,
    applied_pre_llm_patch,
    resolve_module,
    subselect_rows,
    take_dim,
)
from longvideo_eval.frontend.encode.encoder import QWEN_TOKENS_KIND  # noqa: E402
from longvideo_eval.frontend.prune.random_prune import (  # noqa: E402
    RandomPruner,
    keep_count,
    random_keep_indices,
    seed_from_query,
)
from longvideo_eval.interfaces import Budget, VisualTokens  # noqa: E402


def _spec(pre=8, keep=(1, 3, 5), compression="random", **kw):
    return PatchSpec(method=compression, patch_point="pre_llm", pre_prune_tokens=pre,
                     keep_indices=keep, compression=compression, **kw)


# --------------------------------------------------------------------------- #
# PatchSpec: validation of the declarative patch
# --------------------------------------------------------------------------- #
def test_patchspec_derives_keep_count_and_normalizes_indices():
    spec = _spec(keep=[1, 3, 5])                 # a list is accepted and frozen to a tuple
    assert spec.keep_count == 3
    assert spec.keep_indices == (1, 3, 5) and isinstance(spec.keep_indices, tuple)
    assert spec.tower_target == "model.visual"
    assert spec.params == {}


def test_patchspec_rejects_unsorted_dup_oob_empty():
    with pytest.raises(ValueError, match="sorted"):
        _spec(keep=(3, 1))
    with pytest.raises(ValueError, match="unique"):
        _spec(keep=(1, 1))
    with pytest.raises(ValueError, match="out of range"):
        _spec(keep=(1, 8))
    with pytest.raises(ValueError, match="out of range"):
        _spec(keep=(-1, 2))
    with pytest.raises(ValueError, match="at least one"):
        _spec(keep=())
    with pytest.raises(ValueError, match="pre_prune_tokens must be positive"):
        _spec(pre=0, keep=(0,))


def test_patchspec_only_knows_the_pre_llm_point():
    assert PATCH_POINTS == ("pre_llm",)
    for point in ("vision_tower", "post_llm", ""):
        with pytest.raises(ValueError, match="patch_point"):
            PatchSpec(method="m", patch_point=point, pre_prune_tokens=8,
                      keep_indices=(0,), compression="random")


def test_patchspec_is_frozen():
    spec = _spec()
    with pytest.raises(Exception):
        spec.compression = "mmtok"


def test_patchspec_has_no_per_layer_or_per_frame_fields():
    # The reduced seam prunes at one point, at token granularity, with no selection-signal
    # switch: the dataclass carries exactly these fields.
    import dataclasses

    assert [f.name for f in dataclasses.fields(PatchSpec)] == [
        "method", "patch_point", "pre_prune_tokens", "keep_indices",
        "tower_target", "compression", "params",
    ]


# --------------------------------------------------------------------------- #
# take_dim / subselect_rows / resolve_module
# --------------------------------------------------------------------------- #
def test_take_dim_numpy_along_each_axis():
    x = np.arange(24).reshape(2, 3, 4)
    assert take_dim(x, [0, 2], 1).tolist() == x[:, [0, 2], :].tolist()
    assert take_dim(x, [3, 1], -1).tolist() == x[:, :, [3, 1]].tolist()   # order preserved
    assert take_dim(x, (1,), 0).shape == (1, 3, 4)


def test_take_dim_torch_matches_numpy():
    torch = pytest.importorskip("torch")
    x = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    t = torch.from_numpy(x)
    for dim, idx in ((1, [0, 2]), (-1, [3, 1]), (0, [1])):
        got = take_dim(t, idx, dim)
        assert isinstance(got, torch.Tensor)
        assert got.tolist() == take_dim(x, idx, dim).tolist()
    # A tensor index vector is accepted as well.
    assert take_dim(t, torch.tensor([2, 0]), 1).tolist() == x[:, [2, 0], :].tolist()


def test_subselect_rows_list_preserves_order_and_type():
    assert subselect_rows(["a", "b", "c", "d"], [0, 2]) == ["a", "c"]
    assert subselect_rows(("a", "b", "c", "d"), [3, 1]) == ("d", "b")


def test_subselect_rows_array_takes_axis_zero():
    arr = np.arange(12).reshape(4, 3)
    assert subselect_rows(arr, [1, 3]).tolist() == [[3, 4, 5], [9, 10, 11]]


def test_resolve_module_walks_the_dotted_path():
    leaf = object()
    model = SimpleNamespace(model=SimpleNamespace(visual=leaf))
    assert resolve_module(model, "model.visual") is leaf


def test_resolve_module_fails_loud():
    model = SimpleNamespace(model=SimpleNamespace(language_model=1))
    with pytest.raises(AttributeError, match="cannot resolve.*visual") as exc:
        resolve_module(model, "model.visual")
    assert "language_model" in str(exc.value)      # the message lists what WAS found


# --------------------------------------------------------------------------- #
# Which tower signals each compression needs
# --------------------------------------------------------------------------- #
def test_signal_table_matches_the_compressions():
    assert set(PRE_LLM_SIGNALS) == set(PRE_LLM_COMPRESSIONS)
    assert PRE_LLM_SIGNALS["mmtok"] == ("img_features",)
    assert PRE_LLM_SIGNALS["visionzip"] == ("cls_attention", "attn_keys")
    assert PRE_LLM_SIGNALS["flashvid"] == ("cls_attention",)
    assert PRE_LLM_SIGNALS["random"] == ()
    # The static controls are exactly the compressions that need no signal.
    assert set(PRE_LLM_STATIC) == {c for c, s in PRE_LLM_SIGNALS.items() if not s}


# --------------------------------------------------------------------------- #
# _install_tower_capture: only the requested hooks go in, and they come out again
# --------------------------------------------------------------------------- #
def _fake_tower_model():
    """A model whose vision tower has just the two hook points: the last block's attention
    and the patch merger. Both forwards record that they ran and return a marker."""
    calls = []

    def attn_forward(*args, **kwargs):
        calls.append("attn")
        return "ATTN_OUT"

    def merger_forward(*args, **kwargs):
        calls.append("merger")
        return "MERGER_OUT"

    first_attn = SimpleNamespace(forward=lambda *a, **k: "FIRST")
    attn = SimpleNamespace(forward=attn_forward)
    merger = SimpleNamespace(forward=merger_forward)
    tower = SimpleNamespace(
        blocks=[SimpleNamespace(attn=first_attn), SimpleNamespace(attn=attn)], merger=merger,
    )
    model = SimpleNamespace(model=SimpleNamespace(visual=tower))
    return model, tower, attn, merger, attn_forward, merger_forward, calls


def _restore(restore):
    for mod, fwd in restore:
        mod.forward = fwd


def test_capture_installs_only_the_merger_hook_for_img_features():
    model, tower, attn, merger, attn_fwd, merger_fwd, _ = _fake_tower_model()
    restore = _install_tower_capture(model, {}, "model.visual", None, signals=("img_features",))
    assert attn.forward is attn_fwd                 # attention left alone
    assert merger.forward is not merger_fwd         # merger wrapped
    assert [m for m, _ in restore] == [merger]
    _restore(restore)
    assert merger.forward is merger_fwd


def test_capture_installs_only_the_attention_hook_for_cls_attention():
    model, tower, attn, merger, attn_fwd, merger_fwd, _ = _fake_tower_model()
    first_fwd = tower.blocks[0].attn.forward
    restore = _install_tower_capture(model, {}, "model.visual", None, signals=("cls_attention",))
    assert attn.forward is not attn_fwd             # the LAST block's attention is wrapped
    assert tower.blocks[0].attn.forward is first_fwd   # earlier blocks are not
    assert merger.forward is merger_fwd             # merger left alone
    assert [m for m, _ in restore] == [attn]
    _restore(restore)
    assert attn.forward is attn_fwd


def test_capture_with_no_signals_installs_nothing():
    model, tower, attn, merger, attn_fwd, merger_fwd, _ = _fake_tower_model()
    assert _install_tower_capture(model, {}, "model.visual", None, signals=()) == []
    assert attn.forward is attn_fwd and merger.forward is merger_fwd


def test_capture_default_installs_both_hooks_and_restores_both():
    model, tower, attn, merger, attn_fwd, merger_fwd, _ = _fake_tower_model()
    restore = _install_tower_capture(model, {}, "model.visual", None)
    assert attn.forward is not attn_fwd and merger.forward is not merger_fwd
    assert {id(m) for m, _ in restore} == {id(attn), id(merger)}
    _restore(restore)
    assert attn.forward is attn_fwd and merger.forward is merger_fwd


def test_capture_fails_loud_on_an_unexpected_tower():
    no_merger = SimpleNamespace(model=SimpleNamespace(visual=SimpleNamespace(blocks=[])))
    with pytest.raises(AttributeError, match="lacks blocks/merger"):
        _install_tower_capture(no_merger, {}, "model.visual", None)
    no_attn = SimpleNamespace(model=SimpleNamespace(visual=SimpleNamespace(
        blocks=[SimpleNamespace()], merger=SimpleNamespace(forward=lambda x: x))))
    with pytest.raises(AttributeError, match="lacks `attn`"):
        _install_tower_capture(no_attn, {}, "model.visual", None, signals=("cls_attention",))
    with pytest.raises(AttributeError, match="cannot resolve"):
        _install_tower_capture(SimpleNamespace(), {}, "model.visual", None)


def test_attention_hook_is_read_only_and_captures_the_recomputed_signal(monkeypatch):
    model, tower, attn, merger, attn_fwd, merger_fwd, calls = _fake_tower_model()
    seen = {}

    def fake_recompute(attn_module, hidden_states, cu_seqlens, position_embeddings, grid_thw):
        seen.update(module=attn_module, hidden=hidden_states, cu=cu_seqlens,
                    pos=position_embeddings, grid=grid_thw)
        return "ATTENTION_SIGNAL", "KEY_SIGNAL"

    monkeypatch.setattr(prune_seam, "_recompute_vision_signals", fake_recompute)
    captured = {}
    _install_tower_capture(model, captured, "model.visual", "GRID", signals=("cls_attention",))

    # Without the inputs the recompute needs, the hook passes through and captures nothing.
    assert attn.forward("hidden") == "ATTN_OUT"
    assert captured == {}
    # With them, the original output is returned unchanged and the signal is stored.
    out = attn.forward("hidden", cu_seqlens="CU", position_embeddings=("cos", "sin"))
    assert out == "ATTN_OUT"
    # The recompute also yields the key vectors; a kernel that did not ask for them does
    # not get them stored.
    assert captured == {"cls_attention": "ATTENTION_SIGNAL"}
    assert seen == {"module": attn, "hidden": "hidden", "cu": "CU",
                    "pos": ("cos", "sin"), "grid": "GRID"}
    assert calls == ["attn", "attn"]                # the real forward ran both times


def test_attention_hook_stores_the_key_vectors_only_when_they_are_requested(monkeypatch):
    monkeypatch.setattr(prune_seam, "_recompute_vision_signals",
                        lambda *args: ("ATTENTION_SIGNAL", "KEY_SIGNAL"))

    def capture(signals):
        model, tower, attn, merger, attn_fwd, merger_fwd, _ = _fake_tower_model()
        captured = {}
        restore = _install_tower_capture(model, captured, "model.visual", "GRID",
                                         signals=signals)
        assert [m for m, _ in restore] == [attn]    # one attention hook, never the merger
        attn.forward("hidden", cu_seqlens="CU", position_embeddings=("cos", "sin"))
        return captured

    # FlashVID's request: the attention only.
    assert capture(PRE_LLM_SIGNALS["flashvid"]) == {"cls_attention": "ATTENTION_SIGNAL"}
    # VisionZip's request: the attention and the key vectors.
    assert capture(PRE_LLM_SIGNALS["visionzip"]) == {
        "cls_attention": "ATTENTION_SIGNAL", "attn_keys": "KEY_SIGNAL"}
    # Asking for the key vectors alone still installs the attention hook.
    assert capture(("attn_keys",))["attn_keys"] == "KEY_SIGNAL"


def test_merger_hook_is_read_only_and_averages_each_merge_window():
    torch = pytest.importorskip("torch")
    model, tower, attn, merger, attn_fwd, merger_fwd, calls = _fake_tower_model()
    captured = {}
    _install_tower_capture(model, captured, "model.visual", None, signals=("img_features",))
    # 3 merged tokens, each built from a window of 4 pre-merger rows of width 2.
    pre = torch.arange(24, dtype=torch.float32).reshape(12, 2)
    assert merger.forward(pre) == "MERGER_OUT"       # the original output is returned
    assert calls == ["merger"]
    assert captured["img_features"].shape == (3, 2)
    assert torch.equal(captured["img_features"], pre.view(3, 4, 2).mean(dim=1))
    assert set(captured) == {"img_features"}


# --------------------------------------------------------------------------- #
# applied_pre_llm_patch: asks for exactly the signals the compression needs
# --------------------------------------------------------------------------- #
def _lm_only_model():
    lm = SimpleNamespace(forward=lambda **kw: "OUT")
    return SimpleNamespace(model=SimpleNamespace(language_model=lm)), lm


@pytest.mark.parametrize("compression,expected", [
    ("mmtok", ("img_features",)),
    ("visionzip", ("cls_attention", "attn_keys")),
    ("flashvid", ("cls_attention",)),
])
def test_patch_requests_only_the_signals_its_compression_needs(
        monkeypatch, compression, expected):
    requests = []

    def spy(model, captured, tower_target, grid_thw, signals=None):
        requests.append({"signals": tuple(signals), "tower_target": tower_target,
                         "grid_thw": grid_thw})
        return []

    monkeypatch.setattr(prune_seam, "_install_tower_capture", spy)
    model, lm = _lm_only_model()
    original = lm.forward
    spec = _spec(pre=4, keep=(0, 1), compression=compression)
    with applied_pre_llm_patch(model, spec, [2, 3, 4, 5], [1, 6], prefill_len=8,
                               grid_thw="GRID", tower_target="model.visual"):
        assert lm.forward is not original
    assert lm.forward is original
    assert requests == [{"signals": expected, "tower_target": "model.visual",
                         "grid_thw": "GRID"}]


def test_random_patch_never_touches_the_tower(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("the random control must not install tower hooks")

    monkeypatch.setattr(prune_seam, "_install_tower_capture", forbidden)
    model, lm = _lm_only_model()
    with applied_pre_llm_patch(model, _spec(pre=4, keep=(0, 1)), [2, 3, 4, 5], [1, 6],
                               prefill_len=8):
        pass


@pytest.mark.parametrize("compression", ["mmtok", "visionzip", "flashvid", "random"])
def test_injected_capture_installer_replaces_the_default(monkeypatch, compression):
    def forbidden(*args, **kwargs):
        raise AssertionError("an injected installer must be used instead of the default")

    monkeypatch.setattr(prune_seam, "_install_tower_capture", forbidden)
    hooked = SimpleNamespace(forward="WRAPPED")
    installs = []

    def install_capture():
        installs.append(1)
        return [(hooked, "ORIGINAL")]

    model, lm = _lm_only_model()
    with applied_pre_llm_patch(model, _spec(pre=4, keep=(0, 1), compression=compression),
                               [2, 3, 4, 5], [1, 6], prefill_len=8,
                               install_capture=install_capture):
        assert hooked.forward == "WRAPPED"
    assert installs == [1]
    assert hooked.forward == "ORIGINAL"             # the installer's restore list is honoured


def test_real_hooks_follow_the_compression_and_are_restored():
    # No monkeypatching: the context manager itself decides which hooks go on a fake tower.
    for compression, attn_wrapped, merger_wrapped in (
        ("mmtok", False, True), ("visionzip", True, False), ("flashvid", True, False),
        ("random", False, False),
    ):
        model, tower, attn, merger, attn_fwd, merger_fwd, _ = _fake_tower_model()
        lm = SimpleNamespace(forward=lambda **kw: "OUT")
        model.model.language_model = lm
        lm_fwd = lm.forward
        with applied_pre_llm_patch(model, _spec(pre=4, keep=(0, 1), compression=compression),
                                   [2, 3, 4, 5], [1, 6], prefill_len=8):
            assert (attn.forward is not attn_fwd) is attn_wrapped, compression
            assert (merger.forward is not merger_fwd) is merger_wrapped, compression
            assert lm.forward is not lm_fwd
        assert attn.forward is attn_fwd and merger.forward is merger_fwd
        assert lm.forward is lm_fwd


# --------------------------------------------------------------------------- #
# The real kernels driven through the seam on a tiny fake model (torch only)
# --------------------------------------------------------------------------- #
class _WordTokenizer:
    """One id per word, with a leading begin-of-sequence id that the seam must strip."""

    bos_token_id = 0

    def __call__(self, words, is_split_into_words=True, return_tensors="pt", **kwargs):
        import torch

        ids = [self.bos_token_id] + [1 + (sum(map(ord, w)) % 15) for w in words]
        return {"input_ids": torch.tensor([ids])}


def _torch_model(torch, dim):
    model, tower, attn, merger, attn_fwd, merger_fwd, _ = _fake_tower_model()
    seen = {}

    def lm_forward(**kwargs):
        seen.update(kwargs)
        return "OUT"

    model.model.language_model = SimpleNamespace(forward=lm_forward)
    torch.manual_seed(0)
    embed = torch.nn.Embedding(16, dim)
    model.get_input_embeddings = lambda: embed
    return model, tower, embed, seen


def test_mmtok_through_the_seam_matches_the_numpy_reference():
    torch = pytest.importorskip("torch")
    nf, nt, dim, pre_dim = 3, 4, 6, 5
    n = nf * nt
    model, tower, embed, seen = _torch_model(torch, dim)
    g = torch.Generator().manual_seed(1)
    seq = 2 + n + 2                                   # text, START, n visual, END, text
    vis_pos = list(range(2, 2 + n))
    inputs_embeds = torch.randn(1, seq, dim, generator=g)
    pre_merger = torch.randn(n * 4, pre_dim, generator=g)
    question = "What colour is the car?"
    spec = _spec(pre=n, keep=tuple(range(3)), compression="mmtok",
                 params={"keep_ratio": 0.25, "expansion": 1.0, "token_budget": None})
    arm = {}
    with applied_pre_llm_patch(model, spec, vis_pos, [1, 2 + n], prefill_len=seq,
                               grid_thw=[[nf, 2, 2]], question=question,
                               tokenizer=_WordTokenizer(), arm_out=arm):
        tower.merger.forward(pre_merger)              # the tower runs first and is captured
        model.model.language_model.forward(
            inputs_embeds=inputs_embeds, cache_position=torch.arange(seq))

    # The same selection computed with the numpy reference, from the same three inputs.
    words = mmtok_extract_keywords(f"Question: {question}").split()
    ids = _WordTokenizer()(words)["input_ids"][0, 1:]             # begin-of-sequence stripped
    text = embed(ids).detach().numpy()
    video = inputs_embeds[0, vis_pos].numpy()
    img = pre_merger.view(n, 4, pre_dim).mean(dim=1).numpy()
    expected = greedy_max_coverage(mmtok_combined_np(text, video, img), 3)

    assert arm == {"pre_prune_visual": n, "post_prune_visual": 3}
    kept_positions = [p for p in seen["cache_position"].tolist() if p in vis_pos]
    assert kept_positions == [vis_pos[i] for i in expected]
    # Pure selection: the surviving visual rows are the original embeddings, unchanged.
    out_rows = seen["inputs_embeds"][0, 1:1 + 3]
    assert torch.equal(out_rows, inputs_embeds[0, kept_positions])
    # The markers at positions 1 and 2+n are gone, the text at both ends stays.
    assert seen["cache_position"].tolist() == [0] + kept_positions + [seq - 1]


def test_flashvid_through_the_seam_matches_a_direct_kernel_call(monkeypatch):
    torch = pytest.importorskip("torch")
    nf, nt, dim = 4, 8, 6
    n = nf * nt
    model, tower, embed, seen = _torch_model(torch, dim)
    g = torch.Generator().manual_seed(2)
    attention = torch.rand(nf, nt, generator=g)
    monkeypatch.setattr(prune_seam, "_recompute_vision_signals",
                        lambda *args: (attention, None))
    seq = 2 + n + 2
    vis_pos = list(range(2, 2 + n))
    inputs_embeds = torch.rand(1, seq, dim, generator=g)
    params = {"keep_ratio": 0.5, "expansion": 1.0, "alpha": 0.7, "segment_threshold": 0.9,
              "min_segment_num": 2, "temporal_threshold": 0.8, "do_segment": False,
              "complementary_segment": True}
    spec = _spec(pre=n, keep=tuple(range(n // 2)), compression="flashvid", params=params)
    arm = {}
    with applied_pre_llm_patch(model, spec, vis_pos, [1, 2 + n], prefill_len=seq,
                               grid_thw=[[nf, 2, 2]], arm_out=arm):
        tower.blocks[-1].attn.forward("hidden", cu_seqlens="CU", position_embeddings=("c", "s"))
        model.model.language_model.forward(
            inputs_embeds=inputs_embeds, cache_position=torch.arange(seq))

    cfg = FlashVidVisionConfig(retention_ratio=0.5, expansion=1.0, alpha=0.7,
                               do_segment=False, segment_threshold=0.9, min_segment_num=2,
                               temporal_threshold=0.8)
    tokens, idx = flashvid_compression(inputs_embeds[0, vis_pos].view(nf, nt, dim),
                                       attention, cfg)
    kept_positions = [vis_pos[i] for i in idx.tolist()]
    assert 0 < len(kept_positions) < n
    assert len(set(kept_positions)) == len(kept_positions)
    assert arm == {"pre_prune_visual": n, "post_prune_visual": len(kept_positions)}
    assert seen["cache_position"].tolist() == [0] + kept_positions + [seq - 1]
    # The merged token values (not the original embeddings) are what the LLM receives.
    assert torch.allclose(seen["inputs_embeds"][0, 1:1 + len(kept_positions)], tokens)


@pytest.mark.parametrize("params,expected_kept", [
    ({"keep_ratio": 0.5, "alpha": 0.75, "expansion": 1.0, "token_budget": None}, 16),
    # An absolute budget wins over the ratio.
    ({"keep_ratio": 0.5, "alpha": 0.75, "expansion": 1.0, "token_budget": 8}, 8),
])
def test_visionzip_through_the_seam_matches_a_direct_kernel_call(
        monkeypatch, params, expected_kept):
    torch = pytest.importorskip("torch")
    nf, nt, dim, key_dim = 4, 8, 6, 5
    n = nf * nt
    model, tower, embed, seen = _torch_model(torch, dim)
    g = torch.Generator().manual_seed(3)
    attention = torch.rand(nf, nt, generator=g)
    keys = torch.randn(nf, nt, key_dim, generator=g)
    monkeypatch.setattr(prune_seam, "_recompute_vision_signals",
                        lambda *args: (attention, keys))
    seq = 2 + n + 2
    vis_pos = list(range(2, 2 + n))
    inputs_embeds = torch.rand(1, seq, dim, generator=g)
    spec = _spec(pre=n, keep=tuple(range(expected_kept)), compression="visionzip",
                 params=params)
    arm = {}
    with applied_pre_llm_patch(model, spec, vis_pos, [1, 2 + n], prefill_len=seq,
                               grid_thw=[[nf, 2, 2]], arm_out=arm):
        tower.blocks[-1].attn.forward("hidden", cu_seqlens="CU", position_embeddings=("c", "s"))
        model.model.language_model.forward(
            inputs_embeds=inputs_embeds, cache_position=torch.arange(seq))

    tokens, idx = visionzip_compression(
        inputs_embeds[0, vis_pos].view(nf, nt, dim), attention, keys,
        retention_ratio=expected_kept / n, expansion=1.0, alpha=0.75)
    kept_positions = [vis_pos[i] for i in idx.tolist()]
    assert len(kept_positions) == expected_kept == spec.keep_count
    assert arm == {"pre_prune_visual": n, "post_prune_visual": expected_kept}
    assert seen["cache_position"].tolist() == [0] + kept_positions + [seq - 1]
    # The contextual rows carry merged values, so the LLM receives the kernel's tokens and
    # not a plain subset of the original embeddings.
    out_rows = seen["inputs_embeds"][0, 1:1 + expected_kept]
    assert torch.allclose(out_rows, tokens)
    assert not torch.allclose(out_rows, inputs_embeds[0, kept_positions])


def test_mmtok_through_the_seam_needs_tokenizer_and_model():
    torch = pytest.importorskip("torch")
    model, tower, embed, seen = _torch_model(torch, 4)
    spec = _spec(pre=4, keep=(0, 1), compression="mmtok")
    with pytest.raises(RuntimeError, match="requires the tokenizer"):
        with applied_pre_llm_patch(model, spec, [1, 2, 3, 4], [0, 5], prefill_len=6,
                                   grid_thw=[[1, 2, 2]], tokenizer=None):
            tower.merger.forward(torch.zeros(16, 3))
            model.model.language_model.forward(inputs_embeds=torch.zeros(1, 6, 4))


# --------------------------------------------------------------------------- #
# RandomPruner: determinism, keep-count arithmetic, patch declaration, accounting
# --------------------------------------------------------------------------- #
def test_seed_and_indices_are_deterministic():
    s1 = seed_from_query("what happens next?", 0)
    assert s1 == seed_from_query("what happens next?", 0)
    assert seed_from_query("other question", 0) != s1
    assert seed_from_query("what happens next?", 1) != s1
    idx1 = random_keep_indices(100, 25, s1)
    assert idx1 == random_keep_indices(100, 25, s1)
    assert len(idx1) == 25 and idx1 == tuple(sorted(set(idx1)))
    assert all(0 <= i < 100 for i in idx1)


def test_keep_count_ratio_and_budget_override():
    assert keep_count(100, Budget(), 0.25) == 25                   # ratio path
    assert keep_count(100, Budget(token_budget=30), 0.25) == 30    # token_budget overrides
    assert keep_count(100, Budget(token_budget=999), 0.25) == 100  # clamped to the input
    assert keep_count(3, Budget(), 0.1) == 1                       # never below 1
    assert keep_count(100, Budget(token_budget=0), 0.25) == 1      # never below 1
    with pytest.raises(ValueError, match="must be positive"):
        keep_count(0, Budget(), 0.25)


def test_keep_count_rounds_up():
    # Ceiling, not round-to-nearest: round(2.5) would give 2 on the first case.
    assert keep_count(10, Budget(), 0.25) == 3        # ceil(2.5)
    assert keep_count(7, Budget(), 0.1) == 1          # ceil(0.7)
    assert keep_count(9600, Budget(), 0.0625) == 600  # an exact ratio is unaffected


def _qwen_pkg(pre=120, grid_t=4):
    return {"kind": QWEN_TOKENS_KIND, "model_id": "qwen3-vl-4b",
            "num_visual_tokens": pre, "grid_thw": (grid_t, 10, 12)}


def test_random_pruner_pre_llm_spec_and_accounting():
    pruner = RandomPruner(keep_ratio=0.25, seed=7)
    vt = VisualTokens(tokens=_qwen_pkg(pre=100), num_tokens=100)
    out = pruner.prune(vt, "a query", Budget())
    spec = out.tokens[PATCH_SPEC_KEY]
    assert spec.method == "random"
    assert spec.patch_point == "pre_llm" and spec.compression == "random"
    assert spec.pre_prune_tokens == 100 and spec.keep_count == 25
    # keep_indices is the real seeded surviving set, not a placeholder.
    assert spec.keep_indices == random_keep_indices(100, 25, seed_from_query("a query", 7))
    assert spec.keep_indices != tuple(range(25))
    assert spec.params == {"keep_ratio": 0.25, "seed": 7, "token_budget": None}
    # The pruned sequence is what the language model prefills: num_tokens is the kept count.
    assert out.num_tokens == 25
    assert PATCH_SPEC_KEY not in vt.tokens  # the encoder's dict is not mutated
    assert out.tokens["num_visual_tokens"] == 100   # the rest of the package is carried over


def test_random_pruner_determinism_same_query():
    pruner = RandomPruner(keep_ratio=0.5, seed=1)
    a = pruner.prune(VisualTokens(tokens=_qwen_pkg(40), num_tokens=40), "q", Budget())
    b = pruner.prune(VisualTokens(tokens=_qwen_pkg(40), num_tokens=40), "q", Budget())
    c = pruner.prune(VisualTokens(tokens=_qwen_pkg(40), num_tokens=40), "another q", Budget())
    assert a.tokens[PATCH_SPEC_KEY].keep_indices == b.tokens[PATCH_SPEC_KEY].keep_indices
    assert a.tokens[PATCH_SPEC_KEY].keep_indices != c.tokens[PATCH_SPEC_KEY].keep_indices


def test_random_pruner_token_budget_overrides_ratio():
    out = RandomPruner().prune(VisualTokens(tokens=_qwen_pkg(100), num_tokens=100), "q",
                               Budget(token_budget=40))
    assert out.num_tokens == 40 and out.tokens[PATCH_SPEC_KEY].keep_count == 40
    assert out.tokens[PATCH_SPEC_KEY].params["token_budget"] == 40


def test_random_pruner_no_spec_when_keep_equals_pre():
    pruner = RandomPruner(keep_ratio=1.0, seed=0)
    out = pruner.prune(VisualTokens(tokens=_qwen_pkg(10), num_tokens=10), "q", Budget())
    assert out.num_tokens == 10 and PATCH_SPEC_KEY not in out.tokens


def test_random_pruner_direct_mode_subselects_rows():
    # On a materialized token list (no backend package) the rows are subselected directly.
    pruner = RandomPruner(keep_ratio=0.25, seed=3)
    out = pruner.prune(VisualTokens(tokens=list(range(100)), num_tokens=100), "q", Budget())
    assert out.num_tokens == 25 and len(out.tokens) == 25
    assert out.tokens == sorted(out.tokens) and all(0 <= v < 100 for v in out.tokens)
    assert out.tokens == list(random_keep_indices(100, 25, seed_from_query("q", 3)))


def test_random_pruner_zero_token_edge():
    out = RandomPruner().prune(VisualTokens(tokens=None, num_tokens=0), "q", Budget())
    assert out.num_tokens == 0


def test_random_pruner_rejects_bad_params():
    with pytest.raises(ValueError, match="keep_ratio"):
        RandomPruner(keep_ratio=0.0)
    with pytest.raises(ValueError, match="keep_ratio"):
        RandomPruner(keep_ratio=1.5)


def test_random_pruner_spec_drives_the_seam_to_the_same_keep_set():
    # Pruner -> PatchSpec -> seam: the tokens that reach the language model are exactly the
    # pruner's seeded draw.
    out = RandomPruner(keep_ratio=0.5, seed=5).prune(
        VisualTokens(tokens=_qwen_pkg(pre=8), num_tokens=8), "q", Budget())
    spec = out.tokens[PATCH_SPEC_KEY]
    seen = {}
    lm = SimpleNamespace(forward=lambda **kw: seen.update(kw))
    model = SimpleNamespace(model=SimpleNamespace(language_model=lm))
    vis_pos = list(range(2, 10))                      # text, START, 8 visual, END, text
    with applied_pre_llm_patch(model, spec, vis_pos, [1, 10], prefill_len=12):
        lm.forward(inputs_embeds=np.zeros((1, 12, 2)), cache_position=np.arange(12))
    assert seen["cache_position"].tolist() == [0] + [vis_pos[i] for i in spec.keep_indices] + [11]


def test_build_random_prune_wires_random_pruner():
    from longvideo_eval.models.build import build_pipeline

    orch = build_pipeline("qwen3-vl-4b", "random-prune", dry_run=True,
                          prune_kwargs={"keep_ratio": 0.5, "seed": 3})
    assert type(orch.pruner).__name__ == "RandomPruner"
    assert orch.pruner.keep_ratio == 0.5 and orch.pruner.seed == 3


# --------------------------------------------------------------------------- #
# The backend's run record must reach the result rows, so each row identifies its own run
# --------------------------------------------------------------------------- #
class _ArmOrchestrator:
    """Stub orchestrator returning a fixed Answer with (or without) a run record."""

    def __init__(self, arm):
        from longvideo_eval.interfaces import Answer, CostRecord

        trace = [{"arm": arm}] if arm is not None else []
        self._answer = Answer(text="A", cost=CostRecord(prefill_tokens=7),
                              rounds_trace=trace)

    def run(self, sample, budget):
        return self._answer


def test_evaluate_lifts_arm_from_rounds_trace():
    from longvideo_eval.interfaces import Sample
    from longvideo_eval.runners.evaluate import evaluate

    arm = {"model_id": "qwen3-vl-4b",
           "prune": {"method": "random", "pre_prune_visual": 100, "post_prune_visual": 25}}
    samples = [Sample(video_id="v1", video_path="p", query="q?")]
    results = evaluate(_ArmOrchestrator(arm), samples, Budget(), golds={"v1": "A"})
    assert results[0].arm == arm
    # No trace (the fakes emit none) means arm is None, never a KeyError.
    results = evaluate(_ArmOrchestrator(None), samples, Budget(), golds={"v1": "A"})
    assert results[0].arm is None


def test_write_results_persists_arm(tmp_path):
    import json

    from longvideo_eval.interfaces import Sample
    from longvideo_eval.runners.evaluate import evaluate
    from longvideo_eval.runners.results import write_results

    arm = {"enable_thinking": False, "presentation": "video",
           "prune": {"patch_point": "pre_llm", "method": "mmtok"}}
    samples = [Sample(video_id="v1", video_path="p", query="q?")]
    results = evaluate(_ArmOrchestrator(arm), samples, Budget(), golds={"v1": "A"})
    run_dir = write_results(
        results, str(tmp_path), {"task": "t", "method": "m", "setup": "s"}
    )
    row = json.loads((run_dir / "results.jsonl").read_text().strip())
    assert row["arm"] == arm                    # the row identifies its own run
    assert row["prefill_tokens"] == 7           # cost fields are still flattened alongside
