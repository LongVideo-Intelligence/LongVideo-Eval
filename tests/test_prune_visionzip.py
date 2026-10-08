"""Hardware-free tests for the VisionZip pruner and its compression kernel.

VisionZip keeps two kinds of visual tokens: the DOMINANT tokens that receive the most
attention in the vision tower, unchanged, and a few CONTEXTUAL tokens into which the remaining
tokens are averaged by the similarity of their attention key vectors. Covered here:

  * ``visionzip_plan_np``: the torch-free index plan (budget split, dominant set, contextual
    targets, merge assignment), checked against a reference written separately in this file;
  * ``VisionZipPruner``: the patch it declares for the pre-LLM prune point and its accounting;
  * how an absolute token budget reaches the kernel;
  * ``visionzip_compression``: the torch kernel that applies the plan to token values. These
    tests are skipped without torch.
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from longvideo_eval.backend.hf.pre_llm_compress import (  # noqa: E402
    visionzip_compression,
    visionzip_plan_np,
)
from longvideo_eval.backend.hf.prune_seam import PATCH_SPEC_KEY  # noqa: E402
from longvideo_eval.frontend.encode.encoder import QWEN_TOKENS_KIND  # noqa: E402
from longvideo_eval.frontend.prune.visionzip import VisionZipPruner  # noqa: E402
from longvideo_eval.interfaces import Budget, VisualTokens  # noqa: E402

DEFAULT_ALPHA = 0.928571            # 13 of every 14 kept tokens are dominant


# --------------------------------------------------------------------------- #
# visionzip_plan_np: checked against a separately written reference
# --------------------------------------------------------------------------- #
def _budget_split(n, retention, alpha, expansion=1.0):
    """(total, dominant, contextual) token counts for ``n`` input tokens."""
    total = max(1, int(round(n * retention * expansion)))
    dominant = max(1, int(round(total * alpha)))
    contextual = max(1, total - dominant)
    if dominant + contextual > n:
        dominant = min(dominant, n - 1)
        contextual = max(1, min(contextual, n - dominant))
    return total, dominant, contextual


def _plan_reference(attn, keys, retention, expansion, alpha):
    """The plan in plain Python loops: the most-attended tokens are dominant, the contextual
    targets are evenly spaced over the rest, and every other token goes to the target whose
    key vector is most similar (cosine)."""
    attn = [float(a) for a in np.asarray(attn).reshape(-1)]
    keys = np.asarray(keys, dtype=np.float64).reshape(len(attn), -1)
    n = len(attn)
    _, dominant_n, contextual_n = _budget_split(n, retention, alpha, expansion)

    dominant = sorted(range(n), key=lambda i: (-attn[i], i))[:dominant_n]
    rest = [i for i in range(n) if i not in set(dominant)]
    contextual_n = min(contextual_n, len(rest))
    step = max(1, len(rest) // contextual_n)
    target_slots = list(range(0, len(rest), step))[:contextual_n]
    targets = [rest[j] for j in target_slots]
    others = [rest[j] for j in range(len(rest)) if j not in set(target_slots)]

    def unit(v):
        return v / max(float(np.linalg.norm(v)), 1e-8)

    assign = []
    for o in others:
        sims = [float(unit(keys[o]) @ unit(keys[t])) for t in targets]
        assign.append(max(range(len(sims)), key=lambda j: (sims[j], -j)))   # first best
    return dominant, targets, others, assign, contextual_n


@pytest.mark.parametrize("retention,alpha", [(0.5, 0.7), (0.25, DEFAULT_ALPHA), (0.3, 0.5)])
def test_visionzip_plan_matches_the_reference_on_random_inputs(retention, alpha):
    rng = np.random.default_rng(3)
    for _ in range(8):
        n = int(rng.integers(8, 60))
        attn = rng.random(n)
        keys = rng.standard_normal((n, 5))
        dom, targets, others, assign, contextual_n = visionzip_plan_np(
            attn, keys, retention, 1.0, alpha)
        r_dom, r_targets, r_others, r_assign, r_contextual_n = _plan_reference(
            attn, keys, retention, 1.0, alpha)
        assert dom.tolist() == r_dom                     # most attended first
        assert targets.tolist() == r_targets
        assert others.tolist() == r_others
        assert assign.tolist() == r_assign
        assert contextual_n == r_contextual_n


def test_visionzip_plan_dominant_are_the_most_attended_tokens():
    attn = np.array([0.1, 0.9, 0.2, 0.8, 0.05, 0.7])
    # retention 0.5 of 6 tokens keeps 3; alpha=1 asks for all of them to be dominant.
    dom, _, _, _, _ = visionzip_plan_np(attn, np.eye(6), retention_ratio=0.5, expansion=1.0,
                                        alpha=1.0)
    assert sorted(dom.tolist()) == [1, 3, 5]
    assert dom.tolist() == [1, 3, 5]                     # in order of attention: .9, .8, .7


def test_visionzip_plan_breaks_attention_ties_by_the_earlier_token():
    dom, _, _, _, _ = visionzip_plan_np(np.ones(10), np.eye(10), 0.5, 1.0, 0.6)
    assert dom.tolist() == [0, 1, 2]                     # round(5 * 0.6) == 3


@pytest.mark.parametrize("n,retention,alpha", [
    (64, 0.25, DEFAULT_ALPHA), (256, 0.25, DEFAULT_ALPHA), (256, 0.0625, DEFAULT_ALPHA),
    (100, 0.5, 0.7), (37, 0.3, 0.5), (1000, 0.111, DEFAULT_ALPHA),
])
def test_visionzip_plan_splits_the_budget_between_dominant_and_contextual(n, retention, alpha):
    rng = np.random.default_rng(n)
    dom, targets, others, assign, contextual_n = visionzip_plan_np(
        rng.random(n), rng.standard_normal((n, 4)), retention, 1.0, alpha)
    total = round(n * retention)
    assert len(dom) == round(total * alpha)
    assert len(targets) == contextual_n == total - len(dom)
    assert len(dom) + len(targets) == total              # the kept count
    # Every token is exactly one of: dominant, a contextual target, or merged into a target.
    groups = dom.tolist() + targets.tolist() + others.tolist()
    assert sorted(groups) == list(range(n))
    assert len(assign) == len(others)
    assert assign.min() >= 0 and assign.max() < contextual_n


def test_visionzip_plan_spaces_the_contextual_targets_evenly():
    # 20 tokens, attention rising with the index: keep 10, of which 5 dominant (the last five)
    # and 5 contextual, spaced every 15 // 5 == 3 tokens over the other fifteen.
    dom, targets, others, _, contextual_n = visionzip_plan_np(
        np.arange(20, dtype=float), np.eye(20), 0.5, 1.0, 0.5)
    assert sorted(dom.tolist()) == [15, 16, 17, 18, 19]
    assert targets.tolist() == [0, 3, 6, 9, 12] and contextual_n == 5
    assert others.tolist() == [1, 2, 4, 5, 7, 8, 10, 11, 13, 14]


def test_visionzip_plan_assigns_each_token_to_the_most_similar_target():
    # Six tokens, none dominant except the most attended one (token 5). Targets are tokens 0
    # and 2; the keys of tokens 1, 3 and 4 point at one target or the other.
    attn = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.9])
    keys = np.array([[1.0, 0.0], [0.1, 1.0], [0.0, 1.0], [5.0, 0.2], [-1.0, 3.0], [1.0, 1.0]])
    dom, targets, others, assign, contextual_n = visionzip_plan_np(attn, keys, 0.5, 1.0, 0.34)
    assert dom.tolist() == [5]
    assert targets.tolist() == [0, 2] and contextual_n == 2
    assert others.tolist() == [1, 3, 4]
    assert assign.tolist() == [1, 0, 1]                  # 1 -> token 2, 3 -> token 0, 4 -> token 2


def test_visionzip_plan_is_scale_invariant_in_the_keys():
    # Cosine similarity: rescaling every key vector leaves the assignment as it was.
    rng = np.random.default_rng(5)
    attn, keys = rng.random(40), rng.standard_normal((40, 6))
    a = visionzip_plan_np(attn, keys, 0.5, 1.0, 0.6)
    b = visionzip_plan_np(attn, keys * rng.uniform(0.5, 4.0, size=(40, 1)), 0.5, 1.0, 0.6)
    for x, y in zip(a[:4], b[:4]):
        assert x.tolist() == y.tolist()


def test_visionzip_plan_accepts_per_frame_shaped_signals():
    # The seam hands over (frames, tokens) attention and (frames, tokens, dim) keys.
    rng = np.random.default_rng(9)
    attn, keys = rng.random((4, 8)), rng.standard_normal((4, 8, 3))
    shaped = visionzip_plan_np(attn, keys, 0.5, 1.0, 0.75)
    flat = visionzip_plan_np(attn.reshape(-1), keys.reshape(32, 3), 0.5, 1.0, 0.75)
    for x, y in zip(shaped[:4], flat[:4]):
        assert x.tolist() == y.tolist()


# --------------------------------------------------------------------------- #
# VisionZipPruner: the declared patch, accounting, guards
# --------------------------------------------------------------------------- #
def _qwen_pkg(pre=100):
    return {"kind": QWEN_TOKENS_KIND, "model_id": "qwen3-vl-4b",
            "num_visual_tokens": pre, "grid_thw": (4, 10, 12)}


def test_visionzip_builds_pre_llm_spec():
    pruner = VisionZipPruner(keep_ratio=0.25)
    vt = VisualTokens(tokens=_qwen_pkg(pre=100), num_tokens=100)
    out = pruner.prune(vt, "a query", Budget())
    spec = out.tokens[PATCH_SPEC_KEY]
    assert spec.method == "visionzip" and spec.patch_point == "pre_llm"
    assert spec.compression == "visionzip"
    assert spec.pre_prune_tokens == 100 and spec.keep_count == 25
    assert spec.params == {"keep_ratio": 0.25, "alpha": DEFAULT_ALPHA,
                           "token_budget": None, "expansion": 1.0}
    assert out.num_tokens == 25                 # the LLM prefill shrinks to the kept count
    assert out.cost.wall_seconds == 0.0         # the work happens (and is timed) in the backend
    assert PATCH_SPEC_KEY not in vt.tokens      # the input package is not mutated
    assert out.tokens["grid_thw"] == (4, 10, 12)


def test_visionzip_defaults():
    pruner = VisionZipPruner()
    assert pruner.keep_ratio == 0.25 and pruner.alpha == DEFAULT_ALPHA


@pytest.mark.parametrize("pre,ratio", [(100, 0.25), (5760, 0.25), (23040, 0.0625),
                                       (90, 0.25), (37, 0.3), (10, 0.25)])
def test_visionzip_target_count_is_the_rounded_ratio(pre, ratio):
    out = VisionZipPruner(keep_ratio=ratio).prune(
        VisualTokens(tokens=_qwen_pkg(pre), num_tokens=pre), "q", Budget())
    assert out.num_tokens == round(pre * ratio)
    assert out.tokens[PATCH_SPEC_KEY].keep_count == round(pre * ratio)


def test_visionzip_token_budget_overrides_ratio():
    out = VisionZipPruner().prune(VisualTokens(tokens=_qwen_pkg(100), num_tokens=100),
                                  "q", Budget(token_budget=40))
    spec = out.tokens[PATCH_SPEC_KEY]
    assert spec.keep_count == 40 and out.num_tokens == 40
    assert spec.params["token_budget"] == 40    # carried to the backend, which applies it


def test_visionzip_no_spec_when_keep_ge_pre():
    out = VisionZipPruner(keep_ratio=1.0).prune(
        VisualTokens(tokens=_qwen_pkg(10), num_tokens=10), "q", Budget())
    assert out.num_tokens == 10 and PATCH_SPEC_KEY not in out.tokens
    # A budget larger than the input keeps everything too.
    out = VisionZipPruner().prune(
        VisualTokens(tokens=_qwen_pkg(10), num_tokens=10), "q", Budget(token_budget=64))
    assert out.num_tokens == 10 and PATCH_SPEC_KEY not in out.tokens


def test_visionzip_keeps_at_least_one_token():
    out = VisionZipPruner(keep_ratio=0.01).prune(
        VisualTokens(tokens=_qwen_pkg(10), num_tokens=10), "q", Budget())
    assert out.num_tokens == 1


def test_visionzip_zero_token_edge():
    out = VisionZipPruner().prune(VisualTokens(tokens=None, num_tokens=0), "q", Budget())
    assert out.num_tokens == 0


def test_visionzip_rejects_non_qwen_package():
    with pytest.raises(NotImplementedError, match="seam-only"):
        VisionZipPruner().prune(VisualTokens(tokens=list(range(100)), num_tokens=100),
                                "q", Budget())
    with pytest.raises(NotImplementedError, match="seam-only"):
        VisionZipPruner().prune(VisualTokens(tokens={"kind": "other"}, num_tokens=100),
                                "q", Budget())


def test_visionzip_rejects_bad_params():
    for ratio in (0.0, 1.5, -0.1):
        with pytest.raises(ValueError, match="keep_ratio"):
            VisionZipPruner(keep_ratio=ratio)
    for alpha in (0.0, 1.5):
        with pytest.raises(ValueError, match="alpha"):
            VisionZipPruner(alpha=alpha)


def test_visionzip_ignores_the_question():
    def keep(query):
        out = VisionZipPruner().prune(
            VisualTokens(tokens=_qwen_pkg(100), num_tokens=100), query, Budget())
        return out.tokens[PATCH_SPEC_KEY]

    assert keep("first question") == keep("a completely different question")


def test_visionzip_registered_and_wired_by_build():
    from longvideo_eval import _registry
    from longvideo_eval.models.build import build_pipeline

    assert _registry.get("prune", "visionzip") is VisionZipPruner
    orch = build_pipeline("qwen3-vl-4b", "visionzip", dry_run=True)
    assert type(orch.pruner) is VisionZipPruner
    assert orch.pruner.alpha == DEFAULT_ALPHA


def test_build_prune_kwargs_reach_the_visionzip_pruner():
    from longvideo_eval.models.build import build_pipeline

    orch = build_pipeline("qwen3-vl-4b", "visionzip", dry_run=True,
                          prune_kwargs={"keep_ratio": 0.1, "alpha": 0.8})
    assert orch.pruner.keep_ratio == 0.1 and orch.pruner.alpha == 0.8
    out = orch.pruner.prune(
        VisualTokens(tokens=_qwen_pkg(pre=100), num_tokens=100), "q", Budget())
    assert out.tokens[PATCH_SPEC_KEY].keep_count == 10


# --------------------------------------------------------------------------- #
# An absolute token budget reaches the kernel as a retention ratio
# --------------------------------------------------------------------------- #
class _Features:
    shape = (4, 64, 8)                          # 4 frames x 64 tokens = 256 tokens


def _dispatch(monkeypatch, params, captured=None):
    """Run the seam's dispatch with the kernel replaced by a recorder."""
    from longvideo_eval.backend.hf import pre_llm_compress as C
    from longvideo_eval.backend.hf import prune_seam

    seen = {}

    def fake(video_features, attn_logits, attn_keys, retention_ratio, expansion, alpha):
        seen.update(features=video_features, attention=attn_logits, keys=attn_keys,
                    ratio=retention_ratio, expansion=expansion, alpha=alpha)
        return video_features, [0]

    monkeypatch.setattr(C, "visionzip_compression", fake)
    captured = captured or {"cls_attention": "ATTENTION", "attn_keys": "KEYS"}
    feats = _Features()
    out = prune_seam._run_pre_llm_compression("visionzip", feats, captured, params)
    assert out == (feats, [0])                  # the kernel's result is returned as is
    assert seen["features"] is feats
    return seen


def test_visionzip_token_budget_reaches_the_kernel_as_a_ratio(monkeypatch):
    """An absolute token budget is turned into the retention ratio that yields it."""
    seen = _dispatch(monkeypatch, {"keep_ratio": 0.25, "token_budget": 16})
    assert seen["ratio"] == 16 / 256
    assert round(256 * seen["ratio"] * seen["expansion"]) == 16


def test_visionzip_without_a_token_budget_the_kernel_gets_the_keep_ratio(monkeypatch):
    seen = _dispatch(monkeypatch, {"keep_ratio": 0.125, "token_budget": None})
    assert seen["ratio"] == 0.125 and seen["expansion"] == 1.0


def test_visionzip_token_budget_ratio_is_capped_at_one(monkeypatch):
    seen = _dispatch(monkeypatch, {"keep_ratio": 0.25, "token_budget": 4096})
    assert seen["ratio"] == 1.0


def test_visionzip_token_budget_accounts_for_the_expansion(monkeypatch):
    seen = _dispatch(monkeypatch, {"keep_ratio": 0.25, "token_budget": 16, "expansion": 1.25})
    assert seen["expansion"] == 1.25
    assert round(256 * seen["ratio"] * seen["expansion"]) == 16


def test_visionzip_dispatch_hands_over_both_signals_and_alpha(monkeypatch):
    seen = _dispatch(monkeypatch, {"keep_ratio": 0.25, "alpha": 0.8})
    assert (seen["attention"], seen["keys"], seen["alpha"]) == ("ATTENTION", "KEYS", 0.8)
    # Without an explicit alpha the kernel gets the default dominant share.
    assert _dispatch(monkeypatch, {"keep_ratio": 0.25})["alpha"] == DEFAULT_ALPHA


def test_visionzip_dispatch_needs_the_key_vectors(monkeypatch):
    # The attention alone is not enough: the contextual merge reads the key vectors.
    with pytest.raises(KeyError, match="attn_keys"):
        _dispatch(monkeypatch, {"keep_ratio": 0.25}, captured={"cls_attention": "ATTENTION"})


def test_pruner_budget_and_dispatch_agree_on_the_kept_count(monkeypatch):
    # The count the pruner reports is the count the kernel is asked to keep.
    out = VisionZipPruner().prune(VisualTokens(tokens=_qwen_pkg(256), num_tokens=256),
                                  "q", Budget(token_budget=40))
    spec = out.tokens[PATCH_SPEC_KEY]
    seen = _dispatch(monkeypatch, spec.params)
    assert round(256 * seen["ratio"] * seen["expansion"]) == out.num_tokens == 40


# --------------------------------------------------------------------------- #
# torch kernel (runs only where torch is installed)
# --------------------------------------------------------------------------- #
def _video(torch, nf=8, nt=16, d=12, dk=6, seed=0):
    """Positive features, so the mean of any merged group is non-zero."""
    g = torch.Generator().manual_seed(seed)
    feats = torch.rand(nf, nt, d, generator=g) + 0.1
    attn = torch.rand(nf, nt, generator=g)
    keys = torch.randn(nf, nt, dk, generator=g)
    return feats, attn, keys


# (frames, tokens per frame, retention, alpha): every case leaves at least one contextual slot.
KERNEL_CASES = [
    (8, 16, 0.25, DEFAULT_ALPHA), (8, 32, 0.25, DEFAULT_ALPHA), (16, 16, 0.0625, DEFAULT_ALPHA),
    (4, 25, 0.5, 0.7), (3, 13, 0.3, 0.5), (5, 20, 0.111, 0.8),
]


@pytest.mark.parametrize("nf,nt,retention,alpha", KERNEL_CASES)
def test_visionzip_kernel_keeps_the_rounded_share_of_the_tokens(nf, nt, retention, alpha):
    torch = pytest.importorskip("torch")
    feats, attn, keys = _video(torch, nf=nf, nt=nt, seed=nf * nt)
    n, d = nf * nt, feats.shape[-1]
    tokens, idx = visionzip_compression(feats.clone(), attn, keys, retention, 1.0, alpha)
    idx_list = idx.tolist()
    total = round(n * retention)
    assert len(idx_list) == total                        # kept count
    assert tokens.shape == (total, d)                    # one row per kept index
    assert idx_list == sorted(set(idx_list))             # ascending, no token kept twice
    assert 0 <= idx_list[0] and idx_list[-1] < n
    # The same count the pruner reports for this input.
    out = VisionZipPruner(keep_ratio=retention, alpha=alpha).prune(
        VisualTokens(tokens=_qwen_pkg(n), num_tokens=n), "q", Budget())
    assert out.num_tokens == total


@pytest.mark.parametrize("nf,nt,retention,alpha", KERNEL_CASES)
def test_visionzip_kernel_indices_are_the_numpy_plan(nf, nt, retention, alpha):
    torch = pytest.importorskip("torch")
    feats, attn, keys = _video(torch, nf=nf, nt=nt, seed=1 + nf * nt)
    _, idx = visionzip_compression(feats.clone(), attn, keys, retention, 1.0, alpha)
    dom, targets, _, _, _ = visionzip_plan_np(attn.numpy(), keys.numpy(), retention, 1.0, alpha)
    assert len(dom) == round(round(nf * nt * retention) * alpha)     # dominant count
    assert idx.tolist() == sorted(dom.tolist() + targets.tolist())


@pytest.mark.parametrize("nf,nt,retention,alpha", KERNEL_CASES)
def test_visionzip_kernel_keeps_dominant_rows_and_merges_into_contextual_rows(
        nf, nt, retention, alpha):
    torch = pytest.importorskip("torch")
    feats, attn, keys = _video(torch, nf=nf, nt=nt, seed=2 + nf * nt)
    n, d = nf * nt, feats.shape[-1]
    original = feats.clone()
    flat = original.reshape(n, d)
    tokens, idx = visionzip_compression(feats, attn, keys, retention, 1.0, alpha)
    assert torch.equal(feats, original)                  # the input is left as it was

    dom, targets, others, assign, _ = _plan_reference(
        attn.numpy(), keys.numpy(), retention, 1.0, alpha)
    row_of = {token: row for row, token in enumerate(idx.tolist())}
    assert set(row_of) == set(dom) | set(targets)

    # Dominant tokens come out exactly as they went in.
    for token in dom:
        assert torch.equal(tokens[row_of[token]], flat[token])

    # Each contextual row is its target plus the mean of the tokens assigned to it.
    assert others, "the case must leave tokens to merge"
    merged_into = {slot: [] for slot in range(len(targets))}
    for token, slot in zip(others, assign):
        merged_into[slot].append(token)
    assert any(merged_into.values())
    for slot, target in enumerate(targets):
        row = tokens[row_of[target]]
        group = merged_into[slot]
        if group:
            assert torch.allclose(row, flat[target] + flat[group].mean(dim=0), atol=1e-5)
            assert not torch.allclose(row, flat[target])         # the value has changed
        else:
            assert torch.equal(row, flat[target])                # nothing was assigned to it


def test_visionzip_kernel_folds_every_dropped_token_into_the_output():
    # With a single contextual token, everything that is not dominant is averaged into it.
    torch = pytest.importorskip("torch")
    feats, attn, keys = _video(torch, nf=2, nt=10, seed=7)
    flat = feats.reshape(20, -1).clone()
    tokens, idx = visionzip_compression(feats, attn, keys, 0.5, 1.0, 0.9)   # 9 dominant + 1
    dom = set(torch.topk(attn.reshape(-1), 9).indices.tolist())
    rest = [i for i in range(20) if i not in dom]
    assert sorted(idx.tolist()) == sorted(dom | {rest[0]})
    row = tokens[idx.tolist().index(rest[0])]
    assert torch.allclose(row, flat[rest[0]] + flat[rest[1:]].mean(dim=0), atol=1e-5)


def test_visionzip_kernel_keeps_everything_unchanged_at_full_retention():
    torch = pytest.importorskip("torch")
    feats, attn, keys = _video(torch, nf=2, nt=8, seed=4)
    tokens, idx = visionzip_compression(feats.clone(), attn, keys, 1.0, 1.0, 0.75)
    assert idx.tolist() == list(range(16))
    assert torch.equal(tokens, feats.reshape(16, -1))    # nothing left to merge


def test_visionzip_kernel_is_deterministic():
    torch = pytest.importorskip("torch")
    feats, attn, keys = _video(torch, seed=3)
    a_tok, a_idx = visionzip_compression(feats.clone(), attn, keys, 0.25, 1.0, DEFAULT_ALPHA)
    b_tok, b_idx = visionzip_compression(feats.clone(), attn, keys, 0.25, 1.0, DEFAULT_ALPHA)
    assert a_idx.tolist() == b_idx.tolist()
    assert torch.equal(a_tok, b_tok)


def test_visionzip_kernel_keeps_fewer_tokens_at_a_lower_retention():
    torch = pytest.importorskip("torch")
    feats, attn, keys = _video(torch, nf=8, nt=32, seed=5)
    _, small = visionzip_compression(feats.clone(), attn, keys, 0.125, 1.0, DEFAULT_ALPHA)
    _, large = visionzip_compression(feats.clone(), attn, keys, 0.5, 1.0, DEFAULT_ALPHA)
    assert len(small) == 32 and len(large) == 128
    # The dominant set grows with the budget: every dominant token of the small run is one
    # of the most attended tokens of the large run too.
    small_dom = set(visionzip_plan_np(attn.numpy(), keys.numpy(), 0.125, 1.0, DEFAULT_ALPHA)[0]
                    .tolist())
    large_dom = set(visionzip_plan_np(attn.numpy(), keys.numpy(), 0.5, 1.0, DEFAULT_ALPHA)[0]
                    .tolist())
    assert small_dom <= large_dom


def test_visionzip_kernel_preserves_dtype():
    torch = pytest.importorskip("torch")
    feats, attn, keys = _video(torch, nf=4, nt=16, seed=6)
    tokens, idx = visionzip_compression(feats.half(), attn.half(), keys.half(),
                                        0.25, 1.0, DEFAULT_ALPHA)
    assert tokens.dtype == torch.float16 and idx.dtype == torch.long
