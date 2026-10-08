"""The lazy greedy maximiser must pick the same tokens as the textbook greedy loop.

MMTok selects tokens by greedy maximum coverage. The textbook loop re-scores every column at
every step; the lazy version re-scores only the columns whose stale upper bound could still
win. That is a speed-up only: the selection must not change. These tests compare the two

  * directly, on the coverage matrices, for several shapes, seeds and keep ratios, including
    a video whose consecutive frames are near-duplicates;
  * through ``mmtok_compression``, on both the normal and the low-memory (half precision)
    branch;

and check ``mmtok_compression`` against the torch-free numpy reference.

Everything here needs torch and is skipped without it. Sizes are small: the whole file runs
in a few seconds on a CPU.
"""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from longvideo_eval.backend.hf import pre_llm_compress as C  # noqa: E402
from longvideo_eval.backend.hf.pre_llm_compress import (  # noqa: E402
    _plain_greedy,
    greedy_max_coverage,
    lazy_greedy_max_coverage,
    mmtok_combined_np,
    mmtok_compression,
)

KEEP_RATIOS = (1 / 16, 1 / 4, 1 / 2, 1.0)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Each test starts from the documented defaults, whatever the caller's shell exports."""
    for name in ("MMTok_GREEDY", "MMTok_LOWMEM_N_THRESHOLD", "MMTok_LOWMEM_SOFTMAX_CHUNK"):
        monkeypatch.delenv(name, raising=False)


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #
def _random_video(nf, nt, d, d_img, m, seed):
    """Unstructured features: (video [nf, nt, d], image [nf, nt, d_img], text [m, d])."""
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(nf, nt, d, generator=g),
            torch.randn(nf, nt, d_img, generator=g),
            torch.randn(m, d, generator=g))


def _clustered_video(nf, nt, d, d_img, m, seed, run=4, noise=0.01):
    """A video made of short runs of near-duplicate frames.

    Every ``run`` consecutive frames share one base frame plus a little noise, as a static
    shot does. Many columns then have almost the same coverage, which is where a lazy
    evaluation is most likely to go wrong (stale bounds that are nearly tied).
    """
    g = torch.Generator().manual_seed(seed)
    n_base = math.ceil(nf / run)
    base_v = torch.randn(n_base, nt, d, generator=g)
    base_i = torch.randn(n_base, nt, d_img, generator=g)
    owner = torch.arange(nf) // run
    video = base_v[owner] + noise * torch.randn(nf, nt, d, generator=g)
    img = base_i[owner] + noise * torch.randn(nf, nt, d_img, generator=g)
    return video, img, torch.randn(m, d, generator=g)


def _parts(video, img, text, alpha=0.5, tv_temp=0.01, vv_temp=0.2):
    """The two coverage blocks ``[P, alpha * Q]`` in single precision, built here from the
    definition (text->video softmax over m, image->image softmax over n)."""
    d = video.shape[-1]
    x = torch.nn.functional.normalize(video.reshape(-1, d), dim=-1)
    xc = torch.nn.functional.normalize(img.reshape(x.shape[0], -1), dim=-1)
    z = torch.nn.functional.normalize(text, dim=-1)
    m, n = z.shape[0], x.shape[0]
    P = torch.softmax((z @ x.T) / tv_temp, dim=1) / m
    Q = torch.softmax((xc @ xc.T) / vv_temp, dim=1) * (alpha / n)
    return [P, Q]


def _k(n, ratio):
    return max(1, math.ceil(n * ratio))


# name -> (builder, nf, nt, d, d_img, m, seed)
CASES = {
    "random-small": (_random_video, 4, 16, 16, 12, 3, 0),
    "random-wide": (_random_video, 6, 32, 24, 16, 5, 1),
    "random-one-text-token": (_random_video, 8, 24, 8, 8, 1, 2),
    "clustered-runs-of-4": (_clustered_video, 16, 16, 16, 12, 4, 3),
    "clustered-runs-of-4-b": (_clustered_video, 12, 24, 24, 16, 2, 4),
}


def _build(name):
    builder, nf, nt, d, d_img, m, seed = CASES[name]
    return builder(nf, nt, d, d_img, m, seed)


# --------------------------------------------------------------------------- #
# lazy vs plain, directly on the coverage matrices
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("ratio", KEEP_RATIOS)
def test_lazy_selects_the_same_columns_in_the_same_order_as_plain(case, ratio):
    parts = _parts(*_build(case))
    n = parts[0].shape[1]
    k = _k(n, ratio)
    plain = _plain_greedy(parts, k).tolist()
    lazy = lazy_greedy_max_coverage(parts, k).tolist()
    assert len(plain) == k and len(set(plain)) == k       # k distinct columns
    assert lazy == plain                                  # same picks, same order


@pytest.mark.parametrize("batch", [1, 2, 7, 64, 10_000])
def test_lazy_result_does_not_depend_on_the_refresh_batch(batch):
    parts = _parts(*_build("clustered-runs-of-4"))
    n = parts[0].shape[1]
    k = _k(n, 1 / 4)
    expected = _plain_greedy(parts, k).tolist()
    assert lazy_greedy_max_coverage(parts, k, batch=batch).tolist() == expected


def test_lazy_result_does_not_depend_on_the_column_chunk():
    parts = _parts(*_build("random-wide"))
    k = _k(parts[0].shape[1], 1 / 4)
    expected = _plain_greedy(parts, k).tolist()
    assert lazy_greedy_max_coverage(parts, k, col_chunk=5).tolist() == expected
    assert _plain_greedy(parts, k, col_chunk=5).tolist() == expected


@pytest.mark.parametrize("seed", range(6))
def test_lazy_matches_plain_on_generic_non_negative_matrices(seed):
    # Not only on softmax-shaped inputs: any non-negative blocks sharing their column axis.
    g = torch.Generator().manual_seed(100 + seed)
    n = 40 + 7 * seed
    parts = [torch.rand(3 + seed, n, generator=g), torch.rand(n, n, generator=g) * 0.1]
    for k in (1, n // 8, n // 2, n):
        assert lazy_greedy_max_coverage(parts, k).tolist() == _plain_greedy(parts, k).tolist()


def test_both_greedy_versions_break_ties_towards_the_lowest_index():
    # Columns 0/1 are identical and so are columns 2/3: exact ties at every step.
    block = torch.tensor([[1.0, 1.0, 0.0, 0.0],
                          [0.0, 0.0, 0.5, 0.5]])
    for fn in (lazy_greedy_max_coverage, _plain_greedy):
        assert fn([block], 2).tolist() == [0, 2]
        assert fn([block], 4).tolist() == [0, 2, 1, 3]


def test_greedy_picks_by_marginal_gain_not_by_column_total():
    # Column 2 has the second-largest total but only re-covers row 0; column 1 adds new rows.
    block = torch.tensor([[10.0, 0.0, 9.0],
                          [0.0, 1.0, 0.0]])
    for fn in (lazy_greedy_max_coverage, _plain_greedy):
        assert fn([block], 2).tolist() == [0, 1]


def test_greedy_clamps_k_and_handles_zero():
    block = torch.eye(4)
    for fn in (lazy_greedy_max_coverage, _plain_greedy):
        assert fn([block], 0).tolist() == []
        assert fn([block], -3).tolist() == []
        assert sorted(fn([block], 99).tolist()) == [0, 1, 2, 3]     # k > n clamps to n
        assert fn([block], 2).dtype == torch.long


def test_greedy_does_not_modify_its_inputs():
    parts = _parts(*_build("random-small"))
    before = [p.clone() for p in parts]
    lazy_greedy_max_coverage(parts, 8)
    _plain_greedy(parts, 8)
    assert all(torch.equal(a, b) for a, b in zip(parts, before))


def test_lazy_evaluates_far_fewer_column_gains_than_plain(monkeypatch):
    # The point of the lazy version. Count how many column gains each maximiser computes.
    parts = _parts(*_build("clustered-runs-of-4"))
    n = parts[0].shape[1]
    k = _k(n, 1 / 4)
    real = C._column_gains
    counted = {"cols": 0}

    def counting(M, best, cols, chunk=4096):
        counted["cols"] += int(cols.shape[0])
        return real(M, best, cols, chunk)

    monkeypatch.setattr(C, "_column_gains", counting)
    plain = _plain_greedy(parts, k).tolist()
    plain_cols, counted["cols"] = counted["cols"], 0
    lazy = lazy_greedy_max_coverage(parts, k, batch=8).tolist()
    lazy_cols = counted["cols"]

    assert lazy == plain
    assert plain_cols == len(parts) * k * n            # every column, every step, every block
    assert lazy_cols < plain_cols / 2


# --------------------------------------------------------------------------- #
# lazy vs plain through mmtok_compression, normal and low-memory branches
# --------------------------------------------------------------------------- #
def _compress(case, ratio, greedy):
    video, img, text = _build(case)
    n = video.shape[0] * video.shape[1]
    feats, selected = mmtok_compression(video, img, text, target_vision_tokens=_k(n, ratio),
                                        greedy=greedy)
    return video, feats, selected


@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("ratio", KEEP_RATIOS)
def test_mmtok_compression_lazy_equals_plain(case, ratio):
    video, lazy_feats, lazy_sel = _compress(case, ratio, "lazy")
    _, plain_feats, plain_sel = _compress(case, ratio, "plain")
    n, d = video.shape[0] * video.shape[1], video.shape[2]
    k = _k(n, ratio)
    assert lazy_sel.tolist() == plain_sel.tolist()
    assert torch.equal(lazy_feats, plain_feats)
    # Shape and ordering of the result.
    sel = lazy_sel.tolist()
    assert len(sel) == k and sel == sorted(set(sel))
    assert sel[0] >= 0 and sel[-1] < n
    # Pure selection: the returned rows are the input tokens, unchanged.
    assert torch.equal(lazy_feats, video.reshape(n, d)[lazy_sel])
    if ratio == 1.0:
        assert sel == list(range(n))


@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("ratio", KEEP_RATIOS)
def test_mmtok_compression_lazy_equals_plain_on_the_low_memory_branch(
        monkeypatch, case, ratio):
    # Any video above the threshold builds Q in half precision, softmaxed in row chunks.
    monkeypatch.setenv("MMTok_LOWMEM_N_THRESHOLD", "8")
    monkeypatch.setenv("MMTok_LOWMEM_SOFTMAX_CHUNK", "50")     # several chunks per matrix
    video, lazy_feats, lazy_sel = _compress(case, ratio, "lazy")
    _, plain_feats, plain_sel = _compress(case, ratio, "plain")
    n = video.shape[0] * video.shape[1]
    assert lazy_sel.tolist() == plain_sel.tolist()
    assert torch.equal(lazy_feats, plain_feats)
    sel = lazy_sel.tolist()
    assert len(sel) == _k(n, ratio) and sel == sorted(set(sel))


def test_low_memory_threshold_switches_the_precision_of_q(monkeypatch):
    # Confirms the two tests above really take different branches.
    video, img, text = _build("random-small")
    seen = []

    def spy(parts, k_max, *args, **kwargs):
        seen.append([p.dtype for p in parts])
        return torch.arange(int(k_max))

    monkeypatch.setattr(C, "lazy_greedy_max_coverage", spy)
    mmtok_compression(video, img, text, target_vision_tokens=4, greedy="lazy")
    monkeypatch.setenv("MMTok_LOWMEM_N_THRESHOLD", "8")
    mmtok_compression(video, img, text, target_vision_tokens=4, greedy="lazy")
    assert seen == [[torch.float32, torch.float32], [torch.float32, torch.float16]]


def test_low_memory_softmax_chunk_size_does_not_change_the_selection(monkeypatch):
    video, img, text = _build("clustered-runs-of-4")
    monkeypatch.setenv("MMTok_LOWMEM_N_THRESHOLD", "8")
    picks = []
    for chunk in ("7", "64", "100000"):
        monkeypatch.setenv("MMTok_LOWMEM_SOFTMAX_CHUNK", chunk)
        picks.append(mmtok_compression(video, img, text, target_vision_tokens=32)[1].tolist())
    assert picks[0] == picks[1] == picks[2]


# --------------------------------------------------------------------------- #
# choosing the maximiser
# --------------------------------------------------------------------------- #
def _spy_on_maximisers(monkeypatch):
    used = []

    def lazy(parts, k_max, *args, **kwargs):
        used.append("lazy")
        return torch.arange(int(k_max))

    def plain(parts, k_max, *args, **kwargs):
        used.append("plain")
        return torch.arange(int(k_max))

    monkeypatch.setattr(C, "lazy_greedy_max_coverage", lazy)
    monkeypatch.setattr(C, "_plain_greedy", plain)
    return used


def test_default_maximiser_is_lazy(monkeypatch):
    used = _spy_on_maximisers(monkeypatch)
    video, img, text = _build("random-small")
    mmtok_compression(video, img, text, target_vision_tokens=4)
    assert used == ["lazy"]


def test_environment_variable_overrides_the_default_maximiser(monkeypatch):
    used = _spy_on_maximisers(monkeypatch)
    video, img, text = _build("random-small")
    monkeypatch.setenv("MMTok_GREEDY", "plain")
    mmtok_compression(video, img, text, target_vision_tokens=4)
    monkeypatch.setenv("MMTok_GREEDY", "LAZY")                 # case-insensitive
    mmtok_compression(video, img, text, target_vision_tokens=4)
    assert used == ["plain", "lazy"]


def test_explicit_argument_wins_over_the_environment_variable(monkeypatch):
    used = _spy_on_maximisers(monkeypatch)
    video, img, text = _build("random-small")
    monkeypatch.setenv("MMTok_GREEDY", "plain")
    mmtok_compression(video, img, text, target_vision_tokens=4, greedy="lazy")
    monkeypatch.setenv("MMTok_GREEDY", "lazy")
    mmtok_compression(video, img, text, target_vision_tokens=4, greedy="plain")
    assert used == ["lazy", "plain"]


def test_unknown_maximiser_is_rejected(monkeypatch):
    video, img, text = _build("random-small")
    with pytest.raises(ValueError, match="greedy must be 'lazy' or 'plain'"):
        mmtok_compression(video, img, text, target_vision_tokens=4, greedy="fastest")
    monkeypatch.setenv("MMTok_GREEDY", "fastest")
    with pytest.raises(ValueError, match="greedy must be 'lazy' or 'plain'"):
        mmtok_compression(video, img, text, target_vision_tokens=4)


# --------------------------------------------------------------------------- #
# torch kernel vs the numpy reference
# --------------------------------------------------------------------------- #
def _numpy_reference(video, img, text, k):
    n = video.shape[0] * video.shape[1]
    combined = mmtok_combined_np(text.numpy(), video.reshape(n, -1).numpy(),
                                 img.reshape(n, -1).numpy())
    return list(greedy_max_coverage(combined, k))


@pytest.mark.parametrize("greedy", ["lazy", "plain"])
@pytest.mark.parametrize("case", ["random-small", "random-one-text-token"])
@pytest.mark.parametrize("ratio", [1 / 16, 1 / 4, 1 / 2])
def test_mmtok_compression_agrees_with_the_numpy_reference(greedy, case, ratio):
    video, img, text = _build(case)
    n = video.shape[0] * video.shape[1]
    k = _k(n, ratio)
    _, selected = mmtok_compression(video, img, text, target_vision_tokens=k, greedy=greedy)
    assert selected.tolist() == _numpy_reference(video, img, text, k)


def test_mmtok_compression_honours_alpha_and_temperatures_like_the_reference():
    video, img, text = _build("random-small")
    n = video.shape[0] * video.shape[1]
    k = 12
    for alpha, tv_temp, vv_temp in ((0.5, 0.01, 0.2), (2.0, 0.05, 0.1), (0.05, 0.5, 1.0)):
        _, selected = mmtok_compression(video, img, text, target_vision_tokens=k,
                                        alpha=alpha, tv_temp=tv_temp, vv_temp=vv_temp)
        combined = mmtok_combined_np(text.numpy(), video.reshape(n, -1).numpy(),
                                     img.reshape(n, -1).numpy(),
                                     alpha=alpha, tv_temp=tv_temp, vv_temp=vv_temp)
        assert selected.tolist() == list(greedy_max_coverage(combined, k))


def test_mmtok_compression_text_covered_token_is_selected_first():
    # The text points at token 5 and the image term is switched almost off.
    video = torch.eye(8).reshape(2, 4, 8)
    img = torch.eye(8).reshape(2, 4, 8)
    text = torch.zeros(1, 8)
    text[0, 5] = 1.0
    for greedy in ("lazy", "plain"):
        _, selected = mmtok_compression(video, img, text, target_vision_tokens=1,
                                        alpha=0.001, greedy=greedy)
        assert selected.tolist() == [5]


def test_mmtok_compression_clamps_the_target_to_the_token_count():
    video, img, text = _build("random-small")
    n = video.shape[0] * video.shape[1]
    feats, selected = mmtok_compression(video, img, text, target_vision_tokens=10 * n)
    assert selected.tolist() == list(range(n))
    assert feats.shape == (n, video.shape[2])
