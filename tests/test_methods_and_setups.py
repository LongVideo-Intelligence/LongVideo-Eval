"""Tests for the token-pruning and dual-stream setups and for the methods that use them.

Two things are checked without any hardware:

  * the named setups carry the documented allocation;
  * ``build_pipeline(..., dry_run=True)`` assembles each new method and, where the fakes allow
    it, runs a sample end to end with sensible cost accounting.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pytest

from longvideo_eval.__main__ import main
from longvideo_eval.interfaces import Budget, Sample
from longvideo_eval.models.build import METHODS, build_pipeline
from longvideo_eval.runners.setups import SETUPS, get_setup
from longvideo_eval.testing import FakeEncoder

MODEL = "qwen3-vl-4b"
SAMPLE = Sample(video_id="v1", video_path="/fake/v1.mp4", query="What is the man doing?",
                choices=["cooking", "running", "reading", "sleeping"])

# name -> (N, r_l, K)
LOHI_SETUPS = {
    "lohi_128f_r25_k8": (128, 0.25, 8),
    "lohi_128f_r25_k4": (128, 0.25, 4),
}


# --------------------------------------------------------------------------- #
# setups
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", sorted(LOHI_SETUPS))
def test_lohi_setups_carry_the_documented_allocation(name):
    n, r_l, k = LOHI_SETUPS[name]
    b = get_setup(name)
    assert b.presentation == "lohi"
    assert (b.frame_count, b.resolution, b.hi_i_count) == (n, r_l, k)
    # The image scale is not fixed: the images fill the budget of 16 native frames per video.
    assert b.hi_resolution is None and b.reference_frames == 16
    assert b.decode_budget == n                    # every decoded frame is used
    assert b.hi_i_count <= b.frame_count           # the K frames are a subset of the N
    assert b.token_budget is None                  # the realized count is metered, not pinned
    assert b.max_rounds == 0


def test_lohi_setup_names_spell_out_their_allocation():
    for name, (n, r_l, k) in LOHI_SETUPS.items():
        assert name == f"lohi_{n}f_r{round(r_l * 100)}_k{k}"


def test_lohi_video_stream_is_half_the_reference_budget():
    # With G the tokens of one native frame: N video frames at scale r cost (N / 2) * r^2 * G
    # (two frames share a temporal position). 16 native frames cost 8 G; the low-resolution
    # video costs 4 G and leaves the other half to the images.
    for name in LOHI_SETUPS:
        b = get_setup(name)
        assert math.isclose((b.frame_count / 2) * b.resolution ** 2, 4.0)
        assert b.reference_frames / 2 == 8.0


@pytest.mark.parametrize("name,frames", [("native_64f", 64), ("native_256f", 256)])
def test_pruning_source_setups_decode_every_frame_they_use(name, frames):
    b = get_setup(name)
    assert b.frame_count == frames
    assert b.decode_budget == b.frame_count        # no over-decode
    assert b.resolution == 1.0                     # native resolution
    assert b.presentation == "video"
    assert b.hi_i_count is None and b.hi_resolution is None
    assert b.token_budget is None                  # a pruner's keep_ratio sets the budget


def test_pruning_source_setups_land_on_16_native_frames_at_the_documented_ratios():
    # 64 frames keeping 1/4, or 256 frames keeping 1/16, both end at 16 frames' worth of tokens.
    assert get_setup("native_64f").frame_count * (1 / 4) == 16
    assert get_setup("native_256f").frame_count * (1 / 16) == 16


def test_only_the_lohi_setups_use_the_lohi_presentation():
    for name, b in SETUPS.items():
        expected = "lohi" if name in LOHI_SETUPS else "video"
        assert b.presentation == expected, name
        if expected == "video":
            assert b.hi_i_count is None and b.hi_resolution is None, name


def test_budget_dual_stream_fields_default_to_single_stream():
    b = Budget()
    assert b.presentation == "video"
    assert b.hi_i_count is None and b.hi_resolution is None


# --------------------------------------------------------------------------- #
# method table
# --------------------------------------------------------------------------- #
def test_new_methods_resolve_to_the_expected_stage_components():
    expected = {
        "random-prune": ("uniform", "random"),
        "mmtok": ("uniform", "mmtok"),
        "flashvid": ("uniform", "flashvid"),
        "visionzip": ("uniform", "visionzip"),
        "cliptopk": ("cliptopk", "identity"),
        "aks": ("aks", "identity"),
        "lohi-uniform": ("lohi-uniform", "identity"),
        "lohi-semdiv": ("lohi-semdiv", "identity"),
    }
    for method, (select, prune) in expected.items():
        ids = METHODS[method]
        assert (ids.select, ids.prune) == (select, prune), method
        assert (ids.decode, ids.encode, ids.backend) == ("decord", "vit", "qwen-hf"), method
        assert ids.orchestrate == "single_shot"


def test_dry_run_gives_scoring_selectors_the_fake_scorer():
    from longvideo_eval.testing import FakeClipScorer

    orch = build_pipeline(MODEL, "lohi-semdiv", dry_run=True)
    assert type(orch.selector).__name__ == "LoHiSemDivSelector"
    assert isinstance(orch.selector._inner._get_scorer(), FakeClipScorer)   # no CLIP weights


def test_real_assembly_keeps_the_real_scorer_factory():
    # Outside a dry run nothing is loaded at assembly, and the selector keeps its default
    # scorer factory. (Only the selector is inspected; the backend is not touched.)
    from longvideo_eval import _registry
    from longvideo_eval.frontend.select.clip_scorer import default_clip_scorer

    selector = _registry.get("select", METHODS["lohi-semdiv"].select)()
    assert selector._inner._scorer_factory is default_clip_scorer
    assert selector._inner._scorer is None


# --------------------------------------------------------------------------- #
# dry runs, end to end
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("method", ["lohi-uniform", "lohi-semdiv"])
@pytest.mark.parametrize("setup", sorted(LOHI_SETUPS))
def test_lohi_methods_dry_run_end_to_end(method, setup):
    if method == "lohi-semdiv":
        pytest.importorskip("numpy")
    budget = get_setup(setup)
    orch = build_pipeline(MODEL, method, dry_run=True)
    answer = orch.run(SAMPLE, budget)
    assert answer.text == "A"
    # The fake decoder holds 64 frames; all of them go to the video stream, and the K
    # high-resolution frames are encoded a second time.
    assert answer.cost.frames_decoded == 64
    assert answer.cost.frames_encoded == 64 + budget.hi_i_count
    assert answer.cost.prefill_tokens == (64 + budget.hi_i_count) * FakeEncoder.tokens_per_frame
    assert answer.cost.llm_visual_tokens == answer.cost.prefill_tokens     # nothing pruned
    assert answer.cost.rounds == 0 and answer.cost.wall_seconds > 0.0


@pytest.mark.parametrize("method", ["lohi-uniform", "lohi-semdiv"])
def test_lohi_methods_dry_run_selection_is_a_dual_stream_allocation(method):
    if method == "lohi-semdiv":
        pytest.importorskip("numpy")
    budget = get_setup("lohi_128f_r25_k8")
    orch = build_pipeline(MODEL, method, dry_run=True)
    pool = orch.decoder.decode(SAMPLE.video_path, budget)
    sel = orch.selector.select(pool, SAMPLE.query, budget)
    assert list(sel.indices) == list(range(64))                 # the whole pool
    assert sel.per_frame_resolution == [0.25] * 64
    hi = list(sel.hi_indices)
    assert len(hi) == budget.hi_i_count
    assert hi == sorted(set(hi)) and set(hi) <= set(sel.indices)
    if method == "lohi-uniform":
        assert hi == [0, 9, 18, 27, 36, 45, 54, 63]             # 8 evenly spaced frames


def test_lohi_methods_fail_loudly_under_a_single_stream_setup():
    # Without K the dual-stream selectors have nothing to allocate.
    for method in ("lohi-uniform", "lohi-semdiv"):
        with pytest.raises(ValueError, match="hi_i_count"):
            build_pipeline(MODEL, method, dry_run=True).run(SAMPLE, get_setup("native_16f"))


@pytest.mark.parametrize("setup,keep_ratio", [("native_64f", 0.25), ("native_256f", 0.0625)])
def test_random_prune_dry_run_end_to_end(setup, keep_ratio):
    budget = get_setup(setup)
    orch = build_pipeline(MODEL, "random-prune", dry_run=True,
                          prune_kwargs={"keep_ratio": keep_ratio})
    answer = orch.run(SAMPLE, budget)
    pre = 64 * FakeEncoder.tokens_per_frame                     # the fake decoder's 64 frames
    kept = math.ceil(pre * keep_ratio)
    assert answer.text == "A"
    assert answer.cost.frames_encoded == 64                     # every frame is still encoded
    assert answer.cost.vit_patch_tokens == pre * FakeEncoder.merge_size ** 2   # at full size
    assert answer.cost.prefill_tokens == kept                   # only the LLM side shrinks
    assert answer.cost.llm_visual_tokens == kept


def test_random_prune_dry_run_is_reproducible_and_query_dependent():
    budget = get_setup("native_64f")

    def kept_tokens(query):
        orch = build_pipeline(MODEL, "random-prune", dry_run=True)
        pool = orch.decoder.decode("/fake/v.mp4", budget)
        tokens = orch.encoder.encode(pool, orch.selector.select(pool, query, budget), budget)
        tokens.tokens = list(range(tokens.num_tokens))          # make the rows identifiable
        return orch.pruner.prune(tokens, query, budget).tokens

    assert kept_tokens("first question") == kept_tokens("first question")
    assert kept_tokens("first question") != kept_tokens("second question")


@pytest.mark.parametrize("method,cls", [
    ("mmtok", "MMTokPruner"), ("flashvid", "FlashVidPruner"), ("visionzip", "VisionZipPruner"),
])
def test_seam_only_pruners_dry_run_raise_not_implemented(method, cls):
    # These pruners take their signal from the served model's vision tower, which only runs
    # inside generate. The fake encoder emits a plain token list, not the backend package, so
    # a dry run assembles the pipeline and then stops with a clear message.
    orch = build_pipeline(MODEL, method, dry_run=True)
    assert type(orch.pruner).__name__ == cls
    with pytest.raises(NotImplementedError) as exc:
        orch.run(SAMPLE, get_setup("native_64f"))
    msg = str(exc.value)
    assert msg.startswith(f"{cls} is seam-only (pre_llm)")
    assert "model.generate" in msg
    assert "Requires kind='qwen-hf-video'; got 'list'" in msg


@pytest.mark.parametrize("method", ["cliptopk", "aks"])
def test_keyframe_methods_dry_run_end_to_end(method):
    budget = get_setup("keyframe_16f_pool256")
    orch = build_pipeline(MODEL, method, dry_run=True)
    answer = orch.run(SAMPLE, budget)
    assert answer.text == "A"
    # The fake decoder holds 64 frames: the whole pool is decoded and scored, and only the
    # 16 selected frames are encoded.
    assert answer.cost.frames_decoded == 64
    assert answer.cost.frames_encoded == budget.frame_count == 16
    assert answer.cost.prefill_tokens == 16 * FakeEncoder.tokens_per_frame
    assert answer.cost.llm_visual_tokens == answer.cost.prefill_tokens     # nothing pruned
    assert answer.cost.rounds == 0 and answer.cost.wall_seconds > 0.0


@pytest.mark.parametrize("method", ["cliptopk", "aks"])
def test_keyframe_methods_dry_run_selection(method):
    budget = get_setup("keyframe_16f_pool256")
    orch = build_pipeline(MODEL, method, dry_run=True)
    pool = orch.decoder.decode(SAMPLE.video_path, budget)
    assert len(pool.frames) == 64
    sel = orch.selector.select(pool, SAMPLE.query, budget)
    idx = list(sel.indices)
    assert len(idx) == 16 and idx == sorted(set(idx))
    assert set(idx) <= set(range(64))
    assert sel.per_frame_resolution == [1.0] * 16
    assert sel.hi_indices is None                               # single stream
    # The fake scorer's relevance falls off linearly from one frame chosen by the question:
    # peaked nowhere, so CLIP-TopK takes the 16 frames around it and AKS spreads out.
    peak = sum(ord(ch) for ch in SAMPLE.query) % 64
    assert peak in idx
    quarters = {i // 16 for i in idx}
    if method == "cliptopk":
        assert idx == list(range(idx[0], idx[0] + 16))          # one contiguous run
        assert len(quarters) <= 2
    else:
        assert quarters == {0, 1, 2, 3}                         # every quarter of the timeline
        assert {i // 4 for i in idx} == set(range(16))          # one frame per sixteenth


def test_keyframe_methods_dry_run_depends_on_the_question():
    budget = get_setup("keyframe_16f_pool256")
    for method in ("cliptopk", "aks"):
        orch = build_pipeline(MODEL, method, dry_run=True)
        pool = orch.decoder.decode(SAMPLE.video_path, budget)
        a = orch.selector.select(pool, "What is the man doing?", budget).indices
        b = orch.selector.select(pool, "Who opens the door?", budget).indices
        assert list(a) != list(b), method


# --------------------------------------------------------------------------- #
# the documented option tables
# --------------------------------------------------------------------------- #
def _documented_names(option):
    """The names in the first column of the model zoo's tables for ``option`` (the setups
    are split over two tables, one per kind)."""
    doc = Path(__file__).resolve().parents[1] / "docs" / "MODEL_ZOO.md"
    lines = doc.read_text(encoding="utf-8").splitlines()
    headers = [i for i, line in enumerate(lines) if line.startswith(f"| `{option}` |")]
    assert headers
    names = []
    for header in headers:
        for line in lines[header + 2:]:             # skip the header and its separator
            if not line.startswith("|"):
                break
            names.extend(re.findall(r"`([^`]+)`", line.split("|")[1]))
    return names


def test_model_zoo_lists_every_method_once():
    names = _documented_names("--method")
    assert sorted(names) == sorted(METHODS)


def test_model_zoo_lists_every_setup_once():
    names = _documented_names("--setup")
    assert sorted(names) == sorted(SETUPS)


def test_model_zoo_typical_setups_exist():
    # The last column of the method table names a setup to run the method under.
    doc = Path(__file__).resolve().parents[1] / "docs" / "MODEL_ZOO.md"
    rows = [line for line in doc.read_text(encoding="utf-8").splitlines()
            if line.startswith("| `") and line.split("|")[1].strip().strip("`") in METHODS]
    assert len(rows) == len(METHODS)
    for row in rows:
        (setup,) = re.findall(r"`([^`]+)`", row.split("|")[-2])
        assert setup in SETUPS, row


# --------------------------------------------------------------------------- #
# command line, dry run
# --------------------------------------------------------------------------- #
def _make_dataset(tmp_path, n=2):
    videos = tmp_path / "videos"
    videos.mkdir()
    rows = []
    for i in range(n):
        (videos / f"v{i}.mp4").touch()             # the fake decoder never opens these
        rows.append({"video_id": f"v{i}", "question": f"What happens in clip {i}?",
                     "options": ["first", "second", "third", "fourth"], "answer": "A"})
    (tmp_path / "qa.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return tmp_path


def test_cli_dry_run_records_the_dual_stream_budget(tmp_path, monkeypatch):
    monkeypatch.setenv("VIDEOMME_DATA_ROOT", str(_make_dataset(tmp_path)))
    out = tmp_path / "runs"
    code = main([
        "--model", MODEL, "--method", "lohi-uniform", "--task", "videomme",
        "--setup", "lohi_128f_r25_k8", "--output", str(out), "--dry-run",
    ])
    assert code == 0
    (run_dir,) = list(out.glob("videomme_lohi-uniform_lohi_128f_r25_k8_*"))
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["n"] == 2 and summary["accuracy"] == 1.0
    recorded = json.dumps(summary)
    for fragment in ('"presentation": "lohi"', '"hi_i_count": 8', '"hi_resolution": null', '"reference_frames": 16'):
        assert fragment in recorded


def test_cli_dry_run_random_prune_with_prune_kwargs(tmp_path, monkeypatch):
    monkeypatch.setenv("VIDEOMME_DATA_ROOT", str(_make_dataset(tmp_path)))
    out = tmp_path / "runs"
    code = main([
        "--model", MODEL, "--method", "random-prune", "--task", "videomme",
        "--setup", "native_64f", "--output", str(out), "--dry-run",
        "--prune-kwargs", '{"keep_ratio": 0.5}',
    ])
    assert code == 0
    (run_dir,) = list(out.glob("videomme_random-prune_native_64f_*"))
    rows = [json.loads(line)
            for line in (run_dir / "results.jsonl").read_text().strip().splitlines()]
    assert len(rows) == 2
    assert all(r["prefill_tokens"] == 64 * FakeEncoder.tokens_per_frame // 2 for r in rows)
