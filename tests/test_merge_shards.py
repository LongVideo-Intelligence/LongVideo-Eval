"""Tests for scripts/merge_shards.py — dedupe-by-key merge, fail-loud on a real conflict or a
run_meta mismatch across shards, and a recomputed summary (accuracy + cost means) checked
against a hand-computed value. Hardware-free (house pattern): shard dirs are either produced
by a real `--dry-run --shard i/N` CLI invocation (the "clean merge" end-to-end case) or built
directly as tiny synthetic results.jsonl/summary.json pairs (the isolated dedupe/conflict/
meta-mismatch/recompute cases).
"""
import json

import pytest

from longvideo_eval.__main__ import main
from scripts import merge_shards


def _row(video_id, question_id=None, prediction="A", gold="A", correct=True, **cost):
    """A minimal results.jsonl row — CostRecord fields default 0/0.0, override via **cost."""
    base = {
        "decode_seconds": 0.0, "frames_decoded": 0, "vit_flops": 0.0, "frames_encoded": 0,
        "prefill_tokens": 0, "decode_tokens": 0, "thinking_tokens": 0, "rounds": 0,
        "wall_seconds": 0.0,
    }
    base.update(cost)
    return {
        "video_id": video_id, "question_id": question_id, "prediction": prediction,
        "gold": gold, "correct": correct, "arm": None, **base,
    }


def _write_shard(tmp_path, name, rows, meta):
    """Write a shard dir with results.jsonl + summary.json; return its Path."""
    shard_dir = tmp_path / name
    shard_dir.mkdir()
    with (shard_dir / "results.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    summary = {
        "model": "qwen3-vl-4b", "method": "lowres-base", "task": "videomme", "setup": "model_default_16f",
        "thinking": False, "dry_run": True, "n": len(rows), "shard": None,
        "git_sha": None, "timestamp": "20260707-000000",
        **meta,
    }
    (shard_dir / "summary.json").write_text(json.dumps(summary) + "\n", encoding="utf-8")
    return shard_dir


def _make_dataset(tmp_path, n):
    """Same synthetic VideoMME slice test_cli.py uses: n QA rows + empty video files."""
    videos = tmp_path / "videos"
    videos.mkdir()
    rows = []
    for i in range(n):
        vid = f"v{i}"
        (videos / f"{vid}.mp4").touch()
        rows.append({
            "video_id": vid, "question": f"What happens in clip {i}?",
            "options": ["first", "second", "third", "fourth"], "answer": "A",
        })
    qa = tmp_path / "qa.jsonl"
    qa.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return tmp_path


# --------------------------------------------------------------------------- #
# Clean merge (end-to-end: real --shard i/N runs recombined)
# --------------------------------------------------------------------------- #
def test_merge_clean_merge_recombines_all_shards(tmp_path, monkeypatch):
    data_root = _make_dataset(tmp_path, n=9)
    monkeypatch.setenv("VIDEOMME_DATA_ROOT", str(data_root))
    n_shards = 3
    shard_dirs = []
    for i in range(n_shards):
        out = tmp_path / f"runs_{i}"
        main([
            "--task", "videomme", "--setup", "model_default_16f", "--output", str(out), "--dry-run",
            "--shard", f"{i}/{n_shards}",
        ])
        (run_dir,) = out.glob(f"videomme_lowres-base_model_default_16f_s{i}of{n_shards}_*")
        shard_dirs.append(run_dir)

    merged_rows, summary = merge_shards.merge(shard_dirs)

    assert len(merged_rows) == 9
    assert {r["video_id"] for r in merged_rows} == {f"v{i}" for i in range(9)}
    assert summary["n"] == 9
    assert summary["accuracy"] == 1.0                 # FakeBackend "A" == gold "A" throughout
    assert summary["model"] == "qwen3-vl-4b" and summary["method"] == "lowres-base"
    assert "prefill_tokens" in summary["cost_means"]
    assert len(summary["shards"]) == 3
    assert {s["shard"]["index"] for s in summary["shards"]} == {0, 1, 2}

    # CLI entry point writes the same merged data to disk.
    out_dir = tmp_path / "merged"
    code = merge_shards.main([str(d) for d in shard_dirs] + ["--output", str(out_dir)])
    assert code == 0
    written_rows = [json.loads(line) for line in
                    (out_dir / "results.jsonl").read_text().strip().splitlines()]
    assert len(written_rows) == 9
    written_summary = json.loads((out_dir / "summary.json").read_text())
    assert written_summary["n"] == 9 and written_summary["accuracy"] == 1.0


def test_merge_resolves_glob_pattern(tmp_path, monkeypatch):
    data_root = _make_dataset(tmp_path, n=4)
    monkeypatch.setenv("VIDEOMME_DATA_ROOT", str(data_root))
    out = tmp_path / "runs"
    for i in range(2):
        main([
            "--task", "videomme", "--setup", "model_default_16f", "--output", str(out), "--dry-run",
            "--shard", f"{i}/2",
        ])
    merged_rows, summary = merge_shards.merge(
        merge_shards._resolve_shard_dirs([str(out / "videomme_lowres-base_model_default_16f_s*of2_*")])
    )
    assert len(merged_rows) == 4
    assert summary["n"] == 4


# --------------------------------------------------------------------------- #
# Dedupe / conflict / meta-mismatch (isolated, synthetic shard dirs)
# --------------------------------------------------------------------------- #
def test_merge_dedupes_identical_duplicate_rows(tmp_path):
    shared_meta = {"model": "qwen3-vl-4b", "method": "lowres-base", "task": "videomme",
                   "setup": "model_default_16f", "thinking": False}
    row_v0 = _row("v0", prediction="A", gold="A", correct=True, prefill_tokens=100)
    shard0 = _write_shard(tmp_path, "s0", [row_v0, _row("v1", prediction="B", gold="A",
                                                          correct=False)], shared_meta)
    # v0 reappears byte-for-byte identical in shard1 (e.g. an overlapping resubmit) — dedupe,
    # don't error.
    shard1 = _write_shard(tmp_path, "s1", [row_v0, _row("v2")], shared_meta)

    merged_rows, summary = merge_shards.merge([shard0, shard1])

    assert {r["video_id"] for r in merged_rows} == {"v0", "v1", "v2"}
    assert summary["n"] == 3


def test_merge_raises_on_conflicting_duplicate(tmp_path):
    shared_meta = {"model": "qwen3-vl-4b", "method": "lowres-base", "task": "videomme",
                   "setup": "model_default_16f", "thinking": False}
    shard0 = _write_shard(tmp_path, "s0", [_row("v0", prediction="A")], shared_meta)
    # Same key (v0), DIFFERENT prediction — a real conflict, must fail loud, never pick one
    # silently.
    shard1 = _write_shard(tmp_path, "s1", [_row("v0", prediction="B")], shared_meta)

    with pytest.raises(ValueError, match="conflicting duplicate"):
        merge_shards.merge([shard0, shard1])


def test_merge_raises_on_run_meta_mismatch(tmp_path):
    meta_a = {"model": "qwen3-vl-4b", "method": "lowres-base", "task": "videomme",
              "setup": "model_default_16f", "thinking": False}
    meta_b = {**meta_a, "method": "random-prune"}     # method differs -> not the same run
    shard0 = _write_shard(tmp_path, "s0", [_row("v0")], meta_a)
    shard1 = _write_shard(tmp_path, "s1", [_row("v1")], meta_b)

    with pytest.raises(ValueError, match="run_meta mismatch"):
        merge_shards.merge([shard0, shard1])


def test_merge_raises_on_missing_shard_files(tmp_path):
    empty_dir = tmp_path / "not-a-shard"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="results.jsonl"):
        merge_shards.merge([empty_dir])


def test_merge_raises_when_no_dirs_match():
    with pytest.raises(FileNotFoundError, match="no shard directory matched"):
        merge_shards._resolve_shard_dirs(["/no/such/glob/*pattern*"])


# --------------------------------------------------------------------------- #
# Summary recompute vs hand-computed
# --------------------------------------------------------------------------- #
def test_merge_summary_recompute_matches_hand_computed(tmp_path):
    shared_meta = {"model": "qwen3-vl-4b", "method": "lowres-base", "task": "videomme",
                   "setup": "model_default_16f", "thinking": True}
    rows_a = [
        _row("v0", correct=True, prefill_tokens=100, decode_tokens=1),
        _row("v1", correct=False, prefill_tokens=200, decode_tokens=2),
    ]
    rows_b = [
        _row("v2", correct=True, prefill_tokens=300, decode_tokens=3),
        _row("v3", correct=None, gold=None, prefill_tokens=400, decode_tokens=4),  # unscored
    ]
    shard0 = _write_shard(tmp_path, "s0", rows_a, shared_meta)
    shard1 = _write_shard(tmp_path, "s1", rows_b, shared_meta)

    merged_rows, summary = merge_shards.merge([shard0, shard1])

    all_rows = rows_a + rows_b
    scored = [r["correct"] for r in all_rows if r["correct"] is not None]
    expected_accuracy = sum(scored) / len(scored)     # 2/3
    expected_prefill_mean = sum(r["prefill_tokens"] for r in all_rows) / len(all_rows)  # 250.0
    expected_decode_mean = sum(r["decode_tokens"] for r in all_rows) / len(all_rows)    # 2.5

    assert len(merged_rows) == 4
    assert summary["accuracy"] == pytest.approx(expected_accuracy)
    assert summary["cost_means"]["prefill_tokens"] == pytest.approx(expected_prefill_mean)
    assert summary["cost_means"]["decode_tokens"] == pytest.approx(expected_decode_mean)
    assert summary["thinking"] is True


def test_merge_refuses_shards_from_different_budgets(tmp_path):
    """Two shards can agree on (model, method, task, setup) and still have run under different
    budgets. Merging those would silently average incomparable token budgets, so the meta
    check must reject it.
    """
    import json

    import pytest

    from scripts.merge_shards import merge

    dirs = []
    for i, decode_budget in enumerate((16, 256)):
        d = tmp_path / f"s{i}"
        d.mkdir()
        (d / "results.jsonl").write_text(
            json.dumps({"question_id": f"q{i}", "video_id": "v", "correct": True}) + "\n",
            encoding="utf-8",
        )
        (d / "summary.json").write_text(json.dumps({
            "model": "m", "method": "x", "task": "t", "setup": "s", "thinking": False,
            "presentation": "video", "n": 1,
            "budget": {"frame_count": 16, "resolution": 1.0, "decode_budget": decode_budget},
        }), encoding="utf-8")
        dirs.append(d)

    with pytest.raises(ValueError, match="run_meta mismatch"):
        merge(dirs)
