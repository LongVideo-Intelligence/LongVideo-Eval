"""CLI tests: dry-run E2E, task YAML, setups, method assembly.

All hardware-free. The CLI test builds a tiny synthetic Video-MME slice (a jsonl + empty .mp4
files the FakeDecoder never opens) so the full parse -> load task -> build pipeline -> iterate
-> evaluate -> write-results path is exercised end-to-end in CI without weights or real decode.
"""
import json
import re

import pytest

from longvideo_eval.__main__ import main
from longvideo_eval.data.videomme import VideoMME
from longvideo_eval.models.build import build_pipeline
from longvideo_eval.runners.setups import get_setup
from longvideo_eval.tasks.loader import load_task
from longvideo_eval.interfaces import Sample


def _make_dataset(tmp_path, n=3):
    """Write n QA rows + empty video files under a fresh VideoMME data root; return the root."""
    videos = tmp_path / "videos"
    videos.mkdir()
    rows = []
    for i in range(n):
        vid = f"v{i}"
        (videos / f"{vid}.mp4").touch()  # FakeDecoder never opens these; existence is enough
        rows.append({
            "video_id": vid,
            "question": f"What happens in clip {i}?",
            "options": ["first", "second", "third", "fourth"],
            "answer": "A",
        })
    qa = tmp_path / "qa.jsonl"
    qa.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return tmp_path


def _make_bucketed_dataset(tmp_path, n_per_bucket=6):
    """`n_per_bucket` "short" videos followed by `n_per_bucket` "long" videos — a contiguous
    per-bucket ordering, the shape a stratified loader (e.g. duration tiers) tends to emit.
    `video_id` encodes the bucket so tests can assert round-robin sharding balances it.
    """
    videos = tmp_path / "videos"
    videos.mkdir()
    rows = []
    for bucket in ("short", "long"):
        for i in range(n_per_bucket):
            vid = f"{bucket}{i}"
            (videos / f"{vid}.mp4").touch()
            rows.append({
                "video_id": vid,
                "question": f"What happens in {vid}?",
                "options": ["first", "second", "third", "fourth"],
                "answer": "A",
            })
    qa = tmp_path / "qa.jsonl"
    qa.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return tmp_path


def test_cli_dry_run_end_to_end(tmp_path, monkeypatch):
    data_root = _make_dataset(tmp_path, n=3)
    monkeypatch.setenv("VIDEOMME_DATA_ROOT", str(data_root))
    out = tmp_path / "runs"

    code = main([
        "--model", "qwen3-vl-4b", "--method", "lowres-base", "--task", "videomme",
        "--setup", "model_default_16f", "--limit", "8", "--output", str(out), "--dry-run",
    ])
    assert code == 0

    run_dirs = list(out.glob("videomme_lowres-base_model_default_16f_*"))
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    lines = (run_dir / "results.jsonl").read_text().strip().splitlines()
    assert len(lines) == 3                      # all 3 samples scored (limit 8 does not cap)
    first = json.loads(lines[0])
    assert first["prediction"] == "A" and first["correct"] is True
    assert "frames_decoded" in first            # full CostRecord folded into each row

    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["accuracy"] == 1.0           # FakeBackend answers "A", golds are "A"
    assert summary["n"] == 3
    assert summary["model"] == "qwen3-vl-4b" and summary["dry_run"] is True
    assert "cost_means" in summary


def test_videomme_loader_disambiguates_multiple_questions_per_video(tmp_path):
    """Real Video-MME ships ~3 questions/video; `question_id` must keep golds from colliding
    (lmms-eval avoids this by indexing on the HF dataset row `doc_id`, not `videoID`)."""
    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "v0.mp4").touch()
    rows = [
        {"video_id": "v0", "question_id": "v0-0", "question": "Q0",
         "options": ["a", "b", "c", "d"], "answer": "A"},
        {"video_id": "v0", "question_id": "v0-1", "question": "Q1",
         "options": ["a", "b", "c", "d"], "answer": "B"},
        {"video_id": "v0", "question_id": "v0-2", "question": "Q2",
         "options": ["a", "b", "c", "d"], "answer": "C"},
    ]
    qa = tmp_path / "qa.jsonl"
    qa.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    dataset = VideoMME(video_root=str(videos), qa_path=str(qa))
    samples = list(dataset)

    assert [s.question_id for s in samples] == ["v0-0", "v0-1", "v0-2"]
    assert all(s.video_id == "v0" for s in samples)
    assert dataset.golds == {"v0-0": "A", "v0-1": "B", "v0-2": "C"}


def test_videomme_loader_raises_on_gold_key_collision(tmp_path):
    """Without a distinguishing question_id, reused video_ids must fail loud, never silently
    overwrite an earlier gold (fail-loud discipline)."""
    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "v0.mp4").touch()
    rows = [
        {"video_id": "v0", "question": "Q0", "options": ["a", "b", "c", "d"], "answer": "A"},
        {"video_id": "v0", "question": "Q1", "options": ["a", "b", "c", "d"], "answer": "B"},
    ]
    qa = tmp_path / "qa.jsonl"
    qa.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    dataset = VideoMME(video_root=str(videos), qa_path=str(qa))
    with pytest.raises(ValueError, match="duplicate gold key"):
        list(dataset)


def test_task_yaml_loads_and_renders_prompt():
    task = load_task("videomme")
    assert task.metric == "mcq_accuracy"
    prompt = task.build_prompt(
        Sample("v1", "/v1.mp4", "What is shown?", choices=["red", "green", "blue", "black"])
    )
    assert "A. red" in prompt and "D. black" in prompt
    assert "What is shown?" in prompt
    assert prompt.rstrip().endswith("The best answer is:")


def test_load_task_unknown_raises():
    with pytest.raises(FileNotFoundError, match="available tasks"):
        load_task("does-not-exist")


def test_get_setup_known_and_unknown():
    assert get_setup("model_default_16f").frame_count == 16
    assert get_setup("model_default_32f").frame_count == 32
    with pytest.raises(KeyError, match="available"):
        get_setup("999F")


def test_frames_seen_setups_are_native_resolution():
    # The ① family: no resize (resolution None), decode a generous pool then select frame_count.
    for name in ("model_default_16f", "model_default_32f", "model_default_64f"):
        assert get_setup(name).resolution is None


def test_iso_token_setups_hold_frame_resolution_tradeoff():
    # The ② family: N * r^2 constant (~5760 tokens), decode_budget == frame_count (no over-decode),
    # token_budget left None (metered by qwen_tokens, not pinned — Decisions log #1).
    expected = {
        "native_16f": (16, 1.0),
        "uniform_64f_r50": (64, 0.5),
        "lowres_base_256f": (256, 0.25),
    }
    n_r2 = set()
    for name, (frames, r) in expected.items():
        b = get_setup(name)
        assert b.frame_count == frames
        assert b.resolution == r
        assert b.decode_budget == frames          # every decoded frame is used
        assert b.token_budget is None             # not pinned; realized count is metered
        n_r2.add(round(frames * r * r, 6))
    assert len(n_r2) == 1                          # N * r^2 identical across the iso triplet


def test_build_pipeline_unknown_method_lists_options():
    with pytest.raises(ValueError, match="valid methods"):
        build_pipeline("qwen3-vl-4b", "no-such-method", dry_run=True)


def test_build_pipeline_unknown_model_lists_options():
    with pytest.raises(ValueError, match="valid models"):
        build_pipeline("no-such-model", "lowres-base", dry_run=True)


def test_cli_records_thinking_arm_in_summary(tmp_path, monkeypatch):
    # The run's arm is recorded beside the (model, method, setup) triple in summary.json —
    # default off, and --thinking flips it (dry-run: fakes carry no thinking span, flag is
    # still recorded so runs stay identifiable).
    data_root = _make_dataset(tmp_path, n=1)
    monkeypatch.setenv("VIDEOMME_DATA_ROOT", str(data_root))

    for flag, expected in (([], False), (["--thinking"], True)):
        out = tmp_path / f"runs_{expected}"
        code = main([
            "--model", "qwen3.5-4b", "--method", "lowres-base", "--task", "videomme",
            "--setup", "model_default_16f", "--output", str(out), "--dry-run", *flag,
        ])
        assert code == 0
        (run_dir,) = out.glob("videomme_lowres-base_model_default_16f_*")
        summary = json.loads((run_dir / "summary.json").read_text())
        assert summary["thinking"] is expected


def test_cli_thinking_rejected_for_none_mode_model():
    # Real (non-dry) assembly validates the arm against ModelSpec.thinking at build time,
    # BEFORE any dataset/video is touched — so no fixture data is needed.
    with pytest.raises(ValueError, match="no.*thinking mode"):
        main(["--model", "qwen3-vl-4b", "--method", "lowres-base", "--task", "videomme",
              "--thinking"])


def test_build_pipeline_real_mode_fails_loudly_on_execute(tmp_path):
    """Without --dry-run, assembly succeeds and the real decoder fails loudly on a missing video.

    (Until #15 the decode stage was a skeleton and this expected NotImplementedError; the real
    DecordDecoder's first fail-loud guard is the path check.)
    """
    orch = build_pipeline("qwen3-vl-4b", "lowres-base", dry_run=False)
    with pytest.raises(FileNotFoundError, match="video not found"):
        orch.run(Sample("v1", str(tmp_path / "x.mp4"), "q"), get_setup("model_default_16f"))


# --------------------------------------------------------------------------- #
# --shard i/N
# --------------------------------------------------------------------------- #
def test_shard_spec_rejects_bad_format_and_out_of_range():
    # argparse's `type=` raises ArgumentTypeError -> argparse turns that into a usage error
    # (SystemExit(2)), never a raw traceback.
    with pytest.raises(SystemExit):
        main(["--task", "videomme", "--shard", "not-a-shard"])
    with pytest.raises(SystemExit):
        main(["--task", "videomme", "--shard", "4/4"])       # i must be < N
    with pytest.raises(SystemExit):
        main(["--task", "videomme", "--shard", "0/0"])       # N must be >= 1


def test_shard_partition_disjoint_cover_and_deterministic(tmp_path, monkeypatch):
    """Every sample lands in EXACTLY one of the N shards (disjoint cover), and re-running the
    same shard spec against the same data yields the identical set (determinism)."""
    n_per_bucket = 6
    data_root = _make_bucketed_dataset(tmp_path, n_per_bucket=n_per_bucket)
    monkeypatch.setenv("VIDEOMME_DATA_ROOT", str(data_root))
    n_shards = 4

    shard_video_ids = []
    for i in range(n_shards):
        out = tmp_path / f"runs_{i}"
        code = main([
            "--task", "videomme", "--setup", "model_default_16f", "--output", str(out), "--dry-run",
            "--shard", f"{i}/{n_shards}",
        ])
        assert code == 0
        (run_dir,) = out.glob(f"videomme_lowres-base_model_default_16f_s{i}of{n_shards}_*")
        rows = [json.loads(line) for line in
                (run_dir / "results.jsonl").read_text().strip().splitlines()]
        shard_video_ids.append([r["video_id"] for r in rows])

    all_ids = [vid for shard in shard_video_ids for vid in shard]
    expected = sorted(f"{b}{i}" for b in ("short", "long") for i in range(n_per_bucket))
    assert sorted(all_ids) == expected          # cover: union == the full sample set
    assert len(set(all_ids)) == len(all_ids)    # disjoint: no id repeated across shards

    # determinism: re-running shard 0 against the same data reproduces the identical row set.
    out_rerun = tmp_path / "runs_0_rerun"
    main([
        "--task", "videomme", "--setup", "model_default_16f", "--output", str(out_rerun), "--dry-run",
        "--shard", f"0/{n_shards}",
    ])
    (rerun_dir,) = out_rerun.glob(f"videomme_lowres-base_model_default_16f_s0of{n_shards}_*")
    rerun_ids = [json.loads(line)["video_id"] for line in
                 (rerun_dir / "results.jsonl").read_text().strip().splitlines()]
    assert rerun_ids == shard_video_ids[0]


def test_shard_partition_preserves_bucket_balance(tmp_path, monkeypatch):
    """Round-robin (index % N == i) keeps every shard's bucket mix even when the dataset's
    own ordering groups a bucket into a contiguous run (6 short then 6 long here) — a
    CONTIGUOUS slice would instead hand shard 0 all-short and the last shard all-long."""
    data_root = _make_bucketed_dataset(tmp_path, n_per_bucket=6)
    monkeypatch.setenv("VIDEOMME_DATA_ROOT", str(data_root))
    n_shards = 3   # 12 samples / 3 shards = 4 each -> 2 short + 2 long per shard, exactly

    for i in range(n_shards):
        out = tmp_path / f"runs_{i}"
        main([
            "--task", "videomme", "--setup", "model_default_16f", "--output", str(out), "--dry-run",
            "--shard", f"{i}/{n_shards}",
        ])
        (run_dir,) = out.glob(f"videomme_lowres-base_model_default_16f_s{i}of{n_shards}_*")
        video_ids = [json.loads(line)["video_id"] for line in
                     (run_dir / "results.jsonl").read_text().strip().splitlines()]
        n_short = sum(1 for v in video_ids if v.startswith("short"))
        n_long = sum(1 for v in video_ids if v.startswith("long"))
        assert (n_short, n_long) == (2, 2), f"shard {i} imbalanced: {video_ids}"


def test_shard_applies_after_limit(tmp_path, monkeypatch):
    """--shard partitions the POST-limit sequence: --limit 6 first caps 12 samples down to the
    first 6 (v0..v5), THEN --shard 0/2 keeps index%2==0 of THOSE 6 (v0, v2, v4) — not of the
    full 12."""
    data_root = _make_dataset(tmp_path, n=12)
    monkeypatch.setenv("VIDEOMME_DATA_ROOT", str(data_root))
    out = tmp_path / "runs"
    main([
        "--task", "videomme", "--setup", "model_default_16f", "--output", str(out), "--dry-run",
        "--limit", "6", "--shard", "0/2",
    ])
    (run_dir,) = out.glob("videomme_lowres-base_model_default_16f_s0of2_*")
    video_ids = [json.loads(line)["video_id"] for line in
                 (run_dir / "results.jsonl").read_text().strip().splitlines()]
    assert video_ids == ["v0", "v2", "v4"]


def test_shard_flag_adds_results_dir_suffix(tmp_path, monkeypatch):
    data_root = _make_dataset(tmp_path, n=2)
    monkeypatch.setenv("VIDEOMME_DATA_ROOT", str(data_root))

    out_plain = tmp_path / "plain"
    main(["--task", "videomme", "--setup", "model_default_16f", "--output", str(out_plain), "--dry-run"])
    (plain_dir,) = out_plain.glob("videomme_lowres-base_model_default_16f_*")
    assert re.search(r"_s\d+of\d+_", plain_dir.name) is None    # unsharded: no suffix

    out_shard = tmp_path / "shard"
    main([
        "--task", "videomme", "--setup", "model_default_16f", "--output", str(out_shard), "--dry-run",
        "--shard", "1/3",
    ])
    (shard_dir,) = out_shard.glob("videomme_lowres-base_model_default_16f_*")
    assert "_s1of3_" in shard_dir.name

    summary = json.loads((shard_dir / "summary.json").read_text())
    assert summary["shard"] == {"index": 1, "n": 3}


# --------------------------------------------------------------------------- #
# --prune-kwargs
# --------------------------------------------------------------------------- #
def test_prune_kwargs_invalid_json_fails_loud():
    with pytest.raises(ValueError, match="not valid JSON"):
        main(["--task", "videomme", "--prune-kwargs", "{not json}"])


def test_prune_kwargs_must_be_json_object_fails_loud():
    with pytest.raises(ValueError, match="must be a JSON object"):
        main(["--task", "videomme", "--prune-kwargs", "[1, 2, 3]"])
