"""Results writer — per-sample jsonl + run summary json, lmms-eval-like layout.

Every run lands in `<output>/<task>_<method>_<setup>[_s{i}of{N}]_<YYYYmmdd-HHMMSS>/` (the
`_s{i}of{N}` segment appears only for a `--shard i/N` run, so sharded and unsharded runs, and
every shard of one run, never collide; `scripts/merge_shards.py` recombines them):
  results.jsonl  one line per sample: video_id, question_id, prediction, gold, correct, arm
                 (the backend's run-arm record, lifted from Answer.rounds_trace so rows
                 self-identify; null when the backend records none), + EVERY CostRecord field
                 (front-end + back-end costs are first-class). `question_id` disambiguates
                 multi-Q/video benchmarks (Video-MME ~3 Q/video); it is None for 1-QA/video
                 sources.
  summary.json   the run meta (model/method/task/setup/git-sha) + accuracy + n + the mean of
                 each cost field, so a run is comparable on the accuracy-cost axis at a glance.
  rounds_trace.jsonl  (written when any row carries a trace) one line per sample: key
                 (question_id else video_id) + the backend's per-call trace.

Keeping the full CostRecord per sample (not just accuracy) is what makes these outputs the raw
material for fair-cost tables.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, fields
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from ..interfaces import CostRecord
from .evaluate import Result, accuracy

_COST_FIELDS = [f.name for f in fields(CostRecord)]


def _git_sha() -> Optional[str]:
    """Current commit sha, or None if git is unavailable / this is not a repo (loud-narrow)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return out.stdout.strip()


def write_results(results: List[Result], out_dir: str, run_meta: dict) -> Path:
    """Write results.jsonl + summary.json under a fresh timestamped run directory; return it."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    shard = run_meta.get("shard")   # {"index": i, "n": N} for a --shard run, else None/absent
    shard_suffix = f"_s{shard['index']}of{shard['n']}" if shard else ""
    run_name = (
        f"{run_meta['task']}_{run_meta['method']}_{run_meta['setup']}{shard_suffix}_{timestamp}"
    )
    run_dir = Path(out_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "results.jsonl").open("w", encoding="utf-8") as fh:
        for r in results:
            row = {
                "video_id": r.video_id,
                "question_id": r.question_id,
                "prediction": r.prediction,
                "gold": r.gold,
                "correct": r.correct,
                "arm": r.arm,   # run-arm metadata — rows self-identify
                # Per-sample error capture: "ok" | "error" (+ the exception
                # text) — a failed sample is loud in its ROW while the run's other rows stand.
                "status": getattr(r, "status", "ok"),
                "error": getattr(r, "error", None),
                **asdict(r.cost),
            }
            fh.write(json.dumps(row) + "\n")

    # Per-call traces: only when at least one row has one.
    if any(r.rounds_trace for r in results):
        with (run_dir / "rounds_trace.jsonl").open("w", encoding="utf-8") as fh:
            for r in results:
                fh.write(json.dumps({
                    "key": r.question_id if r.question_id is not None else r.video_id,
                    "video_id": r.video_id,
                    "question_id": r.question_id,
                    "rounds_trace": r.rounds_trace or [],
                }) + "\n")

    n = len(results)
    cost_means = {
        name: (sum(getattr(r.cost, name) for r in results) / n if n else 0.0)
        for name in _COST_FIELDS
    }
    summary = {
        **run_meta,
        "git_sha": _git_sha(),
        "timestamp": timestamp,
        "n": n,
        "errors": sum(1 for r in results if getattr(r, "status", "ok") == "error"),
        "accuracy": accuracy(results),
        "cost_means": cost_means,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return run_dir
