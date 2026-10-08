#!/usr/bin/env python3
"""Merge `--shard i/N` run outputs into one `results.jsonl` + `summary.json`.

Merges by question id in the harness's own row shape (`runners/results.py`): dedupe rows by
identity key, fail loud
on a genuine conflict, recompute accuracy + cost means over the merged set, and verify the
contributing shards actually are shards of the SAME run before trusting any of it.

Steps:
  1. Resolve each CLI arg to one or more shard result directories (glob-expanded if it
     contains a wildcard, else treated as a literal directory).
  2. Load every shard's `results.jsonl` + `summary.json`.
  3. Check `summary.json["model"/"method"/"task"/"setup"/"thinking"]` MATCH across all shards
     — a mismatch means these are shards of DIFFERENT runs, not partitions of one, and merging
     them would silently corrupt the accuracy/cost numbers. Fails loud, naming the mismatch.
  4. Merge rows keyed like `Sample.key` (interfaces.py): `question_id` when present, else
     `video_id`. Two shards may legitimately both carry a row for the SAME key only if it is
     an EXACT duplicate (e.g. an overlapping resubmit) — those are silently deduped, keeping
     the first-seen copy. If two shards disagree on that key's row (different prediction /
     cost / anything) that is a real conflict and this raises loud rather than picking one.
  5. Recompute `accuracy` (mean of `correct` over rows where it is not None) and `cost_means`
     (mean of every `CostRecord` field) over the deduped set — never trust the per-shard
     summaries' own accuracy/cost_means, since those are means over a PARTIAL set.
  6. Write the merged `results.jsonl` + a `summary.json` carrying the recomputed numbers, the
     agreed-on run_meta, and a `shards` provenance list (source dir, n, its shard index/N,
     git_sha, timestamp) for audit.

Usage:
  python scripts/merge_shards.py "runs/videomme_lowres-base_model_default_16f_s*of4_*" --output runs/merged
  python scripts/merge_shards.py runs/.../..._s0of4_... runs/.../..._s1of4_... \\
      runs/.../..._s2of4_... runs/.../..._s3of4_... --output runs/merged
"""
from __future__ import annotations

import argparse
import glob
import json
from dataclasses import fields
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from longvideo_eval.interfaces import CostRecord

_COST_FIELDS = [f.name for f in fields(CostRecord)]
# Must MATCH across shards, and are carried onto the merged summary. `budget` is here because
# a merged artifact has to be SELF-IDENTIFYING: two runs can share (model, method, task, setup)
# and still have been produced under different budgets. `presentation` is reserved for runs
# that record how the frames were presented to the model; it is None otherwise.
_META_KEYS = ("model", "method", "task", "setup", "thinking", "budget", "presentation")
_GLOB_CHARS = set("*?[")


def _resolve_shard_dirs(patterns: List[str]) -> List[Path]:
    """Expand each CLI arg (glob or literal dir) into a sorted, deduped list of shard dirs."""
    dirs: List[Path] = []
    for pat in patterns:
        if _GLOB_CHARS & set(pat):
            matches = sorted(glob.glob(pat))
            if not matches:
                raise FileNotFoundError(f"no shard directory matched glob {pat!r}")
            dirs.extend(Path(m) for m in matches)
        else:
            dirs.append(Path(pat))

    seen = set()
    unique: List[Path] = []
    for d in dirs:
        key = d.resolve()
        if key not in seen:
            seen.add(key)
            unique.append(d)
    if not unique:
        raise ValueError("no shard directories resolved from the given arguments")
    return unique


def _row_key(row: dict) -> str:
    """Mirror `Sample.key` (interfaces.py): `question_id` when present, else `video_id`."""
    qid = row.get("question_id")
    return qid if qid is not None else row["video_id"]


def _load_shard(shard_dir: Path) -> Tuple[List[dict], dict]:
    results_path = shard_dir / "results.jsonl"
    summary_path = shard_dir / "summary.json"
    if not results_path.exists():
        raise FileNotFoundError(f"{shard_dir}: missing results.jsonl")
    if not summary_path.exists():
        raise FileNotFoundError(f"{shard_dir}: missing summary.json")
    rows = [
        json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return rows, summary


def _check_meta_consistent(summaries: "Dict[Path, dict]") -> dict:
    """All shards must agree on _META_KEYS; return those agreed-on values, else fail loud."""
    dirs = list(summaries)
    reference = {k: summaries[dirs[0]].get(k) for k in _META_KEYS}
    for d in dirs[1:]:
        this = {k: summaries[d].get(k) for k in _META_KEYS}
        if this != reference:
            mismatches = {
                k: (reference[k], this[k]) for k in _META_KEYS if reference[k] != this[k]
            }
            raise ValueError(
                f"run_meta mismatch between {dirs[0]} and {d}: {mismatches} — shards must be "
                "partitions of the SAME run (model/method/task/setup/thinking must match)"
            )
    return reference


def merge(shard_dirs: List[Path]) -> Tuple[List[dict], dict]:
    """Load + merge shard rows (dedupe-by-key, fail loud on conflicts) and recompute summary."""
    summaries: Dict[Path, dict] = {}
    merged: Dict[str, dict] = {}
    order: List[str] = []          # first-seen key order, for stable output
    provenance: List[dict] = []

    for shard_dir in shard_dirs:
        rows, summary = _load_shard(shard_dir)
        summaries[shard_dir] = summary
        provenance.append({
            "dir": str(shard_dir),
            "n": summary.get("n"),
            "shard": summary.get("shard"),
            "git_sha": summary.get("git_sha"),
            "timestamp": summary.get("timestamp"),
        })
        for row in rows:
            key = _row_key(row)
            if key in merged:
                if merged[key] != row:
                    raise ValueError(
                        f"conflicting duplicate prediction for key {key!r}: "
                        f"{merged[key]!r} (earlier shard) != {row!r} (from {shard_dir})"
                    )
                continue   # exact duplicate (e.g. an overlapping resubmit) — keep the first
            merged[key] = row
            order.append(key)

    if not merged:
        raise ValueError("no rows found across the given shard directories")

    common_meta = _check_meta_consistent(summaries)
    merged_rows = [merged[k] for k in order]

    n = len(merged_rows)
    scored = [r["correct"] for r in merged_rows if r.get("correct") is not None]
    accuracy = (sum(scored) / len(scored)) if scored else None
    cost_means = {
        name: (sum(r.get(name, 0.0) for r in merged_rows) / n if n else 0.0)
        for name in _COST_FIELDS
    }

    summary = {
        **common_meta,
        "n": n,
        "accuracy": accuracy,
        "cost_means": cost_means,
        "shards": provenance,
    }
    return merged_rows, summary


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="merge_shards",
        description="Merge --shard i/N run outputs into one results.jsonl + summary.json.",
    )
    p.add_argument(
        "shard_dirs", nargs="+",
        help="shard result directories, and/or glob pattern(s) matching them "
             "(e.g. 'runs/videomme_lowres-base_model_default_16f_s*of4_*'). Quote globs so the shell doesn't "
             "expand them first.",
    )
    p.add_argument("--output", required=True, help="directory to write the merged results into")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    shard_dirs = _resolve_shard_dirs(args.shard_dirs)
    merged_rows, summary = merge(shard_dirs)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "results.jsonl").open("w", encoding="utf-8") as fh:
        for row in merged_rows:
            fh.write(json.dumps(row) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    acc = summary["accuracy"]
    acc_str = "n/a" if acc is None else f"{acc:.4f}"
    print(
        f"[merge_shards] merged {len(shard_dirs)} shard(s) -> n={summary['n']} "
        f"accuracy={acc_str} -> {out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
