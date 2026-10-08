"""CLI entry point — assemble a pipeline for one (model, method, task, setup) and run it.

It parses the run spec, loads the lmms-eval-style task YAML, resolves the named Budget preset,
builds the stage pipeline via the registry, iterates the dataset into `Sample`s, and writes
per-sample + summary results.

Two modes, both fully wired:
  --dry-run  swaps decode/encode/backend for the hardware-free fakes (testing.py) so CI covers
             the end-to-end path without weights or real video decode.
  (default)  reaches the real registry components.

Fail-loud discipline: unknown model/method/task/setup, a missing video, or an unknown metric
all raise with the valid options named — never a silent degrade.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import re
from dataclasses import replace
from typing import List, Optional, Tuple

from .interfaces import Budget, Sample
from .models.build import build_pipeline
from .runners.evaluate import accuracy, evaluate
from .runners.results import write_results
from .runners.setups import get_setup
from .tasks.loader import Task, load_task


def _shard_spec(spec: str) -> Tuple[int, int]:
    """argparse `type=` for `--shard i/N`: parse + validate, fail loud with the offending spec.

    Raising `argparse.ArgumentTypeError` (not a bare ValueError) lets argparse report this as
    a normal CLI usage error (with the flag name + spec echoed) instead of a raw traceback.
    """
    m = re.fullmatch(r"(\d+)/(\d+)", spec.strip())
    if not m:
        raise argparse.ArgumentTypeError(f"--shard must be 'i/N' (e.g. '0/4'), got {spec!r}")
    i, n = int(m.group(1)), int(m.group(2))
    if n < 1:
        raise argparse.ArgumentTypeError(f"--shard N must be >= 1, got {spec!r}")
    if not (0 <= i < n):
        raise argparse.ArgumentTypeError(f"--shard i must satisfy 0 <= i < N, got {spec!r}")
    return i, n


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="longvideo-eval",
        description="Fair-cost long-video understanding harness.",
    )
    p.add_argument("--model", default="qwen3-vl-4b", help="base model id (see config.BASE_MODELS)")
    p.add_argument("--method", default="lowres-base", help="method id (see models.build.METHODS)")
    p.add_argument("--task", required=True, help="task name (a longvideo_eval/tasks/<name>.yaml)")
    p.add_argument("--setup", default="lowres_base_256f", help="Budget preset name (see runners.setups.SETUPS)")
    p.add_argument("--limit", type=int, default=None, help="cap the number of samples evaluated")
    p.add_argument("--only-videos", type=str, default=None,
                   help="comma-separated video_id list; evaluate ONLY those videos' samples. "
                        "Applied BEFORE --limit/--shard. Fails loud if an id matches nothing.")
    p.add_argument("--shard", type=_shard_spec, default=None,
                   help="'i/N': partition the POST-LIMIT sample sequence into N shards by "
                        "strided round-robin (sample index %% N == i) and evaluate only shard "
                        "i. Round-robin, not contiguous blocks, so a stratified dataset's "
                        "buckets (e.g. duration tiers, usually emitted as contiguous runs) "
                        "stay balanced across every shard regardless of N — a contiguous slice "
                        "could instead hand one shard an entire bucket and another none. "
                        "Sharding runs AFTER --limit so a --limit smoke test stays small and "
                        "reproducible when also sharded. The results dir name gets a "
                        "'_s{i}of{N}' suffix so shards never collide; recombine with "
                        "scripts/merge_shards.py.")
    p.add_argument("--output", default="runs/", help="base directory for run outputs")
    p.add_argument("--dry-run", action="store_true",
                   help="use hardware-free fakes (no weights / no real decode)")
    p.add_argument("--subtitles", action="store_true",
                   help="feed the subtitle channel (parity: to method AND baseline)")
    p.add_argument("--thinking", action="store_true",
                   help="run the thinking arm (default off). Validity is per model — see "
                        "config.ModelSpec.thinking: 'none' rejects it, 'hybrid' allows both "
                        "arms, 'always' requires it. Metered into CostRecord.thinking_tokens.")
    p.add_argument("--max-new-tokens", type=int, default=None,
                   help="override QwenHFBackend's generation cap (default: the backend's own "
                        "_DEFAULT_MAX_NEW_TOKENS=32, sized for a bare MCQ letter). Thinking arms "
                        "need headroom for the <think> span on top of the answer; pass e.g. 2048 "
                        "for --thinking runs. Ignored under --dry-run (the fake backend has no "
                        "generation cap).")
    p.add_argument("--prune-kwargs", default=None,
                   help="JSON object forwarded to build_pipeline(prune_kwargs=...) -> the "
                        "pruner constructor. Parsed eagerly so a malformed value fails loud "
                        "before any compute is spent; recorded verbatim in "
                        "run_meta['prune_kwargs'].")
    return p


def _load_dataset(task: Task, use_subtitles: bool):
    """Instantiate the dataset loader named in the task YAML, expanding env vars in its kwargs.

    Env expansion lets a task YAML point at `${VIDEOMME_DATA_ROOT}/...` so data roots stay out
    of version control; an unset var expands to the literal and fails loudly at file resolution.
    """
    module_path, _, cls_name = task.dataset_loader.partition(":")
    if not module_path or not cls_name:
        raise ValueError(
            f"task {task.name!r} dataset loader {task.dataset_loader!r} must be 'module:Class'"
        )
    module = importlib.import_module(module_path)
    loader_cls = getattr(module, cls_name)
    kwargs = {k: (os.path.expandvars(v) if isinstance(v, str) else v)
              for k, v in task.dataset_kwargs.items()}
    return loader_cls(use_subtitles=use_subtitles, **kwargs)


def _parse_prune_kwargs(raw: Optional[str]) -> Optional[dict]:
    """Parse `--prune-kwargs` eagerly; fail loud (before any compute) on bad JSON / shape."""
    if raw is None:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"--prune-kwargs is not valid JSON: {raw!r} ({e})") from e
    if not isinstance(parsed, dict):
        raise ValueError(
            f"--prune-kwargs must be a JSON object, got {type(parsed).__name__}: {raw!r}"
        )
    return parsed


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    prune_kwargs = _parse_prune_kwargs(args.prune_kwargs)
    task = load_task(args.task)
    budget = get_setup(args.setup)
    orchestrator = build_pipeline(
        args.model, args.method, dry_run=args.dry_run, enable_thinking=args.thinking,
        prune_kwargs=prune_kwargs, max_new_tokens=args.max_new_tokens,
    )

    use_subtitles = args.subtitles or task.subtitles
    dataset = _load_dataset(task, use_subtitles)

    # Materialize samples first: iterating the loader also populates `dataset.golds`, and it
    # resolves every video path up front so a missing file fails before any compute is spent.
    samples: List[Sample] = list(dataset)
    if args.only_videos is not None:
        wanted = {v for v in args.only_videos.split(",") if v}
        samples = [s for s in samples if s.video_id in wanted]
        missing = wanted - {s.video_id for s in samples}
        if missing:
            raise SystemExit(
                f"--only-videos: no samples for {sorted(missing)} in task {args.task!r}; "
                "check the id against the dataset's video_id column"
            )
    if args.limit is not None:
        samples = samples[: args.limit]

    # --shard partitions AFTER --limit (decision: --limit is a debug cap on the WHOLE sequence;
    # sharding a debug-capped run should still be small + reproducible, so the shard filter runs
    # on whatever --limit left). Strided round-robin (index % N == i), never a contiguous slice
    # — see the --shard help string for why: it keeps a stratified dataset's buckets balanced
    # across every shard regardless of N.
    if args.shard is not None:
        shard_i, shard_n = args.shard
        samples = [s for idx, s in enumerate(samples) if idx % shard_n == shard_i]

    # Render the task prompt into each sample's `query` (the string the backend sees). The raw
    # question + choices stay available on the sample for the prompt template.
    prompt_samples = [replace(s, query=task.build_prompt(s)) for s in samples]

    # golds is keyed over the FULL dataset (populated by the `list(dataset)` above, before
    # --limit/--shard filtered the local `samples` list) — a superset is harmless since
    # evaluate() only looks up the keys it actually iterates (per-shard scoring stays correct).
    results = evaluate(orchestrator, prompt_samples, budget, golds=dataset.golds)

    run_meta = {
        "model": args.model,
        "method": args.method,
        "task": args.task,
        "setup": args.setup,
        "thinking": args.thinking,   # the run's arm, beside the (model, method, setup) triple
        "dry_run": args.dry_run,
        "subtitles": use_subtitles,
        "budget": _budget_dict(budget),
        "shard": {"index": args.shard[0], "n": args.shard[1]} if args.shard is not None else None,
        "prune_kwargs": prune_kwargs,
        "max_new_tokens": args.max_new_tokens,
    }
    run_dir = write_results(results, args.output, run_meta)

    acc = accuracy(results)
    acc_str = "n/a" if acc is None else f"{acc:.4f}"
    print(f"[longvideo-eval] {args.task}/{args.method}/{args.setup}: "
          f"n={len(results)} accuracy={acc_str} -> {run_dir}")
    return 0


def _budget_dict(budget: Budget) -> dict:
    return {
        "token_budget": budget.token_budget,
        "frame_count": budget.frame_count,
        "resolution": budget.resolution,
        "decode_budget": budget.decode_budget,
        "max_rounds": budget.max_rounds,
        "hi_i_count": budget.hi_i_count,
        "hi_resolution": budget.hi_resolution,
        "reference_frames": budget.reference_frames,
        "presentation": budget.presentation,
    }


if __name__ == "__main__":
    raise SystemExit(main())
