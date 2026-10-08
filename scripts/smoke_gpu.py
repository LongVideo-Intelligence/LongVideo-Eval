#!/usr/bin/env python
"""One-command GPU smoke: assemble the REAL single-shot pipeline on one video + question and
print the Answer + CostRecord. Needs a GPU machine (decord + torch + transformers + weights);
imports nothing GPU-specific until invoked, so this file lints/collects on a GPU-free machine.

    python scripts/smoke_gpu.py \
        --model qwen3.5-4b --method lowres-base --setup lowres_base_256f \
        --video /path/clip.mp4 \
        --question "What happens first? A. x B. y C. z D. w" \
        --thinking            # or --no-thinking (default)

The point is that a GPU check is a single command: it exercises decode -> select -> encode
(HF preprocess + exact-token assert) -> prune -> generate (vision tower + LLM), and surfaces
the run's arm (model / setup / thinking on|off) so results stay identifiable.
"""
from __future__ import annotations

import argparse
import sys

from longvideo_eval.interfaces import Sample
from longvideo_eval.models.build import build_pipeline
from longvideo_eval.runners.setups import SETUPS, get_setup


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="qwen3-vl-4b",
                   help="config.BASE_MODELS key (qwen3-vl-4b | qwen3.5-4b)")
    p.add_argument("--method", default="lowres-base",
                   help="build.METHODS key (qwen-default | lowres-base = dense low-res @ lowres_base_256f)")
    p.add_argument("--setup", default="lowres_base_256f",
                   help=f"runners.setups key; available: {sorted(SETUPS)}")
    p.add_argument("--video", required=True, help="path to a video file")
    p.add_argument("--question", required=True, help="MCQ prompt text (question + options)")
    p.add_argument("--subtitles", default=None, help="optional subtitle text (parity channel)")
    thinking = p.add_mutually_exclusive_group()
    thinking.add_argument("--thinking", dest="thinking", action="store_true",
                          help="enable the thinking arm (valid for hybrid/always models per "
                               "config.ModelSpec.thinking; REQUIRED for qwen3-vl-4b-thinking; "
                               "metered into thinking_tokens)")
    thinking.add_argument("--no-thinking", dest="thinking", action="store_false",
                          help="disable thinking (default; invalid for always-thinking models)")
    p.set_defaults(thinking=False)
    p.add_argument("--max-new-tokens", type=int, default=None,
                   help="override backend max_new_tokens (default: backend default)")
    args = p.parse_args()

    budget = get_setup(args.setup)
    orch = build_pipeline(args.model, args.method, dry_run=False, enable_thinking=args.thinking)
    if args.max_new_tokens is not None:
        orch.backend.max_new_tokens = args.max_new_tokens  # per-run override

    sample = Sample(
        video_id=args.video, video_path=args.video, query=args.question, subtitles=args.subtitles,
    )

    print(f"[smoke] model={args.model} method={args.method} setup={args.setup} "
          f"thinking={args.thinking}")
    print(f"[smoke] budget={budget}")
    answer = orch.run(sample, budget)

    print("\n=== Answer ===")
    print(answer.text)
    print("\n=== CostRecord ===")
    print(answer.cost)
    if answer.rounds_trace:
        print("\n=== Arm ===")
        print(answer.rounds_trace[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
