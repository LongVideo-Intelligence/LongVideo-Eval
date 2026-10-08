"""Eval loop — run an Orchestrator over samples and collect (prediction, cost) pairs.

The accuracy–cost pairs this returns are the raw material for the fair-cost tables. Scoring
for MCQ is a simple exact match on the chosen option letter.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional

from ..interfaces import Answer, Budget, Orchestrator, Sample


@dataclass
class Result:
    video_id: str
    prediction: str
    gold: Optional[str]
    correct: Optional[bool]
    cost: object   # CostRecord
    question_id: Optional[str] = None   # disambiguates multi-Q/video benchmarks (Video-MME)
    # Compact run-arm metadata (model / thinking mode / token counts) lifted
    # from Answer.rounds_trace[0]["arm"] so results.jsonl rows are SELF-IDENTIFYING — without
    # it the run arm never reached disk and result files couldn't be told apart.
    arm: Optional[dict] = None
    # The backend's per-call trace (carries the run arm).
    rounds_trace: Optional[list] = None
    # Per-sample error capture (fail loud in the ROW, not fatal to the RUN). "ok" for a normal
    # row; "error" when
    # orchestrator.run raised for THIS sample — the exception text lands in ``error``, the row
    # scores as incorrect-equivalent (correct=None -> excluded from accuracy but counted in
    # summary["errors"]), and the run continues.
    status: str = "ok"
    error: Optional[str] = None


_WRAP_ANSWER = re.compile(r"\[\[\s*([A-Za-z])\s*\]\]|\{\{\s*([A-Za-z])\s*\}\}")


def extract_choice(text: str) -> Optional[str]:
    """Best-effort MCQ option-letter extraction.

    PRIORITY: an explicitly bracket-wrapped final answer — ``[[A]]`` or ``{{A}}``. Verbose /
    thinking models (e.g. qwen3.5) bury the choice
    in a long response where the positional heuristics below grab an early *mentioned* letter, not
    the *chosen* one; the wrap is an unambiguous anchor, and the LAST wrap is taken (the model's
    final choice, after any reasoning). Falls back to the positional heuristics when absent —
    handles 'A', '(A)', 'A. foo', 'The answer is A', 'Option B'. Heuristic; a real benchmark
    ships its own parser."""
    wraps = _WRAP_ANSWER.findall(text or "")
    if wraps:
        last = wraps[-1]
        return (last[0] or last[1]).upper()
    t = (text or "").strip().upper()
    if not t:
        return None
    m = re.match(r"^\(?\s*([A-Z])\s*[\).:\-\s]", t) or re.fullmatch(r"([A-Z])", t)
    if m:
        return m.group(1)
    m = re.search(r"\b(?:ANSWER|OPTION|CHOICE)\b[^A-Z]{0,8}([A-Z])\b", t)
    if m:
        return m.group(1)
    m = re.search(r"\b([A-Z])\b", t)      # fallback: first standalone capital letter
    return m.group(1) if m else None


def score_mcq(prediction: str, gold: Optional[str]) -> Optional[bool]:
    if gold is None:
        return None
    g = gold.strip().upper()
    gold_letter = g[0] if g else None
    return extract_choice(prediction) == gold_letter


def evaluate(orchestrator: Orchestrator, samples: Iterable[Sample], budget: Budget,
             golds: Optional[dict] = None) -> List[Result]:
    """Run every sample once through `orchestrator`.

    `golds` maps `Sample.key` -> answer (i.e. `question_id` when the dataset has one, else
    `video_id`) — see `Sample.key`. Never index golds by `video_id` directly: real Video-MME
    has ~3 questions per video and `video_id` alone collides across them.
    """
    golds = golds or {}
    results: List[Result] = []
    for s in samples:
        try:
            ans: Answer = orchestrator.run(s, budget)
        except Exception as exc:  # noqa: BLE001 — deliberate per-sample capture
            # Fail loud in the ROW, not fatally for the RUN: one structurally broken sample
            # (e.g. an unreadable video) must never zero out an entire multi-hour benchmark shard. The row self-reports the failure verbatim; downstream
            # accuracy treats it as unanswered; summary["errors"] surfaces the count.
            from ..interfaces import CostRecord

            results.append(Result(
                video_id=s.video_id,
                question_id=s.question_id,
                prediction="",
                gold=golds.get(s.key),
                correct=None,
                cost=CostRecord(),
                status="error",
                error=f"{type(exc).__name__}: {exc}",
            ))
            continue
        gold = golds.get(s.key)
        # Lift the backend's arm record off the debug trace so it
        # persists into results.jsonl; None when the backend recorded none (e.g. fakes).
        trace0 = ans.rounds_trace[0] if ans.rounds_trace else None
        arm = trace0.get("arm") if isinstance(trace0, dict) else None
        results.append(Result(
            video_id=s.video_id,
            question_id=s.question_id,
            prediction=ans.text,
            gold=gold,
            correct=score_mcq(ans.text, gold),
            cost=ans.cost,
            arm=arm,
            rounds_trace=ans.rounds_trace or None,
        ))
    return results


def accuracy(results: List[Result]) -> Optional[float]:
    scored = [r.correct for r in results if r.correct is not None]
    return sum(scored) / len(scored) if scored else None
