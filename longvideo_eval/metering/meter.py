"""Wall-time metering helper.

Cost accounting follows one convention: each stage's output data object carries ONLY its
own incremental cost in `.cost` (FramePool.cost = decode, VisualTokens.cost = encode or
prune increment, Answer.cost = generation). The Orchestrator merges them into the total,
so nothing is double-counted and nothing is self-reported by a paper.

`Meter.stage(kind)` times a stage's wall-clock and folds it into the record it yields;
the component fills in the stage-specific fields (decode_seconds, vit_flops, ...).
"""
from __future__ import annotations

import time
from contextlib import contextmanager

from ..interfaces import CostRecord


class Meter:
    """Accumulates a CostRecord across stages/rounds for one sample."""

    def __init__(self) -> None:
        self.record = CostRecord()

    def fold(self, other: CostRecord) -> None:
        """Merge another record (e.g. a stage output's .cost) into the running total."""
        self.record = self.record.merge(other)

    @contextmanager
    def stage(self, kind: str):
        """Time a stage; the elapsed wall-time is added to the running total.

        Yields a fresh CostRecord the caller fills with stage-specific fields; it is folded
        in on exit. `kind` is advisory (for future per-stage breakdowns).
        """
        rec = CostRecord()
        t0 = time.perf_counter()
        try:
            yield rec
        finally:
            rec.wall_seconds += time.perf_counter() - t0
            self.fold(rec)
