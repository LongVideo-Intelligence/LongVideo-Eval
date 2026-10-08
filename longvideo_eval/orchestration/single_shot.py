"""Single-shot orchestrator — the baseline path and every single-pass method.

Wires decode -> select -> encode -> prune -> generate exactly once, merging each stage's
incremental cost into one CostRecord.
"""
from __future__ import annotations

from .._registry import register
from ..interfaces import (
    Answer, Budget, Decoder, Encoder, LLMBackend, Orchestrator, Pruner, Sample, Selector,
)
from ..metering.meter import Meter


@register("orchestrate", "single_shot")
class SingleShotOrchestrator(Orchestrator):
    def __init__(self, decoder: Decoder, selector: Selector, encoder: Encoder,
                 pruner: Pruner, backend: LLMBackend) -> None:
        self.decoder = decoder
        self.selector = selector
        self.encoder = encoder
        self.pruner = pruner
        self.backend = backend

    def run(self, sample: Sample, budget: Budget) -> Answer:
        meter = Meter()

        # Each stage is timed (wall_seconds) AND has its incremental cost folded in.
        with meter.stage("decode"):
            pool = self.decoder.decode(sample.video_path, budget)
        meter.fold(pool.cost)

        with meter.stage("select"):
            selection = self.selector.select(pool, sample.query, budget, context=None)
        meter.fold(selection.cost)       # select increment (e.g. keyframe scoring — frontend cost)

        with meter.stage("encode"):
            tokens = self.encoder.encode(pool, selection, budget)
        meter.fold(tokens.cost)          # encode increment

        with meter.stage("prune"):
            tokens = self.pruner.prune(tokens, sample.query, budget)
        meter.fold(tokens.cost)          # prune increment (identity = 0)

        with meter.stage("generate"):
            answer = self.backend.generate(tokens, sample.query, budget,
                                           subtitles=sample.subtitles)
        meter.fold(answer.cost)

        meter.record.rounds = 0          # single-shot (interface: rounds == rounds actually used)
        answer.cost = meter.record
        return answer
