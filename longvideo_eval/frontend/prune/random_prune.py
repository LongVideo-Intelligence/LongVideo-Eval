"""Random token pruning — the control every learned pruner is compared against.

Same prune point, token granularity and accounting as the learned pruners (the ``pre_llm``
seam, see ``backend/hf/prune_seam.py``); the only difference is the keep set, which is a
seeded uniform draw with no signal from the video or the question. A learned pruner that
does not beat this at the same retention has not shown that its signal matters.

This module also owns :func:`keep_count`, the retention arithmetic shared by every pruner.
"""
from __future__ import annotations

import hashlib
import math
import random

from ..._registry import register
from ...backend.hf.prune_seam import PATCH_SPEC_KEY, PatchSpec, subselect_rows
from ...frontend.encode.encoder import QWEN_TOKENS_KIND
from ...interfaces import Budget, CostRecord, Pruner, VisualTokens

_DEFAULT_KEEP_RATIO = 0.25


def keep_count(pre_prune_tokens: int, budget: Budget, keep_ratio: float) -> int:
    """Surviving token count: ``token_budget`` if set, else ``ceil(pre * keep_ratio)`` (>= 1).

    This is the PRUNER-level count (the control's real keep set; the learned pruners'
    placeholder / accounting count). Each learned kernel applies its own published rounding at
    generate time, and the backend records the realized count.
    """
    if pre_prune_tokens <= 0:
        raise ValueError(f"pre_prune_tokens must be positive, got {pre_prune_tokens}")
    if budget.token_budget is not None:
        k = min(pre_prune_tokens, int(budget.token_budget))
    else:
        k = math.ceil(pre_prune_tokens * keep_ratio)
    return max(1, min(k, pre_prune_tokens))


def seed_from_query(query: str, base_seed: int) -> int:
    """A per-sample seed that is stable across processes (``hash()`` is salted per process)."""
    digest = hashlib.sha256(f"{base_seed}:{query}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def random_keep_indices(pre_prune_tokens: int, keep: int, seed: int) -> tuple[int, ...]:
    """Seeded, sorted, unique subset of ``range(pre_prune_tokens)`` of size ``keep``."""
    rng = random.Random(seed)
    return tuple(sorted(rng.sample(range(pre_prune_tokens), keep)))


@register("prune", "random")
class RandomPruner(Pruner):
    """Uniformly random visual-token drop to ``token_budget`` (else ``keep_ratio``).

    On the Qwen package it declares a ``pre_llm`` patch; on a materialized token array (fakes,
    eager backends) it subselects the rows directly.
    """

    def __init__(self, keep_ratio: float = _DEFAULT_KEEP_RATIO, seed: int = 0) -> None:
        if not (0.0 < keep_ratio <= 1.0):
            raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}")
        self.keep_ratio = keep_ratio
        self.seed = seed

    def prune(self, tokens: VisualTokens, query: str, budget: Budget) -> VisualTokens:
        pre = tokens.num_tokens
        if pre <= 0:
            return VisualTokens(
                tokens=tokens.tokens, num_tokens=0,
                realized_resolution=tokens.realized_resolution, cost=CostRecord(),
            )

        keep = keep_count(pre, budget, self.keep_ratio)
        indices = random_keep_indices(pre, keep, seed_from_query(query, self.seed))
        pkg = tokens.tokens
        if not (isinstance(pkg, dict) and pkg.get("kind") == QWEN_TOKENS_KIND):
            return VisualTokens(
                tokens=subselect_rows(pkg, indices), num_tokens=keep,
                realized_resolution=tokens.realized_resolution, cost=CostRecord(),
            )
        if len(indices) == pre:
            return VisualTokens(  # nothing dropped — leave the dense package untouched
                tokens=pkg, num_tokens=pre,
                realized_resolution=tokens.realized_resolution, cost=CostRecord(),
            )
        spec = PatchSpec(
            method="random", patch_point="pre_llm", pre_prune_tokens=pre,
            keep_indices=indices, compression="random",
            params={"keep_ratio": self.keep_ratio, "seed": self.seed,
                    "token_budget": budget.token_budget},
        )
        new_pkg = dict(pkg)                  # shallow copy; never mutate the encoder's dict
        new_pkg[PATCH_SPEC_KEY] = spec
        return VisualTokens(
            tokens=new_pkg, num_tokens=keep,  # the effective LLM prefill shrinks to this
            realized_resolution=tokens.realized_resolution, cost=CostRecord(),
        )
