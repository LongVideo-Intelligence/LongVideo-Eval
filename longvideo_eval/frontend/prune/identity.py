"""Identity pruner — keeps all tokens. This IS the dense baseline."""
from __future__ import annotations

from ..._registry import register
from ...interfaces import Budget, CostRecord, Pruner, VisualTokens


@register("prune", "identity")
class IdentityPruner(Pruner):
    """No-op. Zero incremental cost; every method is measured against this."""

    def prune(self, tokens: VisualTokens, query: str, budget: Budget) -> VisualTokens:
        return VisualTokens(
            tokens=tokens.tokens,
            num_tokens=tokens.num_tokens,
            realized_resolution=tokens.realized_resolution,
            cost=CostRecord(),
        )
