"""VisionZip token pruning (Yang et al., CVPR 2025), adapted to the Qwen3-VL family.

VisionZip keeps two kinds of visual tokens: a few DOMINANT tokens that attract the most
attention in the vision tower, and CONTEXTUAL tokens that summarise everything else.

THE TWO STAGES (``backend/hf/pre_llm_compress.visionzip_compression``).
  * Dominant tokens: the top-K tokens by attention received, kept unchanged.
  * Contextual tokens: the remaining tokens are assigned to evenly spaced targets by the
    cosine similarity of their attention key vectors and VALUE-AVERAGED into them
    (``contextual = target + mean(assigned others)``). This changes token values, which is
    why the seam writes tokens back instead of only dropping.

THIS ADAPTATION. The paper targets LLaVA-style models and reads the CLS token's attention.
Qwen3-VL has no CLS token, so the dominance signal is the mean self-attention of the last
vision block (the attention each token RECEIVES), and the method runs on post-merger tokens
at the ``pre_llm`` seam (see ``backend/hf/prune_seam.py``).

DEFAULTS. ``alpha=0.928571`` is the dominant share of the kept tokens (13 of every 14).
``keep_ratio=0.25``; ``Budget.token_budget`` overrides it. The kept count is
``round(N * keep_ratio)``, and the backend records the realized count in the results arm.
"""
from __future__ import annotations

from ..._registry import register
from ...backend.hf.prune_seam import PATCH_SPEC_KEY, PatchSpec
from ...frontend.encode.encoder import QWEN_TOKENS_KIND
from ...interfaces import Budget, CostRecord, Pruner, VisualTokens

_DEFAULT_ALPHA = 0.928571     # dominant share of the kept tokens
_DEFAULT_KEEP_RATIO = 0.25


@register("prune", "visionzip")
class VisionZipPruner(Pruner):
    """VisionZip dominant tokens + contextual value-merge at the pre_llm seam.

    Seam-only: the dominance signal is computed from the served model's vision tower inside
    ``model.generate``.
    """

    def __init__(self, keep_ratio: float = _DEFAULT_KEEP_RATIO,
                 alpha: float = _DEFAULT_ALPHA) -> None:
        if not (0.0 < keep_ratio <= 1.0):
            raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}")
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.keep_ratio = keep_ratio
        self.alpha = alpha

    def prune(self, tokens: VisualTokens, query: str, budget: Budget) -> VisualTokens:
        pre = tokens.num_tokens
        if pre <= 0:
            return VisualTokens(
                tokens=tokens.tokens, num_tokens=0,
                realized_resolution=tokens.realized_resolution, cost=CostRecord(),
            )

        pkg = tokens.tokens
        if not (isinstance(pkg, dict) and pkg.get("kind") == QWEN_TOKENS_KIND):
            raise NotImplementedError(
                "VisionZipPruner is seam-only (pre_llm): its dominance signal is computed from "
                "the served model's vision tower, which runs only inside model.generate. "
                f"Requires kind={QWEN_TOKENS_KIND!r}; got "
                f"{type(pkg).__name__ if not isinstance(pkg, dict) else pkg.get('kind')!r}."
            )

        # The target count, with VisionZip's own rounding (round), so that it equals what the
        # kernel keeps inside the backend.
        if budget.token_budget is not None:
            keep = int(budget.token_budget)
        else:
            keep = int(round(pre * self.keep_ratio))
        keep = max(1, min(keep, pre))
        if keep >= pre:
            return VisualTokens(
                tokens=pkg, num_tokens=pre,
                realized_resolution=tokens.realized_resolution, cost=CostRecord(),
            )

        spec = PatchSpec(
            method="visionzip", patch_point="pre_llm", pre_prune_tokens=pre,
            keep_indices=tuple(range(keep)), compression="visionzip",
            params={"keep_ratio": self.keep_ratio, "alpha": self.alpha,
                    "token_budget": budget.token_budget, "expansion": 1.0},
        )
        new_pkg = dict(pkg)
        new_pkg[PATCH_SPEC_KEY] = spec
        return VisualTokens(
            tokens=new_pkg, num_tokens=keep,  # the effective LLM prefill shrinks to this
            realized_resolution=tokens.realized_resolution, cost=CostRecord(),
        )
