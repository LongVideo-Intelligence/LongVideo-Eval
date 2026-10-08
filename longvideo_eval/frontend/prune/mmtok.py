"""MMTok token pruning (Dong et al., ICLR 2026), adapted to the Qwen3-VL family.

MMTok selects the subset of visual tokens that best COVERS two things at once: the question
(text->vision similarity) and the rest of the image content (vision->vision similarity). The
selection is a greedy maximum-coverage problem over ``Combined = [P; alpha*Q]``.

THIS ADAPTATION. The paper targets LLaVA-style models and selects pre-projector CLIP tokens.
Qwen3-VL has no CLIP tower, so here MMTok runs at the ``pre_llm`` seam (see
``backend/hf/prune_seam.py``) on the model's own features:
  * ``P`` = text->vision softmax (``tv_temp=0.01``) over the POST-merger video tokens;
  * ``Q`` = vision->vision softmax (``vv_temp=0.2``) over the PRE-merger tower states, averaged
    over each 2x2 merge window; ``alpha=0.5`` weights ``Q``;
  * the text side is the question's keywords embedded with the LLM's own input-embedding
    table. The question comes from the harness ``query``.
It is pure SELECTION: the kept tokens are written back unchanged.

SPEED. The maximiser is lazy greedy (``pre_llm_compress.lazy_greedy_max_coverage``), which
picks the same tokens as the textbook loop without re-scoring every token at every step.
``prune_kwargs={"greedy": "plain"}`` restores the textbook loop for cross-checks.

DEFAULTS. ``keep_ratio=0.25``; ``Budget.token_budget`` overrides it. The realized count
equals the target, and the backend records it in the results arm.
"""
from __future__ import annotations

from ..._registry import register
from ...backend.hf.prune_seam import PATCH_SPEC_KEY, PatchSpec
from ...frontend.encode.encoder import QWEN_TOKENS_KIND
from ...interfaces import Budget, CostRecord, Pruner, VisualTokens

_DEFAULT_ALPHA = 0.5
_DEFAULT_TV_TEMP = 0.01
_DEFAULT_VV_TEMP = 0.2
_DEFAULT_KEEP_RATIO = 0.25


@register("prune", "mmtok")
class MMTokPruner(Pruner):
    """MMTok greedy multimodal max-coverage at the pre_llm seam. Deterministic.

    Seam-only: the coverage signal is the served model's own tower features plus the
    LLM-embedded question, which exist only inside ``model.generate``.
    """

    def __init__(
        self,
        keep_ratio: float = _DEFAULT_KEEP_RATIO,
        alpha: float = _DEFAULT_ALPHA,
        tv_temp: float = _DEFAULT_TV_TEMP,
        vv_temp: float = _DEFAULT_VV_TEMP,
        greedy: str = "lazy",
    ) -> None:
        if not (0.0 < keep_ratio <= 1.0):
            raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}")
        if greedy not in ("lazy", "plain"):
            raise ValueError(f"greedy must be 'lazy' or 'plain', got {greedy!r}")
        self.keep_ratio = keep_ratio
        self.alpha = alpha
        self.tv_temp = tv_temp
        self.vv_temp = vv_temp
        self.greedy = greedy

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
                "MMTokPruner is seam-only (pre_llm): its coverage signal is the served model's "
                "own tower features + LLM-embedded question, which exist only inside "
                f"model.generate. Requires kind={QWEN_TOKENS_KIND!r}; got "
                f"{type(pkg).__name__ if not isinstance(pkg, dict) else pkg.get('kind')!r}."
            )

        # The target count, with MMTok's own rounding (round, not ceil) so that it equals what
        # the kernel selects inside the backend. The backend arm records the realized count.
        if budget.token_budget is not None:
            keep = int(budget.token_budget)
        else:
            keep = int(round(pre * self.keep_ratio))
        keep = max(1, min(keep, pre))
        if keep >= pre:
            return VisualTokens(  # nothing dropped — leave the dense package untouched
                tokens=pkg, num_tokens=pre,
                realized_resolution=tokens.realized_resolution, cost=CostRecord(),
            )

        spec = PatchSpec(
            method="mmtok", patch_point="pre_llm", pre_prune_tokens=pre,
            keep_indices=tuple(range(keep)), compression="mmtok",
            params={"keep_ratio": self.keep_ratio, "alpha": self.alpha,
                    "tv_temp": self.tv_temp, "vv_temp": self.vv_temp,
                    "greedy": self.greedy,
                    "token_budget": budget.token_budget, "expansion": 1.0},
        )
        new_pkg = dict(pkg)                  # shallow copy; never mutate the encoder's dict
        new_pkg[PATCH_SPEC_KEY] = spec
        return VisualTokens(
            tokens=new_pkg, num_tokens=keep,  # the effective LLM prefill shrinks to this
            realized_resolution=tokens.realized_resolution, cost=CostRecord(),
        )
