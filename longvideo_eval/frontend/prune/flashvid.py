"""FlashVID token merging (Fan et al., 2026), adapted to the Qwen3-VL family.

FlashVID compresses video tokens with a training-free tree: it segments the video where
consecutive frames stop looking alike, then inside each segment keeps a set of
attention-salient and mutually diverse tokens and merges the rest across time and space.

THE FOUR STAGES (``backend/hf/pre_llm_compress.flashvid_compression`` + ``_flashvid_torch``).
  1. Dynamic segmentation on frame-mean transition similarity < ``segment_threshold``;
     ``min_segment_num`` / ``complementary_segment`` enforce a floor on the segment count.
  2. Attention/diversity token selection per segment: greedy farthest-point over
     attention-calibrated cosine distance. The per-frame budget is
     ``ceil(tokens_per_frame * keep_ratio)``, of which ``ceil(budget * alpha)`` goes to this
     stage (``alpha`` is the split between the two stages, NOT a prune ratio).
  3. Temporal average merging when ``temporal_threshold < 1.0`` (VALUE-MODIFYING).
  4. Density-peak clustering that merges the remaining tokens onto cluster centers.

THIS ADAPTATION. The paper targets LLaVA-style models with a CLS attention. Qwen3-VL has no
CLS token, so the attention signal is the mean self-attention of the last vision block, and
the compression runs on post-merger tokens at the ``pre_llm`` seam (see
``backend/hf/prune_seam.py``). FlashVID's optional second stage, pruning inside the LLM, is
not used here: the whole reduction happens before the language model.

DEFAULTS. ``alpha=0.7``, ``segment_threshold=0.9``, ``min_segment_num=8``,
``complementary_segment=True``, ``temporal_threshold=0.8``, ``keep_ratio=0.25``
(``Budget.token_budget`` overrides it, as the ratio that yields that count).

THE REALIZED COUNT IS DATA-DEPENDENT. The budget is applied per frame and the merge stages
decide how many tokens survive, so the count the LLM sees differs slightly from
``keep_ratio * tokens``; ``VisualTokens.num_tokens`` here is the target, and the backend
records the realized count in the results arm. As in the reference implementation, a segment
that is a single frame clusters over the whole frame, so a cluster centre can coincide with
a token the selection stage already kept; that position is then counted twice.
"""
from __future__ import annotations

from ..._registry import register
from ...backend.hf.prune_seam import PATCH_SPEC_KEY, PatchSpec
from ...frontend.encode.encoder import QWEN_TOKENS_KIND
from ...interfaces import Budget, CostRecord, Pruner, VisualTokens
from .random_prune import keep_count

_DEFAULT_KEEP_RATIO = 0.25
_DEFAULT_ALPHA = 0.7                # split between selection and merging (NOT a prune ratio)
_DEFAULT_SEGMENT_THRESHOLD = 0.9
_DEFAULT_MIN_SEGMENT_NUM = 8
_DEFAULT_TEMPORAL_THRESHOLD = 0.8


@register("prune", "flashvid")
class FlashVidPruner(Pruner):
    """FlashVID vision-side tree merge at the pre_llm seam.

    Seam-only: its attention signal is computed from the served model's vision tower inside
    ``model.generate``.
    """

    def __init__(
        self,
        keep_ratio: float = _DEFAULT_KEEP_RATIO,
        alpha: float = _DEFAULT_ALPHA,
        segment_threshold: float = _DEFAULT_SEGMENT_THRESHOLD,
        min_segment_num: int = _DEFAULT_MIN_SEGMENT_NUM,
        temporal_threshold: float = _DEFAULT_TEMPORAL_THRESHOLD,
        do_segment: bool = True,
        complementary_segment: bool = True,
    ) -> None:
        if not (0.0 < keep_ratio <= 1.0):
            raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}")
        self.keep_ratio = keep_ratio
        self.alpha = alpha
        self.segment_threshold = segment_threshold
        self.min_segment_num = min_segment_num
        self.temporal_threshold = temporal_threshold
        self.do_segment = do_segment
        self.complementary_segment = complementary_segment

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
                "FlashVidPruner is seam-only (pre_llm): its attention signal is computed from "
                "the served model's vision tower, which runs only inside model.generate. "
                f"Requires kind={QWEN_TOKENS_KIND!r}; got "
                f"{type(pkg).__name__ if not isinstance(pkg, dict) else pkg.get('kind')!r}."
            )

        keep = keep_count(pre, budget, self.keep_ratio)
        if keep >= pre:
            return VisualTokens(
                tokens=pkg, num_tokens=pre,
                realized_resolution=tokens.realized_resolution, cost=CostRecord(),
            )

        spec = PatchSpec(
            method="flashvid", patch_point="pre_llm", pre_prune_tokens=pre,
            keep_indices=tuple(range(keep)), compression="flashvid",
            params={"keep_ratio": self.keep_ratio, "alpha": self.alpha,
                    "segment_threshold": self.segment_threshold,
                    "min_segment_num": self.min_segment_num,
                    "temporal_threshold": self.temporal_threshold,
                    "do_segment": self.do_segment,
                    "complementary_segment": self.complementary_segment,
                    "token_budget": budget.token_budget, "expansion": 1.0,
                    "token_selection_method": "attn_div", "llm_pruning": "off"},
        )
        new_pkg = dict(pkg)
        new_pkg[PATCH_SPEC_KEY] = spec
        return VisualTokens(
            tokens=new_pkg, num_tokens=keep,  # the effective LLM prefill shrinks to this
            realized_resolution=tokens.realized_resolution, cost=CostRecord(),
        )
