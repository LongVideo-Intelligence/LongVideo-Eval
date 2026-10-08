"""Core pluggable interfaces + the fair-cost data types.

Every efficiency method (token pruning / keyframe selection / resolution-aware allocation)
plugs into ONE of the stage protocols below, is fed the SAME decoded frame pool under the
SAME budget, and is metered in ONE place (`CostRecord`). Keeping this protocol in code is
what makes the harness an artifact rather than a script.

Pipeline (front-end grouped). A `Sample` (video + query + subtitles) enters the
Orchestrator, which drives:

    raw video
      -> Decoder    (frontend.decode)    -> FramePool
      -> Selector   (frontend.select)    -> Selection
      -> Encoder    (frontend.encode)    -> VisualTokens
      -> Pruner     (frontend.prune)     -> VisualTokens
      -> LLMBackend (backend)            -> Answer
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, Sequence


# --------------------------------------------------------------------------- #
# Budget + cost accounting (the fair axis)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Budget:
    """The shared allocation fed identically to every method.

    A method may spend UP TO these; the harness enforces the same ceiling on all
    methods so comparisons are cost-matched, not self-reported.
    """

    token_budget: Optional[int] = None      # visual tokens into the LLM
    frame_count: Optional[int] = None        # frames kept after selection
    resolution: Optional[float] = None       # per-frame scale factor r in (0, 1] vs NATIVE
                                             #   (r=1.0 native, r=0.25 = quarter PER SIDE =>
                                             #   ~r^2 visual tokens/frame; 16F@1.0 / 64F@0.5 /
                                             #   256F@0.25 all land at about the same token
                                             #   count because N * r^2 is held constant).
                                             #   None = the model's own default resizing.
                                             #   Applied at decode time.
    decode_budget: Optional[int] = None      # N_dec: frames allowed to decode
    max_rounds: int = 0                      # evidence-gathering rounds (0 = single-shot)
    # --- dual-stream (LoHi) allocation: (N, r_l, K, r_h) -------------------------------
    hi_i_count: Optional[int] = None         # K: frames that ALSO go through the image
                                             #   pathway at `hi_resolution`. They are a subset
                                             #   of the `frame_count` low-resolution frames.
    hi_resolution: Optional[float] = None    # r_h in (0, 1] vs NATIVE for those K frames.
                                             #   None = set per video: the largest size that
                                             #   keeps the run within `reference_frames`
                                             #   (native size when that is None too).
    reference_frames: Optional[int] = None   # the token budget of the setup, expressed as
                                             #   this many NATIVE-resolution video frames.
                                             #   The K images get what the low-resolution
                                             #   video leaves of it.
    presentation: str = "video"              # how the frames reach the LLM:
                                             #   "video" — one video block (default);
                                             #   "lohi"  — one video block at `resolution`
                                             #             (Lo-V) + K timestamped image blocks
                                             #             at `hi_resolution` (Hi-I).


@dataclass
class CostRecord:
    """One place for every cost the harness insists on measuring (not per-paper self-report).

    Front-end costs (decode, ViT) are the ones prior work under-reports; keep them
    first-class here. Accumulated across stages by the Meter.
    """

    # front-end
    decode_seconds: float = 0.0
    frames_decoded: int = 0
    vit_flops: float = 0.0
    frames_encoded: int = 0
    # ViT INGRESS: t*h*w from the REALIZED video_grid_thw, PRE merge_size^2 reduction — the
    # exact patch count entering the tower, the quantity that drives real ViT compute. Resize/
    # resample + heterogeneous native resolutions make realized counts drift from expected;
    # this makes the drift auditable per sample. Set by the encode stage.
    vit_patch_tokens: int = 0
    # back-end
    prefill_tokens: int = 0
    decode_tokens: int = 0        # generated answer tokens
    thinking_tokens: int = 0      # reasoning tokens (attributed separately, not "free")
    # LLM INGRESS: surviving VISUAL tokens actually entering the language model. Set by the
    # backend per generate call.
    llm_visual_tokens: int = 0
    # orchestration
    rounds: int = 0               # rounds actually used (0 = single-shot)
    wall_seconds: float = 0.0

    def merge(self, other: "CostRecord") -> "CostRecord":
        """Sum two records (e.g. across stages)."""
        return CostRecord(
            decode_seconds=self.decode_seconds + other.decode_seconds,
            frames_decoded=self.frames_decoded + other.frames_decoded,
            vit_flops=self.vit_flops + other.vit_flops,
            frames_encoded=self.frames_encoded + other.frames_encoded,
            vit_patch_tokens=self.vit_patch_tokens + other.vit_patch_tokens,
            prefill_tokens=self.prefill_tokens + other.prefill_tokens,
            decode_tokens=self.decode_tokens + other.decode_tokens,
            thinking_tokens=self.thinking_tokens + other.thinking_tokens,
            llm_visual_tokens=self.llm_visual_tokens + other.llm_visual_tokens,
            rounds=self.rounds + other.rounds,
            wall_seconds=self.wall_seconds + other.wall_seconds,
        )


# --------------------------------------------------------------------------- #
# Data flowing through the front-end
# --------------------------------------------------------------------------- #
@dataclass
class FramePool:
    """Decoded frames + how much they cost to produce. Output of the decode stage.

    All selectors operate on the SAME pool for a given (video, decode_budget) so no
    method secretly decodes more than another.
    """

    video_id: str
    frames: Any                     # backend-specific tensor/array [N,H,W,C]
    timestamps: Sequence[float]     # seconds, len == N
    fps: float
    cost: CostRecord = field(default_factory=CostRecord)
    # Realized per-frame (H, W) of `frames` AFTER any decode-time resize (Budget.resolution=r).
    # All frames in a pool share one (H, W) — uniform allocation. None = unknown/not recorded,
    # which the native path (resolution=None) may leave unset. Lets the resolution axis be
    # metered from the pool rather than re-derived downstream.
    frame_hw: Optional[tuple[int, int]] = None
    # NATIVE frame rate of the source video. `fps` above is the effective SAMPLING rate; the
    # encoder must not use it to rebuild frame indices for timestamps: round(t * sampling_rate)
    # snaps every timestamp to a grid of 1/sampling_rate seconds (up to ~112 s off for 16 frames
    # over an hour) and flattens irregular timestamps (keyframe picks) onto an even grid.
    # Decoders set this; None falls back to `fps`.
    native_fps: Optional[float] = None


@dataclass
class Selection:
    """A chosen frame subset + per-frame resolution (joint allocation). Output of select."""

    indices: Sequence[int]                    # into FramePool.frames
    per_frame_resolution: Optional[Sequence[float]] = None  # per-frame scale r (see Budget.resolution).
                                                            # Echoes what the DECODER realized: since
                                                            # Budget.resolution is applied at DECODE time,
                                                            # every shipped selector expands it into per-
                                                            # frame values when set, and leaves this None
                                                            # only when Budget.resolution is also None.
                                                            # None => native (no decode-time resize); the
                                                            # encoder then runs the family smart-resize
                                                            # path. It does NOT mean "re-apply Budget.
                                                            # resolution at encode" — frames are already
                                                            # sampled by then.
    # Dual-stream (LoHi): the subset of `indices` that additionally goes through the image
    # pathway at Budget.hi_resolution. None => single stream.
    hi_indices: Optional[Sequence[int]] = None
    signal: str = "uniform"                   # what drove the selection (uniform, query similarity, ...)
    cost: CostRecord = field(default_factory=CostRecord)  # select-stage increment (e.g. CLIP scoring)


@dataclass
class VisualTokens:
    """Encoded (and possibly pruned) visual tokens fed to the LLM. Output of encode/prune."""

    tokens: Any                     # [num_tokens, dim]
    num_tokens: int
    # Actual per-frame scale factor r the Encoder used; MUST echo Selection.per_frame_resolution
    # so the resolution axis is verifiable/meterable rather than silently discarded.
    realized_resolution: Optional[Sequence[float]] = None
    cost: CostRecord = field(default_factory=CostRecord)


@dataclass
class Answer:
    """Final model output for one sample."""

    text: str
    cost: CostRecord = field(default_factory=CostRecord)
    rounds_trace: list = field(default_factory=list)  # per-call debug trace (carries the run arm)


@dataclass
class Sample:
    """One evaluation instance — the unit an Orchestrator consumes.

    `subtitles` is the caption channel: whatever is fed here MUST be fed identically to the
    method AND to its baseline, so the harness can enforce/meter caption parity instead of it
    being smuggled through `query`.

    `question_id` is additive (appended last, default None) so existing positional
    `Sample(video_id, video_path, query, ...)` call sites are untouched. It exists because
    `video_id` is a data-SOURCE key, not a row identity: real Video-MME has ~3 questions per
    video, so golds/results indexing must use `.key`, never `video_id` alone, or same-video
    questions collide.
    """

    video_id: str
    video_path: str
    query: str
    subtitles: Optional[str] = None
    choices: Optional[Sequence[str]] = None   # MCQ options
    question_id: Optional[str] = None         # per-question id when the dataset has one

    @property
    def key(self) -> str:
        """Unique identity for golds/result indexing.

        `question_id` when the dataset provides one (multi-question-per-video benchmarks
        like Video-MME); falls back to `video_id` for 1-QA/video sources.
        """
        return self.question_id if self.question_id is not None else self.video_id


@dataclass
class RoundContext:
    """State threaded across rounds so a Selector can pick NEW evidence.

    round_index=0 with empty lists == the single-shot case.
    """

    round_index: int = 0
    prior_selections: list = field(default_factory=list)  # list[Selection] from past rounds
    evidence: list = field(default_factory=list)          # accumulated intermediate outputs
    # The Sample's question + MCQ options. Additive (defaulted) — existing RoundContext() call
    # sites and the single-shot path are untouched.
    query: str = ""
    choices: Optional[Sequence] = None


# --------------------------------------------------------------------------- #
# Stage protocols — one per pipeline stage. A "method" implements exactly one.
# --------------------------------------------------------------------------- #
class Decoder(ABC):
    """frontend.decode — raw video path -> FramePool under a decode budget."""

    @abstractmethod
    def decode(self, video_path: str, budget: Budget) -> FramePool: ...


class Selector(ABC):
    """frontend.select — pick a frame subset (+ per-frame resolution) for a query.

    Implemented by uniform, keyframe, and resolution-aware allocators. Single-shot methods
    select once (context=None).
    """

    @abstractmethod
    def select(self, pool: FramePool, query: str, budget: Budget,
               context: Optional[RoundContext] = None) -> Selection: ...


class Encoder(ABC):
    """frontend.encode — selected frames -> VisualTokens (ViT forward).

    MUST honor selection.per_frame_resolution (the resolution-aware allocation) and record
    what it actually used in VisualTokens.realized_resolution, so the resolution axis is
    metered rather than silently dropped.
    """

    @abstractmethod
    def encode(self, pool: FramePool, selection: Selection, budget: Budget) -> VisualTokens: ...


class Pruner(ABC):
    """frontend.prune — post-encoder token reduction under Budget.token_budget.

    Implemented by token-pruning methods. Identity pruner = dense baseline.
    """

    @abstractmethod
    def prune(self, tokens: VisualTokens, query: str, budget: Budget) -> VisualTokens: ...


class LLMBackend(ABC):
    """backend — visual tokens + prompt -> Answer.

    `subtitles` is the caption channel: when present it is included in the prompt, and the
    harness feeds the SAME subtitles to the baseline (parity).

    `max_new_tokens` is a per-CALL generation-cap override. None => the backend's own
    configured cap; the effective cap is recorded in the run arm.
    """

    @abstractmethod
    def generate(self, tokens: VisualTokens, prompt: str, budget: Budget,
                 subtitles: Optional[str] = None,
                 max_new_tokens: Optional[int] = None) -> Answer: ...


class Orchestrator(ABC):
    """orchestration — wires the stages for one Sample.

    Single-shot orchestrator: decode -> select -> encode -> prune -> generate, once.
    sample.subtitles flow to the backend identically for method and baseline.
    """

    @abstractmethod
    def run(self, sample: Sample, budget: Budget) -> Answer: ...


# --------------------------------------------------------------------------- #
# Registry glue — components self-register under a stable id.
# --------------------------------------------------------------------------- #
class StageComponent(Protocol):
    """Anything registrable (Decoder/Selector/Encoder/Pruner/Orchestrator)."""

    id: str
