"""Exact Qwen visual-token math — pure, GPU-free, transformers-free at runtime.

Why this module exists: the harness must enforce the visual-token budget ITSELF, *before*
handing frames to the HF processor. Qwen's processor auto-rescales any input whose token count
would exceed its `max_pixels` (`longest_edge`) cap — silently changing the resolution under us
and voiding a cost-matched comparison. So we replicate Qwen's `smart_resize` + token-count
arithmetic here as pure integer math, decide (frame_count, max_pixels) up front, and then
cross-check the processor's reported token count against our own (assert_token_count,
fail-loudly).

Everything here mirrors the transformers source (URLs in comments next to each constant /
algorithm). No torch / transformers import — it unit-tests without a GPU.

Token model (both families):
  * `smart_resize(h, w)` rounds each side to a multiple of `factor = patch_size * merge_size`,
    preserving aspect ratio, clamping the *frame* pixel area into [min_pixels, max_pixels].
  * one frame -> (h_bar/factor) * (w_bar/factor) LLM tokens after the spatial merge.
  * a video groups `temporal_patch_size` consecutive frames into one temporal position; a clip
    of `num_frames` frames therefore costs `ceil(num_frames / temporal_patch_size)` temporal
    groups, each costing one frame's worth of tokens (the tail group is padded by repeating the
    last frame — see the `-T % temporal_patch_size` pad in the video processor).

3-D video clamp (Qwen3-VL): transformers 5.x's Qwen3-VL *video* smart_resize
(video_processing_qwen3_vl.py) clamps the 3-D total `t_bar * h_bar * w_bar` against
`longest_edge`/`shortest_edge`, where `t_bar` is the PADDED FRAME COUNT
(`ceil(frames/temporal)*temporal` — frames, NOT temporal groups) and
`beta = sqrt(frames * H * W / longest_edge)`. A per-frame 2-D clamp times temporal grouping
agrees with it whenever no clamp engages, but when clamping engages they diverge BOTH ways:
  * max side: the realized per-frame budget is `longest_edge / t_bar` = `max_pixels / tps` —
    HALF the 2-D per-frame assumption (786432 -> 393216 effective at any even frame count);
  * min side: the 3-D total-min trigger is much harder to trip than the per-frame 2-D min, so
    small-res clips the 2-D model would upscale pass through natively (e.g. 128F @ 240x336 ->
    2-D model 8960 vs processor 5120; see tests).
`video_tokens` therefore implements the 3-D clamp for the Qwen3 family (`video_clamp="3d"`),
mirroring the encoder's size mapping (per-frame recipe x temporal groups). Because the encoder
PRE-PADS the clip before the processor call, the processor's `frames` (beta's numerator) is the
PADDED count `t_bar`. Qwen2/2.5-VL video keeps the per-frame 2-D budget (`video_clamp="2d"`),
matching its processor unconditionally. `assert_token_count` remains the runtime tripwire
either way.

Sources (huggingface/transformers):
  * qwen2_vl/image_processing_qwen2_vl.py       -> canonical 2-D smart_resize + factor=28 family
  * qwen2_vl/video_processing_qwen2_vl.py        -> video min/max defaults, temporal padding, grid_t
  * qwen3_vl/video_processing_qwen3_vl.py         -> factor=32 family, video min/max defaults
  * qwen3_vl/modular_qwen3_vl.py                  -> vision config: patch=16, merge=2, temporal=2
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# --------------------------------------------------------------------------- #
# Verified constants (per model family). See module docstring for source URLs.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class QwenVisionConstants:
    """Vision-tower geometry + processor pixel budget for one Qwen-VL family.

    `factor` (= patch_size * merge_size) is the rounding granularity every resized side is a
    multiple of; it is derived in __post_init__ and cannot be set independently.
    `min_pixels` / `max_pixels` are the processor's *video-path* `shortest_edge` / `longest_edge`
    area budget (per-frame area for this module's 2-D model). Override them per call when the
    harness hands the processor an explicit cap.

    TWO temporal knobs, deliberately separate:

      * `temporal_patch_size` — the PROCESSOR's frame grouping. It decides the tail padding
        (`-n % temporal_patch_size` frames repeated) and the reported `video_grid_thw[0]`
        (`grid_t = padded_frames // temporal_patch_size`).
      * `temporal_merge_size` — an LLM-side temporal merge applied to the grid AFTER the
        processor: `llm_temporal_positions = ceil(grid_t / temporal_merge_size)`.

    On Qwen the whole temporal reduction happens in the processor (tps=2, then the grid IS the
    LLM's temporal axis), so `temporal_merge_size` defaults to **1**. It is a field rather than
    an assumption so that a family whose model merges further inside the LLM cannot be silently
    mis-counted.
    """

    patch_size: int
    merge_size: int
    temporal_patch_size: int
    min_pixels: int
    max_pixels: int
    # LLM-side temporal merge over the processor's grid_t (see the class docstring). 1 == "the
    # processor grid already IS the LLM temporal axis" (every Qwen family).
    temporal_merge_size: int = 1
    # Which video-path clamp the family's processor applies (module docstring "3-D video clamp"):
    #   "3d" — Qwen3 lineage: total t_bar*h_bar*w_bar clamped against the size budget
    #          (video_processing_qwen3_vl.py);
    #   "2d" — Qwen2/2.5-VL lineage: per-frame h_bar*w_bar clamp (canonical smart_resize).
    video_clamp: str = "3d"
    factor: int = 0  # derived = patch_size * merge_size

    def __post_init__(self) -> None:
        object.__setattr__(self, "factor", self.patch_size * self.merge_size)
        if self.patch_size <= 0 or self.merge_size <= 0 or self.temporal_patch_size <= 0:
            raise ValueError("patch_size, merge_size, temporal_patch_size must be positive")
        if self.temporal_merge_size <= 0:
            raise ValueError(
                f"temporal_merge_size must be positive, got {self.temporal_merge_size}"
            )
        if not (0 < self.min_pixels <= self.max_pixels):
            raise ValueError(
                f"require 0 < min_pixels <= max_pixels, got {self.min_pixels}, {self.max_pixels}"
            )
        if self.video_clamp not in ("3d", "2d"):
            raise ValueError(f"video_clamp must be '3d' or '2d', got {self.video_clamp!r}")


# Qwen3-VL family. Vision config patch=16, merge=2, temporal=2 (=> factor 32).
#   qwen3_vl/modular_qwen3_vl.py: patch_size=16, spatial_merge_size=2, temporal_patch_size=2
#     https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/modular_qwen3_vl.py
#   video min/max = size {shortest_edge: 128*32*32, longest_edge: 32*32*768}
#     https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/video_processing_qwen3_vl.py
#   TRANSFORMERS 5.x NOTE: the kwarg surface moved (`min_pixels`/`max_pixels` per-call kwargs
#   are GONE; the budget is `size={"shortest_edge","longest_edge"}`, still AREA semantics), and
#   the LOADED processor defaults now express the 3-D total budget (4096 / 25165824 = this
#   per-frame recipe scaled by a 32-group nominal). These constants are therefore the HARNESS's
#   per-frame budget recipe — not a claim about the installed processor's defaults. The encoder
#   passes them explicitly per call, scaled by temporal_groups where the 3-D clamp applies (see
#   frontend/encode/encoder.py), with assert_token_count as the runtime tripwire.
QWEN3_VL = QwenVisionConstants(
    patch_size=16,
    merge_size=2,
    temporal_patch_size=2,
    min_pixels=128 * 32 * 32,   # 131072
    max_pixels=32 * 32 * 768,   # 786432
)

# Qwen2.5-VL family (reuses Qwen2-VL's image + video processors verbatim). patch=14 (=> factor 28).
#   qwen2_vl/image_processing_qwen2_vl.py: patch_size=14, temporal_patch_size=2, merge_size=2;
#     canonical smart_resize(factor=28, min=56*56, max=14*14*4*1280) [image-path defaults]
#     https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py
#   video min/max = size {shortest_edge: 128*28*28, longest_edge: 28*28*768}
#     https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_vl/video_processing_qwen2_vl.py
#   (image-path defaults differ: min_pixels=56*56=3136, max_pixels=28*28*1280=1003520)
#   The `qwen2_5_vl` video path clamps PER-FRAME 2-D: qwen2_vl/video_processing_qwen2_vl.py
#   ::_preprocess (imported verbatim by qwen2_5_vl) applies the canonical smart_resize per
#   frame with min_pixels=size["shortest_edge"], max_pixels=size["longest_edge"]; no grid_t
#   scaling, no 3-D total. So size bounds scale by grid_t only for "3d" families.
QWEN2_5_VL = QwenVisionConstants(
    patch_size=14,
    merge_size=2,
    temporal_patch_size=2,
    min_pixels=128 * 28 * 28,   # 100352
    max_pixels=28 * 28 * 768,   # 602112
    video_clamp="2d",           # qwen2_5_vl video processor: per-frame 2-D (see above)
)

# --------------------------------------------------------------------------- #
# smart_resize — EXACT replication of transformers' 2-D algorithm.
# --------------------------------------------------------------------------- #
def smart_resize(
    height: int, width: int, *, factor: int, min_pixels: int, max_pixels: int
) -> tuple[int, int]:
    """Return (h_bar, w_bar), each a multiple of `factor`, area in [min_pixels, max_pixels].

    Byte-for-byte the transformers Qwen2-VL image `smart_resize` (also imported by the
    Qwen2/2.5-VL video processor), including its exact round/floor/ceil placement:
        h_bar = round(h/factor)*factor
        if h_bar*w_bar > max: floor(h/beta/factor)*factor,  beta = sqrt(h*w / max)
        elif h_bar*w_bar < min: ceil(h*beta/factor)*factor, beta = sqrt(min / (h*w))
    Source: qwen2_vl/image_processing_qwen2_vl.py.

    Raises ValueError on the upstream's degenerate condition (aspect ratio > 200) and on
    non-positive dimensions (upstream would ZeroDivisionError; we fail loudly and explicitly).
    """
    if height <= 0 or width <= 0:
        raise ValueError(f"height and width must be positive, got {height}x{width}")
    if factor <= 0:
        raise ValueError(f"factor must be positive, got {factor}")
    # Upstream guard: `max(h, w) / min(h, w) > 200` -> ValueError (aspect too extreme).
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got "
            f"{max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


# --------------------------------------------------------------------------- #
# Token counts.
# --------------------------------------------------------------------------- #
def frame_tokens(height: int, width: int, c: QwenVisionConstants) -> int:
    """LLM visual tokens for ONE frame at native (height, width), after smart_resize + merge.

    = (h_bar / factor) * (w_bar / factor), where (h_bar, w_bar) = smart_resize(...). Since h_bar,
    w_bar are multiples of factor (= patch*merge), this equals grid_h*grid_w / merge^2 exactly.
    """
    h_bar, w_bar = smart_resize(
        height, width, factor=c.factor, min_pixels=c.min_pixels, max_pixels=c.max_pixels
    )
    return (h_bar // c.factor) * (w_bar // c.factor)


def temporal_groups(num_frames: int, c: QwenVisionConstants) -> int:
    """Number of temporal patches for `num_frames` frames (tail zero-padded by frame repeat).

    grid_t = ceil(num_frames / temporal_patch_size). Mirrors the video processor's
    `if pad := -T % temporal_patch_size` padding then `grid_t = T // temporal_patch_size`.
    """
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    return math.ceil(num_frames / c.temporal_patch_size)


def merge_groups(num_frames: int, c: QwenVisionConstants) -> int:
    """LLM-side temporal positions for `num_frames`: ``ceil(temporal_groups(n) / merge)``.

    The second half of the two-stage temporal reduction documented on `QwenVisionConstants`:
    `temporal_groups` is what the PROCESSOR reports as ``video_grid_thw[0]``, this is what the
    LANGUAGE MODEL actually charges. Identity on every Qwen family (`temporal_merge_size == 1`).
    """
    return math.ceil(temporal_groups(num_frames, c) / c.temporal_merge_size)


def merged_tokens(grid_thw, c: QwenVisionConstants) -> int:
    """LLM visual tokens implied by a REALIZED processor grid ``(t, h, w)``.

    ``ceil(t / temporal_merge_size) * h * w // merge_size^2``. This is the one place the
    grid -> token reduction is written down, so a family whose processor grid is NOT already
    the LLM temporal axis cannot be mis-counted by an open-coded ``t*h*w // merge^2``.
    Byte-identical to that expression whenever ``temporal_merge_size == 1`` — i.e. for every
    Qwen path, including the image pathway (per-image grids are ``(1, h, w)``, and
    ``ceil(1/m) == 1`` for any m).
    """
    t, h, w = (int(v) for v in grid_thw)
    return (math.ceil(t / c.temporal_merge_size) * h * w) // (c.merge_size * c.merge_size)


def video_tokens(num_frames: int, height: int, width: int, c: QwenVisionConstants) -> int:
    """Total LLM visual tokens for a uniformly-sized clip of `num_frames` frames at (h, w).

    ``video_clamp="2d"`` (Qwen2/2.5-VL): temporal_groups * frame_tokens — the per-frame 2-D
    clamp, matching that family's video processor unconditionally.

    ``video_clamp="3d"`` (Qwen3 lineage, module docstring "3-D video clamp"): transformers'
    video smart_resize clamps the 3-D total ``t_bar * h_bar * w_bar`` against the size budget
    with ``t_bar`` = the PADDED frame count (video_processing_qwen3_vl.py). The budget mirrored
    here is what the encoder actually hands the processor: the per-frame recipe scaled by
    temporal groups (``size = {shortest: min_pixels*grid_t, longest: max_pixels*grid_t}``).
    Because the encoder pre-pads the clip before the processor call, the processor's
    ``num_frames`` (beta's numerator in the source) IS ``t_bar``. Identical to the 2-D model
    whenever neither clamp triggers; when a clamp engages the two diverge.

    Accounts for temporal padding either way: a clip whose frame count is not a multiple of
    temporal_patch_size still pays for the full tail group. The LLM-side temporal merge
    (`merge_groups`) is applied on top of the grid in both branches; it is the identity for
    every Qwen family.
    """
    grid_t = temporal_groups(num_frames, c)
    llm_t = merge_groups(num_frames, c)
    if c.video_clamp == "2d":
        return llm_t * frame_tokens(height, width, c)

    # --- Qwen3-VL 3-D video clamp (video_processing_qwen3_vl.py) ---
    if height <= 0 or width <= 0:
        raise ValueError(f"height and width must be positive, got {height}x{width}")
    factor = c.factor
    t_bar = grid_t * c.temporal_patch_size          # padded frame count (frames, NOT groups)
    max_pixels = c.max_pixels * grid_t              # encoder size mapping: recipe x groups
    min_pixels = c.min_pixels * grid_t
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if t_bar * h_bar * w_bar > max_pixels:
        # Effective per-frame budget = max_pixels/t_bar = c.max_pixels/temporal_patch_size —
        # HALF the per-frame recipe at tps=2.
        beta = math.sqrt((t_bar * height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif t_bar * h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (t_bar * height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return llm_t * (h_bar // factor) * (w_bar // factor)


# --------------------------------------------------------------------------- #
# Budget solvers (pure, fail loudly). See round-trip properties in the tests.
# --------------------------------------------------------------------------- #
def max_frames_for_budget(
    token_budget: int, height: int, width: int, c: QwenVisionConstants
) -> int:
    """Most frames of native (h, w) whose video_tokens fit `token_budget`.

    Each temporal group of `temporal_patch_size` frames costs `frame_tokens(h, w)`; the answer is
    `(token_budget // frame_tokens) * temporal_patch_size`. Raises ValueError if even the temporal
    minimum (a single frame == one padded group) does not fit.

    Round-trip: video_tokens(answer, ...) <= token_budget always. TIGHTNESS
    (video_tokens(answer + 1, ...) > token_budget) holds exactly under the 2-D model
    (`video_clamp="2d"`, or "3d" when no clamp engages); when the Qwen3-VL 3-D max clamp
    engages, the per-frame cost SHRINKS as frames grow (beta grows with t_bar), so this solver
    is CONSERVATIVE — the answer always fits, but more frames might too.
    """
    if token_budget <= 0:
        raise ValueError(f"token_budget must be positive, got {token_budget}")
    per_group = frame_tokens(height, width, c)  # tokens for one LLM temporal position
    max_groups = token_budget // per_group
    if max_groups < 1:
        raise ValueError(
            f"infeasible: one frame at {height}x{width} costs {per_group} tokens "
            f"> budget {token_budget}"
        )
    # Frames per LLM temporal position = processor grouping x LLM merge (the latter is 1 on
    # every Qwen family, so this is byte-identical there).
    return max_groups * c.temporal_patch_size * c.temporal_merge_size


def max_pixels_for_budget(token_budget: int, num_frames: int, c: QwenVisionConstants) -> int:
    """Max `max_pixels` (longest_edge) cap to hand the processor so `num_frames` fit the budget.

    With llm_t = merge_groups(num_frames) (== temporal_groups on every Qwen family), the
    per-frame token allowance is `token_budget // llm_t`; smart_resize guarantees
    h_bar*w_bar <= max_pixels, i.e.
    tokens/frame = h_bar*w_bar / factor^2 <= max_pixels / factor^2. So the cap is
    `(token_budget // grid_t) * factor^2`. Raises ValueError if grid_t frames cannot fit even at
    one token per frame (the smallest possible frame, factor x factor).

    Round-trip: any frame resized under this cap yields video_tokens <= token_budget — under the
    Qwen3-VL 3-D clamp (`video_clamp="3d"`) with ~2x extra headroom (its effective per-frame
    budget is cap/temporal_patch_size when the max clamp engages), so the guarantee holds
    conservatively. TIGHTNESS (cap + factor^2 admits an over-budget frame) is a 2-D-model
    property only.
    """
    if token_budget <= 0:
        raise ValueError(f"token_budget must be positive, got {token_budget}")
    llm_t = merge_groups(num_frames, c)   # == temporal_groups on every Qwen family
    tokens_per_frame = token_budget // llm_t
    if tokens_per_frame < 1:
        raise ValueError(
            f"infeasible: {num_frames} frames need >= {llm_t} tokens (1/frame) "
            f"> budget {token_budget}"
        )
    return tokens_per_frame * c.factor * c.factor


# --------------------------------------------------------------------------- #
# Fail-loudly cross-check (used by the HF backend).
# --------------------------------------------------------------------------- #
def assert_token_count(expected: int, processor_reported: int) -> None:
    """Raise if our computed visual-token count disagrees with the processor's reported count.

    This is the guard that catches a silent processor auto-rescale (or a math drift) at runtime:
    the harness computes `expected` up front, then asserts the processor honored it.
    """
    if expected != processor_reported:
        raise ValueError(
            "Qwen visual-token count mismatch: harness computed "
            f"{expected} tokens but the processor reported {processor_reported}. "
            "The processor likely auto-rescaled the input (max_pixels cap engaged) or the token "
            "math is out of sync with the installed transformers version."
        )
