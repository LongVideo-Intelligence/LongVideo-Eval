"""Hardware-free test doubles so the pipeline runs end-to-end without weights or videos.

These are NOT registered components (kept out of the registry) — they exist to exercise
the wiring + metering in tests and demos. Real components live under frontend/ and backend/.
"""
from __future__ import annotations

from typing import Optional

from .interfaces import (
    Answer, Budget, CostRecord, Decoder, Encoder, FramePool, LLMBackend,
    Selection, VisualTokens,
)
from .models.qwen_tokens import QWEN3_VL, QwenVisionConstants


class FakeDecoder(Decoder):
    """Emits `n` placeholder frames with a plausible decode cost."""

    def __init__(self, n: int = 64) -> None:
        self.n = n

    @property
    def duration_s(self) -> float:
        """The fake clip's duration: ``n`` frames at the 2.0 fps fiction ``decode`` uses."""
        return self.n / 2.0

    def decode(self, video_path: str, budget: Budget) -> FramePool:
        db = self.n if budget.decode_budget is None else budget.decode_budget  # 0 means 0
        n = min(self.n, db)
        return FramePool(
            video_id=video_path,
            frames=list(range(n)),
            timestamps=[i / 2 for i in range(n)],
            fps=2.0,
            cost=CostRecord(decode_seconds=0.01 * n, frames_decoded=n),
        )


class FakeEncoder(Encoder):
    """Turns selected frames into `tokens_per_frame` tokens each; echoes resolution.

    Mirrors the REAL encoder's ingress accounting: ``num_tokens`` is the
    MERGED count and ``cost.vit_patch_tokens`` the PRE-merge patch count (x merge_size^2),
    so the dense invariant ``vit_patch_tokens == llm_visual_tokens * merge^2`` is testable
    hardware-free.
    """

    tokens_per_frame = 64          # merged visual tokens per frame
    merge_size = 2                 # Qwen 2x2 patch merger

    def encode(self, pool: FramePool, selection: Selection, budget: Budget) -> VisualTokens:
        # Dual stream: the high-resolution frames are encoded a second time.
        k = len(selection.indices) + len(selection.hi_indices or [])
        nt = k * self.tokens_per_frame
        return VisualTokens(
            tokens=[0] * nt,
            num_tokens=nt,
            realized_resolution=selection.per_frame_resolution,
            cost=CostRecord(vit_flops=1e9 * k, frames_encoded=k,
                            vit_patch_tokens=nt * self.merge_size ** 2),
        )


class FakeClipScorer:
    """Deterministic stand-in for ``frontend.select.clip_scorer.ClipScorer`` (no weights).

    Frame ``i`` of ``n`` gets a smooth unit embedding, so neighbours are similar and distant
    frames are not; the query similarity peaks at a frame chosen from the query
    text. Enough structure for selection logic to be exercised without CLIP.
    """

    def embed_frames(self, frames):
        import math

        n = len(frames)
        out = []
        for i in range(n):
            x = i / max(1, n - 1)
            # 8 frequencies -> 16 dimensions, so up to 16 frames are linearly independent.
            v = [f(0.5 * math.pi * k * x) for k in range(1, 9) for f in (math.cos, math.sin)]
            norm = math.sqrt(sum(c * c for c in v))
            out.append([c / norm for c in v])
        return out

    def __call__(self, pool, query: str):
        n = len(pool.frames)
        peak = (sum(ord(ch) for ch in query) % n) if n else 0
        return [1.0 - abs(i - peak) / max(1, n) for i in range(n)]


class FakeBackend(LLMBackend):
    """Returns a fixed answer with a cost proportional to the token count.

    Mirrors the REAL backend's generate surface and RECORDS the per-call ``max_new_tokens`` it
    received (None per call when unset) so tests can assert what was forwarded.
    """

    def __init__(self) -> None:
        self.seen_max_new_tokens: list = []

    def generate(self, tokens: VisualTokens, prompt: str, budget: Budget,
                 subtitles: Optional[str] = None, max_new_tokens=None) -> Answer:
        self.seen_max_new_tokens.append(max_new_tokens)
        thinking = 50 if budget.max_rounds else 0
        return Answer(
            text="A",
            # llm_visual_tokens mirrors the real backend's dense ingress (== the merged count).
            cost=CostRecord(prefill_tokens=tokens.num_tokens, decode_tokens=1,
                            thinking_tokens=thinking, llm_visual_tokens=tokens.num_tokens),
        )


# The per-call kwarg surface Qwen3VLVideoProcessorInitKwargs accepts on transformers 5.x;
# anything else must raise exactly like 5.x validate_typed_dict.
_QWEN_VIDEO_KWARGS_5X = frozenset({
    "do_convert_rgb", "do_resize", "size", "default_to_square", "resample", "do_rescale",
    "rescale_factor", "do_normalize", "image_mean", "image_std", "do_center_crop", "do_pad",
    "crop_size", "data_format", "input_data_format", "device", "do_sample_frames",
    "video_metadata", "fps", "num_frames", "return_metadata", "return_tensors",
    "patch_size", "temporal_patch_size", "merge_size", "min_frames", "max_frames",
    "videos",
})


class _Grid:
    """Minimal stand-in for a torch tensor row: supports ``.tolist()``."""

    def __init__(self, values):
        self._values = list(values)

    def tolist(self):
        return list(self._values)


class FakeQwenVideoProcessor:
    """Callable double for ``AutoProcessor.video_processor`` — returns ``{'video_grid_thw': ...}``.

    ``min_pixels``/``max_pixels`` attributes are None, matching the loaded 5.x processor
    (the attrs exist but are unset — the budget lives in ``size``), so
    ``require_transformers_5``'s 4.x-surface probe passes against this fake.
    """

    min_pixels = None
    max_pixels = None

    def __init__(self, c: QwenVisionConstants = QWEN3_VL, corrupt_by: int = 0):
        self.c = c
        self.corrupt_by = corrupt_by  # add to grid_w to simulate a silent rescale (assert fires)

    def __call__(self, **kwargs):
        # Strict 5.x kwarg validation — the failure mode of passing 4.x kwargs.
        for key in kwargs:
            if key not in _QWEN_VIDEO_KWARGS_5X:
                raise TypeError(
                    "Qwen3VLVideoProcessorInitKwargs.__init__() got an unexpected keyword "
                    f"argument '{key}'"
                )
        arr = kwargs["videos"][0]             # [T, H, W, C] channels-last (encoder's layout)
        size = kwargs["size"]                 # {"shortest_edge": int, "longest_edge": int} AREAS
        t, h, w = int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2])
        # Internal temporal pad exactly like the real processor (`-T % temporal_patch_size`);
        # the encoder pre-pads, so t == t_bar on the encoder path.
        t_bar = t + (-t) % self.c.temporal_patch_size
        grid_t = t_bar // self.c.temporal_patch_size
        if kwargs["do_resize"]:
            import math

            factor = self.c.factor
            h_bar = round(h / factor) * factor
            w_bar = round(w / factor) * factor
            if self.c.video_clamp == "2d":
                # Qwen2/2.5-VL lineage (factor 28): per-FRAME 2-D clamp. Mirrors
                # models.qwen_tokens.video_tokens video_clamp="2d". assert_token_count is the
                # runtime tripwire against the real processor.
                per_frame_max = size["longest_edge"] // grid_t
                per_frame_min = size["shortest_edge"] // grid_t
                if h_bar * w_bar > per_frame_max:
                    beta = math.sqrt((h * w) / per_frame_max)
                    h_bar = max(factor, math.floor(h / beta / factor) * factor)
                    w_bar = max(factor, math.floor(w / beta / factor) * factor)
                elif h_bar * w_bar < per_frame_min:
                    beta = math.sqrt(per_frame_min / (h * w))
                    h_bar = math.ceil(h * beta / factor) * factor
                    w_bar = math.ceil(w * beta / factor) * factor
            else:
                # The real 3-D video clamp (video_processing_qwen3_vl.py): trigger uses t_bar
                # (padded FRAMES, not groups); beta uses the INPUT frame count t (== t_bar after
                # the encoder's pre-pad). NOT a per-frame 2-D clamp with divided bounds.
                if t_bar * h_bar * w_bar > size["longest_edge"]:
                    beta = math.sqrt((t * h * w) / size["longest_edge"])
                    h_bar = max(factor, math.floor(h / beta / factor) * factor)
                    w_bar = max(factor, math.floor(w / beta / factor) * factor)
                elif t_bar * h_bar * w_bar < size["shortest_edge"]:
                    beta = math.sqrt(size["shortest_edge"] / (t * h * w))
                    h_bar = math.ceil(h * beta / factor) * factor
                    w_bar = math.ceil(w * beta / factor) * factor
            h, w = h_bar, w_bar
        grid_h = h // self.c.patch_size
        grid_w = w // self.c.patch_size + self.corrupt_by
        return {"video_grid_thw": [_Grid([grid_t, grid_h, grid_w])]}


class FakeQwenProcessor:
    """Double for ``AutoProcessor`` exposing just what ViTEncoder touches (``video_processor``)."""

    def __init__(self, c: QwenVisionConstants = QWEN3_VL, corrupt_by: int = 0):
        self.video_processor = FakeQwenVideoProcessor(c, corrupt_by=corrupt_by)
