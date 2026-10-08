"""Real video decode into a FramePool via decord.

Decode is a first-class cost on long video: this stage turns a raw video path into the SAME
frame pool every downstream selector sees, under ``Budget.decode_budget`` (N_dec), and meters
the wall time + frame count so no method can secretly out-decode another.

Sampling convention: ``np.linspace(0, total-1, n, dtype=int)`` — uniform float spacing, then
TRUNCATION toward zero. We take ``n = min(budget, total)`` so indices are always unique.

numpy is deliberately NOT a runtime dependency of this module: the index math is pure Python
(``int((total-1)*i/(n-1))``), which reproduces ``np.linspace(0, total-1, n, dtype=int)``
exactly, so the decoder (and its pure-function tests) run in a numpy-free environment such as
CI. numpy only enters transitively via decord's ``asnumpy()`` on the real-decode path, where
decord is by construction installed.
"""
from __future__ import annotations

import math

import time
from pathlib import Path
from typing import Optional

from ..._registry import register
from ...interfaces import Budget, CostRecord, Decoder, FramePool
from ...models.qwen_tokens import QWEN3_VL


def uniform_decode_indices(total_frames: int, decode_budget: Optional[int]) -> list[int]:
    """Uniformly spaced frame indices across ``[0, total_frames - 1]``.

    Kept a PURE function because the sampling rule *is* the fairness contract (every method
    decodes the same frames for a given budget); isolating it makes that contract unit-testable
    with no decord/video dependency.

    Semantics:
      * ``decode_budget is None`` -> every frame, ``[0 .. total_frames - 1]``.
      * ``decode_budget == 0``    -> ``[]`` (decode nothing).
      * ``decode_budget < 0``     -> ``ValueError`` (a negative budget is a caller bug).
      * otherwise                 -> ``n = min(decode_budget, total_frames)`` indices at
        ``int((total-1)*i/(n-1))`` for ``i in 0..n-1`` — identical to
        ``np.linspace(0, total-1, n, dtype=int)`` (truncation; see module docstring).

    The result is ascending and deduplicated — a no-op while ``n <= total`` (spacing >= 1 keeps
    truncated indices distinct), kept only as a defensive safety net.
    """
    if total_frames < 0:
        raise ValueError(f"total_frames must be >= 0, got {total_frames}")
    if decode_budget is None:
        return list(range(total_frames))
    if decode_budget < 0:
        raise ValueError(f"decode_budget must be >= 0 or None, got {decode_budget}")
    n = min(decode_budget, total_frames)
    if n <= 0:
        return []
    if n == 1:
        # linspace(0, total-1, 1) == [0.0]; avoids a divide-by-(n-1) below.
        return [0]
    idx = [int((total_frames - 1) * i / (n - 1)) for i in range(n)]  # truncation == linspace(dtype=int)
    return list(dict.fromkeys(idx))  # order-preserving dedup; no-op for n <= total_frames


def resize_target(
    height: int,
    width: int,
    resolution: float,
    factor: int = QWEN3_VL.factor,
) -> tuple[int, int]:
    """Target ``(H, W)`` to decode-resize to for scale factor ``resolution`` (r), each side a
    multiple of ``factor``.

    Kept a PURE function (no decord/numpy) because it *is* the resolution-axis contract: r is a
    per-side scale relative to native (``r in (0, 1]``: r=1.0 native, r=0.25 a quarter PER
    SIDE), so tokens/frame scale as ~r^2 and the triplet 16F@1.0 / 64F@0.5 / 256F@0.25 sits at
    approximately the same visual-token budget.

    Each scaled side is rounded to the NEAREST ``factor`` multiple, byte-for-byte the round style
    of ``qwen_tokens.smart_resize`` (``round(side / factor) * factor``), so the downstream Qwen
    encode sees already-aligned frames and its own ``smart_resize`` is a pass-through (no silent
    re-clamp). ``factor`` defaults to the Qwen3-VL vision patch factor (patch 16 * merge 2 = 32).
    Because the two sides round independently, the realized token count is only approximately
    ``r^2`` of the native one; the harness meters the realized count per sample.

    Degenerate small videos are clamped to a ``>= factor`` floor on each side, so a tiny clip never
    collapses below one vision patch. Fails loudly on non-positive dims, ``factor``, or an r outside
    ``(0, 1]`` (r > 1 would upsample beyond native, which the axis does not define).
    """
    if height <= 0 or width <= 0:
        raise ValueError(f"height and width must be positive, got {height}x{width}")
    if factor <= 0:
        raise ValueError(f"factor must be positive, got {factor}")
    if not (0.0 < resolution <= 1.0):
        raise ValueError(f"resolution (scale r) must be in (0, 1], got {resolution}")
    h_bar = max(factor, round(height * resolution / factor) * factor)
    w_bar = max(factor, round(width * resolution / factor) * factor)
    return h_bar, w_bar


def fit_target(
    height: int,
    width: int,
    max_tokens: int,
    factor: int = QWEN3_VL.factor,
    max_aspect_error: float = 0.08,
) -> tuple[int, int]:
    """The largest ``(H, W)`` whose token count ``(H/factor) * (W/factor)`` fits in
    ``max_tokens``, keeping the video's aspect ratio and never exceeding its native size.

    Used when a stream is sized from the video rather than from a fixed scale: the native
    size decides the shape, the remaining budget decides how much of it is kept. Each side is
    a multiple of ``factor``. Among the grids that fit, the one with the most tokens wins as
    long as its aspect ratio is within ``max_aspect_error`` (log ratio) of the native one;
    ties go to the closer aspect ratio. Pure, like ``resize_target``.
    """
    if height <= 0 or width <= 0:
        raise ValueError(f"height and width must be positive, got {height}x{width}")
    if factor <= 0:
        raise ValueError(f"factor must be positive, got {factor}")
    if max_tokens < 1:
        raise ValueError(f"max_tokens must be at least 1, got {max_tokens}")
    native_h, native_w = resize_target(height, width, 1.0, factor=factor)
    top_h, top_w = native_h // factor, native_w // factor
    if top_h * top_w <= max_tokens:
        return native_h, native_w
    aspect = width / height
    scale = (max_tokens / (top_h * top_w)) ** 0.5
    best = None
    for g_h in range(max(1, int(top_h * scale) - 2), min(top_h, int(top_h * scale) + 3) + 1):
        for g_w in range(max(1, int(top_w * scale) - 2), min(top_w, int(top_w * scale) + 3) + 1):
            if g_h * g_w > max_tokens:
                continue
            err = abs(math.log((g_w / g_h) / aspect))
            # in tolerance first, then most tokens, then closest aspect ratio
            key = (err <= max_aspect_error, g_h * g_w if err <= max_aspect_error else -err, -err)
            if best is None or key > best[0]:
                best = (key, g_h, g_w)
    if best is None:                      # budget below one row or column: a single patch
        return factor, factor
    return best[1] * factor, best[2] * factor


@register("decode", "decord")
class DecordDecoder(Decoder):
    """Decode with decord under ``Budget.decode_budget`` (N_dec).

    Only this stage's increment (``decode_seconds`` + ``frames_decoded``) is written to the
    returned ``FramePool.cost``; the orchestrator folds per-stage costs together
    (interfaces.CostRecord.merge). No caching / JPEG dumping here — predecode caching is a
    separate, later concern.
    """

    def __init__(self, num_threads: int = 0, factor: int = QWEN3_VL.factor) -> None:
        # 0 == decord's own default thread count; exposed so callers can pin decode threads.
        self.num_threads = num_threads
        # Vision patch factor (patch_size * merge_size) of the TARGET model — 32 for the qwen3
        # family, 28 for qwen2.5-vl. The decoder snaps its resize to this so the processor's own
        # smart_resize is a pass-through; a wrong factor makes the processor silently re-resize
        # and trips assert_token_count downstream. build_pipeline passes the model's value.
        if factor <= 0:
            raise ValueError(f"factor must be positive, got {factor}")
        self.factor = factor

    def decode(self, video_path: str, budget: Budget) -> FramePool:
        # Fail loudly on a bad path BEFORE importing/opening decord, so a missing video is a
        # clean FileNotFoundError rather than an opaque decord internal error.
        if not Path(video_path).is_file():
            raise FileNotFoundError(f"video not found: {video_path}")

        try:
            import decord
        except ImportError as exc:  # fail loudly; no fallback reader by design
            raise ImportError(
                "DecordDecoder requires the 'decord' package (pip install decord); "
                "no fallback video reader is provided by design"
            ) from exc

        # Meter wall time strictly around VideoReader construction + get_batch (the decode work).
        t0 = time.perf_counter()
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0), num_threads=self.num_threads)
        total = len(vr)
        native_fps = float(vr.get_avg_fps())
        if native_fps <= 0.0:
            # A non-positive average fps means a corrupt/unreadable stream — fail loudly.
            raise ValueError(f"decord reported non-positive fps ({native_fps}) for {video_path}")
        indices = uniform_decode_indices(total, budget.decode_budget)
        if not indices:
            # decode_budget == 0 (or empty clip): return an empty [0, ...] array, no get_batch.
            import numpy as np

            frames = np.empty((0, 0, 0, 0), dtype=np.uint8)
        elif budget.resolution is None or budget.presentation == "lohi":
            # Native decode. The dual-stream ("lohi") presentation needs TWO sizes of the same
            # frames, both defined as a scale of the native size, so the pool stays native and
            # the encoder renders each stream from it (one decode, two scales).
            frames = vr.get_batch(indices).asnumpy()  # [N, H, W, C] uint8, native size
        else:
            # Decode-at-target-size: decord's native width/height resize. It still runs the full
            # IDCT and only fuses the resize into decode, so decode+resize land in ONE metered
            # decode_seconds. We probe native (H, W)
            # from frame 0 (one native decode) to scale by r, then reopen at the target size.
            native_h, native_w = (int(d) for d in vr[0].shape[:2])
            target_h, target_w = resize_target(
                native_h, native_w, budget.resolution, factor=self.factor,
            )
            vr_resized = decord.VideoReader(
                video_path, ctx=decord.cpu(0),
                width=target_w, height=target_h, num_threads=self.num_threads,
            )
            frames = vr_resized.get_batch(indices).asnumpy()  # [N, target_h, target_w, C] uint8
        decode_seconds = time.perf_counter() - t0

        n = len(indices)
        # Realized (H, W) of the pooled frames after any decode-time resize; None when empty.
        frame_hw = (int(frames.shape[1]), int(frames.shape[2])) if n else None
        timestamps = [i / native_fps for i in indices]  # seconds of each sampled frame

        # METADATA CONVENTION: FramePool carries the absolute timestamps, the EFFECTIVE sampled
        # fps (below) and the source's NATIVE fps (`native_fps`). The encoder rebuilds Qwen
        # `frames_indices` as round(t * native_fps), which recovers the original frame indices
        # exactly. (Rounding with the effective fps instead would quantise timestamps to a
        # ~duration/N grid: 16 frames over an hour can be up to ~112 s off in the rendered
        # `<t seconds>` tokens, and irregular samples collapse to an even grid.)
        # FramePool.fps is the EFFECTIVE sampled fps: sampled-frame count / full clip duration.
        # Uniform sampling spans the whole clip, so this reports the pool's true temporal density
        # (frames kept per second of video), NOT the native fps — downstream MRoPE/timestamp
        # logic should treat `fps` as the sampling rate and `timestamps` as the absolute times.
        duration = total / native_fps
        effective_fps = (n / duration) if duration > 0 else 0.0

        return FramePool(
            video_id=Path(video_path).stem,
            frames=frames,
            timestamps=timestamps,
            fps=effective_fps,
            frame_hw=frame_hw,
            native_fps=native_fps,
            cost=CostRecord(decode_seconds=decode_seconds, frames_decoded=n),
        )
