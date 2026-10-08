"""Vision encode stage for the HF Qwen path.

STAGE SPLIT (why this stage does NOT run the vision tower). Token-pruning methods patch the
Qwen vision tower / language model at *generate* time, so the tower has to run behind the
backend seam (``backend/hf/qwen.py``) inside ``model.generate``. This encode stage therefore
runs only the HF *preprocessing*: it patchifies the already-decoded, already-resized frames
into a video grid, computes the EXACT visual-token count (``qwen_tokens``) and asserts it
against the processor's reported grid — the fail-loud guard against a silent rescale (at
r=0.25 a 192x320=61440 px frame is BELOW Qwen's video ``min_pixels`` 131072, so a resize path
would silently UPSCALE and void the cost match).

Because ``Encoder.encode(pool, selection, budget)`` has no query, it cannot tokenize the
text+video sequence — ``input_ids`` depend on the prompt. So it packages the text-INDEPENDENT
visual evidence + timestamp metadata + pixel budget into ``VisualTokens.tokens`` (opaque dict,
schema below); ``backend/hf/qwen.py`` assembles the prompt, re-runs the processor with the
text, and generates.

``VisualTokens.tokens`` schema (``kind == QWEN_TOKENS_KIND``)::

    {
      "kind":              QWEN_TOKENS_KIND,
      "model_id":          str,              # config.BASE_MODELS key (family + repo)
      "frames":            np.ndarray|None,  # [T,H,W,C] uint8, selected + decode-resized +
                                             #   temporal-padded to a temporal_patch_size multiple
                                             #   (None on the zero-frame edge)
      "video_metadata":    dict,             # {fps, frames_indices, total_num_frames, duration}
      "size":              dict,             # transformers-5.x video area budget, passed per
                                             #   call: {"shortest_edge": int, "longest_edge": int}
                                             #   — AREA bounds (pixel counts) despite the names
      "do_resize":         bool,             # False on the resized (r<1) path, True on native
      "num_visual_tokens": int,              # asserted == processor grid (merged)
      "grid_thw":          tuple|None,       # (t,h,w) from the encode-time assert run
    }

TRANSFORMERS 5.x. 5.x strictly validates video-processor kwargs and DROPPED
``min_pixels``/``max_pixels`` — the area budget now lives in ``size={"shortest_edge",
"longest_edge"}`` (AREA semantics preserved). Mapping from our per-frame budget to 5.x
``size``: the Qwen3-VL VIDEO budget clamps the 3-D total ``t_bar*h_bar*w_bar`` (see
``models/qwen_tokens.py``), so the per-frame allowance ``b`` maps to
``longest_edge = b * temporal_groups``. No 4.x fallback by design;
``require_transformers_5`` gates loudly, ``assert_token_count`` is the runtime tripwire.

Timestamps. ``video_metadata`` satisfies ``frames_indices[k] / fps == pool.timestamps[k]``
(real elapsed seconds). We build ``frames_indices`` ONCE from the decoder's own sampled
timestamps and never re-sample or overwrite them, so the frame indices the processor sees are
the decoded frames' indices and the rendered ``<t seconds>`` text is correct.

torch / transformers / numpy are imported LAZILY inside the real path: the package must import
on a GPU-free machine.
"""
from __future__ import annotations

from ..._registry import register
from ...interfaces import Budget, CostRecord, Encoder, FramePool, Selection, VisualTokens
from ...models.qwen_tokens import (
    QWEN2_5_VL,
    QWEN3_VL,
    QwenVisionConstants,
    assert_token_count,
    merge_groups,
    merged_tokens,
    temporal_groups,
    video_tokens,
)

# Package tag on VisualTokens.tokens so backend/hf/qwen.py can validate the seam contract.
QWEN_TOKENS_KIND = "qwen-hf-video"

# The environment the harness is developed and verified against.
PINNED_ENV = "transformers >= 5 / torch >= 2.10 / decord 0.6.0"


def require_transformers_5(processor=None) -> None:
    """Loud, early gate: the harness targets transformers >= 5 only (no fallback).

    5.x moved the video area budget from ``min_pixels``/``max_pixels`` kwargs to
    ``size={"shortest_edge", "longest_edge"}`` and strictly validates call kwargs; supporting
    both surfaces silently would mean two divergent cost paths, so anything older fails here —
    at first processor use, before any GPU work. When a ``processor`` is given, additionally
    reject a video processor that still carries a CONFIGURED ``min_pixels`` attribute (a
    4.x-style pixel surface).
    """
    import transformers  # lazy; this module must import GPU-free

    major = int(transformers.__version__.split(".", 1)[0])
    if major < 5:
        raise RuntimeError(
            f"longvideo-eval requires transformers >= 5 (verified env: {PINNED_ENV}); "
            f"found {transformers.__version__}. The 4.x min_pixels/max_pixels video-processor "
            "surface is unsupported by design — no dual-path fallback."
        )
    vp = getattr(processor, "video_processor", None) if processor is not None else None
    if vp is not None and getattr(vp, "min_pixels", None) is not None:
        raise RuntimeError(
            "loaded video processor exposes a configured min_pixels attribute — a 4.x-style "
            f"pixel surface the harness no longer drives (verified env: {PINNED_ENV}); "
            "refusing to guess which budget applies."
        )

# Qwen3.5-4B shares Qwen3-VL vision geometry (patch=16, merge=2, temporal=2 => factor 32) but
# its processor's default video pixel budget is far smaller, blowing the token count up if left
# as-is; the recipe that keeps token counts comparable is min=200704 / max=1605632. Only the
# AREA budget differs from Qwen3-VL — the factor/temporal geometry is identical, so
# smart_resize/token math is shared.
_QWEN3_5 = QwenVisionConstants(
    patch_size=16,
    merge_size=2,
    temporal_patch_size=2,
    min_pixels=200704,   # 256*28*28 (an area literal, not 28-geometry)
    max_pixels=1605632,
)


def qwen_family(model_id: str) -> str:
    """Family key from a config.BASE_MODELS id: ``"qwen2_5_vl"`` for Qwen2.5-VL,
    ``"qwen3_5"`` for Qwen3.5, else ``"qwen3_vl"``.

    Kept a pure string test (no import of transformers/AutoConfig) so both the encoder and the
    backend derive the family the same way without loading weights. The backend additionally
    confirms against ``AutoConfig.model_type`` when it loads the model.
    """
    m = model_id.lower()
    if "2.5" in m or "2_5" in m:
        return "qwen2_5_vl"
    return "qwen3_5" if ("3.5" in m or "3_5" in m) else "qwen3_vl"


def vision_constants(model_id: str) -> QwenVisionConstants:
    """Per-model vision constants (geometry + processor pixel budget) for token math."""
    fam = qwen_family(model_id)
    if fam == "qwen2_5_vl":
        return QWEN2_5_VL       # factor 28, per-frame 2-D video clamp
    return _QWEN3_5 if fam == "qwen3_5" else QWEN3_VL


def processor_load_kwargs(model_id: str) -> dict:
    """``from_pretrained`` kwargs shared by the encode stage and the backend for one model.

    Empty for every Qwen family (plain in-library classes). Kept here (not in the backend) so
    the encoder's processor and the backend's processor are guaranteed to be loaded the same
    way — a drift between them would move the token count.
    """
    return {}


def _resize_frames(frames_thwc, target_h: int, target_w: int):
    """Resize a [T,H,W,C] uint8 stack to an EXACT (target_h, target_w) with PIL bicubic.

    Exact, not "budgeted": the caller already decided the grid and passes ``do_resize=False``,
    so the processor must see that grid and nothing else. Bicubic matches decord's decode-time
    resize and the Qwen processors, keeping these frames comparable with the single-stream
    setups' decode-time-resized ones.
    """
    import numpy as np
    from PIL import Image

    if frames_thwc.shape[1] == target_h and frames_thwc.shape[2] == target_w:
        return frames_thwc
    out = [
        np.asarray(
            Image.fromarray(np.asarray(f, dtype=np.uint8)).resize(
                (target_w, target_h), Image.BICUBIC
            ),
            dtype=np.uint8,
        )
        for f in frames_thwc
    ]
    return np.stack(out, axis=0)


def to_processor_video(frames_thwc):
    """The frame array layout handed to the HF video processor (channels-LAST [T,H,W,C]).

    Single source of truth shared with the backend so the layout is fixed in one place. HF
    image/video processors infer channel dimension and accept channels-last uint8; we keep the
    decord-native [T,H,W,C] and let the processor rescale/normalize (do_resize handled by the
    caller). If a transformers build needs [T,C,H,W], flip it HERE.
    """
    return frames_thwc


def metadata_fps(pool: FramePool) -> float:
    """The rate used to turn frame timestamps back into frame indices for the processor.

    Must be the source's NATIVE fps: indices are rebuilt as round(t * fps) and the processor
    renders `<t seconds>` as index / fps, so only a rate at which every sampled time is (close
    to) an integer multiple of 1/fps round-trips. The effective sampling rate does not: for 16
    frames over an hour it quantises timestamps to a ~225 s grid. Pools without `native_fps`
    (fakes) fall back to the sampling rate.
    """
    if pool.native_fps and pool.native_fps > 0:
        return float(pool.native_fps)
    return float(pool.fps) if pool.fps and pool.fps > 0 else 2.0


def _mrope_metadata(timestamps, fps: float, pad: int) -> dict:
    """Timestamp ``video_metadata`` for pre-decoded frames.

    ``frames_indices[k] / fps == timestamps[k]`` so the processor recovers the real elapsed
    seconds of each frame. Indices are derived ONCE from the decoder's sampled timestamps and
    made strictly increasing (a defensive guard for coarse fps; uniform sampling keeps spacing
    ~1 so this is a no-op in practice). Temporal-pad frames repeat the last real frame, so they
    reuse the last index. ``fps`` must be a rate at which the timestamps are (close to) integer
    multiples of 1/fps — the source's native fps (see ``metadata_fps``); rounding with a coarse
    rate breaks the ``index/fps == seconds`` invariant the processor relies on.
    """
    idx: list[int] = []
    prev = -1
    for t in timestamps:
        i = int(round(t * fps))
        if i <= prev:
            i = prev + 1  # keep strictly increasing; never re-sample/reorder
        idx.append(i)
        prev = i
    for _ in range(pad):
        idx.append(idx[-1] if idx else 0)  # padding = a copy of the last real frame
    duration = float(timestamps[-1]) if timestamps else 0.0
    return {
        "fps": float(fps),
        "frames_indices": idx,
        "total_num_frames": (idx[-1] + 1) if idx else 0,
        "duration": duration,
    }


@register("encode", "vit")
class ViTEncoder(Encoder):
    """HF Qwen video preprocessing + exact-token-count assert. See module docstring for the
    stage split and the ``VisualTokens.tokens`` schema.

    ``model_id`` keys into ``config.BASE_MODELS`` (family + HF repo); build.py passes it so the
    encoder and backend agree on the processor. The processor is loaded lazily and cached.
    """

    def __init__(self, model_id: str = "qwen3-vl-4b") -> None:
        self.model_id = model_id
        self._processor = None  # lazy (transformers import deferred to the real path)

    def _load_processor(self):
        if self._processor is None:
            from transformers import AutoProcessor  # lazy; GPU-free import of this module

            from ...config import BASE_MODELS

            if self.model_id not in BASE_MODELS:
                raise KeyError(
                    f"unknown model {self.model_id!r}; valid: {sorted(BASE_MODELS)}"
                )
            # No pixel kwargs here: 5.x from_pretrained silently ignores min/max_pixels
            # ("Unused or unrecognized kwargs") — the budget is passed explicitly as `size`
            # on EVERY processor call instead, so no default can drift under us.
            self._processor = AutoProcessor.from_pretrained(
                BASE_MODELS[self.model_id].hf_repo, **processor_load_kwargs(self.model_id)
            )
            require_transformers_5(self._processor)  # loud gate
        return self._processor

    def _video_processor_kwargs(self, proc, *, do_resize: bool, size: dict) -> dict:
        """Kwargs for a DIRECT ``proc.video_processor(...)`` call (the 5.x surface)."""
        return {"do_resize": do_resize, "do_sample_frames": False, "size": size}

    def _encode_lohi(
        self, pool: FramePool, selection: Selection, budget: Budget
    ) -> VisualTokens:
        """Dual stream: ONE video block at ``r_l`` (Lo-V) + K image blocks at ``r_h`` (Hi-I).

        The pool arrives NATIVE (the decoder keeps it native for ``presentation="lohi"``),
        and both streams are sized from that native size. The video block is at
        ``Budget.resolution``. The images are at ``Budget.hi_resolution`` when it is set;
        otherwise they are sized PER VIDEO to fill what the video block leaves of the
        reference budget (``Budget.reference_frames`` native frames), so the total lands on
        that budget whatever the source resolution is. Both are handed to the processor with
        ``do_resize=False``, so the realized grids are ours.

        Token accounting::

            Lo-V  = temporal_groups(N) * (lo_h/f) * (lo_w/f)   # temporal merge APPLIES
            Hi-I  = K * (hi_h/f) * (hi_w/f)                    # image pathway: NO merge

        An image costs twice a video frame of the same size on the Qwen families, because the
        video pathway folds two frames into one temporal position and the image pathway does
        not. Both streams are asserted against the processor's realized grids, so a silent
        re-clamp on either one stops the run instead of quietly changing the budget.
        """
        import numpy as np

        from ..decode.decoder import fit_target, resize_target

        c = vision_constants(self.model_id)
        factor, tps = c.factor, c.temporal_patch_size
        indices = list(selection.indices)
        hi_indices = list(selection.hi_indices or [])
        n = len(indices)
        if n == 0:
            raise ValueError("lohi presentation needs a non-empty Lo-V selection")
        if not set(hi_indices).issubset(set(indices)):
            raise ValueError(
                "Hi-I indices must be a SUBSET of the Lo-V frames (each frame is decoded once "
                f"and reused at both scales); got hi={hi_indices[:8]}... not in Lo-V"
            )
        if budget.resolution is None:
            raise ValueError("lohi presentation needs Budget.resolution (the Lo-V scale r_l)")

        frames = np.asarray(pool.frames)
        native_h, native_w = int(frames.shape[1]), int(frames.shape[2])
        lo_h, lo_w = resize_target(native_h, native_w, budget.resolution, factor=factor)
        lo_tokens = merge_groups(n, c) * (lo_h // factor) * (lo_w // factor)
        if budget.hi_resolution is not None:
            hi_h, hi_w = resize_target(
                native_h, native_w, float(budget.hi_resolution), factor=factor
            )
        elif budget.reference_frames is not None and hi_indices:
            full_h, full_w = resize_target(native_h, native_w, 1.0, factor=factor)
            reference = (
                merge_groups(int(budget.reference_frames), c)
                * (full_h // factor) * (full_w // factor)
            )
            per_image = max(1, (reference - lo_tokens) // len(hi_indices))
            hi_h, hi_w = fit_target(native_h, native_w, per_image, factor=factor)
        else:
            hi_h, hi_w = resize_target(native_h, native_w, 1.0, factor=factor)

        lo_frames = _resize_frames(frames[indices], lo_h, lo_w)
        hi_frames = _resize_frames(frames[hi_indices], hi_h, hi_w) if hi_indices else None

        # --- Lo-V: video pathway (temporal merge intact) ---
        grid_t = temporal_groups(n, c)
        expected_lo = merge_groups(n, c) * (lo_h // factor) * (lo_w // factor)
        sel = lo_frames
        pad = (-n) % tps
        if pad:
            sel = np.concatenate([sel, np.repeat(sel[-1:], pad, axis=0)], axis=0)
        timestamps = [float(pool.timestamps[i]) for i in indices]
        video_metadata = _mrope_metadata(timestamps, metadata_fps(pool), pad)
        lo_area = lo_h * lo_w
        lo_size = {
            "shortest_edge": min(lo_area, c.min_pixels),
            "longest_edge": max(lo_area, c.max_pixels) * grid_t,
        }
        proc = self._load_processor()
        vp_out = proc.video_processor(
            videos=[to_processor_video(sel)],
            video_metadata=[video_metadata],
            return_tensors="pt",
            **self._video_processor_kwargs(proc, do_resize=False, size=lo_size),
        )
        grid = vp_out["video_grid_thw"][0].tolist()
        assert_token_count(expected_lo, merged_tokens(grid, c))

        # --- Hi-I: image pathway (no temporal merge) ---
        expected_hi, hi_grids, hi_size = 0, [], None
        if hi_frames is not None:
            hi_area = hi_h * hi_w
            hi_size = {
                "shortest_edge": min(hi_area, c.min_pixels),
                "longest_edge": max(hi_area, c.max_pixels),
            }
            ip_out = proc.image_processor(
                images=[f for f in hi_frames], do_resize=False, size=hi_size, return_tensors="pt"
            )
            hi_grids = ip_out["image_grid_thw"].tolist()
            realized_hi = sum(merged_tokens(g, c) for g in hi_grids)
            expected_hi = len(hi_indices) * (hi_h // factor) * (hi_w // factor)
            assert_token_count(expected_hi, realized_hi)

        return VisualTokens(
            tokens={
                "kind": QWEN_TOKENS_KIND,
                "model_id": self.model_id,
                "frames": sel,
                "video_metadata": video_metadata,
                "size": lo_size,
                "do_resize": False,
                "num_visual_tokens": expected_lo,
                "grid_thw": tuple(grid),
                "frame_hw": (lo_h, lo_w),
                # --- the Hi-I stream (present only for presentation="lohi") ---
                "hi_frames": hi_frames,
                "hi_timestamps": [float(pool.timestamps[i]) for i in hi_indices],
                "hi_num_visual_tokens": expected_hi,
                "hi_frame_hw": (hi_h, hi_w),
                "hi_grid_thw": [tuple(g) for g in hi_grids],
                "hi_size": hi_size,
                "native_hw": (native_h, native_w),
                # realized per-side scale of the images against the native size
                "hi_scale": round(((hi_h * hi_w) / (native_h * native_w)) ** 0.5, 4),
            },
            num_tokens=expected_lo + expected_hi,
            realized_resolution=selection.per_frame_resolution,
            cost=CostRecord(
                frames_encoded=n + len(hi_indices),
                vit_patch_tokens=int(grid[0] * grid[1] * grid[2])
                + sum(int(g[0] * g[1] * g[2]) for g in hi_grids),
            ),
        )

    def encode(self, pool: FramePool, selection: Selection, budget: Budget) -> VisualTokens:
        if budget.presentation == "lohi":
            return self._encode_lohi(pool, selection, budget)
        indices = list(selection.indices)
        n = len(indices)
        c = vision_constants(self.model_id)

        if n == 0:
            # 0 frames selected (frame_count == 0) -> zero visual tokens; no processor call.
            # `size` carries the inert per-frame family budget (nothing consumes it downstream).
            return VisualTokens(
                tokens={
                    "kind": QWEN_TOKENS_KIND, "model_id": self.model_id, "frames": None,
                    "video_metadata": {},
                    "size": {"shortest_edge": c.min_pixels, "longest_edge": c.max_pixels},
                    "do_resize": True, "num_visual_tokens": 0, "grid_thw": None,
                    "frame_hw": None,
                },
                num_tokens=0,
                realized_resolution=selection.per_frame_resolution,
                cost=CostRecord(frames_encoded=0),
            )

        import numpy as np  # lazy (real path only)

        frames = pool.frames
        sel = np.asarray(frames)[np.asarray(indices, dtype=int)]  # [n,H,W,C] uint8
        height, width = int(sel.shape[1]), int(sel.shape[2])
        # CONTRACT: Selection.per_frame_resolution echoes what the DECODER already realized.
        # Budget.resolution is applied at DECODE time (frontend.decode via decord width/height),
        # so by the time frames reach here they are ALREADY at their target size. Every shipped
        # selector expands budget.resolution into per_frame_resolution when it is set, and passes
        # None only when budget.resolution is also None. So None here means "decoder did not
        # resize -> native smart-resize path", which is exactly what this branch does.
        resized = selection.per_frame_resolution is not None  # honor the resolution allocation
        # NOTE: despite the Sequence type, this encoder only supports a uniform realized frame
        # shape in one FramePool. It echoes per-frame resolution metadata, but does not actually
        # encode heterogeneous per-frame sizes; true dynamic-resolution allocation would need a
        # ragged/multi-call processor path rather than this single [T,H,W,C] array.
        factor = c.factor

        # --- expected token count + 5.x `size` area budget (branch on resolution axis) ---
        grid_t = temporal_groups(n, c)
        if resized:
            # Decoder already resized each side to r and snapped to `factor` (resize_target); the
            # processor runs do_resize=False, so its grid is exactly ours. Fail loud if the
            # decoder handed us an unaligned frame (decode/selection contract break).
            if height % factor or width % factor:
                raise ValueError(
                    f"resized frame {height}x{width} not a multiple of factor {factor}; "
                    "decoder resize_target / selection resolution are out of sync"
                )
            per_frame = (height // factor) * (width // factor)
            expected = merge_groups(n, c) * per_frame   # == grid_t * per_frame on every Qwen
            area = height * width
            # Belt against a silent upscale: bracket the decoded frames so a resize path, were
            # it ever to engage despite do_resize=False, would be a no-op under EITHER area
            # reading — shortest_edge <= area <= area*grid_t <= longest_edge.
            size = {
                "shortest_edge": min(area, c.min_pixels),
                "longest_edge": max(area, c.max_pixels) * grid_t,
            }
            do_resize = False
        else:
            # Native path: processor smart_resizes to the family budget; we mirror its math.
            # 5.x mapping, family-branched on video_clamp:
            #   "3d" (qwen3_vl): the video budget clamps the 3-D total t*h*w, so the size bounds
            #   handed over scale by grid_t. The 3-D clamp is NOT equivalent to a per-frame 2-D
            #   clamp with divided bounds — its trigger uses t_bar = PADDED FRAMES (not groups),
            #   making the effective per-frame budget max_pixels/tps (half, at tps=2), and its
            #   total-min trigger is much laxer than the per-frame min. video_tokens() mirrors it.
            #   "2d" (qwen2_5_vl): the processor applies the canonical per-frame smart_resize
            #   with the size bounds DIRECTLY — bounds go over unscaled.
            expected = video_tokens(n, height, width, c)
            scale_t = grid_t if c.video_clamp == "3d" else 1
            size = {
                "shortest_edge": c.min_pixels * scale_t,
                "longest_edge": c.max_pixels * scale_t,
            }
            do_resize = True

        # --- temporal padding (repeat last frame to a temporal_patch_size multiple) ---
        tps = c.temporal_patch_size
        pad = (-n) % tps
        if pad:
            sel = np.concatenate([sel, np.repeat(sel[-1:], pad, axis=0)], axis=0)

        timestamps = [float(pool.timestamps[i]) for i in indices]
        fps = metadata_fps(pool)
        video_metadata = _mrope_metadata(timestamps, fps, pad)

        # --- fail-loud cross-check: the processor grid must equal our computed count ---
        # do_sample_frames=False: frames already sampled by the decoder (no re-sample).
        # size = the 5.x area budget (was min_pixels/max_pixels on 4.x).
        proc = self._load_processor()
        vp_out = proc.video_processor(
            videos=[to_processor_video(sel)],
            video_metadata=[video_metadata],
            return_tensors="pt",
            **self._video_processor_kwargs(proc, do_resize=do_resize, size=size),
        )
        grid = vp_out["video_grid_thw"][0].tolist()  # [t, h, w]
        merged = merged_tokens(grid, c)
        assert_token_count(expected, merged)  # raises on a silent auto-rescale / math drift

        return VisualTokens(
            tokens={
                "kind": QWEN_TOKENS_KIND,
                "model_id": self.model_id,
                "frames": sel,
                "video_metadata": video_metadata,
                "size": size,
                "do_resize": do_resize,
                "num_visual_tokens": expected,
                "grid_thw": tuple(grid),
                # Realized-shape audit trail: the per-frame (H, W) the pool actually delivered
                # to the processor — with grid_thw this makes any expected-vs-actual clamp
                # investigation one jq away (the backend arm records both).
                "frame_hw": (int(sel.shape[1]), int(sel.shape[2])),
            },
            num_tokens=expected,
            realized_resolution=selection.per_frame_resolution,  # echo the r allocation
            # vit_flops stays 0: the tower runs behind the backend seam; front-end encode
            # wall-time is metered by the orchestrator's stage timer.
            # vit_patch_tokens: t*h*w of the REALIZED (asserted) grid — the PRE-merge patch
            # count entering the tower (ViT ingress).
            cost=CostRecord(frames_encoded=n,
                            vit_patch_tokens=int(grid[0] * grid[1] * grid[2])),
        )
