"""Hardware-free tests for the HF Qwen encode stage (ViTEncoder).

No torch / transformers / weights: a fake video processor (longvideo_eval.testing) reproduces
the REAL Qwen video-grid math, so the encoder's exact-token-count cross-check
(assert_token_count) is genuinely exercised, not stubbed past. numpy IS required (the encoder's
real path is numpy-native, like the decoder); tests importorskip it.
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from longvideo_eval.frontend.encode.encoder import (  # noqa: E402
    QWEN_TOKENS_KIND,
    ViTEncoder,
    qwen_family,
    vision_constants,
)
from longvideo_eval.interfaces import Budget, CostRecord, FramePool, Selection  # noqa: E402
from longvideo_eval.models.qwen_tokens import QWEN3_VL, temporal_groups, video_tokens  # noqa: E402
from longvideo_eval.testing import FakeQwenProcessor  # noqa: E402


def _pool(n, h, w, fps=2.0):
    frames = np.zeros((n, h, w, 3), dtype=np.uint8)
    timestamps = [i / fps for i in range(n)]
    return FramePool(video_id="v", frames=frames, timestamps=timestamps, fps=fps,
                     frame_hw=(h, w), cost=CostRecord())


def _encoder(model_id="qwen3-vl-4b", corrupt_by=0):
    enc = ViTEncoder(model_id=model_id)
    enc._processor = FakeQwenProcessor(vision_constants(model_id), corrupt_by=corrupt_by)  # inject
    return enc


# --------------------------------------------------------------------------- #
# family helpers
# --------------------------------------------------------------------------- #
def test_family_and_constants():
    assert qwen_family("qwen3-vl-4b") == "qwen3_vl"
    assert qwen_family("qwen3.5-4b") == "qwen3_5"
    assert vision_constants("qwen3-vl-4b") is QWEN3_VL
    assert vision_constants("qwen3.5-4b").max_pixels == 1605632  # Qwen3.5 override recipe
    assert vision_constants("qwen3.5-4b").factor == 32           # shares Qwen3-VL geometry
    # The always-thinking checkpoint is plain Qwen3-VL vision-wise (same processor budget).
    assert qwen_family("qwen3-vl-4b-thinking") == "qwen3_vl"
    assert vision_constants("qwen3-vl-4b-thinking") is QWEN3_VL


# --------------------------------------------------------------------------- #
# resized (r<1) path — the dense low-res case, do_resize=False, aligned frames
# --------------------------------------------------------------------------- #
def test_resized_path_token_count_and_schema():
    pool = _pool(4, 64, 128)                          # 64x128 both factor-32 aligned
    sel = Selection(indices=[0, 1, 2, 3], per_frame_resolution=[0.5] * 4, signal="uniform")
    vt = _encoder().encode(pool, sel, Budget(frame_count=4, resolution=0.5))

    # (64/32)*(128/32) = 8 tokens/frame; temporal_groups(4)=2 -> 16 visual tokens.
    assert vt.num_tokens == temporal_groups(4, QWEN3_VL) * (2 * 4) == 16
    assert vt.realized_resolution == [0.5, 0.5, 0.5, 0.5]     # echoes the r allocation
    assert vt.cost.frames_encoded == 4                        # real selected frames (not padded)
    assert vt.cost.vit_flops == 0.0                           # tower runs at the backend seam

    pkg = vt.tokens
    assert pkg["kind"] == QWEN_TOKENS_KIND
    assert pkg["model_id"] == "qwen3-vl-4b"
    assert pkg["do_resize"] is False
    assert pkg["num_visual_tokens"] == 16
    assert pkg["grid_thw"] == (2, 4, 8)
    assert pkg["frames"].shape == (4, 64, 128, 3)            # no padding needed (even count)
    # 5.x `size` bracket belt: shortest <= area <= area*grid_t <= longest under either reading.
    area, grid_t = 64 * 128, 2
    assert pkg["size"]["shortest_edge"] == min(area, QWEN3_VL.min_pixels)
    assert pkg["size"]["longest_edge"] == max(area, QWEN3_VL.max_pixels) * grid_t
    assert pkg["size"]["shortest_edge"] <= area <= pkg["size"]["longest_edge"]


def test_mrope_metadata_invariant():
    # frames_indices[k] / fps == real timestamp[k].
    pool = _pool(4, 64, 128, fps=2.0)
    sel = Selection(indices=[0, 1, 2, 3], per_frame_resolution=[0.5] * 4)
    md = _encoder().encode(pool, sel, Budget()).tokens["video_metadata"]
    fps = md["fps"]
    recovered = [i / fps for i in md["frames_indices"]]
    assert recovered == pytest.approx(pool.timestamps)
    assert md["frames_indices"] == [0, 1, 2, 3]


def test_odd_frame_count_temporal_pads():
    pool = _pool(3, 64, 128)
    sel = Selection(indices=[0, 1, 2], per_frame_resolution=[0.5] * 3)
    vt = _encoder().encode(pool, sel, Budget())
    # temporal_groups(3)=2 -> still 16 tokens; frames padded to a temporal_patch_size multiple.
    assert vt.num_tokens == 16
    assert vt.tokens["frames"].shape[0] == 4                  # 3 -> padded to 4
    assert len(vt.tokens["video_metadata"]["frames_indices"]) == 4
    assert vt.cost.frames_encoded == 3                        # padding not counted as encoded
    # padded frame reuses the last real index (a copy of the last frame).
    assert vt.tokens["video_metadata"]["frames_indices"][-1] == \
        vt.tokens["video_metadata"]["frames_indices"][-2]


def test_unaligned_resized_frame_raises():
    pool = _pool(2, 70, 128)                                  # 70 not a multiple of 32
    sel = Selection(indices=[0, 1], per_frame_resolution=[0.5, 0.5])
    with pytest.raises(ValueError, match="not a multiple of factor"):
        _encoder().encode(pool, sel, Budget())


# --------------------------------------------------------------------------- #
# native path — do_resize=True, processor smart_resizes to the family budget
# --------------------------------------------------------------------------- #
def test_native_path_matches_video_tokens():
    pool = _pool(2, 100, 100)                                 # unaligned native frame
    sel = Selection(indices=[0, 1], per_frame_resolution=None, signal="uniform")
    vt = _encoder().encode(pool, sel, Budget(frame_count=2, resolution=None))
    assert vt.tokens["do_resize"] is True
    # 64 under the REAL 3-D total-min clamp (beta = sqrt(min*grid_t/(2*100*100)) -> 256x256,
    # 8x8/frame x 1 group). A per-frame 2-D model would claim 144 — see qwen_tokens.py
    # "3-D video clamp".
    assert vt.num_tokens == video_tokens(2, 100, 100, QWEN3_VL) == 64
    assert vt.realized_resolution is None                    # native => no r echoed
    # 5.x mapping: per-frame family budget scaled by temporal_groups (the 3-D video clamp).
    assert vt.tokens["size"] == {
        "shortest_edge": QWEN3_VL.min_pixels * 1,            # temporal_groups(2) == 1
        "longest_edge": QWEN3_VL.max_pixels * 1,
    }


def test_native_path_exact_when_max_clamp_engages():
    # 720p native: rounded 704*1280 * t_bar 4 = 3.6M > longest 786432*2 = 1.57M — the 3-D
    # DOWN-clamp engages (its trigger uses PADDED FRAMES, so the effective per-frame budget is
    # max_pixels/2 = 393216, NOT the per-frame recipe). Fake (real 3-D math) and video_tokens
    # (the mirror) must agree EXACTLY — 364/frame x 2 groups = 728, not the 2-D model's 1440.
    pool = _pool(4, 720, 1280)
    sel = Selection(indices=[0, 1, 2, 3], per_frame_resolution=None)
    vt = _encoder().encode(pool, sel, Budget(frame_count=4, resolution=None))
    assert vt.num_tokens == video_tokens(4, 720, 1280, QWEN3_VL) == 728


def test_native_path_exact_when_min_clamp_engages():
    # Tiny native frames: 4*96*96 = 36864 < shortest 131072*2 — the 3-D total UP-clamp engages;
    # beta = sqrt(262144/36864) -> 256x256 = 64/frame x 2 groups = 128 (2-D model said 288).
    pool = _pool(4, 96, 96)
    sel = Selection(indices=[0, 1, 2, 3], per_frame_resolution=None)
    vt = _encoder().encode(pool, sel, Budget(frame_count=4, resolution=None))
    assert vt.num_tokens == video_tokens(4, 96, 96, QWEN3_VL) == 128


def test_native_path_repro_8960_vs_5120():
    # 128F @ 240x336 native. A 2-D mirror computes 8960 (per-frame min upscale) while the
    # processor reports 5120 (the 3-D total min does NOT trigger), which would trip
    # assert_token_count. With the 3-D mirror AND the 3-D fake, encoder expected == processor
    # grid == 5120 and encode succeeds.
    pool = _pool(128, 240, 336)
    sel = Selection(indices=list(range(128)), per_frame_resolution=None)
    vt = _encoder().encode(pool, sel, Budget(frame_count=128, resolution=None))
    assert vt.num_tokens == video_tokens(128, 240, 336, QWEN3_VL) == 5120


# --------------------------------------------------------------------------- #
# edge + fail-loud guard
# --------------------------------------------------------------------------- #
def test_zero_frames_yields_zero_tokens_no_processor():
    enc = ViTEncoder(model_id="qwen3-vl-4b")                  # no processor injected on purpose
    sel = Selection(indices=[], per_frame_resolution=[])
    vt = enc.encode(_pool(4, 64, 128), sel, Budget(frame_count=0))
    assert vt.num_tokens == 0
    assert vt.tokens["frames"] is None
    assert vt.cost.frames_encoded == 0


def test_token_count_mismatch_fires_loudly():
    # A silent processor rescale (grid off by one patch) must raise via assert_token_count.
    pool = _pool(4, 64, 128)
    sel = Selection(indices=[0, 1, 2, 3], per_frame_resolution=[0.5] * 4)
    with pytest.raises(ValueError, match="visual-token count mismatch"):
        _encoder(corrupt_by=1).encode(pool, sel, Budget())


# --------------------------------------------------------------------------- #
# transformers 5.x surface
# --------------------------------------------------------------------------- #
def test_fake_processor_rejects_4x_pixel_kwargs():
    # The fake mirrors 5.x validate_typed_dict: the removed 4.x kwargs raise — the CI tripwire
    # against drifting back to the 4.x surface.
    from longvideo_eval.testing import FakeQwenVideoProcessor

    vp = FakeQwenVideoProcessor(QWEN3_VL)
    frames = np.zeros((2, 64, 128, 3), dtype=np.uint8)
    with pytest.raises(TypeError, match="unexpected keyword argument 'min_pixels'"):
        vp(videos=[frames], video_metadata=[{}], do_resize=False, do_sample_frames=False,
           min_pixels=4096, max_pixels=786432, return_tensors="pt")


def _fake_transformers(monkeypatch, version):
    import sys
    import types

    monkeypatch.setitem(sys.modules, "transformers",
                        types.SimpleNamespace(__version__=version))


def test_require_transformers_5_rejects_4x(monkeypatch):
    from longvideo_eval.frontend.encode.encoder import require_transformers_5

    _fake_transformers(monkeypatch, "4.57.0")
    with pytest.raises(RuntimeError, match="requires transformers >= 5"):
        require_transformers_5()


def test_require_transformers_5_rejects_configured_min_pixels_surface(monkeypatch):
    # A 5.x install whose loaded video processor still carries a CONFIGURED min_pixels is a
    # 4.x-style surface — refuse rather than guess which budget applies.
    from types import SimpleNamespace

    from longvideo_eval.frontend.encode.encoder import require_transformers_5

    _fake_transformers(monkeypatch, "5.12.1")
    proc_4x = SimpleNamespace(video_processor=SimpleNamespace(min_pixels=131072))
    with pytest.raises(RuntimeError, match="4.x-style"):
        require_transformers_5(proc_4x)


def test_require_transformers_5_passes_on_pinned_surface(monkeypatch):
    # The 5.x shape: min_pixels attr present but None (budget lives in `size`).
    from longvideo_eval.frontend.encode.encoder import require_transformers_5

    _fake_transformers(monkeypatch, "5.12.1")
    require_transformers_5(FakeQwenProcessor(QWEN3_VL))  # must not raise
