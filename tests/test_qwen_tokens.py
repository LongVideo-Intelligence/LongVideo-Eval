"""Unit tests for the pure Qwen visual-token math (no GPU, no network).

Covers: smart_resize replication (pass-through, aspect preservation, factor-multiple output,
min/max clamping), frame/video token counts incl. temporal padding, budget-solver round-trips
and infeasibility, and an OPTIONAL cross-check against transformers if it happens to be installed.
"""
from __future__ import annotations

import dataclasses
import math

import pytest

from longvideo_eval.models.qwen_tokens import (
    QWEN2_5_VL,
    QWEN3_VL,
    QwenVisionConstants,
    assert_token_count,
    frame_tokens,
    max_frames_for_budget,
    max_pixels_for_budget,
    smart_resize,
    temporal_groups,
    video_tokens,
)

FAMILIES = [QWEN3_VL, QWEN2_5_VL]


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
def test_derived_factor_and_family_constants():
    assert QWEN3_VL.factor == 32 == QWEN3_VL.patch_size * QWEN3_VL.merge_size
    assert QWEN2_5_VL.factor == 28 == QWEN2_5_VL.patch_size * QWEN2_5_VL.merge_size
    assert QWEN3_VL.min_pixels == 128 * 32 * 32 and QWEN3_VL.max_pixels == 32 * 32 * 768
    assert QWEN2_5_VL.min_pixels == 128 * 28 * 28 and QWEN2_5_VL.max_pixels == 28 * 28 * 768
    for c in FAMILIES:
        assert c.temporal_patch_size == 2
    # Video clamp lineage: Qwen3 = 3-D total clamp, Qwen2/2.5 = per-frame 2-D (see
    # qwen_tokens.py).
    assert QWEN3_VL.video_clamp == "3d" and QWEN2_5_VL.video_clamp == "2d"
    with pytest.raises(ValueError, match="video_clamp"):
        dataclasses.replace(QWEN3_VL, video_clamp="4d")


def _3d_max_engages(n: int, H: int, W: int, c: QwenVisionConstants) -> bool:
    """Whether the Qwen3-VL 3-D MAX clamp triggers for this clip (the trigger inequality,
    video_processing_qwen3_vl.py:35-63: t_bar*h_bar*w_bar > longest with t_bar = padded FRAMES
    and longest = per-frame recipe x groups)."""
    if c.video_clamp != "3d":
        return False
    g = temporal_groups(n, c)
    t_bar = g * c.temporal_patch_size
    h_bar = round(H / c.factor) * c.factor
    w_bar = round(W / c.factor) * c.factor
    return t_bar * h_bar * w_bar > c.max_pixels * g


def test_constants_validate_bounds():
    with pytest.raises(ValueError):
        QwenVisionConstants(patch_size=16, merge_size=2, temporal_patch_size=2,
                            min_pixels=1000, max_pixels=10)  # min > max
    with pytest.raises(ValueError):
        QwenVisionConstants(patch_size=0, merge_size=2, temporal_patch_size=2,
                            min_pixels=10, max_pixels=1000)  # bad geometry


# --------------------------------------------------------------------------- #
# smart_resize
# --------------------------------------------------------------------------- #
def test_smart_resize_passthrough_when_already_valid():
    # 448x448 is a multiple of both 32 and 28 and within both families' [min, max].
    for c in FAMILIES:
        h, w = smart_resize(448, 448, factor=c.factor, min_pixels=c.min_pixels,
                            max_pixels=c.max_pixels)
        assert (h, w) == (448, 448)


def test_smart_resize_outputs_are_factor_multiples():
    for c in FAMILIES:
        for (H, W) in [(720, 1280), (1080, 1920), (37, 5000), (100, 100), (33, 640)]:
            h, w = smart_resize(H, W, factor=c.factor, min_pixels=c.min_pixels,
                                max_pixels=c.max_pixels)
            assert h % c.factor == 0 and w % c.factor == 0
            assert c.min_pixels <= h * w <= c.max_pixels or h == c.factor or w == c.factor


def test_smart_resize_preserves_aspect_within_rounding():
    # 16:9 frame, QWEN3 factor 32: 640 -> 640 (20*32), 360 -> 352 (11*32).
    h, w = smart_resize(360, 640, factor=32, min_pixels=QWEN3_VL.min_pixels,
                        max_pixels=QWEN3_VL.max_pixels)
    assert (h, w) == (352, 640)


def test_smart_resize_max_clamp_engages():
    # 2048x2048 exceeds QWEN3 max_pixels=786432 -> clamps to 864x864 (area <= max).
    h, w = smart_resize(2048, 2048, factor=32, min_pixels=QWEN3_VL.min_pixels,
                        max_pixels=QWEN3_VL.max_pixels)
    assert (h, w) == (864, 864)
    assert h * w <= QWEN3_VL.max_pixels


def test_smart_resize_min_clamp_engages():
    # 32x32 is below QWEN3 min_pixels=131072 -> scales up to 384x384 (area >= min).
    h, w = smart_resize(32, 32, factor=32, min_pixels=QWEN3_VL.min_pixels,
                        max_pixels=QWEN3_VL.max_pixels)
    assert (h, w) == (384, 384)
    assert h * w >= QWEN3_VL.min_pixels


def test_smart_resize_rejects_extreme_aspect_and_degenerate():
    with pytest.raises(ValueError):
        smart_resize(40, 10000, factor=32, min_pixels=1, max_pixels=10_000_000)  # ratio 250 > 200
    with pytest.raises(ValueError):
        smart_resize(0, 640, factor=32, min_pixels=1, max_pixels=10_000)
    with pytest.raises(ValueError):
        smart_resize(640, -1, factor=32, min_pixels=1, max_pixels=10_000)


# --------------------------------------------------------------------------- #
# frame_tokens / video_tokens
# --------------------------------------------------------------------------- #
def test_frame_tokens_hand_computed():
    # QWEN3: 448x448 -> (448/32)^2 = 14^2 = 196.
    assert frame_tokens(448, 448, QWEN3_VL) == 196
    # QWEN3 16:9 640x360 -> 20 * 11 = 220.
    assert frame_tokens(360, 640, QWEN3_VL) == 220
    # QWEN2.5: 448x448 is a multiple of 28 (16*28) -> (448/28)^2 = 16^2 = 256.
    assert frame_tokens(448, 448, QWEN2_5_VL) == 256


def test_temporal_groups_padding():
    assert temporal_groups(1, QWEN3_VL) == 1     # single frame padded to one group
    assert temporal_groups(2, QWEN3_VL) == 1
    assert temporal_groups(4, QWEN3_VL) == 2
    assert temporal_groups(5, QWEN3_VL) == 3     # 5 frames pad to 6 -> 3 groups
    with pytest.raises(ValueError):
        temporal_groups(0, QWEN3_VL)


def test_video_tokens_hand_computed_and_padding():
    ft = frame_tokens(448, 448, QWEN3_VL)  # 196
    # 448x448 never trips the 3-D clamp (2*200704 < 786432 both ways), so 3-D == 2-D here.
    assert video_tokens(4, 448, 448, QWEN3_VL) == 2 * ft   # 392
    assert video_tokens(5, 448, 448, QWEN3_VL) == 3 * ft   # padded tail group -> 588
    assert video_tokens(1, 448, 448, QWEN3_VL) == 1 * ft   # 196


# --------------------------------------------------------------------------- #
# Qwen3-VL 3-D video clamp (video_processing_qwen3_vl.py).
# The 2-D-per-frame model diverges from the real processor whenever a clamp engages; these pin
# the 3-D mirror against hand-computed values.
# --------------------------------------------------------------------------- #
def test_video_tokens_3d_not_engaged_agrees_with_2d_model():
    # No clamp engaged (even n: trigger <=> 2*rounded_area outside [min, max]) -> identical to
    # temporal_groups x frame_tokens for a spread of sizes/counts.
    for (n, H, W) in [(4, 448, 448), (5, 448, 448), (1, 448, 448), (16, 360, 640),
                      (128, 448, 448), (2, 512, 512)]:
        assert not _3d_max_engages(n, H, W, QWEN3_VL)
        g = temporal_groups(n, QWEN3_VL)
        assert video_tokens(n, H, W, QWEN3_VL) == g * frame_tokens(H, W, QWEN3_VL)


def test_video_tokens_3d_max_clamp_hand_computed():
    # 1080p @ 128F: rounded 1088x1920; trigger 128*1088*1920 = 267M > 786432*64 = 50.3M ->
    # beta = sqrt(128*1080*1920/50331648) = sqrt(5.2734) = 2.29639;
    # h = floor(1080/2.29639/32)*32 = 14*32, w = floor(1920/2.29639/32)*32 = 26*32
    # -> 14*26 = 364/frame x 64 groups = 23296. (The 2-D model would claim 46080 — the ~2x
    # over-prediction: the effective per-frame budget is longest/t_bar = 786432/2 = 393216.)
    assert video_tokens(128, 1080, 1920, QWEN3_VL) == 23296
    two_d = temporal_groups(128, QWEN3_VL) * frame_tokens(1080, 1920, QWEN3_VL)
    assert two_d == 46080  # the old model's wrong answer, pinned to document the divergence


def test_video_tokens_3d_exact_repro_8960_vs_5120():
    # 128F @ 240x336 native.
    #   rounded: round(240/32)=8 -> 256, round(336/32)=10 (banker's 10.5->10) -> 320;
    #   3-D min trigger: 128*256*320 = 10.49M NOT < 131072*64 = 8.39M -> NO clamp ->
    #     per-frame (256/32)*(320/32) = 80 -> 64 groups * 80 = 5120  (the processor's count);
    #   2-D per-frame min WOULD engage (256*320 = 81920 < 131072) and upscale to
    #     ceil-rounded 320x448 -> 140/frame -> 8960  (a 2-D mirror's wrong count).
    assert video_tokens(128, 240, 336, QWEN3_VL) == 5120
    two_d = temporal_groups(128, QWEN3_VL) * frame_tokens(240, 336, QWEN3_VL)
    assert two_d == 8960   # what a per-frame 2-D model would compute


def test_video_tokens_3d_min_clamp_hand_computed():
    # Small clip that DOES trip the 3-D total min: 2 frames @ 100x100.
    #   rounded 96x96; trigger 2*96*96 = 18432 < 131072*1 -> min engages;
    #   beta = sqrt(131072/(2*100*100)) = 2.56 -> ceil(100*2.56/32)*32 = 256 -> 8*8 = 64.
    assert video_tokens(2, 100, 100, QWEN3_VL) == 64
    # Qwen2.5 (2-D lineage) keeps the per-frame model.
    assert video_tokens(2, 100, 100, QWEN2_5_VL) == \
        temporal_groups(2, QWEN2_5_VL) * frame_tokens(100, 100, QWEN2_5_VL)


# --------------------------------------------------------------------------- #
# Budget solvers — round-trip properties
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("c", FAMILIES)
@pytest.mark.parametrize("budget", [200, 500, 1000, 4096, 12000])
@pytest.mark.parametrize("size", [(448, 448), (360, 640), (720, 1280)])
def test_max_frames_round_trip(c, budget, size):
    H, W = size
    ft = frame_tokens(H, W, c)
    if ft > budget:
        with pytest.raises(ValueError):
            max_frames_for_budget(budget, H, W, c)
        return
    n = max_frames_for_budget(budget, H, W, c)
    assert n >= 1
    assert n % c.temporal_patch_size == 0
    assert video_tokens(n, H, W, c) <= budget      # the answer ALWAYS fits (both clamp models)
    if not _3d_max_engages(n + 1, H, W, c):
        # TIGHTNESS is exact only when the 3-D max clamp does not engage (solver docstring):
        # under the engaged 3-D clamp the per-frame cost shrinks as frames grow, so the
        # 2-D-based solver is conservative — more frames may still fit.
        assert video_tokens(n + 1, H, W, c) > budget   # one more frame opens a new group


def test_max_frames_infeasible_raises():
    # A single frame at 720x1280 costs more than a tiny budget.
    with pytest.raises(ValueError):
        max_frames_for_budget(10, 720, 1280, QWEN3_VL)


@pytest.mark.parametrize("c", FAMILIES)
@pytest.mark.parametrize("budget,num_frames", [(40, 8), (300, 16), (1000, 8), (5000, 32)])
def test_max_pixels_upper_bound(c, budget, num_frames):
    grid_t = temporal_groups(num_frames, c)
    tpf = budget // grid_t
    if tpf < 1:
        with pytest.raises(ValueError):
            max_pixels_for_budget(budget, num_frames, c)
        return
    p = max_pixels_for_budget(budget, num_frames, c)
    assert p == tpf * c.factor * c.factor
    # ANY frame resized under this cap keeps the whole clip within budget.
    capped = dataclasses.replace(c, min_pixels=1, max_pixels=p)
    for (H, W) in [(4096, 4096), (2160, 3840), (1000, 1000), (512, 4096)]:
        assert video_tokens(num_frames, H, W, capped) <= budget


@pytest.mark.parametrize("c", FAMILIES)
# budgets chosen so tokens/frame (tpf) stays <= 199 -> a 1-row fill frame of ratio (tpf+1)
# stays within smart_resize's aspect-ratio<=200 guard.
@pytest.mark.parametrize("budget,num_frames", [(40, 8), (300, 16), (800, 32), (3000, 32)])
def test_max_pixels_tightness(c, budget, num_frames):
    grid_t = temporal_groups(num_frames, c)
    tpf = budget // grid_t
    assert 1 <= tpf and tpf + 1 <= 200
    p = max_pixels_for_budget(budget, num_frames, c)
    looser = dataclasses.replace(c, min_pixels=1, max_pixels=p + c.factor * c.factor)
    fill_h, fill_w = (tpf + 1) * c.factor, c.factor   # area == (tpf+1)*factor^2, one grid row
    if c.video_clamp == "2d":
        # A frame that exactly fills the next-larger cap (p + factor^2) pushes the clip over
        # budget, proving p is the largest feasible cap (in factor^2 quanta) — a 2-D property.
        assert video_tokens(num_frames, fill_h, fill_w, looser) > budget
    else:
        # 3-D clamp: the fill frame trips the TOTAL trigger (t_bar x area > cap x groups even
        # at the cap itself, since t_bar = 2 x groups), so beta shrinks it and the clip stays
        # UNDER budget — the solver's cap is conservative here, never violated (docstring).
        assert video_tokens(num_frames, fill_h, fill_w, looser) <= budget


def test_max_pixels_infeasible_raises():
    # 8 frames -> grid_t 4; budget 3 can't give even 1 token/frame.
    with pytest.raises(ValueError):
        max_pixels_for_budget(3, 8, QWEN3_VL)


# --------------------------------------------------------------------------- #
# assert_token_count
# --------------------------------------------------------------------------- #
def test_assert_token_count_ok_and_mismatch():
    assert_token_count(196, 196)  # no raise
    with pytest.raises(ValueError, match="mismatch"):
        assert_token_count(196, 200)


# --------------------------------------------------------------------------- #
# OPTIONAL cross-check vs installed transformers (skipped if absent).
# --------------------------------------------------------------------------- #
def test_cross_check_against_transformers_if_present():
    pytest.importorskip("transformers")
    try:
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import (  # type: ignore
            smart_resize as hf_smart_resize,
        )
    except Exception:  # pragma: no cover - import shape varies across versions
        pytest.skip("transformers present but qwen2_vl smart_resize not importable")

    factor, min_p, max_p = 28, 56 * 56, 28 * 28 * 1280  # Qwen2-VL image-path defaults
    for (H, W) in [(720, 1280), (1080, 1920), (100, 100), (33, 640), (2048, 2048), (40, 40)]:
        ours = smart_resize(H, W, factor=factor, min_pixels=min_p, max_pixels=max_p)
        theirs = tuple(hf_smart_resize(H, W, factor=factor, min_pixels=min_p, max_pixels=max_p))
        assert ours == theirs, f"{(H, W)}: {ours} != {theirs}"


def test_temporal_grouping_matches_ceil_definition():
    # Property: video_tokens == ceil(N / tps) * frame_tokens for random-ish N.
    for c in FAMILIES:
        ft = frame_tokens(448, 448, c)
        for n in range(1, 20):
            assert video_tokens(n, 448, 448, c) == math.ceil(n / c.temporal_patch_size) * ft
