"""Tests for the real decord decoder.

Three layers:
  * ``uniform_decode_indices`` — a pure function (no video, no numpy required); exhaustively
    tested here since it *is* the fairness contract (every method decodes the same frames).
  * ``resize_target`` — a pure function (no decord/numpy); the resolution-axis contract (r is a
    per-side scale => ~r^2 tokens/frame), factor alignment, and small-video clamps.
  * ``DecordDecoder`` — the fail-loud missing-file guard runs everywhere (it trips BEFORE the
    decord import); the full decode / decode-at-r paths are gated on ``importorskip`` for
    decord/numpy + a video writer (CI has none — the skip is acceptable in the TEST only; the
    decoder itself never skips or falls back).
"""
from __future__ import annotations

import pytest

from longvideo_eval.frontend.decode.decoder import (
    DecordDecoder,
    resize_target,
    uniform_decode_indices,
)
from longvideo_eval.models.qwen_tokens import QWEN3_VL
from longvideo_eval.interfaces import Budget


# --------------------------------------------------------------------------- #
# uniform_decode_indices — pure, numpy-free
# --------------------------------------------------------------------------- #
def test_none_budget_returns_all_frames():
    assert uniform_decode_indices(5, None) == [0, 1, 2, 3, 4]
    assert uniform_decode_indices(0, None) == []


def test_budget_exceeding_total_clamps_to_total():
    # n = min(budget, total) = total, so every frame exactly once.
    assert uniform_decode_indices(4, 10) == [0, 1, 2, 3]
    assert uniform_decode_indices(4, 4) == [0, 1, 2, 3]


def test_budget_zero_is_empty():
    assert uniform_decode_indices(100, 0) == []
    assert uniform_decode_indices(0, 0) == []


def test_negative_budget_raises():
    with pytest.raises(ValueError):
        uniform_decode_indices(100, -1)


def test_negative_total_raises():
    with pytest.raises(ValueError):
        uniform_decode_indices(-1, 4)


def test_budget_one_returns_first_frame():
    assert uniform_decode_indices(100, 1) == [0]
    assert uniform_decode_indices(1, 1) == [0]


def test_hand_computed_spacing_truncates_like_linspace():
    # Convention: np.linspace(0, total-1, n, dtype=int) == truncation toward zero.
    # total=10, n=5 -> floats [0, 2.25, 4.5, 6.75, 9] -> truncated [0, 2, 4, 6, 9].
    assert uniform_decode_indices(10, 5) == [0, 2, 4, 6, 9]
    # total=16, n=8 -> [0, 2.14, 4.29, 6.43, 8.57, 10.71, 12.86, 15] -> [0, 2, 4, 6, 8, 10, 12, 15].
    assert uniform_decode_indices(16, 8) == [0, 2, 4, 6, 8, 10, 12, 15]


def test_indices_are_sorted_unique_and_in_range():
    for total in (7, 13, 64, 128):
        for budget in (1, 3, total // 2 or 1, total):
            idx = uniform_decode_indices(total, budget)
            assert idx == sorted(idx)                     # ascending
            assert len(idx) == len(set(idx))              # unique (clamped n <= total)
            assert len(idx) == min(budget, total)         # count semantics preserved
            assert all(0 <= i <= total - 1 for i in idx)  # in range
            assert idx[0] == 0                            # always starts at frame 0
            if len(idx) >= 2:
                assert idx[-1] == total - 1               # multi-frame sampling hits the end


def test_matches_numpy_linspace_reference():
    # The compatibility target: np.linspace(0, total-1, n, dtype=int). Skips if numpy absent.
    np = pytest.importorskip("numpy")
    for total in (5, 16, 37, 128):
        for n in range(1, total + 1):
            ref = np.linspace(0, total - 1, n, dtype=int).tolist()
            assert uniform_decode_indices(total, n) == ref, (total, n)


# --------------------------------------------------------------------------- #
# resize_target — pure, decord-free, numpy-free
# --------------------------------------------------------------------------- #
def test_resize_target_native_aligns_to_factor():
    # r=1.0: keep native size but snap each side to a multiple of factor (32), smart_resize's
    # round style. 720 -> round(720/32)*32 = 704; 1280 is already 40*32 -> 1280.
    assert resize_target(720, 1280, 1.0, factor=32) == (704, 1280)
    # A size already factor-aligned at r=1.0 is a pass-through (no-op).
    assert resize_target(352, 640, 1.0, factor=32) == (352, 640)


def test_resize_target_scales_each_side_per_r():
    # r is PER SIDE, not per pixel: half r halves each side (=> ~quarter the tokens/frame).
    assert resize_target(720, 1280, 0.5, factor=32) == (352, 640)   # 360->352, 640->640
    assert resize_target(720, 1280, 0.25, factor=32) == (192, 320)  # 180->192, 320->320


def test_resize_target_token_cost_scales_like_r_squared():
    # The iso-token property: N * tokens_per_frame is ~constant when N * r^2 is held constant.
    # tokens_per_frame = (H/factor) * (W/factor) on the aligned target.
    def ft(h, w):
        return (h // 32) * (w // 32)
    tpf = {r: ft(*resize_target(720, 1280, r, factor=32)) for r in (1.0, 0.5, 0.25)}
    # 16F@1.0 / 64F@0.5 / 256F@0.25 -> per-frame token cost drops ~4x each step (r halves).
    assert 16 * tpf[1.0] == pytest.approx(64 * tpf[0.5], rel=0.15)
    assert 64 * tpf[0.5] == pytest.approx(256 * tpf[0.25], rel=0.15)


def test_resize_target_clamps_tiny_video_to_one_patch():
    # A video smaller than one patch (or scaled below it) never collapses below `factor`.
    assert resize_target(10, 10, 1.0, factor=32) == (32, 32)
    assert resize_target(64, 64, 0.25, factor=32) == (32, 32)  # 16 -> clamped up to 32


def test_resize_target_default_factor_is_qwen3_vl():
    assert resize_target(720, 1280, 0.5) == resize_target(720, 1280, 0.5, factor=QWEN3_VL.factor)
    assert QWEN3_VL.factor == 32


@pytest.mark.parametrize("bad_r", [0.0, -0.5, 1.5, 2.0])
def test_resize_target_rejects_out_of_range_r(bad_r):
    with pytest.raises(ValueError):
        resize_target(720, 1280, bad_r, factor=32)


@pytest.mark.parametrize("h,w,factor", [(0, 100, 32), (100, -1, 32), (100, 100, 0)])
def test_resize_target_rejects_bad_dims_or_factor(h, w, factor):
    with pytest.raises(ValueError):
        resize_target(h, w, 0.5, factor=factor)


# --------------------------------------------------------------------------- #
# DecordDecoder — fail-loud guard (runs without decord installed)
# --------------------------------------------------------------------------- #
def test_missing_file_raises_before_decord():
    with pytest.raises(FileNotFoundError):
        DecordDecoder().decode("/no/such/video.mp4", Budget(decode_budget=8))


# --------------------------------------------------------------------------- #
# DecordDecoder — real decode path (needs decord + numpy + a video writer)
# --------------------------------------------------------------------------- #
def _make_test_video(path, n_frames: int = 16, size: int = 32, fps: int = 8,
                     height: int = None, width: int = None):
    """Write a tiny per-frame-distinct clip; skip the test if no encoder is importable.

    ``size`` gives a square frame; pass explicit ``height``/``width`` for a non-square clip
    (used to check decode-at-resolution produces the exact target (H, W)).
    """
    np = pytest.importorskip("numpy")
    h = height if height is not None else size
    w = width if width is not None else size
    frames = [np.full((h, w, 3), (i * 16) % 256, dtype=np.uint8) for i in range(n_frames)]

    try:
        import imageio.v2 as iio  # imageio + imageio-ffmpeg
    except ImportError:
        iio = None
    if iio is not None:
        try:
            writer = iio.get_writer(str(path), fps=fps, macro_block_size=1)
            for f in frames:
                writer.append_data(f)
            writer.close()
            return
        except Exception:  # noqa: BLE001 - imageio without its ffmpeg plugin; try PyAV below
            pass

    try:
        import av  # PyAV fallback
    except ImportError:
        pytest.skip("no video writer available (need imageio-ffmpeg or av)")
    container = av.open(str(path), mode="w")
    stream = container.add_stream("mpeg4", rate=fps)
    stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
    for f in frames:
        for packet in stream.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def test_decord_decode_respects_budget_and_meters(tmp_path):
    pytest.importorskip("decord")
    pytest.importorskip("numpy")
    video = tmp_path / "clip.mp4"
    _make_test_video(video, n_frames=16, size=32, fps=8)

    pool = DecordDecoder().decode(str(video), Budget(decode_budget=8))

    assert pool.frames.shape[0] == 8                      # decode_budget honored
    assert pool.frames.ndim == 4                          # [N, H, W, C]
    assert pool.frames.dtype.name == "uint8"
    assert pool.cost.frames_decoded == 8                  # metered frame count
    assert pool.cost.decode_seconds > 0.0                 # real wall time measured
    assert len(pool.timestamps) == 8
    assert all(b > a for a, b in zip(pool.timestamps, pool.timestamps[1:]))  # monotonic
    assert pool.video_id == "clip"                        # path stem
    assert pool.fps > 0.0                                 # effective sampled fps
    assert pool.frame_hw == (pool.frames.shape[1], pool.frames.shape[2])  # realized (H, W) recorded


def test_decord_decode_at_resolution_resizes_to_target(tmp_path):
    pytest.importorskip("decord")
    pytest.importorskip("numpy")
    # Native 128x256 (H x W), both factor-32 multiples so the target math is exact.
    video = tmp_path / "clip.mp4"
    _make_test_video(video, n_frames=16, size=None, fps=8, height=128, width=256)

    pool = DecordDecoder().decode(str(video), Budget(decode_budget=8, resolution=0.5))

    exp_h, exp_w = resize_target(128, 256, 0.5, factor=QWEN3_VL.factor)  # (64, 128)
    assert (exp_h, exp_w) == (64, 128)
    assert pool.frames.shape[0] == 8                       # decode_budget still honored
    assert pool.frames.shape[1:3] == (exp_h, exp_w)        # decode-at-r produced target size
    assert pool.frame_hw == (exp_h, exp_w)                 # realized (H, W) recorded on the pool
    assert pool.cost.frames_decoded == 8                   # only pooled frames metered (probe folded in)
    assert pool.cost.decode_seconds > 0.0


def test_decord_decode_none_budget_decodes_all(tmp_path):
    pytest.importorskip("decord")
    pytest.importorskip("numpy")
    video = tmp_path / "clip.mp4"
    _make_test_video(video, n_frames=16, size=32, fps=8)

    pool = DecordDecoder().decode(str(video), Budget(decode_budget=None))
    assert pool.frames.shape[0] == pool.cost.frames_decoded
    assert pool.cost.frames_decoded >= 1                  # decoded the whole clip
