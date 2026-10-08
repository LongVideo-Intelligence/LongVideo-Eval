"""Hardware-free tests for the dual-stream (LoHi) encode path.

The dual-stream presentation sends N frames through the video pathway at a low scale and K of
those frames through the image pathway at a high scale. The rule under test: BOTH sizes are a
scale of the pool's NATIVE frame size, snapped to the model's patch factor by the same
``resize_target`` rule the single-stream setups use. The video stream uses
``Budget.resolution`` and the image stream ``Budget.hi_resolution`` (unset means native).

Expected token counts, with f the patch factor::

    video stream = merge_groups(N) * (lo_h / f) * (lo_w / f)
    image stream = K * (hi_h / f) * (hi_w / f)

A fake processor reports grids computed from the arrays it is actually handed, so the
encoder's own cross-check runs for real. Resizing uses PIL; the tests skip without it.
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("PIL")

from longvideo_eval.frontend.decode.decoder import resize_target  # noqa: E402
from longvideo_eval.frontend.encode.encoder import (  # noqa: E402
    QWEN_TOKENS_KIND,
    ViTEncoder,
    _resize_frames,
    vision_constants,
)
from longvideo_eval.interfaces import Budget, CostRecord, FramePool, Selection  # noqa: E402
from longvideo_eval.models.qwen_tokens import merge_groups, temporal_groups  # noqa: E402
from longvideo_eval.testing import FakeQwenProcessor  # noqa: E402


# --------------------------------------------------------------------------- #
# fake processor with an image pathway
# --------------------------------------------------------------------------- #
class _GridRows:
    """Stand-in for the ``image_grid_thw`` tensor: supports ``.tolist()``."""

    def __init__(self, rows):
        self._rows = [list(r) for r in rows]

    def tolist(self):
        return [list(r) for r in self._rows]


class _FakeImageProcessor:
    """Reports one ``(1, H/patch, W/patch)`` grid per image, from the image it is handed.

    It records every call. Like the real processor it would resize when asked to; the encoder
    must pass ``do_resize=False``, so a request to resize is treated as an error here.
    """

    def __init__(self, c, corrupt_by=0):
        self.c = c
        self.corrupt_by = corrupt_by
        self.calls = []

    def __call__(self, images=None, do_resize=True, size=None, return_tensors=None):
        if do_resize:
            raise AssertionError("the image stream must be handed over with do_resize=False")
        shapes = [tuple(int(v) for v in np.asarray(im).shape) for im in images]
        self.calls.append({"shapes": shapes, "size": size, "return_tensors": return_tensors})
        return {"image_grid_thw": _GridRows(
            [1, h // self.c.patch_size, w // self.c.patch_size + self.corrupt_by]
            for (h, w, _c) in shapes
        )}


class _RecordingVideoProcessor:
    """Wraps the package's fake video processor and records what it was called with."""

    def __init__(self, inner):
        self.inner = inner
        self.min_pixels = inner.min_pixels
        self.max_pixels = inner.max_pixels
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append({"shape": tuple(int(v) for v in kwargs["videos"][0].shape),
                           "do_resize": kwargs["do_resize"], "size": kwargs["size"]})
        return self.inner(**kwargs)


def _encoder(model_id="qwen3-vl-4b", image_corrupt_by=0, video_corrupt_by=0):
    c = vision_constants(model_id)
    proc = FakeQwenProcessor(c, corrupt_by=video_corrupt_by)
    proc.video_processor = _RecordingVideoProcessor(proc.video_processor)
    proc.image_processor = _FakeImageProcessor(c, corrupt_by=image_corrupt_by)
    enc = ViTEncoder(model_id=model_id)
    enc._processor = proc                      # inject: no transformers, no weights
    return enc, proc


def _pool(n, h, w, fps=2.0):
    # Frame i is filled with value i, so a frame can be recognised after resizing.
    frames = np.stack([np.full((h, w, 3), i, dtype=np.uint8) for i in range(n)])
    return FramePool(video_id="v", frames=frames, timestamps=[i / fps for i in range(n)],
                     fps=fps, frame_hw=(h, w), cost=CostRecord())


def _lohi_budget(n, r_l, k, r_h):
    return Budget(frame_count=n, resolution=r_l, decode_budget=n,
                  hi_i_count=k, hi_resolution=r_h, presentation="lohi")


NATIVE_SIZES = [(720, 1280), (360, 640), (1080, 1920), (1280, 720), (480, 854)]
SCALES = [(0.25, 1.0), (0.25, 0.5)]


# --------------------------------------------------------------------------- #
# both stream sizes come from the native size
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("native", NATIVE_SIZES)
@pytest.mark.parametrize("scales", SCALES)
def test_both_streams_are_scaled_from_the_native_size(native, scales):
    native_h, native_w = native
    r_l, r_h = scales
    n, hi_indices = 6, [1, 4]
    enc, proc = _encoder()
    c = vision_constants("qwen3-vl-4b")
    f = c.factor
    sel = Selection(indices=list(range(n)), per_frame_resolution=[r_l] * n,
                    hi_indices=hi_indices)
    vt = enc.encode(_pool(n, native_h, native_w), sel, _lohi_budget(n, r_l, 2, r_h))
    pkg = vt.tokens

    lo_h, lo_w = resize_target(native_h, native_w, r_l, f)
    hi_h, hi_w = resize_target(native_h, native_w, r_h, f)
    expected_lo = merge_groups(n, c) * (lo_h // f) * (lo_w // f)
    expected_hi = len(hi_indices) * (hi_h // f) * (hi_w // f)

    assert pkg["kind"] == QWEN_TOKENS_KIND
    assert pkg["native_hw"] == (native_h, native_w)
    assert pkg["frame_hw"] == (lo_h, lo_w)
    assert pkg["hi_frame_hw"] == (hi_h, hi_w)
    assert pkg["num_visual_tokens"] == expected_lo
    assert pkg["hi_num_visual_tokens"] == expected_hi
    assert vt.num_tokens == expected_lo + expected_hi

    # The arrays themselves have those sizes, and that is what the processors were handed.
    assert pkg["frames"].shape == (n, lo_h, lo_w, 3)
    assert pkg["hi_frames"].shape == (2, hi_h, hi_w, 3)
    assert proc.video_processor.calls[0]["shape"] == (n, lo_h, lo_w, 3)
    assert proc.video_processor.calls[0]["do_resize"] is False
    assert proc.image_processor.calls[0]["shapes"] == [(hi_h, hi_w, 3)] * 2
    assert pkg["do_resize"] is False

    # Realized grids as reported by the processors.
    p = c.patch_size
    assert pkg["grid_thw"] == (temporal_groups(n, c), lo_h // p, lo_w // p)
    assert pkg["hi_grid_thw"] == [(1, hi_h // p, hi_w // p)] * 2
    assert vt.cost.frames_encoded == n + 2
    assert vt.cost.vit_patch_tokens == (
        temporal_groups(n, c) * (lo_h // p) * (lo_w // p) + 2 * (hi_h // p) * (hi_w // p)
    )
    assert vt.realized_resolution == [r_l] * n


@pytest.mark.parametrize("native", [(720, 1280), (1080, 1920), (1280, 720)])
def test_image_stream_is_not_scaled_from_the_low_resolution_frames(native):
    # At r_h = 0.5 the image stream is half the NATIVE size. Halving twice (taking the already
    # reduced video frames as the base) would give a much smaller frame.
    native_h, native_w = native
    enc, _ = _encoder()
    sel = Selection(indices=[0, 1, 2, 3], per_frame_resolution=[0.25] * 4, hi_indices=[2])
    pkg = enc.encode(_pool(4, native_h, native_w), sel, _lohi_budget(4, 0.25, 1, 0.5)).tokens
    lo_h, lo_w = pkg["frame_hw"]
    assert pkg["hi_frame_hw"] == resize_target(native_h, native_w, 0.5, 32)
    assert pkg["hi_frame_hw"] != resize_target(lo_h, lo_w, 0.5, 32)
    assert pkg["hi_frame_hw"][0] > lo_h and pkg["hi_frame_hw"][1] > lo_w


def test_unset_hi_resolution_means_native():
    enc, _ = _encoder()
    sel = Selection(indices=[0, 1, 2, 3], per_frame_resolution=[0.25] * 4, hi_indices=[0, 3])
    budget = Budget(frame_count=4, resolution=0.25, hi_i_count=2, presentation="lohi")
    assert budget.hi_resolution is None
    pkg = enc.encode(_pool(4, 360, 640), sel, budget).tokens
    assert pkg["hi_frame_hw"] == resize_target(360, 640, 1.0, 32) == (352, 640)
    explicit = enc.encode(_pool(4, 360, 640), sel, _lohi_budget(4, 0.25, 2, 1.0)).tokens
    assert explicit["hi_frame_hw"] == pkg["hi_frame_hw"]
    assert explicit["hi_num_visual_tokens"] == pkg["hi_num_visual_tokens"] == 2 * 11 * 20


def test_patch_factor_follows_the_model_family():
    # Qwen2.5-VL uses a patch factor of 28, so the same native frame snaps to other sizes.
    model_id = "qwen2.5-vl-7b"
    c = vision_constants(model_id)
    assert c.factor == 28
    enc, proc = _encoder(model_id)
    n = 4
    sel = Selection(indices=list(range(n)), per_frame_resolution=[0.25] * n, hi_indices=[1])
    vt = enc.encode(_pool(n, 720, 1280), sel, _lohi_budget(n, 0.25, 1, 1.0))
    pkg = vt.tokens
    lo_h, lo_w = resize_target(720, 1280, 0.25, 28)
    hi_h, hi_w = resize_target(720, 1280, 1.0, 28)
    assert (lo_h, lo_w) != resize_target(720, 1280, 0.25, 32)
    assert pkg["frame_hw"] == (lo_h, lo_w) and pkg["hi_frame_hw"] == (hi_h, hi_w)
    assert pkg["num_visual_tokens"] == merge_groups(n, c) * (lo_h // 28) * (lo_w // 28)
    assert pkg["hi_num_visual_tokens"] == (hi_h // 28) * (hi_w // 28)


def test_an_image_costs_twice_a_video_frame_of_the_same_size():
    # The video pathway folds two frames into one temporal position; the image pathway does
    # not. With both streams at the same scale, 2 images cost as much as 4 video frames.
    enc, _ = _encoder()
    sel = Selection(indices=[0, 1, 2, 3], per_frame_resolution=[0.5] * 4, hi_indices=[0, 3])
    pkg = enc.encode(_pool(4, 256, 512), sel, _lohi_budget(4, 0.5, 2, 0.5)).tokens
    assert pkg["frame_hw"] == pkg["hi_frame_hw"] == (128, 256)
    assert pkg["num_visual_tokens"] == 2 * 4 * 8          # 2 temporal positions x 32 tokens
    assert pkg["hi_num_visual_tokens"] == 2 * 4 * 8       # 2 images x 32 tokens


# --------------------------------------------------------------------------- #
# frames, timestamps, padding
# --------------------------------------------------------------------------- #
def test_hi_frames_and_timestamps_are_the_selected_pool_frames():
    enc, _ = _encoder()
    n = 8
    sel = Selection(indices=list(range(n)), per_frame_resolution=[0.25] * n,
                    hi_indices=[1, 5, 6])
    pool = _pool(n, 256, 512, fps=4.0)
    pkg = enc.encode(pool, sel, _lohi_budget(n, 0.25, 3, 1.0)).tokens
    assert pkg["hi_timestamps"] == [0.25, 1.25, 1.5]
    # Each frame is a constant image, so resizing leaves its value: the frames can be told apart.
    assert [int(f[0, 0, 0]) for f in pkg["hi_frames"]] == [1, 5, 6]
    assert [int(f[0, 0, 0]) for f in pkg["frames"]] == list(range(n))
    assert pkg["frames"].dtype == np.uint8 and pkg["hi_frames"].dtype == np.uint8
    # Timestamps of the video stream round-trip through the metadata.
    meta = pkg["video_metadata"]
    assert [i / meta["fps"] for i in meta["frames_indices"]] == list(pool.timestamps)
    # The pool itself is left at native size for any later use.
    assert pool.frames.shape == (n, 256, 512, 3)


def test_lo_v_selection_may_be_a_subset_of_the_pool():
    enc, _ = _encoder()
    pool = _pool(8, 256, 512)
    sel = Selection(indices=[0, 2, 4, 6], per_frame_resolution=[0.25] * 4, hi_indices=[2, 6])
    pkg = enc.encode(pool, sel, _lohi_budget(4, 0.25, 2, 1.0)).tokens
    assert [int(f[0, 0, 0]) for f in pkg["frames"]] == [0, 2, 4, 6]
    assert [int(f[0, 0, 0]) for f in pkg["hi_frames"]] == [2, 6]
    assert pkg["hi_timestamps"] == [1.0, 3.0]


def test_odd_frame_count_is_padded_for_the_video_stream_only():
    enc, _ = _encoder()
    c = vision_constants("qwen3-vl-4b")
    n = 5
    sel = Selection(indices=list(range(n)), per_frame_resolution=[0.25] * n, hi_indices=[4])
    vt = enc.encode(_pool(n, 256, 512), sel, _lohi_budget(n, 0.25, 1, 1.0))
    pkg = vt.tokens
    assert pkg["frames"].shape[0] == 6                    # padded to a multiple of 2
    assert int(pkg["frames"][5][0, 0, 0]) == 4            # by repeating the last frame
    assert pkg["num_visual_tokens"] == merge_groups(5, c) * 2 * 4 == 3 * 8
    assert pkg["hi_frames"].shape[0] == 1                 # images are not padded
    assert vt.cost.frames_encoded == 5 + 1                # real frames only
    assert len(pkg["video_metadata"]["frames_indices"]) == 6


def test_no_hi_frames_gives_a_video_only_package():
    enc, proc = _encoder()
    sel = Selection(indices=[0, 1, 2, 3], per_frame_resolution=[0.25] * 4, hi_indices=[])
    vt = enc.encode(_pool(4, 256, 512), sel, _lohi_budget(4, 0.25, 0, 1.0))
    pkg = vt.tokens
    assert pkg["hi_frames"] is None and pkg["hi_num_visual_tokens"] == 0
    assert pkg["hi_grid_thw"] == [] and pkg["hi_timestamps"] == []
    assert vt.num_tokens == pkg["num_visual_tokens"] == 2 * 2 * 4
    assert proc.image_processor.calls == []               # the image pathway is not called
    # hi_indices=None (a single-stream selector under a lohi budget) behaves the same way.
    none_sel = Selection(indices=[0, 1, 2, 3], per_frame_resolution=[0.25] * 4)
    assert enc.encode(_pool(4, 256, 512), none_sel,
                      _lohi_budget(4, 0.25, 0, 1.0)).tokens["hi_frames"] is None


def test_size_bounds_bracket_each_stream():
    # With do_resize=False the area bounds must leave the frames untouched under any reading:
    # shortest_edge <= frame area <= longest_edge for both streams.
    enc, proc = _encoder()
    c = vision_constants("qwen3-vl-4b")
    n = 4
    sel = Selection(indices=list(range(n)), per_frame_resolution=[0.25] * n, hi_indices=[0])
    pkg = enc.encode(_pool(n, 1080, 1920), sel, _lohi_budget(n, 0.25, 1, 1.0)).tokens
    lo_area = pkg["frame_hw"][0] * pkg["frame_hw"][1]
    hi_area = pkg["hi_frame_hw"][0] * pkg["hi_frame_hw"][1]
    assert pkg["size"]["shortest_edge"] <= lo_area
    assert lo_area * temporal_groups(n, c) <= pkg["size"]["longest_edge"]
    assert pkg["hi_size"]["shortest_edge"] <= hi_area <= pkg["hi_size"]["longest_edge"]
    assert proc.video_processor.calls[0]["size"] == pkg["size"]
    assert proc.image_processor.calls[0]["size"] == pkg["hi_size"]
    assert hi_area > c.max_pixels                         # a native 1080p frame is above the
    #                                                       default cap, and is still not resized


# --------------------------------------------------------------------------- #
# errors
# --------------------------------------------------------------------------- #
def test_hi_frames_must_be_a_subset_of_the_video_frames():
    enc, _ = _encoder()
    sel = Selection(indices=[0, 2, 4, 6], per_frame_resolution=[0.25] * 4, hi_indices=[2, 3])
    with pytest.raises(ValueError, match="SUBSET"):
        enc.encode(_pool(8, 256, 512), sel, _lohi_budget(4, 0.25, 2, 1.0))


def test_missing_lo_resolution_is_an_error():
    enc, _ = _encoder()
    sel = Selection(indices=[0, 1, 2, 3], hi_indices=[1])
    budget = Budget(frame_count=4, hi_i_count=1, hi_resolution=1.0, presentation="lohi")
    with pytest.raises(ValueError, match="Budget.resolution"):
        enc.encode(_pool(4, 256, 512), sel, budget)


def test_empty_video_selection_is_an_error():
    enc, _ = _encoder()
    with pytest.raises(ValueError, match="non-empty"):
        enc.encode(_pool(4, 256, 512), Selection(indices=[], hi_indices=[]),
                   _lohi_budget(0, 0.25, 0, 1.0))


def test_out_of_range_scales_are_rejected():
    enc, _ = _encoder()
    sel = Selection(indices=[0, 1], hi_indices=[0])
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        enc.encode(_pool(2, 256, 512), sel, _lohi_budget(2, 0.25, 1, 1.5))
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        enc.encode(_pool(2, 256, 512), sel, _lohi_budget(2, 0.0, 1, 1.0))


def test_image_grid_mismatch_fires_loudly():
    # A processor that silently changes an image's grid must stop the run.
    enc, _ = _encoder(image_corrupt_by=1)
    sel = Selection(indices=[0, 1, 2, 3], per_frame_resolution=[0.25] * 4, hi_indices=[1])
    with pytest.raises(ValueError, match="visual-token count mismatch"):
        enc.encode(_pool(4, 256, 512), sel, _lohi_budget(4, 0.25, 1, 1.0))


def test_video_grid_mismatch_fires_loudly():
    enc, _ = _encoder(video_corrupt_by=1)
    sel = Selection(indices=[0, 1, 2, 3], per_frame_resolution=[0.25] * 4, hi_indices=[1])
    with pytest.raises(ValueError, match="visual-token count mismatch"):
        enc.encode(_pool(4, 256, 512), sel, _lohi_budget(4, 0.25, 1, 1.0))


# --------------------------------------------------------------------------- #
# dispatch and the resize helper
# --------------------------------------------------------------------------- #
def test_only_the_lohi_presentation_takes_the_dual_stream_path():
    enc, proc = _encoder()
    pool = _pool(4, 64, 128)                      # already aligned to the patch factor
    sel = Selection(indices=[0, 1, 2, 3], per_frame_resolution=[0.5] * 4, hi_indices=[1])
    single = enc.encode(pool, sel, Budget(frame_count=4, resolution=0.5, hi_i_count=1))
    assert "hi_frames" not in single.tokens and "native_hw" not in single.tokens
    assert single.tokens["frame_hw"] == (64, 128)         # frames taken as they are
    assert proc.image_processor.calls == []
    dual = enc.encode(pool, sel, _lohi_budget(4, 0.5, 1, 1.0))
    assert dual.tokens["frame_hw"] == (32, 64) and dual.tokens["hi_frame_hw"] == (64, 128)
    assert len(proc.image_processor.calls) == 1


def test_resize_frames_gives_the_exact_target_size():
    frames = np.stack([np.full((90, 160, 3), v, dtype=np.uint8) for v in (10, 200)])
    out = _resize_frames(frames, 64, 96)
    assert out.shape == (2, 64, 96, 3) and out.dtype == np.uint8
    assert int(out[0].min()) == int(out[0].max()) == 10   # a constant image stays constant
    assert int(out[1].min()) == int(out[1].max()) == 200


def test_resize_frames_returns_the_input_when_already_at_the_target_size():
    frames = np.zeros((3, 64, 96, 3), dtype=np.uint8)
    assert _resize_frames(frames, 64, 96) is frames


def test_resize_frames_downscales_content_smoothly():
    # Left half black, right half white: after a 4x reduction the halves are still there.
    frame = np.zeros((64, 128, 3), dtype=np.uint8)
    frame[:, 64:] = 255
    out = _resize_frames(frame[None], 16, 32)[0]
    assert out.shape == (16, 32, 3)
    assert int(out[8, 2, 0]) < 16 and int(out[8, 29, 0]) > 239


# --------------------------------------------------------------------------- #
# image stream sized per video to fill the reference budget
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("factor", [32, 28])
@pytest.mark.parametrize("hw", [(720, 1280), (1080, 1920), (480, 854), (360, 640),
                                (1280, 720), (2160, 3840)])
@pytest.mark.parametrize("k", [8, 4])
def test_fit_target_fills_the_reference_budget_without_exceeding_it(hw, k, factor):
    """128 frames at a quarter size plus K fitted images never exceed 16 native frames, and
    stay within a few percent of it, for every source resolution."""
    from longvideo_eval.frontend.decode.decoder import fit_target, resize_target

    h, w = hw
    tokens = lambda size: (size[0] // factor) * (size[1] // factor)  # noqa: E731
    reference = 8 * tokens(resize_target(h, w, 1.0, factor=factor))
    video = 64 * tokens(resize_target(h, w, 0.25, factor=factor))
    fitted = fit_target(h, w, (reference - video) // k, factor=factor)
    total = video + k * tokens(fitted)
    assert total <= reference
    assert total >= 0.93 * reference
    assert fitted[0] % factor == 0 and fitted[1] % factor == 0
    # the shape of the video is kept
    assert abs((fitted[1] / fitted[0]) / (w / h) - 1) < 0.1


def test_fit_target_returns_native_when_the_budget_allows_it():
    from longvideo_eval.frontend.decode.decoder import fit_target, resize_target

    assert fit_target(720, 1280, 10 ** 6) == resize_target(720, 1280, 1.0)


def test_fit_target_rejects_bad_arguments():
    from longvideo_eval.frontend.decode.decoder import fit_target

    with pytest.raises(ValueError):
        fit_target(0, 1280, 100)
    with pytest.raises(ValueError):
        fit_target(720, 1280, 0)
