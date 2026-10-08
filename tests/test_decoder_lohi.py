"""The dual-stream presentation keeps the decoded pool at native size.

Both streams of the LoHi presentation are defined as a scale of the source video's native
size, so the decoder must not resize at decode time: the encoder renders the two sizes from
one native pool.

No video file and no real decoder are needed. A small fake ``decord`` module is placed in
``sys.modules``; it records how each reader was opened (with or without a target size) and
returns frames of that size, which is all the decoder's control flow depends on.
"""
from __future__ import annotations

import sys
import types

import pytest

np = pytest.importorskip("numpy")

from longvideo_eval.frontend.decode.decoder import DecordDecoder, resize_target  # noqa: E402
from longvideo_eval.interfaces import Budget  # noqa: E402
from longvideo_eval.runners.setups import get_setup  # noqa: E402

NATIVE_H, NATIVE_W, TOTAL, FPS = 360, 640, 240, 24.0


class _Batch:
    def __init__(self, arr):
        self._arr = arr

    def asnumpy(self):
        return self._arr


@pytest.fixture()
def fake_decord(monkeypatch, tmp_path):
    """Install the fake module and return ``(video_path, opened_readers)``."""
    opened = []

    class VideoReader:
        def __init__(self, path, ctx=None, width=-1, height=-1, num_threads=0):
            self.h = NATIVE_H if height in (-1, None) else height
            self.w = NATIVE_W if width in (-1, None) else width
            self.batches = []
            opened.append(self)

        def __len__(self):
            return TOTAL

        def get_avg_fps(self):
            return FPS

        def __getitem__(self, i):
            return np.zeros((self.h, self.w, 3), dtype=np.uint8)

        def get_batch(self, indices):
            self.batches.append(list(indices))
            # Frame j of the batch is filled with j, so the order is recognisable.
            arr = np.stack([np.full((self.h, self.w, 3), j % 256, dtype=np.uint8)
                            for j in range(len(indices))])
            return _Batch(arr)

    module = types.ModuleType("decord")
    module.VideoReader = VideoReader
    module.cpu = lambda index=0: ("cpu", index)
    monkeypatch.setitem(sys.modules, "decord", module)
    video = tmp_path / "clip.mp4"
    video.touch()                              # the decoder checks the path exists first
    return str(video), opened


def test_lohi_presentation_decodes_at_native_size(fake_decord):
    video, opened = fake_decord
    budget = Budget(frame_count=16, resolution=0.25, decode_budget=16,
                    hi_i_count=4, hi_resolution=1.0, presentation="lohi")
    pool = DecordDecoder().decode(video, budget)
    assert pool.frames.shape == (16, NATIVE_H, NATIVE_W, 3)     # native, despite resolution=0.25
    assert pool.frame_hw == (NATIVE_H, NATIVE_W)
    # One reader, opened without a target size: a single decode serves both streams.
    assert len(opened) == 1
    assert (opened[0].h, opened[0].w) == (NATIVE_H, NATIVE_W)
    assert len(opened[0].batches) == 1 and len(opened[0].batches[0]) == 16
    assert pool.cost.frames_decoded == 16


def test_video_presentation_with_the_same_resolution_resizes_at_decode(fake_decord):
    # The contrast case: the same budget as a single video stream is resized by the decoder.
    video, opened = fake_decord
    pool = DecordDecoder().decode(
        video, Budget(frame_count=16, resolution=0.25, decode_budget=16))
    target = resize_target(NATIVE_H, NATIVE_W, 0.25)
    assert target == (96, 160)
    assert pool.frames.shape == (16,) + target + (3,)
    assert pool.frame_hw == target
    assert len(opened) == 2                                     # size lookup + resized reader
    assert (opened[1].h, opened[1].w) == target
    assert opened[0].batches == []                              # frames come from the resized one


def test_lohi_and_video_presentations_sample_the_same_frames(fake_decord):
    video, opened = fake_decord
    DecordDecoder().decode(video, Budget(resolution=0.25, decode_budget=16,
                                         hi_i_count=4, presentation="lohi"))
    DecordDecoder().decode(video, Budget(resolution=0.25, decode_budget=16))
    lohi_indices = opened[0].batches[0]
    video_indices = opened[2].batches[0]
    assert lohi_indices == video_indices                        # only the size differs
    assert lohi_indices[0] == 0 and lohi_indices[-1] == TOTAL - 1


def test_lohi_pool_keeps_timestamps_and_native_fps(fake_decord):
    video, _ = fake_decord
    pool = DecordDecoder().decode(video, Budget(resolution=0.25, decode_budget=8,
                                                hi_i_count=2, presentation="lohi"))
    assert pool.native_fps == FPS
    assert len(pool.timestamps) == 8
    assert pool.timestamps[0] == 0.0 and pool.timestamps[-1] == (TOTAL - 1) / FPS
    assert pool.fps == pytest.approx(8 / (TOTAL / FPS))         # effective sampling rate
    assert pool.video_id == "clip"


def test_lohi_presentation_ignores_the_decoder_patch_factor(fake_decord):
    # The patch factor only matters when the decoder resizes; a native pool is the same for
    # every model family.
    video, _ = fake_decord
    budget = Budget(resolution=0.25, decode_budget=4, hi_i_count=1, presentation="lohi")
    a = DecordDecoder(factor=32).decode(video, budget)
    b = DecordDecoder(factor=28).decode(video, budget)
    assert a.frames.shape == b.frames.shape == (4, NATIVE_H, NATIVE_W, 3)


@pytest.mark.parametrize("setup", ["lohi_128f_r25_k8", "lohi_128f_r25_k4"])
def test_lohi_setups_decode_a_native_pool(fake_decord, setup):
    video, opened = fake_decord
    budget = get_setup(setup)
    pool = DecordDecoder().decode(video, budget)
    assert pool.frames.shape == (budget.frame_count, NATIVE_H, NATIVE_W, 3)
    assert len(opened) == 1


def test_lohi_native_pool_feeds_the_selector_and_the_frame_counts_line_up(fake_decord):
    from longvideo_eval.frontend.select.uniform import LoHiUniformSelector

    video, _ = fake_decord
    budget = get_setup("lohi_128f_r25_k8")
    pool = DecordDecoder().decode(video, budget)
    sel = LoHiUniformSelector().select(pool, "q", budget)
    assert len(sel.indices) == budget.frame_count == pool.frames.shape[0]
    assert len(sel.hi_indices) == budget.hi_i_count
