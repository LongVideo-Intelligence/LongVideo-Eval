"""Timestamps must survive the frame-index round trip.

The encoder rebuilds ``frames_indices`` as round(t * fps) and the processor renders
timestamps as index / fps. Using the effective SAMPLING rate quantised sparse timestamps (up to
half the frame spacing) and flattened irregular ones; the native fps round-trips exactly.
"""
from longvideo_eval.frontend.decode.decoder import uniform_decode_indices
from longvideo_eval.frontend.encode.encoder import _mrope_metadata, metadata_fps
from longvideo_eval.interfaces import FramePool


def _shown(timestamps, fps):
    md = _mrope_metadata(timestamps, fps, 0)
    return [i / md["fps"] for i in md["frames_indices"]]


def _pool(ts, fps, native):
    return FramePool(video_id="v", frames=None, timestamps=ts, fps=fps, native_fps=native)


def test_uniform_sparse_roundtrip_uses_native_fps():
    native, dur, n = 25.0, 3600.0, 16
    ts = [i / native for i in uniform_decode_indices(int(native * dur), n)]
    pool = _pool(ts, n / dur, native)
    assert metadata_fps(pool) == native
    shown = _shown(ts, metadata_fps(pool))
    assert max(abs(a - b) for a, b in zip(shown, ts)) < 1e-6


def test_effective_rate_would_quantise():
    native, dur, n = 25.0, 3600.0, 16
    ts = [i / native for i in uniform_decode_indices(int(native * dur), n)]
    shown = _shown(ts, n / dur)
    assert max(abs(a - b) for a, b in zip(shown, ts)) > 60.0   # the old failure mode


def test_irregular_timestamps_are_kept():
    native = 30.0
    ts = [20.0, 25.0, 30.0, 1630.0, 1631.0, 1632.0, 3000.0, 3590.0]
    shown = _shown(ts, metadata_fps(_pool(ts, 8 / 3600.0, native)))
    assert max(abs(a - b) for a, b in zip(shown, ts)) < 1 / native


def test_pool_without_native_fps_keeps_old_behaviour():
    assert metadata_fps(_pool([0.0, 1.0], 2.5, None)) == 2.5
    assert metadata_fps(_pool([0.0, 1.0], 0.0, None)) == 2.0
