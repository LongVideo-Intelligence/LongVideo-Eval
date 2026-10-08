"""Tests for ClipScorer on the transformers 5.x CLIP surface.

On transformers 5.x, ``CLIPModel.get_text_features`` / ``get_image_features`` return a pooling
output object, not the projected embedding tensor that 4.x returned. The scorer therefore
takes the projection explicitly: ``text_projection(text_model(...).pooler_output)`` and the
vision twin. Normalizing the pooling object directly would raise.

The fake CLIP model below reproduces that surface:

  * ``text_model`` / ``vision_model`` return an object with a ``pooler_output`` attribute and
    no tensor operations;
  * ``text_projection`` / ``visual_projection`` return tensors;
  * ``get_text_features`` / ``get_image_features`` return the same pooling-shaped objects, so
    a scorer that went back to calling them would fail in these tests.

torch, transformers and PIL are replaced in ``sys.modules`` (the scorer imports them lazily),
so the scorer runs end to end with no weights and the cosine scores are checked against
hand-computed values. numpy is real.
"""
from __future__ import annotations

import contextlib
import math
import sys
import types

import pytest

np = pytest.importorskip("numpy")

from longvideo_eval.frontend.select.clip_scorer import (  # noqa: E402
    CLIP_MODEL_ID,
    ClipScorer,
    default_clip_scorer,
)
from longvideo_eval.interfaces import FramePool  # noqa: E402


# --------------------------------------------------------------------------- #
# minimal tensor stand-ins (only the operations ClipScorer touches)
# --------------------------------------------------------------------------- #
class _T:
    """A 2-D tensor double: rows of floats, with the ``.float().cpu().numpy()`` export chain
    the scorer uses."""

    def __init__(self, rows):
        self.rows = [[float(x) for x in r] for r in rows]

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self.rows, dtype=np.float32)


def _normalize(t, dim=-1):
    """Fake ``F.normalize``: unit-length rows. It only accepts a tensor double (something with
    ``.rows``); handing it a pooling output raises AttributeError, as the real function does."""
    assert dim == -1
    out = []
    for row in t.rows:
        n = math.sqrt(sum(x * x for x in row)) or 1.0
        out.append([x / n for x in row])
    return _T(out)


class _PoolingOutput:
    """The 5.x pooling output shape: a ``pooler_output`` attribute and no tensor operations."""

    def __init__(self, pooler_output):
        self.pooler_output = pooler_output


class _Inputs(dict):
    """Processor output double: a dict (so ``**`` unpacking works) with ``.to(device)``."""

    def to(self, device):
        return self


# --------------------------------------------------------------------------- #
# fake CLIP model and processor with hand-checkable embeddings
# --------------------------------------------------------------------------- #
# Pooled vision feature of a frame, keyed by the frame's constant pixel value.
_VISION_POOLED = {0: [1.0, 0.0], 1: [0.0, 1.0], 2: [1.0, 1.0]}
_TEXT_POOLED = [0.0, 1.0]  # text_projection swaps the coordinates -> [1, 0] in joint space


class _FakeCLIPModel:
    loaded_ids: list = []

    @classmethod
    def from_pretrained(cls, model_id):
        cls.loaded_ids.append(model_id)
        return cls()

    def to(self, device):
        return self

    def eval(self):
        return self

    # --- the surface the scorer must use ---------------------------------------- #
    def text_model(self, **kwargs):
        return _PoolingOutput(_T([_TEXT_POOLED]))

    def text_projection(self, pooled):
        # Swap the coordinates: not the identity, so skipping the projection changes the scores.
        return _T([[r[1], r[0]] for r in pooled.rows])

    def vision_model(self, pixel_values=None, **kwargs):
        return _PoolingOutput(_T(pixel_values))

    def visual_projection(self, pooled):
        # Scale by 2: leaves cosines unchanged, but proves a tensor (not the pooling output)
        # is what flows on.
        return _T([[2.0 * x for x in r] for r in pooled.rows])

    # --- the older entry points, returning what 5.x really returns ---------------- #
    def get_text_features(self, **kwargs):
        return self.text_model(**kwargs)

    def get_image_features(self, **kwargs):
        return self.vision_model(**kwargs)


class _FakeCLIPProcessor:
    @classmethod
    def from_pretrained(cls, model_id):
        return cls()

    def __call__(self, text=None, images=None, return_tensors=None, **kwargs):
        if text is not None:
            return _Inputs(input_ids=[[1, 2]])
        # Each "image" is the raw uint8 array (the fake Image.fromarray is the identity); its
        # constant fill value picks the pooled vision feature.
        feats = [_VISION_POOLED[int(np.asarray(img)[0, 0, 0])] for img in images]
        return _Inputs(pixel_values=feats)


@pytest.fixture()
def fake_modules(monkeypatch):
    """Install fake torch / transformers / PIL into sys.modules for the lazy imports."""
    torch = types.ModuleType("torch")
    torch.no_grad = contextlib.nullcontext
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.nn = types.SimpleNamespace(
        functional=types.SimpleNamespace(normalize=_normalize)
    )

    transformers = types.ModuleType("transformers")
    transformers.CLIPModel = _FakeCLIPModel
    transformers.CLIPProcessor = _FakeCLIPProcessor

    pil = types.ModuleType("PIL")
    pil.Image = types.SimpleNamespace(fromarray=lambda arr: arr)

    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "PIL", pil)
    _FakeCLIPModel.loaded_ids = []


def _pool(n_frames: int) -> FramePool:
    # Frame i is filled with pixel value i, which keys into _VISION_POOLED.
    frames = np.stack([np.full((2, 2, 3), i, dtype=np.uint8) for i in range(n_frames)]) \
        if n_frames else np.empty((0, 2, 2, 3), dtype=np.uint8)
    return FramePool(video_id="v", frames=frames,
                     timestamps=[i / 2 for i in range(n_frames)], fps=2.0)


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #
def test_clip_scorer_projection_path_end_to_end(fake_modules):
    # Text: pooled [0, 1], swapped by the projection to [1, 0] (unit length).
    # Frames (pooled, scaled by 2, normalized):
    #   f0 [1, 0] -> [1, 0]          cosine 1.0
    #   f1 [0, 1] -> [0, 1]          cosine 0.0
    #   f2 [1, 1] -> [.707, .707]    cosine sqrt(2)/2
    scores = ClipScorer()(_pool(3), "query")
    assert scores == pytest.approx([1.0, 0.0, math.sqrt(2) / 2])
    # Had the scorer skipped text_projection (raw pooled [0, 1]), the scores would be
    # [0.0, 1.0, sqrt(2)/2]: the swap makes a silent bypass visible.
    assert all(isinstance(s, float) for s in scores)


def test_clip_scorer_batches_frames_without_changing_scores(fake_modules):
    # batch_size=2 over 3 frames exercises the batching loop; the cosines are identical.
    scores = ClipScorer(batch_size=2)(_pool(3), "query")
    assert scores == pytest.approx([1.0, 0.0, math.sqrt(2) / 2])
    assert ClipScorer(batch_size=1)(_pool(3), "query") == pytest.approx(scores)


def test_fake_feature_entry_points_break_normalize(fake_modules):
    # The tripwire behind the end-to-end test: the 5.x get_*_features shape cannot be
    # normalized. A scorer that called get_*_features would die with this same error.
    model = _FakeCLIPModel.from_pretrained("x")
    with pytest.raises(AttributeError):
        _normalize(model.get_text_features(input_ids=[[1]]))
    with pytest.raises(AttributeError):
        _normalize(model.get_image_features(pixel_values=[[1.0, 0.0]]))


def test_clip_scorer_empty_pool_short_circuits(fake_modules):
    scores = ClipScorer()(_pool(0), "query")
    assert scores == []
    assert _FakeCLIPModel.loaded_ids == []          # nothing is loaded for an empty pool


def test_clip_scorer_embed_accessors_return_normalized_numpy(fake_modules):
    # embed_text / embed_frames expose the projected, normalized embeddings that __call__
    # scores with, as numpy arrays.
    scorer = ClipScorer()
    text = scorer.embed_text("query")
    assert text.shape == (1, 2) and text.tolist() == [[1.0, 0.0]]  # swapped by text_projection
    frames = scorer.embed_frames(_pool(3).frames)
    assert frames.shape == (3, 2)
    assert frames.dtype == np.float32
    assert np.allclose(np.linalg.norm(frames, axis=-1), 1.0)       # unit rows
    # __call__'s scores are exactly the products of the two (one shared forward path).
    assert (frames @ text.T).reshape(-1).tolist() == pytest.approx(
        ClipScorer()(_pool(3), "query"))


def test_clip_scorer_embed_frames_empty_input(fake_modules):
    out = ClipScorer().embed_frames(np.empty((0, 2, 2, 3), dtype=np.uint8))
    assert out.shape[0] == 0


def test_clip_scorer_loads_default_model_lazily_and_once(fake_modules):
    scorer = ClipScorer()
    assert _FakeCLIPModel.loaded_ids == []          # nothing loaded at construction
    scorer(_pool(1), "q")
    assert _FakeCLIPModel.loaded_ids == ["openai/clip-vit-base-patch32"]
    assert scorer.device == "cpu"                   # no GPU reported by the fake torch
    scorer(_pool(1), "q")
    assert len(_FakeCLIPModel.loaded_ids) == 1       # cached, not reloaded


def test_default_clip_scorer_uses_the_pinned_model():
    scorer = default_clip_scorer()
    assert isinstance(scorer, ClipScorer)
    assert scorer.model_id == CLIP_MODEL_ID == "openai/clip-vit-base-patch32"
    assert scorer._model is None                    # constructing it loads nothing
