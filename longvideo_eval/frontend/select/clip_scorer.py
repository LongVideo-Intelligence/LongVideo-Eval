"""Query-conditioned per-frame relevance scoring with CLIP.

A "scorer" is any ``Callable[[FramePool, str], Sequence[float]]`` returning one relevance
score per pool frame (len == len(pool.frames)). Selectors take an injectable scorer so their
selection LOGIC is unit-testable with a trivial fake: no torch, no weights, no GPU. Only the
real ``ClipScorer`` pulls in torch/transformers, and it does so LAZILY (the package must
import on a GPU-free machine).

MODEL. ``openai/clip-vit-base-patch32`` (CLIP ViT-B/32), the scorer the LoHi paper uses for
its CLIP-based frame selection.

COST. The CLIP forward is real front-end compute. Its wall time is metered by the
orchestrator's ``select`` stage timer, so a selector leaves ``Selection.cost`` empty rather
than writing ``wall_seconds`` itself (that would double-count). Decoding the frames it scores
is charged by the decoder.
"""
from __future__ import annotations

from typing import Optional

from ...interfaces import FramePool

CLIP_MODEL_ID = "openai/clip-vit-base-patch32"  # CLIP ViT-B/32


class ClipScorer:
    """CLIP-B/32 image-text cosine similarity, one score per pool frame.

    Encodes the query text once, encodes the frames in batches, and scores each frame by the
    cosine similarity of the (normalized) embeddings.

    TRANSFORMERS 5.x API. On 5.x ``CLIPModel.get_text_features`` / ``get_image_features``
    return a ``BaseModelOutputWithPooling``, NOT the projected-embedding tensor 4.x returned.
    The projection is therefore taken explicitly —
    ``text_projection(text_model(...).pooler_output)`` /
    ``visual_projection(vision_model(...).pooler_output)`` — which is what 4.x computed
    internally, so the scores are unchanged.

    torch / transformers are imported lazily on first call; the model + processor are cached.
    ``device`` defaults to CUDA when available (GPU path) else CPU.
    """

    def __init__(self, model_id: str = CLIP_MODEL_ID, device: Optional[str] = None,
                 batch_size: int = 64) -> None:
        self.model_id = model_id
        self.device = device
        self.batch_size = batch_size
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is None:
            import torch  # lazy; this module must import GPU-free
            from transformers import CLIPModel, CLIPProcessor

            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = CLIPModel.from_pretrained(self.model_id).to(self.device).eval()
            self._processor = CLIPProcessor.from_pretrained(self.model_id)
        return self._model, self._processor

    def embed_text(self, query: str):
        """L2-normalized CLIP text embedding for ``query`` -> numpy ``[1, D]`` float32.

        ``__call__``'s text branch, exposed for selectors that need the raw embedding.
        """
        import torch  # lazy

        model, processor = self._load()
        text_inputs = processor(
            text=[query], return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        with torch.no_grad():
            # 5.x projection path (class docstring "TRANSFORMERS 5.x API"): get_text_features
            # returns BaseModelOutputWithPooling on >=5, so project the pooled output ourselves.
            text_feat = model.text_projection(model.text_model(**text_inputs).pooler_output)  # [1, D]
        text_feat = torch.nn.functional.normalize(text_feat, dim=-1)
        return text_feat.float().cpu().numpy()

    def embed_frames(self, frames):
        """L2-normalized CLIP image embeddings, one per frame -> numpy ``[N, D]`` float32.

        ``frames`` is any ``[N, H, W, C]`` uint8 array-like (a FramePool's frames or the
        encoder package's frame buffer). ``__call__``'s batched image branch, exposed for
        selectors that need the raw embeddings (e.g. a diversity term).
        """
        import numpy as np  # lazy (real path only)
        import torch
        from PIL import Image

        model, processor = self._load()
        arr = np.asarray(frames)
        n = int(arr.shape[0])
        if n == 0:
            return np.zeros((0, 1), dtype=np.float32)

        feats = []
        for start in range(0, n, self.batch_size):
            batch = arr[start:start + self.batch_size]
            images = [Image.fromarray(np.asarray(f)) for f in batch]
            image_inputs = processor(images=images, return_tensors="pt").to(self.device)
            with torch.no_grad():
                # 5.x projection path — see embed_text / class docstring.
                image_feat = model.visual_projection(
                    model.vision_model(**image_inputs).pooler_output
                )  # [b, D]
            image_feat = torch.nn.functional.normalize(image_feat, dim=-1)
            feats.append(image_feat.float().cpu().numpy())
        return np.concatenate(feats, axis=0)

    def __call__(self, pool: FramePool, query: str) -> list[float]:
        import numpy as np  # lazy (real path only)

        frames = np.asarray(pool.frames)  # [N, H, W, C] uint8
        if int(frames.shape[0]) == 0:
            return []
        # Cosine similarity of the (normalized) embeddings.
        text_feat = self.embed_text(query)      # [1, D]
        image_feat = self.embed_frames(frames)  # [N, D]
        return (image_feat @ text_feat.T).reshape(-1).astype(float).tolist()


def default_clip_scorer() -> ClipScorer:
    """The CLIP-B/32 scorer used when no scorer is injected."""
    return ClipScorer()
