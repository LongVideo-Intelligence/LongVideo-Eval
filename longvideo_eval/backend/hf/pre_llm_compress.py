"""Pre-LLM visual-token compression kernels for the Qwen3-VL family.

WHERE THESE RUN. After the vision tower and its 2x2 patch merger have run in full, after the
merged video tokens are scattered into ``inputs_embeds`` and the position ids are computed on
the full sequence, and BEFORE the language model. Every kernel returns
``(compressed_tokens, keep_visual_local_indices)``; ``prune_seam.applied_pre_llm_patch``
captures the signals a kernel needs, writes the compressed tokens back and subselects the
sequence. This module owns only the per-method selection / merge math.

Token pruning at this point is a Qwen3-VL adaptation: the methods were proposed for
LLaVA-style models with a CLIP tower, where they act on pre-projector tokens and a real CLS
attention. Here they act on post-merger LLM-space tokens, and the attention signal is the mean
self-attention of the last vision block (Qwen3-VL has no CLS token).

TWO LAYERS PER METHOD.
  * Torch-free numpy cores (``*_np`` / ``greedy_max_coverage``): the selection decisions, small
    enough to unit-test without torch.
  * Torch kernels (``*_compression``): run at generate time on the model's device. Torch is
    imported lazily, so this module imports on a GPU-free machine.

SIGNALS (captured by the seam's tower hooks):
  * ``video_features`` (num_frames, num_tokens, hidden): post-merger LLM-space embeddings, read
    back from ``inputs_embeds`` at the video placeholder positions.
  * ``cls_attention``  (num_frames, num_tokens): per-frame ``softmax(QK^T/sqrt(d))`` of the last
    vision block, averaged over heads and queries, then 2x2-merged to match the patch merger.
  * ``attn_keys``      (num_frames, num_tokens, head_dim): mean-over-heads key vectors of the
    last vision block, 2x2-merged. VisionZip's contextual similarity.
  * ``img_features``   (num_frames, num_tokens, hidden_pre): PRE-merger tower hidden states,
    averaged over each 2x2 merge window. MMTok's vision-vision term.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


# --------------------------------------------------------------------------- #
# Shared numpy helpers (torch-free).
# --------------------------------------------------------------------------- #
def to_numpy(x):
    """Seam→planner boundary conversion: accept LIVE torch tensors (cuda / bf16 / grad-attached)
    as well as plain arrays/lists, return a numpy array.

    The ``*_np`` entries below may receive live tensors at generate time, and
    ``np.asarray(cuda_tensor)`` raises (bf16 additionally has no numpy dtype, and
    grad-attached tensors refuse ``__array__``). The required chain
    ``.detach().float().cpu()`` is done ONCE here at the boundary. Duck-typed on ``detach``;
    plain arrays/lists fall through to ``np.asarray`` untouched.
    """
    if hasattr(x, "detach"):
        x = x.detach().float().cpu()
    import numpy as np

    return np.asarray(x)


def _np_l2_normalize(x, eps: float = 1e-8):
    """``x / clamp(||x||, eps)``."""
    import numpy as np

    x = np.asarray(x, dtype=np.float64)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, eps)


def _np_softmax_rows(x):
    """Numerically-stable row softmax — ``F.softmax(dim=1)``."""
    import numpy as np

    x = np.asarray(x, dtype=np.float64)
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


# =========================================================================== #
# MMTok: greedy multimodal maximum-coverage subset selection.
# alpha=0.5, tv_temp=0.01, vv_temp=0.2.
# =========================================================================== #

# Stopwords with little visual relevance.
MMTOK_NON_VISUAL_WORDS = frozenset({
    "a", "an", "the",
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "it", "they", "he", "she", "we", "you", "i",
    "what", "where", "when", "why", "how", "who", "which",
    "and", "or", "but", "so", "if", "then",
    "question", "answer",
    "to", "for", "with", "by", "from", "at", "into", "onto", "upon",
})
_MMTOK_PROMPT_1 = "Answer the question using a single word or phrase."
_MMTOK_PROMPT_2 = "Answer with the option's letter from the given choices directly."


def mmtok_extract_keywords(text: str) -> str:
    """Filter stopwords, return descriptive text.

    The question is wrapped as ``f"Question: {q}"`` BEFORE extraction; the caller passes
    that wrapped string.
    """
    import re

    if not text:
        return ""
    text = text.replace(_MMTOK_PROMPT_1, "").replace(_MMTOK_PROMPT_2, "")
    words = re.findall(r"\b\w+\b", text)
    filtered = [w for w in words if w.lower() not in MMTOK_NON_VISUAL_WORDS]
    return " ".join(filtered) if filtered else text


def mmtok_combined_np(text_embedding, video_features, img_features,
                      alpha: float = 0.5, tv_temp: float = 0.01, vv_temp: float = 0.2):
    """Assemble ``Combined = [P; alpha*Q]``.

    ``P = softmax((z @ x.T)/tv_temp, rows)/m`` over post-merger ``video_features`` (x);
    ``Q = softmax((xc @ xc.T)/vv_temp, rows)/n`` over pre-merger ``img_features`` (xc);
    ``z`` = normalized ``text_embedding``. Torch-free; the real path uses the torch kernel
    below (Q is [N,N], GPU-only for real video). Shapes: text [m, D], video [n, D],
    img [n, Dc]. Returns Combined [m+n, n].
    """
    import numpy as np

    # Boundary conversion (live cuda/bf16 tensors -> numpy) happens ONCE here (see to_numpy).
    x = _np_l2_normalize(to_numpy(video_features))
    xc = _np_l2_normalize(to_numpy(img_features))
    z = _np_l2_normalize(to_numpy(text_embedding))
    if x.ndim != 2 or z.ndim != 2 or xc.ndim != 2:
        raise ValueError(
            f"mmtok_combined_np expects 2-D arrays, got text {z.shape}, video {x.shape}, "
            f"img {xc.shape}"
        )
    m, n = z.shape[0], x.shape[0]
    P = _np_softmax_rows((z @ x.T) / tv_temp) / m
    Q = _np_softmax_rows((xc @ xc.T) / vv_temp) / n
    return np.concatenate([P, alpha * Q], axis=0)


def greedy_max_coverage(combined, k_max: int, exclude_indices=()) -> tuple:
    """Greedy maximum coverage, torch-free reference (the plain loop).

    Per step: ``delta[c] = sum_rows(clamp(Combined[:, c] - best, 0))`` (+ -inf mask on picked/
    excluded columns), pick ``argmax`` (ties -> lowest index, matching torch.argmax), update
    ``best = max(best, picked column)``. Returns SORTED indices.
    """
    import numpy as np

    # Boundary conversion (live cuda/bf16 tensors -> numpy) happens ONCE here (see to_numpy).
    C = to_numpy(combined).astype(np.float64)
    if C.ndim != 2:
        raise ValueError(f"combined must be 2-D [m+n, n], got shape {C.shape}")
    n = C.shape[1]
    k = max(0, min(int(k_max), n))
    best = np.zeros(C.shape[0], dtype=np.float64)
    score_mask = np.zeros(n, dtype=np.float64)
    for i in exclude_indices:
        score_mask[int(i)] = -np.inf
    selected = []
    for _ in range(k):
        delta = np.clip(C - best[:, None], 0.0, None).sum(axis=0) + score_mask
        best_idx = int(np.argmax(delta))
        selected.append(best_idx)
        score_mask[best_idx] = -np.inf
        best = np.maximum(best, C[:, best_idx])
    return tuple(sorted(selected))


def _column_gains(M, best, cols, chunk: int = 4096):
    """Exact marginal coverage gain of each column in ``cols``:
    ``sum_rows(clamp(M[:, c] - best, 0))``. Chunked over columns to bound the temporary."""
    import torch

    out = torch.empty(cols.shape[0], device=M.device, dtype=torch.float32)
    for s in range(0, cols.shape[0], chunk):
        c = cols[s:s + chunk]
        out[s:s + chunk] = (M.index_select(1, c) - best.unsqueeze(1)).clamp_min_(0).float().sum(0)
    return out


def lazy_greedy_max_coverage(parts, k_max: int, batch: int = 256, col_chunk: int = 4096):
    """Greedy maximum coverage over ``sum_p sum_rows max_{c in S} parts[p][r, c]``, evaluated
    lazily. Returns the selected column indices in SELECTION order (a LongTensor).

    ``parts`` is a list of matrices that share their column axis (here ``[P, alpha*Q]``).
    The objective is monotone submodular, so a column's marginal gain can only shrink as the
    selection grows: a gain computed at an earlier step is an UPPER BOUND on its current value.
    Each step therefore re-evaluates only the columns whose stale bound could still win, in
    batches of ``batch``, instead of every column. The pick is the same as plain greedy's
    (largest exact gain, lowest index on a tie), up to floating-point summation order.

    Cost per step drops from ``rows x columns`` to ``rows x (a few batches)``. Plain greedy
    needs ``k_max`` full passes; on 100k tokens that is hours, this is seconds.
    """
    import torch

    device = parts[0].device
    n = int(parts[0].shape[1])
    k = max(0, min(int(k_max), n))
    best = [torch.zeros(p.shape[0], device=device, dtype=p.dtype) for p in parts]
    all_cols = torch.arange(n, device=device)
    # Step 0: every bound is exact (one full pass, the cost of ONE plain-greedy step).
    ub = torch.zeros(n, device=device, dtype=torch.float32)
    for p, b in zip(parts, best):
        ub += _column_gains(p, b, all_cols, col_chunk)
    fresh = torch.ones(n, device=device, dtype=torch.bool)
    taken = torch.zeros(n, device=device, dtype=torch.bool)
    sel = torch.empty(k, dtype=torch.long, device=device)
    neg_inf = float("-inf")
    for i in range(k):
        while True:
            j = torch.argmax(ub)                 # first index on a tie
            if bool(fresh[j]):
                break
            # The leader's bound is stale: refresh the `batch` largest stale bounds at once.
            stale = torch.where(fresh | taken, torch.full_like(ub, neg_inf), ub)
            cand = torch.topk(stale, min(batch, n)).indices
            cand = cand[torch.isfinite(stale[cand])]
            g = torch.zeros(cand.shape[0], device=device, dtype=torch.float32)
            for p, b in zip(parts, best):
                g += _column_gains(p, b, cand, col_chunk)
            ub[cand] = g
            fresh[cand] = True
        sel[i] = j
        taken[j] = True
        ub[j] = neg_inf
        for p, b in zip(parts, best):
            torch.maximum(b, p[:, j], out=b)
        fresh.zero_()                            # every other bound is stale again
    return sel


def mmtok_compression(video_features, img_features, text_embedding,
                      target_vision_tokens: int, alpha: float = 0.5,
                      tv_temp: float = 0.01, vv_temp: float = 0.2,
                      greedy: Optional[str] = None):
    """Torch kernel: MMTok subset selection (pure SELECTION, no value change).

    Flattens ``(num_frames, num_tokens, D)`` to ``N`` tokens and greedily picks the subset
    that maximises coverage of ``Combined = [P; alpha*Q]``: ``P`` is the text->vision softmax
    over the post-merger ``video_features``, ``Q`` the vision->vision softmax over the
    pre-merger ``img_features``. Returns ``(feats[selected], selected)`` with ``selected``
    SORTED global indices into the flattened video-token space.

    ``greedy`` selects the maximiser: ``"lazy"`` (default) is :func:`lazy_greedy_max_coverage`;
    ``"plain"`` re-evaluates every column at every step (the textbook loop, kept as the
    reference the lazy version is tested against). The environment variable
    ``MMTok_GREEDY`` overrides the default. Above ``MMTok_LOWMEM_N_THRESHOLD`` tokens
    (default 50000) ``Q`` is built in half precision, row-chunked, to fit in memory.
    """
    import torch.nn.functional as F

    nf, nt, d = video_features.shape
    N = nf * nt
    feats = video_features.reshape(N, d)
    xc = img_features.reshape(N, -1)
    k_max = min(int(target_vision_tokens), N)
    mode = (greedy or os.getenv("MMTok_GREEDY", "lazy")).lower()
    if mode not in ("lazy", "plain"):
        raise ValueError(f"greedy must be 'lazy' or 'plain', got {mode!r}")

    def _l2(t):
        return t / t.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    z = _l2(text_embedding).float()
    x_norm = _l2(feats).float()
    P = z @ x_norm.T
    m, n = P.shape
    P = F.softmax(P * (1.0 / tv_temp), dim=1) / m

    lowmem = n > int(os.getenv("MMTok_LOWMEM_N_THRESHOLD", "50000"))
    if lowmem:
        xc_norm = _l2(xc).half()
        Q = xc_norm @ xc_norm.T
        Q.mul_(1.0 / vv_temp)
        soft_chunk = int(os.getenv("MMTok_LOWMEM_SOFTMAX_CHUNK", "4096"))
        for s in range(0, n, soft_chunk):
            e = min(s + soft_chunk, n)
            Q[s:e] = F.softmax(Q[s:e].float(), dim=1).to(Q.dtype)
        Q.mul_(alpha / float(n))
    else:
        xc_norm = _l2(xc).float()
        Q = F.softmax((xc_norm @ xc_norm.T) * (1.0 / vv_temp), dim=1) * (alpha / n)

    if mode == "lazy":
        selected = lazy_greedy_max_coverage([P, Q], k_max)
    else:
        selected = _plain_greedy([P, Q], k_max)
    del Q
    selected, _ = selected.sort()
    return feats[selected], selected


def _plain_greedy(parts, k_max: int, col_chunk: int = 8192):
    """Textbook greedy: every column's gain is recomputed at every step. O(k * rows * cols)."""
    import torch

    device = parts[0].device
    n = int(parts[0].shape[1])
    k = max(0, min(int(k_max), n))
    best = [torch.zeros(p.shape[0], device=device, dtype=p.dtype) for p in parts]
    all_cols = torch.arange(n, device=device)
    mask = torch.zeros(n, device=device, dtype=torch.float32)
    sel = torch.empty(k, dtype=torch.long, device=device)
    for i in range(k):
        delta = mask.clone()
        for p, b in zip(parts, best):
            delta += _column_gains(p, b, all_cols, col_chunk)
        j = torch.argmax(delta)
        sel[i] = j
        mask[j] = float("-inf")
        for p, b in zip(parts, best):
            torch.maximum(b, p[:, j], out=b)
    return sel


# =========================================================================== #
# VisionZip: dominant top-K by attention + contextual tokens merged by key similarity.
# alpha=0.928571 (the dominant share of the kept tokens).
# =========================================================================== #
def visionzip_plan_np(attn_logits, attn_keys, retention_ratio: float,
                      expansion: float, alpha: float):
    """Index/assignment PLAN for VisionZip (torch-free).

    Returns ``(dominant_indices, target_global, others_global, assign, contextual_num)``:
    ``dominant_indices`` top-``dominant_num`` by attention; ``target_global`` uniform-spaced
    contextual targets from the non-dominant set; ``others_global`` the remaining non-dominant
    tokens; ``assign[i]`` the target-slot (0..contextual_num-1) each ``others_global[i]`` merges
    into by K-vector cosine similarity. The torch kernel does the value-average from this plan.

    ``attn_logits`` [N], ``attn_keys`` [N, Dk]. Budgets: ``total = round(N*ret*exp)``,
    ``dominant = round(total*alpha)``, remainder contextual.
    """
    import numpy as np

    # Boundary conversion (live cuda/bf16 tensors -> numpy) happens ONCE here (see to_numpy).
    attn = to_numpy(attn_logits).astype(np.float64).reshape(-1)
    keys = to_numpy(attn_keys).astype(np.float64).reshape(attn.shape[0], -1)
    N = attn.shape[0]

    total_kept = max(1, int(round(N * retention_ratio * expansion)))
    dominant_num = max(1, int(round(total_kept * alpha)))
    contextual_num = max(1, total_kept - dominant_num)
    if dominant_num + contextual_num > N:
        dominant_num = min(dominant_num, N - 1)
        contextual_num = max(1, min(contextual_num, N - dominant_num))

    # 1) Dominant: top-K by attention (torch.topk largest, ties -> lower index like torch here).
    dom_indices = np.array(
        sorted(range(N), key=lambda i: (-attn[i], i))[:dominant_num], dtype=np.int64
    )

    # 2) Contextual targets: uniform-spaced from non-dominant.
    dom_mask = np.zeros(N, dtype=bool)
    dom_mask[dom_indices] = True
    nondom_global = np.where(~dom_mask)[0]
    n_nd = nondom_global.shape[0]
    contextual_num = min(contextual_num, n_nd)
    step = max(1, n_nd // contextual_num)
    target_local = np.arange(0, n_nd, step)[:contextual_num]
    target_global = nondom_global[target_local]

    # 3) Merge assignment via key-vector cosine similarity (of the NON-DOMINANT keys).
    keys_nd = _np_l2_normalize(keys[nondom_global])
    targets_norm = keys_nd[target_local]
    other_local_mask = np.ones(n_nd, dtype=bool)
    other_local_mask[target_local] = False
    others_norm = keys_nd[other_local_mask]
    others_global = nondom_global[other_local_mask]
    if others_norm.shape[0] > 0:
        sim = others_norm @ targets_norm.T
        assign = sim.argmax(axis=-1).astype(np.int64)
    else:
        assign = np.zeros(0, dtype=np.int64)
    return dom_indices, target_global, others_global, assign, int(contextual_num)


def visionzip_compression(video_features, attn_logits, attn_keys,
                          retention_ratio: float, expansion: float, alpha: float):
    """Torch kernel: VisionZip dominant + contextual value-MERGE.

    Dominant tokens kept verbatim; non-dominant tokens value-averaged into uniform-spaced
    target slots (``contextual = target + mean(assigned others)``), then dominant+contextual
    concatenated and sorted by global index. Returns ``(compressed_tokens, keep_indices)``.
    The keep_indices span dominant + target positions; the value at each target is MODIFIED
    (this is why the seam writes tokens back instead of only dropping).
    """
    import numpy as np
    import torch

    nf, nt, d = video_features.shape
    N = nf * nt
    feats = video_features.reshape(N, d)
    device = feats.device

    dom_idx, target_global, others_global, assign, contextual_num = visionzip_plan_np(
        attn_logits, attn_keys, retention_ratio, expansion, alpha
    )
    dom_idx_t = torch.as_tensor(dom_idx, dtype=torch.long, device=device)
    target_t = torch.as_tensor(np.asarray(target_global), dtype=torch.long, device=device)

    target_feats = feats[target_t]
    if others_global.shape[0] > 0:
        others_t = torch.as_tensor(np.asarray(others_global), dtype=torch.long, device=device)
        assign_t = torch.as_tensor(assign, dtype=torch.long, device=device)
        one_hot = torch.zeros(others_t.shape[0], contextual_num, dtype=feats.dtype, device=device)
        one_hot.scatter_(1, assign_t.unsqueeze(-1), 1.0)
        counts = one_hot.sum(dim=0).clamp_min(1.0).unsqueeze(-1)
        aggregated = (one_hot.T @ feats[others_t]) / counts
    else:
        aggregated = torch.zeros(contextual_num, d, dtype=feats.dtype, device=device)
    contextual_feats = target_feats + aggregated

    all_indices = torch.cat([dom_idx_t, target_t])
    all_feats = torch.cat([feats[dom_idx_t], contextual_feats], dim=0)
    order = torch.argsort(all_indices)
    return all_feats[order], all_indices[order]


# =========================================================================== #
# FlashVID: dynamic segmentation -> attention/diversity selection -> temporal merge ->
# density-peak clustering. alpha=0.7.
# =========================================================================== #
@dataclass
class FlashVidVisionConfig:
    """Vision-side FlashVid hyperparameters (mirror of the used FlashVidConfig fields)."""

    retention_ratio: float = 0.25
    expansion: float = 1.0        # no in-LLM pruning stage here -> 1.0
    alpha: float = 0.7            # ADTS/STTM split (NOT a prune ratio)
    do_segment: bool = True
    segment_threshold: float = 0.9
    min_segment_num: int = 8
    complementary_segment: bool = True
    temporal_threshold: float = 0.8
    token_selection_method: str = "attn_div"
    num_attn_div_tokens: Optional[int] = field(default=None)
    num_sttm_tokens: Optional[int] = field(default=None)


def pairwise_cosine_distances_np(feats):
    """``1 - cos_sim`` pairwise. feats [B, n, d] -> [B, n, n]."""
    import numpy as np

    f = np.asarray(feats, dtype=np.float64)
    normed = f / np.maximum(np.linalg.norm(f, axis=-1, keepdims=True), 1e-12)
    return 1.0 - normed @ np.transpose(normed, (0, 2, 1))


def attn_div_v2_select_np(features, cls_attention, num_retained: int):
    """Farthest-point attention/diversity selection, torch-free.

    features [B, n, d], cls_attention [B, n], returns keep_indices [B, num_retained] SORTED.
    Distance = pairwise-cosine calibrated by (1) global cls_attention*1e6 and (2) event
    relevance ``einsum(features, pooled).mean``. First token = argmax of the 2nd-smallest row
    distance; then greedy min-distance-to-selected argmax (no re-pick guard in v2).
    """
    import numpy as np

    # Boundary conversion (live cuda/bf16 tensors -> numpy) happens ONCE here (see to_numpy).
    feats = to_numpy(features).astype(np.float64)
    pooled = feats.mean(axis=1)                                   # [B, d]
    gca = to_numpy(cls_attention).astype(np.float64) * 1e6        # [B, n]
    B, n, d = feats.shape
    dist = pairwise_cosine_distances_np(feats)                    # [B, n, n]
    term1 = gca[:, None, :]                                       # [B, 1, n]
    # Event relevance: einsum "b n d, c d -> b c n" (pooled is [c=B, d]), mean over c.
    local = np.einsum("bnd,cd->bcn", feats, pooled).mean(axis=1)  # [B, n]
    term2 = local[:, None, :]
    dist = dist * term1 * term2
    keep = np.zeros((B, num_retained), dtype=np.int64)
    # first token: argmax over the 2nd-smallest distance per column.
    part = np.sort(dist, axis=1)[:, 1, :]                         # [B, n] (k=2 smallest -> idx 1)
    keep[:, 0] = np.argmax(part, axis=-1)
    for i in range(1, num_retained):
        sub = np.take_along_axis(dist, keep[:, :i][:, :, None].repeat(n, axis=2), axis=1)
        min_dist = sub.min(axis=1)                                # [B, n]
        keep[:, i] = np.argmax(min_dist, axis=-1)
    keep.sort(axis=1)
    return keep


def segment_np(frame_means, segment_threshold: float, min_segment_num: int,
               complementary_segment: bool = True):
    """DySeg segment lengths from frame-mean transition similarity, torch-free.

    ``frame_means`` [num_frames, d]. Cuts where consecutive-frame cosine sim < threshold; if
    fewer than ``min_segment_num`` segments and ``complementary_segment``, adds the lowest-sim
    remaining transitions as extra cuts (the topk-lowest fill). Returns segment
    lengths (numpy int array summing to num_frames).
    """
    import numpy as np

    # Boundary conversion (live cuda/bf16 tensors -> numpy) happens ONCE here (see to_numpy).
    fm = to_numpy(frame_means).astype(np.float64)
    num_frames = fm.shape[0]
    normed = fm / np.maximum(np.linalg.norm(fm, axis=-1, keepdims=True), 1e-12)
    trans = np.sum(normed[:-1] * normed[1:], axis=-1)             # [num_frames-1]
    cut = np.where(trans < segment_threshold)[0]

    num_segments = cut.shape[0] + 1
    if num_segments < min_segment_num and complementary_segment:
        remaining = min_segment_num - num_segments
        masked = trans.copy()
        masked[masked < segment_threshold] = 1.0
        k = min(remaining, masked.shape[0])
        if k > 0:
            # lowest-k transitions (ties -> lower index, matching torch.topk largest=False here).
            extra = np.array(sorted(range(masked.shape[0]),
                                    key=lambda i: (masked[i], i))[:k], dtype=np.int64)
            cut = np.sort(np.concatenate([cut, extra]))
    padded = np.concatenate([[-1], cut, [num_frames - 1]]).astype(np.int64)
    return np.diff(padded)


def flashvid_compression(video_features, cls_attention, config: FlashVidVisionConfig):
    """Torch kernel: FlashVID vision-side compression.

    DySeg segmentation -> per-segment ADTS selection + TAM temporal-average-merge + DPC-kNN
    spatial cluster-merge. VALUE-MODIFYING (TAM sums features, DPC-kNN averages clusters), which
    is why pre_llm writes the compressed tokens back. Returns
    ``(sorted_tokens, sorted_global_indices)`` into the flattened video-token space.

    ``expansion=1.0`` (no in-LLM pruning stage), so ``token_budget = ceil(num_tokens *
    retention_ratio)``. Imports torch lazily. The temporal merge writes into
    ``video_features`` in place: pass a copy if the caller still needs the original.
    """

    from . import _flashvid_torch as fv  # lazy torch helpers, torch-free at module import

    return fv.flashvid_compression_torch(video_features, cls_attention, config)


