"""Pruner seam for the HF Qwen backend: how a token-pruning method reaches the model.

WHY A SEAM AND NOT A PRUNER-STAGE FORWARD. For an HF Qwen checkpoint, the vision tower, the
visual-token drop and the LLM prefill all happen inside ONE ``model.generate`` call. The
encode stage (``frontend/encode/encoder.py``) only *preprocesses*: its ``VisualTokens.tokens``
is a CPU package (frames + timestamp metadata + pixel budget), NOT embeddings. A
``Pruner``-stage component therefore cannot drop tokens itself. It *declares* a
:class:`PatchSpec` (stashed under ``VisualTokens.tokens[PATCH_SPEC_KEY]``) and
``QwenHFBackend`` applies it around ``generate``.

THE PATCH POINT: ``"pre_llm"``. Post-merger, post-scatter, post-position-ids, PRE-language
model, at TOKEN granularity. The wrapper reads the merged video tokens back from
``inputs_embeds``, runs the method's compression over signals captured read-only from the
tower, writes the (possibly value-merged) tokens back at the kept positions, and subselects
``inputs_embeds`` / ``position_ids`` / ``attention_mask`` / ``cache_position`` /
``visual_pos_masks`` / deepstack features by the surviving positions. Position ids are
index-gathered, not recomputed, so survivors keep their original spatio-temporal positions.
Because this happens after the position ids exist, arbitrary per-token pruning is valid.

COST ACCOUNTING. The vision tower still runs on the FULL frame set, and is metered as such.
The reduced sequence enters EVERY decoder layer, so the LLM prefill genuinely shrinks, while
the raw ``input_ids`` stay full length (the reduction happens on ``inputs_embeds`` inside the
forward). The honest ``prefill_tokens`` is the raw input width minus the dropped visual
tokens; the backend records that, plus the pre/post visual counts, in the results arm.

This module is torch-free at import; tensor ops duck-type over list / numpy / torch so the
alignment logic unit-tests without weights.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

# Key under which a Pruner stashes its PatchSpec inside VisualTokens.tokens (the qwen-hf-video
# package). Absent => no pruning (the identity/dense path); the backend runs untouched.
PATCH_SPEC_KEY = "patch_spec"

PATCH_POINTS = ("pre_llm",)

# pre_llm compression kernels (dispatch id -> pre_llm_compress). "random" is the control: the
# same prune point, token granularity and subselect, with a STATIC seeded keep set (no tower
# signal, so no capture) — ``spec.keep_indices`` carries the real surviving set.
PRE_LLM_COMPRESSIONS = ("mmtok", "visionzip", "flashvid", "random")
# The static-control subset (keep set decided upstream; no capture).
PRE_LLM_STATIC = ("random",)
# Which tower signals each kernel consumes. Capturing only these matters for cost: the
# attention proxy re-runs the last vision block's QK^T softmax for every frame.
PRE_LLM_SIGNALS = {
    "mmtok": ("img_features",),
    "visionzip": ("cls_attention", "attn_keys"),
    "flashvid": ("cls_attention",),
    "random": (),
}


@dataclass(frozen=True)
class PatchSpec:
    """A declarative patch a Pruner hands the backend (the Pruner does NOT prune itself).

    ``keep_indices`` are sorted, unique MERGED-visual-token positions into
    ``[0, pre_prune_tokens)``. For the static control they are the tokens that survive; for
    the learned kernels they are a placeholder whose length is the target count (the kernel
    decides the real set at generate time). ``params`` / ``method`` ride into the results arm.
    """

    method: str                          # pruner id (e.g. "mmtok"), for results metadata
    patch_point: str                     # "pre_llm"
    pre_prune_tokens: int                # merged visual tokens BEFORE prune (== encoder count)
    keep_indices: tuple[int, ...]        # SORTED, unique, in [0, pre_prune_tokens)
    tower_target: str = "model.visual"   # dotted path to the vision tower
    compression: Optional[str] = None    # one of PRE_LLM_COMPRESSIONS
    params: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.patch_point not in PATCH_POINTS:
            raise ValueError(f"patch_point {self.patch_point!r} not in {PATCH_POINTS}")
        if self.pre_prune_tokens <= 0:
            raise ValueError(f"pre_prune_tokens must be positive, got {self.pre_prune_tokens}")
        ki = tuple(int(i) for i in self.keep_indices)
        if not ki:
            raise ValueError("keep_indices is empty — a pruner must keep at least one token")
        if len(set(ki)) != len(ki):
            raise ValueError("keep_indices must be unique")
        if ki != tuple(sorted(ki)):
            raise ValueError("keep_indices must be sorted ascending")
        if ki[0] < 0 or ki[-1] >= self.pre_prune_tokens:
            raise ValueError(
                f"keep_indices out of range [0, {self.pre_prune_tokens}): "
                f"min={ki[0]}, max={ki[-1]}"
            )
        object.__setattr__(self, "keep_indices", ki)
        if self.compression not in PRE_LLM_COMPRESSIONS:
            raise ValueError(
                f"pre_llm patch requires compression in {PRE_LLM_COMPRESSIONS}, got "
                f"{self.compression!r}"
            )

    @property
    def keep_count(self) -> int:
        return len(self.keep_indices)


def take_dim(x: Any, indices: Any, dim: int) -> Any:
    """Subselect ``x`` along ``dim`` at ``indices`` — torch ``index_select`` / numpy ``take``."""
    try:
        import torch  # lazy

        if isinstance(x, torch.Tensor):
            idx = indices if isinstance(indices, torch.Tensor) else torch.as_tensor(
                [int(i) for i in indices], dtype=torch.long)
            return x.index_select(dim, idx.to(device=x.device, dtype=torch.long))
    except ImportError:
        pass
    import numpy as np  # tests use numpy stand-ins; the real path is torch above

    return np.take(x, [int(i) for i in indices], axis=dim)


def subselect_rows(rows: Any, keep_indices: Sequence[int]) -> Any:
    """Return ``rows`` restricted to ``keep_indices`` along axis 0, order preserved."""
    if isinstance(rows, (list, tuple)):
        return type(rows)(rows[i] for i in keep_indices)
    return take_dim(rows, keep_indices, 0)


def resolve_module(model: Any, dotted: str) -> Any:
    """Walk a dotted attribute path to the target submodule; fail loud with what WAS found."""
    obj = model
    for part in dotted.split("."):
        if not hasattr(obj, part):
            avail = [a for a in dir(obj) if not a.startswith("_")][:40]
            raise AttributeError(
                f"prune seam: cannot resolve target {dotted!r} — {part!r} missing on "
                f"{type(obj).__name__}. Available (first 40): {avail}"
            )
        obj = getattr(obj, part)
    return obj


def scatter_seq_rows(x: Any, positions: Sequence[int], values: Any) -> Any:
    """Return a copy of ``x`` with ``x[:, positions, :] = values`` (batch dim 0, seq dim 1).

    The value-MODIFYING write-back of the compressed (possibly merged) visual tokens into their
    kept sequence positions before the subselect. Duck-types over numpy and torch.
    """
    pos = [int(p) for p in positions]
    try:
        import torch  # lazy

        if isinstance(x, torch.Tensor):
            out = x.clone()
            idx = torch.as_tensor(pos, dtype=torch.long, device=x.device)
            vals = values if isinstance(values, torch.Tensor) else torch.as_tensor(values)
            out[:, idx, :] = vals.to(device=out.device, dtype=out.dtype)
            return out
    except ImportError:
        pass
    import numpy as np

    out = np.array(x, copy=True)
    out[:, pos, :] = np.asarray(values)
    return out


def pre_llm_subselect(
    *,
    inputs_embeds: Any,
    position_ids: Any,
    attention_mask: Any,
    cache_position: Any,
    visual_pos_masks: Any,
    deepstack_visual_embeds: Any,
    visual_positions: Sequence[int],
    marker_positions: Sequence[int],
    keep_visual_local: Sequence[int],
    compressed_tokens: Any,
) -> dict:
    """Write the compressed tokens back and drop everything that did not survive.

    ``visual_positions`` — absolute seq positions of the video placeholders, in scatter order
    (the m-th placeholder holds merged token m); length == pre_prune_tokens.
    ``marker_positions`` — the vision_start/vision_end token positions, DROPPED along with the
    pruned video tokens. Text positions survive. ``keep_visual_local`` — sorted local indices
    into ``[0, N)`` that survive; ``compressed_tokens`` — the ``len(keep_visual_local)``
    (value-merged) rows to write at the kept visual positions.

    Returns the subselected ``inputs_embeds`` / ``position_ids`` (index-gathered, position gaps
    intentionally kept) / ``attention_mask`` / ``cache_position`` / ``visual_pos_masks`` /
    ``deepstack_visual_embeds`` plus ``keep_global`` and ``post_prune_visual``.
    Sequence-shaped tensors gather the surviving GLOBAL positions; deepstack (one row per
    visual token) gathers the LOCAL visual indices.
    """
    seq_len = int(inputs_embeds.shape[1])
    vis = [int(p) for p in visual_positions]
    keep_local = [int(i) for i in keep_visual_local]
    keep_visual_abs = [vis[i] for i in keep_local]
    drop = set(vis) | {int(p) for p in marker_positions}
    non_visual = [p for p in range(seq_len) if p not in drop]
    keep_global = sorted(keep_visual_abs + non_visual)

    ie = scatter_seq_rows(inputs_embeds, keep_visual_abs, compressed_tokens)
    out: dict = {
        "inputs_embeds": take_dim(ie, keep_global, 1),
        "position_ids": None if position_ids is None else take_dim(position_ids, keep_global, -1),
        "attention_mask": None if attention_mask is None else take_dim(attention_mask, keep_global, -1),
        "cache_position": None if cache_position is None else take_dim(cache_position, keep_global, -1),
        "visual_pos_masks": None if visual_pos_masks is None else take_dim(visual_pos_masks, keep_global, -1),
        "deepstack_visual_embeds": (
            None if deepstack_visual_embeds is None
            else [subselect_rows(d, keep_local) for d in deepstack_visual_embeds]
        ),
        "keep_global": keep_global,
        "post_prune_visual": len(keep_visual_abs),
    }
    return out


def _run_pre_llm_compression(compression, video_features, captured, params, *,
                             model=None, tokenizer=None, question=None):
    """Dispatch to the pre_llm_compress kernel for ``compression``. Returns ``(compressed_tokens,
    keep_visual_local)`` — ``keep_visual_local`` sorted local indices into ``[0, N)``."""
    from . import pre_llm_compress as C

    nf, nt, _ = video_features.shape
    N = nf * nt
    retention = params.get("keep_ratio", 0.25)
    expansion = params.get("expansion", 1.0)
    if compression == "visionzip":
        if params.get("token_budget"):
            retention = min(1.0, float(params["token_budget"]) / (N * expansion))
        return C.visionzip_compression(
            video_features, captured["cls_attention"], captured["attn_keys"],
            retention_ratio=retention, expansion=expansion,
            alpha=params.get("alpha", 0.928571),
        )
    if compression == "mmtok":
        # Each kernel keeps its own rounding: MMTok and VisionZip round, FlashVID takes the
        # ceiling. The
        # pruner-level keep_count() (frontend/prune/random_prune.py) is the ceiling.
        target = params.get("token_budget") or max(1, int(round(N * retention * expansion)))
        target = min(int(target), N)
        text_emb = _mmtok_text_embedding(model, tokenizer, question)
        img = captured["img_features"].view(nf, nt, -1)
        return C.mmtok_compression(
            video_features, img, text_emb, target_vision_tokens=target,
            alpha=params.get("alpha", 0.5), tv_temp=params.get("tv_temp", 0.01),
            vv_temp=params.get("vv_temp", 0.2), greedy=params.get("greedy"),
        )
    if compression == "flashvid":
        # FlashVID budgets per frame from a ratio, so an absolute token budget is expressed
        # as the ratio that yields it.
        if params.get("token_budget"):
            retention = min(1.0, float(params["token_budget"]) / N)
        cfg = C.FlashVidVisionConfig(
            retention_ratio=retention, expansion=expansion,
            alpha=params.get("alpha", 0.7),
            do_segment=params.get("do_segment", True),
            segment_threshold=params.get("segment_threshold", 0.9),
            min_segment_num=params.get("min_segment_num", 8),
            complementary_segment=params.get("complementary_segment", True),
            temporal_threshold=params.get("temporal_threshold", 0.8),
        )
        return C.flashvid_compression(video_features, captured["cls_attention"], cfg)
    raise ValueError(f"unknown pre_llm compression {compression!r}")


def _mmtok_text_embedding(model, tokenizer, question):
    """The question's keywords embedded with the LLM's own input-embedding table, on device."""
    import torch

    from . import pre_llm_compress as C

    if tokenizer is None or model is None:
        raise RuntimeError(
            "pre_llm mmtok requires the tokenizer + model (LLM embed_tokens) to embed the "
            "question — the backend must pass them into applied_pre_llm_patch"
        )
    text = C.mmtok_extract_keywords(f"Question: {question or ''}")
    words = text.split() or [""]
    embed_tokens = model.get_input_embeddings()
    device = next(embed_tokens.parameters()).device
    enc = tokenizer(words, is_split_into_words=True, return_tensors="pt",
                    padding=True, truncation=True)
    input_ids = enc["input_ids"].to(device)
    with torch.no_grad():
        tok_emb = embed_tokens(input_ids)[0]
    start = 0
    bos = getattr(tokenizer, "bos_token_id", None)
    if input_ids.shape[1] > 1 and bos is not None and int(input_ids[0, 0]) == int(bos):
        start = 1
    return tok_emb[start:]


# Cap on the per-frame fp32 softmax intermediate the attention recompute may materialize
# (bytes). A correctly budgeted frame peaks well under 1 GB; a frame that reaches the tower
# without its pixel budget applied can ask for tens of GB and OOM-kill the run mid-way. 8 GiB
# sits far above the former and far below the latter, so the guard only fires on budget bugs.
VISION_SOFTMAX_GUARD_BYTES = 8 * 1024**3


def vision_softmax_bytes(num_heads: int, seq_per_frame: int) -> int:
    """fp32 QK^T-softmax intermediate for ONE frame: heads x seq^2 x 4 bytes. Pure."""
    return int(num_heads) * int(seq_per_frame) ** 2 * 4


def check_vision_softmax_budget(
    num_heads: int, seq_per_frame: int, limit_bytes: int = VISION_SOFTMAX_GUARD_BYTES
) -> None:
    """Fail LOUD before the recompute materializes a pathological fp32 softmax (OOM guard).

    Raises with a message naming the actual fix — the run's pixel budget — instead of letting a
    CUDA OOM kill the run. Deliberately does NOT chunk or downcast above the limit: silently
    changing numerics would change which tokens a method keeps.
    """
    need = vision_softmax_bytes(num_heads, seq_per_frame)
    if need > limit_bytes:
        raise RuntimeError(
            f"pre_llm signal capture: the attention recompute would materialize a "
            f"{need / 1024**3:.1f} GiB fp32 softmax intermediate for ONE frame "
            f"({num_heads} heads x {seq_per_frame}^2 keys > {limit_bytes / 1024**3:.0f} GiB "
            "guard). A frame this large only reaches the tower when no pixel budget was "
            "applied upstream. Fix the run's budget (Budget.resolution / frame count); do NOT "
            "raise this guard or chunk the softmax."
        )


def _recompute_vision_signals(attn_module, hidden_states, cu_seqlens, position_embeddings,
                              grid_thw):
    """Read-only recompute of the last vision block's mean self-attention.

    Returns ``(cls_attention[nf, nt], attn_keys[nf, nt, head_dim])``: per-frame
    ``softmax(QK^T/sqrt(d))`` averaged over heads and queries, and the mean-over-heads key
    vectors, both averaged over each 2x2 merge window so they line up with the merged tokens. Coupled to Qwen3-VL vision attention internals (``.qkv`` / ``.num_heads`` /
    ``.head_dim`` / ``apply_rotary_pos_emb_vision``); fails loud if that surface drifts.
    """
    import torch
    import torch.nn as nn

    try:
        from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb_vision
    except Exception as exc:  # pragma: no cover - import-time surface drift
        raise RuntimeError(
            "pre_llm signal capture: cannot import apply_rotary_pos_emb_vision — Qwen3-VL vision "
            "internals drifted; re-pin the tower recompute against the installed transformers"
        ) from exc
    for attr in ("qkv", "num_heads", "head_dim"):
        if not hasattr(attn_module, attr):
            raise AttributeError(
                f"pre_llm signal capture: vision attention module lacks {attr!r} — cannot "
                "recompute the attention signal (backbone API drift)"
            )
    seq_length = hidden_states.shape[0]
    q, k, _v = (
        attn_module.qkv(hidden_states)
        .reshape(seq_length, 3, attn_module.num_heads, -1)
        .permute(1, 0, 2, 3)
        .unbind(0)
    )
    cos, sin = position_embeddings
    q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
    num_frames = cu_seqlens.shape[0] - 1
    q, k = q.transpose(0, 1), k.transpose(0, 1)
    q = q.reshape(num_frames, -1, attn_module.num_heads, attn_module.head_dim).permute(0, 2, 1, 3).contiguous()
    k = k.reshape(num_frames, -1, attn_module.num_heads, attn_module.head_dim).permute(0, 2, 1, 3).contiguous()
    seq_per_frame = q.shape[2]
    check_vision_softmax_budget(attn_module.num_heads, seq_per_frame)
    attn_weights = torch.empty(num_frames, seq_per_frame, dtype=q.dtype, device=q.device)
    for f in range(num_frames):
        attn_f = torch.matmul(q[f], k[f].transpose(-1, -2)) / attn_module.head_dim ** 0.5
        attn_f = nn.functional.softmax(attn_f, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights[f] = attn_f.mean(0).mean(0)
    attn_keys = k.mean(dim=1)  # (num_frames, seq_per_frame, head_dim)
    # 2x2 spatial merge to match the patch merger.
    gt_num_frames = int(grid_thw[0][0])
    merged_seq = attn_weights.shape[-1] // 4
    cls_attention = attn_weights.view(gt_num_frames, merged_seq, -1).mean(-1)
    head_dim = attn_keys.shape[-1]
    attn_keys = attn_keys.view(gt_num_frames, merged_seq, 4, head_dim).mean(dim=2)
    return cls_attention, attn_keys


def _install_tower_capture(model, captured: dict, tower_target: str, grid_thw,
                           signals: Sequence[str] = ("cls_attention", "img_features")) -> list:
    """Install read-only hooks capturing the tower ``signals`` a kernel needs. Returns the
    restore list.

    ``cls_attention`` / ``attn_keys`` wrap the last vision block's attention and re-run its
    QK^T softmax;
    ``img_features`` wraps the merger and keeps its (pre-merger) input. Only the requested
    hooks are installed, so a method that needs no attention does not pay for the recompute.
    Fails loud if the tower lacks the expected ``blocks`` / ``merger`` structure.
    """
    tower = resolve_module(model, tower_target)
    if not hasattr(tower, "blocks") or not hasattr(tower, "merger"):
        raise AttributeError(
            f"pre_llm signal capture: tower {tower_target!r} lacks blocks/merger — cannot install "
            "the read-only signal hooks (backbone API drift)"
        )
    restore = []

    if "cls_attention" in signals or "attn_keys" in signals:
        last_block = tower.blocks[-1]
        if not hasattr(last_block, "attn"):
            raise AttributeError("pre_llm signal capture: last vision block lacks `attn`")
        attn_mod = last_block.attn
        orig_attn = attn_mod.forward

        def attn_wrap(*args, **kwargs):
            out = orig_attn(*args, **kwargs)
            hidden_states = args[0] if args else kwargs.get("hidden_states")
            cu_seqlens = kwargs.get("cu_seqlens", args[1] if len(args) > 1 else None)
            position_embeddings = kwargs.get("position_embeddings")
            if hidden_states is not None and cu_seqlens is not None and position_embeddings is not None:
                cls_attention, attn_keys = _recompute_vision_signals(
                    attn_mod, hidden_states, cu_seqlens, position_embeddings, grid_thw
                )
                captured["cls_attention"] = cls_attention
                if "attn_keys" in signals:
                    captured["attn_keys"] = attn_keys
            return out

        attn_mod.forward = attn_wrap
        restore.append((attn_mod, orig_attn))

    if "img_features" in signals:
        merger = tower.merger
        orig_merger = merger.forward

        def merger_wrap(*args, **kwargs):
            pre = args[0] if args else kwargs.get("x", kwargs.get("hidden_states"))
            if pre is not None:
                captured["img_features"] = pre.view(-1, 4, pre.shape[-1]).mean(dim=1)
            return orig_merger(*args, **kwargs)

        merger.forward = merger_wrap
        restore.append((merger, orig_merger))
    return restore


@contextmanager
def applied_pre_llm_patch(
    model: Any, spec: PatchSpec, visual_positions: Sequence[int],
    marker_positions: Sequence[int], prefill_len: int, *, grid_thw=None,
    question: Optional[str] = None, tokenizer: Any = None,
    lm_target: str = "model.language_model", tower_target: str = "model.visual",
    arm_out: Optional[dict] = None, compress_fn: Optional[Callable] = None,
    install_capture: Optional[Callable] = None,
):
    """Install the pre_llm compression for one generate.

    Wraps the vision tower (read-only signal capture) and the language model's ``forward`` (the
    boundary that receives ``inputs_embeds`` post-scatter and ``position_ids`` already
    computed, plus ``attention_mask`` / ``cache_position`` / ``visual_pos_masks`` /
    ``deepstack_visual_embeds``). On the PREFILL pass it reads the merged video tokens back
    from ``inputs_embeds`` at the placeholder positions, runs the method's compression, writes
    the compressed tokens back and subselects every sequence-shaped tensor
    (:func:`pre_llm_subselect`). Decode steps pass through untouched. All wrappers are restored
    in ``finally``.

    ``compress_fn(video_features, captured) -> (compressed_tokens, keep_visual_local)`` and
    ``install_capture() -> restore_list`` are injectable so the wrap lifecycle + subselect
    unit-test on fakes; None => the real kernels + tower hooks.
    """
    if spec.patch_point != "pre_llm":
        raise ValueError(f"applied_pre_llm_patch requires patch_point=pre_llm, got {spec.patch_point!r}")
    if len(visual_positions) != spec.pre_prune_tokens:
        raise ValueError(
            f"prune seam (pre_llm): {len(visual_positions)} visual placeholder positions but "
            f"spec.pre_prune_tokens={spec.pre_prune_tokens} — placeholder/count drift"
        )
    vis = [int(p) for p in visual_positions]
    n_vis = len(vis)
    captured: dict = {}
    signals = PRE_LLM_SIGNALS.get(spec.compression, ())

    if compress_fn is None:
        def compress_fn(video_features, captured):  # noqa: ANN001 - injectable closure
            if spec.compression in PRE_LLM_STATIC:
                return _pre_llm_static_select(video_features, spec.keep_indices)
            return _run_pre_llm_compression(
                spec.compression, video_features, captured, spec.params,
                model=model, tokenizer=tokenizer, question=question,
            )

    lm = resolve_module(model, lm_target)
    orig_lm = lm.forward
    restore: list = []

    def lm_wrap(*args, **kwargs):
        inputs_embeds = kwargs.get("inputs_embeds")
        if inputs_embeds is None or int(inputs_embeds.shape[1]) != prefill_len:
            return orig_lm(*args, **kwargs)  # decode step (or no embeds): untouched
        if grid_thw is not None:
            nf = int(grid_thw[0][0])
        elif "cls_attention" in captured:
            nf = int(captured["cls_attention"].shape[0])
        else:
            nf = 1  # static keeps are granularity-agnostic (flat indices)
        nt = n_vis // nf
        vf = _read_video_features(inputs_embeds, vis, nf, nt)
        compressed, keep_local = compress_fn(vf, captured)
        # Keep the kernel's (compressed[j] <-> keep_local[j]) pairing: every kernel returns
        # them sorted-and-aligned by global index; do NOT re-sort keep_local alone.
        keep_local = _int_list(keep_local)
        sub = pre_llm_subselect(
            inputs_embeds=inputs_embeds, position_ids=kwargs.get("position_ids"),
            attention_mask=kwargs.get("attention_mask"), cache_position=kwargs.get("cache_position"),
            visual_pos_masks=kwargs.get("visual_pos_masks"),
            deepstack_visual_embeds=kwargs.get("deepstack_visual_embeds"),
            visual_positions=vis, marker_positions=marker_positions,
            keep_visual_local=keep_local, compressed_tokens=compressed,
        )
        new_kwargs = dict(kwargs)
        new_kwargs["inputs_embeds"] = sub["inputs_embeds"]
        for key in ("position_ids", "attention_mask", "cache_position", "visual_pos_masks",
                    "deepstack_visual_embeds"):
            if key in new_kwargs:
                new_kwargs[key] = sub[key]
        if arm_out is not None:
            arm_out["pre_prune_visual"] = n_vis
            arm_out["post_prune_visual"] = sub["post_prune_visual"]
        return orig_lm(*args, **new_kwargs)

    try:
        if install_capture is not None:
            capture_restore = install_capture()
        elif signals:
            capture_restore = _install_tower_capture(
                model, captured, tower_target, grid_thw, signals=signals
            )
        else:
            capture_restore = []  # static control: no tower signal needed
        restore.extend(capture_restore or [])
        restore.append((lm, orig_lm))
        lm.forward = lm_wrap
        yield
    finally:
        for mod, fwd in restore:
            mod.forward = fwd


def _read_video_features(inputs_embeds, visual_positions, nf, nt):
    """Read the merged video tokens back from ``inputs_embeds`` at the placeholder positions
    -> ``(nf, nt, D)``."""
    try:
        import torch  # lazy

        if isinstance(inputs_embeds, torch.Tensor):
            idx = torch.as_tensor([int(p) for p in visual_positions], dtype=torch.long,
                                  device=inputs_embeds.device)
            return inputs_embeds[0].index_select(0, idx).view(nf, nt, -1)
    except ImportError:
        pass
    import numpy as np

    rows = np.asarray(inputs_embeds)[0][[int(p) for p in visual_positions]]
    return rows.reshape(nf, nt, -1)


def _pre_llm_static_select(video_features, keep_indices):
    """Static control: return ``(flat_features[keep], keep)`` — pure selection, VALUE UNCHANGED."""
    keep = [int(i) for i in keep_indices]
    d = int(video_features.shape[-1])
    try:
        import torch  # lazy

        if isinstance(video_features, torch.Tensor):
            flat = video_features.reshape(-1, d)
            idx = torch.as_tensor(keep, dtype=torch.long, device=flat.device)
            return flat.index_select(0, idx), keep
    except ImportError:
        pass
    import numpy as np

    flat = np.asarray(video_features).reshape(-1, d)
    return flat[keep], keep


def _int_list(x) -> list:
    """Python ints from a torch/numpy/list index vector, ORDER PRESERVED (kernel pairing)."""
    try:
        import torch  # lazy

        if isinstance(x, torch.Tensor):
            return [int(v) for v in x.detach().cpu().tolist()]
    except ImportError:
        pass
    return [int(v) for v in x]
