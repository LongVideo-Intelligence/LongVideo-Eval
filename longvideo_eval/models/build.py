"""Method assembly — a (model, method) pair -> a wired `Orchestrator`.

A "method" here is a named binding of stage components (decoder, selector, encoder, pruner,
backend). This is the single place the registry ids for a method are declared, so a method's
code location and its stage ids line up 1:1 with the folder layout.

`--dry-run` substitutes the hardware-free fakes (testing.py) for the decode/encode/backend
stages so the whole pipeline runs in CI without weights or real decode; selector + pruner are
the real registered components in both modes (they are hardware-free already).

Fail-loud: an unknown model or method raises with the valid options listed.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Optional

from .. import _registry
from ..config import BASE_MODELS
from ..interfaces import Orchestrator
from ..orchestration.single_shot import SingleShotOrchestrator
from ..testing import FakeBackend, FakeClipScorer, FakeDecoder, FakeEncoder

# Import stage modules for their registration side effects (register under stable ids).
from ..frontend.decode import decoder as _decoder  # noqa: F401  registers decode:decord
from ..frontend.select import uniform as _uniform  # noqa: F401  select:uniform, lohi-uniform
from ..frontend.select import semdiv as _semdiv  # noqa: F401  select:semdiv, lohi-semdiv
from ..frontend.select import cliptopk as _cliptopk  # noqa: F401  registers select:cliptopk
from ..frontend.select import aks as _aks  # noqa: F401  registers select:aks
from ..frontend.encode import encoder as _encoder  # noqa: F401  registers encode:vit
from ..frontend.prune import identity as _identity  # noqa: F401  registers prune:identity
from ..frontend.prune import random_prune as _random_prune  # noqa: F401  registers prune:random
from ..frontend.prune import mmtok as _mmtok  # noqa: F401  registers prune:mmtok
from ..frontend.prune import flashvid as _flashvid  # noqa: F401  registers prune:flashvid
from ..frontend.prune import visionzip as _visionzip  # noqa: F401  registers prune:visionzip
from ..backend.hf import qwen as _qwen  # noqa: F401  registers backend:qwen-hf


@dataclass(frozen=True)
class ComponentIds:
    """The registry ids each stage of a method resolves to."""

    decode: str
    select: str
    encode: str
    prune: str
    backend: str
    orchestrate: str = "single_shot"


METHODS = {
    # Native Qwen processing: uniform frames, no token pruning, HF backend.
    "qwen-default": ComponentIds(
        decode="decord", select="uniform", encode="vit", prune="identity", backend="qwen-hf",
    ),
    # Low-Res-Base, the project's default baseline: every decoded frame, at a low resolution.
    # Default setup lowres_base_256f = 256 decoded frames at a quarter of the native size.
    # The decoded-frame count is the setting: lowres_base_64f / 128f / 256f / 512f.
    "lowres-base": ComponentIds(
        decode="decord", select="uniform", encode="vit", prune="identity", backend="qwen-hf",
    ),
    # --- token pruning: dense frames in, a fraction of the visual tokens into the LLM ---
    # The control: a seeded uniform draw at the same prune point as the learned pruners.
    "random-prune": ComponentIds(
        decode="decord", select="uniform", encode="vit", prune="random", backend="qwen-hf",
    ),
    "mmtok": ComponentIds(
        decode="decord", select="uniform", encode="vit", prune="mmtok", backend="qwen-hf",
    ),
    "flashvid": ComponentIds(
        decode="decord", select="uniform", encode="vit", prune="flashvid", backend="qwen-hf",
    ),
    "visionzip": ComponentIds(
        decode="decord", select="uniform", encode="vit", prune="visionzip", backend="qwen-hf",
    ),
    # --- keyframe selection: score a large decoded pool, keep a few frames (keyframe_16f_pool256) ---
    "cliptopk": ComponentIds(
        decode="decord", select="cliptopk", encode="vit", prune="identity", backend="qwen-hf",
    ),
    "aks": ComponentIds(
        decode="decord", select="aks", encode="vit", prune="identity", backend="qwen-hf",
    ),
    # --- dual stream (LoHi): a low-res video of every frame plus K of them as images ---
    # Run under a `lohi*` setup. On Qwen, two video frames merge into one temporal position
    # and an image does not, so the image stream is scaled down, per video, to keep the total
    # visual tokens equal to the 16-frame reference (K=8: about 0.7 of the native size).
    # K high-resolution frames at regular intervals (no selection cost).
    "lohi-uniform": ComponentIds(
        decode="decord", select="lohi-uniform", encode="vit", prune="identity", backend="qwen-hf",
    ),
    # K high-resolution frames by query relevance x visual diversity (one CLIP pass).
    "lohi-semdiv": ComponentIds(
        decode="decord", select="lohi-semdiv", encode="vit", prune="identity", backend="qwen-hf",
    ),
}


def build_pipeline(
    model_id: str, method_id: str, dry_run: bool = False, enable_thinking: bool = False,
    prune_kwargs: Optional[dict] = None, max_new_tokens: Optional[int] = None,
) -> Orchestrator:
    """Resolve (model, method) into a single-shot orchestrator with its stages bound.

    ``enable_thinking`` selects the backend's thinking arm: thinking is an explicit on/off
    evaluation dimension, never a silent chat-template default. The arm is validated at
    backend construction against ``config.ModelSpec.thinking``: "none" rejects True, "always"
    rejects False, "hybrid" allows both. Ignored under ``dry_run`` (the fakes have no thinking
    span).

    ``prune_kwargs`` are forwarded to the pruner constructor.

    ``max_new_tokens``, when given, overrides the backend's generation cap (``None`` keeps its
    own default). Thinking arms need headroom beyond a bare MCQ answer for the ``<think>`` span;
    ignored under ``dry_run`` (the fake backend has no generation cap to set).
    """
    if model_id not in BASE_MODELS:
        raise ValueError(
            f"unknown model {model_id!r}; valid models: {sorted(BASE_MODELS)}"
        )
    if method_id not in METHODS:
        raise ValueError(
            f"unknown method {method_id!r}; valid methods: {sorted(METHODS)}"
        )

    ids = METHODS[method_id]
    selector_cls = _registry.get("select", ids.select)
    # A dry run must not load CLIP: selectors that score frames take an injectable scorer.
    selector_params = inspect.signature(selector_cls).parameters
    if dry_run and "scorer_factory" in selector_params:
        selector = selector_cls(scorer_factory=FakeClipScorer)
    elif dry_run and "scorer" in selector_params:
        selector = selector_cls(scorer=FakeClipScorer())
    else:
        selector = selector_cls()
    pruner = _registry.get("prune", ids.prune)(**(prune_kwargs or {}))

    if dry_run:
        decoder = FakeDecoder()
        encoder = FakeEncoder()
        backend = FakeBackend()
    else:
        from ..frontend.encode.encoder import vision_constants
        decoder = _registry.get("decode", ids.decode)(
            factor=vision_constants(model_id).factor
        )
        encoder = _registry.get("encode", ids.encode)(model_id=model_id)
        backend_kwargs = {"model_id": model_id, "enable_thinking": enable_thinking}
        if max_new_tokens is not None:
            backend_kwargs["max_new_tokens"] = max_new_tokens
        backend = _registry.get("backend", ids.backend)(**backend_kwargs)

    return SingleShotOrchestrator(decoder, selector, encoder, pruner, backend)
