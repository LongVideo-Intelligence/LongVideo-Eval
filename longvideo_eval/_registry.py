"""Tiny runtime component registry.

Stage components (Selector/Pruner/Decoder/Encoder/LLMBackend/Orchestrator) self-register
under a stable string id. The runner looks components up by (kind, id).

    @register("prune", "identity")
    class IdentityPruner(Pruner): ...

    get("prune", "identity")   -> IdentityPruner
"""
from __future__ import annotations

from typing import Callable, Dict, Tuple, Type

_REGISTRY: Dict[Tuple[str, str], Type] = {}

_KINDS = {"decode", "select", "encode", "prune", "backend", "orchestrate"}


def register(kind: str, id: str) -> Callable[[Type], Type]:
    if kind not in _KINDS:
        raise ValueError(f"unknown component kind {kind!r}; expected one of {sorted(_KINDS)}")

    def deco(cls: Type) -> Type:
        key = (kind, id)
        if key in _REGISTRY and _REGISTRY[key] is not cls:
            raise ValueError(f"duplicate component id: {kind}:{id} already registered")
        cls.id = id
        _REGISTRY[key] = cls
        return cls

    return deco


def get(kind: str, id: str) -> Type:
    try:
        return _REGISTRY[(kind, id)]
    except KeyError:
        avail = sorted(i for (k, i) in _REGISTRY if k == kind)
        raise KeyError(f"no {kind} component {id!r}; registered: {avail}") from None


def available(kind: str) -> list:
    return sorted(i for (k, i) in _REGISTRY if k == kind)
