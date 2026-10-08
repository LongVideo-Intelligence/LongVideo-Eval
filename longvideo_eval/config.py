"""Declared base models and datasets the harness targets.

Model repo ids are exact Hugging Face paths.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class ModelSpec:
    id: str
    hf_repo: str          # HuggingFace repo id
    backend: str          # "hf"
    family: str = "qwen"
    # Thinking capability. Thinking is an explicit experiment arm, metered via
    # CostRecord.thinking_tokens, never a chat-template default:
    #   "none"   — no thinking mode; enable_thinking=True must raise loudly
    #   "hybrid" — same checkpoint toggles via enable_thinking (both arms runnable)
    #   "always" — the checkpoint always thinks; enable_thinking=False must raise loudly
    thinking: str = "none"


@dataclass(frozen=True)
class DatasetSpec:
    id: str
    bibkey: str           # citation key of the benchmark paper
    task_type: str        # "mcq"
    loader: str           # dotted module:Class
    has_subtitles: bool = False


BASE_MODELS = {
    # Size-specific ids are the CLI-facing handles (`--model qwen3-vl-4b`).
    "qwen3-vl-4b": ModelSpec("qwen3-vl-4b", "Qwen/Qwen3-VL-4B-Instruct", "hf", thinking="none"),
    "qwen3.5-4b": ModelSpec("qwen3.5-4b", "Qwen/Qwen3.5-4B", "hf", thinking="hybrid"),
    # The always-thinking Qwen3-VL checkpoint, the thinking-arm counterpart to qwen3-vl-4b.
    "qwen3-vl-4b-thinking": ModelSpec(
        "qwen3-vl-4b-thinking", "Qwen/Qwen3-VL-4B-Thinking", "hf", thinking="always"
    ),
    "qwen3-vl-8b": ModelSpec("qwen3-vl-8b", "Qwen/Qwen3-VL-8B-Instruct", "hf", thinking="none"),
    # Plain Qwen2.5-VL-7B. Its vision geometry is patch 14 * merge 2 (factor 28), not the
    # qwen3 family's 32, so the decoder snaps resized frames to 28 for this model
    # (build_pipeline passes the factor). The Instruct checkpoint has no thinking arm.
    "qwen2.5-vl-7b": ModelSpec(
        "qwen2.5-vl-7b", "Qwen/Qwen2.5-VL-7B-Instruct", "hf", thinking="none"
    ),
}

DATASETS = {
    "videomme": DatasetSpec(
        id="videomme",
        bibkey="fu2025videomme",
        task_type="mcq",
        loader="longvideo_eval.data.videomme:VideoMME",
        has_subtitles=True,   # VideoMME ships subtitles; feed to method AND baseline (parity)
    ),
}

# Pruners registered in the runtime component registry (frontend/prune/).
PRUNERS: Tuple[str, ...] = ("identity",)
