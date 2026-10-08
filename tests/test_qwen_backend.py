"""Hardware-free tests for the HF Qwen backend — the torch-free surfaces only.

The backend's generate() imports torch lazily and needs weights, so it cannot run end-to-end
in CI. What IS testable without torch: constructor validation of the thinking arm, the seam
contract (VisualTokens.tokens must come from ViTEncoder) which is checked BEFORE torch is
imported, and the pure thinking-accounting helper split_thinking(). The full generate path is
covered by scripts/smoke_gpu.py on a GPU machine.
"""
from __future__ import annotations

import pytest

from longvideo_eval.backend.hf.qwen import (
    QwenHFBackend,
    split_thinking,
    template_thinking_kwargs,
    validate_thinking_arm,
)
from longvideo_eval.frontend.encode.encoder import QWEN_TOKENS_KIND
from longvideo_eval.interfaces import Budget, VisualTokens


# --------------------------------------------------------------------------- #
# thinking arm — validated against config.ModelSpec.thinking (three-way matrix)
# --------------------------------------------------------------------------- #
def test_thinking_matrix_none_mode():
    # "none" (qwen3-vl-4b Instruct): thinking=False ok; thinking=True rejected loudly,
    # pointing at the Thinking checkpoint.
    assert QwenHFBackend(model_id="qwen3-vl-4b").thinking_mode == "none"
    with pytest.raises(ValueError, match="no.*thinking mode.*qwen3-vl-4b-thinking"):
        QwenHFBackend(model_id="qwen3-vl-4b", enable_thinking=True)


def test_thinking_matrix_hybrid_mode():
    # "hybrid" (qwen3.5-4b): both arms valid; default is off.
    on = QwenHFBackend(model_id="qwen3.5-4b", enable_thinking=True)
    off = QwenHFBackend(model_id="qwen3.5-4b")
    assert on.enable_thinking is True and on.thinking_mode == "hybrid"
    assert off.enable_thinking is False and off.thinking_mode == "hybrid"


def test_thinking_matrix_always_mode():
    # "always" (qwen3-vl-4b-thinking): cannot not-think; only thinking=True constructs.
    be = QwenHFBackend(model_id="qwen3-vl-4b-thinking", enable_thinking=True)
    assert be.thinking_mode == "always"
    with pytest.raises(ValueError, match="cannot not-think"):
        QwenHFBackend(model_id="qwen3-vl-4b-thinking", enable_thinking=False)
    with pytest.raises(ValueError, match="cannot not-think"):
        QwenHFBackend(model_id="qwen3-vl-4b-thinking")  # default False is also invalid


def test_validate_thinking_arm_unknown_model():
    with pytest.raises(KeyError, match="unknown model"):
        validate_thinking_arm("qwen99-vl", enable_thinking=False)


def test_build_pipeline_propagates_thinking_validation():
    # The matrix fires at BUILD time (assembly constructs the backend, no weights loaded).
    from longvideo_eval.models.build import build_pipeline

    orch = build_pipeline("qwen3-vl-4b-thinking", "lowres-base", enable_thinking=True)
    assert orch.backend.thinking_mode == "always"
    with pytest.raises(ValueError, match="cannot not-think"):
        build_pipeline("qwen3-vl-4b-thinking", "lowres-base", enable_thinking=False)


# --------------------------------------------------------------------------- #
# template kwargs — hybrid/none pass an explicit boolean; always passes none
# --------------------------------------------------------------------------- #
def test_template_kwargs_explicit_for_hybrid_and_none():
    assert template_thinking_kwargs("hybrid", True) == {"enable_thinking": True}
    assert template_thinking_kwargs("hybrid", False) == {"enable_thinking": False}
    assert template_thinking_kwargs("none", False) == {"enable_thinking": False}


def test_template_kwargs_empty_for_always():
    # The always-thinking template defines no enable_thinking input — pass nothing.
    assert template_thinking_kwargs("always", True) == {}


# --------------------------------------------------------------------------- #
# seam contract — validated before any torch import
# --------------------------------------------------------------------------- #
def test_generate_rejects_foreign_tokens():
    be = QwenHFBackend(model_id="qwen3-vl-4b")
    bad = VisualTokens(tokens={"kind": "not-ours"}, num_tokens=0)
    with pytest.raises(TypeError, match="ViTEncoder"):
        be.generate(bad, "q", Budget())


def test_generate_rejects_model_mismatch():
    be = QwenHFBackend(model_id="qwen3-vl-4b")
    mismatched = VisualTokens(
        tokens={"kind": QWEN_TOKENS_KIND, "model_id": "qwen3.5-4b"}, num_tokens=0
    )
    with pytest.raises(ValueError, match="build.py must wire the same model"):
        be.generate(mismatched, "q", Budget())


# --------------------------------------------------------------------------- #
# split_thinking — the metering rule
# --------------------------------------------------------------------------- #
def _wc(s):
    return s.split()  # length == whitespace token count (a stand-in tokenizer)


def test_split_thinking_on_with_block():
    ans, dec, think = split_thinking(
        "<think> a b c </think> Answer A", total_gen_tokens=10, enable_thinking=True, encode_fn=_wc
    )
    assert ans == "Answer A"
    assert dec == 2                      # "Answer A" -> 2 tokens
    assert think == 8                    # remaining generated tokens attributed to thinking


def test_split_thinking_off_keeps_all_as_answer():
    ans, dec, think = split_thinking(
        "Answer A", total_gen_tokens=2, enable_thinking=False, encode_fn=_wc
    )
    assert ans == "Answer A" and dec == 2 and think == 0


def test_split_thinking_on_but_unclosed_keeps_answer():
    # Malformed generation (no </think>): never drop the answer; charge nothing to thinking.
    ans, dec, think = split_thinking(
        "<think> reasoning that never closed", total_gen_tokens=5, enable_thinking=True, encode_fn=_wc
    )
    assert ans == "<think> reasoning that never closed"
    assert dec == 5 and think == 0
