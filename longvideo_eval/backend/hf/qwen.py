"""Qwen3-VL / Qwen3.5 / Qwen2.5-VL backend via HuggingFace transformers.

The vision tower runs inside ``model.generate`` here (not in the encode stage, which only
preprocesses — see ``frontend/encode/encoder.py``). The backend consumes the encoder's
``VisualTokens.tokens`` package (kind ``qwen-hf-video``), adds the prompt, and runs the full
HF forward.

Generation is GREEDY in both thinking arms (temperature/top_p unset, do_sample=False,
num_beams=1) — explicit, never a template default (an unset temperature falls back to the
model's own sampling default).

THINKING is an explicit evaluation dimension, not a hardcoded default. Each model's valid arms
are declared by ``config.ModelSpec.thinking`` ("none" | "hybrid" | "always") and validated at
construction — never by hardcoded model ids:
  * ``"none"`` (Qwen3-VL-4B-Instruct): no thinking mode; ``enable_thinking=True`` is rejected
    loudly, naming the Qwen3-VL Thinking checkpoint as the alternative.
  * ``"hybrid"`` (Qwen3.5-4B): both arms valid; the chat template ALWAYS receives an explicit
    boolean (both ways: Qwen3.5's template force-appends ``<think>`` if not told otherwise).
  * ``"always"`` (Qwen3-VL-4B-Thinking): the checkpoint cannot not-think, so
    ``enable_thinking=False`` is rejected loudly; its template unconditionally opens the think
    span and takes NO ``enable_thinking`` kwarg, so none is passed (see
    ``template_thinking_kwargs``).
  * When thinking is ON, the ``<think>...</think>`` span is metered into
    ``CostRecord.thinking_tokens`` (separate from ``decode_tokens`` = post-think answer tokens,
    so thinking cost lands on the same axis) and stripped from ``Answer.text`` so MCQ
    extraction sees only the answer — identical metering for "hybrid" and "always".

Qwen3.5-4B processor overrides (per-frame min=200704 / max=1605632) are applied by family
(``frontend/encode/encoder.py::vision_constants``) in BOTH arms; without them the token count
blows up. On transformers 5.x they reach the processor as the per-call
``size={"shortest_edge","longest_edge"}`` area budget the encoder packages in
``VisualTokens.tokens["size"]`` — the 4.x ``min_pixels``/``max_pixels`` kwargs no longer exist.

torch / transformers imported LAZILY: the package must import on a GPU-free machine.
"""
from __future__ import annotations

from typing import Optional

from ..._registry import register
from ...config import BASE_MODELS
from ...frontend.encode.encoder import (
    QWEN_TOKENS_KIND,
    qwen_family,
    require_transformers_5,
    to_processor_video,
    vision_constants,
)
from ...interfaces import Answer, Budget, CostRecord, LLMBackend, VisualTokens
from .prune_seam import PATCH_SPEC_KEY, applied_pre_llm_patch

# MCQ answers are short (a letter, optionally a brief justification); a small cap keeps decode
# cheap. 32 is a safe default (the thinking arm needs headroom for the <think> span; raise per
# run if truncated).
_DEFAULT_MAX_NEW_TOKENS = 32

# The ModelSpec.thinking enum this backend understands; a spec outside it is a config drift
# bug and fails loudly at construction rather than silently mis-arming a run.
_THINKING_MODES = ("none", "hybrid", "always")


def validate_thinking_arm(model_id: str, enable_thinking: bool) -> str:
    """Check ``enable_thinking`` against ``BASE_MODELS[model_id].thinking``; return the mode.

    Pure config lookup (no torch/transformers) so the arm matrix unit-tests hardware-free and
    the error fires at BUILD time, not mid-run. See module docstring for the
    per-mode semantics.
    """
    if model_id not in BASE_MODELS:
        raise KeyError(f"unknown model {model_id!r}; valid: {sorted(BASE_MODELS)}")
    mode = BASE_MODELS[model_id].thinking
    if mode not in _THINKING_MODES:
        raise ValueError(
            f"ModelSpec.thinking={mode!r} for {model_id!r} is not one of {_THINKING_MODES}; "
            "config.py and backend/hf/qwen.py are out of sync"
        )
    if mode == "none" and enable_thinking:
        raise ValueError(
            f"enable_thinking=True is invalid for {model_id!r}: this checkpoint has no "
            "thinking mode. Use the Qwen3-VL Thinking checkpoint (qwen3-vl-4b-thinking), "
            "or Qwen3.5-4B (hybrid)."
        )
    if mode == "always" and not enable_thinking:
        raise ValueError(
            f"enable_thinking=False is invalid for {model_id!r}: an always-thinking checkpoint "
            "cannot not-think. Run it with enable_thinking=True, or use qwen3-vl-4b (none) / "
            "qwen3.5-4b (hybrid) for the no-thinking arm."
        )
    return mode


def template_thinking_kwargs(mode: str, enable_thinking: bool) -> dict:
    """chat-template kwargs for a validated (mode, arm) pair.

    "hybrid": an EXPLICIT boolean, both ways (Qwen3.5's template force-appends ``<think>`` if
    unset). "none": explicit False (the Instruct template treats it as an
    unused var; harmless, and keeps the no-default discipline uniform). "always": NO kwarg —
    the Thinking checkpoint's template unconditionally opens the think span and defines no
    ``enable_thinking`` input; passing one would at best be ignored and at worst break on a
    template that rejects unknown kwargs, so the checkpoint's one valid arm gets exactly the
    template it ships with.
    """
    if mode == "always":
        return {}
    return {"enable_thinking": enable_thinking}


def split_thinking(full_text: str, total_gen_tokens: int, enable_thinking: bool, encode_fn):
    """Split a ``<think>...</think>`` span off the answer and account its cost. Pure +
    torch-free so it unit-tests without weights.

    Returns ``(answer_text, decode_tokens, thinking_tokens)`` where ``decode_tokens`` counts
    only the post-think answer and ``thinking_tokens`` absorbs the rest (the ``<think>`` markers
    + reasoning), so ``prefill + decode + thinking`` sums to the real generated cost.
    Thinking OFF, or a missing closing tag, keeps the whole span as the
    answer (never drop the answer on a malformed generation).

    ``encode_fn(str) -> Sequence`` is the tokenizer's no-special-tokens encode; only its length
    is used, to count the answer tokens exactly rather than by character heuristics.
    """
    if enable_thinking and "</think>" in full_text:
        answer_text = full_text.split("</think>", 1)[1].strip()
        decode_tokens = len(encode_fn(answer_text))
        thinking_tokens = max(0, total_gen_tokens - decode_tokens)
        return answer_text, decode_tokens, thinking_tokens
    return full_text, total_gen_tokens, 0


@register("backend", "qwen-hf")
class QwenHFBackend(LLMBackend):
    """Loads a Qwen VL model from ``config.BASE_MODELS`` and runs prefill + decode.

    ``subtitles``, when present, are prepended to the prompt (the parity channel) and fed
    identically to the baseline.
    """

    def __init__(
        self,
        model_id: str = "qwen3-vl-4b",
        enable_thinking: bool = False,
        max_new_tokens: int = _DEFAULT_MAX_NEW_TOKENS,
        system_prompt: str = "You are a helpful assistant.",
    ) -> None:
        self.model_id = model_id
        self.enable_thinking = enable_thinking
        self.max_new_tokens = max_new_tokens
        self.system_prompt = system_prompt
        self._model = None      # lazy
        self._processor = None  # lazy
        # Fail loud at construction: the arm must be valid for the spec'd thinking mode.
        self.thinking_mode = validate_thinking_arm(model_id, enable_thinking)

    # ---------------------------------------------------------------- model loading
    def _load(self):
        """Lazily load (model, processor). The model class is chosen from
        ``AutoConfig.model_type``, NOT the repo name, so Qwen3-VL-4B-Thinking — which shares
        model_type="qwen3_vl" with Instruct (same architecture, different post-training) —
        lands on Qwen3VLForConditionalGeneration via the else-branch."""
        if self._model is None:
            from transformers import AutoConfig, AutoProcessor  # lazy

            repo = BASE_MODELS[self.model_id].hf_repo  # id validated at construction

            cfg = AutoConfig.from_pretrained(repo, trust_remote_code=True)
            model_type = getattr(cfg, "model_type", "")
            if "qwen3_5" in model_type:
                from transformers import Qwen3_5ForConditionalGeneration

                model_cls, dtype_key = Qwen3_5ForConditionalGeneration, "torch_dtype"
            elif "qwen2_5_vl" in model_type:
                # Route to the Qwen2.5-VL class, NOT the else-branch Qwen3VL (which would
                # silently mis-load a 2.5 checkpoint).
                from transformers import Qwen2_5_VLForConditionalGeneration

                model_cls, dtype_key = Qwen2_5_VLForConditionalGeneration, "dtype"
            else:
                from transformers import Qwen3VLForConditionalGeneration

                model_cls, dtype_key = Qwen3VLForConditionalGeneration, "dtype"

            self._model = model_cls.from_pretrained(
                repo, **{dtype_key: "bfloat16", "device_map": "auto"}
            )
            self._model.eval()
            # No pixel kwargs: 5.x from_pretrained silently ignores min/max_pixels; the budget
            # arrives per call via the encoder's `size` package (never a loadable default).
            self._processor = AutoProcessor.from_pretrained(repo)
            require_transformers_5(self._processor)  # loud gate
        return self._model, self._processor

    # ---------------------------------------------------------------- generation
    def generate(self, tokens: VisualTokens, prompt: str, budget: Budget,
                 subtitles: Optional[str] = None,
                 max_new_tokens: Optional[int] = None) -> Answer:
        """See ``LLMBackend``. ``max_new_tokens`` is a per-call cap override; None keeps the
        backend's configured cap."""
        pkg = tokens.tokens
        if not isinstance(pkg, dict) or pkg.get("kind") != QWEN_TOKENS_KIND:
            raise TypeError(
                "QwenHFBackend requires VisualTokens.tokens produced by ViTEncoder "
                f"(kind={QWEN_TOKENS_KIND!r}); got {type(pkg).__name__}"
            )
        if pkg["model_id"] != self.model_id:
            raise ValueError(
                f"encoder packaged model {pkg['model_id']!r} but backend is {self.model_id!r}; "
                "build.py must wire the same model into both stages"
            )

        import torch  # lazy

        model, proc = self._load()
        family = qwen_family(self.model_id)
        c = vision_constants(self.model_id)

        # Parity channel: subtitles prepended, fed identically to method AND baseline.
        text_body = f"{subtitles}\n\n{prompt}" if subtitles else prompt

        frames = pkg["frames"]
        # A Pruner declares its patch in the package; absent => the dense path, untouched.
        spec = pkg.get(PATCH_SPEC_KEY)
        # Dual stream: one video block (Lo-V) + K timestamped image blocks (Hi-I).
        lohi = frames is not None and budget.presentation == "lohi"
        lohi_visual_tokens = 0
        # Per-mode chat-template kwargs — hybrid gets an explicit boolean, always gets none
        # (its template defines no enable_thinking input). See template_thinking_kwargs.
        tmpl_kwargs = template_thinking_kwargs(self.thinking_mode, self.enable_thinking)

        if frames is None:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": [{"type": "text", "text": text_body}]},
            ]
            text = proc.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **tmpl_kwargs
            )
            inputs = proc(text=[text], return_tensors="pt")
        elif lohi:
            if spec is not None:
                raise ValueError(
                    "the lohi presentation does not support a Pruner PatchSpec; run token "
                    "pruning on the single video stream"
                )
            import numpy as _np
            from PIL import Image  # lazy

            from ...models.qwen_tokens import assert_token_count
            from ..chat.interleave import build_lohi_content

            hi_frames = pkg.get("hi_frames")
            content = build_lohi_content(text_body, list(pkg.get("hi_timestamps") or []))
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": content},
            ]
            text = proc.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **tmpl_kwargs
            )
            # `hi_frames or []` would raise: it is a numpy array (ambiguous truth value).
            pil_hi = [
                Image.fromarray(_np.asarray(f, dtype=_np.uint8))
                for f in (hi_frames if hi_frames is not None else [])
            ]
            inputs = proc(
                text=[text],
                videos=[to_processor_video(frames)],
                video_metadata=[pkg["video_metadata"]],
                images=pil_hi or None,
                do_sample_frames=False,
                do_resize=False,   # the encoder already produced the exact grid of each stream
                return_tensors="pt",
            )
            # Both streams must survive text assembly at the size the encoder asserted.
            vgrid = inputs["video_grid_thw"][0].tolist()
            lo_merged = (vgrid[0] * vgrid[1] * vgrid[2]) // (c.merge_size * c.merge_size)
            assert_token_count(int(pkg["num_visual_tokens"]), lo_merged)
            hi_merged = 0
            if pil_hi:
                hi_merged = sum(
                    (g[0] * g[1] * g[2]) // (c.merge_size * c.merge_size)
                    for g in inputs["image_grid_thw"].tolist()
                )
                assert_token_count(int(pkg["hi_num_visual_tokens"]), hi_merged)
            lohi_visual_tokens = lo_merged + hi_merged
        else:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": [
                    {"type": "video", "video": ""},  # placeholder; real frames via videos= below
                    {"type": "text", "text": text_body},
                ]},
            ]
            text = proc.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **tmpl_kwargs
            )
            inputs = proc(
                text=[text],
                videos=[to_processor_video(frames)],
                video_metadata=[pkg["video_metadata"]],
                do_resize=pkg["do_resize"],
                do_sample_frames=False,
                size=pkg["size"],   # 5.x area budget (was min_pixels/max_pixels on 4.x)
                return_tensors="pt",
            )
            # Re-assert the visual budget survived text assembly (fail loud on silent rescale).
            grid = inputs["video_grid_thw"][0].tolist()
            merged = (grid[0] * grid[1] * grid[2]) // (c.merge_size * c.merge_size)
            from ...models.qwen_tokens import assert_token_count

            # Pruning does not change this: the processor still expands the FULL grid, so the
            # PRE-prune count is asserted here.
            assert_token_count(pkg["num_visual_tokens"], merged)

        # Pruner seam: build the per-generate patch context (see prune_seam.py).
        from contextlib import nullcontext

        patch_ctx = nullcontext()
        prune_arm: Optional[dict] = None
        marker_pos: list = []
        if spec is not None:
            if frames is None:
                raise ValueError("prune seam: a PatchSpec was declared but no frames reached the backend")
            video_token_id = getattr(model.config, "video_token_id", None)
            vs_id = getattr(model.config, "vision_start_token_id", None)
            ve_id = getattr(model.config, "vision_end_token_id", None)
            if video_token_id is None or vs_id is None or ve_id is None:
                raise ValueError(
                    "prune seam (pre_llm): model.config lacks video_token_id / "
                    "vision_start_token_id / vision_end_token_id; cannot locate the visual "
                    "placeholders and markers"
                )
            ids_row = [int(v) for v in inputs["input_ids"][0].tolist()]
            vis_pos = [i for i, v in enumerate(ids_row) if v == int(video_token_id)]
            marker_pos = [i for i, v in enumerate(ids_row) if v in (int(vs_id), int(ve_id))]
            prune_arm = {}
            patch_ctx = applied_pre_llm_patch(
                model, spec, vis_pos, marker_pos, prefill_len=len(ids_row),
                grid_thw=inputs["video_grid_thw"], question=prompt,
                tokenizer=proc.tokenizer, tower_target=spec.tower_target, arm_out=prune_arm,
            )

        # device_map="auto" shards the model; move inputs to "cuda" (not model.device, which
        # under sharding may report meta/cpu).
        inputs = inputs.to("cuda")
        # `input_width` is the width of the input_ids fed to generate (used to slice off the
        # answer). Pruning keeps it FULL: the reduction happens on inputs_embeds inside the
        # forward, and `prefill_tokens` is corrected to the reduced width after generate.
        input_width = int(inputs["input_ids"].shape[1])
        prefill_tokens = input_width

        # Greedy, explicit (never a template default). Do NOT pass temperature/top_p when not
        # sampling (some transformers versions warn/error on their presence with do_sample=False).
        pad_id = proc.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = proc.tokenizer.eos_token_id
        effective_cap = int(max_new_tokens) if max_new_tokens is not None else self.max_new_tokens
        gen_kwargs = {
            "max_new_tokens": effective_cap,
            "do_sample": False,
            "num_beams": 1,
            "use_cache": True,
            "eos_token_id": proc.tokenizer.eos_token_id,
            "pad_token_id": pad_id,
        }

        with torch.no_grad(), patch_ctx:
            out = model.generate(**inputs, **gen_kwargs)

        trimmed = out[0][input_width:]                    # generated ids only
        total_gen = int(trimmed.shape[0])

        # LLM INGRESS: the visual tokens actually entering the language model this call.
        # 0 on the frames=None (text-only) path.
        llm_visual_tokens = int(pkg["num_visual_tokens"]) if frames is not None else 0
        if lohi:
            llm_visual_tokens = lohi_visual_tokens
        prune_record = None
        if spec is not None:
            # Honest prefill: the language model ran on the reduced sequence, i.e. the raw
            # width minus the dropped visual tokens and the dropped vision markers.
            post = int(prune_arm.get("post_prune_visual", spec.pre_prune_tokens))
            prefill_tokens = max(1, input_width - (spec.pre_prune_tokens - post) - len(marker_pos))
            llm_visual_tokens = post
            prune_record = {
                "method": spec.method,
                "patch_point": spec.patch_point,
                "compression": spec.compression,
                "pre_prune_visual": spec.pre_prune_tokens,
                "post_prune_visual": post,
                "target_keep": spec.keep_count,
                "prefill_shrinks": True,
                "params": dict(spec.params),
            }

        full_text = proc.decode(trimmed, skip_special_tokens=True,
                                clean_up_tokenization_spaces=False)

        # Thinking accounting — see split_thinking().
        def _encode(s):
            return proc.tokenizer.encode(s, add_special_tokens=False)

        answer_text, decode_tokens, thinking_tokens = split_thinking(
            full_text, total_gen, self.enable_thinking, _encode
        )

        trace0: dict = {
            "arm": {
                "model_id": self.model_id,
                "enable_thinking": self.enable_thinking,
                "thinking_mode": self.thinking_mode,
                "family": family,
                "greedy": True,
                "max_new_tokens": effective_cap,                     # the cap actually applied
                "max_new_tokens_override": max_new_tokens,           # None => backend default
                "visual_tokens": pkg["num_visual_tokens"],
                # Realized-shape audit trail: the processor grid + per-frame (H, W) the encoder
                # asserted, so expected-vs-actual clamp investigations are one jq away.
                "grid_thw": list(pkg["grid_thw"]) if pkg.get("grid_thw") else None,
                "frame_hw": list(pkg["frame_hw"]) if pkg.get("frame_hw") else None,
                "prune": prune_record,                           # None on the dense path
                "presentation": budget.presentation,
                # Dual-stream audit trail (None unless presentation == "lohi").
                "hi_visual_tokens": pkg.get("hi_num_visual_tokens") if lohi else None,
                "hi_frame_hw": list(pkg["hi_frame_hw"]) if lohi and pkg.get("hi_frame_hw") else None,
                "native_hw": list(pkg["native_hw"]) if lohi and pkg.get("native_hw") else None,
                "hi_scale": pkg.get("hi_scale") if lohi else None,
            }
        }

        return Answer(
            text=answer_text.strip(),
            cost=CostRecord(
                prefill_tokens=prefill_tokens,
                decode_tokens=decode_tokens,
                thinking_tokens=thinking_tokens,
                llm_visual_tokens=llm_visual_tokens,
            ),
            # Record the run's arm so results stay identifiable.
            rounds_trace=[trace0],
        )
