# AGENTS.md

Guidance for coding agents working in this repository. Humans may find it a useful map too.

## What this repository is

LongVideo-Eval is a system-aware, complete-cost evaluation harness for long-video
intelligence. It compares long-video pipelines under fixed backbones, matched evidence
budgets, and one shared cost ledger spanning video-side and model-side work.

The package is `longvideo_eval/`. A run is one (model, method, task, setup).

## The one design you must understand

Every run is a fixed chain of five stages. A method implements exactly one of them:

```
raw video
  -> Decoder     frontend/decode    -> FramePool      frames + decode cost
  -> Selector    frontend/select    -> Selection      which frames, at what resolution
  -> Encoder     frontend/encode    -> VisualTokens   exact visual-token count
  -> Pruner      frontend/prune     -> VisualTokens   token reduction
  -> LLMBackend  backend            -> Answer         prefill + generation
```

Read `longvideo_eval/interfaces.py` first. It holds the stage protocols, `Budget` (the
allocation every method is run under) and `CostRecord` (every cost of one sample).

| Path | What lives there |
|---|---|
| `interfaces.py` | Stage protocols, `Budget`, `CostRecord`, the data passed between stages |
| `frontend/` | `decode/`, `select/`, `encode/`, `prune/`: one folder per stage |
| `backend/hf/` | Hugging Face backend for the Qwen family; `prune_seam.py` applies a pruner inside `generate`; `pre_llm_compress.py` holds the pruning kernels |
| `backend/chat/` | Prompt content builders (the LoHi video + images layout) |
| `orchestration/` | `SingleShotOrchestrator`: runs the five stages once and sums their cost |
| `metering/` | `Meter`: accumulates a `CostRecord` across stages |
| `models/build.py` | `METHODS`: binds a method id to one component per stage |
| `models/qwen_tokens.py` | Pure-Python mirror of the Qwen processors' token arithmetic |
| `config.py` | `BASE_MODELS`, `DATASETS` |
| `runners/` | `setups.py` (named budgets), `evaluate.py`, `results.py` |
| `data/`, `tasks/` | Dataset loaders and task YAML files |
| `testing.py` | Hardware-free fakes used by `--dry-run` and the tests |

## Commands

```bash
pip install -e ".[dev]"      # base install: no GPU stack needed
ruff check .
pytest -q                    # hardware-free; decord and transformers tests skip if absent

pip install -r requirements-gpu.txt                      # to run real models
python scripts/prepare_videomme.py --help                # build the Video-MME layout once
python -m longvideo_eval --task videomme --model qwen3-vl-4b --method qwen-default --setup native_16f
python -m longvideo_eval --task videomme --dry-run --limit 5   # full pipeline, no weights
python scripts/smoke_gpu.py --help                       # one video, one question, on a GPU
```

Useful flags: `--limit N`, `--shard i/N` (recombine with `scripts/merge_shards.py`),
`--only-videos`, `--subtitles`, `--thinking`, `--max-new-tokens`, `--output`.

A run writes `results.jsonl` (one row per question, with every cost field) and
`summary.json` under `runs/<task>_<method>_<setup>_<timestamp>/`.

## Rules that must hold

These are what make results comparable. Do not trade them away to make something work.

1. **One stage per method.** A method is one registered component in one stage folder. If
   an idea seems to need two stages, it is two components bound together by one `METHODS` row.
2. **The harness meters cost, never the method.** Each stage returns only its own increment
   in its output's `.cost`. The orchestrator sums them. Do not let a component report a
   total, and do not fold another stage's cost into yours.
3. **Everyone reads the same pool.** Selectors operate on the `FramePool` the decoder
   produced under `Budget.decode_budget`. A selector must not decode extra frames itself.
   If a method scans 256 frames to keep 16, the setup charges it for 256.
4. **Token counts are computed, then asserted.** The encode stage computes the visual-token
   count with `models/qwen_tokens.py` and checks it against the processor's own grid
   (`assert_token_count`). The backend asserts it again after the prompt is assembled. If
   the assert fires, fix the arithmetic or the size mapping. Never catch or bypass it.
5. **Fail loudly.** An unknown model, method, task or setup, a missing video, or an unknown
   metric raises with the valid options named. Rows are never skipped silently. A sample
   that raises during a run is recorded as an error row and the run continues.
6. **The package imports without a GPU stack.** `torch`, `transformers`, `decord` and
   `numpy`-heavy paths are imported lazily inside the functions that need them. Keep it so:
   CI runs on the base install.
7. **Thinking is an explicit arm.** Whether a model thinks is set by `--thinking` and
   validated against `ModelSpec.thinking`. Thinking tokens are metered in their own field.
8. **Resolution is a scale of the source video.** `Budget.resolution` and
   `Budget.hi_resolution` are per-side scales of the native frame size
   (`frontend/decode/decoder.py::resize_target`). A setup's token budget is frames × r².

## Adding a method

1. Implement one protocol from `interfaces.py` in the matching stage folder, for example a
   `Selector` in `frontend/select/my_method.py`.
2. Register it under a stable id: `@register("select", "my-method")`.
3. Import the module in `models/build.py` so the decorator runs, and add a `METHODS` row
   naming your component for its stage and the existing components for the others.
4. Add a hardware-free test. Use the fakes in `testing.py`; see `tests/test_smoke.py`.
5. If the method needs a new budget, add a named preset to `runners/setups.py` rather than
   hard-coding numbers in the component.
6. Add a row to `docs/MODEL_ZOO.md`. It is the user-facing list of backbones, methods,
   benchmarks and setups; the README stays short and links to it.

A selector must return indices in temporal order and expand `budget.resolution` into
`per_frame_resolution` when it is set (see `frontend/select/uniform.py`). A pruner returns a
fresh `VisualTokens` whose `.cost` holds only the pruning increment
(see `frontend/prune/identity.py`).

**Token pruners are declarative.** For a Hugging Face Qwen model the vision tower runs
inside `model.generate`, so the visual tokens do not exist yet at the `Pruner` stage. A
pruner attaches a `PatchSpec` to the token package (see `frontend/prune/mmtok.py`), and the
backend applies it between the vision tower and the language model
(`backend/hf/prune_seam.py`). To add one: write the selection math as a kernel in
`backend/hf/pre_llm_compress.py`, add its id to `PRE_LLM_COMPRESSIONS` and to
`PRE_LLM_SIGNALS` (the tower signals it needs, so nothing else is computed), dispatch it in
`_run_pre_llm_compression`, and write the small `Pruner` class that declares it.

**A dual-stream method is a selector.** It returns every low-resolution frame in
`Selection.indices` and the high-resolution subset in `Selection.hi_indices`, and runs under
a `presentation="lohi"` setup (see `frontend/select/semdiv.py`).

## Adding a benchmark or a model

- **Benchmark**: a loader class in `data/` that yields `Sample`s and exposes `golds`
  (`Sample.key` to answer), a task YAML in `tasks/`, and a `DATASETS` entry in `config.py`.
  Follow `data/videomme.py` and `tasks/videomme.yaml`. Key gold answers by `Sample.key`,
  not by `video_id`: one video usually has several questions.
- **Model**: a `BASE_MODELS` entry in `config.py`. A new Qwen-family checkpoint may also
  need vision constants in `frontend/encode/encoder.py::vision_constants`. A different
  model family needs its own backend under `backend/`.

## Conventions

- Python 3.10+, `ruff` clean, line length 100.
- Tests must pass without a GPU, weights or video files. Guard anything heavier with
  `pytest.importorskip`.
- Do not commit videos, weights, datasets or run outputs. `.gitignore` covers the usual
  extensions.
- Keep data paths out of the repository: task YAML files read them from environment
  variables such as `VIDEOMME_DATA_ROOT`.
