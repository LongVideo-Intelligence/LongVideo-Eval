# Model zoo

Everything a run can be built from: backbones, methods, benchmarks and setups. A run is one
of each:

```bash
python -m longvideo_eval --task <benchmark> --model <backbone> --method <method> --setup <setup>
```

The lists below mirror the code. `BASE_MODELS` and `DATASETS` live in
[`longvideo_eval/config.py`](../longvideo_eval/config.py), `METHODS` in
[`longvideo_eval/models/build.py`](../longvideo_eval/models/build.py) and `SETUPS` in
[`longvideo_eval/runners/setups.py`](../longvideo_eval/runners/setups.py). An unknown name
fails with the valid ones listed.

## Backbones

| `--model` | Checkpoint | Thinking |
|---|---|---|
| `qwen3-vl-4b` | `Qwen/Qwen3-VL-4B-Instruct` | off only |
| `qwen3-vl-8b` | `Qwen/Qwen3-VL-8B-Instruct` | off only |
| `qwen3-vl-4b-thinking` | `Qwen/Qwen3-VL-4B-Thinking` | on only (`--thinking`) |
| `qwen3.5-4b` | `Qwen/Qwen3.5-4B` | on or off |
| `qwen2.5-vl-7b` | `Qwen/Qwen2.5-VL-7B-Instruct` | off only |

All run on Hugging Face transformers 5. Thinking tokens are metered in their own field.

## Benchmarks

| `--task` | Benchmark | Preparation |
|---|---|---|
| `videomme` | Video-MME | `scripts/prepare_videomme.py`, then `export VIDEOMME_DATA_ROOT=...` |

## Methods

A method is one component in one pipeline stage. Everything else in the pipeline is shared.

| `--method` | Stage | What it does | Typical setup |
|---|---|---|---|
| `qwen-default` | none | Uniform frames, every visual token kept | `native_16f` |
| `lowres-base` | none | **Low-Res-Base**, the default baseline: every decoded frame, at a low resolution | `lowres_base_256f` |
| `mmtok` | prune | Keeps the tokens that best cover the question and the image content | `native_64f` |
| `visionzip` | prune | Keeps high-attention tokens and merges the rest into contextual tokens | `native_64f` |
| `flashvid` | prune | Segments the video, then selects and merges tokens across time and space | `native_64f` |
| `random-prune` | prune | Control: a random subset at the same retention | `native_64f` |
| `cliptopk` | select | Keeps the frames most similar to the question | `keyframe_16f_pool256` |
| `aks` | select | Balances relevance against coverage of the timeline | `keyframe_16f_pool256` |
| `lohi-uniform` | select | Low-res video plus K higher-res images at regular intervals | `lohi_128f_r25_k8` |
| `lohi-semdiv` | select | The same, with the K images chosen by relevance and diversity | `lohi_128f_r25_k8` |

Token pruners take their retention from `--prune-kwargs`, for example
`--prune-kwargs '{"keep_ratio": 0.0625}'`. The default keeps a quarter of the visual tokens.
On the Qwen families pruning happens after the vision tower and before the language model,
so the tower is charged for every frame and the language model only for what survives.

The pruning methods were proposed for LLaVA-style models. Here they are adapted to the
Qwen3-VL family: they act on the tokens after the patch merger, and methods that read a CLS
attention use the mean self-attention of the last vision block instead. Each method's module
docstring states its adaptation.

## Setups

A setup is the budget every method in a comparison shares: how many frames are decoded, how
many are kept, and at what resolution. There are two kinds.

### Model-default setups

The model's released preprocessing. Use these to reproduce a model's published numbers. The
model decides the resolution, and it changes with the number of frames, because the
processor caps the total pixels of a video.

| `--setup` | Decoded | Kept |
|---|---|---|
| `model_default_16f` | 128 | 16 |
| `model_default_32f` | 128 | 32 |
| `model_default_64f` | 256 | 64 |

### Fixed-resolution setups (this repository's default)

The setup fixes the resolution as a scale `r` of the source video: `r50` is half the native
width and height. It does not change with the frame count or with a token limit. A frame at
scale `r` costs `r²` of a native frame, so a setup's visual-token budget is **frames × r²**:

| | frames × r² |
|---|---|
| 16 frames at 1.0 | 16 × 1 = 16 |
| 64 frames at 0.5 | 64 × ¼ = 16 |
| 256 frames at 0.25 | 256 × 1⁄16 = 16 |

| `--setup` | Decoded | Kept | Scale | Use |
|---|---|---|---|---|
| `native_16f` | 16 | 16 | 100% | The reference budget |
| `native_64f` | 64 | 64 | 100% | Input for a pruner that keeps 1/4 |
| `native_256f` | 256 | 256 | 100% | Input for a pruner that keeps 1/16 |
| `uniform_64f_r50` | 64 | 64 | 50% | The reference budget with more frames |
| `lowres_base_256f` | 256 | 256 | 25% | **Low-Res-Base**, the default `--setup` |
| `lowres_base_64f` | 64 | 64 | 25% | Low-Res-Base at a quarter of the frames |
| `lowres_base_128f` | 128 | 128 | 25% | Low-Res-Base at half the frames |
| `lowres_base_512f` | 512 | 512 | 25% | Low-Res-Base at twice the frames |
| `keyframe_16f_pool256` | 256 | 16 | 100% | Keyframe selection: the 256-frame decode is charged |
| `lohi_128f_r25_k8` | 128 | 128 + 8 | 25%, images per video | Dual stream at the reference budget |
| `lohi_128f_r25_k4` | 128 | 128 + 4 | 25%, images per video | Dual stream, fewer and sharper images |

**Names** read `<kind>_<frames>f`, plus `_r<percent>` for a resolution as a percentage of the
native size. A name without `r` is at the native size (`model_default_*` aside).

**Low-Res-Base** is the project's default baseline: every decoded frame, at a quarter of the
native size. What the setup fixes is the number of decoded frames, and `lowres_base_<N>f`
scales the same recipe over frame counts. Its budget is N/16, so 256 frames match the
reference of 16 native frames.

A dual-stream setup is N frames at a low scale as a video, plus K of them as images.

**Qwen note.** Qwen merges two video frames into one temporal position. An image gets no
such merge, so it counts as two video frames. The image stream is therefore scaled down to
keep the total at the reference budget, and the scale is set **per video**: the images take
exactly what the low-resolution video leaves of the budget of 16 native frames. A 720p and a
1080p video get different image sizes and the same total.

| | video + images | typical image scale |
|---|---|---|
| `lohi_128f_r25_k8` | 128 × 1⁄16 + 2 × 8 × ½ = 8 + 8 = 16 | about 70% per side |
| `lohi_128f_r25_k4` | 128 × 1⁄16 + 2 × 4 × 1 = 8 + 8 = 16 | about 100% per side |

**Why the source video is the reference.** The goal is fewer decoded frames, fewer input
tokens, fewer generated tokens, faster inference and better accuracy. Generation is part of
the bill: answer tokens and thinking tokens are metered separately for every sample.
Measuring all of it from the video as it was shot gives every method the same starting
point. A method that picks its own resolution, per video, per frame or per region, then gets
full credit for every pixel it leaves out.

## Adding to the zoo

See [AGENTS.md](../AGENTS.md) for the rules a new method, benchmark or backbone must follow,
and open a pull request.
