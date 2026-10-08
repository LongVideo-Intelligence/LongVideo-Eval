<div align="center">

<img src="docs/assets/longvideo_eval_logo.png#gh-light-mode-only" alt="LongVideo-Eval logo" width="300">
<img src="docs/assets/longvideo_eval_logo_dark.png#gh-dark-mode-only" alt="LongVideo-Eval logo" width="300">

# LongVideo-Eval: Evaluation Infrastructure for Long-Video Intelligence

[![Project Page](https://img.shields.io/badge/Project-Page-8A2BE2?logo=googlechrome&logoColor=white)](https://sixundong.com/projects/longvideo-eval)
[![Release log](https://img.shields.io/badge/Release-log-orange)](CHANGELOG.md)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)](#-get-involved)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)

</div>

<p align="center">
  <em>LongVideo-Eval is a system-aware, complete-cost evaluation harness for long-video
  intelligence. It compares long-video pipelines under fixed backbones, matched evidence
  budgets, and one shared cost ledger spanning video-side and model-side work.</em>
</p>

<p align="center">
  ⭐ <b>Star this repository to follow the release.</b> The harness core is out; methods and benchmarks are landing continuously.
</p>

---

## 🚀 Overview

Long-video methods are compared by accuracy at a matched number of visual tokens. That
leaves out most of what a long video costs. A keyframe selector decodes and scores
hundreds of candidate frames to keep sixteen. A token pruner runs the full vision encoder
before discarding most of its output. Neither shows up in a token count, and each paper
measures it differently, if at all.

LongVideo-Eval makes four things the same for every method, so the only thing left to
compare is the idea:

- **One entry.** Every method is a plug-in for one stage of the same pipeline.
- **One input.** Same backbone, same decoded frame pool, same frame and token budget.
- **One meter.** Decoding, vision encoding, prefill and generation are recorded per sample
  by the harness, never self-reported by a method.
- **One picture.** Every run lands as a point on the same accuracy-cost plane, next to the
  strongest baselines.

What counts is the same for every method: **fewer decoded frames, fewer input tokens, fewer
generated tokens, faster inference, better accuracy**, all measured from the source video as
it was shot.

It is a development platform as much as a benchmark: a new method is one stage, and it is
compared with every baseline from the first run.

```
raw video
  -> Decoder      video -> frame pool          decode time, frames decoded
  -> Selector     pool -> frames               selection time
  -> Encoder      frames -> visual tokens      frames encoded, vision-tower load
  -> Pruner       tokens -> fewer tokens       tokens kept
  -> LLMBackend   tokens + prompt -> answer    prefill, generated tokens
```

## 📅 News

- **2026.10** — First code release: the harness, token pruning (MMTok, VisionZip, FlashVID), keyframe selection (CLIP-TopK, AKS) and the LoHi dual stream, with the Qwen family on Video-MME.
- **2026.10** — The repository and the [project page](https://sixundong.com/projects/longvideo-eval) are public.
- **2026.10** — Our paper **LoHi** is accepted to NeurIPS 2026 and released on arXiv. [[Paper](https://arxiv.org/abs/2610.04318)] [[Homepage](https://sixundong.com/projects/lohi)]

## ⚡ Quickstart

```bash
git clone https://github.com/LongVideo-Intelligence/LongVideo-Eval.git
cd LongVideo-Eval
pip install -e ".[dev]"
pytest -q                      # hardware-free: no GPU, weights or videos needed
```

To run a real model, install the GPU stack and prepare Video-MME once:

```bash
pip install -r requirements-gpu.txt

# Download lmms-lab/Video-MME from Hugging Face and unzip the video archives, then:
python scripts/prepare_videomme.py \
    --parquet /data/Video-MME/videomme/test-00000-of-00001.parquet \
    --videos  /data/Video-MME/data \
    --out     /data/videomme_root
export VIDEOMME_DATA_ROOT=/data/videomme_root
```

Then evaluate. A run is a model, a method, a task and a setup (the shared budget):

```bash
# Low-Res-Base, the default baseline: 256 decoded frames at a quarter of the native size
python -m longvideo_eval --task videomme --model qwen3-vl-4b --method lowres-base --setup lowres_base_256f

# 16 frames at native resolution, the same visual-token budget
python -m longvideo_eval --task videomme --model qwen3-vl-4b --method qwen-default --setup native_16f
```

Every other method runs the same way. Only `--method` and `--setup` change:

```bash
python -m longvideo_eval --task videomme --model qwen3-vl-4b --method mmtok --setup native_64f
```

Add `--limit 20` for a quick check, `--shard i/N` to split a run across GPUs (recombine with
`scripts/merge_shards.py`), and `--dry-run` to exercise the pipeline without weights.

Each run writes `results.jsonl` (one row per question, with the prediction and every cost
field) and `summary.json` (accuracy and the mean of each cost) under `runs/`.

📖 The **[model zoo](docs/MODEL_ZOO.md)** lists every backbone, method, benchmark and setup,
and what each setup means.

The stage protocols live in [`longvideo_eval/interfaces.py`](longvideo_eval/interfaces.py),
which is the place to start reading. A new method implements one of them, registers under an
id, and gets a row in `METHODS` in [`longvideo_eval/models/build.py`](longvideo_eval/models/build.py).

## 📦 Release plan

LongVideo-Eval is built to hold many methods, not a fixed list. The first batch is out, and
each direction below keeps growing. The names in the "available now" column are where a
direction starts, not where it ends.

| Direction | Available now | Status |
|---|---|---|
| Harness | Stage protocols, shared budgets, one cost ledger, command line | ✅ Released |
| Backbones | The Qwen family on Hugging Face transformers | ✅ First batch, more coming |
| Benchmarks | Video-MME | ✅ First batch, more coming |
| Token pruning and merging | MMTok, VisionZip, FlashVID | ✅ First batch, more coming |
| Keyframe selection | CLIP-TopK, AKS | ✅ First batch, more coming |
| Resolution allocation | LoHi-Uniform, LoHi-SemDiv | ✅ First batch, more coming |
| Leaderboard | Every method under every shared setup, with its full cost | 🚧 Coming |
| Agentic and multi-turn methods | | 🚧 Coming |
| Streaming and memory models | | 🚧 Coming |

**More features coming:** a public leaderboard, more backbones and inference engines, more
long-video benchmarks, cached decoding for fast re-runs, and documentation for adding a
method in an afternoon. The [project page](https://sixundong.com/projects/longvideo-eval) has
the long-term picture and the [release log](CHANGELOG.md) tracks what has landed. If a method
or benchmark you care about is missing, open an issue or send a pull request.

## 🤝 Get involved

This repository is updated continuously, and contributions are welcome:

- **Add a method or a benchmark** with a pull request.
- **Propose a dataset, a budget or a cost** the comparison is missing by opening an issue.
- **Collaborate** on joint evaluations or anything larger: sixundong.ai@gmail.com

## 📚 Citation

LongVideo-Eval was inspired by LoHi and builds on the work below. If you find this project
useful, please consider citing it:

- **LoHi**: Rethinking Long-Video Efficiency (NeurIPS 2026) · [arXiv](https://arxiv.org/abs/2610.04318) · [project page](https://sixundong.com/projects/lohi)
- **MMTok**: Multimodal Coverage Maximization for Efficient Inference of VLMs (ICLR 2026) · [arXiv](https://arxiv.org/abs/2508.18264) · [code](https://github.com/Ironieser/MMTok)
- **Rethinking Model Efficiency**: Multi-Agent Inference with Large Models · [arXiv](https://arxiv.org/abs/2604.04929)

Each method in the harness has its own paper. If you use one, please cite it too:

- **FlashVID**: Fan et al., 2026 · [arXiv](https://arxiv.org/abs/2602.08024) · [code](https://github.com/Fanziyang-v/FlashVID)
- **VisionZip**: Yang et al., CVPR 2025 · [arXiv](https://arxiv.org/abs/2412.04467) · [code](https://github.com/JIA-Lab-research/VisionZip)
- **AKS**: Tang et al., CVPR 2025 · [arXiv](https://arxiv.org/abs/2502.21271) · [code](https://github.com/ncTimTang/AKS)

<details>
<summary>BibTeX</summary>

```bibtex
@inproceedings{dong2026lohi,
  title     = {Rethinking Long-Video Efficiency: A Joint Allocation Perspective on Frames, Pixels, and Front-End Latency},
  author    = {Dong, Sixun and Li, Wei and Deng, Andong and Qian, Qi and Zhu, Victor and Ji, Zhengping and Chen, Chen},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026},
  eprint    = {2610.04318},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/2610.04318}
}

@inproceedings{dong2026mmtok,
  title={Mmtok: Multimodal coverage maximization for efficient inference of vlms},
  author={Dong, Sixun and Hu, Juhua and Zhang, Mian and Yin, Ming and Fu, Yanjie and Qian, Qi},
  booktitle={International Conference on Learning Representations},
  volume={2026},
  pages={48075--48099},
  year={2026}
}

@article{dong2026rethinking,
  title={Rethinking Model Efficiency: Multi-Agent Inference with Large Models},
  author={Dong, Sixun and Hu, Juhua and Li, Steven and Wen, Wei and Qian, Qi},
  journal={arXiv preprint arXiv:2604.04929},
  year={2026}
}
```

</details>

## 🙏 Acknowledgements

We thank [Rethinking Model Efficiency](https://arxiv.org/abs/2604.04929),
[MMTok](https://github.com/Ironieser/MMTok),
[FlashVID](https://github.com/Fanziyang-v/FlashVID),
[VisionZip](https://github.com/JIA-Lab-research/VisionZip),
[AKS](https://github.com/ncTimTang/AKS) and
[lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval), whose ideas, code and conventions
this project builds on, and the authors of the methods and benchmarks evaluated in the
harness for releasing their code and data.

## License

Apache-2.0, see [LICENSE](LICENSE). Code adapted from other projects is listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Model weights, datasets and third-party
methods are subject to their own licenses.
