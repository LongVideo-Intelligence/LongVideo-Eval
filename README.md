<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/longvideo_eval_logo_dark.png">
  <img src="docs/assets/longvideo_eval_logo.png" alt="LongVideo-Eval logo" width="300">
</picture>

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
  ⭐ <b>Star this repository to follow the release.</b> Code, methods and benchmarks are landing continuously.
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

- **2026.10** — The repository and the [project page](https://sixundong.com/projects/longvideo-eval) are public. Code release is in preparation.
- **2026.10** — Our paper **LoHi** is accepted to NeurIPS 2026 and released on arXiv. [[Paper](https://arxiv.org/abs/2610.04318)] [[Homepage](https://sixundong.com/projects/lohi)]

## 📦 Release plan

| Component | What | Status |
|---|---|---|
| Harness | Stage protocols, shared budgets, unified cost metering, CLI | Waiting to be released |
| Base models | Qwen3-VL family | Waiting to be released |
| Benchmarks | Video-MME, MLVU, LVBench | Waiting to be released |
| Token pruning | VisionZip, MMTok, FlashVID | Waiting to be released |
| Keyframe selection | Query-aware keyframe baselines | Waiting to be released |
| LoHi | LoHi-Uniform, LoHi-SemDiv | Waiting to be released |

Next in line: agentic and multi-turn methods, streaming and memory models, more backbones
and benchmarks. The [project page](https://sixundong.com/projects/longvideo-eval) has the
long-term picture and the [release log](CHANGELOG.md) tracks what has landed.

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
[MMTok](https://github.com/Ironieser/MMTok) and
[lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval), whose ideas and conventions
this project builds on, and the authors of the methods and benchmarks evaluated in the
harness for releasing their code and data.

## License

Apache-2.0, see [LICENSE](LICENSE). Model weights, datasets and third-party methods are
subject to their own licenses.
