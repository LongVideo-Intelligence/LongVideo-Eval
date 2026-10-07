<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/longvideo_eval_logo_dark.png">
  <img src="docs/assets/longvideo_eval_logo.png" alt="LongVideo-Eval logo" width="300">
</picture>

# LongVideo-Eval: Evaluation Infrastructure for Long-Video Intelligence

[![Project Page](https://img.shields.io/badge/Project-Page-8A2BE2?logo=googlechrome&logoColor=white)](https://sixundong.com/projects/longvideo-eval)
[![GitHub](https://img.shields.io/badge/GitHub-Code-black?logo=github)](https://github.com/Ironieser/LongVideo-Eval)
[![Release log](https://img.shields.io/badge/Release-log-orange)](CHANGELOG.md)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)](#-get-involved)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)

Maintained by **[Sixun Dong](https://sixundong.com)** and [contributors](https://sixundong.com/projects/longvideo-eval#contributors)

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

**LongVideo-Eval** is a **system-aware, complete-cost** evaluation harness for long-video
intelligence. It compares long-video pipelines under **fixed backbones**, **matched
evidence budgets**, and **one shared cost ledger** spanning video-side and model-side
work: decoding, vision encoding, prefill and generation, all measured in one place.

It is a development platform as much as a benchmark. A new method is one stage of a shared
pipeline, and it is compared with every baseline under the same input and the same budget
from the first run.

## ✨ Key Highlights

- **One entry.** Every method is a plug-in for one stage of the same pipeline, whether it
  selects frames, allocates resolution, prunes tokens or runs several rounds. Everything
  around it is shared.
- **One input.** Same backbone, same decoded frame pool, same frame and token budget for
  every method in a comparison.
- **One meter.** Decode time, frames decoded, vision-encoder load, visual tokens into the
  LLM, prefill and generated tokens are recorded per sample by the harness, never
  self-reported by a method.
- **One picture.** Every run lands as a point on the same accuracy-cost plane, next to the
  strongest baselines.
- **Built to grow.** A new benchmark is one task file, a new budget is one line, and a new
  kind of cost is one field that is then metered for every method.

## 📅 News

- **2026.10** — The repository and the [project page](https://sixundong.com/projects/longvideo-eval) are public. Code release is in preparation.
- **2026.10** — Our paper **LoHi** is accepted to NeurIPS 2026 and released on arXiv. [[Paper](https://arxiv.org/abs/2610.04318)] [[Homepage](https://sixundong.com/projects/lohi)]

## 🧩 Framework

A run is a fixed chain of stages. Each stage is a small protocol, and a method implements
exactly one of them:

```
raw video
  -> Decoder      frontend/decode     -> FramePool       frames + decode cost
  -> Selector     frontend/select     -> Selection       which frames, at what resolution
  -> Encoder      frontend/encode     -> VisualTokens    exact visual-token count
  -> Pruner       frontend/prune      -> VisualTokens    token reduction
  -> LLMBackend   backend             -> Answer          prefill + decode
```

```
longvideo_eval/
├─ interfaces.py     stage protocols, Budget, CostRecord
├─ frontend/         decode/  select/  encode/  prune/
├─ backend/          LLM inference backends
├─ orchestration/    wires the stages for one sample
├─ metering/         unified cost accounting
├─ data/  tasks/     dataset loaders and task definitions
├─ models/           binds a (model, method) pair into a pipeline
└─ runners/          evaluation loop, budgets, results
```

Adding a method will mean implementing one stage protocol and registering it under an id.
The budget, the rest of the pipeline and the metering are inherited, so every method is
directly comparable with every other one.

## 📦 Release plan

Everything below is implemented. Items are checked off as they are released.

| Component | What | Status |
|---|---|---|
| Harness | Stage protocols, shared budgets, unified cost metering, CLI | Waiting to be released |
| Base models | Qwen3-VL family | Waiting to be released |
| Benchmarks | Video-MME, MLVU, LVBench | Waiting to be released |
| Token pruning | VisionZip, MMTok, FlashVID | Waiting to be released |
| Keyframe selection | Query-aware keyframe baselines | Waiting to be released |
| LoHi | LoHi-Uniform, LoHi-SemDiv | Waiting to be released |

## 🔭 Where this is going

The harness is a long-term project: a common ground where the whole field is measured the
same way, and a workbench for building what comes next.

- **Now.** Token pruning, keyframe selection and resolution allocation on the Qwen3-VL
  family, across Video-MME, MLVU and LVBench, every run with a full bill.
- **Next.** Agentic and multi-turn methods, where every extra round is charged. Streaming
  and memory models. More backbones, faster inference backends, more benchmarks, with full
  documentation and an API reference.
- **Then.** A public accuracy-cost leaderboard that anyone can add a point to, with
  methods, datasets and budgets contributed by the people who build them.

See the [project page](https://sixundong.com/projects/longvideo-eval) for the full picture
and the [release log](CHANGELOG.md) for what has landed.

## 🤝 Get involved

This repository is updated continuously, and contributions are welcome:

- **Add a method or a benchmark** with a pull request.
- **Propose a dataset, a budget or a cost** the comparison is missing by opening an issue.
- **Collaborate** on joint evaluations or anything larger: sixundong.ai@gmail.com

## 💡 Inspiration: LoHi

LongVideo-Eval was inspired by [LoHi](https://sixundong.com/projects/lohi) (NeurIPS 2026),
whose three lessons are the principles the harness is built to test at scale:

1. **Your model needs more frames, not more pixels.**
2. **Low-resolution frames answer most questions; a few need detail.**
3. **Don't ignore the front-end: decoding can outlast the model.**

LoHi is one method family in the harness, and its code is released here.

## 📚 Papers

If you find this project useful, please consider citing the work it builds on:

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

This project stands on the following work:

- [Rethinking Model Efficiency: Multi-Agent Inference with Large Models](https://arxiv.org/abs/2604.04929),
  earlier work on inference efficiency that motivates this project.
- [MMTok](https://github.com/Ironieser/MMTok), a multimodal coverage-maximization token
  selection method, which is one of the pruning baselines in the harness.
- [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval), whose task and model
  conventions the harness follows so that benchmarks stay easy to port.

We also thank the authors of the methods and benchmarks evaluated in the harness for
releasing their code and data.

## License

Apache-2.0, see [LICENSE](LICENSE). Model weights, datasets and third-party methods are
subject to their own licenses.
