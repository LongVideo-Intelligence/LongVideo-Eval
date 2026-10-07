<p align="center">
  <img src="docs/assets/longvideo_eval_logo.png" alt="LongVideo-Eval logo" width="320">
</p>

<h1 align="center">LongVideo-Eval</h1>

<p align="center">
  <b>A fair-cost evaluation harness for long-video understanding.</b><br>
  The official code home of <b>LoHi</b> (NeurIPS 2026).
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2610.04318"><img src="https://img.shields.io/badge/arXiv-2610.04318-b31b1b.svg" alt="arXiv"></a>
  <a href="https://openreview.net/forum?id=a9xLyT4hG4"><img src="https://img.shields.io/badge/NeurIPS-2026-4b44ce.svg" alt="NeurIPS 2026"></a>
  <a href="https://sixundong.com/projects/longvideo-eval"><img src="https://img.shields.io/badge/Project-Page-2ea44f.svg" alt="Project page"></a>
  <a href="https://sixundong.com/projects/lohi"><img src="https://img.shields.io/badge/LoHi-Page-d2382c.svg" alt="LoHi page"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-blue.svg" alt="License"></a>
</p>

> **Status: code release in progress.** The harness and the methods listed below are
> implemented and are being prepared for release. Star or watch the repository to be
> notified when they land.

## LoHi (NeurIPS 2026)

**Rethinking Long-Video Efficiency: A Joint Allocation Perspective on Frames, Pixels, and Front-End Latency**

Sixun Dong<sup>1</sup>, Wei Li<sup>1</sup>, Andong Deng<sup>1</sup>, Qi Qian<sup>2</sup>,
Victor Zhu<sup>3</sup>, Zhengping Ji<sup>3</sup>, Chen Chen<sup>1</sup>

<sup>1</sup>University of Central Florida &nbsp; <sup>2</sup>Meta Reality Labs &nbsp; <sup>3</sup>Axon

[Paper](https://arxiv.org/abs/2610.04318) ·
[OpenReview](https://openreview.net/forum?id=a9xLyT4hG4) ·
[Project page](https://sixundong.com/projects/lohi) ·
[Blog](https://sixundong.com/blog.html?post=lohi-rethinking-long-video-efficiency)

Three lessons from the paper:

1. **Your model needs more frames, not more pixels.**
2. **Low-resolution frames answer most questions; a few need detail.**
3. **Don't ignore the front-end: decoding can outlast the model.**

LoHi will be released here, as part of the harness.

## Why a harness

Long-video methods are usually compared by accuracy at a matched number of visual tokens.
That leaves out most of what a long video actually costs. A keyframe selector may decode
and score hundreds of candidate frames to keep sixteen. A token pruner may run the full
vision encoder before discarding most of its output. Neither cost shows up in a token
count, and each paper measures it differently, if at all.

LongVideo-Eval evaluates long-video VLMs and efficiency methods on **one accuracy-cost
axis**, with decoding, vision encoding and prefill all measured in one place:

- **One pipeline, one stage per method.** Every method plugs into exactly one stage of a
  fixed chain and inherits everything else unchanged.
- **One budget.** Every method in a comparison reads the same decoded frame pool under the
  same frame and token budget.
- **One cost record.** Decode time, frames decoded, vision-encoder load, visual tokens into
  the LLM, prefill and generated tokens are recorded per sample by the harness, never
  self-reported by a method.

## Framework

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

## Release plan

Everything below is implemented. Items are checked off as they are released.

| Component | What | Status |
|---|---|---|
| Harness | Stage protocols, shared budgets, unified cost metering, CLI | Waiting to be released |
| Base models | Qwen3-VL family | Waiting to be released |
| Benchmarks | Video-MME, MLVU, LVBench | Waiting to be released |
| Token pruning | VisionZip, MMTok, FlashVID | Waiting to be released |
| Keyframe selection | Query-aware keyframe baselines | Waiting to be released |
| LoHi | LoHi-Uniform, LoHi-SemDiv | Waiting to be released |

## Where this is going

LoHi was the first paper to come out of this harness. The harness itself is the long-term
project: a common ground where the whole field is measured the same way, and a workbench
for building what comes next.

- **Evaluate.** Run any method on any benchmark under the same budget and get its full
  bill, not only its accuracy.
- **Develop.** Write a new method as one stage, inherit the rest of the pipeline, and see
  it next to every baseline the same afternoon.
- **Extend.** A new benchmark is one task file, a new budget is one line, and a new kind
  of cost is one field that is then metered for every method.
- **Cover the field.** Token pruning, keyframe selection and resolution allocation first,
  then agentic and multi-turn methods, streaming and memory models, more backbones and
  faster backends.

See the [project page](https://sixundong.com/projects/longvideo-eval) for the full picture
and the [release log](CHANGELOG.md) for what has landed.

## Get involved

This repository is updated continuously, and contributions are welcome:

- **Add a method or a benchmark** with a pull request.
- **Propose a dataset, a budget or a cost** the comparison is missing by opening an issue.
- **Collaborate** on joint evaluations or anything larger: sixundong.ai@gmail.com

## Papers

If you find this project useful, please consider citing our work:

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
  title     = {{MMT}ok: Multimodal Coverage Maximization for Efficient Inference of {VLM}s},
  author    = {Dong, Sixun and Hu, Juhua and Zhang, Mian and Yin, Ming and Fu, Yanjie and Qian, Qi},
  booktitle = {The Fourteenth International Conference on Learning Representations (ICLR)},
  year      = {2026},
  url       = {https://openreview.net/forum?id=GvPdSWZT31}
}

@article{dong2026rethinking,
  title   = {Rethinking Model Efficiency: Multi-Agent Inference with Large Models},
  author  = {Dong, Sixun and Hu, Juhua and Li, Steven and Wen, Wei and Qian, Qi},
  journal = {arXiv preprint arXiv:2604.04929},
  year    = {2026}
}
```

</details>

## Acknowledgements

This project stands on the following work:

- [Rethinking Model Efficiency: Multi-Agent Inference with Large Models](https://arxiv.org/abs/2604.04929),
  earlier work from the authors on inference efficiency that motivates this project.
- [MMTok](https://github.com/Ironieser/MMTok), our multimodal coverage-maximization token
  selection method, which is one of the pruning baselines in the harness.
- [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval), whose task and model
  conventions the harness follows so that benchmarks stay easy to port.

We also thank the authors of the methods and benchmarks evaluated in the harness for
releasing their code and data.

## License

Apache-2.0, see [LICENSE](LICENSE). Model weights, datasets and third-party methods are
subject to their own licenses.
