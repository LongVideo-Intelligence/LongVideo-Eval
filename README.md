<p align="center">
  <img src="docs/assets/longvideo_eval_logo.png" alt="LongVideo-Eval logo" width="320">
</p>

<h1 align="center">LongVideo-Eval</h1>

<p align="center">
  A fair-cost evaluation harness for long-video understanding.
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2610.04318">LoHi paper (arXiv)</a> ·
  <a href="https://sixundong.com/projects/lohi">LoHi project page</a>
</p>

> **Code is coming soon.** We are preparing the first public release. Star or watch the
> repository to be notified when it lands.

## What this is

Long-video methods are usually compared by accuracy at a matched number of visual tokens.
That leaves out most of what a long video actually costs. A keyframe selector may decode
and score hundreds of candidate frames to keep sixteen. A token pruner may run the full
vision encoder before discarding most of its output. Neither cost shows up in a token
count, and each paper measures it differently, if at all.

LongVideo-Eval is an open-source harness that evaluates long-video VLMs and efficiency
methods on **one accuracy-cost axis**, with decoding, vision encoding and prefill all
measured in one place:

- **One pipeline, one stage per method.** Every method plugs into exactly one stage of a
  fixed chain (decode, select, encode, prune, generate) and inherits everything else.
- **One budget.** Every method in a comparison reads the same decoded frame pool under the
  same frame and token budget.
- **One cost record.** Decode time, frames decoded, vision-encoder load, visual tokens into
  the LLM, prefill and generated tokens are recorded per sample by the harness, never
  self-reported by a method.

## What is being released

- [ ] The harness: stage protocols, shared budgets, unified cost metering
- [ ] Qwen3-VL on Video-MME as the reference pipeline
- [ ] Token-pruning baselines: VisionZip, MMTok, FlashVID
- [ ] Keyframe-selection baselines
- [ ] LoHi, released as part of this harness
- [ ] More benchmarks and base models

## Citation

LongVideo-Eval grows out of LoHi. If you find this project useful, please cite:

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
```

## License

Apache-2.0, see [LICENSE](LICENSE).
