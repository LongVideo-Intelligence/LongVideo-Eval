# Release log

All notable changes to LongVideo-Eval are recorded here, newest first.

## v0.1.0

First code release.

**Harness**
- Stage protocols, the shared `Budget` and the unified `CostRecord` (`longvideo_eval/interfaces.py`).
- Front-end stages: decord decoder, selectors, a Qwen encode stage with an exact visual-token
  count asserted against the processor, pruners.
- Hugging Face backend for Qwen3-VL (4B, 8B, 4B-Thinking), Qwen3.5-4B and Qwen2.5-VL-7B,
  with the thinking arm metered separately.
- Command line runner with `--limit`, `--shard`, `--only-videos`, `--subtitles`, `--thinking`
  and `--dry-run`; `scripts/merge_shards.py` to recombine shards.
- Video-MME task and loader, plus `scripts/prepare_videomme.py` for the official release.

**Setups**
- Resolution is a scale of the source video's native size, fixed by the setup. A setup's
  visual-token budget is frames × r².
- **Low-Res-Base** is the default baseline: `lowres_base_<N>f` is N decoded frames at a
  quarter of the native size, and `lowres_base_256f` is the default `--setup`.
- `native_*`, `uniform_*`, `keyframe_*` and `lohi_*` setups at the same reference budget, and
  `model_default_*` setups that reproduce a model's released preprocessing.

**Methods**
- **Token pruning**: MMTok, VisionZip and FlashVID, adapted to the Qwen3-VL family, plus a
  random control. A pruner declares what to drop and the backend applies it after the vision
  tower and before the language model (`backend/hf/prune_seam.py`), so the tower is charged
  for every frame and the language model only for what survives. MMTok uses a lazy greedy
  maximiser that selects the same tokens as the textbook loop, much faster.
- **Keyframe selection**: CLIP-TopK and AKS over a shared decoded pool, scored with CLIP
  ViT-B/32. The whole pool's decode is charged.
- **LoHi dual stream**: LoHi-Uniform and LoHi-SemDiv. Low-resolution frames through the video
  pathway plus K of them through the image pathway.

`docs/MODEL_ZOO.md` lists every backbone, method, benchmark and setup.

## 2026-10-07

- The repository is public.
- Project page: https://sixundong.com/projects/longvideo-eval
- LoHi (NeurIPS 2026) is on arXiv: https://arxiv.org/abs/2610.04318
