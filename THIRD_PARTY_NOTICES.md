# Third-party notices

LongVideo-Eval is released under the Apache License 2.0 (see `LICENSE`). Parts of it are
adapted from the projects below, which keep their own licenses.

## FlashVID

`longvideo_eval/backend/hf/_flashvid_torch.py` and the FlashVID section of
`longvideo_eval/backend/hf/pre_llm_compress.py` are adapted from the official FlashVID
implementation, https://github.com/Fanziyang-v/FlashVID.

```
MIT License

Copyright (c) 2026 Turbo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## VisionZip

`longvideo_eval/frontend/prune/visionzip.py` and the VisionZip section of
`longvideo_eval/backend/hf/pre_llm_compress.py` implement VisionZip
(https://github.com/JIA-Lab-research/VisionZip, Apache License 2.0) for the Qwen3-VL family.

## AKS

`longvideo_eval/frontend/select/aks.py` is an independent implementation of Adaptive Keyframe
Sampling, written from the method's description (Tang et al., CVPR 2025,
https://github.com/ncTimTang/AKS). It contains no code from that repository.

## MMTok

`longvideo_eval/frontend/prune/mmtok.py` and the MMTok section of
`longvideo_eval/backend/hf/pre_llm_compress.py` implement MMTok
(https://github.com/Ironieser/MMTok) for the Qwen3-VL family.
