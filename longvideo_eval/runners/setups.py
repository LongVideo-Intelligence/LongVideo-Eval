"""Named Budget presets — the shared allocation every method is cost-matched against.

A "setup" fixes the joint axes (frames / resolution / decode budget) so the baselines and
every efficiency method run under the SAME ceiling. The setup name is what the CLI `--setup`
selects and what result tables group by.

NAMING. `<kind>_<frames>f[_r<percent>]...`: the kind in full, the number of frames, and a
resolution as a percentage of the source video's native size (`r25` = a quarter per side).
A name without `r` is at the native size, except `model_default_*`.

Fail-loud: an unknown setup name raises with the available names listed.
"""
from __future__ import annotations

from ..interfaces import Budget

# TWO KINDS OF SETUP.
#
# A. MODEL-DEFAULT (`model_default_*`). The model's released preprocessing, used to reproduce
#    its published numbers. The model decides the resolution, and it changes with the number
#    of frames, because the processor caps the total pixels of a video.
#
# B. FIXED-RESOLUTION, this repository's default (everything else). The setup fixes the
#    resolution as a scale r of the source video's native size. It does not change with the
#    frame count or with a token limit. A frame at scale r costs r^2 of a native frame, so a
#    setup's visual-token budget is  N * r^2:
#
#   16 frames @ r=1.0    16 * 1      = 16      `native_16f`, the reference budget
#   64 frames @ r=0.5    64 * 1/4    = 16      `uniform_64f_r50`
#  256 frames @ r=0.25  256 * 1/16   = 16      `lowres_base_256f`, the default
#
#    - `native_<N>f`: N frames at the native size. 16 is the reference; 64 and 256 are the
#      inputs for token pruners, which keep 1/4 or 1/16 to land on the reference budget.
#    - `uniform_<N>f_r<p>`: N frames at p percent of the native size.
#    - `lowres_base_<N>f`: LOW-RES-BASE, the project's default baseline. Every decoded frame,
#      at a quarter of the native size. The setup fixes the number of DECODED frames, and the
#      family scales the same recipe over frame counts (budget N/16).
#    - `keyframe_<N>f_pool<M>`: N native frames chosen from an M-frame decoded pool. The
#      M-frame decode is charged.
#    - `lohi_<N>f_r<p>_k<K>`: dual stream. N frames at p percent as a video plus K of them
#      as images, sized per video to fill the reference budget.
#
# QWEN NOTE. Qwen merges two video frames into one temporal position; an image gets no such
# merge, so it counts as two video frames. The image stream is therefore scaled down to keep
# the total at the reference budget. The scale is set PER VIDEO: the images take exactly what
# the low-resolution video leaves of the budget of 16 native frames, whatever the source
# resolution. For K=8 that is about 0.7 of the native size per side; for K=4 about 1.0.
#
# Realized token counts are metered per sample.
SETUPS = {
    # A. model-default preprocessing
    "model_default_16f": Budget(frame_count=16, resolution=None, decode_budget=128),
    "model_default_32f": Budget(frame_count=32, resolution=None, decode_budget=128),
    "model_default_64f": Budget(frame_count=64, resolution=None, decode_budget=256),
    # B. fixed resolution (this repository's default)
    "native_16f": Budget(frame_count=16, resolution=1.0, decode_budget=16),
    "native_64f": Budget(frame_count=64, resolution=1.0, decode_budget=64),
    "native_256f": Budget(frame_count=256, resolution=1.0, decode_budget=256),
    "uniform_64f_r50": Budget(frame_count=64, resolution=0.5, decode_budget=64),
    "lowres_base_64f": Budget(frame_count=64, resolution=0.25, decode_budget=64),
    "lowres_base_128f": Budget(frame_count=128, resolution=0.25, decode_budget=128),
    "lowres_base_256f": Budget(frame_count=256, resolution=0.25, decode_budget=256),
    "lowres_base_512f": Budget(frame_count=512, resolution=0.25, decode_budget=512),
    "keyframe_16f_pool256": Budget(frame_count=16, resolution=1.0, decode_budget=256),
    "lohi_128f_r25_k8": Budget(
        frame_count=128, resolution=0.25, decode_budget=128,
        hi_i_count=8, reference_frames=16, presentation="lohi",
    ),
    "lohi_128f_r25_k4": Budget(
        frame_count=128, resolution=0.25, decode_budget=128,
        hi_i_count=4, reference_frames=16, presentation="lohi",
    ),
}


def get_setup(name: str) -> Budget:
    try:
        return SETUPS[name]
    except KeyError:
        raise KeyError(
            f"unknown setup {name!r}; available: {sorted(SETUPS)}"
        ) from None
