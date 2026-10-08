"""Chat-content builders for presentations that mix a video block with image blocks."""
from __future__ import annotations

from typing import Sequence

# Cue placed between the low-resolution video and the high-resolution images.
LOHI_CUE = (
    "The above low-resolution video provides temporal context. "
    "The following {k} high-resolution images show selected key frames:"
)


def build_lohi_content(
    query: str, hi_timestamps: Sequence[float], video_placeholder: bool = True
) -> list[dict]:
    """LoHi dual-stream user content: [video] + cue + K x (<t s> + image) + question.

    The per-image ``<x.x seconds>`` stamps carry the Hi-I frames' TRUE source timestamps, so
    each high-resolution view is tied back to a point on the Lo-V timeline. This matters most
    on Qwen2.5-VL, whose video pathway encodes time in its position ids and emits no timestamp
    text: for that family this cue is the ONLY time signal the image blocks receive.
    """
    content: list[dict] = []
    if video_placeholder:
        content.append({"type": "video", "video": ""})  # real frames via videos= at generate
    content.append({"type": "text", "text": LOHI_CUE.format(k=len(hi_timestamps))})
    for t in hi_timestamps:
        content.append({"type": "text", "text": f"<{t:.1f} seconds>"})
        content.append({"type": "image", "image": ""})  # real frames via images=
    content.append({"type": "text", "text": query})
    return content
