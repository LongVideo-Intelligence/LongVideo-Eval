"""Tests for the dual-stream chat content builder (backend/chat/interleave.py).

The LoHi presentation sends one low-resolution video block followed by K high-resolution
images. Each image is preceded by a ``<t seconds>`` stamp that ties it back to a point on the
video's timeline, and a short cue separates the two streams.
"""
from __future__ import annotations

from longvideo_eval.backend.chat.interleave import LOHI_CUE, build_lohi_content


def test_lohi_content_layout_video_cue_images_question():
    content = build_lohi_content("what happens?", [0.0, 1.5, 3.0])
    assert content[0] == {"type": "video", "video": ""}
    assert content[1] == {"type": "text", "text": LOHI_CUE.format(k=3)}
    # One timestamp text followed by one image block per high-resolution frame, in order.
    assert content[2] == {"type": "text", "text": "<0.0 seconds>"}
    assert content[3] == {"type": "image", "image": ""}
    assert content[4] == {"type": "text", "text": "<1.5 seconds>"}
    assert content[5] == {"type": "image", "image": ""}
    assert content[6] == {"type": "text", "text": "<3.0 seconds>"}
    assert content[7] == {"type": "image", "image": ""}
    # The question comes last.
    assert content[8] == {"type": "text", "text": "what happens?"}
    assert len(content) == 9


def test_lohi_content_has_one_video_and_one_image_block_per_frame():
    stamps = [float(i) for i in range(8)]
    content = build_lohi_content("q", stamps)
    assert sum(1 for c in content if c["type"] == "video") == 1
    assert sum(1 for c in content if c["type"] == "image") == 8
    # Every image block is immediately preceded by its own timestamp text.
    for i, c in enumerate(content):
        if c["type"] == "image":
            assert content[i - 1]["type"] == "text"
            assert content[i - 1]["text"].endswith(" seconds>")


def test_lohi_cue_states_the_number_of_images():
    assert "{k}" in LOHI_CUE
    cue = build_lohi_content("q", [1.0, 2.0, 3.0, 4.0])[1]["text"]
    assert "4 high-resolution images" in cue
    assert "low-resolution video" in cue


def test_lohi_timestamps_are_formatted_to_one_decimal():
    content = build_lohi_content("q", [12.3456, 7, 3599.96])
    texts = [c["text"] for c in content if c["type"] == "text"]
    assert texts[1:4] == ["<12.3 seconds>", "<7.0 seconds>", "<3600.0 seconds>"]


def test_lohi_content_without_video_placeholder():
    content = build_lohi_content("q", [2.0], video_placeholder=False)
    assert all(c["type"] != "video" for c in content)
    assert content[0] == {"type": "text", "text": LOHI_CUE.format(k=1)}
    assert content[-1] == {"type": "text", "text": "q"}


def test_lohi_content_with_no_images_keeps_video_cue_and_question():
    content = build_lohi_content("only q", [])
    assert content == [
        {"type": "video", "video": ""},
        {"type": "text", "text": LOHI_CUE.format(k=0)},
        {"type": "text", "text": "only q"},
    ]


def test_lohi_content_blocks_are_independent_objects():
    # Each block is its own dict, so a caller filling in one image does not alter the others.
    content = build_lohi_content("q", [0.0, 1.0])
    images = [c for c in content if c["type"] == "image"]
    images[0]["image"] = "filled"
    assert images[1]["image"] == ""
