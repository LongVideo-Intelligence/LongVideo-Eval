"""Video-MME loader tests — synthetic-slice pattern, hardware-free.

Covers the per-category sidecars (`VideoMME.categories` / `.domains`) exposed when the QA rows
carry `task_type`/`domain`, and left EMPTY (non-breaking) when a QA file omits them.
"""
import json

from longvideo_eval.data.videomme import VideoMME

_OPTS = ["cooking", "travel", "sports", "news"]


def _make_dataset(tmp_path, rows):
    """Write QA rows + one empty video file per distinct video_id; return the videos root."""
    videos = tmp_path / "videos"
    videos.mkdir(exist_ok=True)
    for vid in {r["video_id"] for r in rows}:
        (videos / f"{vid}.mp4").touch()  # FakeDecoder never opens these; existence is enough
    qa = tmp_path / "qa.jsonl"
    qa.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return videos, qa


def _row(vid, qid, answer="A", question="What happens?", task_type=None, domain=None):
    row = {"video_id": vid, "question_id": qid, "question": question,
           "options": list(_OPTS), "answer": answer}
    if task_type is not None:
        row["task_type"] = task_type
    if domain is not None:
        row["domain"] = domain
    return row


def test_videomme_exposes_category_and_domain_sidecars(tmp_path):
    videos, qa = _make_dataset(tmp_path, [
        _row("v0", "v0-0", answer="A", task_type="Temporal Perception", domain="Knowledge"),
        _row("v1", "v1-0", answer="C", task_type="Action Reasoning", domain="Film & Television"),
    ])
    ds = VideoMME(video_root=str(videos), qa_path=str(qa), use_subtitles=False)
    samples = list(ds)

    assert [s.key for s in samples] == ["v0-0", "v1-0"]
    assert ds.golds == {"v0-0": "A", "v1-0": "C"}
    # per-category sidecar keyed identically to golds (verbatim, unvalidated string)
    assert ds.categories == {"v0-0": "Temporal Perception", "v1-0": "Action Reasoning"}
    assert ds.domains == {"v0-0": "Knowledge", "v1-0": "Film & Television"}


def test_videomme_category_sidecar_empty_without_task_type(tmp_path):
    """Older 300-QA slices ship no task_type: the sidecars stay empty and the run still works
    (non-breaking) — this is why the field is optional."""
    videos, qa = _make_dataset(tmp_path, [
        _row("v0", "v0-0", answer="A"),   # no task_type / domain
        _row("v0", "v0-1", answer="B"),
    ])
    ds = VideoMME(video_root=str(videos), qa_path=str(qa), use_subtitles=False)
    samples = list(ds)

    assert [s.key for s in samples] == ["v0-0", "v0-1"]
    assert ds.golds == {"v0-0": "A", "v0-1": "B"}
    assert ds.categories == {}
    assert ds.domains == {}


def test_videomme_partial_category_coverage(tmp_path):
    """A mixed file (some rows tagged, some not) keeps only the tagged entries — no None keys."""
    videos, qa = _make_dataset(tmp_path, [
        _row("v0", "v0-0", task_type="Object Reasoning"),
        _row("v1", "v1-0"),   # untagged
    ])
    ds = VideoMME(video_root=str(videos), qa_path=str(qa), use_subtitles=False)
    list(ds)
    assert ds.categories == {"v0-0": "Object Reasoning"}
    assert "v1-0" not in ds.categories
