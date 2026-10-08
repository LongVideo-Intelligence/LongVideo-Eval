"""Video-MME loader (fu2025videomme) — QA file + local video root -> `Sample`s.

MCQ benchmark with an optional subtitle channel; subtitles are fed to the method AND its
baseline (parity). Follows lmms-eval's task-file convention so porting stays cheap, but the
class is our own runtime.

Fail-loud: a missing video file raises immediately — NO skipping, so a run's `n` always
reflects the full requested slice. A `.parquet` QA file needs pandas; if pandas is absent that
is an ImportError, never a silent fallback to some other format.

Gold answers are exposed via `self.golds` (`Sample.key` -> answer letter), populated during
iteration, because `Sample` (interfaces.py) intentionally carries no label. Real Video-MME has
~3 questions per video, so golds are keyed by `Sample.key` (= `question_id` when the row has
one, else `video_id`) — NOT `video_id` alone, which would collide across a video's questions.
If a QA file omits `question_id` AND reuses a `video_id` across rows, that is a genuine key
collision and raises loudly rather than silently overwriting a gold.

The Video-MME task category (`task_type`) is exposed via `self.categories` (`Sample.key` ->
task_type), a sidecar dict populated during iteration, so per-category accuracy (Video-MME's
own 12-way task taxonomy — temporal perception, action reasoning, ...) can be computed later
without touching `Sample`. This is OPTIONAL and UNVALIDATED: a row WITHOUT `task_type` simply
contributes no category entry, and the string is taken verbatim rather than checked against a
closed set (the QA file is the authority). `domain` is captured alongside in `self.domains`
for coarser rollups.

`scripts/prepare_videomme.py` converts the official `lmms-lab/Video-MME` release into the QA
file this loader reads.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List

from ..interfaces import Sample

_VIDEO_EXTS = (".mp4", ".mkv", ".webm")


@dataclass
class VideoMME:
    """Iterate a Video-MME QA file, resolving each video under `video_root`.

    QA rows (jsonl: one JSON object per line; or parquet: one row each) carry:
      video_id (str)      — resolves to a file under video_root by extension probing
      question_id (str, optional) — unique per-question id; disambiguates multi-Q/video data
      question (str)
      options (list[str]) — the MCQ choices, in order
      answer (str)        — the gold option letter
      subtitle_path (str, optional) — read only when use_subtitles is on
    """

    video_root: str
    qa_path: str
    use_subtitles: bool = True
    golds: Dict[str, str] = field(default_factory=dict, init=False, repr=False)
    categories: Dict[str, str] = field(default_factory=dict, init=False, repr=False)
    domains: Dict[str, str] = field(default_factory=dict, init=False, repr=False)

    def __iter__(self) -> Iterator[Sample]:
        self.golds = {}
        self.categories = {}
        self.domains = {}
        for row in self._read_rows():
            video_id = row["video_id"]
            question_id = row.get("question_id")
            video_path = self._resolve_video(video_id)
            subtitles = self._read_subtitles(row.get("subtitle_path"))
            sample = Sample(
                video_id=video_id,
                video_path=video_path,
                query=row["question"],
                subtitles=subtitles,
                choices=list(row["options"]),
                question_id=question_id,
            )
            if sample.key in self.golds:
                raise ValueError(
                    f"duplicate gold key {sample.key!r} (video_id={video_id!r}, "
                    f"question_id={question_id!r}) — rows collide; add a unique 'question_id' "
                    f"to disambiguate videos with more than one question"
                )
            self.golds[sample.key] = row["answer"]
            # Per-category sidecars (optional; see module docstring): absent => no entry,
            # verbatim string.
            task_type = row.get("task_type")
            if task_type is not None:
                self.categories[sample.key] = task_type
            domain = row.get("domain")
            if domain is not None:
                self.domains[sample.key] = domain
            yield sample

    # -- QA file reading ---------------------------------------------------- #
    def _read_rows(self) -> List[dict]:
        path = Path(self.qa_path)
        if not path.exists():
            raise FileNotFoundError(
                f"VideoMME QA file not found: {path} (set VIDEOMME_DATA_ROOT or fix the task "
                f"YAML dataset.kwargs.qa_path)"
            )
        if path.suffix == ".parquet":
            return self._read_parquet(path)
        return self._read_jsonl(path)

    @staticmethod
    def _read_jsonl(path: Path) -> List[dict]:
        rows: List[dict] = []
        with path.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)  # malformed line -> loud JSONDecodeError with position
                if not isinstance(obj, dict):
                    raise ValueError(f"{path}:{lineno}: expected a JSON object, got {type(obj)}")
                rows.append(obj)
        return rows

    @staticmethod
    def _read_parquet(path: Path) -> List[dict]:
        try:
            import pandas as pd
        except ImportError as exc:  # narrow: only the missing optional dep, re-raised loudly
            raise ImportError(
                f"reading {path} needs pandas; install it (`pip install pandas`) or convert the "
                f"QA file to jsonl — there is no silent fallback"
            ) from exc
        return pd.read_parquet(path).to_dict(orient="records")

    # -- resolution helpers ------------------------------------------------- #
    def _resolve_video(self, video_id: str) -> str:
        root = Path(self.video_root)
        candidates: List[Path] = []
        # Honor an explicit extension on the id first, then probe the common ones.
        if Path(video_id).suffix:
            candidates.append(root / video_id)
        candidates += [root / f"{video_id}{ext}" for ext in _VIDEO_EXTS]
        for cand in candidates:
            if cand.exists():
                return str(cand)
        raise FileNotFoundError(
            f"no video for id {video_id!r} under {root} (tried {[str(c) for c in candidates]}); "
            f"missing videos are not skipped"
        )

    def _read_subtitles(self, subtitle_path):
        if not self.use_subtitles or not subtitle_path:
            return None
        path = Path(subtitle_path)
        if not path.is_absolute():
            path = Path(self.video_root) / path
        if not path.exists():
            raise FileNotFoundError(
                f"subtitles enabled but subtitle file not found: {path}"
            )
        return path.read_text(encoding="utf-8")
