#!/usr/bin/env python3
"""Convert the official Video-MME release into the layout the `videomme` task reads.

The task (`longvideo_eval/tasks/videomme.yaml`) expects, under `$VIDEOMME_DATA_ROOT`:

    qa.jsonl     one JSON object per question
    videos/      <video_id>.mp4

The `lmms-lab/Video-MME` dataset on Hugging Face ships a parquet QA table (whose `videoID`
column names the video file, and whose options carry an "A. " letter prefix) plus zipped
videos and subtitles. Unzip the archives, then run:

    python scripts/prepare_videomme.py \\
        --parquet  /data/Video-MME/videomme/test-00000-of-00001.parquet \\
        --videos   /data/Video-MME/data \\
        --subtitles /data/Video-MME/subtitle \\
        --out      /data/videomme_root

    export VIDEOMME_DATA_ROOT=/data/videomme_root

`videos/` is created as a symlink to `--videos`, so nothing is copied. `--subtitles` is
optional; when given, each row whose `<videoID>.srt` exists gets a `subtitle_path`, which the
loader reads only when a run enables the subtitle channel.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_OPTION_PREFIX = re.compile(r"^\s*[A-Z][.)]\s*")
_PASSTHROUGH = ("duration", "domain", "sub_category", "task_type")


def convert_row(row: dict, subtitle_dir: Path | None) -> dict:
    video_id = str(row["videoID"])
    out = {
        "video_id": video_id,
        "question_id": str(row["question_id"]),
        "question": str(row["question"]),
        "options": [_OPTION_PREFIX.sub("", str(o)) for o in row["options"]],
        "answer": str(row["answer"]).strip(),
    }
    for key in _PASSTHROUGH:
        if row.get(key) is not None:
            out[key] = str(row[key])
    if subtitle_dir is not None:
        srt = subtitle_dir / f"{video_id}.srt"
        if srt.is_file():
            out["subtitle_path"] = str(srt.resolve())
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--parquet", required=True, help="the Video-MME QA parquet file")
    p.add_argument("--videos", required=True, help="directory holding <videoID>.mp4")
    p.add_argument("--subtitles", default=None, help="directory holding <videoID>.srt (optional)")
    p.add_argument("--out", required=True, help="output root (becomes VIDEOMME_DATA_ROOT)")
    args = p.parse_args(argv)

    try:
        import pandas as pd
    except ImportError:
        sys.exit("prepare_videomme.py needs pandas and pyarrow: pip install pandas pyarrow")

    videos = Path(args.videos).resolve()
    if not videos.is_dir():
        sys.exit(f"--videos is not a directory: {videos}")
    subtitle_dir = Path(args.subtitles).resolve() if args.subtitles else None
    if subtitle_dir is not None and not subtitle_dir.is_dir():
        sys.exit(f"--subtitles is not a directory: {subtitle_dir}")

    rows = pd.read_parquet(args.parquet).to_dict(orient="records")
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    link = out_root / "videos"
    if link.is_symlink():
        link.unlink()
    elif link.exists():
        sys.exit(f"{link} exists and is not a symlink; remove it or choose another --out")
    link.symlink_to(videos, target_is_directory=True)

    converted = [convert_row(r, subtitle_dir) for r in rows]
    missing = sorted({r["video_id"] for r in converted
                      if not (videos / f"{r['video_id']}.mp4").is_file()})
    with (out_root / "qa.jsonl").open("w", encoding="utf-8") as fh:
        for r in converted:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    with_subs = sum("subtitle_path" in r for r in converted)
    print(f"wrote {len(converted)} questions over {len({r['video_id'] for r in converted})} "
          f"videos to {out_root / 'qa.jsonl'} ({with_subs} with subtitles)")
    if missing:
        print(f"WARNING: {len(missing)} videos are missing under {videos}, e.g. {missing[:5]}. "
              "Runs fail loudly on a missing video; use --only-videos or --limit to avoid them.")
    print(f"export VIDEOMME_DATA_ROOT={out_root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
