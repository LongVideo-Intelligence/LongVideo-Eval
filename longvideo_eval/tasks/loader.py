"""Task YAML loader — lmms-eval-style task definitions -> a runnable `Task`.

We mirror lmms-eval's task-file convention (a YAML per task holding the dataset kwargs, the
`doc_to_text` prompt template, and the metric) but keep our OWN runtime — nothing here imports
lmms_eval. Keeping tasks declarative (YAML, not code) is what lets an upstream task be ported
near-mechanically.

Fail-loud: an unknown task file or an unknown metric raises with the valid options named, so a
typo never silently scores nothing.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

from ..interfaces import Sample

# Metrics the runner knows how to score. A task YAML naming anything else is rejected up front.
KNOWN_METRICS = {"mcq_accuracy"}

# Option letters for multiple-choice rendering (A, B, C, D, ...).
_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

_TASKS_DIR = Path(__file__).resolve().parent


@dataclass
class Task:
    """One benchmark task: how to load its data, prompt the model, and score it."""

    name: str
    dataset_loader: str          # dotted "module:Class" for the dataset loader
    dataset_kwargs: dict         # kwargs passed to the loader (env vars expanded by the CLI)
    doc_to_text: str             # prompt template with {question} and {options} fields
    metric: str                  # must be in KNOWN_METRICS
    output_type: str             # e.g. "multiple_choice"
    subtitles: bool = False      # default subtitle channel state (CLI --subtitles can force on)

    def build_prompt(self, sample: Sample) -> str:
        """Render `doc_to_text` for one sample: lettered choices + the question stem.

        The subtitle channel is fed to the backend separately (parity), NOT
        spliced into the prompt here.
        """
        choices = list(sample.choices or [])
        if len(choices) > len(_LETTERS):
            raise ValueError(
                f"sample {sample.key!r} has {len(choices)} choices; "
                f"only {len(_LETTERS)} option letters are supported"
            )
        options = "\n".join(f"{_LETTERS[i]}. {c}" for i, c in enumerate(choices))
        return self.doc_to_text.format(question=sample.query, options=options)


def available_tasks() -> list:
    return sorted(p.stem for p in _TASKS_DIR.glob("*.yaml"))


def load_task(name: str) -> Task:
    """Read `longvideo_eval/tasks/<name>.yaml` into a validated `Task`."""
    path = _TASKS_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"no task {name!r}; available tasks: {available_tasks()} (add {name}.yaml under "
            f"{_TASKS_DIR})"
        )
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"task file {path} must be a YAML mapping, got {type(data).__name__}")

    metric = data.get("metric")
    if metric not in KNOWN_METRICS:
        raise ValueError(
            f"task {name!r} metric {metric!r} is not known; valid metrics: {sorted(KNOWN_METRICS)}"
        )

    dataset = data.get("dataset")
    if not isinstance(dataset, dict) or "loader" not in dataset:
        raise ValueError(
            f"task {name!r} needs a 'dataset' mapping with a 'loader' (module:Class) field"
        )

    doc_to_text: Optional[str] = data.get("doc_to_text")
    if not isinstance(doc_to_text, str) or not doc_to_text.strip():
        raise ValueError(f"task {name!r} needs a non-empty 'doc_to_text' prompt template")

    return Task(
        name=data.get("task", name),
        dataset_loader=dataset["loader"],
        dataset_kwargs=dict(dataset.get("kwargs", {})),
        doc_to_text=doc_to_text,
        metric=metric,
        output_type=data.get("output_type", "multiple_choice"),
        subtitles=bool(data.get("subtitles", False)),
    )
