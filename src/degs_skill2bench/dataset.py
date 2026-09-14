from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contract import TEST_SHA256, TRAIN_SHA256


def load_split(path: Path | str, *, split: str) -> tuple[dict[str, Any], ...]:
    source = Path(path).expanduser().resolve()
    expected_sha = {"train": TRAIN_SHA256, "test": TEST_SHA256}.get(split)
    if expected_sha is None:
        raise ValueError("Skill2Bench split differs")
    if hashlib.sha256(source.read_bytes()).hexdigest() != expected_sha:
        raise ValueError(f"Skill2Bench {split} split hash differs")
    rows = tuple(json.loads(line) for line in source.read_text().splitlines() if line)
    expected_count = 100 if split == "train" else 200
    if len(rows) != expected_count or any(type(row) is not dict for row in rows):
        raise ValueError(f"Skill2Bench {split} population differs")
    ids = [str(row.get("instance_id") or "") for row in rows]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError(f"Skill2Bench {split} task identity differs")
    return rows


def public_task_view(task: Mapping[str, Any]) -> dict[str, Any]:
    questions = task.get("questions")
    if isinstance(questions, Sequence) and not isinstance(questions, (str, bytes)):
        instance_id = str(task.get("instance_id") or "")
        if not instance_id:
            raise ValueError("Skill2Bench task identity differs")
        return {
            "instance_id": instance_id,
            "scenario": str(task.get("scenario") or ""),
            "questions": [str(question) for question in questions],
        }
    steps = task.get("steps")
    if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
        raise ValueError("Skill2Bench task steps differ")
    instance_id = str(task.get("instance_id") or "")
    if not instance_id:
        raise ValueError("Skill2Bench task identity differs")
    return {
        "instance_id": instance_id,
        "scenario": str(task.get("scenario") or ""),
        "questions": [str(step.get("question") or "") for step in steps],
    }


__all__ = ["load_split", "public_task_view"]
