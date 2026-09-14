"""Pinned train-query authority with no answer/gold fields in its public output."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any

TRAIN_COUNT = 200
DEVELOPMENT_START = 200
DEVELOPMENT_END = 400
DEVELOPMENT_COUNT = DEVELOPMENT_END - DEVELOPMENT_START
DATASET_SHA256 = (
    "bcecaa89a005bd4e3bbe98da150a86e8062c27f262e575d5e47bd9861b3525e7"
)
DATASET_PREFIX_BYTES = 160126
DATASET_PREFIX_SHA256 = (
    "f7e1569378ec2df807e643dde425c3c7f1ef11719ab5c04b1f0c74d90a0eecc7"
)
EXPECTED_QUERY_PROJECTION_SHA256 = (
    "a4f07c48655999fc5b7628fe7dfb1e5b8c72e574964100b031d8737fb0efc538"
)
EXPECTED_DEVELOPMENT_QUERY_PROJECTION_SHA256 = (
    "4c30aca5f39ec29b1b1cae5ec5ccfc519297cc1284a86976f1c5de269710bf44"
)
_QUERY_FIELDS = {"train_index", "task_id", "instruction"}
_DEVELOPMENT_QUERY_FIELDS = {
    "development_index",
    "dataset_index",
    "task_id",
    "instruction",
}
_SAFE_TASK_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def validate_train_queries(value: Any) -> list[dict[str, Any]]:
    if type(value) is not list or len(value) != TRAIN_COUNT:
        raise ValueError("train query population differs")
    task_ids: set[str] = set()
    for index, row in enumerate(value):
        if type(row) is not dict or set(row) != _QUERY_FIELDS:
            raise ValueError("train query fields differ")
        if row["train_index"] != index or type(row["train_index"]) is not int:
            raise ValueError("train query index differs")
        task_id = row["task_id"]
        instruction = row["instruction"]
        if (
            type(task_id) is not str
            or _SAFE_TASK_ID.fullmatch(task_id) is None
            or task_id in task_ids
        ):
            raise ValueError("train query task identity differs")
        if (
            type(instruction) is not str
            or not instruction.strip()
            or instruction != instruction.strip()
            or len(instruction) > 8000
            or any(
                ord(character) < 32 and character not in "\t\n\r"
                for character in instruction
            )
        ):
            raise ValueError("train query instruction differs")
        task_ids.add(task_id)
    return value


def query_projection_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(validate_train_queries(value))).hexdigest()


def validate_development_queries(value: Any) -> list[dict[str, Any]]:
    if type(value) is not list or len(value) != DEVELOPMENT_COUNT:
        raise ValueError("development query population differs")
    task_ids: set[str] = set()
    for development_index, row in enumerate(value):
        if type(row) is not dict or set(row) != _DEVELOPMENT_QUERY_FIELDS:
            raise ValueError("development query fields differ")
        if (
            row["development_index"] != development_index
            or row["dataset_index"] != DEVELOPMENT_START + development_index
            or type(row["development_index"]) is not int
            or type(row["dataset_index"]) is not int
        ):
            raise ValueError("development query index differs")
        task_id = row["task_id"]
        instruction = row["instruction"]
        if (
            type(task_id) is not str
            or _SAFE_TASK_ID.fullmatch(task_id) is None
            or task_id in task_ids
        ):
            raise ValueError("development query task identity differs")
        if (
            type(instruction) is not str
            or not instruction.strip()
            or instruction != instruction.strip()
            or len(instruction) > 8000
            or any(
                ord(character) < 32 and character not in "\t\n\r"
                for character in instruction
            )
        ):
            raise ValueError("development query instruction differs")
        task_ids.add(task_id)
    return value


def development_query_projection_sha256(value: Any) -> str:
    return hashlib.sha256(
        _canonical_json_bytes(validate_development_queries(value))
    ).hexdigest()


def _read_pinned_prefix(path: Path) -> bytes:
    try:
        value = path.expanduser().read_bytes()[:DATASET_PREFIX_BYTES]
    except OSError as exc:
        raise ValueError("dataset is not readable") from exc
    if len(value) != DATASET_PREFIX_BYTES:
        raise ValueError("pinned dataset prefix is unavailable")
    if hashlib.sha256(value).hexdigest() != DATASET_PREFIX_SHA256:
        raise ValueError("pinned dataset prefix identity differs")
    return value


def _read_pinned_dataset(path: Path) -> bytes:
    try:
        value = path.expanduser().read_bytes()
    except OSError as exc:
        raise ValueError("dataset is not readable") from exc
    if hashlib.sha256(value).hexdigest() != DATASET_SHA256:
        raise ValueError("pinned dataset identity differs")
    return value


def _dataset_rows(path: Path) -> list[dict[str, Any]]:
    try:
        rows = json.loads(_read_pinned_dataset(path).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("pinned dataset is not valid JSON") from exc
    if (
        type(rows) is not list
        or len(rows) != DEVELOPMENT_END
        or any(type(row) is not dict for row in rows)
    ):
        raise ValueError("pinned dataset population differs")
    return rows


def _prefix_rows(value: bytes) -> list[dict[str, Any]]:
    try:
        rows = json.loads(value.decode("utf-8") + "\n]")
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("pinned dataset prefix is not the expected JSON fragment") from exc
    if type(rows) is not list or len(rows) != TRAIN_COUNT or any(type(row) is not dict for row in rows):
        raise ValueError("pinned dataset prefix population differs")
    return rows


def load_train_queries(dataset_path: Path) -> list[dict[str, Any]]:
    """Read the pinned train prefix and expose only index, ID, and instruction."""

    source_rows = _prefix_rows(_read_pinned_prefix(Path(dataset_path)))
    queries = [
        {
            "train_index": train_index,
            "task_id": str(row["id"]),
            "instruction": row["instruction"],
        }
        for train_index, row in enumerate(source_rows)
    ]
    if query_projection_sha256(queries) != EXPECTED_QUERY_PROJECTION_SHA256:
        raise ValueError("frozen query projection identity differs")
    return queries


def load_development_queries(dataset_path: Path) -> list[dict[str, Any]]:
    """Expose only query fields from fixed development dataset rows [200,400)."""

    rows = _dataset_rows(Path(dataset_path))[DEVELOPMENT_START:DEVELOPMENT_END]
    queries = [
        {
            "development_index": development_index,
            "dataset_index": DEVELOPMENT_START + development_index,
            "task_id": str(row["id"]),
            "instruction": row["instruction"],
        }
        for development_index, row in enumerate(rows)
    ]
    if (
        development_query_projection_sha256(queries)
        != EXPECTED_DEVELOPMENT_QUERY_PROJECTION_SHA256
    ):
        raise ValueError("frozen development query projection identity differs")
    return queries


def _load_train_harness_records(dataset_path: Path) -> list[dict[str, Any]]:
    """Project only the fixed Agent-facing fields from the pinned train prefix."""

    source_rows = _prefix_rows(_read_pinned_prefix(Path(dataset_path)))
    records: list[dict[str, Any]] = []
    for train_index, row in enumerate(source_rows):
        task_id = str(row["id"])
        instruction = row["instruction"]
        spreadsheet_path = row.get("spreadsheet_path", task_id)
        instruction_type = row.get("instruction_type", "")
        answer_position = row.get("answer_position", "")
        if any(
            type(value) is not str
            for value in (
                instruction,
                spreadsheet_path,
                instruction_type,
                answer_position,
            )
        ):
            raise ValueError(f"train harness row {train_index} fields differ")
        records.append(
            {
                "train_index": train_index,
                "task_id": task_id,
                "instruction": instruction,
                "spreadsheet_path": spreadsheet_path,
                "instruction_type": instruction_type,
                "answer_position": answer_position,
            }
        )
    queries = [
        {
            "train_index": row["train_index"],
            "task_id": row["task_id"],
            "instruction": row["instruction"],
        }
        for row in records
    ]
    if query_projection_sha256(queries) != EXPECTED_QUERY_PROJECTION_SHA256:
        raise ValueError("frozen train harness projection identity differs")
    return records


def _load_development_harness_records(dataset_path: Path) -> list[dict[str, Any]]:
    """Project Agent-facing fields from fixed development rows [200,400)."""

    rows = _dataset_rows(Path(dataset_path))[DEVELOPMENT_START:DEVELOPMENT_END]
    records: list[dict[str, Any]] = []
    for development_index, row in enumerate(rows):
        task_id = str(row["id"])
        instruction = row["instruction"]
        spreadsheet_path = row.get("spreadsheet_path", task_id)
        instruction_type = row.get("instruction_type", "")
        answer_position = row.get("answer_position", "")
        if any(
            type(value) is not str
            for value in (
                instruction,
                spreadsheet_path,
                instruction_type,
                answer_position,
            )
        ):
            raise ValueError(
                f"development harness row {development_index} fields differ"
            )
        records.append(
            {
                "development_index": development_index,
                "dataset_index": DEVELOPMENT_START + development_index,
                "task_id": task_id,
                "instruction": instruction,
                "spreadsheet_path": spreadsheet_path,
                "instruction_type": instruction_type,
                "answer_position": answer_position,
            }
        )
    queries = [
        {
            "development_index": row["development_index"],
            "dataset_index": row["dataset_index"],
            "task_id": row["task_id"],
            "instruction": row["instruction"],
        }
        for row in records
    ]
    if (
        development_query_projection_sha256(queries)
        != EXPECTED_DEVELOPMENT_QUERY_PROJECTION_SHA256
    ):
        raise ValueError("frozen development harness projection identity differs")
    return records


__all__ = [
    "DATASET_SHA256",
    "DEVELOPMENT_COUNT",
    "DEVELOPMENT_END",
    "DEVELOPMENT_START",
    "EXPECTED_DEVELOPMENT_QUERY_PROJECTION_SHA256",
    "EXPECTED_QUERY_PROJECTION_SHA256",
    "DATASET_PREFIX_BYTES",
    "DATASET_PREFIX_SHA256",
    "TRAIN_COUNT",
    "development_query_projection_sha256",
    "load_development_queries",
    "load_train_queries",
    "query_projection_sha256",
    "validate_development_queries",
    "validate_train_queries",
]
