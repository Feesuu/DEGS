from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol


def _read_regular_file_snapshot(path: Path) -> tuple[bytes, Any]:
    absolute = path.expanduser().absolute()
    try:
        metadata = absolute.stat()
        payload = absolute.read_bytes()
        return payload, metadata
    except OSError as exc:
        raise ValueError(f"cannot read experience file {absolute}: {exc}") from exc


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


@dataclass(frozen=True)
class ExperiencePayload:
    """Opaque guidance produced by an external algorithm for one task."""

    experience: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


class ExperienceProvider(Protocol):
    """The only interface between an external method and the benchmark runner."""

    def for_instance(self, instance_id: str) -> ExperiencePayload: ...

    def identity(self) -> Mapping[str, Any]: ...


class EmptyExperienceProvider:
    def for_instance(self, instance_id: str) -> ExperiencePayload:
        return ExperiencePayload()

    def identity(self) -> Mapping[str, Any]:
        return {"provider": "empty", "task_conditioned": False}


class FileExperienceProvider:
    """Exact lookup over the documented JSONL experience interchange."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().absolute()
        if self.path.suffix.casefold() != ".jsonl":
            raise ValueError("experience file must use the documented .jsonl format")
        payload, _ = _read_regular_file_snapshot(self.path)
        self._sha256 = hashlib.sha256(payload).hexdigest()
        self._payloads = self._load_bytes(payload, source=self.path)

    def for_instance(self, instance_id: str) -> ExperiencePayload:
        return self._payloads.get(
            str(instance_id),
            self._payloads.get("*", ExperiencePayload()),
        )

    def has_exact_instance(self, instance_id: str) -> bool:
        return str(instance_id) in self._payloads and str(instance_id) != "*"

    @property
    def exact_instance_ids(self) -> tuple[str, ...]:
        return tuple(sorted(key for key in self._payloads if key != "*"))

    @property
    def has_wildcard(self) -> bool:
        return "*" in self._payloads

    @property
    def resource_formats(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    payload.metadata["format"]
                    for payload in self._payloads.values()
                    if isinstance(payload.metadata.get("format"), str)
                }
            )
        )

    def identity(self) -> Mapping[str, Any]:
        return {
            "provider": "file_exact_id_lookup",
            "file_name": self.path.name,
            "sha256": self._sha256,
            "entry_count": len(self._payloads),
            "task_conditioned": any(key != "*" for key in self._payloads),
            "selection_logic": "none_exact_instance_id_lookup_only",
        }

    @classmethod
    def _load_bytes(
        cls,
        payload: bytes,
        *,
        source: Path,
    ) -> dict[str, ExperiencePayload]:
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"experience file is not UTF-8: {source}") from exc
        rows = []
        for line_number, line in enumerate(
            text.splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                rows.append(
                    json.loads(line, object_pairs_hook=_object_without_duplicate_keys)
                )
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {source}:{line_number}: {exc}") from exc
        return cls._rows_to_payloads(rows, source=source)

    @classmethod
    def _rows_to_payloads(
        cls,
        rows: list[Any],
        *,
        source: Path,
    ) -> dict[str, ExperiencePayload]:
        payloads: dict[str, ExperiencePayload] = {}
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError(f"experience row {index} in {source} must be an object")
            instance_id = row.get("instance_id")
            if instance_id is None:
                raise ValueError(
                    f"experience row {index} in {source} lacks instance_id"
                )
            key = str(instance_id)
            if key in payloads:
                raise ValueError(f"duplicate experience for instance {key!r}")
            value = row.get("experience")
            if not isinstance(value, str):
                raise ValueError(f"experience for {key!r} must be text")
            metadata = row.get("metadata", {})
            if not isinstance(metadata, dict):
                raise ValueError(f"metadata for {key!r} must be an object")
            normalized = value.strip()
            payloads[key] = ExperiencePayload(normalized, dict(metadata))
        return payloads
