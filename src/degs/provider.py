from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .bundle import (
    EXPERIENCE_FORMAT,
    FORMAT,
    METHOD_FAMILY,
    _VerifiedExperienceBundle,
    _RESULT_STATUSES,
)
from .core import canonical_json_bytes


@dataclass(frozen=True)
class ExperiencePayload:
    experience: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _strict_json(value: bytes, *, label: str) -> Any:
    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate key {key!r}")
            result[key] = item
        return result

    return json.loads(
        value.decode("utf-8"),
        object_pairs_hook=no_duplicates,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant {token}")
        ),
    )


class DEGSExperienceProvider:
    """Exact task lookup for a bundle returned by the strict verifier."""

    def __init__(self, verified_bundle: _VerifiedExperienceBundle) -> None:
        if type(verified_bundle) is not _VerifiedExperienceBundle:
            raise TypeError("provider requires a bundle returned by verify_from_paths")
        self._load(verified_bundle.root, verified_bundle.manifest)

    def _load(self, root: Path, verified_manifest: Mapping[str, Any]) -> None:
        self.path = root / "experience.jsonl"
        manifest_path = root / "bundle_manifest.json"
        manifest_bytes = manifest_path.read_bytes()
        manifest = _strict_json(manifest_bytes, label="bundle manifest")
        unsigned = {key: value for key, value in manifest.items() if key != "self_sha256"}
        if (
            type(manifest) is not dict
            or manifest != dict(verified_manifest)
            or manifest_bytes != canonical_json_bytes(manifest)
            or manifest.get("format") != FORMAT
            or manifest.get("method_family") != METHOD_FAMILY
            or manifest.get("fixed_denominator") != 200
            or manifest.get("experience_file") != "experience.jsonl"
            or manifest.get("self_sha256")
            != hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
        ):
            raise ValueError("verified bundle manifest identity differs")
        payload = self.path.read_bytes()
        lines = payload.splitlines()
        if (
            payload != b"".join(line + b"\n" for line in lines)
            or len(lines) != 200
            or manifest.get("experience_sha256") != hashlib.sha256(payload).hexdigest()
        ):
            raise ValueError("experience file identity differs")
        tasks = manifest.get("tasks")
        if type(tasks) is not list or len(tasks) != 200:
            raise ValueError("bundle task audit differs")
        rows: dict[str, ExperiencePayload] = {}
        for index, (line, task) in enumerate(zip(lines, tasks, strict=True)):
            row = _strict_json(line, label=f"experience row {index}")
            metadata = row.get("metadata") if type(row) is dict else None
            experience = row.get("experience") if type(row) is dict else None
            instance_id = row.get("instance_id") if type(row) is dict else None
            audit = metadata.get("retrieval_audit") if type(metadata) is dict else None
            if (
                type(row) is not dict
                or set(row) != {"instance_id", "experience", "metadata"}
                or line != canonical_json_bytes(row)
                or type(instance_id) is not str
                or not instance_id
                or instance_id in rows
                or type(experience) is not str
                or type(metadata) is not dict
                or metadata.get("format") != EXPERIENCE_FORMAT
                or metadata.get("method_family") != METHOD_FAMILY
                or metadata.get("query_index") != index
                or metadata.get("dataset_index") != 200 + index
                or metadata.get("status") != task.get("status")
                or task.get("task_id") != instance_id
                or task.get("row_sha256") != hashlib.sha256(canonical_json_bytes(row)).hexdigest()
                or metadata.get("experience_sha256") != hashlib.sha256(experience.encode()).hexdigest()
                or type(audit) is not dict
                or metadata.get("retrieval_audit_sha256")
                != hashlib.sha256(canonical_json_bytes(audit)).hexdigest()
                or metadata.get("status") not in _RESULT_STATUSES
                or not experience
            ):
                raise ValueError(f"experience row {index} differs")
            rows[instance_id] = ExperiencePayload(experience, metadata)
        self._payloads = rows
        self._sha256 = hashlib.sha256(payload).hexdigest()
        self._bundle_self_sha256 = manifest["self_sha256"]

    def for_instance(self, instance_id: str) -> ExperiencePayload:
        try:
            return self._payloads[str(instance_id)]
        except KeyError as exc:
            raise KeyError(f"experience is absent for task {instance_id}") from exc

    def identity(self) -> Mapping[str, Any]:
        return {
            "provider": "experience_simgrag_exact_lookup",
            "task_conditioned": True,
            "path": str(self.path),
            "sha256": self._sha256,
            "bundle_self_sha256": self._bundle_self_sha256,
            "row_count": len(self._payloads),
        }


__all__ = ["DEGSExperienceProvider", "ExperiencePayload"]
