from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any, Callable, Collection, Mapping, Sequence

from openai import APIError
from react_agent.models import (
    OpenAIClient,
    RequestCompletionLengthExceeded,
    RequestContextLengthExceeded,
    RequestRuntimeTimeout,
)
from sb_adapter.transport import validate_service_url

from .core import TRAIN_INSTRUCTION_COUNT, canonical_json_bytes
from .section_graph import SECTION_GRAPH_FORMAT, SOURCE_SPLIT
from .source_replay import validate_source_replay_outcome_protocol
from .source_review import (
    SOURCE_REVIEW_PROMPT_SHA256,
    SOURCE_REVIEW_PROTOCOL_FORMAT,
    ExperienceSourceReviewer,
    SourceReviewResult,
    openai_source_review_llm,
    parse_source_review_response,
    source_review_payload,
    source_review_response_schema,
)
from .successful_source import (
    SUCCESS_EXTRACTION_KIND,
    SUCCESS_PROMPT_SHA256,
    SUCCESS_SOURCE_PROTOCOL_FORMAT,
    SuccessfulTrajectoryExtraction,
    SuccessfulTrajectoryExperienceExtractor,
    openai_success_llm,
)
from .validated_repair import (
    OpenAIJsonObjectLLM,
    REPAIR_PROMPT_SHA256,
    REPAIR_SOURCE_MODEL,
    REPAIR_SOURCE_TIMEOUT_SECONDS,
    PRODUCER_RUNTIME_TIMEOUT_RETRIES,
    PRODUCER_TRANSPORT_RETRY_WAITS,
    ProducerTransportGuard,
    SystemicProducerTransportFailure,
    gather_cancel_on_error,
    producer_transport_failure_policy,
    REPAIR_EXTRACTION_KIND,
    SOURCE_RAW_RESPONSE_FORMAT,
    REPAIR_SOURCE_PROTOCOL_FORMAT,
    ValidatedRepairExtraction,
    ValidatedRepairExperienceExtractor,
    _read_bound_replay_trajectory,
    _source_generation_config,
    _strict_json_object,
    _write_json_output,
    _parse_llm_experience_graph,
    render_successful_replay,
    select_validated_repair_example,
    repair_response_schema,
)


ReplayTrajectoryLoader = Callable[[str], Mapping[str, Any]]
SourceExtractorFactory = Callable[
    [str, Path],
    SuccessfulTrajectoryExperienceExtractor | ValidatedRepairExperienceExtractor,
]
SourceReviewerFactory = Callable[[Path], ExperienceSourceReviewer]
SourceProgressCallback = Callable[[Mapping[str, Any]], None]
SOURCE_REBUILD_WORKERS = 16
INCREMENTAL_BATCH_SIZE = 8
SOURCE_EXTRACTION_SEMANTIC_ATTEMPTS = 3
SOURCE_EXTRACTION_CHECKPOINT_FORMAT = "degs_source_extraction_checkpoint_v4"
SOURCE_REVIEW_CHECKPOINT_FORMAT = "degs_source_review_checkpoint_v6"
SOURCE_REBUILD_AUDIT_FORMAT = "degs_reviewed_source_audit_v14"
SOURCE_REVIEW_RETRY_STATUS = "REVIEW_RETRY_EXHAUSTED_DRAFT_FALLBACK"
SOURCE_TERMINAL_EXCLUSION_CHECKPOINT_FORMAT = (
    "degs_source_terminal_exclusion_checkpoint_v1"
)
SOURCE_SYSTEMIC_TRANSPORT_ARCHIVE_FORMAT = (
    "degs_source_systemic_transport_archive_v2"
)


def _producer_cache_identity(protocol: Mapping[str, Any]) -> dict[str, Any]:
    operational = {
        "service_url",
        "timeout_seconds",
        "retry_waits_seconds",
        "runtime_timeout_retries",
    }
    return {key: value for key, value in protocol.items() if key not in operational}


def _same_producer_cache_identity(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> bool:
    return _producer_cache_identity(left) == _producer_cache_identity(right)


_REPLAY_TERMINAL_STATUSES = {
    "REPLAY_VALIDATED_SUCCESS",
    "REPLAY_EXHAUSTED",
    "REPLAY_RUNTIME_FAILURE",
}
_TASK_CHECKPOINT_DIRECTORY_RE = re.compile(r"[0-9]{3}_[0-9a-f]{12}\Z")
_SOURCE_TRANSPORT_PATTERNS = {
    "source extraction": ("raw_response_attempt_*.json",),
    "source review": (
        "raw_review_response_attempt_*.json",
        "raw_review_retry_response_attempt_*.json",
    ),
}
_SOURCE_TRANSPORT_ATTEMPT_PATTERNS = {
    "source extraction": (
        re.compile(r"(raw_response_attempt_)([0-9]{2})\.json\Z"),
    ),
    "source review": (
        re.compile(r"(raw_review_response_attempt_)([0-9]{2})\.json\Z"),
        re.compile(r"(raw_review_retry_response_attempt_)([0-9]{2})\.json\Z"),
    ),
}


@dataclass(frozen=True)
class ExperienceSourceBuild:
    section_graphs: dict[str, Any]
    source_audit: tuple[dict[str, Any], ...]
    source_exclusions: tuple[dict[str, Any], ...] = ()


def _context_length_exclusion(
    *,
    train_index: int,
    original: Mapping[str, Any],
    origin: str,
    error: RequestContextLengthExceeded,
) -> dict[str, Any]:
    return {
        "train_index": train_index,
        "task_id": original["task_id"],
        "trajectory_id": original["trajectory_id"],
        "origin": origin,
        "status": "SOURCE_EXCLUDED_CONTEXT_LENGTH",
        "error": str(error),
    }


def _generation_failure_exclusion(
    *,
    train_index: int,
    original: Mapping[str, Any],
    origin: str,
    invalid_response_attempts: Sequence[str],
) -> dict[str, Any]:
    if (
        len(invalid_response_attempts) != SOURCE_EXTRACTION_SEMANTIC_ATTEMPTS
        or any(
            type(reason) is not str or not reason
            for reason in invalid_response_attempts
        )
    ):
        raise ValueError("generation failure evidence differs")
    return {
        "train_index": train_index,
        "task_id": original["task_id"],
        "trajectory_id": original["trajectory_id"],
        "origin": origin,
        "status": "SOURCE_EXCLUDED_GENERATION_FAILURE",
        "semantic_attempt_count": SOURCE_EXTRACTION_SEMANTIC_ATTEMPTS,
        "invalid_response_attempts": list(invalid_response_attempts),
    }


def _no_reusable_experience_exclusion(
    *,
    train_index: int,
    original: Mapping[str, Any],
    origin: str,
) -> dict[str, Any]:
    return {
        "train_index": train_index,
        "task_id": original["task_id"],
        "trajectory_id": original["trajectory_id"],
        "origin": origin,
        "status": "SOURCE_EXCLUDED_NO_REUSABLE_EXPERIENCE",
    }


def _write_terminal_exclusion_checkpoint(
    path: Path,
    *,
    train_index: int,
    original: Mapping[str, Any],
    origin: str,
    protocol: Mapping[str, Any],
    request_payload_sha256: str,
    exclusion: Mapping[str, Any],
) -> None:
    _write_json_output(
        path,
        _terminal_exclusion_checkpoint_payload(
            train_index=train_index,
            original=original,
            origin=origin,
            protocol=protocol,
            request_payload_sha256=request_payload_sha256,
            exclusion=exclusion,
        ),
    )


def _terminal_exclusion_checkpoint_payload(
    *,
    train_index: int,
    original: Mapping[str, Any],
    origin: str,
    protocol: Mapping[str, Any],
    request_payload_sha256: str,
    exclusion: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "format": SOURCE_TERMINAL_EXCLUSION_CHECKPOINT_FORMAT,
        "train_index": train_index,
        "task_id": original["task_id"],
        "trajectory_id": original["trajectory_id"],
        "origin": origin,
        "source_protocol": dict(protocol),
        "source_protocol_sha256": hashlib.sha256(
            canonical_json_bytes(protocol)
        ).hexdigest(),
        "request_payload_sha256": request_payload_sha256,
        "exclusion": dict(exclusion),
    }


def _load_terminal_exclusion_checkpoint(
    path: Path,
    *,
    train_index: int,
    original: Mapping[str, Any],
    origin: str,
    expected_protocol: Mapping[str, Any],
    expected_request_payload_sha256: str,
) -> dict[str, Any]:
    payload = _strict_json_object_file(path, label="source terminal exclusion checkpoint")
    exclusion = payload.get("exclusion")
    protocol = payload.get("source_protocol")
    expected_exclusion: dict[str, Any] | None = None
    if type(exclusion) is dict:
        if exclusion.get("status") == "SOURCE_EXCLUDED_GENERATION_FAILURE":
            expected_exclusion = _generation_failure_exclusion(
                train_index=train_index,
                original=original,
                origin=origin,
                invalid_response_attempts=exclusion.get(
                    "invalid_response_attempts", ()
                ),
            )
        elif exclusion.get("status") == (
            "SOURCE_EXCLUDED_NO_REUSABLE_EXPERIENCE"
        ):
            expected_exclusion = _no_reusable_experience_exclusion(
                train_index=train_index,
                original=original,
                origin=origin,
            )
        elif exclusion.get("status") == "SOURCE_EXCLUDED_CONTEXT_LENGTH":
            error = exclusion.get("error")
            if type(error) is str and error:
                expected_exclusion = {
                    "train_index": train_index,
                    "task_id": original["task_id"],
                    "trajectory_id": original["trajectory_id"],
                    "origin": origin,
                    "status": "SOURCE_EXCLUDED_CONTEXT_LENGTH",
                    "error": error,
                }
    if (
        set(payload)
        != {
            "format",
            "train_index",
            "task_id",
            "trajectory_id",
            "origin",
            "source_protocol",
            "source_protocol_sha256",
            "request_payload_sha256",
            "exclusion",
        }
        or payload.get("format") != SOURCE_TERMINAL_EXCLUSION_CHECKPOINT_FORMAT
        or payload.get("train_index") != train_index
        or payload.get("task_id") != original["task_id"]
        or payload.get("trajectory_id") != original["trajectory_id"]
        or payload.get("origin") != origin
        or type(protocol) is not dict
        or not _same_producer_cache_identity(protocol, expected_protocol)
        or payload.get("source_protocol_sha256")
        != hashlib.sha256(canonical_json_bytes(protocol)).hexdigest()
        or payload.get("request_payload_sha256")
        != expected_request_payload_sha256
        or type(exclusion) is not dict
        or expected_exclusion is None
        or exclusion != expected_exclusion
    ):
        raise ValueError("source terminal exclusion checkpoint identity differs")
    return exclusion


def _no_validated_success_exclusion(
    *,
    train_index: int,
    original: Mapping[str, Any],
    replay_status: str,
) -> dict[str, Any]:
    if replay_status not in _REPLAY_TERMINAL_STATUSES or (
        replay_status == "REPLAY_VALIDATED_SUCCESS"
    ):
        raise ValueError("replay exclusion status differs")
    return {
        "train_index": train_index,
        "task_id": original["task_id"],
        "trajectory_id": original["trajectory_id"],
        "origin": "ORIGINAL_FAILURE",
        "status": "SOURCE_EXCLUDED_NO_VALIDATED_SUCCESS",
        "replay_terminal_status": replay_status,
    }


def _ensure_directory(path: Path) -> None:
    if path.exists() and not path.is_dir():
        raise ValueError("source extraction checkpoint directory differs")
    path.mkdir(parents=True, exist_ok=True)


def _strict_json_array(path: Path, *, label: str) -> list[Mapping[str, Any]]:
    def without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=without_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {token}")
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} is not readable strict JSON") from exc
    if type(value) is not list or any(not isinstance(row, Mapping) for row in value):
        raise ValueError(f"{label} must be an array of objects")
    return value


def _strict_json_object_file(path: Path, *, label: str) -> dict[str, Any]:
    def without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=without_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {token}")
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} is not readable strict JSON") from exc
    if type(value) is not dict:
        raise ValueError(f"{label} must be an object")
    return value


def _replace_output(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically replace one checkpoint after a review retry."""

    output = Path(path).expanduser().absolute()
    if not output.is_file():
        raise FileNotFoundError("review retry requires an existing regular checkpoint")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.retry.", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        remaining = memoryview(canonical_json_bytes(payload))
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("review retry checkpoint write made no progress")
            remaining = remaining[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.replace(temporary, output)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _write_or_verify_output(
    path: Path, payload: Mapping[str, Any]
) -> None:
    """Finish split publication idempotently after a process interruption."""

    output = Path(path).expanduser().absolute()
    expected = canonical_json_bytes(payload)
    if output.exists():
        if not output.is_file() or output.read_bytes() != expected:
            raise ValueError("existing source output differs")
        return
    _write_json_output(output, payload)


def _validate_directory(path: Path, *, label: str) -> None:
    if not path.is_dir():
        raise ValueError(f"{label} differs")


def _validate_regular_file(path: Path, *, label: str) -> None:
    if not path.is_file():
        raise ValueError(f"{label} differs")


def _source_transport_failure_paths(
    checkpoint_root: Path,
    *,
    stage: str,
) -> tuple[Path, ...]:
    if stage not in _SOURCE_TRANSPORT_PATTERNS:
        raise ValueError("source systemic transport stage differs")
    paths: list[Path] = []
    for pattern in _SOURCE_TRANSPORT_PATTERNS[stage]:
        for path in sorted(checkpoint_root.glob(f"*/{pattern}")):
            if (
                path.parent.parent != checkpoint_root
                or not _TASK_CHECKPOINT_DIRECTORY_RE.fullmatch(path.parent.name)
            ):
                raise ValueError("source systemic transport path differs")
            payload = _strict_json_object_file(
                path, label="source systemic transport response"
            )
            if payload.get("outcome") == "TRANSPORT_EXHAUSTED":
                _validate_directory(
                    path.parent, label="source systemic transport path"
                )
                _validate_regular_file(
                    path, label="source systemic transport path"
                )
                paths.append(path)
    return tuple(sorted(paths))


def _source_transport_attempt_identity(
    path: Path, *, stage: str
) -> tuple[str, int]:
    patterns = _SOURCE_TRANSPORT_ATTEMPT_PATTERNS.get(stage)
    if patterns is None:
        raise ValueError("source systemic transport stage differs")
    for pattern in patterns:
        match = pattern.fullmatch(path.name)
        if match is not None:
            return match.group(1), int(match.group(2))
    raise ValueError("source systemic transport path differs")


def _source_systemic_transport_files(
    checkpoint_root: Path,
    *,
    stage: str,
    current_transport_paths: Sequence[Path],
) -> tuple[tuple[Path, ...], tuple[str, ...]]:
    if stage not in _SOURCE_TRANSPORT_PATTERNS:
        raise ValueError("source systemic transport stage differs")
    raw_files: set[Path] = set()
    parents: set[Path] = set()
    request_ids: set[str] = set()
    affected_prefixes: dict[tuple[Path, str], int] = {}
    for path in sorted(set(current_transport_paths)):
        if path.parent.parent != checkpoint_root:
            raise ValueError("source systemic transport path differs")
        payload = _strict_json_object_file(
            path, label="source systemic transport response"
        )
        request_id = payload.get("request_id")
        if (
            payload.get("outcome") != "TRANSPORT_EXHAUSTED"
            or type(request_id) is not str
            or not request_id
        ):
            raise ValueError("source systemic transport response differs")
        prefix, index = _source_transport_attempt_identity(path, stage=stage)
        key = (path.parent, prefix)
        affected_prefixes[key] = min(index, affected_prefixes.get(key, index))
        parents.add(path.parent)
        request_ids.add(request_id)
    for (parent, prefix), first_affected_index in affected_prefixes.items():
        for path in sorted(parent.glob(f"{prefix}*.json")):
            suffix = path.name.removeprefix(prefix).removesuffix(".json")
            if not suffix.isdigit() or len(suffix) != 2:
                raise ValueError("source systemic transport path differs")
            if int(suffix) >= first_affected_index:
                _validate_regular_file(
                    path, label="source systemic transport path"
                )
                raw_files.add(path)
    terminal_name = (
        "terminal_exclusion.json"
        if stage == "source extraction"
        else "review.json"
    )
    terminal_files: list[Path] = []
    for parent in sorted(parents):
        terminal = parent / terminal_name
        if not terminal.is_file():
            continue
        payload = _strict_json_object_file(
            terminal, label="source systemic transport terminal checkpoint"
        )
        if stage == "source extraction":
            status = (
                payload.get("exclusion", {}).get("status")
                if type(payload.get("exclusion")) is dict
                else None
            )
            include = status == "SOURCE_EXCLUDED_GENERATION_FAILURE"
        else:
            audit = payload.get("review_audit")
            status = (
                audit.get("source_review_status")
                if type(audit) is dict
                else None
            )
            parent_prefixes = {
                prefix
                for candidate_parent, prefix in affected_prefixes
                if candidate_parent == parent
            }
            include = (
                status == SOURCE_REVIEW_RETRY_STATUS
                or (
                    status == "REVIEW_DRAFT_FALLBACK"
                    and "raw_review_response_attempt_" in parent_prefixes
                )
            )
        if include:
            terminal_files.append(terminal)
    return tuple(sorted({*raw_files, *terminal_files})), tuple(sorted(request_ids))


def _finish_source_systemic_transport_archive(
    checkpoint_root: Path,
    archive: Path,
) -> None:
    checkpoint_root = checkpoint_root.absolute()
    archive = archive.absolute()
    archive_root = checkpoint_root / "_systemic_transport_waves"
    _validate_directory(checkpoint_root, label="source checkpoint root")
    _validate_directory(
        archive_root, label="source systemic transport archive root"
    )
    if archive.parent != archive_root:
        raise ValueError("source systemic transport archive path differs")
    _validate_directory(
        archive, label="source systemic transport archive path"
    )
    manifest_path = archive / "manifest.json"
    _validate_regular_file(
        manifest_path, label="source systemic transport archive manifest"
    )
    manifest = _strict_json_object_file(
        manifest_path, label="source systemic transport archive"
    )
    if (
        set(manifest)
        != {
            "format",
            "status",
            "cause",
            "stage",
            "failed_request_ids",
            "producer_transport_failure_policy",
            "files",
        }
        or manifest.get("format") != SOURCE_SYSTEMIC_TRANSPORT_ARCHIVE_FORMAT
        or manifest.get("status") not in {"BUILDING", "COMMITTED"}
        or manifest.get("cause") not in {
            "SYSTEMIC_WAVE",
            "INTERRUPTED_WAVE",
        }
        or manifest.get("stage") not in {"source extraction", "source review"}
        or manifest.get("producer_transport_failure_policy")
        != producer_transport_failure_policy()
        or type(manifest.get("failed_request_ids")) is not list
        or any(
            type(request_id) is not str or not request_id
            for request_id in manifest["failed_request_ids"]
        )
        or manifest.get("failed_request_ids")
        != sorted(set(manifest["failed_request_ids"]))
        or type(manifest.get("files")) is not list
        or not manifest["files"]
    ):
        raise ValueError("source systemic transport archive differs")
    for row in manifest["files"]:
        if type(row) is not dict or set(row) != {
            "source_relative_path",
            "archive_relative_path",
            "sha256",
        }:
            raise ValueError("source systemic transport archive file differs")
        source_relative = row["source_relative_path"]
        archive_relative = row["archive_relative_path"]
        if (
            type(source_relative) is not str
            or type(archive_relative) is not str
            or type(row["sha256"]) is not str
            or len(row["sha256"]) != 64
        ):
            raise ValueError("source systemic transport archive path differs")
        source_relative_path = Path(source_relative)
        archive_relative_path = Path(archive_relative)
        if (
            source_relative_path.is_absolute()
            or archive_relative_path.is_absolute()
            or ".." in source_relative_path.parts
            or ".." in archive_relative_path.parts
            or len(source_relative_path.parts) != 2
            or not _TASK_CHECKPOINT_DIRECTORY_RE.fullmatch(
                source_relative_path.parts[0]
            )
            or archive_relative_path
            != Path("files") / source_relative_path
        ):
            raise ValueError("source systemic transport archive path differs")
        allowed_names = {
            "source extraction": (
                re.compile(r"raw_response_attempt_[0-9]{2}\.json\Z"),
                re.compile(r"terminal_exclusion\.json\Z"),
            ),
            "source review": (
                re.compile(r"raw_review_response_attempt_[0-9]{2}\.json\Z"),
                re.compile(r"raw_review_retry_response_attempt_[0-9]{2}\.json\Z"),
                re.compile(r"review\.json\Z"),
            ),
        }
        if not any(
            pattern.fullmatch(source_relative_path.name)
            for pattern in allowed_names[manifest["stage"]]
        ):
            raise ValueError("source systemic transport archive path differs")
        source = checkpoint_root / source_relative_path
        destination = archive / archive_relative_path
        archive_files_root = archive / "files"
        _ensure_directory(archive_files_root)
        _ensure_directory(destination.parent)
        _validate_directory(
            source.parent, label="source systemic transport archive path"
        )
        _validate_directory(
            archive_files_root,
            label="source systemic transport archive path",
        )
        _validate_directory(
            destination.parent, label="source systemic transport archive path"
        )
        if destination.is_file():
            _validate_regular_file(
                destination, label="source systemic transport archive path"
            )
            if hashlib.sha256(destination.read_bytes()).hexdigest() != row["sha256"]:
                raise ValueError("archived systemic transport evidence differs")
            if source.exists():
                raise ValueError("systemic transport evidence exists twice")
        elif source.is_file():
            _validate_regular_file(
                source, label="source systemic transport archive path"
            )
            if hashlib.sha256(source.read_bytes()).hexdigest() != row["sha256"]:
                raise ValueError("systemic transport evidence changed before archive")
            os.replace(source, destination)
        else:
            raise FileNotFoundError("systemic transport evidence is missing")
    if manifest["status"] == "BUILDING":
        _replace_output(
            manifest_path, {**manifest, "status": "COMMITTED"}
        )


def _recover_source_systemic_transport_archives(
    checkpoint_root: Path,
) -> None:
    archive_root = checkpoint_root / "_systemic_transport_waves"
    if not archive_root.exists():
        return
    if not archive_root.is_dir():
        raise ValueError("source systemic transport archive root differs")
    _validate_directory(
        archive_root, label="source systemic transport archive root"
    )
    for archive in sorted(archive_root.glob(".wave_*.building")):
        if not archive.is_dir():
            raise ValueError("source systemic transport archive path differs")
        _validate_directory(
            archive, label="source systemic transport archive path"
        )
        manifest_path = archive / "manifest.json"
        manifest_temps = sorted(archive.glob(".manifest.json.*"))
        for temporary in manifest_temps:
            _validate_regular_file(
                temporary,
                label="source systemic transport archive manifest temporary",
            )
        if not manifest_path.exists():
            unexpected = [
                path for path in archive.iterdir() if path not in manifest_temps
            ]
            if unexpected:
                raise ValueError(
                    "source systemic transport archive is missing its manifest"
                )
            for temporary in manifest_temps:
                temporary.unlink()
            archive.rmdir()
            continue
        for temporary in manifest_temps:
            temporary.unlink()
        _finish_source_systemic_transport_archive(checkpoint_root, archive)
        final = archive.with_name(archive.name[1:-9])
        if final.exists():
            raise FileExistsError("source systemic transport archive conflicts")
        os.replace(archive, final)


def _has_reusable_complete_response_after_transport(
    transport_path: Path,
    *,
    stage: str,
) -> bool:
    """Return whether a later saved response can finish this interrupted item."""

    prefix, transport_index = _source_transport_attempt_identity(
        transport_path, stage=stage
    )
    transport_payload = _strict_json_object_file(
        transport_path, label="interrupted source transport response"
    )
    identity_fields = (
        "request_kind",
        "request_id",
        "system_prompt_sha256",
        "payload_sha256",
        "source_protocol",
        "source_protocol_sha256",
    )
    later_paths = sorted(transport_path.parent.glob(f"{prefix}*.json"))
    for candidate in reversed(later_paths):
        _candidate_prefix, candidate_index = _source_transport_attempt_identity(
            candidate, stage=stage
        )
        if candidate_index <= transport_index:
            continue
        payload = _strict_json_object_file(
            candidate, label="interrupted source saved response"
        )
        if any(
            payload.get(field) != transport_payload.get(field)
            for field in identity_fields
        ):
            raise ValueError("interrupted source response identity differs")
        if payload.get("outcome") != "COMPLETE":
            continue
        response_text = payload.get("response")
        if type(response_text) is not str:
            raise ValueError("interrupted source response differs")
        # Recovery only determines whether a complete, identity-matching raw
        # response exists.  The normal extraction/review resume path owns the
        # context-dependent schema validation and will retry semantic failures.
        return True
    return False


def _archive_source_systemic_transport_wave(
    checkpoint_root: Path,
    stage: str,
    *,
    current_transport_paths: Sequence[Path],
    cause: str = "SYSTEMIC_WAVE",
) -> None:
    if cause not in {"SYSTEMIC_WAVE", "INTERRUPTED_WAVE"}:
        raise ValueError("source transport archive cause differs")
    files, request_ids = _source_systemic_transport_files(
        checkpoint_root,
        stage=stage,
        current_transport_paths=current_transport_paths,
    )
    if not files:
        raise RuntimeError("systemic source transport failure has no saved evidence")
    archive_root = checkpoint_root / "_systemic_transport_waves"
    _ensure_directory(archive_root)
    sequence = 1 + sum(
        1
        for path in archive_root.iterdir()
        if path.name.startswith("wave_") or path.name.startswith(".wave_")
    )
    stem = f"wave_{sequence:04d}"
    staging = archive_root / f".{stem}.building"
    final = archive_root / stem
    if staging.exists() or final.exists():
        raise FileExistsError("source systemic transport archive identity conflicts")
    staging.mkdir()
    rows = [
        {
            "source_relative_path": str(path.relative_to(checkpoint_root)),
            "archive_relative_path": str(
                Path("files") / path.relative_to(checkpoint_root)
            ),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in files
    ]
    _write_json_output(
        staging / "manifest.json",
        {
            "format": SOURCE_SYSTEMIC_TRANSPORT_ARCHIVE_FORMAT,
            "status": "BUILDING",
            "cause": cause,
            "stage": stage,
            "failed_request_ids": list(request_ids),
            "producer_transport_failure_policy": (
                producer_transport_failure_policy()
            ),
            "files": rows,
        },
    )
    _finish_source_systemic_transport_archive(checkpoint_root, staging)
    os.replace(staging, final)


def _archive_interrupted_source_transport_waves(
    checkpoint_root: Path,
) -> None:
    """Remove crash-orphaned transport attempts from active retry state."""

    for stage in ("source extraction", "source review"):
        unresolved: list[Path] = []
        for path in _source_transport_failure_paths(
            checkpoint_root, stage=stage
        ):
            parent = path.parent
            if stage == "source extraction":
                resolved = (
                    (parent / "extraction.json").is_file()
                    or (parent / "terminal_exclusion.json").is_file()
                )
            else:
                review_checkpoint = parent / "review.json"
                if not review_checkpoint.is_file():
                    resolved = False
                else:
                    review_payload = _strict_json_object_file(
                        review_checkpoint,
                        label="interrupted source review checkpoint",
                    )
                    review_audit = review_payload.get("review_audit")
                    status = (
                        review_audit.get("source_review_status")
                        if type(review_audit) is dict
                        else None
                    )
                    resolved = status in {
                        "REVIEW_ACCEPTED",
                        SOURCE_REVIEW_RETRY_STATUS,
                    } or (
                        status == "REVIEW_DRAFT_FALLBACK"
                        and not path.name.startswith(
                            "raw_review_retry_response_attempt_"
                        )
                    )
            if not resolved and _has_reusable_complete_response_after_transport(
                path,
                stage=stage,
            ):
                resolved = True
            if not resolved:
                unresolved.append(path)
        if not unresolved:
            continue
        _archive_source_systemic_transport_wave(
            checkpoint_root,
            stage,
            current_transport_paths=unresolved,
            cause="INTERRUPTED_WAVE",
        )


def _unique_by_task(
    rows: Sequence[Mapping[str, Any]], *, label: str
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError(f"{label} row must be an object")
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or not task_id or task_id in result:
            raise ValueError(f"{label} task identity differs")
        result[task_id] = row
    return result


def _accepted_attempt(outcome: Mapping[str, Any]) -> Mapping[str, Any]:
    accepted_index = outcome.get("accepted_attempt_index")
    attempts = outcome.get("attempts")
    rows = [
        row
        for row in attempts
        if isinstance(row, Mapping) and row.get("attempt_index") == accepted_index
    ] if type(attempts) is list else []
    if len(rows) != 1:
        raise ValueError("accepted replay attempt is missing or duplicated")
    return rows[0]


def _validated_source_outcomes(
    original_records: Sequence[Mapping[str, Any]],
    replay_outcomes: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    if len(original_records) != TRAIN_INSTRUCTION_COUNT:
        raise ValueError("source rebuild requires exactly train[0,200) original records")
    originals = _unique_by_task(original_records, label="original trajectory")
    if len(originals) != TRAIN_INSTRUCTION_COUNT:
        raise ValueError("original train task identity differs")
    outcomes = _unique_by_task(replay_outcomes, label="replay outcome")
    failed_task_ids = {
        row.get("task_id")
        for row in original_records
        if isinstance(row, Mapping) and row.get("success") is not True
    }
    if set(outcomes) != failed_task_ids:
        raise ValueError("replay outcomes must cover every original failure exactly once")
    return outcomes


def _extract_source_row(
    *,
    train_index: int,
    original: Mapping[str, Any],
    outcomes: Mapping[str, Mapping[str, Any]],
    replay_trajectory_loader: ReplayTrajectoryLoader,
    success_extractor: SuccessfulTrajectoryExperienceExtractor | None,
    repair_extractor: ValidatedRepairExperienceExtractor | None,
    extraction_override: Any | None = None,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    task_id = original.get("task_id")
    trajectory_id = original.get("trajectory_id")
    query_text = original.get("instruction")
    if (
        not isinstance(task_id, str)
        or not task_id
        or not isinstance(trajectory_id, str)
        or not trajectory_id
        or not isinstance(query_text, str)
        or not query_text.strip()
        or query_text != query_text.strip()
        or type(original.get("success")) is not bool
    ):
        raise ValueError("original trajectory identity differs")

    if original.get("success") is True:
        if not isinstance(success_extractor, SuccessfulTrajectoryExperienceExtractor):
            raise TypeError("successful trajectory extractor is required")
        extraction = extraction_override or success_extractor.extract(original)
        experience_nodes = extraction.experience_node_dicts()
        experience_edges = extraction.edge_dicts()
        audit_row = {
            "train_index": train_index,
            "task_id": task_id,
            "trajectory_id": trajectory_id,
            "origin": "ORIGINAL_SUCCESS",
            "status": "INGESTED",
            "prompt_sha256": SUCCESS_PROMPT_SHA256,
            "source_protocol": extraction.source_protocol,
            "source_protocol_sha256": extraction.source_protocol_sha256,
            "request_payload_sha256": extraction.request_payload_sha256,
            "response_schema_sha256": extraction.response_schema_sha256,
        }
    else:
        outcome = outcomes[task_id]
        validate_source_replay_outcome_protocol(outcome)
        status = outcome.get("status")
        if status not in _REPLAY_TERMINAL_STATUSES:
            raise ValueError("replay outcome terminal status differs")
        if (
            outcome.get("parent_trajectory_id") != trajectory_id
            or outcome.get("task_id") != task_id
        ):
            raise ValueError("replay outcome does not belong to the original failure")
        if status != "REPLAY_VALIDATED_SUCCESS":
            return None
        if not isinstance(repair_extractor, ValidatedRepairExperienceExtractor):
            raise TypeError("validated repair extractor is required")
        attempt = _accepted_attempt(outcome)
        trajectory_path = attempt.get("replay_trajectory_path")
        if not isinstance(trajectory_path, str) or not trajectory_path:
            raise ValueError("accepted replay trajectory path differs")
        successful_replay = replay_trajectory_loader(trajectory_path)
        example = select_validated_repair_example(outcome, successful_replay)
        extraction = extraction_override or repair_extractor.extract(example)
        experience_nodes = extraction.experience_node_dicts()
        experience_edges = extraction.edge_dicts()
        audit_row = {
            "train_index": train_index,
            "task_id": task_id,
            "trajectory_id": example.trajectory_id,
            "origin": "REPLAY_VALIDATED_SUCCESS",
            "status": "INGESTED",
            "accepted_patch_id": example.accepted_patch_id,
            "accepted_attempt_index": example.accepted_attempt_index,
            "accepted_patch_sha256": attempt["patch_sha256"],
            "successful_replay_sha256": attempt["replay_trajectory_sha256"],
            "repair_memory_sha256": hashlib.sha256(
                canonical_json_bytes(example.memory.to_dict())
            ).hexdigest(),
            "source_replay_protocol": outcome["source_replay_protocol"],
            "source_replay_protocol_sha256": outcome[
                "source_replay_protocol_sha256"
            ],
            "replay_evidence_mode": "CURRENT_NO_TRUNCATION",
            "prompt_sha256": REPAIR_PROMPT_SHA256,
            "source_protocol": extraction.source_protocol,
            "source_protocol_sha256": extraction.source_protocol_sha256,
            "request_payload_sha256": extraction.request_payload_sha256,
            "response_schema_sha256": extraction.response_schema_sha256,
        }
    audit_row["query_text"] = query_text
    audit_row["query_text_sha256"] = hashlib.sha256(
        query_text.encode("utf-8")
    ).hexdigest()
    audit_row["discarded_edge_reasons"] = list(
        extraction.discarded_edge_reasons
    )
    workflow = {
        "train_index": train_index,
        "task_id": task_id,
        "query_text": query_text,
        "experience_nodes": experience_nodes,
        "edges": experience_edges,
    }
    return workflow, audit_row


def _checkpoint_path(checkpoint_dir: Path, train_index: int, task_id: str) -> Path:
    task_digest = hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:12]
    return checkpoint_dir / f"{train_index:03d}_{task_digest}" / "extraction.json"


def _review_checkpoint_path(
    checkpoint_dir: Path, train_index: int, task_id: str
) -> Path:
    return _checkpoint_path(checkpoint_dir, train_index, task_id).with_name(
        "review.json"
    )


def _source_graph_sha256(workflow: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "experience_nodes": workflow["experience_nodes"],
                "edges": workflow["edges"],
            }
        )
    ).hexdigest()


def _load_review_checkpoint(
    path: Path,
    *,
    draft_workflow: Mapping[str, Any],
    expected_protocol: Mapping[str, Any],
    expected_payload_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = _strict_json_object_file(path, label="source review checkpoint")
    protocol = payload.get("review_protocol")
    expected_fields = {
        "format",
        "draft_workflow_sha256",
        "review_protocol",
        "review_protocol_sha256",
        "review_request_payload_sha256",
        "review_response_schema_sha256",
        "final_workflow",
        "review_audit",
    }
    final_workflow = payload.get("final_workflow")
    review_audit = payload.get("review_audit")
    if (
        set(payload) != expected_fields
        or payload.get("format") != SOURCE_REVIEW_CHECKPOINT_FORMAT
        or payload.get("draft_workflow_sha256")
        != _source_graph_sha256(draft_workflow)
        or type(protocol) is not dict
        or not _same_producer_cache_identity(protocol, expected_protocol)
        or payload.get("review_protocol_sha256")
        != hashlib.sha256(canonical_json_bytes(protocol)).hexdigest()
        or payload.get("review_request_payload_sha256")
        != expected_payload_sha256
        or payload.get("review_response_schema_sha256")
        != hashlib.sha256(
            canonical_json_bytes(source_review_response_schema())
        ).hexdigest()
        or type(final_workflow) is not dict
        or set(final_workflow)
        != {"train_index", "task_id", "query_text", "experience_nodes", "edges"}
        or final_workflow.get("train_index") != draft_workflow.get("train_index")
        or final_workflow.get("task_id") != draft_workflow.get("task_id")
        or final_workflow.get("query_text") != draft_workflow.get("query_text")
        or type(review_audit) is not dict
    ):
        raise ValueError("source review checkpoint identity differs")
    nodes, edges, discarded = _parse_llm_experience_graph(
        {
            "experience_nodes": final_workflow["experience_nodes"],
            "edges": final_workflow["edges"],
        }
    )
    if (
        discarded
        or [row.to_dict() for row in nodes] != final_workflow["experience_nodes"]
        or [row.to_dict() for row in edges] != final_workflow["edges"]
    ):
        raise ValueError("source review checkpoint graph is not canonical")
    return final_workflow, review_audit


def _load_extraction_checkpoint(
    path: Path,
    *,
    train_index: int,
    task_id: str,
    expected_origin: str,
    expected_protocol: Mapping[str, Any],
    expected_audit_bindings: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = _strict_json_object_file(path, label="source extraction checkpoint")
    if set(payload) != {"format", "workflow", "audit"} or payload.get("format") != (
        SOURCE_EXTRACTION_CHECKPOINT_FORMAT
    ):
        raise ValueError("source extraction checkpoint fields differ")
    workflow = payload.get("workflow")
    audit = payload.get("audit")
    if (
        type(workflow) is not dict
        or set(workflow)
        != {"train_index", "task_id", "query_text", "experience_nodes", "edges"}
        or workflow.get("train_index") != train_index
        or workflow.get("task_id") != task_id
        or workflow.get("query_text") != expected_audit_bindings.get("query_text")
        or type(audit) is not dict
        or audit.get("train_index") != train_index
        or audit.get("task_id") != task_id
        or audit.get("origin") != expected_origin
        or any(
            audit.get(key) != value
            for key, value in expected_audit_bindings.items()
        )
        or type(audit.get("discarded_edge_reasons")) is not list
        or any(
            type(reason) is not str or not reason
            for reason in audit.get("discarded_edge_reasons", [])
        )
    ):
        raise ValueError("source extraction checkpoint identity differs")
    prompt_sha256 = (
        SUCCESS_PROMPT_SHA256
        if expected_origin == "ORIGINAL_SUCCESS"
        else REPAIR_PROMPT_SHA256
    )
    protocol = audit.get("source_protocol")
    if (
        audit.get("prompt_sha256") != prompt_sha256
        or type(protocol) is not dict
        or audit.get("source_protocol_sha256")
        != hashlib.sha256(canonical_json_bytes(protocol)).hexdigest()
        or not _same_producer_cache_identity(protocol, expected_protocol)
    ):
        raise ValueError("source extraction checkpoint protocol differs")
    nodes, edges, discarded_edge_reasons = _parse_llm_experience_graph(
        {
            "experience_nodes": workflow["experience_nodes"],
            "edges": workflow["edges"],
        }
    )
    if (
        discarded_edge_reasons
        or [node.to_dict() for node in nodes] != workflow["experience_nodes"]
        or [edge.to_dict() for edge in edges] != workflow["edges"]
    ):
        raise ValueError("source extraction checkpoint is not canonical")
    return workflow, audit


def _write_transport_failure_attempt(
    path: Path,
    *,
    protocol: Mapping[str, Any],
    request_id: str,
    payload_sha256: str,
    error: Exception,
) -> str:
    failure = f"{type(error).__name__}: {error}"
    _write_json_output(
        path,
        {
            "format": SOURCE_RAW_RESPONSE_FORMAT,
            "outcome": "TRANSPORT_EXHAUSTED",
            "error": failure,
            "request_kind": protocol["request_kind"],
            "request_id": request_id,
            "system_prompt_sha256": protocol["prompt_sha256"],
            "payload_sha256": payload_sha256,
            "source_protocol": dict(protocol),
            "source_protocol_sha256": hashlib.sha256(
                canonical_json_bytes(protocol)
            ).hexdigest(),
            "response": "",
        },
    )
    return failure


def _load_saved_invalid_response_attempts(
    checkpoint_parent: Path,
    *,
    expected_protocol: Mapping[str, Any],
    expected_request_id: str,
    expected_payload_sha256: str,
) -> tuple[tuple[str, ...], Mapping[str, Any] | None]:
    actual_paths = sorted(checkpoint_parent.glob("raw_response_attempt_*.json"))
    if not 1 <= len(actual_paths) <= SOURCE_EXTRACTION_SEMANTIC_ATTEMPTS:
        raise ValueError("saved source response attempt count differs")
    expected_paths = [
        checkpoint_parent / f"raw_response_attempt_{index:02d}.json"
        for index in range(1, len(actual_paths) + 1)
    ]
    if actual_paths != expected_paths:
        raise ValueError("saved source response attempts are not a contiguous prefix")
    invalid_response_attempts: list[str] = []
    valid_response: Mapping[str, Any] | None = None
    for path_index, path in enumerate(expected_paths):
        payload = _strict_json_object_file(path, label="saved source LLM response")
        protocol = payload.get("source_protocol")
        outcome = payload.get("outcome")
        expected_fields = {
            "format",
            "outcome",
            "request_kind",
            "request_id",
            "system_prompt_sha256",
            "payload_sha256",
            "source_protocol",
            "source_protocol_sha256",
            "response",
        }
        if outcome in {"COMPLETION_LENGTH_EXCEEDED", "TRANSPORT_EXHAUSTED"}:
            expected_fields.add("error")
        if (
            set(payload) != expected_fields
            or payload.get("format") != SOURCE_RAW_RESPONSE_FORMAT
            or outcome
            not in {
                "COMPLETE",
                "COMPLETION_LENGTH_EXCEEDED",
                "TRANSPORT_EXHAUSTED",
            }
            or payload.get("request_kind") != expected_protocol.get("request_kind")
            or payload.get("request_id") != expected_request_id
            or payload.get("system_prompt_sha256")
            != expected_protocol.get("prompt_sha256")
            or payload.get("payload_sha256") != expected_payload_sha256
            or type(protocol) is not dict
            or not _same_producer_cache_identity(protocol, expected_protocol)
            or payload.get("source_protocol_sha256")
            != hashlib.sha256(canonical_json_bytes(protocol)).hexdigest()
            or type(payload.get("response")) is not str
        ):
            raise ValueError("saved generation failure identity differs")
        if outcome in {"COMPLETION_LENGTH_EXCEEDED", "TRANSPORT_EXHAUSTED"}:
            error = payload.get("error")
            if type(error) is not str or not error:
                raise ValueError("saved source failure error differs")
            if outcome == "TRANSPORT_EXHAUSTED":
                if payload["response"]:
                    raise ValueError("saved transport failure response differs")
                invalid_response_attempts.append(error)
                continue
            partial = payload["response"]
            invalid_response_attempts.append(
                f"RequestCompletionLengthExceeded: {error}; "
                f"partial_chars={len(partial)}; "
                f"partial_sha256={hashlib.sha256(partial.encode('utf-8')).hexdigest()}"
            )
            continue
        try:
            response = _strict_json_object(payload["response"])
            _parse_llm_experience_graph(response)
        except ValueError as exc:
            invalid_response_attempts.append(f"{type(exc).__name__}: {exc}")
        else:
            if path_index != len(expected_paths) - 1:
                raise ValueError("a valid saved response precedes another attempt")
            valid_response = response
    return tuple(invalid_response_attempts), valid_response


def _load_saved_review_response_attempts(
    checkpoint_parent: Path,
    *,
    expected_protocol: Mapping[str, Any],
    expected_request_id: str,
    expected_payload_sha256: str,
    draft_count: int,
    attempt_prefix: str = "raw_review_response_attempt",
) -> tuple[tuple[str, ...], SourceReviewResult | None]:
    if attempt_prefix not in {
        "raw_review_response_attempt",
        "raw_review_retry_response_attempt",
    }:
        raise ValueError("source review attempt prefix differs")
    actual_paths = sorted(
        checkpoint_parent.glob(f"{attempt_prefix}_*.json")
    )
    if not 1 <= len(actual_paths) <= SOURCE_EXTRACTION_SEMANTIC_ATTEMPTS:
        raise ValueError("saved source review response attempt count differs")
    expected_paths = [
        checkpoint_parent / f"{attempt_prefix}_{index:02d}.json"
        for index in range(1, len(actual_paths) + 1)
    ]
    if actual_paths != expected_paths:
        raise ValueError(
            "saved source review response attempts are not a contiguous prefix"
        )
    invalid_response_attempts: list[str] = []
    valid_response: SourceReviewResult | None = None
    response_schema_sha256 = hashlib.sha256(
        canonical_json_bytes(source_review_response_schema())
    ).hexdigest()
    for path_index, path in enumerate(expected_paths):
        payload = _strict_json_object_file(
            path, label="saved source review LLM response"
        )
        protocol = payload.get("source_protocol")
        outcome = payload.get("outcome")
        expected_fields = {
            "format",
            "outcome",
            "request_kind",
            "request_id",
            "system_prompt_sha256",
            "payload_sha256",
            "source_protocol",
            "source_protocol_sha256",
            "response",
        }
        if outcome in {"COMPLETION_LENGTH_EXCEEDED", "TRANSPORT_EXHAUSTED"}:
            expected_fields.add("error")
        if (
            set(payload) != expected_fields
            or payload.get("format") != SOURCE_RAW_RESPONSE_FORMAT
            or outcome
            not in {
                "COMPLETE",
                "COMPLETION_LENGTH_EXCEEDED",
                "TRANSPORT_EXHAUSTED",
            }
            or payload.get("request_kind") != expected_protocol.get("request_kind")
            or payload.get("request_id") != expected_request_id
            or payload.get("system_prompt_sha256")
            != expected_protocol.get("prompt_sha256")
            or payload.get("payload_sha256") != expected_payload_sha256
            or type(protocol) is not dict
            or not _same_producer_cache_identity(protocol, expected_protocol)
            or payload.get("source_protocol_sha256")
            != hashlib.sha256(canonical_json_bytes(protocol)).hexdigest()
            or type(payload.get("response")) is not str
        ):
            raise ValueError("saved source review response identity differs")
        if outcome in {"COMPLETION_LENGTH_EXCEEDED", "TRANSPORT_EXHAUSTED"}:
            error = payload.get("error")
            if type(error) is not str or not error:
                raise ValueError("saved source review failure error differs")
            if outcome == "TRANSPORT_EXHAUSTED":
                if payload["response"]:
                    raise ValueError("saved source review transport response differs")
                invalid_response_attempts.append(error)
                continue
            partial = payload["response"]
            invalid_response_attempts.append(
                f"RequestCompletionLengthExceeded: {error}; "
                f"partial_chars={len(partial)}; "
                f"partial_sha256={hashlib.sha256(partial.encode('utf-8')).hexdigest()}"
            )
            continue
        try:
            response = _strict_json_object(payload["response"])
            nodes, edges, discarded, decisions, ledger_errors = (
                parse_source_review_response(response, draft_count=draft_count)
            )
        except ValueError as exc:
            invalid_response_attempts.append(f"{type(exc).__name__}: {exc}")
        else:
            if path_index != len(expected_paths) - 1:
                raise ValueError(
                    "a valid saved source review response precedes another attempt"
                )
            valid_response = SourceReviewResult(
                nodes,
                edges,
                discarded,
                decisions,
                ledger_errors,
                dict(protocol),
                hashlib.sha256(canonical_json_bytes(protocol)).hexdigest(),
                expected_payload_sha256,
                response_schema_sha256,
            )
    return tuple(invalid_response_attempts), valid_response


async def rebuild_section_source_async(
    *,
    original_records: Sequence[Mapping[str, Any]],
    replay_outcomes: Sequence[Mapping[str, Any]],
    replay_trajectory_loader: ReplayTrajectoryLoader,
    extractor_factory: SourceExtractorFactory,
    reviewer_factory: SourceReviewerFactory,
    checkpoint_dir: Path,
    retry_failed_reviews: bool = False,
    selected_train_indices: Collection[int] | None = None,
    progress_callback: SourceProgressCallback | None = None,
) -> ExperienceSourceBuild:
    """Extract independent source workflows with async 16-way orchestration."""

    if (
        not callable(replay_trajectory_loader)
        or not callable(extractor_factory)
        or not callable(reviewer_factory)
        or type(retry_failed_reviews) is not bool
    ):
        raise TypeError("parallel source rebuild dependencies differ")
    outcomes = _validated_source_outcomes(original_records, replay_outcomes)
    selected: frozenset[int] | None = None
    if selected_train_indices is not None:
        selected_rows = tuple(selected_train_indices)
        if (
            len(selected_rows) != INCREMENTAL_BATCH_SIZE
            or len(set(selected_rows)) != INCREMENTAL_BATCH_SIZE
            or any(
                type(index) is not int
                or not 0 <= index < TRAIN_INSTRUCTION_COUNT
                for index in selected_rows
            )
        ):
            raise ValueError("incremental source batch must contain exactly 8 train indices")
        ordered = tuple(sorted(selected_rows))
        if ordered[0] % INCREMENTAL_BATCH_SIZE != 0 or ordered != tuple(
            range(ordered[0], ordered[0] + INCREMENTAL_BATCH_SIZE)
        ):
            raise ValueError(
                "incremental source batch must be one aligned contiguous 8-index block"
            )
        selected = frozenset(ordered)
    checkpoint_root = Path(checkpoint_dir).expanduser().absolute()
    _ensure_directory(checkpoint_root)
    _recover_source_systemic_transport_archives(checkpoint_root)
    _archive_interrupted_source_transport_waves(checkpoint_root)
    active_rows: list[tuple[int, Mapping[str, Any], str]] = []
    passive_exclusions: list[dict[str, Any]] = []
    for train_index, original in enumerate(original_records):
        if selected is not None and train_index not in selected:
            continue
        if not isinstance(original, Mapping):
            raise ValueError("original trajectory row must be an object")
        task_id = original.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("original trajectory identity differs")
        if original.get("success") is True:
            origin = "ORIGINAL_SUCCESS"
        else:
            outcome = outcomes[task_id]
            validate_source_replay_outcome_protocol(outcome)
            if outcome.get("status") not in _REPLAY_TERMINAL_STATUSES:
                raise ValueError("replay outcome terminal status differs")
            if outcome.get("status") != "REPLAY_VALIDATED_SUCCESS":
                passive_exclusions.append(
                    _no_validated_success_exclusion(
                        train_index=train_index,
                        original=original,
                        replay_status=str(outcome["status"]),
                    )
                )
                continue
            origin = "REPLAY_VALIDATED_SUCCESS"
        active_rows.append((train_index, original, origin))
    protocol_by_origin: dict[str, dict[str, Any]] = {}
    for origin in sorted({origin for _index, _row, origin in active_rows}):
        protocol_probe = extractor_factory(
            origin, checkpoint_root / f".{origin.lower()}-protocol-probe.json"
        )
        protocol = getattr(protocol_probe.llm, "protocol_identity", None)
        if not isinstance(protocol, Mapping):
            raise TypeError("source extractor protocol identity differs")
        protocol_by_origin[origin] = dict(protocol)
    review_protocol_probe = reviewer_factory(
        checkpoint_root / ".source-review-protocol-probe.json"
    )
    review_protocol_value = getattr(
        review_protocol_probe.llm, "protocol_identity", None
    )
    if not isinstance(review_protocol_value, Mapping):
        raise TypeError("source review protocol identity differs")
    review_protocol = dict(review_protocol_value)
    if (
        review_protocol.get("format") != SOURCE_REVIEW_PROTOCOL_FORMAT
        or review_protocol.get("prompt_sha256")
        != SOURCE_REVIEW_PROMPT_SHA256
    ):
        raise ValueError("source review protocol identity differs")

    semaphore = asyncio.Semaphore(SOURCE_REBUILD_WORKERS)
    extraction_transport_guard = ProducerTransportGuard(
        stage="source extraction"
    )
    review_transport_guard = ProducerTransportGuard(stage="source review")
    pending_publications: list[tuple[Path, dict[str, Any], bool]] = []

    async def extract_one(
        train_index: int,
        original: Mapping[str, Any],
        origin: str,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        task_id = str(original["task_id"])
        checkpoint = _checkpoint_path(checkpoint_root, train_index, task_id)
        terminal_exclusion_checkpoint = checkpoint.parent / "terminal_exclusion.json"
        raw_response_paths = sorted(
            checkpoint.parent.glob("raw_response_attempt_*.json")
        )
        example = None

        def source_evidence() -> tuple[str, dict[str, Any], dict[str, Any]]:
            nonlocal example
            if origin == "ORIGINAL_SUCCESS":
                request_id = original.get("trajectory_id")
                payload = {
                    "successful_trajectory": render_successful_replay(original).payload
                }
                audit_bindings: dict[str, Any] = {}
            else:
                outcome = outcomes[task_id]
                attempt = _accepted_attempt(outcome)
                trajectory_path = attempt.get("replay_trajectory_path")
                if not isinstance(trajectory_path, str) or not trajectory_path:
                    raise ValueError("accepted replay trajectory path differs")
                successful_replay = replay_trajectory_loader(trajectory_path)
                example = select_validated_repair_example(outcome, successful_replay)
                request_id = example.trajectory_id
                payload = {
                    "validated_repair_memory": example.memory.to_dict(),
                    "successful_replay_trajectory": render_successful_replay(
                        example.successful_replay_trajectory
                    ).payload,
                }
                audit_bindings = {
                    "accepted_patch_id": example.accepted_patch_id,
                    "accepted_attempt_index": example.accepted_attempt_index,
                    "accepted_patch_sha256": attempt["patch_sha256"],
                    "successful_replay_sha256": attempt[
                        "replay_trajectory_sha256"
                    ],
                    "repair_memory_sha256": hashlib.sha256(
                        canonical_json_bytes(example.memory.to_dict())
                    ).hexdigest(),
                    "source_replay_protocol": outcome[
                        "source_replay_protocol"
                    ],
                    "source_replay_protocol_sha256": outcome[
                        "source_replay_protocol_sha256"
                    ],
                    "replay_evidence_mode": "CURRENT_NO_TRUNCATION",
                }
            if not isinstance(request_id, str) or not request_id:
                raise ValueError("source extraction request identity differs")
            return request_id, payload, audit_bindings

        def source_request_identity() -> tuple[str, str, dict[str, Any]]:
            request_id, payload, audit_bindings = source_evidence()
            return (
                request_id,
                hashlib.sha256(canonical_json_bytes(payload)).hexdigest(),
                {
                    "trajectory_id": request_id,
                    "request_payload_sha256": hashlib.sha256(
                        canonical_json_bytes(payload)
                    ).hexdigest(),
                    "response_schema_sha256": hashlib.sha256(
                        canonical_json_bytes(repair_response_schema())
                    ).hexdigest(),
                    **audit_bindings,
                    "query_text": original["instruction"],
                    "query_text_sha256": hashlib.sha256(
                        str(original["instruction"]).encode("utf-8")
                    ).hexdigest(),
                },
            )

        def bind_active_exclusion(exclusion: Mapping[str, Any]) -> dict[str, Any]:
            _request_id, payload_sha256, _audit_bindings = (
                source_request_identity()
            )
            protocol = protocol_by_origin[origin]
            return {
                **dict(exclusion),
                "prompt_sha256": protocol["prompt_sha256"],
                "source_protocol": protocol,
                "source_protocol_sha256": hashlib.sha256(
                    canonical_json_bytes(protocol)
                ).hexdigest(),
                "request_payload_sha256": payload_sha256,
                "response_schema_sha256": hashlib.sha256(
                    canonical_json_bytes(repair_response_schema())
                ).hexdigest(),
            }

        def extraction_from_saved_response(
            response: Mapping[str, Any],
        ) -> SuccessfulTrajectoryExtraction | ValidatedRepairExtraction:
            nodes, edges, discarded = _parse_llm_experience_graph(response)
            _request_id, payload_sha256, _audit_bindings = (
                source_request_identity()
            )
            protocol = protocol_by_origin[origin]
            fields = (
                nodes,
                edges,
                discarded,
                protocol,
                hashlib.sha256(canonical_json_bytes(protocol)).hexdigest(),
                payload_sha256,
                hashlib.sha256(
                    canonical_json_bytes(repair_response_schema())
                ).hexdigest(),
            )
            return (
                SuccessfulTrajectoryExtraction(*fields)
                if origin == "ORIGINAL_SUCCESS"
                else ValidatedRepairExtraction(*fields)
            )

        def finalize_extraction(
            extraction: SuccessfulTrajectoryExtraction
            | ValidatedRepairExtraction,
            *,
            attempt_index: int,
            invalid_responses: Sequence[str],
            extractor: SuccessfulTrajectoryExperienceExtractor
            | ValidatedRepairExperienceExtractor,
        ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
            result = _extract_source_row(
                train_index=train_index,
                original=original,
                outcomes=outcomes,
                replay_trajectory_loader=replay_trajectory_loader,
                success_extractor=(
                    extractor
                    if isinstance(
                        extractor,
                        SuccessfulTrajectoryExperienceExtractor,
                    )
                    else None
                ),
                repair_extractor=(
                    extractor
                    if isinstance(
                        extractor,
                        ValidatedRepairExperienceExtractor,
                    )
                    else None
                ),
                extraction_override=extraction,
            )
            if result is None:
                raise AssertionError("active source extraction was skipped")
            if not extraction.experience_nodes:
                _request_id, payload_sha256, _audit_bindings = (
                    source_request_identity()
                )
                exclusion = _no_reusable_experience_exclusion(
                    train_index=train_index,
                    original=original,
                    origin=origin,
                )
                _write_terminal_exclusion_checkpoint(
                    terminal_exclusion_checkpoint,
                    train_index=train_index,
                    original=original,
                    origin=origin,
                    protocol=protocol_by_origin[origin],
                    request_payload_sha256=payload_sha256,
                    exclusion=exclusion,
                )
                return None, bind_active_exclusion(exclusion)
            workflow, audit = result
            audit["source_extraction_attempt_index"] = attempt_index
            audit["invalid_response_attempts"] = list(invalid_responses)
            _write_json_output(
                checkpoint,
                {
                    "format": SOURCE_EXTRACTION_CHECKPOINT_FORMAT,
                    "workflow": workflow,
                    "audit": audit,
                },
            )
            return workflow, audit

        if checkpoint.exists():
            _request_id, _payload_sha256, audit_bindings = source_request_identity()
            result = _load_extraction_checkpoint(
                checkpoint,
                train_index=train_index,
                task_id=task_id,
                expected_origin=origin,
                expected_protocol=protocol_by_origin[origin],
                expected_audit_bindings=audit_bindings,
            )
            status = "RESUMED"
        elif terminal_exclusion_checkpoint.is_file():
            _request_id, payload_sha256, _audit_bindings = source_request_identity()
            result = (
                None,
                bind_active_exclusion(
                    _load_terminal_exclusion_checkpoint(
                        terminal_exclusion_checkpoint,
                        train_index=train_index,
                        original=original,
                        origin=origin,
                        expected_protocol=protocol_by_origin[origin],
                        expected_request_payload_sha256=payload_sha256,
                    )
                ),
            )
            status = str(result[1]["status"])
        else:
            async def run_async() -> tuple[dict[str, Any] | None, dict[str, Any]]:
                saved_invalid_responses, saved_response = (
                    _load_saved_invalid_response_attempts(
                        checkpoint.parent,
                        expected_protocol=protocol_by_origin[origin],
                        expected_request_id=source_request_identity()[0],
                        expected_payload_sha256=source_request_identity()[1],
                    )
                    if raw_response_paths
                    else ((), None)
                )
                invalid_responses = list(saved_invalid_responses)
                if saved_response is not None:
                    extractor = extractor_factory(origin, raw_response_paths[-1])
                    if (
                        dict(extractor.llm.protocol_identity)
                        != protocol_by_origin[origin]
                    ):
                        raise ValueError(
                            "source extractor protocol identity is not stable"
                        )
                    return finalize_extraction(
                        extraction_from_saved_response(saved_response),
                        attempt_index=len(raw_response_paths),
                        invalid_responses=invalid_responses,
                        extractor=extractor,
                    )
                if origin == "REPLAY_VALIDATED_SUCCESS":
                    source_request_identity()
                for attempt_index in range(
                    len(invalid_responses) + 1,
                    SOURCE_EXTRACTION_SEMANTIC_ATTEMPTS + 1,
                ):
                    raw_response = checkpoint.parent / (
                        f"raw_response_attempt_{attempt_index:02d}.json"
                    )
                    extractor = extractor_factory(origin, raw_response)
                    if (
                        dict(extractor.llm.protocol_identity)
                        != protocol_by_origin[origin]
                    ):
                        raise ValueError(
                            "source extractor protocol identity is not stable"
                        )
                    try:
                        if origin == "ORIGINAL_SUCCESS":
                            if not isinstance(
                                extractor,
                                SuccessfulTrajectoryExperienceExtractor,
                            ):
                                raise TypeError("successful source extractor differs")
                            extraction = await extractor.extract_async(
                                original,
                            )
                        else:
                            if not isinstance(
                                extractor,
                                ValidatedRepairExperienceExtractor,
                            ):
                                raise TypeError("repair source extractor differs")
                            if example is None:
                                raise AssertionError(
                                    "validated repair example is missing"
                                )
                            extraction = await extractor.extract_async(
                                example,
                            )
                        await extraction_transport_guard.record_success()
                    except RequestContextLengthExceeded as exc:
                        await extraction_transport_guard.record_success()
                        exclusion = _context_length_exclusion(
                            train_index=train_index,
                            original=original,
                            origin=origin,
                            error=exc,
                        )
                        _request_id, payload_sha256, _audit_bindings = (
                            source_request_identity()
                        )
                        _write_terminal_exclusion_checkpoint(
                            terminal_exclusion_checkpoint,
                            train_index=train_index,
                            original=original,
                            origin=origin,
                            protocol=protocol_by_origin[origin],
                            request_payload_sha256=payload_sha256,
                            exclusion=exclusion,
                        )
                        return None, bind_active_exclusion(exclusion)
                    except RequestCompletionLengthExceeded as exc:
                        await extraction_transport_guard.record_success()
                        partial = exc.partial_content
                        request_id, payload_sha256, _audit_bindings = (
                            source_request_identity()
                        )
                        protocol = protocol_by_origin[origin]
                        _write_json_output(
                            raw_response,
                            {
                                "format": SOURCE_RAW_RESPONSE_FORMAT,
                                "outcome": "COMPLETION_LENGTH_EXCEEDED",
                                "error": str(exc),
                                "request_kind": protocol["request_kind"],
                                "request_id": request_id,
                                "system_prompt_sha256": protocol["prompt_sha256"],
                                "payload_sha256": payload_sha256,
                                "source_protocol": protocol,
                                "source_protocol_sha256": hashlib.sha256(
                                    canonical_json_bytes(protocol)
                                ).hexdigest(),
                                "response": partial,
                            },
                        )
                        invalid_responses.append(
                            f"{type(exc).__name__}: {exc}; "
                            f"partial_chars={len(partial)}; "
                            f"partial_sha256={hashlib.sha256(partial.encode('utf-8')).hexdigest()}"
                        )
                        continue
                    except (RequestRuntimeTimeout, APIError) as exc:
                        request_id, payload_sha256, _audit_bindings = (
                            source_request_identity()
                        )
                        failure = _write_transport_failure_attempt(
                            raw_response,
                            protocol=protocol_by_origin[origin],
                            request_id=request_id,
                            payload_sha256=payload_sha256,
                            error=exc,
                        )
                        invalid_responses.append(failure)
                        await extraction_transport_guard.record_failure(
                            request_id=request_id, error=exc
                        )
                        continue
                    except ValueError as exc:
                        if not raw_response.is_file():
                            raise
                        await extraction_transport_guard.record_success()
                        invalid_responses.append(f"{type(exc).__name__}: {exc}")
                        continue
                    return finalize_extraction(
                        extraction,
                        attempt_index=attempt_index,
                        invalid_responses=invalid_responses,
                        extractor=extractor,
                    )
                _request_id, payload_sha256, _audit_bindings = source_request_identity()
                exclusion = _generation_failure_exclusion(
                    train_index=train_index,
                    original=original,
                    origin=origin,
                    invalid_response_attempts=invalid_responses,
                )
                pending_publications.append(
                    (
                        terminal_exclusion_checkpoint,
                        _terminal_exclusion_checkpoint_payload(
                            train_index=train_index,
                            original=original,
                            origin=origin,
                            protocol=protocol_by_origin[origin],
                            request_payload_sha256=payload_sha256,
                            exclusion=exclusion,
                        ),
                        False,
                    )
                )
                return None, bind_active_exclusion(exclusion)

            async with semaphore:
                result = await run_async()
            status = (
                result[1]["status"] if result[0] is None else "EXTRACTED"
            )
        if result[0] is not None:
            draft_workflow, draft_audit = result
            draft_nodes, draft_edges, draft_discarded = (
                _parse_llm_experience_graph(
                    {
                        "experience_nodes": draft_workflow[
                            "experience_nodes"
                        ],
                        "edges": draft_workflow["edges"],
                    }
                )
            )
            if draft_discarded:
                raise ValueError("draft source checkpoint contains invalid edges")
            source_request_id, evidence, _audit_bindings = source_evidence()
            review_payload = source_review_payload(
                evidence_mode=origin,
                evidence=evidence,
                draft_nodes=draft_nodes,
                draft_edges=draft_edges,
            )
            review_payload_sha256 = hashlib.sha256(
                canonical_json_bytes(review_payload)
            ).hexdigest()
            review_checkpoint = _review_checkpoint_path(
                checkpoint_root, train_index, task_id
            )
            retrying_review = False
            if review_checkpoint.is_file():
                final_workflow, review_audit = _load_review_checkpoint(
                    review_checkpoint,
                    draft_workflow=draft_workflow,
                    expected_protocol=review_protocol,
                    expected_payload_sha256=review_payload_sha256,
                )
                current_review_status = review_audit.get(
                    "source_review_status"
                )
                if (
                    retry_failed_reviews
                    and current_review_status == "REVIEW_DRAFT_FALLBACK"
                ):
                    retrying_review = True
                else:
                    status = "REVIEW_RESUMED"
            if not review_checkpoint.is_file() or retrying_review:
                attempt_prefix = (
                    "raw_review_retry_response_attempt"
                    if retrying_review
                    else "raw_review_response_attempt"
                )
                raw_review_response_paths = sorted(
                    checkpoint.parent.glob(
                        f"{attempt_prefix}_*.json"
                    )
                )
                if raw_review_response_paths:
                    saved_invalid_reviews, reviewed = (
                        _load_saved_review_response_attempts(
                            checkpoint.parent,
                            expected_protocol=review_protocol,
                            expected_request_id=(
                                f"source-review-{source_request_id}"
                            ),
                            expected_payload_sha256=review_payload_sha256,
                            draft_count=len(draft_nodes),
                            attempt_prefix=attempt_prefix,
                        )
                    )
                    invalid_review_responses = list(saved_invalid_reviews)
                else:
                    invalid_review_responses = []
                    reviewed = None
                for review_attempt_index in range(
                    len(raw_review_response_paths) + 1,
                    (
                        SOURCE_EXTRACTION_SEMANTIC_ATTEMPTS + 1
                        if reviewed is None
                        else len(raw_review_response_paths) + 1
                    ),
                ):
                    raw_review_response = checkpoint.parent / (
                        f"{attempt_prefix}_{review_attempt_index:02d}.json"
                    )
                    reviewer = reviewer_factory(raw_review_response)
                    if (
                        dict(reviewer.llm.protocol_identity)
                        != review_protocol
                    ):
                        raise ValueError(
                            "source review protocol identity is not stable"
                        )
                    try:
                        async with semaphore:
                            reviewed = await reviewer.review_async(
                                request_id=(
                                    f"source-review-{source_request_id}"
                                ),
                                evidence_mode=origin,
                                evidence=evidence,
                                draft_nodes=draft_nodes,
                                draft_edges=draft_edges,
                            )
                        await review_transport_guard.record_success()
                    except RequestContextLengthExceeded as exc:
                        await review_transport_guard.record_success()
                        invalid_review_responses.append(
                            f"{type(exc).__name__}: {exc}"
                        )
                        break
                    except RequestCompletionLengthExceeded as exc:
                        await review_transport_guard.record_success()
                        partial = exc.partial_content
                        _write_json_output(
                            raw_review_response,
                            {
                                "format": SOURCE_RAW_RESPONSE_FORMAT,
                                "outcome": "COMPLETION_LENGTH_EXCEEDED",
                                "error": str(exc),
                                "request_kind": review_protocol[
                                    "request_kind"
                                ],
                                "request_id": (
                                    f"source-review-{source_request_id}"
                                ),
                                "system_prompt_sha256": (
                                    SOURCE_REVIEW_PROMPT_SHA256
                                ),
                                "payload_sha256": review_payload_sha256,
                                "source_protocol": review_protocol,
                                "source_protocol_sha256": hashlib.sha256(
                                    canonical_json_bytes(review_protocol)
                                ).hexdigest(),
                                "response": partial,
                            },
                        )
                        invalid_review_responses.append(
                            f"{type(exc).__name__}: {exc}; "
                            f"partial_chars={len(partial)}; "
                            f"partial_sha256={hashlib.sha256(partial.encode('utf-8')).hexdigest()}"
                        )
                        continue
                    except (RequestRuntimeTimeout, APIError) as exc:
                        failure = _write_transport_failure_attempt(
                                raw_review_response,
                                protocol=review_protocol,
                                request_id=(
                                    f"source-review-{source_request_id}"
                                ),
                                payload_sha256=review_payload_sha256,
                                error=exc,
                            )
                        invalid_review_responses.append(failure)
                        await review_transport_guard.record_failure(
                            request_id=f"source-review-{source_request_id}",
                            error=exc,
                        )
                        continue
                    except ValueError as exc:
                        if not raw_review_response.is_file():
                            raise
                        await review_transport_guard.record_success()
                        invalid_review_responses.append(
                            f"{type(exc).__name__}: {exc}"
                        )
                        continue
                    break

                draft_graph_sha256 = _source_graph_sha256(draft_workflow)
                review_schema_sha256 = hashlib.sha256(
                    canonical_json_bytes(source_review_response_schema())
                ).hexdigest()
                previous_invalid_reviews = (
                    list(
                        review_audit[
                            "source_review_invalid_response_attempts"
                        ]
                    )
                    if retrying_review
                    else []
                )
                all_invalid_reviews = [
                    *previous_invalid_reviews,
                    *invalid_review_responses,
                ]
                if reviewed is None:
                    final_workflow = (
                        final_workflow
                        if retrying_review
                        else dict(draft_workflow)
                    )
                    review_status = (
                        SOURCE_REVIEW_RETRY_STATUS
                        if retrying_review
                        else "REVIEW_DRAFT_FALLBACK"
                    )
                    review_decisions: list[dict[str, Any]] = []
                    review_ledger_status = "REVIEW_LEDGER_UNAVAILABLE"
                    review_ledger_errors = [
                        (
                            "post-run review retry exhausted; draft graph retained"
                            if retrying_review
                            else "review unavailable; draft graph retained"
                        ),
                        *all_invalid_reviews,
                    ]
                    review_discarded_edges: list[str] = []
                else:
                    final_workflow = {
                        "train_index": train_index,
                        "task_id": task_id,
                        "query_text": draft_workflow["query_text"],
                        "experience_nodes": [
                            row.to_dict()
                            for row in reviewed.experience_nodes
                        ],
                        "edges": [
                            row.to_dict() for row in reviewed.edges
                        ],
                    }
                    review_status = "REVIEW_ACCEPTED"
                    review_decisions = [
                        row.to_dict() for row in reviewed.review_decisions
                    ]
                    review_ledger_status = (
                        "REVIEW_LEDGER_INCOMPLETE"
                        if reviewed.review_ledger_errors
                        else "REVIEW_LEDGER_COMPLETE"
                    )
                    review_ledger_errors = list(
                        reviewed.review_ledger_errors
                    )
                    review_discarded_edges = list(
                        reviewed.discarded_edge_reasons
                    )
                review_audit = {
                    "source_review_status": review_status,
                    "source_review_prompt_sha256": (
                        SOURCE_REVIEW_PROMPT_SHA256
                    ),
                    "source_review_protocol": review_protocol,
                    "source_review_protocol_sha256": hashlib.sha256(
                        canonical_json_bytes(review_protocol)
                    ).hexdigest(),
                    "source_review_request_payload_sha256": (
                        review_payload_sha256
                    ),
                    "source_review_response_schema_sha256": (
                        review_schema_sha256
                    ),
                    "source_review_attempt_index": len(
                        all_invalid_reviews
                    ) + (1 if reviewed is not None else 0),
                    "source_review_invalid_response_attempts": (
                        all_invalid_reviews
                    ),
                    "source_review_decisions": review_decisions,
                    "source_review_ledger_status": review_ledger_status,
                    "source_review_ledger_errors": review_ledger_errors,
                    "source_review_discarded_edge_reasons": (
                        review_discarded_edges
                    ),
                    "draft_graph_sha256": draft_graph_sha256,
                    "reviewed_graph_sha256": _source_graph_sha256(
                        final_workflow
                    ),
                }
                review_checkpoint_payload = {
                        "format": SOURCE_REVIEW_CHECKPOINT_FORMAT,
                        "draft_workflow_sha256": draft_graph_sha256,
                        "review_protocol": review_protocol,
                        "review_protocol_sha256": review_audit[
                            "source_review_protocol_sha256"
                        ],
                        "review_request_payload_sha256": (
                            review_payload_sha256
                        ),
                        "review_response_schema_sha256": (
                            review_schema_sha256
                        ),
                        "final_workflow": final_workflow,
                        "review_audit": review_audit,
                    }
                pending_publications.append(
                    (
                        review_checkpoint,
                        review_checkpoint_payload,
                        retrying_review,
                    )
                )
                status = review_status

            final_audit = {
                **draft_audit,
                "draft_graph": {
                    "experience_nodes": draft_workflow[
                        "experience_nodes"
                    ],
                    "edges": draft_workflow["edges"],
                },
                "draft_discarded_edge_reasons": draft_audit[
                    "discarded_edge_reasons"
                ],
                "discarded_edge_reasons": review_audit[
                    "source_review_discarded_edge_reasons"
                ],
                **review_audit,
            }
            if not final_workflow["experience_nodes"]:
                exclusion = {
                    **bind_active_exclusion(
                        _no_reusable_experience_exclusion(
                            train_index=train_index,
                            original=original,
                            origin=origin,
                        )
                    ),
                    **{
                        key: final_audit[key]
                        for key in final_audit
                        if key.startswith("source_review_")
                        or key
                        in {
                            "draft_graph",
                            "draft_discarded_edge_reasons",
                            "draft_graph_sha256",
                            "reviewed_graph_sha256",
                        }
                    },
                }
                result = None, exclusion
                status = str(exclusion["status"])
            else:
                result = final_workflow, final_audit
        if progress_callback is not None:
            progress_callback(
                {
                    "status": status,
                    "train_index": train_index,
                    "task_id": task_id,
                    "origin": origin,
                }
            )
        return result

    extraction_transport_wave = await extraction_transport_guard.begin_wave()
    review_transport_wave = await review_transport_guard.begin_wave()
    transport_baselines = {
        stage: frozenset(
            _source_transport_failure_paths(checkpoint_root, stage=stage)
        )
        for stage in ("source extraction", "source review")
    }
    extracted: tuple[
        tuple[dict[str, Any] | None, dict[str, Any]], ...
    ] | None = None
    gather_error: BaseException | None = None
    try:
        extracted = await gather_cancel_on_error(
            tuple(
                extract_one(train_index, original, origin)
                for train_index, original, origin in active_rows
            )
        )
    except BaseException as exc:
        gather_error = exc

    systemic_failures: dict[str, SystemicProducerTransportFailure] = {}
    if isinstance(gather_error, SystemicProducerTransportFailure):
        systemic_failures[gather_error.stage] = gather_error
    for guard, wave in (
        (extraction_transport_guard, extraction_transport_wave),
        (review_transport_guard, review_transport_wave),
    ):
        try:
            await guard.raise_if_systemic(wave)
        except SystemicProducerTransportFailure as exc:
            systemic_failures.setdefault(exc.stage, exc)
    for stage in sorted(systemic_failures):
        current_paths = tuple(
            path
            for path in _source_transport_failure_paths(
                checkpoint_root, stage=stage
            )
            if path not in transport_baselines[stage]
        )
        _archive_source_systemic_transport_wave(
            checkpoint_root,
            stage,
            current_transport_paths=current_paths,
        )
    if gather_error is not None:
        raise gather_error
    if systemic_failures:
        raise systemic_failures[sorted(systemic_failures)[0]]
    if extracted is None:
        raise AssertionError("source rebuild wave produced no result")

    publication_paths = [path for path, _payload, _replace in pending_publications]
    if len(publication_paths) != len(set(publication_paths)):
        raise ValueError("source checkpoint publication identity is duplicated")
    for path, payload, replace in sorted(
        pending_publications, key=lambda row: str(row[0])
    ):
        if replace:
            _replace_output(path, payload)
        else:
            _write_json_output(path, payload)
    workflows = [workflow for workflow, _row in extracted if workflow is not None]
    audit = [
        {**row, "status": "INGESTED"}
        for workflow, row in extracted
        if workflow is not None
    ]
    exclusions = [
        *passive_exclusions,
        *(row for workflow, row in extracted if workflow is None),
    ]
    exclusions.sort(key=lambda row: int(row["train_index"]))
    if not workflows and selected is None:
        raise ValueError("current source contains no extractable trajectory")
    return ExperienceSourceBuild(
        {
            "format": SECTION_GRAPH_FORMAT,
            "source_split": SOURCE_SPLIT,
            "workflows": workflows,
        },
        tuple(audit),
        tuple(exclusions),
    )


def rebuild_section_source_parallel(**kwargs: Any) -> ExperienceSourceBuild:
    return asyncio.run(rebuild_section_source_async(**kwargs))


def fixed_incremental_source_batches(
    build: ExperienceSourceBuild,
    full_audit: Mapping[str, Any],
) -> tuple[tuple[dict[str, Any], dict[str, Any]], ...]:
    """Split one globally-produced source build into the fixed 25x8 inputs."""
    rows = full_audit.get("rows")
    exclusions = full_audit.get("exclusions")
    if type(rows) is not list or type(exclusions) is not list:
        raise ValueError("full source audit rows differ")
    indexed = [*rows, *exclusions]
    if sorted(row.get("train_index") for row in indexed) != list(
        range(TRAIN_INSTRUCTION_COUNT)
    ):
        raise ValueError("full source audit must cover every train index once")
    workflows = {
        int(row["train_index"]): row
        for row in build.section_graphs["workflows"]
    }
    result: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for start in range(0, TRAIN_INSTRUCTION_COUNT, INCREMENTAL_BATCH_SIZE):
        indices = tuple(range(start, start + INCREMENTAL_BATCH_SIZE))
        selected = set(indices)
        batch_rows = [row for row in rows if row["train_index"] in selected]
        batch_exclusions = [
            row for row in exclusions if row["train_index"] in selected
        ]
        section_graphs = {
            "format": SECTION_GRAPH_FORMAT,
            "source_split": SOURCE_SPLIT,
            "workflows": [workflows[index] for index in indices if index in workflows],
        }
        source_sha256 = hashlib.sha256(
            canonical_json_bytes(section_graphs)
        ).hexdigest()
        review_queue = [
            row
            for row in full_audit["review_retry_queue"]
            if row["train_index"] in selected
        ]
        batch_audit = {
            **dict(full_audit),
            "section_graphs_sha256": source_sha256,
            "source_workflow_count": len(batch_rows),
            "incremental_batch_size": INCREMENTAL_BATCH_SIZE,
            "batch_train_indices": list(indices),
            "review_retry_queue_count": len(review_queue),
            "review_retry_queue": review_queue,
            "review_status_counts": dict(
                sorted(Counter(row["source_review_status"] for row in batch_rows).items())
            ),
            "origin_counts": dict(
                sorted(Counter(row["origin"] for row in batch_rows).items())
            ),
            "discarded_edge_count": sum(
                len(row["discarded_edge_reasons"]) for row in batch_rows
            ),
            "excluded_source_workflow_count": len(batch_exclusions),
            "exclusion_counts": dict(
                sorted(Counter(row["status"] for row in batch_exclusions).items())
            ),
            "exclusions": batch_exclusions,
            "rows": batch_rows,
        }
        result.append((section_graphs, batch_audit))
    return tuple(result)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Re-extract a Scheme-B source from original successes and current replay outcomes."
    )
    parser.add_argument("--original-records", type=Path, required=True)
    parser.add_argument("--replay-outcomes", type=Path, required=True)
    parser.add_argument("--section-graphs-output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key-env", default="DEGS_API_KEY")
    parser.add_argument(
        "--retry-failed-reviews",
        action="store_true",
        help=(
            "Retry only REVIEW_DRAFT_FALLBACK checkpoints; never rerun "
            "source extraction or already accepted reviews."
        ),
    )
    parser.add_argument(
        "--batch-train-index",
        action="append",
        type=int,
        help=(
            "Optionally process one contiguous aligned block of exactly eight "
            "train indices; omit for one global 16-concurrent train[0,200) pass."
        ),
    )
    parser.add_argument(
        "--fixed-batch-output-dir",
        type=Path,
        help=(
            "When running the full train set, also publish the 25 fixed 8-index "
            "section-graph and audit inputs for incremental Canonical building."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    section_output = args.section_graphs_output.expanduser().absolute()
    audit_output = args.audit_output.expanduser().absolute()
    if (
        section_output == audit_output
        or (section_output.exists() and not section_output.is_file())
        or (audit_output.exists() and not audit_output.is_file())
    ):
        raise FileExistsError(
            "experience source and audit outputs must be distinct regular files"
        )
    validate_service_url(args.base_url)
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise ValueError(f"missing generation API key: set {args.api_key_env}")
    original_path = args.original_records.expanduser().absolute()
    outcome_path = args.replay_outcomes.expanduser().absolute()
    originals = _strict_json_array(original_path, label="original records")
    outcomes = _strict_json_array(outcome_path, label="replay outcomes")
    if args.fixed_batch_output_dir is not None and args.batch_train_index is not None:
        raise ValueError("fixed batch publication requires the global source run")

    def load_replay(path: str) -> Mapping[str, Any]:
        resolved = Path(path).expanduser()
        if not resolved.is_absolute():
            resolved = outcome_path.parent / resolved
        matching_attempts = [
            attempt
            for outcome in outcomes
            if type(outcome.get("attempts")) is list
            for attempt in outcome["attempts"]
            if isinstance(attempt, Mapping)
            and attempt.get("replay_trajectory_path") == path
        ]
        if len(matching_attempts) != 1:
            raise ValueError("accepted replay trajectory locator differs")
        return _read_bound_replay_trajectory(
            resolved,
            expected_sha256=matching_attempts[0].get(
                "replay_trajectory_sha256"
            ),
        )

    def extractor_factory(origin: str, raw_response_path: Path):
        client = OpenAIClient(
            model=REPAIR_SOURCE_MODEL,
            api_key=api_key,
            base_url=args.base_url,
            generation_config=_source_generation_config(),
            retry_times=PRODUCER_TRANSPORT_RETRY_WAITS,
            runtime_timeout_retries=PRODUCER_RUNTIME_TIMEOUT_RETRIES,
            timeout=REPAIR_SOURCE_TIMEOUT_SECONDS,
            trust_env=False,
        )
        if origin == "ORIGINAL_SUCCESS":
            return SuccessfulTrajectoryExperienceExtractor(
                openai_success_llm(
                    client,
                    raw_response_output=raw_response_path,
                )
            )
        if origin == "REPLAY_VALIDATED_SUCCESS":
            return ValidatedRepairExperienceExtractor(
                OpenAIJsonObjectLLM(
                    client,
                    raw_response_output=raw_response_path,
                )
            )
        raise ValueError("source extraction origin differs")

    def reviewer_factory(raw_response_path: Path) -> ExperienceSourceReviewer:
        client = OpenAIClient(
            model=REPAIR_SOURCE_MODEL,
            api_key=api_key,
            base_url=args.base_url,
            generation_config=_source_generation_config(),
            retry_times=PRODUCER_TRANSPORT_RETRY_WAITS,
            runtime_timeout_retries=PRODUCER_RUNTIME_TIMEOUT_RETRIES,
            timeout=REPAIR_SOURCE_TIMEOUT_SECONDS,
            trust_env=False,
        )
        return ExperienceSourceReviewer(
            openai_source_review_llm(
                client,
                raw_response_output=raw_response_path,
            )
        )

    def report_progress(row: Mapping[str, Any]) -> None:
        print(json.dumps(dict(row), sort_keys=True), flush=True)

    build = rebuild_section_source_parallel(
        original_records=originals,
        replay_outcomes=outcomes,
        replay_trajectory_loader=load_replay,
        extractor_factory=extractor_factory,
        reviewer_factory=reviewer_factory,
        checkpoint_dir=args.checkpoint_dir,
        retry_failed_reviews=args.retry_failed_reviews,
        selected_train_indices=args.batch_train_index,
        progress_callback=report_progress,
    )
    source_bytes = canonical_json_bytes(build.section_graphs)
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    origin_counts = Counter(row["origin"] for row in build.source_audit)
    exclusion_counts = Counter(row["status"] for row in build.source_exclusions)
    review_status_counts = Counter(
        row["source_review_status"] for row in build.source_audit
    )
    discarded_edge_count = sum(
        len(row["discarded_edge_reasons"]) for row in build.source_audit
    )
    review_retry_queue = [
        {
            "train_index": row["train_index"],
            "task_id": row["task_id"],
            "trajectory_id": row["trajectory_id"],
            "status": (
                "PENDING"
                if row["source_review_status"] == "REVIEW_DRAFT_FALLBACK"
                else "EXHAUSTED"
            ),
            "review_request_payload_sha256": row[
                "source_review_request_payload_sha256"
            ],
        }
        for row in build.source_audit
        if row["source_review_status"]
        in {"REVIEW_DRAFT_FALLBACK", SOURCE_REVIEW_RETRY_STATUS}
    ]
    audit = {
        "format": SOURCE_REBUILD_AUDIT_FORMAT,
        "source_split": SOURCE_SPLIT,
        "section_graphs_sha256": source_sha256,
        "source_workflow_count": len(build.source_audit),
        "source_extraction_workers": SOURCE_REBUILD_WORKERS,
        "source_review_workers": SOURCE_REBUILD_WORKERS,
        "incremental_batch_size": (
            INCREMENTAL_BATCH_SIZE if args.batch_train_index is not None else None
        ),
        "batch_train_indices": (
            sorted(args.batch_train_index)
            if args.batch_train_index is not None
            else None
        ),
        "source_extraction_semantic_attempt_limit": (
            SOURCE_EXTRACTION_SEMANTIC_ATTEMPTS
        ),
        "source_review_semantic_attempt_limit": (
            SOURCE_EXTRACTION_SEMANTIC_ATTEMPTS
        ),
        "producer_transport_failure_policy": (
            producer_transport_failure_policy()
        ),
        "review_retry_mode": args.retry_failed_reviews,
        "review_retry_queue_count": len(review_retry_queue),
        "review_retry_queue": review_retry_queue,
        "review_status_counts": dict(sorted(review_status_counts.items())),
        "origin_counts": dict(sorted(origin_counts.items())),
        "discarded_edge_count": discarded_edge_count,
        "excluded_source_workflow_count": len(build.source_exclusions),
        "exclusion_counts": dict(sorted(exclusion_counts.items())),
        "exclusions": list(build.source_exclusions),
        "rows": list(build.source_audit),
    }
    if args.retry_failed_reviews:
        for output, payload in (
            (section_output, build.section_graphs),
            (audit_output, audit),
        ):
            if output.exists():
                _replace_output(output, payload)
            else:
                _write_json_output(output, payload)
    else:
        _write_or_verify_output(section_output, build.section_graphs)
        _write_or_verify_output(audit_output, audit)
    if args.fixed_batch_output_dir is not None:
        batch_root = args.fixed_batch_output_dir.expanduser().absolute()
        _ensure_directory(batch_root)
        for batch_number, (section_graphs, batch_audit) in enumerate(
            fixed_incremental_source_batches(build, audit)
        ):
            batch_dir = batch_root / f"batch_{batch_number:02d}"
            _ensure_directory(batch_dir)
            _write_or_verify_output(
                batch_dir / "section_graphs.json", section_graphs
            )
            _write_or_verify_output(
                batch_dir / "source_audit.json", batch_audit
            )
    print(
        json.dumps(
            {
                "format": SECTION_GRAPH_FORMAT,
                "section_graphs_sha256": source_sha256,
                "source_workflow_count": len(build.source_audit),
                "source_extraction_workers": SOURCE_REBUILD_WORKERS,
                "source_review_workers": SOURCE_REBUILD_WORKERS,
                "incremental_batch_size": (
                    INCREMENTAL_BATCH_SIZE
                    if args.batch_train_index is not None
                    else None
                ),
                "source_extraction_semantic_attempt_limit": (
                    SOURCE_EXTRACTION_SEMANTIC_ATTEMPTS
                ),
                "source_review_semantic_attempt_limit": (
                    SOURCE_EXTRACTION_SEMANTIC_ATTEMPTS
                ),
                "review_retry_mode": args.retry_failed_reviews,
                "review_retry_queue_count": len(review_retry_queue),
                "review_status_counts": dict(
                    sorted(review_status_counts.items())
                ),
                "origin_counts": dict(sorted(origin_counts.items())),
                "discarded_edge_count": discarded_edge_count,
                "excluded_source_workflow_count": len(build.source_exclusions),
                "exclusion_counts": dict(sorted(exclusion_counts.items())),
                "section_graphs_output": str(section_output),
                "audit_output": str(audit_output),
            },
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "ExperienceSourceBuild",
    "SOURCE_EXTRACTION_SEMANTIC_ATTEMPTS",
    "SOURCE_REBUILD_AUDIT_FORMAT",
    "SOURCE_REBUILD_WORKERS",
    "fixed_incremental_source_batches",
    "SOURCE_REVIEW_RETRY_STATUS",
    "SOURCE_SYSTEMIC_TRANSPORT_ARCHIVE_FORMAT",
    "rebuild_section_source_async",
    "rebuild_section_source_parallel",
]


if __name__ == "__main__":
    raise SystemExit(main())
