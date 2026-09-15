from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.resources
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping, Protocol

from react_agent.models import RequestContextLengthExceeded
from sb_adapter.transport import validate_service_url

from .core import canonical_json_bytes
from .dataset import DATASET_SHA256


SOURCE_REPLAY_PATCH_KIND = "generate_source_replay_patch_v2"
SOURCE_REPLAY_PROTOCOL_FORMAT = "degs_source_replay_protocol_v2"
SOURCE_REPLAY_OUTCOME_FORMAT = "degs_source_replay_outcome_v2"
SOURCE_REPLAY_PATCH_PROMPT_RESOURCE = "SOURCE_REPLAY_PATCH_PROMPT_V2.txt"
SOURCE_REPLAY_MAX_ATTEMPTS = 3
SOURCE_REPLAY_MODEL = os.getenv("DEGS_MODEL", "Qwen3.5-9B-AWQ")
SOURCE_REPLAY_PATCH_MAX_TOKENS = 32_000
SOURCE_REPLAY_TIMEOUT_SECONDS = 600.0
SOURCE_REPLAY_MAX_TURNS = 30
SOURCE_REPLAY_BASH_TIMEOUT_SECONDS = 120
SOURCE_REPLAY_EXECUTOR_RETRY_WAITS = (5, 10, 30)
SOURCE_REPLAY_TASK_WORKERS = 8
SOURCE_REPLAY_PATCH_RETRY_WAITS = (5, 10, 30)
SOURCE_REPLAY_PATCH_RUNTIME_TIMEOUT_RETRIES = 1
_SOURCE_REPLAY_PROTOCOL_FIELDS = {
    "format", "patch_request_kind", "patch_prompt_sha256",
    "patch_response_schema_sha256", "no_input_truncation",
    "fresh_input_each_attempt", "max_attempts", "patch_llm", "attempt_executor",
}
_PATCH_LLM_PROTOCOL_FIELDS = {
    "format", "request_kind", "model", "temperature", "thinking",
    "max_tokens", "timeout_seconds", "generation_config",
    "retry_waits_seconds", "runtime_timeout_retries", "prompt_sha256", "service_url",
}
_ATTEMPT_EXECUTOR_PROTOCOL_FIELDS = {
    "format", "dataset_json_sha256", "model", "base_url",
    "max_turns", "max_completion_tokens", "bash_timeout", "llm_timeout",
    "retry_waits", "workers", "outer_task_workers", "temperature", "thinking",
    "fresh_input_each_attempt", "context_overflow_reporting", "observation_policy",
}

_ABSOLUTE_PATH = re.compile(r"(?:^|\s)(?:/[A-Za-z0-9_.-]+|[A-Za-z]:[\\/])")
_CODE_LINE = re.compile(r"(?m)^\s*(?:def |class |from \S+ import |import \S+|```)")


def _prompt_text() -> str:
    return (
        importlib.resources.files("degs")
        .joinpath("resources", SOURCE_REPLAY_PATCH_PROMPT_RESOURCE)
        .read_text(encoding="utf-8")
        .strip()
    )


SOURCE_REPLAY_PATCH_PROMPT = _prompt_text()
SOURCE_REPLAY_PATCH_PROMPT_SHA256 = hashlib.sha256(
    SOURCE_REPLAY_PATCH_PROMPT.encode("utf-8")
).hexdigest()


class PatchJsonLLM(Protocol):
    @property
    def protocol_identity(self) -> Mapping[str, Any]: ...

    def complete_json(
        self,
        *,
        kind: str,
        request_id: str,
        system_prompt: str,
        payload: Mapping[str, Any],
        response_schema: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


class SourceReplayAttemptExecutor(Protocol):
    @property
    def protocol_identity(self) -> Mapping[str, Any]: ...

    def execute(
        self,
        *,
        failed_record: Mapping[str, Any],
        rendered_patch: str,
        patch_id: str,
        attempt_index: int,
        attempt_dir: Path,
        source_replay_protocol_sha256: str,
    ) -> "ReplayExecution": ...


@dataclass(frozen=True)
class SourceReplayPatch:
    diagnosis: str
    instructions: tuple[str, ...]
    checks: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "diagnosis": self.diagnosis,
            "instructions": list(self.instructions),
            "checks": list(self.checks),
        }


@dataclass(frozen=True)
class RenderedReplayTrajectory:
    payload: dict[str, Any]


@dataclass(frozen=True)
class ReplayExecution:
    task_id: str
    trajectory_id: str
    success: bool
    verifier_score: float
    verifier_feedback: str
    trajectory: Mapping[str, Any]
    error: str | None = None
    executor_run_protocol_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id:
            raise ValueError("replay execution task identity differs")
        if not isinstance(self.trajectory_id, str) or not self.trajectory_id:
            raise ValueError("replay execution trajectory identity differs")
        if type(self.success) is not bool:
            raise ValueError("replay execution success differs")
        if type(self.verifier_score) not in (int, float) or not math.isfinite(
            float(self.verifier_score)
        ):
            raise ValueError("replay execution verifier score differs")
        if not isinstance(self.verifier_feedback, str):
            raise ValueError("replay execution verifier feedback differs")
        if not isinstance(self.trajectory, Mapping):
            raise ValueError("replay execution trajectory differs")
        if self.error is not None and not isinstance(self.error, str):
            raise ValueError("replay execution error differs")
        if self.executor_run_protocol_sha256 is not None and (
            len(self.executor_run_protocol_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.executor_run_protocol_sha256
            )
        ):
            raise ValueError("replay executor run protocol identity differs")


def source_replay_patch_response_schema() -> dict[str, Any]:
    text_row = {"type": "string", "minLength": 1}
    return {
        "type": "object",
        "properties": {
            "diagnosis": text_row,
            "instructions": {
                "type": "array",
                "minItems": 1,
                "items": text_row,
            },
            "checks": {
                "type": "array",
                "minItems": 1,
                "items": text_row,
            },
        },
        "required": ["diagnosis", "instructions", "checks"],
        "additionalProperties": False,
    }


def _text_rows(value: Any, *, label: str) -> tuple[str, ...]:
    if type(value) is not list or not value:
        raise ValueError(f"{label} must contain at least one row")
    rows: list[str] = []
    for index, row in enumerate(value):
        if not isinstance(row, str) or not row.strip():
            raise ValueError(f"{label}[{index}] must be non-empty text")
        rows.append(row.strip())
    if len(rows) != len(set(rows)):
        raise ValueError(f"{label} must contain no duplicates")
    return tuple(rows)


def _reject_obvious_leakage(text: str, *, label: str) -> None:
    for kind, pattern in (("absolute path", _ABSOLUTE_PATH), ("executable code", _CODE_LINE)):
        if pattern.search(text):
            raise ValueError(f"{label} contains forbidden {kind}")


def parse_source_replay_patch(
    payload: Mapping[str, Any],
) -> SourceReplayPatch:
    if type(payload) is not dict or set(payload) != {
        "diagnosis",
        "instructions",
        "checks",
    }:
        raise ValueError("source replay patch fields differ")
    diagnosis = payload["diagnosis"]
    if not isinstance(diagnosis, str) or not diagnosis.strip():
        raise ValueError("source replay diagnosis must be non-empty text")
    instructions = _text_rows(payload["instructions"], label="instructions")
    checks = _text_rows(payload["checks"], label="checks")
    for label, text in (
        ("diagnosis", diagnosis),
        *(("instruction", row) for row in instructions),
        *(("check", row) for row in checks),
    ):
        _reject_obvious_leakage(text, label=label)
    return SourceReplayPatch(diagnosis.strip(), instructions, checks)


def render_patch_for_executor(patch: SourceReplayPatch) -> str:
    if not isinstance(patch, SourceReplayPatch):
        raise TypeError("validated source replay patch is required")
    instructions = "\n".join(
        f"{index}. {row}" for index, row in enumerate(patch.instructions, 1)
    )
    checks = "\n".join(f"- {row}" for row in patch.checks)
    return f"### Required corrections\n{instructions}\n\n### Required checks\n{checks}"


def _copy_mapping(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    try:
        copied = json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not strict JSON") from exc
    if type(copied) is not dict:
        raise ValueError(f"{label} must be an object")
    return copied


def _render_trajectory(
    record: Mapping[str, Any],
    *,
    required_success: bool,
) -> RenderedReplayTrajectory:
    if not isinstance(record, Mapping) or record.get("success") is not required_success:
        raise ValueError("source replay trajectory success identity differs")
    task_id = record.get("task_id")
    trajectory_id = record.get("trajectory_id")
    instruction = record.get("instruction")
    steps = record.get("steps")
    if (
        not isinstance(task_id, str)
        or not task_id
        or not isinstance(trajectory_id, str)
        or not trajectory_id
        or not isinstance(instruction, str)
        or not instruction
        or type(steps) is not list
    ):
        raise ValueError("source replay trajectory schema differs")
    turns: list[dict[str, Any]] = [
        {"turn_id": "turn_task", "kind": "task", "content": instruction}
    ]
    seen: set[int] = set()
    for step in steps:
        if not isinstance(step, Mapping):
            raise ValueError("source replay step differs")
        step_id = step.get("step_id")
        if type(step_id) is not int or step_id < 0 or step_id in seen:
            raise ValueError("source replay step identity differs")
        turns.append(
            {
                "turn_id": f"turn_{step_id:03d}",
                "kind": "react_step",
                "raw_model_output": step.get("raw_model_output")
                if isinstance(step.get("raw_model_output"), str)
                else "",
                "action": step.get("action")
                if isinstance(step.get("action"), str)
                else "",
                "observation": step.get("observation")
                if isinstance(step.get("observation"), str)
                else "",
                "action_valid": step.get("action_valid") is True,
                "tool_name": step.get("tool_name")
                if isinstance(step.get("tool_name"), str)
                else "",
            }
        )
        seen.add(step_id)
    final = record.get("final_response")
    turns.append(
        {
            "turn_id": "turn_final",
            "kind": "final_response",
            "content": final if isinstance(final, str) else "",
        }
    )
    verifier_score = record.get("verifier_score")
    verifier_feedback = record.get("verifier_feedback")
    if verifier_score is not None or verifier_feedback is not None:
        turns.append(
            {
                "turn_id": "turn_verifier",
                "kind": "verifier",
                "score": verifier_score,
                "feedback": verifier_feedback
                if isinstance(verifier_feedback, str)
                else "",
            }
        )
    return RenderedReplayTrajectory({"turns": turns})


def render_failed_trajectory(record: Mapping[str, Any]) -> RenderedReplayTrajectory:
    return _render_trajectory(record, required_success=False)


def _stable_patch_id(
    parent_trajectory_id: str,
    attempt_index: int,
    patch: SourceReplayPatch,
) -> str:
    digest = hashlib.sha256(
        canonical_json_bytes(
            {
                "parent_trajectory_id": parent_trajectory_id,
                "attempt_index": attempt_index,
                "patch": patch.to_dict(),
            }
        )
    ).hexdigest()
    return f"degs_patch_{digest[:20]}"


def _patch_sha256(patch: SourceReplayPatch) -> str:
    return hashlib.sha256(canonical_json_bytes(patch.to_dict())).hexdigest()


def _ensure_directory(path: Path, *, fresh: bool = False) -> None:
    if path.exists() and not path.is_dir():
        raise ValueError(f"source replay directory differs: {path}")
    if fresh and path.exists():
        raise FileExistsError(f"source replay directory must be fresh: {path}")
    path.mkdir(parents=True, exist_ok=not fresh)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("xb") as handle:
        handle.write(canonical_json_bytes(payload))


def _safe_task_dir(task_id: str, trajectory_id: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", task_id).strip("._") or "task"
    suffix = hashlib.sha256(trajectory_id.encode("utf-8")).hexdigest()[:12]
    return f"{stem}_{suffix}"


def validate_source_replay_outcome_protocol(outcome: Mapping[str, Any]) -> None:
    protocol = outcome.get("source_replay_protocol") if isinstance(outcome, Mapping) else None
    protocol_sha256 = (
        outcome.get("source_replay_protocol_sha256")
        if isinstance(outcome, Mapping)
        else None
    )
    actual_protocol_sha256 = (
        hashlib.sha256(canonical_json_bytes(protocol)).hexdigest()
        if type(protocol) is dict
        else None
    )
    expected_schema_sha256 = hashlib.sha256(
        canonical_json_bytes(source_replay_patch_response_schema())
    ).hexdigest()
    patch_llm = protocol.get("patch_llm") if isinstance(protocol, Mapping) else None
    attempt_executor = (
        protocol.get("attempt_executor") if isinstance(protocol, Mapping) else None
    )
    service_url = patch_llm.get("service_url") if type(patch_llm) is dict else None
    base_url = (
        attempt_executor.get("base_url")
        if type(attempt_executor) is dict
        else None
    )
    try:
        validate_service_url(service_url)
        validate_service_url(base_url)
    except (TypeError, ValueError):
        raise ValueError(
            "source replay protocol is not the current no-truncation identity"
        ) from None
    expected_patch_llm = {
        "format": SOURCE_REPLAY_PROTOCOL_FORMAT,
        "request_kind": SOURCE_REPLAY_PATCH_KIND,
        "model": SOURCE_REPLAY_MODEL,
        "temperature": 0,
        "thinking": False,
        "max_tokens": SOURCE_REPLAY_PATCH_MAX_TOKENS,
        "timeout_seconds": SOURCE_REPLAY_TIMEOUT_SECONDS,
        "generation_config": {
            "temperature": 0,
            "max_tokens": SOURCE_REPLAY_PATCH_MAX_TOKENS,
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": False}
            },
        },
        "retry_waits_seconds": list(SOURCE_REPLAY_PATCH_RETRY_WAITS),
        "runtime_timeout_retries": SOURCE_REPLAY_PATCH_RUNTIME_TIMEOUT_RETRIES,
        "prompt_sha256": SOURCE_REPLAY_PATCH_PROMPT_SHA256,
        "service_url": service_url,
    }
    expected_executor = {
        "format": "degs_fresh_replay_executor_v1",
        "dataset_json_sha256": DATASET_SHA256,
        "model": SOURCE_REPLAY_MODEL,
        "base_url": base_url,
        "max_turns": SOURCE_REPLAY_MAX_TURNS,
        "max_completion_tokens": SOURCE_REPLAY_PATCH_MAX_TOKENS,
        "bash_timeout": SOURCE_REPLAY_BASH_TIMEOUT_SECONDS,
        "llm_timeout": SOURCE_REPLAY_TIMEOUT_SECONDS,
        "retry_waits": list(SOURCE_REPLAY_EXECUTOR_RETRY_WAITS),
        "workers": 1,
        "outer_task_workers": SOURCE_REPLAY_TASK_WORKERS,
        "temperature": 0,
        "thinking": False,
        "fresh_input_each_attempt": True,
        "context_overflow_reporting": "machine_readable_marker_v1",
        "observation_policy": "full_no_truncation",
    }
    if (
        type(protocol) is not dict
        or set(protocol) != _SOURCE_REPLAY_PROTOCOL_FIELDS
        or type(patch_llm) is not dict
        or set(patch_llm) != _PATCH_LLM_PROTOCOL_FIELDS
        or patch_llm != expected_patch_llm
        or type(attempt_executor) is not dict
        or set(attempt_executor) != _ATTEMPT_EXECUTOR_PROTOCOL_FIELDS
        or attempt_executor != expected_executor
        or service_url.rstrip("/") != base_url.rstrip("/")
    ):
        raise ValueError(
            "source replay protocol is not the current no-truncation identity"
        )
    if (
        not isinstance(outcome, Mapping)
        or outcome.get("format") != SOURCE_REPLAY_OUTCOME_FORMAT
        or type(protocol) is not dict
        or protocol.get("format") != SOURCE_REPLAY_PROTOCOL_FORMAT
        or protocol.get("patch_request_kind") != SOURCE_REPLAY_PATCH_KIND
        or protocol.get("patch_prompt_sha256") != SOURCE_REPLAY_PATCH_PROMPT_SHA256
        or protocol.get("patch_response_schema_sha256") != expected_schema_sha256
        or protocol.get("no_input_truncation") is not True
        or protocol.get("fresh_input_each_attempt") is not True
        or protocol.get("max_attempts") != SOURCE_REPLAY_MAX_ATTEMPTS
        or not isinstance(patch_llm, Mapping)
        or patch_llm.get("request_kind") != SOURCE_REPLAY_PATCH_KIND
        or patch_llm.get("model") != SOURCE_REPLAY_MODEL
        or patch_llm.get("temperature") != 0
        or patch_llm.get("thinking") is not False
        or patch_llm.get("max_tokens") != SOURCE_REPLAY_PATCH_MAX_TOKENS
        or patch_llm.get("timeout_seconds") != SOURCE_REPLAY_TIMEOUT_SECONDS
        or patch_llm.get("retry_waits_seconds")
        != list(SOURCE_REPLAY_PATCH_RETRY_WAITS)
        or patch_llm.get("runtime_timeout_retries")
        != SOURCE_REPLAY_PATCH_RUNTIME_TIMEOUT_RETRIES
        or not isinstance(attempt_executor, Mapping)
        or attempt_executor.get("format") != "degs_fresh_replay_executor_v1"
        or attempt_executor.get("model") != SOURCE_REPLAY_MODEL
        or attempt_executor.get("max_turns") != SOURCE_REPLAY_MAX_TURNS
        or attempt_executor.get("max_completion_tokens")
        != SOURCE_REPLAY_PATCH_MAX_TOKENS
        or attempt_executor.get("bash_timeout")
        != SOURCE_REPLAY_BASH_TIMEOUT_SECONDS
        or attempt_executor.get("llm_timeout") != SOURCE_REPLAY_TIMEOUT_SECONDS
        or attempt_executor.get("retry_waits")
        != list(SOURCE_REPLAY_EXECUTOR_RETRY_WAITS)
        or attempt_executor.get("workers") != 1
        or attempt_executor.get("outer_task_workers")
        != SOURCE_REPLAY_TASK_WORKERS
        or attempt_executor.get("temperature") != 0
        or attempt_executor.get("thinking") is not False
        or attempt_executor.get("fresh_input_each_attempt") is not True
        or attempt_executor.get("context_overflow_reporting")
        != "machine_readable_marker_v1"
        or attempt_executor.get("observation_policy")
        != "full_no_truncation"
        or not isinstance(protocol_sha256, str)
        or actual_protocol_sha256 != protocol_sha256
    ):
        raise ValueError("source replay protocol is not the current no-truncation identity")


class SourceReplayController:
    def __init__(
        self,
        *,
        patch_llm: PatchJsonLLM,
        attempt_executor: SourceReplayAttemptExecutor,
        run_root: Path,
    ) -> None:
        if not hasattr(patch_llm, "complete_json"):
            raise TypeError("source replay patch LLM differs")
        if not hasattr(attempt_executor, "execute"):
            raise TypeError("source replay attempt executor differs")
        self.patch_llm = patch_llm
        self.attempt_executor = attempt_executor
        self.run_root = Path(run_root).expanduser().absolute()
        _ensure_directory(self.run_root)

    def _protocol(self) -> dict[str, Any]:
        schema_sha256 = hashlib.sha256(
            canonical_json_bytes(source_replay_patch_response_schema())
        ).hexdigest()
        patch_identity = _copy_mapping(
            self.patch_llm.protocol_identity, label="patch LLM protocol identity"
        )
        executor_identity = _copy_mapping(
            self.attempt_executor.protocol_identity,
            label="replay executor protocol identity",
        )
        return {
            "format": SOURCE_REPLAY_PROTOCOL_FORMAT,
            "patch_request_kind": SOURCE_REPLAY_PATCH_KIND,
            "patch_prompt_sha256": SOURCE_REPLAY_PATCH_PROMPT_SHA256,
            "patch_response_schema_sha256": schema_sha256,
            "no_input_truncation": True,
            "fresh_input_each_attempt": True,
            "max_attempts": SOURCE_REPLAY_MAX_ATTEMPTS,
            "patch_llm": patch_identity,
            "attempt_executor": executor_identity,
        }

    def recover(self, failed_record: Mapping[str, Any]) -> dict[str, Any]:
        original = _copy_mapping(failed_record, label="original failed trajectory")
        rendered_original = render_failed_trajectory(original)
        task_id = original["task_id"]
        parent_trajectory_id = original["trajectory_id"]
        task_root = self.run_root / _safe_task_dir(task_id, parent_trajectory_id)
        _ensure_directory(task_root, fresh=True)
        _write_json(task_root / "parent_failure.json", original)
        protocol = self._protocol()
        protocol_sha256 = hashlib.sha256(canonical_json_bytes(protocol)).hexdigest()
        attempts: list[dict[str, Any]] = []
        previous: dict[str, Any] | None = None
        try:
            for attempt_index in range(1, SOURCE_REPLAY_MAX_ATTEMPTS + 1):
                attempt_dir = task_root / f"attempt_{attempt_index:02d}"
                _ensure_directory(attempt_dir, fresh=True)
                previous_payload = None
                if previous is not None:
                    previous_trajectory = previous["trajectory"]
                    rendered_previous = _render_trajectory(
                        previous_trajectory, required_success=False
                    )
                    previous_payload = {
                        "patch": previous["patch"],
                        "replay_trajectory": rendered_previous.payload,
                        "verifier_score": previous["verifier_score"],
                        "verifier_feedback": previous["verifier_feedback"],
                    }
                request_payload = {
                    "attempt_index": attempt_index,
                    "task": {
                        key: original.get(key)
                        for key in (
                            "task_id",
                            "instruction",
                            "instruction_type",
                            "answer_position",
                        )
                    },
                    "original_failure": rendered_original.payload,
                    "previous_attempt": previous_payload,
                }
                _write_json(attempt_dir / "patch_request.json", request_payload)
                raw_patch = self.patch_llm.complete_json(
                    kind=SOURCE_REPLAY_PATCH_KIND,
                    request_id=f"{parent_trajectory_id}::patch::{attempt_index}",
                    system_prompt=SOURCE_REPLAY_PATCH_PROMPT,
                    payload=request_payload,
                    response_schema=source_replay_patch_response_schema(),
                )
                raw_patch_copy = _copy_mapping(raw_patch, label="source replay patch response")
                _write_json(attempt_dir / "patch_response.json", raw_patch_copy)
                patch = parse_source_replay_patch(raw_patch_copy)
                patch_id = _stable_patch_id(
                    parent_trajectory_id, attempt_index, patch
                )
                patch_sha256 = _patch_sha256(patch)
                rendered_patch = render_patch_for_executor(patch)
                execution = self.attempt_executor.execute(
                    failed_record=original,
                    rendered_patch=rendered_patch,
                    patch_id=patch_id,
                    attempt_index=attempt_index,
                    attempt_dir=attempt_dir,
                    source_replay_protocol_sha256=protocol_sha256,
                )
                if not isinstance(execution, ReplayExecution):
                    raise TypeError("source replay executor result differs")
                if execution.task_id != task_id:
                    raise ValueError("source replay execution task differs")
                trajectory = _copy_mapping(
                    execution.trajectory, label="fresh replay trajectory"
                )
                if (
                    trajectory.get("task_id") != task_id
                    or trajectory.get("trajectory_id") != execution.trajectory_id
                    or trajectory.get("success") is not execution.success
                ):
                    raise ValueError("fresh replay trajectory identity differs")
                trajectory["verifier_score"] = float(execution.verifier_score)
                trajectory["verifier_feedback"] = execution.verifier_feedback
                extra = trajectory.get("extra")
                if extra is None:
                    extra = {}
                    trajectory["extra"] = extra
                if type(extra) is not dict or "source_replay" in extra:
                    raise ValueError("fresh replay provenance field differs")
                extra["source_replay"] = {
                    "attempt_index": attempt_index,
                    "patch_id": patch_id,
                    "source_task_id": task_id,
                    "parent_trajectory_id": parent_trajectory_id,
                    "source_verifier_score": float(execution.verifier_score),
                    "source_replay_protocol_sha256": protocol_sha256,
                    "executor_run_protocol_sha256": execution.executor_run_protocol_sha256,
                }
                trajectory_path = attempt_dir / "replay_trajectory.json"
                _write_json(trajectory_path, trajectory)
                replay_trajectory_sha256 = hashlib.sha256(
                    canonical_json_bytes(trajectory)
                ).hexdigest()
                attempt = {
                    "attempt_index": attempt_index,
                    "task_id": task_id,
                    "success": execution.success,
                    "patch_id": patch_id,
                    "patch_sha256": patch_sha256,
                    "replay_trajectory_id": execution.trajectory_id,
                    "replay_trajectory_path": str(trajectory_path),
                    "replay_trajectory_sha256": replay_trajectory_sha256,
                    "patch": patch.to_dict(),
                    "verifier_score": float(execution.verifier_score),
                    "verifier_feedback": execution.verifier_feedback,
                    "executor_run_protocol_sha256": execution.executor_run_protocol_sha256,
                    "error": execution.error,
                }
                _write_json(attempt_dir / "attempt.json", attempt)
                attempts.append(attempt)
                if execution.error == "CONTEXT_LENGTH_EXCEEDED":
                    outcome = self._outcome(
                        task_id,
                        parent_trajectory_id,
                        "REPLAY_RUNTIME_FAILURE",
                        attempts,
                        protocol,
                        protocol_sha256,
                        error="CONTEXT_LENGTH_EXCEEDED",
                    )
                    _write_json(task_root / "replay_outcome.json", outcome)
                    return outcome
                if execution.success:
                    outcome = self._outcome(
                        task_id,
                        parent_trajectory_id,
                        "REPLAY_VALIDATED_SUCCESS",
                        attempts,
                        protocol,
                        protocol_sha256,
                        accepted=attempt,
                    )
                    _write_json(task_root / "replay_outcome.json", outcome)
                    return outcome
                previous = {**attempt, "trajectory": trajectory}
        except RequestContextLengthExceeded:
            outcome = self._outcome(
                task_id,
                parent_trajectory_id,
                "REPLAY_RUNTIME_FAILURE",
                attempts,
                protocol,
                protocol_sha256,
                error="CONTEXT_LENGTH_EXCEEDED",
            )
            _write_json(task_root / "replay_outcome.json", outcome)
            return outcome
        except Exception as exc:
            from .validated_repair import SystemicProducerTransportFailure

            if isinstance(exc, SystemicProducerTransportFailure):
                raise
            outcome = self._outcome(
                task_id,
                parent_trajectory_id,
                "REPLAY_RUNTIME_FAILURE",
                attempts,
                protocol,
                protocol_sha256,
                error=f"{type(exc).__name__}: {exc}",
            )
            _write_json(task_root / "replay_outcome.json", outcome)
            return outcome
        outcome = self._outcome(
            task_id,
            parent_trajectory_id,
            "REPLAY_EXHAUSTED",
            attempts,
            protocol,
            protocol_sha256,
        )
        _write_json(task_root / "replay_outcome.json", outcome)
        return outcome

    @staticmethod
    def _outcome(
        task_id: str,
        parent_trajectory_id: str,
        status: str,
        attempts: list[dict[str, Any]],
        protocol: dict[str, Any],
        protocol_sha256: str,
        *,
        accepted: Mapping[str, Any] | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        return {
            "format": SOURCE_REPLAY_OUTCOME_FORMAT,
            "task_id": task_id,
            "parent_trajectory_id": parent_trajectory_id,
            "status": status,
            "accepted_trajectory_id": accepted.get("replay_trajectory_id") if accepted else None,
            "accepted_attempt_index": accepted.get("attempt_index") if accepted else None,
            "accepted_patch_id": accepted.get("patch_id") if accepted else None,
            "attempts": attempts,
            "source_replay_protocol": protocol,
            "source_replay_protocol_sha256": protocol_sha256,
            "error": error,
        }


__all__ = [
    "SOURCE_REPLAY_BASH_TIMEOUT_SECONDS",
    "SOURCE_REPLAY_EXECUTOR_RETRY_WAITS",
    "SOURCE_REPLAY_MAX_ATTEMPTS",
    "SOURCE_REPLAY_MAX_TURNS",
    "SOURCE_REPLAY_MODEL",
    "SOURCE_REPLAY_OUTCOME_FORMAT",
    "SOURCE_REPLAY_PATCH_KIND",
    "SOURCE_REPLAY_PATCH_MAX_TOKENS",
    "SOURCE_REPLAY_PATCH_PROMPT",
    "SOURCE_REPLAY_PATCH_PROMPT_SHA256",
    "SOURCE_REPLAY_PROTOCOL_FORMAT",
    "SOURCE_REPLAY_TIMEOUT_SECONDS",
    "SOURCE_REPLAY_TASK_WORKERS",
    "ReplayExecution",
    "RenderedReplayTrajectory",
    "SourceReplayAttemptExecutor",
    "SourceReplayController",
    "SourceReplayPatch",
    "parse_source_replay_patch",
    "render_failed_trajectory",
    "render_patch_for_executor",
    "source_replay_patch_response_schema",
    "validate_source_replay_outcome_protocol",
]
