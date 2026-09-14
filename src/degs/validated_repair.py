#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import hashlib
import importlib.resources
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Awaitable, Mapping, Protocol, Sequence

from react_agent.models import Message, ModelSettings, OpenAIClient
from sb_adapter.transport import validate_service_url

from .core import canonical_json_bytes
from .section_graph import (
    ExperienceEdge,
    ExperienceNode,
    _experience_edge,
    _experience_node,
)


REPAIR_EXTRACTION_KIND = "extract_validated_repair_experience_v6"
REPAIR_EXTRACTION_FORMAT = "degs_validated_repair_extraction_v6"
REPAIR_SOURCE_PROTOCOL_FORMAT = "degs_validated_repair_source_protocol_v7"
REPAIR_PROMPT_RESOURCE = "VALIDATED_REPAIR_EXPERIENCE_PROMPT_V4.txt"
REPAIR_SOURCE_MODEL = os.getenv("DEGS_MODEL", "Qwen3.5-9B-AWQ")
REPAIR_SOURCE_TEMPERATURE = 0
REPAIR_SOURCE_THINKING = False
REPAIR_SOURCE_MAX_TOKENS = 32_000
REPAIR_SOURCE_TIMEOUT_SECONDS = 600.0
PRODUCER_TRANSPORT_RETRY_WAITS = (5, 10, 30)
PRODUCER_RUNTIME_TIMEOUT_RETRIES = 1
SOURCE_RAW_RESPONSE_FORMAT = "degs_source_llm_raw_response_v2"
SYSTEMIC_TRANSPORT_DISTINCT_REQUEST_LIMIT = 3
SYSTEMIC_TRANSPORT_POLICY = (
    "http_401_403_404_immediate_else_stage_wave_zero_success_"
    "distinct_request_threshold_v1"
)


class SystemicProducerTransportFailure(RuntimeError):
    """Raised when terminal transport failures are shared across requests."""

    def __init__(
        self,
        message: str,
        *,
        stage: str,
        failed_request_ids: Sequence[str],
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.failed_request_ids = tuple(sorted(set(failed_request_ids)))


@dataclass(frozen=True)
class ProducerTransportWave:
    failure_event_offset: int
    success_count: int


class ProducerTransportGuard:
    """Classify transport failures after observing a concurrent stage wave."""

    def __init__(self, *, stage: str) -> None:
        if type(stage) is not str or not stage:
            raise ValueError("producer transport stage differs")
        self.stage = stage
        self._failed_request_ids: list[str] = []
        self._success_count = 0
        self._lock = asyncio.Lock()

    async def begin_wave(self) -> ProducerTransportWave:
        async with self._lock:
            return ProducerTransportWave(
                failure_event_offset=len(self._failed_request_ids),
                success_count=self._success_count,
            )

    async def record_success(self) -> None:
        async with self._lock:
            self._success_count += 1

    async def record_failure(self, *, request_id: str, error: Exception) -> None:
        if type(request_id) is not str or not request_id:
            raise ValueError("producer transport request identity differs")
        status_code = getattr(error, "status_code", None)
        async with self._lock:
            self._failed_request_ids.append(request_id)
        if status_code in {401, 403, 404}:
            raise SystemicProducerTransportFailure(
                f"{self.stage} producer configuration failed with HTTP {status_code}",
                stage=self.stage,
                failed_request_ids=(request_id,),
            ) from error

    async def raise_if_systemic(self, wave: ProducerTransportWave) -> None:
        if not isinstance(wave, ProducerTransportWave):
            raise TypeError("producer transport wave differs")
        async with self._lock:
            success_count = self._success_count - wave.success_count
            failed_request_ids = set(
                self._failed_request_ids[wave.failure_event_offset :]
            )
        if (
            success_count == 0
            and len(failed_request_ids)
            >= SYSTEMIC_TRANSPORT_DISTINCT_REQUEST_LIMIT
        ):
            raise SystemicProducerTransportFailure(
                f"{self.stage} producer transport failed for "
                f"{len(failed_request_ids)} distinct requests in one stage wave "
                "without a successful response",
                stage=self.stage,
                failed_request_ids=tuple(failed_request_ids),
            )


def producer_transport_failure_policy() -> dict[str, Any]:
    return {
        "policy": SYSTEMIC_TRANSPORT_POLICY,
        "immediate_http_statuses": [401, 403, 404],
        "distinct_request_limit": SYSTEMIC_TRANSPORT_DISTINCT_REQUEST_LIMIT,
        "decision_boundary": "after_concurrent_stage_wave",
        "systemic_condition": "zero_successful_transport_response",
        "systemic_resume": (
            "preserve_transport_evidence_and_reopen_failed_stage"
        ),
        "immediate_failure_cleanup": "cancel_and_drain_sibling_tasks",
    }


async def gather_cancel_on_error(
    awaitables: Sequence[Awaitable[Any]],
) -> tuple[Any, ...]:
    """Gather a wave and drain every sibling before propagating an error."""

    tasks = tuple(asyncio.create_task(awaitable) for awaitable in awaitables)
    try:
        return tuple(await asyncio.gather(*tasks))
    except BaseException:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


def _prompt_text() -> str:
    return (
        importlib.resources.files("degs")
        .joinpath("resources", REPAIR_PROMPT_RESOURCE)
        .read_text(encoding="utf-8")
        .strip()
    )


REPAIR_SYSTEM_PROMPT = _prompt_text()
REPAIR_PROMPT_SHA256 = hashlib.sha256(REPAIR_SYSTEM_PROMPT.encode("utf-8")).hexdigest()


def _source_generation_config(
    max_tokens: int = REPAIR_SOURCE_MAX_TOKENS,
) -> dict[str, Any]:
    if type(max_tokens) is not int or max_tokens <= 0:
        raise ValueError("source generation completion budget differs")
    return {
        "temperature": REPAIR_SOURCE_TEMPERATURE,
        "max_tokens": max_tokens,
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": REPAIR_SOURCE_THINKING}
        },
    }


@dataclass(frozen=True)
class ValidatedRepairMemory:
    instructions: tuple[str, ...]
    checks: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.instructions or not self.checks:
            raise ValueError("accepted repair memory count differs")
        for label, rows in (("instruction", self.instructions), ("check", self.checks)):
            if any(not isinstance(row, str) or not row.strip() or row != row.strip() for row in rows):
                raise ValueError(f"accepted repair {label} differs")

    def to_dict(self) -> dict[str, Any]:
        return {"instructions": list(self.instructions), "checks": list(self.checks)}


@dataclass(frozen=True)
class ValidatedRepairExample:
    task_id: str
    trajectory_id: str
    accepted_patch_id: str
    accepted_attempt_index: int
    memory: ValidatedRepairMemory
    successful_replay_trajectory: Mapping[str, Any]


@dataclass(frozen=True)
class RenderedSuccessfulTrajectory:
    payload: dict[str, Any]


@dataclass(frozen=True)
class ValidatedRepairExtraction:
    experience_nodes: tuple[ExperienceNode, ...]
    edges: tuple[ExperienceEdge, ...]
    discarded_edge_reasons: tuple[str, ...]
    source_protocol: dict[str, Any]
    source_protocol_sha256: str
    request_payload_sha256: str
    response_schema_sha256: str

    def experience_node_dicts(self) -> list[dict[str, Any]]:
        return [row.to_dict() for row in self.experience_nodes]

    def edge_dicts(self) -> list[dict[str, int]]:
        return [row.to_dict() for row in self.edges]


class JsonObjectLLM(Protocol):
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

    async def complete_json_async(
        self,
        *,
        kind: str,
        request_id: str,
        system_prompt: str,
        payload: Mapping[str, Any],
        response_schema: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


def _strict_json_object(text: str) -> dict[str, Any]:
    def without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            text,
            object_pairs_hook=without_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {token}")
            ),
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError("LLM response is not strict JSON") from exc
    if type(value) is not dict:
        raise ValueError("LLM response must be one JSON object")
    return value


def experience_node_schema() -> dict[str, Any]:
    contract = {
        "type": "object",
        "properties": {
            "type": {"type": "string", "minLength": 1},
            "description": {"type": "string", "minLength": 1},
        },
        "required": ["type", "description"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "operation": {"type": "string", "minLength": 1},
            "applicability": {
                "type": "array",
                "minItems": 1,
                "items": {"type": "string", "minLength": 1},
            },
            "inputs": {"type": "array", "minItems": 1, "items": contract},
            "outputs": {"type": "array", "minItems": 1, "items": contract},
        },
        "required": ["operation", "applicability", "inputs", "outputs"],
        "additionalProperties": False,
    }


def repair_response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "experience_nodes": {
                "type": "array",
                "minItems": 0,
                "items": experience_node_schema(),
            },
            "edges": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "integer", "minimum": 0},
                        "target": {"type": "integer", "minimum": 0},
                    },
                    "required": ["source", "target"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["experience_nodes", "edges"],
        "additionalProperties": False,
    }


class OpenAIJsonObjectLLM:
    def __init__(
        self,
        client: OpenAIClient,
        *,
        raw_response_output: Path | None = None,
        request_kind: str = REPAIR_EXTRACTION_KIND,
        source_protocol_format: str = REPAIR_SOURCE_PROTOCOL_FORMAT,
        prompt_sha256: str = REPAIR_PROMPT_SHA256,
        response_schema_name: str = "degs_validated_repair_experience_v5",
        expected_retry_times: tuple[int, ...] = PRODUCER_TRANSPORT_RETRY_WAITS,
        expected_runtime_timeout_retries: int = PRODUCER_RUNTIME_TIMEOUT_RETRIES,
        expected_max_tokens: int = REPAIR_SOURCE_MAX_TOKENS,
    ):
        if client.model != REPAIR_SOURCE_MODEL:
            raise ValueError("source extraction model differs from fixed protocol")
        if client.timeout != REPAIR_SOURCE_TIMEOUT_SECONDS:
            raise ValueError("source extraction timeout differs from fixed protocol")
        if (
            tuple(client.retry_times) != expected_retry_times
            or client.runtime_timeout_retries != expected_runtime_timeout_retries
        ):
            raise ValueError("source extraction retry policy differs from fixed protocol")
        if client.generation_config != _source_generation_config(
            expected_max_tokens
        ):
            raise ValueError("source extraction generation config differs from fixed protocol")
        self.client = client
        self.raw_response_output = raw_response_output
        if any(
            type(value) is not str or not value
            for value in (
                request_kind,
                source_protocol_format,
                prompt_sha256,
                response_schema_name,
            )
        ) or len(prompt_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in prompt_sha256
        ):
            raise ValueError("source LLM protocol identity differs")
        self.request_kind = request_kind
        self.source_protocol_format = source_protocol_format
        self.prompt_sha256 = prompt_sha256
        self.response_schema_name = response_schema_name
        self.max_tokens = expected_max_tokens

    @property
    def protocol_identity(self) -> Mapping[str, Any]:
        return {
            "format": self.source_protocol_format,
            "request_kind": self.request_kind,
            "model": REPAIR_SOURCE_MODEL,
            "temperature": REPAIR_SOURCE_TEMPERATURE,
            "thinking": REPAIR_SOURCE_THINKING,
            "max_tokens": self.max_tokens,
            "timeout_seconds": REPAIR_SOURCE_TIMEOUT_SECONDS,
            "generation_config": _source_generation_config(self.max_tokens),
            "retry_waits_seconds": list(self.client.retry_times),
            "runtime_timeout_retries": self.client.runtime_timeout_retries,
            "prompt_sha256": self.prompt_sha256,
            "service_url": str(self.client.base_url or ""),
        }

    def _request_parts(
        self,
        *,
        kind: str,
        system_prompt: str,
        payload: Mapping[str, Any],
        response_schema: Mapping[str, Any],
    ) -> tuple[list[Message], ModelSettings]:
        if kind != self.request_kind:
            raise ValueError("unsupported source extraction request kind")
        request = {"request_kind": kind, "payload": payload}
        messages = [
            Message(role="system", content=system_prompt),
            Message(
                role="user",
                content=json.dumps(
                    request,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            ),
        ]
        return (
            messages,
            ModelSettings(
                temperature=REPAIR_SOURCE_TEMPERATURE,
                max_tokens=self.max_tokens,
                extra_body={
                    "chat_template_kwargs": {
                        "enable_thinking": REPAIR_SOURCE_THINKING,
                    }
                },
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": self.response_schema_name,
                        "strict": True,
                        "schema": dict(response_schema),
                    },
                },
            ),
        )

    def _finish_response(
        self,
        reply: str,
        *,
        kind: str,
        request_id: str,
        system_prompt: str,
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if not isinstance(reply, str):
            raise ValueError("source extraction LLM returned a non-text response")
        if self.raw_response_output is not None:
            _write_json_output(
                self.raw_response_output,
                {
                    "format": SOURCE_RAW_RESPONSE_FORMAT,
                    "outcome": "COMPLETE",
                    "request_kind": kind,
                    "request_id": request_id,
                    "system_prompt_sha256": hashlib.sha256(
                        system_prompt.encode("utf-8")
                    ).hexdigest(),
                    "payload_sha256": hashlib.sha256(
                        canonical_json_bytes(payload)
                    ).hexdigest(),
                    "source_protocol": dict(self.protocol_identity),
                    "source_protocol_sha256": hashlib.sha256(
                        canonical_json_bytes(self.protocol_identity)
                    ).hexdigest(),
                    "response": reply,
                },
            )
        return _strict_json_object(reply)

    def complete_json(
        self,
        *,
        kind: str,
        request_id: str,
        system_prompt: str,
        payload: Mapping[str, Any],
        response_schema: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        messages, settings = self._request_parts(
            kind=kind,
            system_prompt=system_prompt,
            payload=payload,
            response_schema=response_schema,
        )
        reply = self.client.chat(messages, settings)
        if not isinstance(reply, str):
            raise ValueError("source extraction LLM returned a non-text response")
        return self._finish_response(
            reply,
            kind=kind,
            request_id=request_id,
            system_prompt=system_prompt,
            payload=payload,
        )

    async def complete_json_async(
        self,
        *,
        kind: str,
        request_id: str,
        system_prompt: str,
        payload: Mapping[str, Any],
        response_schema: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        messages, settings = self._request_parts(
            kind=kind,
            system_prompt=system_prompt,
            payload=payload,
            response_schema=response_schema,
        )
        reply = await self.client.chat_async(messages, settings)
        if not isinstance(reply, str):
            raise ValueError("source extraction LLM returned a non-text response")
        return self._finish_response(
            reply,
            kind=kind,
            request_id=request_id,
            system_prompt=system_prompt,
            payload=payload,
        )


def select_validated_repair_example(
    replay_outcome: Mapping[str, Any],
    successful_replay_trajectory: Mapping[str, Any],
) -> ValidatedRepairExample:
    from .source_replay import (
        _patch_sha256,
        _stable_patch_id,
        parse_source_replay_patch,
        validate_source_replay_outcome_protocol,
    )

    if not isinstance(replay_outcome, Mapping):
        raise ValueError("replay outcome must be an object")
    validate_source_replay_outcome_protocol(replay_outcome)
    if replay_outcome.get("status") != "REPLAY_VALIDATED_SUCCESS":
        raise ValueError("replay outcome is not a validated success")
    accepted_index = replay_outcome.get("accepted_attempt_index")
    accepted_patch_id = replay_outcome.get("accepted_patch_id")
    accepted_trajectory_id = replay_outcome.get("accepted_trajectory_id")
    task_id = replay_outcome.get("task_id")
    parent_trajectory_id = replay_outcome.get("parent_trajectory_id")
    attempts = replay_outcome.get("attempts")
    if (
        type(accepted_index) is not int
        or not 1 <= accepted_index <= 3
        or not isinstance(accepted_patch_id, str)
        or not accepted_patch_id
        or not isinstance(accepted_trajectory_id, str)
        or not accepted_trajectory_id
        or not isinstance(task_id, str)
        or not task_id
        or not isinstance(parent_trajectory_id, str)
        or not parent_trajectory_id
        or type(attempts) is not list
    ):
        raise ValueError("accepted replay identity differs")
    accepted_rows = [
        row
        for row in attempts
        if isinstance(row, Mapping) and row.get("attempt_index") == accepted_index
    ]
    if len(accepted_rows) != 1:
        raise ValueError("accepted replay attempt is missing or duplicated")
    if len(attempts) != accepted_index:
        raise ValueError("accepted replay must be the final attempt")
    if [row.get("attempt_index") for row in attempts if isinstance(row, Mapping)] != list(
        range(1, accepted_index + 1)
    ):
        raise ValueError("replay attempts must be consecutive and ordered")
    if any(
        not isinstance(row, Mapping)
        or row.get("task_id") != task_id
        or (row.get("success") is True) != (row.get("attempt_index") == accepted_index)
        for row in attempts
    ):
        raise ValueError("accepted replay must be the unique final success")
    attempt = accepted_rows[0]
    patch = attempt.get("patch")
    if (
        attempt.get("success") is not True
        or attempt.get("task_id") != task_id
        or attempt.get("patch_id") != accepted_patch_id
        or attempt.get("replay_trajectory_id") != accepted_trajectory_id
        or not isinstance(patch, Mapping)
    ):
        raise ValueError("accepted replay attempt differs from outcome")
    parsed_patch = parse_source_replay_patch(patch)
    expected_patch_sha256 = _patch_sha256(parsed_patch)
    expected_patch_id = _stable_patch_id(
        parent_trajectory_id,
        accepted_index,
        parsed_patch,
    )
    memory_payload = parsed_patch.to_dict()
    if (
        accepted_patch_id != expected_patch_id
        or attempt.get("patch_id") != expected_patch_id
        or attempt.get("patch_sha256") != expected_patch_sha256
    ):
        raise ValueError("accepted patch content identity differs")
    instructions = memory_payload.get("instructions")
    checks = memory_payload.get("checks")
    if (
        type(instructions) is not list
        or any(type(row) is not str for row in instructions)
        or type(checks) is not list
        or any(type(row) is not str for row in checks)
    ):
        raise ValueError("accepted repair memory schema differs")
    memory = ValidatedRepairMemory(
        tuple(instructions),
        tuple(checks),
    )
    if not isinstance(successful_replay_trajectory, Mapping):
        raise ValueError("successful replay trajectory must be an object")
    if (
        successful_replay_trajectory.get("success") is not True
        or successful_replay_trajectory.get("task_id") != task_id
        or successful_replay_trajectory.get("trajectory_id") != accepted_trajectory_id
    ):
        raise ValueError("successful replay trajectory differs from accepted attempt")
    replay_trajectory_sha256 = hashlib.sha256(
        canonical_json_bytes(successful_replay_trajectory)
    ).hexdigest()
    if attempt.get("replay_trajectory_sha256") != replay_trajectory_sha256:
        raise ValueError("successful replay trajectory content identity differs")
    extra = successful_replay_trajectory.get("extra")
    source_replay = extra.get("source_replay") if isinstance(extra, Mapping) else None
    if (
        not isinstance(source_replay, Mapping)
        or source_replay.get("attempt_index") != accepted_index
        or source_replay.get("patch_id") != accepted_patch_id
        or source_replay.get("source_task_id") != task_id
        or source_replay.get("parent_trajectory_id")
        != replay_outcome.get("parent_trajectory_id")
        or source_replay.get("source_replay_protocol_sha256")
        != replay_outcome.get("source_replay_protocol_sha256")
    ):
        raise ValueError("successful replay provenance differs from accepted attempt")
    copied_trajectory = json.loads(
        json.dumps(successful_replay_trajectory, ensure_ascii=False, allow_nan=False)
    )
    return ValidatedRepairExample(
        task_id=task_id,
        trajectory_id=accepted_trajectory_id,
        accepted_patch_id=accepted_patch_id,
        accepted_attempt_index=accepted_index,
        memory=memory,
        successful_replay_trajectory=copied_trajectory,
    )


def render_successful_replay(
    record: Mapping[str, Any],
) -> RenderedSuccessfulTrajectory:
    if record.get("success") is not True:
        raise ValueError("experience extraction accepts successful trajectories only")
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
        raise ValueError("successful replay trajectory schema differs")
    turns: list[dict[str, Any]] = [
        {"turn_id": "turn_task", "kind": "task", "content": instruction}
    ]
    seen_step_ids: set[int] = set()
    for step in steps:
        if not isinstance(step, Mapping):
            raise ValueError("successful replay step differs")
        step_id = step.get("step_id")
        if type(step_id) is not int or step_id < 0 or step_id in seen_step_ids:
            raise ValueError("successful replay step identity differs")
        turn_id = f"turn_{step_id:03d}"
        turns.append(
            {
                "turn_id": turn_id,
                "kind": "react_step",
                "raw_model_output": step.get("raw_model_output")
                if isinstance(step.get("raw_model_output"), str)
                else "",
                "action": step.get("action")
                if isinstance(step.get("action"), str)
                else "",
                "observation": (
                    step["observation"]
                    if isinstance(step.get("observation"), str)
                    else ""
                ),
                "action_valid": step.get("action_valid") is True,
                "tool_name": step.get("tool_name") if isinstance(step.get("tool_name"), str) else "",
            }
        )
        seen_step_ids.add(step_id)
    final = record.get("final_response")
    turns.append({"turn_id": "turn_final", "kind": "final_response", "content": final})
    if not isinstance(final, str):
        turns[-1]["content"] = ""
    return RenderedSuccessfulTrajectory({"turns": turns})


def parse_experience_nodes(value: Any) -> tuple[ExperienceNode, ...]:
    """Validate the shared ExperienceNode schema without rewriting LLM vocabulary."""

    if type(value) is not list:
        raise ValueError("experience_nodes must be an array")
    return tuple(_experience_node(row) for row in value)


def _parse_llm_experience_graph(
    value: Any,
) -> tuple[
    tuple[ExperienceNode, ...],
    tuple[ExperienceEdge, ...],
    tuple[str, ...],
]:
    if type(value) is not dict or set(value) != {"experience_nodes", "edges"}:
        raise ValueError("experience extraction response fields differ")
    nodes = parse_experience_nodes(value["experience_nodes"])
    raw_edges = value["edges"]
    if type(raw_edges) is not list:
        raise ValueError("experience edges must be an array")
    if not nodes and raw_edges:
        raise ValueError("an empty experience graph cannot contain edges")
    edges: list[ExperienceEdge] = []
    discarded: list[str] = []
    seen: set[tuple[int, int]] = set()
    for edge_index, raw in enumerate(raw_edges):
        try:
            edge = _experience_edge(raw, node_count=len(nodes))
        except ValueError as exc:
            discarded.append(f"edge[{edge_index}]: {exc}")
            continue
        pair = (edge.source, edge.target)
        if pair in seen:
            discarded.append(f"edge[{edge_index}]: duplicate experience edge")
            continue
        seen.add(pair)
        edges.append(edge)
    return (
        nodes,
        tuple(sorted(edges, key=lambda edge: (edge.source, edge.target))),
        tuple(discarded),
    )


class ValidatedRepairExperienceExtractor:
    def __init__(self, llm: JsonObjectLLM) -> None:
        self.llm = llm

    def extract(
        self,
        example: ValidatedRepairExample,
    ) -> ValidatedRepairExtraction:
        if not isinstance(example, ValidatedRepairExample):
            raise TypeError("validated repair example is required")
        rendered = render_successful_replay(example.successful_replay_trajectory)
        payload = {
            "validated_repair_memory": example.memory.to_dict(),
            "successful_replay_trajectory": rendered.payload,
        }
        response_schema = repair_response_schema()
        source_protocol = dict(self.llm.protocol_identity)
        request_payload_sha256 = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
        response_schema_sha256 = hashlib.sha256(
            canonical_json_bytes(response_schema)
        ).hexdigest()
        source_protocol_sha256 = hashlib.sha256(
            canonical_json_bytes(source_protocol)
        ).hexdigest()
        raw = self.llm.complete_json(
            kind=REPAIR_EXTRACTION_KIND,
            request_id=example.trajectory_id,
            system_prompt=REPAIR_SYSTEM_PROMPT,
            payload=payload,
            response_schema=response_schema,
        )
        nodes, edges, discarded_edge_reasons = _parse_llm_experience_graph(raw)
        return ValidatedRepairExtraction(
            nodes,
            edges,
            discarded_edge_reasons,
            source_protocol,
            source_protocol_sha256,
            request_payload_sha256,
            response_schema_sha256,
        )

    async def extract_async(
        self,
        example: ValidatedRepairExample,
    ) -> ValidatedRepairExtraction:
        if not isinstance(example, ValidatedRepairExample):
            raise TypeError("validated repair example is required")
        rendered = render_successful_replay(example.successful_replay_trajectory)
        payload = {
            "validated_repair_memory": example.memory.to_dict(),
            "successful_replay_trajectory": rendered.payload,
        }
        response_schema = repair_response_schema()
        source_protocol = dict(self.llm.protocol_identity)
        request_payload_sha256 = hashlib.sha256(
            canonical_json_bytes(payload)
        ).hexdigest()
        response_schema_sha256 = hashlib.sha256(
            canonical_json_bytes(response_schema)
        ).hexdigest()
        source_protocol_sha256 = hashlib.sha256(
            canonical_json_bytes(source_protocol)
        ).hexdigest()
        raw = await self.llm.complete_json_async(
            kind=REPAIR_EXTRACTION_KIND,
            request_id=example.trajectory_id,
            system_prompt=REPAIR_SYSTEM_PROMPT,
            payload=payload,
            response_schema=response_schema,
        )
        nodes, edges, discarded_edge_reasons = _parse_llm_experience_graph(raw)
        return ValidatedRepairExtraction(
            nodes,
            edges,
            discarded_edge_reasons,
            source_protocol,
            source_protocol_sha256,
            request_payload_sha256,
            response_schema_sha256,
        )


def load_validated_repair_example(path: Path | str) -> ValidatedRepairExample:
    outcome_path = Path(path).expanduser().absolute()
    try:
        outcome = _strict_json_object(outcome_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError("replay outcome is not readable") from exc
    accepted_index = outcome.get("accepted_attempt_index")
    attempts = outcome.get("attempts")
    accepted_rows = [
        row
        for row in attempts if isinstance(row, Mapping) and row.get("attempt_index") == accepted_index
    ] if type(attempts) is list else []
    if len(accepted_rows) != 1:
        raise ValueError("accepted replay attempt is missing or duplicated")
    trajectory_path = accepted_rows[0].get("replay_trajectory_path")
    if not isinstance(trajectory_path, str) or not trajectory_path:
        raise ValueError("accepted replay trajectory path differs")
    resolved_trajectory_path = Path(trajectory_path).expanduser()
    if not resolved_trajectory_path.is_absolute():
        resolved_trajectory_path = outcome_path.parent / resolved_trajectory_path
    trajectory = _read_bound_replay_trajectory(
        resolved_trajectory_path,
        expected_sha256=accepted_rows[0].get("replay_trajectory_sha256"),
    )
    return select_validated_repair_example(outcome, trajectory)


def _read_bound_replay_trajectory(
    path: Path,
    *,
    expected_sha256: Any,
) -> dict[str, Any]:
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise ValueError("accepted replay trajectory hash differs")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError("accepted replay trajectory is not readable") from exc
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("accepted replay trajectory bytes differ")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("accepted replay trajectory is not UTF-8") from exc
    return _strict_json_object(text)


def _write_json_output(path: Path, payload: Mapping[str, Any]) -> None:
    output = Path(path).expanduser().absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        handle.write(canonical_json_bytes(payload))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract reusable experience nodes from one accepted patch and its successful replay."
    )
    parser.add_argument("--replay-outcome", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key-env", default="DEGS_API_KEY")
    parser.add_argument(
        "--raw-response-output",
        type=Path,
        help="Optional raw model response for source-prompt diagnostics.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = args.output.expanduser().absolute()
    raw_output = (
        args.raw_response_output.expanduser().absolute()
        if args.raw_response_output is not None
        else None
    )
    if output.exists():
        raise FileExistsError("validated repair extraction output must be fresh")
    if raw_output is not None and (
        raw_output == output or raw_output.exists()
    ):
        raise FileExistsError("raw and final source outputs must be distinct and fresh")
    validate_service_url(args.base_url)
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise ValueError(f"missing generation API key: set {args.api_key_env}")
    example = load_validated_repair_example(args.replay_outcome)
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
    result = ValidatedRepairExperienceExtractor(
        OpenAIJsonObjectLLM(
            client,
            raw_response_output=raw_output,
        )
    ).extract(example)
    body = {
        "format": REPAIR_EXTRACTION_FORMAT,
        "task_id": example.task_id,
        "trajectory_id": example.trajectory_id,
        "accepted_patch_id": example.accepted_patch_id,
        "accepted_attempt_index": example.accepted_attempt_index,
        "repair_memory_sha256": hashlib.sha256(
            canonical_json_bytes(example.memory.to_dict())
        ).hexdigest(),
        "prompt_sha256": REPAIR_PROMPT_SHA256,
        "source_protocol": result.source_protocol,
        "source_protocol_sha256": result.source_protocol_sha256,
        "request_payload_sha256": result.request_payload_sha256,
        "response_schema_sha256": result.response_schema_sha256,
        "experience_nodes": result.experience_node_dicts(),
        "edges": result.edge_dicts(),
        "discarded_edge_reasons": list(result.discarded_edge_reasons),
    }
    _write_json_output(output, body)
    print(
        json.dumps(
            {
                "format": REPAIR_EXTRACTION_FORMAT,
                "experience_node_count": len(result.experience_nodes),
                "edge_count": len(result.edges),
                "discarded_edge_count": len(result.discarded_edge_reasons),
                "prompt_sha256": REPAIR_PROMPT_SHA256,
                "source_protocol_sha256": result.source_protocol_sha256,
                "output": str(output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "OpenAIJsonObjectLLM",
    "ProducerTransportGuard",
    "ProducerTransportWave",
    "REPAIR_EXTRACTION_FORMAT",
    "REPAIR_EXTRACTION_KIND",
    "REPAIR_PROMPT_SHA256",
    "REPAIR_SOURCE_MAX_TOKENS",
    "REPAIR_SOURCE_MODEL",
    "REPAIR_SOURCE_TEMPERATURE",
    "REPAIR_SOURCE_THINKING",
    "REPAIR_SOURCE_TIMEOUT_SECONDS",
    "SOURCE_RAW_RESPONSE_FORMAT",
    "SYSTEMIC_TRANSPORT_DISTINCT_REQUEST_LIMIT",
    "SYSTEMIC_TRANSPORT_POLICY",
    "SystemicProducerTransportFailure",
    "ValidatedRepairExample",
    "ValidatedRepairExtraction",
    "ValidatedRepairMemory",
    "ValidatedRepairExperienceExtractor",
    "load_validated_repair_example",
    "main",
    "render_successful_replay",
    "repair_response_schema",
    "experience_node_schema",
    "gather_cancel_on_error",
    "parse_experience_nodes",
    "producer_transport_failure_policy",
    "select_validated_repair_example",
]
