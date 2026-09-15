from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import importlib.resources
from typing import Any, Mapping, Protocol, Sequence

from .core import canonical_json_bytes


BINDING_KIND = "bind_contextual_experience_v1"
BINDING_PROTOCOL_FORMAT = "degs_contextual_binding_protocol_v1"
BINDING_PROMPT_RESOURCE = "CONTEXTUAL_BINDING_PROMPT_V1.txt"


def _prompt_text() -> str:
    return (
        importlib.resources.files("degs")
        .joinpath("resources", BINDING_PROMPT_RESOURCE)
        .read_text(encoding="utf-8")
        .strip()
    )


BINDING_SYSTEM_PROMPT = _prompt_text()
BINDING_PROMPT_SHA256 = hashlib.sha256(
    BINDING_SYSTEM_PROMPT.encode("utf-8")
).hexdigest()


class JsonProducer(Protocol):
    @property
    def protocol_identity(self) -> Mapping[str, Any]: ...

    async def complete_json_async(
        self,
        *,
        kind: str,
        request_id: str,
        system_prompt: str,
        payload: Mapping[str, Any],
        response_schema: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


class BindingCondition(str, Enum):
    SATISFIED = "SATISFIED"
    UNKNOWN = "UNKNOWN"
    CONFLICT = "CONFLICT"


@dataclass(frozen=True)
class BoundParameter:
    name: str
    value: str
    source_evidence_ref: str

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "value": self.value,
            "source_evidence_ref": self.source_evidence_ref,
        }


@dataclass(frozen=True)
class ExperienceExpectation:
    canonical_id: str
    canonical_version: int
    condition: BindingCondition
    condition_evidence_refs: tuple[str, ...]
    expected_role: str
    bound_parameters: tuple[BoundParameter, ...]
    guidance: str
    expected_observation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_id": self.canonical_id,
            "canonical_version": self.canonical_version,
            "condition": self.condition.value,
            "condition_evidence_refs": list(self.condition_evidence_refs),
            "expected_role": self.expected_role,
            "bound_parameters": [row.to_dict() for row in self.bound_parameters],
            "guidance": self.guidance,
            "expected_observation": self.expected_observation,
        }


def experience_expectation_from_dict(value: Mapping[str, Any]) -> ExperienceExpectation:
    if type(value) is not dict:
        raise ValueError("stored expectation differs")
    canonical_id = value.get("canonical_id")
    version = value.get("canonical_version")
    condition = value.get("condition")
    refs = value.get("condition_evidence_refs")
    parameters = value.get("bound_parameters")
    if (
        type(canonical_id) is not str
        or not canonical_id
        or type(version) is not int
        or version <= 0
        or type(refs) is not list
        or any(type(row) is not str for row in refs)
        or len(refs) != len(set(refs))
        or type(parameters) is not list
    ):
        raise ValueError("stored expectation differs")
    try:
        parsed_condition = BindingCondition(condition)
    except (TypeError, ValueError) as exc:
        raise ValueError("stored expectation condition differs") from exc
    bound = tuple(
        BoundParameter(
            _clean_text(row.get("name"), label="bound parameter name"),
            _clean_text(row.get("value"), label="bound parameter value"),
            _clean_text(
                row.get("source_evidence_ref"), label="bound parameter evidence"
            ),
        )
        for row in parameters
        if type(row) is dict
    )
    if len(bound) != len(parameters):
        raise ValueError("stored bound parameters differ")
    result = ExperienceExpectation(
        canonical_id,
        version,
        parsed_condition,
        tuple(refs),
        _clean_text(value.get("expected_role"), label="expected role"),
        bound,
        _clean_text(value.get("guidance"), label="guidance", allow_empty=True),
        _clean_text(value.get("expected_observation"), label="expected observation"),
    )
    if result.to_dict() != dict(value):
        raise ValueError("stored expectation fields differ")
    return result


def contextual_binding_response_schema(
    *, anchor_versions: Mapping[str, int] | None = None
) -> dict[str, Any]:
    canonical_ids = sorted(anchor_versions) if anchor_versions is not None else None
    canonical_id_schema: dict[str, Any] = {"type": "string", "minLength": 1}
    if canonical_ids:
        canonical_id_schema["enum"] = canonical_ids
    return {
        "type": "object",
        "properties": {
            "expectations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "canonical_id": canonical_id_schema,
                        "canonical_version": {"type": "integer", "minimum": 1},
                        "condition": {
                            "type": "string",
                            "enum": [row.value for row in BindingCondition],
                        },
                        "condition_evidence_refs": {
                            "type": "array",
                            "items": {"type": "string", "minLength": 1},
                        },
                        "expected_role": {"type": "string", "minLength": 1},
                        "bound_parameters": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string", "minLength": 1},
                                    "value": {"type": "string", "minLength": 1},
                                    "source_evidence_ref": {
                                        "type": "string",
                                        "minLength": 1,
                                    },
                                },
                                "required": [
                                    "name",
                                    "value",
                                    "source_evidence_ref",
                                ],
                                "additionalProperties": False,
                            },
                        },
                        "guidance": {"type": "string"},
                        "expected_observation": {
                            "type": "string",
                            "minLength": 1,
                        },
                    },
                    "required": [
                        "canonical_id",
                        "canonical_version",
                        "condition",
                        "condition_evidence_refs",
                        "expected_role",
                        "bound_parameters",
                        "guidance",
                        "expected_observation",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["expectations"],
        "additionalProperties": False,
    }


def _clean_text(value: Any, *, label: str, allow_empty: bool = False) -> str:
    if type(value) is not str or value != value.strip() or (not value and not allow_empty):
        raise ValueError(f"{label} differs")
    return value


def _evidence_refs(
    value: Any,
    *,
    label: str,
    allowed: frozenset[str],
) -> tuple[str, ...]:
    if type(value) is not list or any(type(row) is not str for row in value):
        raise ValueError(f"{label} evidence differs")
    refs = tuple(value)
    if len(refs) != len(set(refs)) or any(row not in allowed for row in refs):
        raise ValueError(f"{label} evidence differs")
    return refs


def parse_experience_expectations(
    response: Mapping[str, Any],
    *,
    anchor_versions: Mapping[str, int],
    observable_evidence_ids: set[str] | frozenset[str],
) -> tuple[ExperienceExpectation, ...]:
    if type(response) is not dict or set(response) != {"expectations"}:
        raise ValueError("contextual binding response fields differ")
    if not anchor_versions:
        if response["expectations"] != []:
            raise ValueError("empty retrieval cannot produce expectations")
        return ()
    if any(
        type(key) is not str
        or not key
        or type(version) is not int
        or version <= 0
        for key, version in anchor_versions.items()
    ):
        raise ValueError("anchor version identity differs")
    allowed_evidence = frozenset(observable_evidence_ids)
    raw_rows = response["expectations"]
    if type(raw_rows) is not list:
        raise ValueError("contextual binding expectations differ")
    parsed: dict[str, ExperienceExpectation] = {}
    required_fields = {
        "canonical_id",
        "canonical_version",
        "condition",
        "condition_evidence_refs",
        "expected_role",
        "bound_parameters",
        "guidance",
        "expected_observation",
    }
    for index, raw in enumerate(raw_rows):
        if type(raw) is not dict or set(raw) != required_fields:
            raise ValueError(f"expectation {index} fields differ")
        canonical_id = _clean_text(raw["canonical_id"], label="canonical id")
        if canonical_id not in anchor_versions or canonical_id in parsed:
            raise ValueError("expectation anchor identity differs")
        version = raw["canonical_version"]
        if type(version) is not int or version != anchor_versions[canonical_id]:
            raise ValueError("expectation canonical version differs")
        try:
            condition = BindingCondition(raw["condition"])
        except (TypeError, ValueError) as exc:
            raise ValueError("expectation condition differs") from exc
        evidence_refs = _evidence_refs(
            raw["condition_evidence_refs"],
            label="condition",
            allowed=allowed_evidence,
        )
        expected_role = _clean_text(raw["expected_role"], label="expected role")
        expected_observation = _clean_text(
            raw["expected_observation"], label="expected observation"
        )
        guidance = _clean_text(
            raw["guidance"], label="guidance", allow_empty=True
        )
        if condition is BindingCondition.CONFLICT and guidance:
            raise ValueError("conflict expectation cannot emit guidance")
        if condition is not BindingCondition.CONFLICT and not guidance:
            raise ValueError("usable expectation requires guidance")
        raw_parameters = raw["bound_parameters"]
        if type(raw_parameters) is not list:
            raise ValueError("bound parameters differ")
        parameters: list[BoundParameter] = []
        names: set[str] = set()
        for parameter_index, parameter in enumerate(raw_parameters):
            if type(parameter) is not dict or set(parameter) != {
                "name",
                "value",
                "source_evidence_ref",
            }:
                raise ValueError(f"bound parameter {parameter_index} fields differ")
            name = _clean_text(parameter["name"], label="bound parameter name")
            value = _clean_text(parameter["value"], label="bound parameter value")
            source = _clean_text(
                parameter["source_evidence_ref"], label="bound parameter evidence"
            )
            if name in names or source not in allowed_evidence:
                raise ValueError("bound parameter evidence differs")
            names.add(name)
            parameters.append(BoundParameter(name, value, source))
        parsed[canonical_id] = ExperienceExpectation(
            canonical_id,
            version,
            condition,
            evidence_refs,
            expected_role,
            tuple(parameters),
            guidance,
            expected_observation,
        )
    if set(parsed) != set(anchor_versions):
        raise ValueError("binding must decide every retrieved anchor")
    return tuple(parsed[canonical_id] for canonical_id in anchor_versions)


def render_bound_guidance(
    expectations: Sequence[ExperienceExpectation],
) -> str:
    rows = [
        row.guidance
        for row in expectations
        if row.condition is not BindingCondition.CONFLICT and row.guidance
    ]
    if not rows:
        return ""
    return "\n".join(f"{index}. {value}" for index, value in enumerate(rows, 1))


class ContextualBindingProducer:
    def __init__(self, llm: JsonProducer) -> None:
        self.llm = llm

    async def produce(
        self,
        *,
        request_id: str,
        payload: Mapping[str, Any],
        anchor_versions: Mapping[str, int],
        observable_evidence_ids: set[str] | frozenset[str],
    ) -> tuple[ExperienceExpectation, ...]:
        if not anchor_versions:
            return ()
        response = await self.llm.complete_json_async(
            kind=BINDING_KIND,
            request_id=request_id,
            system_prompt=BINDING_SYSTEM_PROMPT,
            payload=payload,
            response_schema=contextual_binding_response_schema(
                anchor_versions=anchor_versions
            ),
        )
        return parse_experience_expectations(
            response,
            anchor_versions=anchor_versions,
            observable_evidence_ids=observable_evidence_ids,
        )


def contextual_binding_protocol(llm: JsonProducer) -> dict[str, Any]:
    body = {
        "format": BINDING_PROTOCOL_FORMAT,
        "request_kind": BINDING_KIND,
        "prompt_sha256": BINDING_PROMPT_SHA256,
        "producer": dict(llm.protocol_identity),
    }
    return {
        **body,
        "sha256": hashlib.sha256(canonical_json_bytes(body)).hexdigest(),
    }


__all__ = [
    "BINDING_KIND",
    "BINDING_PROMPT_SHA256",
    "BINDING_SYSTEM_PROMPT",
    "BindingCondition",
    "BoundParameter",
    "ContextualBindingProducer",
    "ExperienceExpectation",
    "contextual_binding_protocol",
    "contextual_binding_response_schema",
    "experience_expectation_from_dict",
    "parse_experience_expectations",
    "render_bound_guidance",
]
