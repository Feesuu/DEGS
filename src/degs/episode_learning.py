from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import importlib.resources
from typing import Any, Mapping, Sequence

from .contextual_binding import BindingCondition, ExperienceExpectation, JsonProducer
from .core import canonical_json_bytes
from .episode_evidence import EpisodeEvidence, EpisodeOutcome
from .section_graph import ExperienceEdge, ExperienceNode, _experience_node


REFLECTION_KIND = "reflect_evidence_bounded_episode_v1"
REFLECTION_PROTOCOL_FORMAT = "degs_episode_reflection_protocol_v1"
REFLECTION_PROMPT_RESOURCE = "EPISODE_REFLECTION_PROMPT_V1.txt"


def _prompt_text() -> str:
    return (
        importlib.resources.files("degs")
        .joinpath("resources", REFLECTION_PROMPT_RESOURCE)
        .read_text(encoding="utf-8")
        .strip()
    )


REFLECTION_SYSTEM_PROMPT = _prompt_text()
REFLECTION_PROMPT_SHA256 = hashlib.sha256(
    REFLECTION_SYSTEM_PROMPT.encode("utf-8")
).hexdigest()


class UpdateAction(str, Enum):
    NO_EVIDENCE = "NO_EVIDENCE"
    SUPPORT = "SUPPORT"
    QUALIFY = "QUALIFY"
    CORRECT = "CORRECT"


class ProcedureStepKind(str, Enum):
    CANONICAL = "CANONICAL"
    NEW = "NEW"


@dataclass(frozen=True)
class ExperienceUpdate:
    canonical_id: str
    base_version: int
    action: UpdateAction
    usage_evidence_refs: tuple[str, ...]
    outcome_evidence_refs: tuple[str, ...]
    repair_evidence_refs: tuple[str, ...]
    reason: str
    revised_experience: ExperienceNode | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_id": self.canonical_id,
            "base_version": self.base_version,
            "action": self.action.value,
            "usage_evidence_refs": list(self.usage_evidence_refs),
            "outcome_evidence_refs": list(self.outcome_evidence_refs),
            "repair_evidence_refs": list(self.repair_evidence_refs),
            "reason": self.reason,
            "revised_experience": (
                None
                if self.revised_experience is None
                else self.revised_experience.to_dict()
            ),
        }


@dataclass(frozen=True)
class LearnedExperienceNode:
    experience: ExperienceNode
    evidence_refs: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {**self.experience.to_dict(), "evidence_refs": list(self.evidence_refs)}


@dataclass(frozen=True)
class ProcedureStep:
    kind: ProcedureStepKind
    canonical_id: str | None = None
    node_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "canonical_id": self.canonical_id,
            "node_index": self.node_index,
        }


@dataclass(frozen=True)
class LearningDelta:
    episode_id: str
    base_snapshot_id: str
    updates: tuple[ExperienceUpdate, ...]
    new_nodes: tuple[LearnedExperienceNode, ...]
    new_edges: tuple[ExperienceEdge, ...]
    procedure_steps: tuple[ProcedureStep, ...]
    procedure_edges: tuple[ExperienceEdge, ...]
    discarded_edge_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "base_snapshot_id": self.base_snapshot_id,
            "retrieved_experience_updates": [row.to_dict() for row in self.updates],
            "new_experience_graph": {
                "experience_nodes": [row.to_dict() for row in self.new_nodes],
                "edges": [row.to_dict() for row in self.new_edges],
            },
            "episode_procedure": {
                "steps": [row.to_dict() for row in self.procedure_steps],
                "edges": [row.to_dict() for row in self.procedure_edges],
            },
            "discarded_edge_reasons": list(self.discarded_edge_reasons),
        }


def _contract_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "type": {"type": "string", "minLength": 1},
            "description": {"type": "string", "minLength": 1},
        },
        "required": ["type", "description"],
        "additionalProperties": False,
    }


def _experience_schema(*, with_evidence: bool) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "operation": {"type": "string", "minLength": 1},
        "applicability": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string", "minLength": 1},
        },
        "inputs": {
            "type": "array",
            "minItems": 1,
            "items": _contract_schema(),
        },
        "outputs": {
            "type": "array",
            "minItems": 1,
            "items": _contract_schema(),
        },
    }
    required = ["operation", "applicability", "inputs", "outputs"]
    if with_evidence:
        properties["evidence_refs"] = {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string", "minLength": 1},
        }
        required.append("evidence_refs")
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _edge_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "source": {"type": "integer", "minimum": 0},
            "target": {"type": "integer", "minimum": 0},
        },
        "required": ["source", "target"],
        "additionalProperties": False,
    }


def learning_delta_response_schema(
    *, expectations: Sequence[ExperienceExpectation] = (),
) -> dict[str, Any]:
    ids = [row.canonical_id for row in expectations]
    canonical_schema: dict[str, Any] = {"type": "string", "minLength": 1}
    if ids:
        canonical_schema["enum"] = ids
    return {
        "type": "object",
        "properties": {
            "retrieved_experience_updates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "canonical_id": canonical_schema,
                        "base_version": {"type": "integer", "minimum": 1},
                        "action": {
                            "type": "string",
                            "enum": [row.value for row in UpdateAction],
                        },
                        "usage_evidence_refs": {
                            "type": "array",
                            "items": {"type": "string", "minLength": 1},
                        },
                        "outcome_evidence_refs": {
                            "type": "array",
                            "items": {"type": "string", "minLength": 1},
                        },
                        "repair_evidence_refs": {
                            "type": "array",
                            "items": {"type": "string", "minLength": 1},
                        },
                        "reason": {"type": "string", "minLength": 1},
                        "revised_experience": {
                            "anyOf": [_experience_schema(with_evidence=False), {"type": "null"}]
                        },
                    },
                    "required": [
                        "canonical_id",
                        "base_version",
                        "action",
                        "usage_evidence_refs",
                        "outcome_evidence_refs",
                        "repair_evidence_refs",
                        "reason",
                        "revised_experience",
                    ],
                    "additionalProperties": False,
                },
            },
            "new_experience_graph": {
                "type": "object",
                "properties": {
                    "experience_nodes": {
                        "type": "array",
                        "items": _experience_schema(with_evidence=True),
                    },
                    "edges": {"type": "array", "items": _edge_schema()},
                },
                "required": ["experience_nodes", "edges"],
                "additionalProperties": False,
            },
            "episode_procedure": {
                "type": "object",
                "properties": {
                    "steps": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "kind": {
                                    "type": "string",
                                    "enum": [row.value for row in ProcedureStepKind],
                                },
                                "canonical_id": {
                                    "type": ["string", "null"]
                                },
                                "node_index": {
                                    "type": ["integer", "null"],
                                    "minimum": 0,
                                },
                            },
                            "required": ["kind", "canonical_id", "node_index"],
                            "additionalProperties": False,
                        },
                    },
                    "edges": {"type": "array", "items": _edge_schema()},
                },
                "required": ["steps", "edges"],
                "additionalProperties": False,
            },
        },
        "required": [
            "retrieved_experience_updates",
            "new_experience_graph",
            "episode_procedure",
        ],
        "additionalProperties": False,
    }


def _clean_text(value: Any, *, label: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{label} differs")
    return value


def _refs(
    value: Any,
    *,
    label: str,
    evidence_by_id: Mapping[str, Any],
) -> tuple[str, ...]:
    if type(value) is not list or any(type(row) is not str for row in value):
        raise ValueError(f"{label} evidence differs")
    rows = tuple(value)
    if len(rows) != len(set(rows)) or any(row not in evidence_by_id for row in rows):
        raise ValueError(f"{label} evidence differs")
    return rows


def _has_ref(refs: Sequence[str], prefix: str) -> bool:
    return any(row.startswith(prefix) for row in refs)


def _parse_edges(
    value: Any,
    *,
    node_count: int,
    label: str,
) -> tuple[tuple[ExperienceEdge, ...], tuple[str, ...]]:
    if type(value) is not list:
        raise ValueError(f"{label} edges differ")
    accepted: set[tuple[int, int]] = set()
    discarded: list[str] = []
    for index, raw in enumerate(value):
        if type(raw) is not dict or set(raw) != {"source", "target"}:
            discarded.append(f"{label} edge {index} fields differ")
            continue
        source, target = raw["source"], raw["target"]
        if (
            type(source) is not int
            or type(target) is not int
            or source == target
            or not 0 <= source < node_count
            or not 0 <= target < node_count
        ):
            discarded.append(f"{label} edge {index} endpoints differ")
            continue
        accepted.add((source, target))
    return (
        tuple(ExperienceEdge(source, target) for source, target in sorted(accepted)),
        tuple(discarded),
    )


def parse_learning_delta(
    response: Mapping[str, Any],
    *,
    episode: EpisodeEvidence,
    active_experiences: Mapping[str, ExperienceNode],
) -> LearningDelta:
    if type(response) is not dict or set(response) != {
        "retrieved_experience_updates",
        "new_experience_graph",
        "episode_procedure",
    }:
        raise ValueError("learning delta response fields differ")
    evidence_by_id = episode.evidence_by_id
    expectation_by_id = {row.canonical_id: row for row in episode.expectations}
    raw_updates = response["retrieved_experience_updates"]
    if type(raw_updates) is not list:
        raise ValueError("retrieved experience updates differ")
    update_fields = {
        "canonical_id",
        "base_version",
        "action",
        "usage_evidence_refs",
        "outcome_evidence_refs",
        "repair_evidence_refs",
        "reason",
        "revised_experience",
    }
    updates: dict[str, ExperienceUpdate] = {}
    for index, raw in enumerate(raw_updates):
        if type(raw) is not dict or set(raw) != update_fields:
            raise ValueError(f"experience update {index} fields differ")
        canonical_id = _clean_text(raw["canonical_id"], label="update canonical id")
        expectation = expectation_by_id.get(canonical_id)
        if expectation is None or canonical_id in updates:
            raise ValueError("update target was not a unique retrieved anchor")
        base_version = raw["base_version"]
        if type(base_version) is not int or base_version != expectation.canonical_version:
            raise ValueError("update base version differs")
        try:
            action = UpdateAction(raw["action"])
        except (TypeError, ValueError) as exc:
            raise ValueError("experience update action differs") from exc
        if (
            expectation.condition is BindingCondition.CONFLICT
            and action is not UpdateAction.NO_EVIDENCE
        ):
            raise ValueError("CONFLICT expectation permits NO_EVIDENCE only")
        if (
            action in {UpdateAction.QUALIFY, UpdateAction.CORRECT}
            and episode.outcome is not EpisodeOutcome.REPAIR_SUCCESS
        ):
            raise ValueError("semantic revision requires repair-success evidence")
        usage_refs = _refs(
            raw["usage_evidence_refs"],
            label="usage",
            evidence_by_id=evidence_by_id,
        )
        outcome_refs = _refs(
            raw["outcome_evidence_refs"],
            label="outcome",
            evidence_by_id=evidence_by_id,
        )
        repair_refs = _refs(
            raw["repair_evidence_refs"],
            label="repair",
            evidence_by_id=evidence_by_id,
        )
        reason = _clean_text(raw["reason"], label="update reason")
        revised_raw = raw["revised_experience"]
        revised = None
        if revised_raw is not None:
            revised = _experience_node(revised_raw)
        if action in {UpdateAction.NO_EVIDENCE, UpdateAction.SUPPORT} and revised is not None:
            raise ValueError(f"{action.value} cannot revise experience content")
        if action is UpdateAction.SUPPORT:
            if (
                not episode.successful
                or not usage_refs
                or not _has_ref(usage_refs, "trace:")
                or not outcome_refs
                or not any(
                    evidence_by_id[ref].kind == "verifier_success"
                    for ref in outcome_refs
                )
            ):
                raise ValueError("SUPPORT lacks successful use evidence")
        if action in {UpdateAction.QUALIFY, UpdateAction.CORRECT}:
            if revised is None:
                raise ValueError("semantic revision requires revised experience")
            required_repair_roles = (
                _has_ref(repair_refs, "verifier:original:"),
                _has_ref(repair_refs, "patch:final"),
                _has_ref(repair_refs, "trace:replay:"),
                _has_ref(repair_refs, "verifier:replay:"),
            )
            if not all(required_repair_roles):
                raise ValueError("semantic revision repair evidence is incomplete")
            current = active_experiences.get(canonical_id)
            if current is None:
                raise ValueError("semantic revision target is unavailable")
            if action is UpdateAction.QUALIFY and (
                revised.operation != current.operation
                or revised.inputs != current.inputs
                or revised.outputs != current.outputs
                or revised.applicability == current.applicability
            ):
                raise ValueError("QUALIFY may change applicability only")
        updates[canonical_id] = ExperienceUpdate(
            canonical_id,
            base_version,
            action,
            usage_refs,
            outcome_refs,
            repair_refs,
            reason,
            revised,
        )
    if set(updates) != set(expectation_by_id):
        raise ValueError("Reflection must reconcile every retrieved anchor")

    raw_graph = response["new_experience_graph"]
    if type(raw_graph) is not dict or set(raw_graph) != {"experience_nodes", "edges"}:
        raise ValueError("new experience graph fields differ")
    raw_nodes = raw_graph["experience_nodes"]
    if type(raw_nodes) is not list:
        raise ValueError("new experience nodes differ")
    if not episode.successful and raw_nodes:
        raise ValueError("unresolved episode cannot add positive experience")
    active_payloads = {
        canonical_json_bytes(node.to_dict()): canonical_id
        for canonical_id, node in active_experiences.items()
    }
    new_nodes: list[LearnedExperienceNode] = []
    raw_to_new: dict[int, int] = {}
    discarded: list[str] = []
    for index, raw in enumerate(raw_nodes):
        if type(raw) is not dict or set(raw) != {
            "operation",
            "applicability",
            "inputs",
            "outputs",
            "evidence_refs",
        }:
            raise ValueError(f"new experience node {index} fields differ")
        refs = _refs(
            raw["evidence_refs"],
            label="new node",
            evidence_by_id=evidence_by_id,
        )
        node = _experience_node(
            {key: raw[key] for key in ("operation", "applicability", "inputs", "outputs")}
        )
        if canonical_json_bytes(node.to_dict()) in active_payloads:
            discarded.append(f"new experience node {index} exactly duplicates active experience")
            continue
        if episode.outcome is EpisodeOutcome.ORIGINAL_SUCCESS:
            if not _has_ref(refs, "trace:original:") or not _has_ref(
                refs, "verifier:original:"
            ):
                raise ValueError("original-success new node evidence is incomplete")
        elif episode.outcome is EpisodeOutcome.REPAIR_SUCCESS:
            if (
                not _has_ref(refs, "patch:final")
                or not _has_ref(refs, "trace:replay:")
                or not _has_ref(refs, "verifier:replay:")
            ):
                raise ValueError("repair-success new node evidence is incomplete")
        raw_to_new[index] = len(new_nodes)
        new_nodes.append(LearnedExperienceNode(node, refs))
    raw_new_edges, edge_discards = _parse_edges(
        raw_graph["edges"], node_count=len(raw_nodes), label="new graph"
    )
    discarded.extend(edge_discards)
    remapped_new_edges: set[tuple[int, int]] = set()
    for edge in raw_new_edges:
        if edge.source not in raw_to_new or edge.target not in raw_to_new:
            discarded.append(
                f"new graph edge {edge.source}->{edge.target} referenced a discarded duplicate node"
            )
            continue
        source = raw_to_new[edge.source]
        target = raw_to_new[edge.target]
        if source == target:
            discarded.append(
                f"new graph edge {edge.source}->{edge.target} collapsed to a self-edge"
            )
            continue
        remapped_new_edges.add((source, target))
    new_edges = tuple(
        ExperienceEdge(source, target)
        for source, target in sorted(remapped_new_edges)
    )

    raw_procedure = response["episode_procedure"]
    if type(raw_procedure) is not dict or set(raw_procedure) != {"steps", "edges"}:
        raise ValueError("episode procedure fields differ")
    raw_steps = raw_procedure["steps"]
    if type(raw_steps) is not list:
        raise ValueError("episode procedure steps differ")
    if not episode.successful and raw_steps:
        raise ValueError("unresolved episode cannot add a positive procedure")
    steps: list[ProcedureStep] = []
    raw_to_step: dict[int, int] = {}
    for index, raw in enumerate(raw_steps):
        if type(raw) is not dict or set(raw) != {
            "kind",
            "canonical_id",
            "node_index",
        }:
            raise ValueError(f"procedure step {index} fields differ")
        try:
            kind = ProcedureStepKind(raw["kind"])
        except (TypeError, ValueError) as exc:
            raise ValueError("procedure step kind differs") from exc
        canonical_id, node_index = raw["canonical_id"], raw["node_index"]
        if kind is ProcedureStepKind.CANONICAL:
            update = updates.get(canonical_id) if type(canonical_id) is str else None
            if node_index is not None or update is None or update.action is UpdateAction.NO_EVIDENCE:
                raise ValueError("procedure Canonical step lacks successful-use evidence")
            raw_to_step[index] = len(steps)
            steps.append(ProcedureStep(kind, canonical_id=canonical_id))
        else:
            if canonical_id is not None or type(node_index) is not int or not 0 <= node_index < len(raw_nodes):
                raise ValueError("procedure new-node step differs")
            if node_index not in raw_to_new:
                discarded.append(
                    f"episode procedure step {index} referenced discarded duplicate node {node_index}"
                )
                continue
            raw_to_step[index] = len(steps)
            steps.append(ProcedureStep(kind, node_index=raw_to_new[node_index]))
    raw_procedure_edges, procedure_discards = _parse_edges(
        raw_procedure["edges"], node_count=len(raw_steps), label="episode procedure"
    )
    discarded.extend(procedure_discards)
    remapped_procedure_edges: set[tuple[int, int]] = set()
    for edge in raw_procedure_edges:
        if edge.source not in raw_to_step or edge.target not in raw_to_step:
            discarded.append(
                f"episode procedure edge {edge.source}->{edge.target} referenced a discarded step"
            )
            continue
        source = raw_to_step[edge.source]
        target = raw_to_step[edge.target]
        if source == target:
            discarded.append(
                f"episode procedure edge {edge.source}->{edge.target} collapsed to a self-edge"
            )
            continue
        remapped_procedure_edges.add((source, target))
    procedure_edges = tuple(
        ExperienceEdge(source, target)
        for source, target in sorted(remapped_procedure_edges)
    )
    return LearningDelta(
        episode.episode_id,
        episode.read_snapshot_id,
        tuple(updates[row.canonical_id] for row in episode.expectations),
        tuple(new_nodes),
        new_edges,
        tuple(steps),
        procedure_edges,
        tuple(discarded),
    )


def learning_delta_from_dict(
    value: Mapping[str, Any],
    *,
    episode: EpisodeEvidence,
    active_experiences: Mapping[str, ExperienceNode],
) -> LearningDelta:
    if type(value) is not dict or set(value) != {
        "episode_id",
        "base_snapshot_id",
        "retrieved_experience_updates",
        "new_experience_graph",
        "episode_procedure",
        "discarded_edge_reasons",
    }:
        raise ValueError("stored learning delta fields differ")
    if (
        value["episode_id"] != episode.episode_id
        or value["base_snapshot_id"] != episode.read_snapshot_id
    ):
        raise ValueError("stored learning delta identity differs")
    parsed = parse_learning_delta(
        {
            "retrieved_experience_updates": value["retrieved_experience_updates"],
            "new_experience_graph": value["new_experience_graph"],
            "episode_procedure": value["episode_procedure"],
        },
        episode=episode,
        active_experiences=active_experiences,
    )
    if parsed.to_dict() != dict(value):
        raise ValueError("stored learning delta content differs")
    return parsed


class EpisodeReflectionProducer:
    def __init__(self, llm: JsonProducer) -> None:
        self.llm = llm

    async def produce(
        self,
        *,
        episode: EpisodeEvidence,
        active_experiences: Mapping[str, ExperienceNode],
    ) -> LearningDelta | None:
        if episode.outcome is EpisodeOutcome.ITEM_LOCAL_RUNTIME_FAILURE:
            return None
        response = await self.llm.complete_json_async(
            kind=REFLECTION_KIND,
            request_id=episode.episode_id,
            system_prompt=REFLECTION_SYSTEM_PROMPT,
            payload=episode.to_learning_payload(),
            response_schema=learning_delta_response_schema(
                expectations=episode.expectations
            ),
        )
        return parse_learning_delta(
            response,
            episode=episode,
            active_experiences=active_experiences,
        )


def reflection_protocol(llm: JsonProducer) -> dict[str, Any]:
    body = {
        "format": REFLECTION_PROTOCOL_FORMAT,
        "request_kind": REFLECTION_KIND,
        "prompt_sha256": REFLECTION_PROMPT_SHA256,
        "producer": dict(llm.protocol_identity),
    }
    return {
        **body,
        "sha256": hashlib.sha256(canonical_json_bytes(body)).hexdigest(),
    }


__all__ = [
    "EpisodeReflectionProducer",
    "ExperienceUpdate",
    "LearnedExperienceNode",
    "LearningDelta",
    "ProcedureStep",
    "ProcedureStepKind",
    "REFLECTION_KIND",
    "REFLECTION_PROMPT_SHA256",
    "REFLECTION_SYSTEM_PROMPT",
    "UpdateAction",
    "learning_delta_response_schema",
    "learning_delta_from_dict",
    "parse_learning_delta",
    "reflection_protocol",
]
