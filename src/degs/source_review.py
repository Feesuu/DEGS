from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import importlib.resources
from pathlib import Path
from typing import Any, Mapping, Sequence

from react_agent.models import OpenAIClient

from .core import canonical_json_bytes
from .section_graph import ExperienceEdge, ExperienceNode
from .validated_repair import (
    JsonObjectLLM,
    OpenAIJsonObjectLLM,
    _parse_llm_experience_graph,
    experience_node_schema,
)


SOURCE_REVIEW_KIND = "review_experience_source_graph_v2"
SOURCE_REVIEW_PROTOCOL_FORMAT = "degs_experience_source_review_protocol_v2"
SOURCE_REVIEW_PROMPT_RESOURCE = "EXPERIENCE_SOURCE_REVIEW_PROMPT_V2.txt"
SOURCE_REVIEW_RESPONSE_SCHEMA_NAME = "degs_experience_source_review_v2"


def _prompt_text() -> str:
    return (
        importlib.resources.files("degs")
        .joinpath("resources", SOURCE_REVIEW_PROMPT_RESOURCE)
        .read_text(encoding="utf-8")
        .strip()
    )


SOURCE_REVIEW_SYSTEM_PROMPT = _prompt_text()
SOURCE_REVIEW_PROMPT_SHA256 = hashlib.sha256(
    SOURCE_REVIEW_SYSTEM_PROMPT.encode("utf-8")
).hexdigest()


class ReviewDecisionKind(str, Enum):
    KEEP = "KEEP"
    REWRITE = "REWRITE"
    FOLD = "FOLD"
    REMOVE = "REMOVE"


@dataclass(frozen=True)
class SourceReviewDecision:
    draft_node: int
    decision: ReviewDecisionKind
    final_nodes: tuple[int, ...]
    basis: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "draft_node": self.draft_node,
            "decision": self.decision.value,
            "final_nodes": list(self.final_nodes),
            "basis": self.basis,
        }


@dataclass(frozen=True)
class SourceReviewResult:
    experience_nodes: tuple[ExperienceNode, ...]
    edges: tuple[ExperienceEdge, ...]
    discarded_edge_reasons: tuple[str, ...]
    review_decisions: tuple[SourceReviewDecision, ...]
    review_ledger_errors: tuple[str, ...]
    review_protocol: dict[str, Any]
    review_protocol_sha256: str
    request_payload_sha256: str
    response_schema_sha256: str

    def graph_dict(self) -> dict[str, Any]:
        return {
            "experience_nodes": [row.to_dict() for row in self.experience_nodes],
            "edges": [row.to_dict() for row in self.edges],
        }


def source_review_response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "experience_nodes": {
                "type": "array",
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
            "review_decisions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "draft_node": {"type": "integer", "minimum": 0},
                        "decision": {
                            "type": "string",
                            "enum": [row.value for row in ReviewDecisionKind],
                        },
                        "final_nodes": {
                            "type": "array",
                            "items": {"type": "integer", "minimum": 0},
                        },
                        "basis": {"type": "string", "minLength": 1},
                    },
                    "required": [
                        "draft_node",
                        "decision",
                        "final_nodes",
                        "basis",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["experience_nodes", "edges", "review_decisions"],
        "additionalProperties": False,
    }


def _review_decisions(
    value: Any,
    *,
    draft_count: int,
    final_count: int,
) -> tuple[tuple[SourceReviewDecision, ...], tuple[str, ...]]:
    if type(value) is not list:
        return (), ("review_decisions is not an array",)
    accepted: dict[int, SourceReviewDecision] = {}
    errors: list[str] = []
    for row_index, row in enumerate(value):
        if type(row) is not dict or set(row) != {
            "draft_node",
            "decision",
            "final_nodes",
            "basis",
        }:
            errors.append(f"review_decisions[{row_index}] fields differ")
            continue
        draft_node = row["draft_node"]
        raw_decision = row["decision"]
        raw_final_nodes = row["final_nodes"]
        basis = row["basis"]
        if (
            type(draft_node) is not int
            or not 0 <= draft_node < draft_count
            or draft_node in accepted
            or type(raw_decision) is not str
            or raw_decision not in {item.value for item in ReviewDecisionKind}
            or type(raw_final_nodes) is not list
            or any(type(item) is not int for item in raw_final_nodes)
            or len(raw_final_nodes) != len(set(raw_final_nodes))
            or any(not 0 <= item < final_count for item in raw_final_nodes)
            or type(basis) is not str
            or not basis.strip()
            or basis != basis.strip()
        ):
            errors.append(f"review_decisions[{row_index}] content differs")
            continue
        decision = ReviewDecisionKind(raw_decision)
        final_nodes = tuple(sorted(raw_final_nodes))
        if (
            (decision is ReviewDecisionKind.REMOVE) != (not final_nodes)
            or (
                decision is ReviewDecisionKind.KEEP
                and len(final_nodes) != 1
            )
        ):
            errors.append(f"review_decisions[{row_index}] mapping differs")
            continue
        accepted[draft_node] = SourceReviewDecision(
            draft_node,
            decision,
            final_nodes,
            basis,
        )
    for draft_node in range(draft_count):
        if draft_node not in accepted:
            errors.append(f"draft node {draft_node} has no valid disposition")
    return tuple(accepted[index] for index in sorted(accepted)), tuple(errors)


def _remove_unmapped_final_nodes(
    nodes: tuple[ExperienceNode, ...],
    edges: tuple[ExperienceEdge, ...],
    decisions: tuple[SourceReviewDecision, ...],
    *,
    draft_count: int,
    ledger_errors: tuple[str, ...],
) -> tuple[
    tuple[ExperienceNode, ...],
    tuple[ExperienceEdge, ...],
    tuple[SourceReviewDecision, ...],
    tuple[str, ...],
]:
    """Contract final nodes that a complete review ledger does not retain."""

    if ledger_errors or len(decisions) != draft_count:
        return nodes, edges, decisions, ledger_errors
    retained = {
        final_node
        for decision in decisions
        for final_node in decision.final_nodes
    }
    removed = set(range(len(nodes))) - retained
    if not removed:
        return nodes, edges, decisions, ledger_errors

    adjacency: dict[int, list[int]] = {index: [] for index in range(len(nodes))}
    for edge in edges:
        adjacency[edge.source].append(edge.target)
    contracted: set[tuple[int, int]] = set()
    for source in sorted(retained):
        pending = list(adjacency[source])
        visited_removed: set[int] = set()
        while pending:
            target = pending.pop()
            if target in retained:
                contracted.add((source, target))
                continue
            if target in visited_removed:
                continue
            visited_removed.add(target)
            pending.extend(adjacency[target])

    old_to_new = {
        old_index: new_index
        for new_index, old_index in enumerate(sorted(retained))
    }
    normalized_nodes = tuple(nodes[index] for index in sorted(retained))
    normalized_edges = tuple(
        ExperienceEdge(old_to_new[source], old_to_new[target])
        for source, target in sorted(contracted)
    )
    normalized_decisions = tuple(
        SourceReviewDecision(
            decision.draft_node,
            decision.decision,
            tuple(old_to_new[index] for index in decision.final_nodes),
            decision.basis,
        )
        for decision in decisions
    )
    normalizations = tuple(
        f"final node {index} had no review disposition and was contracted out"
        for index in sorted(removed)
    )
    return (
        normalized_nodes,
        normalized_edges,
        normalized_decisions,
        normalizations,
    )


def parse_source_review_response(
    response: Mapping[str, Any],
    *,
    draft_count: int,
) -> tuple[
    tuple[ExperienceNode, ...],
    tuple[ExperienceEdge, ...],
    tuple[str, ...],
    tuple[SourceReviewDecision, ...],
    tuple[str, ...],
]:
    if type(response) is not dict or set(response) != {
        "experience_nodes",
        "edges",
        "review_decisions",
    }:
        raise ValueError("source review response fields differ")
    nodes, edges, discarded = _parse_llm_experience_graph(
        {
            "experience_nodes": response["experience_nodes"],
            "edges": response["edges"],
        }
    )
    decisions, ledger_errors = _review_decisions(
        response["review_decisions"],
        draft_count=draft_count,
        final_count=len(nodes),
    )
    nodes, edges, decisions, ledger_errors = _remove_unmapped_final_nodes(
        nodes,
        edges,
        decisions,
        draft_count=draft_count,
        ledger_errors=ledger_errors,
    )
    return nodes, edges, discarded, decisions, ledger_errors


def source_review_payload(
    *,
    evidence_mode: str,
    evidence: Mapping[str, Any],
    draft_nodes: Sequence[ExperienceNode],
    draft_edges: Sequence[ExperienceEdge],
) -> dict[str, Any]:
    if evidence_mode not in {"ORIGINAL_SUCCESS", "REPLAY_VALIDATED_SUCCESS"}:
        raise ValueError("source review evidence mode differs")
    return {
        "evidence_mode": evidence_mode,
        "evidence": dict(evidence),
        "draft_graph": {
            "experience_nodes": [row.to_dict() for row in draft_nodes],
            "edges": [row.to_dict() for row in draft_edges],
        },
    }


class ExperienceSourceReviewer:
    def __init__(
        self,
        llm: JsonObjectLLM,
        *,
        request_kind: str = SOURCE_REVIEW_KIND,
        system_prompt: str = SOURCE_REVIEW_SYSTEM_PROMPT,
    ) -> None:
        if not request_kind or not system_prompt:
            raise ValueError("source review prompt identity differs")
        self.llm = llm
        self.request_kind = request_kind
        self.system_prompt = system_prompt

    async def review_async(
        self,
        *,
        request_id: str,
        evidence_mode: str,
        evidence: Mapping[str, Any],
        draft_nodes: Sequence[ExperienceNode],
        draft_edges: Sequence[ExperienceEdge],
    ) -> SourceReviewResult:
        payload = source_review_payload(
            evidence_mode=evidence_mode,
            evidence=evidence,
            draft_nodes=draft_nodes,
            draft_edges=draft_edges,
        )
        schema = source_review_response_schema()
        protocol = dict(self.llm.protocol_identity)
        response = await self.llm.complete_json_async(
            kind=self.request_kind,
            request_id=request_id,
            system_prompt=self.system_prompt,
            payload=payload,
            response_schema=schema,
        )
        nodes, edges, discarded, decisions, ledger_errors = (
            parse_source_review_response(response, draft_count=len(draft_nodes))
        )
        return SourceReviewResult(
            nodes,
            edges,
            discarded,
            decisions,
            ledger_errors,
            protocol,
            hashlib.sha256(canonical_json_bytes(protocol)).hexdigest(),
            hashlib.sha256(canonical_json_bytes(payload)).hexdigest(),
            hashlib.sha256(canonical_json_bytes(schema)).hexdigest(),
        )


def openai_source_review_llm(
    client: OpenAIClient,
    *,
    raw_response_output: Path | None = None,
) -> OpenAIJsonObjectLLM:
    return OpenAIJsonObjectLLM(
        client,
        raw_response_output=raw_response_output,
        request_kind=SOURCE_REVIEW_KIND,
        source_protocol_format=SOURCE_REVIEW_PROTOCOL_FORMAT,
        prompt_sha256=SOURCE_REVIEW_PROMPT_SHA256,
        response_schema_name=SOURCE_REVIEW_RESPONSE_SCHEMA_NAME,
    )


__all__ = [
    "ExperienceSourceReviewer",
    "ReviewDecisionKind",
    "SOURCE_REVIEW_KIND",
    "SOURCE_REVIEW_PROMPT_SHA256",
    "SOURCE_REVIEW_PROTOCOL_FORMAT",
    "SOURCE_REVIEW_SYSTEM_PROMPT",
    "SourceReviewDecision",
    "SourceReviewResult",
    "openai_source_review_llm",
    "parse_source_review_response",
    "source_review_payload",
    "source_review_response_schema",
]
