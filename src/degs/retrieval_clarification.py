from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.resources
from pathlib import Path
import re
from typing import Any, Sequence

from react_agent.models import OpenAIClient

from .validated_repair import OpenAIJsonObjectLLM
from .workflow_retrieval import NeedGraph


CLARIFICATION_KIND = "clarify_need_retrieval_with_input_evidence_v1"
CLARIFICATION_PROTOCOL_FORMAT = "degs_evidence_retrieval_clarification_protocol_v2"
CLARIFICATION_PROMPT_RESOURCE = "EVIDENCE_RETRIEVAL_CLARIFICATION_PROMPT_V1.txt"
CLARIFICATION_SYSTEM_PROMPT = (
    importlib.resources.files("degs")
    .joinpath("resources", CLARIFICATION_PROMPT_RESOURCE)
    .read_text(encoding="utf-8")
    .strip()
)
CLARIFICATION_PROMPT_SHA256 = hashlib.sha256(
    CLARIFICATION_SYSTEM_PROMPT.encode()
).hexdigest()
_REPRESENTATION_PATTERN = re.compile(
    r"\b(?:concatenat\w*|display\w*|format\w*|native|represent\w*|"
    r"string|text|type|value\s+kind)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class NeedRetrievalClarification:
    need_node: int
    evidence_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "need_node": self.need_node,
            "evidence_ids": list(self.evidence_ids),
        }


@dataclass(frozen=True)
class RetrievalClarificationDecision:
    decision: str
    clarifications: tuple[NeedRetrievalClarification, ...]

    @property
    def by_need_node(self) -> dict[int, tuple[str, ...]]:
        return {
            row.need_node: row.evidence_ids
            for row in self.clarifications
        }


def clarifiable_need_indices(need_graph: NeedGraph) -> tuple[int, ...]:
    indices = []
    for index, need in enumerate(need_graph.nodes):
        output_text = " ".join(
            (need.description,)
            + tuple(
                part
                for contract in need.outputs
                for part in (contract.type, contract.description)
            )
        )
        if _REPRESENTATION_PATTERN.search(output_text):
            indices.append(index)
    return tuple(indices)


def clarification_response_schema(
    need_graph: NeedGraph, *, evidence_ids: Sequence[str]
) -> dict[str, Any]:
    if not need_graph.nodes or "Q" not in evidence_ids:
        raise ValueError("clarification schema requires NeedGraph and query evidence")
    eligible = clarifiable_need_indices(need_graph)
    clarification_items: dict[str, Any] = {
        "type": "object",
        "properties": {
            "need_node": {"type": "integer", "enum": list(eligible)},
            "evidence_ids": {
                "type": "array",
                "minItems": 1,
                "items": {"type": "string", "enum": list(evidence_ids)},
            },
        },
        "required": ["need_node", "evidence_ids"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "clarifications": {
                "type": "array",
                "maxItems": len(eligible),
                "items": clarification_items,
            },
        },
        "required": ["clarifications"],
        "additionalProperties": False,
    }


def parse_retrieval_clarification(
    value: Any, *, need_graph: NeedGraph, evidence_ids: Sequence[str]
) -> RetrievalClarificationDecision:
    if type(value) is not dict or set(value) != {"clarifications"}:
        raise ValueError("retrieval clarification response fields differ")
    raw_rows = value["clarifications"]
    if type(raw_rows) is not list:
        raise ValueError("retrieval clarifications must be an array")
    if not raw_rows:
        return RetrievalClarificationDecision("KEEP", ())
    allowed = set(evidence_ids)
    eligible = set(clarifiable_need_indices(need_graph))
    rows: list[NeedRetrievalClarification] = []
    seen_nodes: set[int] = set()
    for raw in raw_rows:
        if type(raw) is not dict or set(raw) != {
            "need_node",
            "evidence_ids",
        }:
            raise ValueError("retrieval clarification fields differ")
        node_index = raw["need_node"]
        cited = raw["evidence_ids"]
        if (
            type(node_index) is not int
            or node_index not in eligible
            or node_index in seen_nodes
            or type(cited) is not list
            or not cited
            or any(type(item) is not str for item in cited)
            or len(cited) != len(set(cited))
            or not any(item != "Q" for item in cited)
            or any(item not in allowed for item in cited)
        ):
            raise ValueError("retrieval clarification evidence differs")
        seen_nodes.add(node_index)
        rows.append(NeedRetrievalClarification(node_index, tuple(cited)))
    rows.sort(key=lambda row: row.need_node)
    return RetrievalClarificationDecision("CLARIFY", tuple(rows))


def openai_retrieval_clarification_llm(
    client: OpenAIClient,
    *,
    expected_retry_times: tuple[int, ...] = (),
    expected_runtime_timeout_retries: int = 0,
) -> OpenAIJsonObjectLLM:
    return OpenAIJsonObjectLLM(
        client,
        request_kind=CLARIFICATION_KIND,
        source_protocol_format=CLARIFICATION_PROTOCOL_FORMAT,
        prompt_sha256=CLARIFICATION_PROMPT_SHA256,
        response_schema_name="degs_evidence_retrieval_clarification_v2",
        expected_retry_times=expected_retry_times,
        expected_runtime_timeout_retries=expected_runtime_timeout_retries,
    )


__all__ = [
    "CLARIFICATION_KIND",
    "CLARIFICATION_PROMPT_SHA256",
    "CLARIFICATION_PROTOCOL_FORMAT",
    "CLARIFICATION_SYSTEM_PROMPT",
    "NeedRetrievalClarification",
    "RetrievalClarificationDecision",
    "clarification_response_schema",
    "clarifiable_need_indices",
    "openai_retrieval_clarification_llm",
    "parse_retrieval_clarification",
]
