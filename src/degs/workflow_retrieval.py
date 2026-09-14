from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.resources
from pathlib import Path
from typing import Any, Mapping, Sequence

from react_agent.models import OpenAIClient

from .core import cosine_similarity
from .section_graph import ExperienceGraph, IOContract, SectionGraphSource
from .validated_repair import OpenAIJsonObjectLLM


NEED_GRAPH_KIND = "extract_query_need_graph_v2"
RETRIEVAL_METHOD_ID = (
    "DEGS 0.77.41 Stable R1 Experience-SimGRAG Retrieval"
)
SELECTOR_KIND = "select_experience_simgrag_subgraph_v2"
NEED_GRAPH_PROTOCOL_FORMAT = "degs_query_need_graph_protocol_v2"
SELECTOR_PROTOCOL_FORMAT = "degs_experience_simgrag_selector_protocol_v2"
NEED_GRAPH_PROMPT_RESOURCE = "QUERY_NEED_GRAPH_PROMPT_V2.txt"
SELECTOR_PROMPT_RESOURCE = "EXPERIENCE_SIMGRAG_SELECTOR_PROMPT_V4.txt"
WORKFLOW_RECALL_K = 8


def _prompt(resource: str) -> str:
    return importlib.resources.files("degs").joinpath("resources", resource).read_text(encoding="utf-8").strip()


NEED_GRAPH_SYSTEM_PROMPT = _prompt(NEED_GRAPH_PROMPT_RESOURCE)
SELECTOR_SYSTEM_PROMPT = _prompt(SELECTOR_PROMPT_RESOURCE)
NEED_GRAPH_PROMPT_SHA256 = hashlib.sha256(NEED_GRAPH_SYSTEM_PROMPT.encode()).hexdigest()
SELECTOR_PROMPT_SHA256 = hashlib.sha256(SELECTOR_SYSTEM_PROMPT.encode()).hexdigest()


@dataclass(frozen=True)
class NeedNode:
    description: str
    applicability_context: tuple[str, ...]
    inputs: tuple[IOContract, ...]
    outputs: tuple[IOContract, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"description": self.description, "applicability_context": list(self.applicability_context), "inputs": [row.to_dict() for row in self.inputs], "outputs": [row.to_dict() for row in self.outputs]}


@dataclass(frozen=True)
class NeedEdge:
    source: int
    target: int

    def to_dict(self) -> dict[str, int]:
        return {"source": self.source, "target": self.target}


@dataclass(frozen=True)
class NeedGraph:
    nodes: tuple[NeedNode, ...]
    edges: tuple[NeedEdge, ...]
    discarded_edge_reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"need_nodes": [row.to_dict() for row in self.nodes], "edges": [row.to_dict() for row in self.edges]}


@dataclass(frozen=True)
class WorkflowRecall:
    train_index: int
    similarity: float
    rank: int

    def to_dict(self) -> dict[str, Any]:
        return {"train_index": self.train_index, "similarity": self.similarity, "rank": self.rank}


@dataclass(frozen=True)
class WorkflowFallbackCandidate:
    candidate_id: str
    train_index: int
    similarity: float
    canonical_node_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "train_index": self.train_index,
            "similarity": self.similarity,
            "canonical_node_ids": list(self.canonical_node_ids),
        }


@dataclass(frozen=True)
class NeedGuidedRetrievalResult:
    status: str
    experience: str
    audit: Mapping[str, Any]


def _nonempty(value: Any, *, label: str) -> str:
    if type(value) is not str or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a stripped non-empty string")
    return value


def _contracts(value: Any, *, label: str) -> tuple[IOContract, ...]:
    if type(value) is not list or not value:
        raise ValueError(f"{label} must be a non-empty array")
    rows = []
    for raw in value:
        if type(raw) is not dict or set(raw) != {"type", "description"}:
            raise ValueError(f"{label} contract fields differ")
        rows.append(IOContract(_nonempty(raw["type"], label=f"{label} type"), _nonempty(raw["description"], label=f"{label} description")))
    return tuple(rows)


def _contract_schema() -> dict[str, Any]:
    return {"type": "object", "properties": {"type": {"type": "string", "minLength": 1}, "description": {"type": "string", "minLength": 1}}, "required": ["type", "description"], "additionalProperties": False}


def need_graph_response_schema() -> dict[str, Any]:
    contract = _contract_schema()
    return {"type": "object", "properties": {"need_nodes": {"type": "array", "minItems": 1, "items": {"type": "object", "properties": {"description": {"type": "string", "minLength": 1}, "applicability_context": {"type": "array", "minItems": 1, "items": {"type": "string", "minLength": 1}}, "inputs": {"type": "array", "minItems": 1, "items": contract}, "outputs": {"type": "array", "minItems": 1, "items": contract}}, "required": ["description", "applicability_context", "inputs", "outputs"], "additionalProperties": False}}, "edges": {"type": "array", "items": {"type": "object", "properties": {"source": {"type": "integer", "minimum": 0}, "target": {"type": "integer", "minimum": 0}}, "required": ["source", "target"], "additionalProperties": False}}}, "required": ["need_nodes", "edges"], "additionalProperties": False}


@dataclass(frozen=True)
class SubgraphSelection:
    selected_candidate_id: str


def selector_response_schema(candidates: Sequence[Any]) -> dict[str, Any]:
    candidate_ids = [row.candidate_id for row in candidates]
    if not candidate_ids:
        raise ValueError("selector requires at least one candidate")
    return {
        "type": "object",
        "properties": {
            "selected_candidate_id": {
                "type": "string",
                "enum": candidate_ids,
            }
        },
        "required": ["selected_candidate_id"],
        "additionalProperties": False,
    }


def parse_need_graph(value: Any) -> NeedGraph:
    if type(value) is not dict or set(value) != {"need_nodes", "edges"}:
        raise ValueError("NeedGraph response fields differ")
    if type(value["need_nodes"]) is not list or not value["need_nodes"]:
        raise ValueError("NeedGraph must contain at least one NeedNode")
    nodes = []
    for raw in value["need_nodes"]:
        if type(raw) is not dict or set(raw) != {"description", "applicability_context", "inputs", "outputs"}:
            raise ValueError("NeedNode fields differ")
        applicability = raw["applicability_context"]
        if type(applicability) is not list or not applicability:
            raise ValueError("NeedNode applicability context differs")
        conditions = tuple(_nonempty(row, label="NeedNode applicability context") for row in applicability)
        if len(conditions) != len(set(conditions)):
            raise ValueError("NeedNode applicability context contains duplicates")
        nodes.append(NeedNode(_nonempty(raw["description"], label="NeedNode description"), conditions, _contracts(raw["inputs"], label="NeedNode inputs"), _contracts(raw["outputs"], label="NeedNode outputs")))
    if type(value["edges"]) is not list:
        raise ValueError("NeedGraph edges must be an array")
    edges, discarded, seen = [], [], set()
    for index, raw in enumerate(value["edges"]):
        if type(raw) is not dict or set(raw) != {"source", "target"}:
            discarded.append(f"edge[{index}]: fields differ")
            continue
        source, target = raw["source"], raw["target"]
        if type(source) is not int or type(target) is not int or not 0 <= source < target < len(nodes):
            discarded.append(f"edge[{index}]: invalid forward Need edge")
        elif (source, target) in seen:
            discarded.append(f"edge[{index}]: duplicate Need edge")
        else:
            seen.add((source, target))
            edges.append(NeedEdge(source, target))
    return NeedGraph(tuple(nodes), tuple(edges), tuple(discarded))


def openai_need_graph_llm(
    client: OpenAIClient,
    *,
    expected_retry_times: tuple[int, ...] = (),
    expected_runtime_timeout_retries: int = 0,
) -> OpenAIJsonObjectLLM:
    return OpenAIJsonObjectLLM(client, request_kind=NEED_GRAPH_KIND, source_protocol_format=NEED_GRAPH_PROTOCOL_FORMAT, prompt_sha256=NEED_GRAPH_PROMPT_SHA256, response_schema_name="degs_query_need_graph_v2", expected_retry_times=expected_retry_times, expected_runtime_timeout_retries=expected_runtime_timeout_retries)


def openai_selector_llm(
    client: OpenAIClient,
    *,
    expected_retry_times: tuple[int, ...] = (),
    expected_runtime_timeout_retries: int = 0,
) -> OpenAIJsonObjectLLM:
    return OpenAIJsonObjectLLM(client, request_kind=SELECTOR_KIND, source_protocol_format=SELECTOR_PROTOCOL_FORMAT, prompt_sha256=SELECTOR_PROMPT_SHA256, response_schema_name="degs_experience_simgrag_selector_v2", expected_retry_times=expected_retry_times, expected_runtime_timeout_retries=expected_runtime_timeout_retries)


def recall_source_workflows(query_vector: Sequence[float], train_query_vectors: Mapping[int, Sequence[float]], *, top_k: int = WORKFLOW_RECALL_K) -> tuple[WorkflowRecall, ...]:
    if type(top_k) is not int or top_k <= 0:
        raise ValueError("workflow recall top-k must be positive")
    ranked = sorted(((float(cosine_similarity(query_vector, vector)), train_index) for train_index, vector in train_query_vectors.items()), key=lambda row: (-row[0], row[1]))[:top_k]
    return tuple(WorkflowRecall(train_index, similarity, rank) for rank, (similarity, train_index) in enumerate(ranked))


def selector_payload(
    *,
    query_text: str,
    need_graph: NeedGraph,
    workflow_recalls: Sequence[WorkflowRecall],
    candidates: Sequence[Any],
    graph: ExperienceGraph,
    source: SectionGraphSource,
) -> dict[str, Any]:
    node_by_id = graph.node_by_id
    workflow_by_index = source.workflow_by_index

    def occurrence_payload(candidate: Any) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for evidence in candidate.occurrence_evidence:
            workflow_index, node_index = evidence.occurrence_id
            workflow = workflow_by_index[workflow_index]
            rows.append(
                {
                    **evidence.to_dict(),
                    "source_task_id": workflow.task_id,
                    "source_experience": workflow.experience_nodes[node_index].to_dict(),
                }
            )
        return rows

    return {
        "query": query_text,
        "need_graph": need_graph.to_dict(),
        "workflow_recalls": [row.to_dict() for row in workflow_recalls],
        "candidates": [
            {
                **candidate.to_dict(),
                "canonical_nodes": [
                    node_by_id[canonical_id].to_dict()
                    for canonical_id in candidate.canonical_node_ids
                ],
                "source_occurrence_evidence": occurrence_payload(candidate),
            }
            for candidate in candidates
        ],
    }


def workflow_fallback_candidates(
    workflow_recalls: Sequence[WorkflowRecall],
    *,
    canonical_ids_by_workflow: Mapping[int, Sequence[str]],
) -> tuple[WorkflowFallbackCandidate, ...]:
    rows = [
        WorkflowFallbackCandidate(
            f"C{position}",
            recall.train_index,
            recall.similarity,
            tuple(sorted(canonical_ids_by_workflow[recall.train_index])),
        )
        for position, recall in enumerate(workflow_recalls)
        if recall.train_index in canonical_ids_by_workflow
    ]
    if not rows:
        raise ValueError("workflow fallback has no source-backed candidate")
    return tuple(rows)


def workflow_fallback_selector_payload(
    *,
    query_text: str,
    candidates: Sequence[WorkflowFallbackCandidate],
    graph: ExperienceGraph,
    source: SectionGraphSource,
    fallback_mode: str = "NEED_GRAPH_UNAVAILABLE",
) -> dict[str, Any]:
    node_by_id = graph.node_by_id
    workflow_by_index = source.workflow_by_index
    return {
        "query": query_text,
        "fallback_mode": fallback_mode,
        "candidates": [
            {
                **candidate.to_dict(),
                "canonical_nodes": [
                    node_by_id[canonical_id].to_dict()
                    for canonical_id in candidate.canonical_node_ids
                ],
                "source_experience_nodes": [
                    node.to_dict()
                    for node in workflow_by_index[
                        candidate.train_index
                    ].experience_nodes
                ],
                "source_edges": [
                    edge.to_dict()
                    for edge in workflow_by_index[candidate.train_index].edges
                ],
            }
            for candidate in candidates
        ],
    }


def parse_selector_response(
    value: Any,
    *,
    candidates: Sequence[Any],
) -> SubgraphSelection:
    if type(value) is not dict or set(value) != {"selected_candidate_id"}:
        raise ValueError("selector response fields differ")
    selected = value["selected_candidate_id"]
    if type(selected) is not str or selected not in {
        row.candidate_id for row in candidates
    }:
        raise ValueError("selector selected an unverified candidate")
    return SubgraphSelection(selected)


def _render_subgraph_candidate(
    candidate: Any,
    *,
    need_graph: NeedGraph,
    graph: ExperienceGraph,
    source: SectionGraphSource,
) -> str:
    canonical_ids = tuple(sorted(candidate.canonical_node_ids))
    labels = {key: f"T{i}" for i, key in enumerate(canonical_ids)}
    lines = [
        "EXPERIENCE-SIMGRAG SUBGRAPH",
        "Templates may be instantiated more than once. This is not a global execution order.",
        "Use an operation only when its applicability and contracts match the current task.",
    ]
    for key in canonical_ids:
        experience = graph.node_by_id[key].experience
        lines.extend([
            f"TEMPLATE {labels[key]}", f"Operation: {experience.operation}", "Applicability:",
            *(f"- {row}" for row in experience.applicability), "Inputs:",
            *(f"- {row.type}: {row.description}" for row in experience.inputs), "Outputs:",
            *(f"- {row.type}: {row.description}" for row in experience.outputs),
        ])
    lines.append("NEED SLOTS (separate operation instances):")
    for need_index, key in candidate.need_mapping:
        need = need_graph.nodes[need_index]
        lines.extend(
            [
                f"NeedNode {need_index} uses {labels[key]}",
                f"Requested micro-operation: {need.description}",
                "Need applicability: " + "; ".join(need.applicability_context),
            ]
        )
    lines.append("SOURCE OCCURRENCE EVIDENCE:")
    workflow_by_index = source.workflow_by_index
    for evidence in candidate.occurrence_evidence:
        workflow_index, node_index = evidence.occurrence_id
        node = workflow_by_index[workflow_index].experience_nodes[node_index]
        lines.extend(
            [
                f"- {labels[evidence.canonical_id]} occurrence ({workflow_index},{node_index})",
                f"  Operation: {node.operation}",
                "  Applicability: " + "; ".join(node.applicability),
                "  Inputs: " + "; ".join(
                    f"{row.type}: {row.description}" for row in node.inputs
                ),
                "  Outputs: " + "; ".join(
                    f"{row.type}: {row.description}" for row in node.outputs
                ),
            ]
        )
    for witness in candidate.dependency_witnesses:
        lines.append(f"NeedNode {witness.source_need_index} -> NeedNode {witness.target_need_index}: "
                     + " -> ".join(labels[key] for key in witness.canonical_path))
        lines.append("Evidence: " + ("continuous source occurrence path" if witness.trace_realizable
                                     else "composed from real edges; not a demonstrated continuous source execution"))
        if witness.source_occurrence_path:
            lines.append("Source occurrences: " + " -> ".join(f"({w},{n})" for w, n in witness.source_occurrence_path))
    if candidate.unsatisfied_need_edges:
        lines.append("SOFT STRUCTURAL MISMATCHES (do not discard otherwise useful operations):")
        lines.extend(
            f"- NeedNode {left} -> NeedNode {right} has no directed witness in the current graph"
            for left, right in candidate.unsatisfied_need_edges
        )
    return "\n".join(lines)


def finalize_subgraph_selection(
    selection: SubgraphSelection,
    *,
    query_text: str,
    need_graph: NeedGraph,
    workflow_recalls: Sequence[WorkflowRecall],
    search_result: Any,
    graph: ExperienceGraph,
    source: SectionGraphSource,
) -> NeedGuidedRetrievalResult:
    candidate_by_id = {row.candidate_id: row for row in search_result.candidates}
    candidate = candidate_by_id.get(selection.selected_candidate_id)
    if candidate is None:
        raise ValueError("selector selected an unverified candidate")
    status = "OK"
    rendered = _render_subgraph_candidate(
        candidate, need_graph=need_graph, graph=graph, source=source
    )
    audit = {
        "query_sha256": hashlib.sha256(query_text.encode()).hexdigest(),
        "need_graph": need_graph.to_dict(),
        "discarded_need_edge_reasons": list(need_graph.discarded_edge_reasons),
        "workflow_recalls": [row.to_dict() for row in workflow_recalls],
        "need_candidates": {
            str(index): [row.to_dict() for row in rows]
            for index, rows in sorted(search_result.need_candidates.items())
        },
        "candidate_region_node_ids": list(search_result.region_node_ids),
        "subgraph_candidates": [row.to_dict() for row in search_result.candidates],
        "search_metrics": dict(search_result.metrics),
        "selected_candidate_id": selection.selected_candidate_id,
        "rendered_experience_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
        "status": status,
    }
    return NeedGuidedRetrievalResult(status, rendered, audit)


def render_workflow_fallback(
    *,
    query_text: str,
    selection: SubgraphSelection,
    candidates: Sequence[WorkflowFallbackCandidate],
    source: SectionGraphSource,
    graph: ExperienceGraph,
    status: str = "NEED_GRAPH_FALLBACK_WORKFLOW",
    reason: str = "NeedGraph generation was unavailable",
) -> NeedGuidedRetrievalResult:
    """Inject one selected, source-backed workflow after an item-local retrieval failure."""
    candidate_by_id = {row.candidate_id: row for row in candidates}
    candidate = candidate_by_id.get(selection.selected_candidate_id)
    if candidate is None:
        raise ValueError("selector selected an unverified workflow fallback")
    workflow = source.workflow_by_index[candidate.train_index]
    lines = [
        "WORKFLOW-RECALL EXPERIENCE FALLBACK",
        f"{reason}; use only operations applicable to the current task.",
    ]
    if candidate.canonical_node_ids:
        lines.append("CANONICAL PROJECTION:")
        lines.extend(
            f"- {graph.node_by_id[canonical_id].experience.operation}"
            for canonical_id in candidate.canonical_node_ids
        )
    for node_index, node in enumerate(workflow.experience_nodes):
        lines.extend(
            [
                f"OPERATION {node_index}: {node.operation}",
                "Applicability: " + "; ".join(node.applicability),
                "Inputs: " + "; ".join(
                    f"{row.type}: {row.description}" for row in node.inputs
                ),
                "Outputs: " + "; ".join(
                    f"{row.type}: {row.description}" for row in node.outputs
                ),
            ]
        )
    if workflow.edges:
        lines.append("CAUSAL ORDER EVIDENCE:")
        lines.extend(f"- {edge.source} -> {edge.target}" for edge in workflow.edges)
    rendered = "\n".join(lines)
    audit = {
        "query_sha256": hashlib.sha256(query_text.encode()).hexdigest(),
        "workflow_fallback_candidates": [row.to_dict() for row in candidates],
        "selected_candidate_id": selection.selected_candidate_id,
        "fallback_train_index": candidate.train_index,
        "rendered_experience_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
        "fallback_reason": reason,
        "status": status,
    }
    return NeedGuidedRetrievalResult(status, rendered, audit)


__all__ = [
    "NeedEdge", "NeedGraph", "NeedGuidedRetrievalResult", "NeedNode", "SELECTOR_KIND",
    "SELECTOR_PROMPT_SHA256", "SELECTOR_SYSTEM_PROMPT", "WORKFLOW_RECALL_K", "WorkflowRecall",
    "RETRIEVAL_METHOD_ID", "SubgraphSelection", "finalize_subgraph_selection",
    "need_graph_response_schema", "openai_need_graph_llm", "openai_selector_llm",
    "parse_need_graph", "parse_selector_response", "recall_source_workflows",
    "WorkflowFallbackCandidate", "render_workflow_fallback", "selector_payload",
    "selector_response_schema", "workflow_fallback_candidates",
    "workflow_fallback_selector_payload",
]
