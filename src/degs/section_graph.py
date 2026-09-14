from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .core import canonical_json_bytes


SECTION_GRAPH_FORMAT = "degs_experience_workflows_v4"
CANONICAL_PARTITION_FORMAT = "degs_canonical_experience_partition_v4"
EXPERIENCE_GRAPH_FORMAT = "degs_experience_graph_v2"
SOURCE_SPLIT = "train[0,200)"
_SAFE_TASK_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_CONTRACT_FIELDS = {"type", "description"}
_NODE_FIELDS = {"operation", "applicability", "inputs", "outputs"}
_EDGE_FIELDS = {"source", "target"}
_CANONICAL_EXPERIENCE_FIELDS = _NODE_FIELDS
_WORKFLOW_FIELDS = {
    "train_index",
    "task_id",
    "query_text",
    "experience_nodes",
    "edges",
}
_SOURCE_FIELDS = {"format", "source_split", "workflows"}
_PARTITION_FIELDS = {"format", "section_graphs_sha256", "groups"}
_GROUP_FIELDS = {"members", "canonical_experience"}
_HOLLOW_APPLICABILITY = {
    "for this task",
    "when appropriate",
    "when needed",
    "when processing data",
    "when using a spreadsheet",
}


@dataclass(frozen=True)
class IOContract:
    type: str
    description: str

    def to_dict(self) -> dict[str, str]:
        return {"type": self.type, "description": self.description}


@dataclass(frozen=True)
class ExperienceNode:
    operation: str
    applicability: tuple[str, ...]
    inputs: tuple[IOContract, ...]
    outputs: tuple[IOContract, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "applicability": list(self.applicability),
            "inputs": [row.to_dict() for row in self.inputs],
            "outputs": [row.to_dict() for row in self.outputs],
        }


@dataclass(frozen=True)
class ExperienceEdge:
    source: int
    target: int

    def to_dict(self) -> dict[str, int]:
        return {"source": self.source, "target": self.target}


@dataclass(frozen=True)
class CanonicalExperience:
    operation: str
    applicability: tuple[str, ...]
    inputs: tuple[IOContract, ...]
    outputs: tuple[IOContract, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "applicability": list(self.applicability),
            "inputs": [row.to_dict() for row in self.inputs],
            "outputs": [row.to_dict() for row in self.outputs],
        }


@dataclass(frozen=True)
class WorkflowGraph:
    train_index: int
    task_id: str
    query_text: str
    experience_nodes: tuple[ExperienceNode, ...]
    edges: tuple[ExperienceEdge, ...]


@dataclass(frozen=True)
class SectionGraphSource:
    workflows: tuple[WorkflowGraph, ...]
    sha256: str

    @property
    def workflow_by_index(self) -> Mapping[int, WorkflowGraph]:
        return MappingProxyType({row.train_index: row for row in self.workflows})


@dataclass(frozen=True)
class CanonicalGroup:
    members: tuple[tuple[int, int], ...]
    canonical_experience: CanonicalExperience


@dataclass(frozen=True)
class CanonicalPartition:
    groups: tuple[CanonicalGroup, ...]
    section_graphs_sha256: str
    sha256: str


@dataclass(frozen=True)
class CanonicalNode:
    canonical_id: str
    experience: CanonicalExperience
    document: str
    document_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_id": self.canonical_id,
            "canonical_experience": self.experience.to_dict(),
            "document": self.document,
            "document_sha256": self.document_sha256,
        }


@dataclass(frozen=True)
class ProjectedEdge:
    source: str
    target: str
    supporting_workflow_ids: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "supporting_workflow_ids": list(self.supporting_workflow_ids),
        }


@dataclass(frozen=True)
class ExperienceGraph:
    nodes: tuple[CanonicalNode, ...]
    edges: tuple[ProjectedEdge, ...]
    section_graphs_sha256: str
    canonical_partition_sha256: str
    experience_graph_sha256: str

    @property
    def node_by_id(self) -> Mapping[str, CanonicalNode]:
        return MappingProxyType({row.canonical_id: row for row in self.nodes})

    def _body(self) -> dict[str, Any]:
        return {
            "format": EXPERIENCE_GRAPH_FORMAT,
            "section_graphs_sha256": self.section_graphs_sha256,
            "canonical_partition_sha256": self.canonical_partition_sha256,
            "nodes": [row.to_dict() for row in self.nodes],
            "edges": [row.to_dict() for row in self.edges],
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "experience_graph_sha256": self.experience_graph_sha256}

    def identity(self) -> dict[str, Any]:
        return {
            "format": EXPERIENCE_GRAPH_FORMAT,
            "section_graphs_sha256": self.section_graphs_sha256,
            "canonical_partition_sha256": self.canonical_partition_sha256,
            "experience_graph_sha256": self.experience_graph_sha256,
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
        }


def _without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _read_canonical_json(path: Path, *, label: str) -> tuple[Any, str]:
    absolute = path.expanduser().absolute()
    if not absolute.is_file():
        raise ValueError(f"{label} must be a readable file")
    try:
        payload = absolute.read_bytes()
    except OSError as exc:
        raise ValueError(f"{label} must be a readable file") from exc
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_without_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} is not strict JSON") from exc
    if payload != canonical_json_bytes(value):
        raise ValueError(f"{label} must use canonical JSON encoding")
    return value, hashlib.sha256(payload).hexdigest()


def _nonempty_string(value: Any, *, label: str) -> str:
    if type(value) is not str or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a stripped non-empty string")
    return value


def _contracts(value: Any, *, label: str) -> tuple[IOContract, ...]:
    if type(value) is not list or not value:
        raise ValueError(f"{label} must be a non-empty array")
    rows: list[IOContract] = []
    for raw in value:
        if type(raw) is not dict or set(raw) != _CONTRACT_FIELDS:
            raise ValueError(f"{label} contract fields differ")
        rows.append(
            IOContract(
                _nonempty_string(raw["type"], label=f"{label} type"),
                _nonempty_string(
                    raw["description"], label=f"{label} description"
                ),
            )
        )
    return tuple(rows)


def _applicability(value: Any, *, label: str) -> tuple[str, ...]:
    if type(value) is not list or not value:
        raise ValueError(f"{label} must be a non-empty array")
    rows = tuple(
        _nonempty_string(row, label=f"{label} item") for row in value
    )
    normalized = [row.casefold().rstrip(" .") for row in rows]
    if any(row in _HOLLOW_APPLICABILITY for row in normalized):
        raise ValueError(f"{label} contains an uninformative statement")
    if len(set(rows)) != len(rows):
        raise ValueError(f"{label} must not contain duplicates")
    return rows


def _experience_node(value: Any) -> ExperienceNode:
    if type(value) is not dict or set(value) != _NODE_FIELDS:
        raise ValueError("ExperienceNode fields differ")
    return ExperienceNode(
        _nonempty_string(value["operation"], label="ExperienceNode operation"),
        _applicability(value["applicability"], label="ExperienceNode applicability"),
        _contracts(value["inputs"], label="ExperienceNode inputs"),
        _contracts(value["outputs"], label="ExperienceNode outputs"),
    )


def _experience_edge(value: Any, *, node_count: int) -> ExperienceEdge:
    if type(value) is not dict or set(value) != _EDGE_FIELDS:
        raise ValueError("experience edge fields differ")
    source = value["source"]
    target = value["target"]
    if (
        type(source) is not int
        or type(target) is not int
        or not 0 <= source < target < node_count
    ):
        raise ValueError(
            "experience edge must reference a forward pair of distinct nodes"
        )
    return ExperienceEdge(source, target)


def _experience_edges(value: Any, *, node_count: int) -> tuple[ExperienceEdge, ...]:
    if type(value) is not list:
        raise ValueError("experience edges must be an array")
    edges = [_experience_edge(raw, node_count=node_count) for raw in value]
    pairs = [(edge.source, edge.target) for edge in edges]
    if len(pairs) != len(set(pairs)):
        raise ValueError("duplicate experience edge")
    return tuple(sorted(edges, key=lambda edge: (edge.source, edge.target)))


def _canonical_experience(value: Any) -> CanonicalExperience:
    if type(value) is not dict or set(value) != _CANONICAL_EXPERIENCE_FIELDS:
        raise ValueError("canonical experience fields differ")
    return CanonicalExperience(
        _nonempty_string(value["operation"], label="canonical experience operation"),
        _applicability(
            value["applicability"], label="canonical experience applicability"
        ),
        _contracts(value["inputs"], label="canonical experience inputs"),
        _contracts(value["outputs"], label="canonical experience outputs"),
    )


def load_section_graphs(
    path: Path | str, *, allow_empty: bool = False
) -> SectionGraphSource:
    value, source_sha256 = _read_canonical_json(Path(path), label="section graph source")
    if type(value) is not dict or set(value) != _SOURCE_FIELDS:
        raise ValueError("section graph source fields differ")
    workflows = value["workflows"]
    if (
        value["format"] != SECTION_GRAPH_FORMAT
        or value["source_split"] != SOURCE_SPLIT
        or type(workflows) is not list
        or not (0 if allow_empty else 1) <= len(workflows) <= 200
    ):
        raise ValueError("section graph source identity differs")
    rows: list[WorkflowGraph] = []
    seen_task_ids: set[str] = set()
    seen_indices: set[int] = set()
    for raw in workflows:
        if type(raw) is not dict or set(raw) != _WORKFLOW_FIELDS:
            raise ValueError("workflow fields differ")
        train_index = raw["train_index"]
        task_id = raw["task_id"]
        query_text = raw["query_text"]
        experience_nodes = raw["experience_nodes"]
        edges = raw["edges"]
        if (
            type(train_index) is not int
            or not 0 <= train_index < 200
            or train_index in seen_indices
            or type(task_id) is not str
            or _SAFE_TASK_ID.fullmatch(task_id) is None
            or task_id in seen_task_ids
            or type(query_text) is not str
            or not query_text.strip()
            or query_text != query_text.strip()
            or type(experience_nodes) is not list
            or not experience_nodes
        ):
            raise ValueError("workflow identity differs")
        rows.append(
            WorkflowGraph(
                train_index,
                task_id,
                query_text,
                tuple(_experience_node(node) for node in experience_nodes),
                _experience_edges(edges, node_count=len(experience_nodes)),
            )
        )
        seen_indices.add(train_index)
        seen_task_ids.add(task_id)
    if [row.train_index for row in rows] != sorted(seen_indices):
        raise ValueError("workflows must be ordered by train_index")
    return SectionGraphSource(tuple(rows), source_sha256)


def load_canonical_partition(
    path: Path | str,
    *,
    source: SectionGraphSource,
    allow_empty: bool = False,
) -> CanonicalPartition:
    value, partition_sha256 = _read_canonical_json(
        Path(path), label="canonical partition"
    )
    if type(value) is not dict or set(value) != _PARTITION_FIELDS:
        raise ValueError("canonical partition fields differ")
    groups = value["groups"]
    if (
        value["format"] != CANONICAL_PARTITION_FORMAT
        or value["section_graphs_sha256"] != source.sha256
        or type(groups) is not list
        or (not groups and not allow_empty)
    ):
        raise ValueError("canonical partition identity differs")
    known = {
        (workflow.train_index, node_index)
        for workflow in source.workflows
        for node_index, _node in enumerate(workflow.experience_nodes)
    }
    parsed: list[CanonicalGroup] = []
    covered: list[tuple[int, int]] = []
    for raw in groups:
        if type(raw) is not dict or set(raw) != _GROUP_FIELDS:
            raise ValueError("canonical group fields differ")
        members = raw["members"]
        if type(members) is not list or not members:
            raise ValueError("canonical group members differ")
        group: list[tuple[int, int]] = []
        for member in members:
            if (
                type(member) is not list
                or len(member) != 2
                or any(type(index) is not int for index in member)
            ):
                raise ValueError("canonical member coordinate differs")
            coordinate = (member[0], member[1])
            if coordinate not in known:
                raise ValueError("canonical member is absent from the experience source")
            group.append(coordinate)
        if group != sorted(set(group)):
            raise ValueError("canonical group members must be sorted and unique")
        canonical_experience = _canonical_experience(raw["canonical_experience"])
        parsed.append(CanonicalGroup(tuple(group), canonical_experience))
        covered.extend(group)
    if parsed != sorted(parsed, key=lambda group: group.members[0]):
        raise ValueError("canonical groups must be ordered by their first member")
    if len(covered) != len(set(covered)) or set(covered) != known:
        raise ValueError("canonical partition must cover every ExperienceNode exactly once")
    return CanonicalPartition(
        tuple(parsed), source.sha256, partition_sha256
    )


def _canonical_id(members: Sequence[tuple[int, int]]) -> str:
    digest = hashlib.sha256(
        canonical_json_bytes([list(member) for member in members])
    ).hexdigest()
    return f"canonical_{digest[:24]}"


def _canonical_document(canonical: CanonicalExperience) -> str:
    applicability = "\n".join(f"- {row}" for row in canonical.applicability)
    inputs = "\n".join(
        f"- {row.type}: {row.description}" for row in canonical.inputs
    )
    outputs = "\n".join(
        f"- {row.type}: {row.description}" for row in canonical.outputs
    )
    return (
        f"OPERATION\n{canonical.operation}\n"
        f"APPLICABILITY\n{applicability}\n"
        f"INPUTS\n{inputs}\n"
        f"OUTPUTS\n{outputs}"
    )


def experience_leaf_id(
    train_index: int,
    node_index: int,
    node: ExperienceNode,
) -> str:
    digest = hashlib.sha256(
        canonical_json_bytes(
            {
                "train_index": train_index,
                "node_index": node_index,
                "experience_node": node.to_dict(),
            }
        )
    ).hexdigest()
    return f"experience_{digest[:24]}"


def experience_document(node: ExperienceNode) -> str:
    return _canonical_document(
        CanonicalExperience(
            node.operation, node.applicability, node.inputs, node.outputs
        )
    )


def compile_experience_graph(
    source: SectionGraphSource,
    partition: CanonicalPartition,
) -> ExperienceGraph:
    if type(source) is not SectionGraphSource or type(partition) is not CanonicalPartition:
        raise TypeError("experience graph compilation requires validated source capabilities")
    if partition.section_graphs_sha256 != source.sha256:
        raise ValueError("canonical partition is not bound to the experience source")
    workflow_by_index = source.workflow_by_index
    canonical_by_member: dict[tuple[int, int], str] = {}
    nodes: list[CanonicalNode] = []
    for group in partition.groups:
        members = group.members
        canonical_id = _canonical_id(members)
        canonical = group.canonical_experience
        document = _canonical_document(canonical)
        nodes.append(
            CanonicalNode(
                canonical_id,
                canonical,
                document,
                hashlib.sha256(document.encode("utf-8")).hexdigest(),
            )
        )
        for member in members:
            canonical_by_member[member] = canonical_id

    edge_workflows: dict[tuple[str, str], set[int]] = defaultdict(set)
    per_workflow_pairs: set[tuple[int, str, str]] = set()
    for workflow in source.workflows:
        for edge in workflow.edges:
            source_id = canonical_by_member[(workflow.train_index, edge.source)]
            target_id = canonical_by_member[(workflow.train_index, edge.target)]
            dependency_key = (workflow.train_index, source_id, target_id)
            if dependency_key in per_workflow_pairs:
                continue
            per_workflow_pairs.add(dependency_key)
            edge_workflows[(source_id, target_id)].add(workflow.train_index)
    edges = tuple(
        ProjectedEdge(source_id, target_id, tuple(sorted(workflow_ids)))
        for (source_id, target_id), workflow_ids in sorted(edge_workflows.items())
    )
    nodes_tuple = tuple(sorted(nodes, key=lambda row: row.canonical_id))
    body = {
        "format": EXPERIENCE_GRAPH_FORMAT,
        "section_graphs_sha256": source.sha256,
        "canonical_partition_sha256": partition.sha256,
        "nodes": [row.to_dict() for row in nodes_tuple],
        "edges": [row.to_dict() for row in edges],
    }
    return ExperienceGraph(
        nodes_tuple,
        edges,
        source.sha256,
        partition.sha256,
        hashlib.sha256(canonical_json_bytes(body)).hexdigest(),
    )


def load_experience_graph(
    section_graphs_path: Path | str,
    *,
    canonical_partition_path: Path | str,
) -> ExperienceGraph:
    source = load_section_graphs(section_graphs_path)
    partition = load_canonical_partition(canonical_partition_path, source=source)
    return compile_experience_graph(source, partition)


__all__ = [
    "CANONICAL_PARTITION_FORMAT",
    "CanonicalExperience",
    "CanonicalGroup",
    "CanonicalNode",
    "CanonicalPartition",
    "EXPERIENCE_GRAPH_FORMAT",
    "ExperienceGraph",
    "ExperienceEdge",
    "ExperienceNode",
    "IOContract",
    "ProjectedEdge",
    "SECTION_GRAPH_FORMAT",
    "SOURCE_SPLIT",
    "SectionGraphSource",
    "WorkflowGraph",
    "compile_experience_graph",
    "experience_document",
    "experience_leaf_id",
    "load_canonical_partition",
    "load_experience_graph",
    "load_section_graphs",
]
