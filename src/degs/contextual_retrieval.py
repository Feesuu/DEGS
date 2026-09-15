from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .section_graph import CanonicalNode, ExperienceGraph, ProjectedEdge


CONTEXTUAL_RETRIEVAL_METHOD = "degs_eir_top5_one_hop_v1"
CONTEXTUAL_TOP_K = 5
CONTEXT_NEIGHBORS_PER_ANCHOR = 2


def _unit_vector(values: Sequence[float], *, label: str) -> tuple[float, ...]:
    if not values or any(type(value) not in (int, float) or not math.isfinite(float(value)) for value in values):
        raise ValueError(f"{label} vector differs")
    vector = tuple(float(value) for value in values)
    norm = math.sqrt(sum(value * value for value in vector))
    if not math.isfinite(norm) or norm == 0.0:
        raise ValueError(f"{label} vector differs")
    return tuple(value / norm for value in vector)


def _similarity(left: Sequence[float], right: Sequence[float]) -> float:
    a = _unit_vector(left, label="query")
    b = _unit_vector(right, label="Canonical")
    if len(a) != len(b):
        raise ValueError("retrieval embedding dimensions differ")
    return sum(x * y for x, y in zip(a, b, strict=True))


@dataclass(frozen=True)
class ContextualNode:
    canonical_id: str
    canonical_version: int
    similarity: float
    node: CanonicalNode

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_id": self.canonical_id,
            "canonical_version": self.canonical_version,
            "similarity": self.similarity,
            "canonical_experience": self.node.experience.to_dict(),
        }


@dataclass(frozen=True)
class ContextualRetrieval:
    snapshot_id: str
    anchors: tuple[ContextualNode, ...]
    context_nodes: tuple[ContextualNode, ...]
    edges: tuple[ProjectedEdge, ...]

    @property
    def anchor_versions(self) -> Mapping[str, int]:
        return MappingProxyType(
            {row.canonical_id: row.canonical_version for row in self.anchors}
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": CONTEXTUAL_RETRIEVAL_METHOD,
            "snapshot_id": self.snapshot_id,
            "anchors": [row.to_dict() for row in self.anchors],
            "context_nodes": [row.to_dict() for row in self.context_nodes],
            "edges": [row.to_dict() for row in self.edges],
        }

    def to_binding_payload(
        self,
        *,
        query: Mapping[str, Any],
        observable_context: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        return {
            "query": dict(query),
            "observable_context": [dict(row) for row in observable_context],
            "snapshot_id": self.snapshot_id,
            "anchors": [row.to_dict() for row in self.anchors],
            "context_nodes": [row.to_dict() for row in self.context_nodes],
            "induced_edges": [row.to_dict() for row in self.edges],
        }


def retrieve_contextual_subgraph(
    *,
    graph: ExperienceGraph,
    snapshot_id: str,
    query_vector: Sequence[float],
    canonical_vectors: Mapping[str, Sequence[float]],
    canonical_versions: Mapping[str, int],
    top_k: int = CONTEXTUAL_TOP_K,
    neighbors_per_anchor: int = CONTEXT_NEIGHBORS_PER_ANCHOR,
) -> ContextualRetrieval:
    if type(graph) is not ExperienceGraph or type(snapshot_id) is not str or not snapshot_id:
        raise ValueError("contextual retrieval identity differs")
    if type(top_k) is not int or top_k <= 0 or type(neighbors_per_anchor) is not int or not 0 <= neighbors_per_anchor <= 2:
        raise ValueError("contextual retrieval bounds differ")
    if not graph.nodes:
        if canonical_vectors or canonical_versions:
            raise ValueError("empty graph retrieval state differs")
        _unit_vector(query_vector, label="query")
        return ContextualRetrieval(snapshot_id, (), (), ())
    node_by_id = graph.node_by_id
    expected_ids = set(node_by_id)
    if set(canonical_vectors) != expected_ids or set(canonical_versions) != expected_ids:
        raise ValueError("active Canonical retrieval population differs")
    if any(type(version) is not int or version <= 0 for version in canonical_versions.values()):
        raise ValueError("active Canonical version differs")
    similarities = {
        canonical_id: _similarity(query_vector, canonical_vectors[canonical_id])
        for canonical_id in sorted(expected_ids)
    }
    anchor_ids = tuple(
        canonical_id
        for canonical_id, _score in sorted(
            similarities.items(), key=lambda row: (-row[1], row[0])
        )[:top_k]
    )
    incoming: dict[str, set[str]] = {canonical_id: set() for canonical_id in expected_ids}
    outgoing: dict[str, set[str]] = {canonical_id: set() for canonical_id in expected_ids}
    for edge in graph.edges:
        if edge.source not in expected_ids or edge.target not in expected_ids:
            raise ValueError("experience graph edge endpoint differs")
        outgoing[edge.source].add(edge.target)
        incoming[edge.target].add(edge.source)

    selected_neighbors: set[str] = set()
    anchor_set = set(anchor_ids)

    def ranked(values: set[str]) -> list[str]:
        return sorted(values - anchor_set, key=lambda value: (-similarities[value], value))

    for anchor_id in anchor_ids:
        chosen: list[str] = []
        inbound = ranked(incoming[anchor_id])
        outbound = ranked(outgoing[anchor_id])
        if inbound:
            chosen.append(inbound[0])
        if outbound and outbound[0] not in chosen and len(chosen) < neighbors_per_anchor:
            chosen.append(outbound[0])
        remaining = sorted(
            (incoming[anchor_id] | outgoing[anchor_id]) - anchor_set - set(chosen),
            key=lambda value: (-similarities[value], value),
        )
        chosen.extend(remaining[: max(0, neighbors_per_anchor - len(chosen))])
        selected_neighbors.update(chosen)

    selected_ids = anchor_set | selected_neighbors

    def contextual_node(canonical_id: str) -> ContextualNode:
        return ContextualNode(
            canonical_id,
            canonical_versions[canonical_id],
            similarities[canonical_id],
            node_by_id[canonical_id],
        )

    anchors = tuple(contextual_node(canonical_id) for canonical_id in anchor_ids)
    context_nodes = tuple(
        contextual_node(canonical_id)
        for canonical_id in sorted(
            selected_neighbors, key=lambda value: (-similarities[value], value)
        )
    )
    edges = tuple(
        edge
        for edge in graph.edges
        if edge.source in selected_ids and edge.target in selected_ids
    )
    return ContextualRetrieval(snapshot_id, anchors, context_nodes, edges)


__all__ = [
    "CONTEXTUAL_RETRIEVAL_METHOD",
    "CONTEXTUAL_TOP_K",
    "CONTEXT_NEIGHBORS_PER_ANCHOR",
    "ContextualNode",
    "ContextualRetrieval",
    "retrieve_contextual_subgraph",
]
