from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
import heapq
import itertools
import math
import time
from typing import Any, Mapping, Sequence

import networkx as nx
import numpy as np

from .core import normalize_embedding_text
from .section_graph import (
    CanonicalExperience,
    CanonicalPartition,
    ExperienceGraph,
    SectionGraphSource,
    _canonical_id,
)
from .workflow_retrieval import NeedGraph, NeedNode, WorkflowRecall


NEED_CANONICAL_TOP_K = 8
SUBGRAPH_TOP_K = 8
OCCURRENCE_EVIDENCE_TOP_K = 8
PARTIAL_BEAM_K = 32
SEMANTIC_RECALL_DOCUMENT_FORMAT = "operation_with_optional_target_clarification_v1"
RECALL_SCORE_FORMAT = "operation_clarification_recall_candidate_workflow_context_mean_v1"


def canonical_retrieval_document(experience: CanonicalExperience) -> str:
    """Return the DEGS operation-only recall document."""
    return experience.operation


def need_retrieval_document(need: NeedNode, clarification: str | None = None) -> str:
    """Preserve the accepted document unless input evidence resolved an ambiguity."""
    if clarification is None:
        return need.description
    if type(clarification) is not str or not clarification.strip():
        raise ValueError("Need retrieval clarification must be non-empty")
    return f"{need.description}\nTarget-evidence clarification: {clarification.strip()}"


@dataclass(frozen=True)
class NeedCandidate:
    need_index: int
    canonical_id: str
    operation_similarity: float
    operation_distance: float
    workflow_context_similarity: float
    workflow_context_distance: float
    recall_similarity: float
    recall_distance: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "need_index": self.need_index,
            "canonical_id": self.canonical_id,
            "operation_similarity": self.operation_similarity,
            "operation_distance": self.operation_distance,
            "workflow_context_similarity": self.workflow_context_similarity,
            "workflow_context_distance": self.workflow_context_distance,
            "recall_similarity": self.recall_similarity,
            "recall_distance": self.recall_distance,
        }


@dataclass(frozen=True)
class DependencyWitness:
    source_need_index: int
    target_need_index: int
    canonical_path: tuple[str, ...]
    supporting_workflow_ids: tuple[int, ...]
    trace_realizable: bool
    workflow_switch_count: int
    source_occurrence_path: tuple[tuple[int, int], ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_need_index": self.source_need_index,
            "target_need_index": self.target_need_index,
            "canonical_path": list(self.canonical_path),
            "supporting_workflow_ids": list(self.supporting_workflow_ids),
            "trace_realizable": self.trace_realizable,
            "workflow_switch_count": self.workflow_switch_count,
            "source_occurrence_path": [list(row) for row in self.source_occurrence_path] if self.source_occurrence_path else None,
        }


@dataclass(frozen=True)
class OccurrenceEvidence:
    occurrence_id: tuple[int, int]
    canonical_id: str
    workflow_recalled: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "occurrence_id": list(self.occurrence_id),
            "canonical_id": self.canonical_id,
            "workflow_recalled": self.workflow_recalled,
        }


@dataclass(frozen=True)
class SubgraphCandidate:
    candidate_id: str
    need_mapping: tuple[tuple[int, str], ...]
    unmatched_need_indices: tuple[int, ...]
    canonical_node_ids: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    dependency_witnesses: tuple[DependencyWitness, ...]
    unsatisfied_need_edges: tuple[tuple[int, int], ...]
    occurrence_evidence: tuple[OccurrenceEvidence, ...]
    matched_need_count: int
    satisfied_need_edge_count: int
    operation_distance_sum: float
    workflow_context_distance_sum: float
    recall_distance_sum: float
    connector_count: int
    non_trace_realized_edge_count: int
    workflow_switch_count: int
    workflow_recall_rank_cost: int

    @property
    def operation_loss(self) -> float:
        return self.operation_distance_sum / max(1, self.matched_need_count)

    @property
    def workflow_context_loss(self) -> float:
        return self.workflow_context_distance_sum / max(1, self.matched_need_count)

    @property
    def recall_loss(self) -> float:
        return self.recall_distance_sum / max(1, self.matched_need_count)

    @property
    def structural_loss(self) -> float:
        edge_count = self.satisfied_need_edge_count + len(self.unsatisfied_need_edges)
        return len(self.unsatisfied_need_edges) / max(1, edge_count)

    @property
    def total_cost(self) -> float:
        return self.recall_loss + self.structural_loss

    @property
    def rank_key(self) -> tuple[Any, ...]:
        return (
            -self.matched_need_count,
            self.total_cost,
            -sum(row.trace_realizable for row in self.dependency_witnesses),
            -self.satisfied_need_edge_count,
            self.connector_count,
            self.non_trace_realized_edge_count,
            self.workflow_switch_count,
            self.recall_distance_sum,
            self.operation_distance_sum,
            self.workflow_context_distance_sum,
            self.workflow_recall_rank_cost,
            self.canonical_node_ids,
            self.edges,
            self.need_mapping,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "need_mapping": [
                {"need_index": need_index, "canonical_id": canonical_id}
                for need_index, canonical_id in self.need_mapping
            ],
            "unmatched_need_indices": list(self.unmatched_need_indices),
            "canonical_node_ids": list(self.canonical_node_ids),
            "edges": [
                {"source": source, "target": target}
                for source, target in self.edges
            ],
            "dependency_witnesses": [
                row.to_dict() for row in self.dependency_witnesses
            ],
            "unsatisfied_need_edges": [
                {"source": source, "target": target}
                for source, target in self.unsatisfied_need_edges
            ],
            "occurrence_evidence": [
                row.to_dict() for row in self.occurrence_evidence
            ],
            "rank": {
                "matched_need_count": self.matched_need_count,
                "satisfied_need_edge_count": self.satisfied_need_edge_count,
                "unsatisfied_need_edge_count": len(self.unsatisfied_need_edges),
                "operation_distance_sum": self.operation_distance_sum,
                "operation_loss": self.operation_loss,
                "workflow_context_distance_sum": self.workflow_context_distance_sum,
                "workflow_context_loss": self.workflow_context_loss,
                "recall_distance_sum": self.recall_distance_sum,
                "recall_loss": self.recall_loss,
                "structural_loss": self.structural_loss,
                "total_cost": self.total_cost,
                "connector_count": self.connector_count,
                "non_trace_realized_edge_count": self.non_trace_realized_edge_count,
                "workflow_switch_count": self.workflow_switch_count,
                "workflow_recall_rank_cost": self.workflow_recall_rank_cost,
            },
        }


@dataclass(frozen=True)
class RetrievalSearchResult:
    need_candidates: Mapping[int, tuple[NeedCandidate, ...]]
    region_node_ids: tuple[str, ...]
    candidates: tuple[SubgraphCandidate, ...]
    metrics: Mapping[str, int | float | str]


@dataclass(frozen=True)
class RetrievalIndex:
    graph: ExperienceGraph
    canonical_ids: tuple[str, ...]
    occurrences_by_canonical: Mapping[str, tuple[tuple[int, int], ...]]
    canonical_ids_by_workflow: Mapping[int, tuple[str, ...]]
    adjacency: Mapping[str, tuple[str, ...]]
    edge_support: Mapping[tuple[str, str], tuple[int, ...]]
    operation_matrix: np.ndarray
    occurrence_adjacency: Mapping[tuple[int, int], tuple[tuple[int, int], ...]]


def build_retrieval_index(
    graph: ExperienceGraph,
    partition: CanonicalPartition,
    source: SectionGraphSource,
    *,
    vectors: Mapping[str, Sequence[float]],
) -> RetrievalIndex:
    graph_ids = {row.canonical_id for row in graph.nodes}
    occurrences_by_canonical: dict[str, tuple[tuple[int, int], ...]] = {}
    workflow_nodes: dict[int, set[str]] = defaultdict(set)
    seen_occurrences: set[tuple[int, int]] = set()
    for group in partition.groups:
        canonical_id = _canonical_id(group.members)
        if canonical_id not in graph_ids or canonical_id in occurrences_by_canonical:
            raise ValueError("Canonical partition and ExperienceGraph differ")
        occurrences_by_canonical[canonical_id] = tuple(group.members)
        for member in group.members:
            if member in seen_occurrences:
                raise ValueError("Canonical partition repeats an occurrence")
            seen_occurrences.add(member)
            workflow_nodes[member[0]].add(canonical_id)
    source_occurrences = {
        (workflow.train_index, node_index)
        for workflow in source.workflows
        for node_index, _node in enumerate(workflow.experience_nodes)
    }
    if (
        set(occurrences_by_canonical) != graph_ids
        or seen_occurrences != source_occurrences
    ):
        raise ValueError("Canonical partition does not exactly cover the source graph")
    edge_support: dict[tuple[str, str], tuple[int, ...]] = {}
    directed = nx.DiGraph()
    directed.add_nodes_from(graph_ids)
    for edge in graph.edges:
        if edge.source not in graph_ids or edge.target not in graph_ids:
            raise ValueError("ExperienceGraph edge endpoint is absent")
        support = tuple(edge.supporting_workflow_ids)
        if not support or support != tuple(sorted(set(support))):
            raise ValueError("ExperienceGraph edge support differs")
        directed.add_edge(edge.source, edge.target)
        edge_support[(edge.source, edge.target)] = support
    occurrence_adjacency: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    canonical_by_occurrence = {m: key for key, members in occurrences_by_canonical.items() for m in members}
    expected_support: dict[tuple[str, str], set[int]] = defaultdict(set)
    for workflow in source.workflows:
        for edge in workflow.edges:
            left, right = (workflow.train_index, edge.source), (workflow.train_index, edge.target)
            occurrence_adjacency[left].append(right)
            expected_support[(canonical_by_occurrence[left], canonical_by_occurrence[right])].add(workflow.train_index)
    if edge_support != {pair: tuple(sorted(ids)) for pair, ids in expected_support.items()}:
        raise ValueError("ExperienceGraph projection differs from real source edges")
    canonical_ids = tuple(sorted(graph_ids))
    canonical_nodes = [graph.node_by_id[row] for row in canonical_ids]
    return RetrievalIndex(
        graph,
        canonical_ids,
        occurrences_by_canonical,
        {index: tuple(sorted(rows)) for index, rows in workflow_nodes.items()},
        {node: tuple(sorted(directed.successors(node))) for node in sorted(graph_ids)},
        edge_support,
        _unit_rows(
            [
                _vector(canonical_retrieval_document(row.experience), vectors)
                for row in canonical_nodes
            ]
        ),
        {key: tuple(sorted(value)) for key, value in occurrence_adjacency.items()},
    )


def _vector(text: str, vectors: Mapping[str, Sequence[float]]) -> Sequence[float]:
    try:
        return vectors[normalize_embedding_text(text)]
    except KeyError as exc:
        raise ValueError("retrieval embedding evidence is incomplete") from exc


def _unit_rows(rows: Sequence[Sequence[float]]) -> np.ndarray:
    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.ndim != 2 or not len(matrix):
        raise ValueError("retrieval embedding matrix differs")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0.0) or not np.isfinite(matrix).all():
        raise ValueError("retrieval embedding vector differs")
    return matrix / norms


def _recall_need_candidates(
    need_graph: NeedGraph,
    index: RetrievalIndex,
    vectors: Mapping[str, Sequence[float]],
    workflow_context_similarities: Mapping[int, float],
    need_documents: Mapping[int, str] | None = None,
    *,
    top_k: int,
) -> dict[int, tuple[NeedCandidate, ...]]:
    if top_k <= 0:
        raise ValueError("Need candidate policy differs")
    result: dict[int, tuple[NeedCandidate, ...]] = {}
    occurrence_workflows = {
        canonical_id: {workflow_index for workflow_index, _ in occurrences}
        for canonical_id, occurrences in index.occurrences_by_canonical.items()
    }
    required_workflows = set().union(*occurrence_workflows.values())
    if not required_workflows.issubset(workflow_context_similarities):
        raise ValueError("workflow context does not cover Canonical occurrences")
    if any(
        not math.isfinite(float(workflow_context_similarities[index]))
        for index in required_workflows
    ):
        raise ValueError("workflow context similarity is not finite")
    documents = need_documents or {}
    if any(
        type(index) is not int
        or not 0 <= index < len(need_graph.nodes)
        or type(document) is not str
        or not document.strip()
        for index, document in documents.items()
    ):
        raise ValueError("Need retrieval documents differ")
    for need_index, need in enumerate(need_graph.nodes):
        need_document = _unit_rows(
            [_vector(documents.get(need_index, need_retrieval_document(need)), vectors)]
        )[0]
        operation_scores = index.operation_matrix @ need_document
        rows: list[NeedCandidate] = []
        for position, canonical_id in enumerate(index.canonical_ids):
            operation_similarity = float(operation_scores[position])
            context_similarity = max(
                float(workflow_context_similarities[workflow_index])
                for workflow_index in occurrence_workflows[canonical_id]
            )
            recall_similarity = (operation_similarity + context_similarity) / 2.0
            rows.append(
                NeedCandidate(
                    need_index,
                    canonical_id,
                    operation_similarity,
                    1.0 - operation_similarity,
                    context_similarity,
                    1.0 - context_similarity,
                    recall_similarity,
                    1.0 - recall_similarity,
                )
            )
        rows.sort(key=lambda row: (row.operation_distance, row.canonical_id))
        result[need_index] = tuple(rows[:top_k])
    return result


def _candidate_region(
    need_candidates: Mapping[int, Sequence[NeedCandidate]],
    workflow_recalls: Sequence[WorkflowRecall],
    index: RetrievalIndex,
) -> tuple[str, ...]:
    # Workflow recall is retained as provenance/ranking evidence, not as a hard
    # candidate-universe expansion.  The induced anchor set is exactly the
    # operation-recall top-k for each NeedNode.
    del workflow_recalls, index
    nodes = {
        row.canonical_id
        for rows in need_candidates.values()
        for row in rows
    }
    return tuple(sorted(nodes))


def _region_graph(region: Sequence[str], index: RetrievalIndex) -> nx.DiGraph:
    allowed = set(region)
    graph = nx.DiGraph()
    graph.add_nodes_from(region)
    for source in region:
        graph.add_edges_from(
            (source, target)
            for target in index.adjacency.get(source, ())
            if target in allowed
        )
    return graph


def _descendant_bits(
    graph: nx.DiGraph,
) -> tuple[dict[str, int], dict[str, int]]:
    # Reachability on SCC condensation; self reachability requires a positive cycle.
    ordered = sorted(graph.nodes)
    positions = {node: i for i, node in enumerate(ordered)}
    components = sorted((tuple(sorted(c)) for c in nx.strongly_connected_components(graph)))
    condensed = nx.condensation(graph, components)
    component_bits = {i: sum(1 << positions[n] for n in members) for i, members in enumerate(components)}
    downstream: dict[int, int] = {}
    for component in reversed(list(nx.topological_sort(condensed))):
        downstream[component] = 0
        for successor in condensed.successors(component):
            downstream[component] |= component_bits[successor] | downstream[successor]
    descendants = {}
    for component, members in enumerate(components):
        for node in members:
            positive_internal = len(members) > 1 or graph.has_edge(node, node)
            descendants[node] = downstream[component] | (component_bits[component] if positive_internal else 0)
    return positions, descendants


def _reachable(
    source: str,
    target: str,
    *,
    positions: Mapping[str, int],
    descendants: Mapping[str, int],
) -> bool:
    return bool(descendants[source] & (1 << positions[target]))


def _need_search_order(
    need_graph: NeedGraph, candidate_counts: Mapping[int, int]
) -> tuple[int, ...]:
    undirected = nx.Graph()
    undirected.add_nodes_from(range(len(need_graph.nodes)))
    undirected.add_edges_from((edge.source, edge.target) for edge in need_graph.edges)
    key = lambda node: (candidate_counts[node], -undirected.degree[node], node)
    ordered: list[int] = []
    remaining = set(undirected.nodes)
    while remaining:
        root = min(remaining, key=key)
        stack = [root]
        while stack:
            node = stack.pop()
            if node not in remaining:
                continue
            remaining.remove(node)
            ordered.append(node)
            stack.extend(
                sorted(
                    (
                        neighbor
                        for neighbor in undirected.neighbors(node)
                        if neighbor in remaining
                    ),
                    key=key,
                    reverse=True,
                )
            )
    return tuple(ordered)


def _best_witness_path(
    source: str,
    target: str,
    *,
    graph: nx.DiGraph,
    index: RetrievalIndex,
    recalled_workflows: frozenset[int],
    forbidden_internal: frozenset[str],
) -> tuple[tuple[str, ...], tuple[int, ...], int] | None:
    if graph.has_edge(source, target):
        return (source, target), index.edge_support[(source, target)], 0
    sequence = itertools.count()
    initial_cost = (0, 0, 0, (source,))
    queue: list[tuple[tuple[Any, ...], int, str, int | None]] = [
        (initial_cost, next(sequence), source, None)
    ]
    best: dict[tuple[str, int | None], tuple[Any, ...]] = {
        (source, None): initial_cost
    }
    while queue:
        cost, _sequence, node, last_workflow = heapq.heappop(queue)
        if best.get((node, last_workflow)) != cost:
            continue
        hops, non_recalled, switches, path = cost
        if node == target and hops > 0:
            supports = [
                set(index.edge_support[(left, right)])
                for left, right in zip(path, path[1:])
            ]
            common = set.intersection(*supports) if supports else set()
            return path, tuple(sorted(common)), switches
        for successor in sorted(graph.successors(node)):
            if successor != target and successor in forbidden_internal:
                continue
            for workflow in index.edge_support[(node, successor)]:
                next_cost = (
                    hops + 1,
                    non_recalled + int(workflow not in recalled_workflows),
                    switches
                    + int(last_workflow is not None and workflow != last_workflow),
                    (*path, successor),
                )
                state = (successor, workflow)
                if state not in best or next_cost < best[state]:
                    best[state] = next_cost
                    heapq.heappush(
                        queue,
                        (next_cost, next(sequence), successor, workflow),
                    )
    return None


def _recover_occurrence_path(
    canonical_path: Sequence[str], index: RetrievalIndex,
) -> tuple[tuple[int, int], ...] | None:
    """Dynamic programming on real source edges, not workflow-set intersection."""
    if len(canonical_path) < 2:
        return None
    paths = {m: (m,) for m in index.occurrences_by_canonical[canonical_path[0]]}
    for canonical_id in canonical_path[1:]:
        allowed = set(index.occurrences_by_canonical[canonical_id])
        next_paths: dict[tuple[int, int], tuple[tuple[int, int], ...]] = {}
        for last, path in sorted(paths.items()):
            for successor in index.occurrence_adjacency.get(last, ()):
                if successor in allowed:
                    candidate = (*path, successor)
                    if successor not in next_paths or candidate < next_paths[successor]:
                        next_paths[successor] = candidate
        paths = next_paths
        if not paths:
            return None
    return min(paths.values(), default=None)


def _occurrence_evidence(
    canonical_ids: Sequence[str],
    *,
    index: RetrievalIndex,
    workflow_recalls: Sequence[WorkflowRecall],
    witnesses: Sequence[DependencyWitness] = (),
) -> tuple[OccurrenceEvidence, ...]:
    recall_rank = {row.train_index: row.rank for row in workflow_recalls}
    witness_occurrences = {m for witness in witnesses for m in (witness.source_occurrence_path or ())}
    rows: list[OccurrenceEvidence] = []
    for canonical_id in canonical_ids:
        occurrences = sorted(
            index.occurrences_by_canonical.get(canonical_id, ()),
            key=lambda row: (
                row not in witness_occurrences,
                row[0] not in recall_rank,
                recall_rank.get(row[0], len(recall_rank) + 1),
                row,
            ),
        )
        if occurrences:
            occurrence = occurrences[0]
            rows.append(
                OccurrenceEvidence(
                    occurrence,
                    canonical_id,
                    occurrence[0] in recall_rank,
                )
            )
    rows.sort(
        key=lambda row: (
            row.occurrence_id not in witness_occurrences,
            not row.workflow_recalled,
            recall_rank.get(row.occurrence_id[0], len(recall_rank) + 1),
            row.canonical_id,
            row.occurrence_id,
        )
    )
    return tuple(rows[:OCCURRENCE_EVIDENCE_TOP_K])


def _candidate_from_mapping(
    mapping: Mapping[int, str],
    *,
    need_graph: NeedGraph,
    need_candidates: Mapping[int, Sequence[NeedCandidate]],
    region_graph: nx.DiGraph,
    index: RetrievalIndex,
    workflow_recalls: Sequence[WorkflowRecall],
) -> SubgraphCandidate | None:
    anchors = set(mapping.values())
    if not anchors:
        return None
    recalled = frozenset(row.train_index for row in workflow_recalls)
    witnesses: list[DependencyWitness] = []
    unsatisfied: list[tuple[int, int]] = []
    selected_edges: set[tuple[str, str]] = set()
    selected_nodes = set(anchors)
    switches = 0
    for edge in need_graph.edges:
        source = mapping[edge.source]
        target = mapping[edge.target]
        witness = _best_witness_path(
            source,
            target,
            graph=region_graph,
            index=index,
            recalled_workflows=recalled,
            forbidden_internal=frozenset(anchors - {source, target}),
        )
        if witness is None:
            unsatisfied.append((edge.source, edge.target))
            continue
        path, supporting_workflows, edge_switches = witness
        occurrence_path = _recover_occurrence_path(path, index)
        selected_nodes.update(path)
        selected_edges.update(zip(path, path[1:]))
        switches += edge_switches
        witnesses.append(
            DependencyWitness(
                edge.source,
                edge.target,
                path,
                supporting_workflows,
                occurrence_path is not None,
                edge_switches,
                occurrence_path,
            )
        )
    mapped = tuple(sorted(mapping.items()))
    candidate_by_pair = {
        (row.need_index, row.canonical_id): row
        for rows in need_candidates.values()
        for row in rows
    }
    operation_distance = float(
        math.fsum(candidate_by_pair[pair].operation_distance for pair in mapped)
    )
    workflow_context_distance = float(
        math.fsum(
            candidate_by_pair[pair].workflow_context_distance for pair in mapped
        )
    )
    recall_distance = float(
        math.fsum(candidate_by_pair[pair].recall_distance for pair in mapped)
    )
    recall_rank = {row.train_index: row.rank for row in workflow_recalls}
    recall_cost = 0
    for _need_index, canonical_id in mapped:
        ranks = [
            recall_rank[occurrence[0]]
            for occurrence in index.occurrences_by_canonical.get(canonical_id, ())
            if occurrence[0] in recall_rank
        ]
        recall_cost += min(ranks) if ranks else len(recall_rank) + 1
    canonical_node_ids = tuple(sorted(selected_nodes))
    return SubgraphCandidate(
        "",
        mapped,
        (),
        canonical_node_ids,
        tuple(sorted(selected_edges)),
        tuple(witnesses),
        tuple(sorted(unsatisfied)),
        _occurrence_evidence(
            canonical_node_ids,
            index=index,
            workflow_recalls=workflow_recalls,
            witnesses=witnesses,
        ),
        len(mapped),
        len(witnesses),
        operation_distance,
        workflow_context_distance,
        recall_distance,
        len(selected_nodes - anchors),
        sum(not row.trace_realizable for row in witnesses),
        switches,
        recall_cost,
    )


def retrieve_experience_subgraphs(
    need_graph: NeedGraph,
    workflow_recalls: Sequence[WorkflowRecall],
    *,
    index: RetrievalIndex,
    vectors: Mapping[str, Sequence[float]],
    workflow_context_similarities: Mapping[int, float],
    need_documents: Mapping[int, str] | None = None,
    need_top_k: int = NEED_CANONICAL_TOP_K,
    result_top_k: int = SUBGRAPH_TOP_K,
    partial_beam_k: int = PARTIAL_BEAM_K,
) -> RetrievalSearchResult:
    started = time.monotonic()
    if result_top_k <= 0 or partial_beam_k <= 0:
        raise ValueError("subgraph search policy differs")
    need_candidates = _recall_need_candidates(
        need_graph,
        index,
        vectors,
        workflow_context_similarities,
        need_documents,
        top_k=need_top_k,
    )
    if any(not rows for rows in need_candidates.values()):
        raise ValueError("ExperienceGraph cannot satisfy operation top-k recall")
    region = _candidate_region(need_candidates, workflow_recalls, index)
    anchor_graph = _region_graph(region, index)
    search_graph = _region_graph(index.canonical_ids, index)
    positions, descendants = _descendant_bits(search_graph)
    order = _need_search_order(
        need_graph, {key: len(value) for key, value in need_candidates.items()}
    )
    distance_by_pair = {
        (row.need_index, row.canonical_id): row.operation_distance
        for rows in need_candidates.values()
        for row in rows
    }

    def partial_rank(mapping: Mapping[int, str]) -> tuple[Any, ...]:
        semantic = math.fsum(
            distance_by_pair[(need_index, canonical_id)]
            for need_index, canonical_id in mapping.items()
        ) / max(1, len(mapping))
        assigned_edges = [
            edge
            for edge in need_graph.edges
            if edge.source in mapping and edge.target in mapping
        ]
        unsatisfied = sum(
            not _reachable(
                mapping[edge.source],
                mapping[edge.target],
                positions=positions,
                descendants=descendants,
            )
            for edge in assigned_edges
        )
        structural = unsatisfied / max(1, len(assigned_edges))
        return (
            semantic + structural,
            semantic,
            structural,
            tuple(sorted(mapping.items())),
        )

    beam: list[dict[int, str]] = [{}]
    states_expanded = 0
    states_pruned = 0
    for need_index in order:
        expanded: list[dict[int, str]] = []
        for mapping in beam:
            for option in need_candidates[need_index]:
                states_expanded += 1
                expanded.append({**mapping, need_index: option.canonical_id})
        expanded.sort(key=partial_rank)
        deduplicated: list[dict[int, str]] = []
        seen: set[tuple[tuple[int, str], ...]] = set()
        for mapping in expanded:
            signature = tuple(sorted(mapping.items()))
            if signature in seen:
                continue
            seen.add(signature)
            deduplicated.append(mapping)
            if len(deduplicated) == partial_beam_k:
                break
        states_pruned += max(0, len(expanded) - len(deduplicated))
        beam = deduplicated

    candidates_by_signature: dict[
        tuple[Any, ...], SubgraphCandidate
    ] = {}
    for mapping in beam:
        candidate = _candidate_from_mapping(
            mapping,
            need_graph=need_graph,
            need_candidates=need_candidates,
            region_graph=search_graph,
            index=index,
            workflow_recalls=workflow_recalls,
        )
        if candidate is None:
            continue
        signature = (
            candidate.need_mapping,
            candidate.canonical_node_ids,
            candidate.edges,
        )
        previous = candidates_by_signature.get(signature)
        if previous is None or candidate.rank_key < previous.rank_key:
            candidates_by_signature[signature] = candidate
    ranked_candidates = sorted(
        candidates_by_signature.values(), key=lambda row: row.rank_key
    )
    finalized = tuple(
        replace(row, candidate_id=f"C{position}")
        for position, row in enumerate(ranked_candidates[:result_top_k])
    )
    metrics: dict[str, int | float | str] = {
        "canonical_node_count": len(index.canonical_ids),
        "region_node_count": len(region),
        "region_edge_count": anchor_graph.number_of_edges(),
        "search_graph_node_count": search_graph.number_of_nodes(),
        "search_graph_edge_count": search_graph.number_of_edges(),
        "partial_beam_k": partial_beam_k,
        "states_expanded": states_expanded,
        "states_pruned": states_pruned,
        "leaf_mappings": len(beam),
        "candidate_count": len(finalized),
        "recall_score_format": RECALL_SCORE_FORMAT,
        "online_pair_enumerations": 0,
        "search_elapsed_seconds": time.monotonic() - started,
    }
    return RetrievalSearchResult(need_candidates, region, finalized, metrics)


__all__ = [
    "DependencyWitness",
    "NEED_CANONICAL_TOP_K",
    "PARTIAL_BEAM_K",
    "NeedCandidate",
    "OccurrenceEvidence",
    "RetrievalIndex",
    "RetrievalSearchResult",
    "SEMANTIC_RECALL_DOCUMENT_FORMAT",
    "RECALL_SCORE_FORMAT",
    "SUBGRAPH_TOP_K",
    "SubgraphCandidate",
    "build_retrieval_index",
    "canonical_retrieval_document",
    "need_retrieval_document",
    "retrieve_experience_subgraphs",
]
