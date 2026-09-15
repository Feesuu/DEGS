from __future__ import annotations

from degs.contextual_retrieval import retrieve_contextual_subgraph
from degs.section_graph import (
    CanonicalExperience,
    CanonicalNode,
    ExperienceGraph,
    IOContract,
    ProjectedEdge,
)


def _node(canonical_id: str) -> CanonicalNode:
    experience = CanonicalExperience(
        f"Perform operation {canonical_id}.",
        ("The task requires this operation.",),
        (IOContract("input", "Current task state."),),
        (IOContract("output", "Updated task state."),),
    )
    return CanonicalNode(canonical_id, experience, canonical_id, canonical_id * 8)


def _graph() -> ExperienceGraph:
    nodes = tuple(_node(f"C{index}") for index in range(7))
    edges = (
        ProjectedEdge("C2", "C0", (0,)),
        ProjectedEdge("C0", "C3", (0,)),
        ProjectedEdge("C0", "C4", (1,)),
        ProjectedEdge("C5", "C0", (2,)),
        ProjectedEdge("C1", "C6", (3,)),
    )
    return ExperienceGraph(nodes, edges, "s", "p", "g")


def test_retrieval_keeps_top_anchors_and_at_most_two_neighbors_each() -> None:
    graph = _graph()
    vectors = {
        "C0": (1.0, 0.0),
        "C1": (0.9, 0.1),
        "C2": (0.8, 0.2),
        "C3": (0.7, 0.3),
        "C4": (0.6, 0.4),
        "C5": (0.5, 0.5),
        "C6": (0.4, 0.6),
    }
    result = retrieve_contextual_subgraph(
        graph=graph,
        snapshot_id="S1",
        query_vector=(1.0, 0.0),
        canonical_vectors=vectors,
        canonical_versions={node.canonical_id: 1 for node in graph.nodes},
        top_k=2,
        neighbors_per_anchor=2,
    )
    assert [row.canonical_id for row in result.anchors] == ["C0", "C1"]
    assert len(result.context_nodes) <= 4
    assert {row.canonical_id for row in result.context_nodes} == {"C2", "C3", "C6"}
    assert all(row.canonical_id not in {"C0", "C1"} for row in result.context_nodes)


def test_neighbor_nodes_are_context_not_selectable_anchors() -> None:
    result = retrieve_contextual_subgraph(
        graph=_graph(),
        snapshot_id="S1",
        query_vector=(1.0, 0.0),
        canonical_vectors={f"C{index}": (1.0 - index / 10, index / 10) for index in range(7)},
        canonical_versions={f"C{index}": 4 for index in range(7)},
        top_k=1,
    )
    payload = result.to_binding_payload(
        query={"evidence_id": "query:0", "content": "Do the task."},
        observable_context=[],
    )
    assert [row["canonical_id"] for row in payload["anchors"]] == ["C0"]
    assert all(row["canonical_id"] != "C0" for row in payload["context_nodes"])
    assert result.anchor_versions == {"C0": 4}


def test_empty_graph_returns_empty_context_without_vectors() -> None:
    result = retrieve_contextual_subgraph(
        graph=ExperienceGraph((), (), "s", "p", "g"),
        snapshot_id="EMPTY",
        query_vector=(1.0, 0.0),
        canonical_vectors={},
        canonical_versions={},
    )
    assert result.anchors == ()
    assert result.context_nodes == ()
    assert result.edges == ()
