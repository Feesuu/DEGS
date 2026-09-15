from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .core import canonical_json_bytes
from .eir_graph import compile_eir_experience_graph
from .graph_dataset_contract import SPREADSHEETBENCH_GRAPH_CONTRACT
from .state_store import EIRStateStore


EIR_GRAPH_QUALITY_FORMAT = "degs_eir_graph_quality_audit_v1"


def _components(nodes: Sequence[str], edges: Sequence[tuple[str, str]]) -> list[set[str]]:
    adjacency: defaultdict[str, set[str]] = defaultdict(set)
    for source, target in edges:
        adjacency[source].add(target)
        adjacency[target].add(source)
    unseen = set(nodes)
    result: list[set[str]] = []
    while unseen:
        root = min(unseen)
        component: set[str] = set()
        queue = deque([root])
        while queue:
            current = queue.popleft()
            if current not in unseen:
                continue
            unseen.remove(current)
            component.add(current)
            queue.extend(sorted(adjacency[current] & unseen))
        result.append(component)
    return sorted(result, key=lambda row: (-len(row), sorted(row)))


def build_eir_graph_quality_audit(
    store: EIRStateStore, *, snapshot_id: str
) -> Mapping[str, Any]:
    graph, versions = compile_eir_experience_graph(store, snapshot_id=snapshot_id)
    node_ids = [row.canonical_id for row in graph.nodes]
    edge_pairs = [(row.source, row.target) for row in graph.edges]
    components = _components(node_ids, edge_pairs)
    member_counts = {
        canonical_id: len(store.canonical_member_ids(canonical_id, snapshot_id=snapshot_id))
        for canonical_id in node_ids
    }
    action_counts = dict(
        Counter(
            str(row[0])
            for row in store.connection.execute(
                "SELECT action FROM eir_experience_events ORDER BY event_id"
            )
        )
    )
    history_version_count = int(
        store.connection.execute("SELECT COUNT(*) FROM eir_canonical_versions").fetchone()[0]
    )
    outcome_counts = dict(
        Counter(
            str(row[0])
            for row in store.connection.execute(
                "SELECT outcome FROM eir_episodes ORDER BY train_index"
            )
        )
    )
    endpoint_episodes: defaultdict[str, set[str]] = defaultdict(set)
    for canonical_id, episode_id in store.connection.execute(
        """
        SELECT member.canonical_id, source.episode_id
        FROM eir_canonical_members AS member
        JOIN eir_source_nodes AS source ON source.source_node_id = member.source_node_id
        """
    ):
        endpoint_episodes[store.resolve_canonical_id(str(canonical_id), snapshot_id=snapshot_id)].add(
            str(episode_id)
        )
    cross_workflow_edges = sum(
        bool(endpoint_episodes[edge.source] - endpoint_episodes[edge.target])
        and bool(endpoint_episodes[edge.target] - endpoint_episodes[edge.source])
        for edge in graph.edges
    )
    body = {
        "format": EIR_GRAPH_QUALITY_FORMAT,
        "role": "diagnostic_only_never_a_runtime_gate",
        "snapshot_id": snapshot_id,
        "graph_identity": graph.identity(),
        "active_canonical_count": len(graph.nodes),
        "historical_canonical_version_count": history_version_count,
        "active_version_counts": {
            str(version): count
            for version, count in sorted(Counter(versions.values()).items())
        },
        "source_node_count": sum(member_counts.values()),
        "singleton_canonical_count": sum(value == 1 for value in member_counts.values()),
        "canonical_member_counts": member_counts,
        "procedure_edge_count": len(graph.edges),
        "cross_workflow_edge_count": cross_workflow_edges,
        "component_count": len(components),
        "largest_component_size": len(components[0]) if components else 0,
        "isolated_node_count": sum(len(row) == 1 for row in components),
        "component_sizes": [len(row) for row in components],
        "experience_action_counts": action_counts,
        "episode_outcome_counts": outcome_counts,
        "learning_delta_status_counts": dict(
            Counter(
                str(row[0])
                for row in store.connection.execute(
                    "SELECT status FROM eir_learning_deltas ORDER BY episode_id"
                )
            )
        ),
        "deferred_revision_count": store.deferred_revision_count(),
    }
    return {
        **body,
        "self_sha256": hashlib.sha256(canonical_json_bytes(body)).hexdigest(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit an EIR graph without gating it.")
    parser.add_argument("--state-db", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    with EIRStateStore(
        args.state_db,
        dataset_contract=SPREADSHEETBENCH_GRAPH_CONTRACT,
    ) as store:
        snapshot_id = store.head_snapshot_id
        if snapshot_id is None:
            raise ValueError("EIR graph audit requires a committed snapshot")
        audit = build_eir_graph_quality_audit(store, snapshot_id=snapshot_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(audit) + b"\n")
    print(json.dumps(audit, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["EIR_GRAPH_QUALITY_FORMAT", "build_eir_graph_quality_audit"]
