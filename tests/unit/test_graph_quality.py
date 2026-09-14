from __future__ import annotations

import hashlib
from pathlib import Path

from degs.core import canonical_json_bytes
from degs.graph_quality import build_graph_quality_artifacts
from degs.section_graph import (
    CanonicalExperience,
    CanonicalGroup,
    CanonicalPartition,
    ExperienceNode,
    IOContract,
    SectionGraphSource,
    WorkflowGraph,
    compile_experience_graph,
    experience_leaf_id,
)
from degs.state_store import IncrementalStateStore


def _node(label: str) -> ExperienceNode:
    return ExperienceNode(
        f"Validate {label}",
        (f"Use when {label} must satisfy a task-specific invariant.",),
        (IOContract("artifact", f"{label} input"),),
        (IOContract("evidence", f"validated {label}"),),
    )


def _source() -> SectionGraphSource:
    workflows = tuple(
        WorkflowGraph(index, f"task-{index}", f"Validate item {index}", (_node(str(index)),), ())
        for index in range(2)
    )
    body = {
        "format": "degs_experience_workflows_v4",
        "source_split": "train[0,200)",
        "workflows": [
            {
                "train_index": row.train_index,
                "task_id": row.task_id,
                "query_text": row.query_text,
                "experience_nodes": [node.to_dict() for node in row.experience_nodes],
                "edges": [],
            }
            for row in workflows
        ],
    }
    return SectionGraphSource(
        workflows, hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    )


def _partition(source: SectionGraphSource, *, merge: bool) -> CanonicalPartition:
    groups = (
        (
            CanonicalGroup(
                ((0, 0), (1, 0)),
                CanonicalExperience(
                    "Validate a task-specific invariant.",
                    ("Use when an artifact has an explicit invariant.",),
                    (IOContract("artifact", "artifact input"),),
                    (IOContract("evidence", "validated invariant"),),
                ),
            ),
        )
        if merge
        else tuple(
            CanonicalGroup(
                ((index, 0),),
                CanonicalExperience(
                    node.operation, node.applicability, node.inputs, node.outputs
                ),
            )
            for index, node in enumerate((_node("0"), _node("1")))
        )
    )
    body = {
        "format": "degs_canonical_experience_partition_v4",
        "section_graphs_sha256": source.sha256,
        "groups": [
            {
                "members": [list(member) for member in group.members],
                "canonical_experience": group.canonical_experience.to_dict(),
            }
            for group in groups
        ],
    }
    return CanonicalPartition(
        groups,
        source.sha256,
        hashlib.sha256(canonical_json_bytes(body)).hexdigest(),
    )


def _ledger() -> dict:
    return {
        "batches": [
            {
                "batch_source_audit": {
                    "rows": [
                        {
                            "train_index": index,
                            "task_id": f"task-{index}",
                            "source_review_status": "REVIEW_ACCEPTED",
                            "source_review_ledger_status": (
                                "REVIEW_LEDGER_INCOMPLETE"
                            ),
                            "draft_graph_sha256": "a" * 64,
                            "draft_graph": {"experience_nodes": [], "edges": []},
                            "source_review_decisions": [],
                        }
                        for index in range(2)
                    ],
                    "exclusions": [],
                }
            }
        ]
    }


def test_graph_quality_reports_connectivity_without_optimizing_for_one_component(
    tmp_path: Path,
) -> None:
    source = _source()
    partition = _partition(source, merge=False)
    graph = compile_experience_graph(source, partition)
    with IncrementalStateStore(tmp_path / "state.sqlite3") as state:
        artifacts = build_graph_quality_artifacts(
            snapshot_id="snapshot-test",
            source=source,
            partition=partition,
            graph=graph,
            canonical_audit={
                "views": [],
                "candidate_graph": {"directed_neighbors": []},
                "cluster_decisions": [],
                "merge_proofs": [],
                "fidelity": [],
            },
            cumulative_source_audit=_ledger(),
            state=state,
        )
    assert artifacts.status == "READY"
    assert artifacts.topology["weak_component_count"] == 2
    assert artifacts.topology["workflow_overlap_component_sizes"] == [1, 1]
    assert artifacts.topology[
        "workflows_with_cross_workflow_canonical_rate"
    ] == 0.0
    assert artifacts.topology["cross_workflow_source_node_rate"] == 0.0
    assert artifacts.audit["metrics"]["singleton_rate"] == 1.0
    assert artifacts.audit["hard_violations"] == []
    assert set(artifacts.serialized_files()) == {
        "graph_quality_audit.json",
        "source_node_audit.jsonl",
        "high_similarity_unmerged.jsonl",
        "candidate_recall_audit.jsonl",
        "canonical_merge_ledger.jsonl",
        "graph_topology.json",
    }


def test_legal_source_exclusion_is_not_an_invalid_review_status(
    tmp_path: Path,
) -> None:
    source = _source()
    partition = _partition(source, merge=False)
    ledger = _ledger()
    ledger["batches"][0]["batch_source_audit"]["exclusions"].append(
        {
            "train_index": 2,
            "task_id": "task-2",
            "trajectory_id": "trajectory-2",
            "origin": "ORIGINAL_FAILURE",
            "status": "SOURCE_EXCLUDED_NO_VALIDATED_SUCCESS",
            "replay_terminal_status": "REPLAY_EXHAUSTED",
        }
    )
    with IncrementalStateStore(tmp_path / "state.sqlite3") as state:
        artifacts = build_graph_quality_artifacts(
            snapshot_id="snapshot-test",
            source=source,
            partition=partition,
            graph=compile_experience_graph(source, partition),
            canonical_audit={
                "views": [],
                "candidate_graph": {"directed_neighbors": []},
                "cluster_decisions": [],
                "merge_proofs": [],
                "fidelity": [],
            },
            cumulative_source_audit=ledger,
            state=state,
        )

    assert artifacts.status == "READY"
    assert artifacts.audit["hard_violations"] == []


def test_source_node_audit_preserves_removed_draft_disposition(
    tmp_path: Path,
) -> None:
    source = _source()
    partition = _partition(source, merge=False)
    graph = compile_experience_graph(source, partition)
    ledger = _ledger()
    first = ledger["batches"][0]["batch_source_audit"]["rows"][0]
    first["draft_graph"] = {
        "experience_nodes": [
            _node("0").to_dict(),
            _node("obsolete").to_dict(),
        ],
        "edges": [],
    }
    first["source_review_ledger_status"] = "REVIEW_LEDGER_COMPLETE"
    first["source_review_decisions"] = [
        {
            "draft_node": 0,
            "decision": "KEEP",
            "final_nodes": [0],
            "basis": "The operation remains necessary.",
        },
        {
            "draft_node": 1,
            "decision": "REMOVE",
            "final_nodes": [],
            "basis": "The step is routine execution mechanics.",
        },
    ]
    with IncrementalStateStore(tmp_path / "state.sqlite3") as state:
        artifacts = build_graph_quality_artifacts(
            snapshot_id="snapshot-test",
            source=source,
            partition=partition,
            graph=graph,
            canonical_audit={
                "views": [],
                "candidate_graph": {"directed_neighbors": []},
                "cluster_decisions": [],
                "merge_proofs": [],
                "fidelity": [],
            },
            cumulative_source_audit=ledger,
            state=state,
        )
    removed = next(
        row
        for row in artifacts.source_node_rows
        if row["train_index"] == 0 and row["draft_node_index"] == 1
    )
    assert removed["review_outcome"] == "REMOVE"
    assert removed["final_nodes"] == []
    assert removed["removal_or_fallback_reason"] == (
        "The step is routine execution mechanics."
    )


def test_source_node_audit_preserves_all_removed_excluded_workflow(
    tmp_path: Path,
) -> None:
    source_body = {
        "format": "degs_experience_workflows_v4",
        "source_split": "train[0,200)",
        "workflows": [],
    }
    source = SectionGraphSource(
        (), hashlib.sha256(canonical_json_bytes(source_body)).hexdigest()
    )
    partition_body = {
        "format": "degs_canonical_experience_partition_v4",
        "section_graphs_sha256": source.sha256,
        "groups": [],
    }
    partition = CanonicalPartition(
        (),
        source.sha256,
        hashlib.sha256(canonical_json_bytes(partition_body)).hexdigest(),
    )
    graph = compile_experience_graph(source, partition)
    removed_node = _node("removed").to_dict()
    ledger = {
        "batches": [
            {
                "batch_source_audit": {
                    "rows": [],
                    "exclusions": [
                        {
                            "train_index": 0,
                            "task_id": "task-removed",
                            "source_review_status": "REVIEW_ACCEPTED",
                            "source_review_ledger_status": (
                                "REVIEW_LEDGER_COMPLETE"
                            ),
                            "draft_graph_sha256": "a" * 64,
                            "draft_graph": {
                                "experience_nodes": [removed_node],
                                "edges": [],
                            },
                            "source_review_decisions": [
                                {
                                    "draft_node": 0,
                                    "decision": "REMOVE",
                                    "final_nodes": [],
                                    "basis": "No independently reusable experience.",
                                }
                            ],
                        }
                    ],
                }
            }
        ]
    }
    with IncrementalStateStore(tmp_path / "state.sqlite3") as state:
        artifacts = build_graph_quality_artifacts(
            snapshot_id="snapshot-empty",
            source=source,
            partition=partition,
            graph=graph,
            canonical_audit={
                "views": [],
                "candidate_graph": {"directed_neighbors": []},
                "cluster_decisions": [],
                "merge_proofs": [],
                "fidelity": [],
            },
            cumulative_source_audit=ledger,
            state=state,
        )
    assert len(artifacts.source_node_rows) == 1
    row = artifacts.source_node_rows[0]
    assert row["draft_source_node"] == removed_node
    assert row["review_outcome"] == "REMOVE"
    assert row["final_nodes"] == []


def test_source_node_audit_includes_generation_exclusion_without_a_draft(
    tmp_path: Path,
) -> None:
    source = _source()
    partition = _partition(source, merge=False)
    ledger = _ledger()
    ledger["batches"][0]["batch_source_audit"]["exclusions"].append(
        {
            "train_index": 2,
            "task_id": "task-2",
            "trajectory_id": "trajectory-2",
            "origin": "ORIGINAL_SUCCESS",
            "status": "SOURCE_EXCLUDED_GENERATION_FAILURE",
            "semantic_attempt_count": 3,
            "invalid_response_attempts": ["bad json"] * 3,
        }
    )
    with IncrementalStateStore(tmp_path / "state.sqlite3") as state:
        artifacts = build_graph_quality_artifacts(
            snapshot_id="snapshot-test",
            source=source,
            partition=partition,
            graph=compile_experience_graph(source, partition),
            canonical_audit={
                "views": [],
                "candidate_graph": {
                    "directed_neighbors": [],
                    "candidate_edges": [],
                },
                "cluster_decisions": [],
                "merge_proofs": [],
                "synthesis_requests": [],
                "fidelity": [],
            },
            cumulative_source_audit=ledger,
            state=state,
        )
    excluded = next(
        row for row in artifacts.source_node_rows if row["train_index"] == 2
    )
    assert excluded["draft_node_index"] is None
    assert excluded["review_outcome"] == "SOURCE_EXCLUDED_GENERATION_FAILURE"
    assert artifacts.audit["metrics"]["source_exclusion_count"] == 1
    assert artifacts.audit["metrics"]["source_exclusion_status_counts"] == {
        "SOURCE_EXCLUDED_GENERATION_FAILURE": 1
    }

def test_unproved_new_group_is_diagnostic_not_ready_not_a_retrieval_gate(tmp_path):
    source = _source()
    partition = _partition(source, merge=True)
    with IncrementalStateStore(tmp_path / "state.sqlite3") as state:
        result = build_graph_quality_artifacts(snapshot_id="test", source=source, partition=partition,
            graph=compile_experience_graph(source, partition), canonical_audit={"prior_groups": [], "merge_events": []},
            cumulative_source_audit=_ledger(), state=state)
    assert result.status == "NOT_READY"
    assert result.audit["retrieval_allowed_by_quality_policy"]
    assert any(row["check"] == "one_call_merge_evidence" for row in result.audit["hard_violations"])


def test_committed_group_is_accepted_as_a_whole_without_new_semantic_proof(tmp_path):
    source = _source()
    partition = _partition(source, merge=True)
    from degs.section_graph import _canonical_id
    prior = [{"canonical_id": _canonical_id(g.members), "members": [list(m) for m in g.members]} for g in partition.groups]
    with IncrementalStateStore(tmp_path / "state.sqlite3") as state:
        result = build_graph_quality_artifacts(snapshot_id="test", source=source, partition=partition,
            graph=compile_experience_graph(source, partition), canonical_audit={"prior_groups": prior, "merge_events": []},
            cumulative_source_audit=_ledger(), state=state)
    assert result.status == "READY"
    assert result.topology["workflow_overlap_component_sizes"] == [2]


def test_splitting_a_committed_group_is_detected(tmp_path):
    source = _source()
    partition = _partition(source, merge=False)
    with IncrementalStateStore(tmp_path / "state.sqlite3") as state:
        result = build_graph_quality_artifacts(snapshot_id="test", source=source, partition=partition,
            graph=compile_experience_graph(source, partition),
            canonical_audit={"prior_groups": [{"canonical_id": "prior", "members": [[0, 0], [1, 0]]}]},
            cumulative_source_audit=_ledger(), state=state)
    assert result.audit["metrics"]["prior_group_split_count"] == 1
    assert result.status == "NOT_READY"
