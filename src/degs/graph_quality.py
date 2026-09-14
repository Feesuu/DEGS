from __future__ import annotations

from collections import Counter
import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import networkx as nx

from .canonicalize import (
    CANONICAL_CANDIDATE_K,
    CANONICAL_RECALL_AUDIT_K,
    CanonicalizationView,
    TemplateRelation,
    canonical_unit_candidates,
    canonicalization_view_embedding_text,
    parse_canonical_merge,
)
from .core import canonical_json_bytes, normalize_embedding_text
from .section_graph import (
    CanonicalPartition,
    ExperienceGraph,
    SectionGraphSource,
    _canonical_id,
    compile_experience_graph,
    experience_leaf_id,
    load_canonical_partition,
    load_section_graphs,
)
from .state_store import IncrementalStateStore
from .source_rebuild import SOURCE_REVIEW_RETRY_STATUS


GRAPH_QUALITY_AUDIT_FORMAT = "degs_graph_quality_audit_v4"
GRAPH_QUALITY_PROTOCOL = {
    "format": GRAPH_QUALITY_AUDIT_FORMAT,
    "hard_checks": ["complete_exact_partition", "traceable_source_edge_projection",
                    "monotonic_prior_groups", "one_call_merge_evidence", "explicit_source_review_status"],
    "diagnostic_only": ["source_nodes_per_workflow", "canonical_group_sizes",
                        "singleton_rate", "cross_workflow_merge_coverage",
                        "weak_and_strong_component_structure", "edge_provenance",
                        "high_similarity_unmerged_candidates", "candidate_recall_ranks_17_64",
                        "hub_removal_sensitivity"],
    "execution_gate": False,
    "canonical_cycles_and_self_loops": "allowed_with_real_source_edge_provenance",
}
GRAPH_QUALITY_PROTOCOL_SHA256 = hashlib.sha256(
    canonical_json_bytes(GRAPH_QUALITY_PROTOCOL)
).hexdigest()


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True)
class GraphQualityArtifacts:
    audit: dict[str, Any]
    source_node_rows: tuple[dict[str, Any], ...]
    high_similarity_unmerged_rows: tuple[dict[str, Any], ...]
    candidate_recall_rows: tuple[dict[str, Any], ...]
    merge_ledger_rows: tuple[dict[str, Any], ...]
    topology: dict[str, Any]

    @property
    def status(self) -> str:
        return str(self.audit["status"])

    def serialized_files(self) -> dict[str, bytes]:
        return {
            "graph_quality_audit.json": canonical_json_bytes(self.audit),
            "source_node_audit.jsonl": _jsonl_bytes(self.source_node_rows),
            "high_similarity_unmerged.jsonl": _jsonl_bytes(
                self.high_similarity_unmerged_rows
            ),
            "candidate_recall_audit.jsonl": _jsonl_bytes(
                self.candidate_recall_rows
            ),
            "canonical_merge_ledger.jsonl": _jsonl_bytes(
                self.merge_ledger_rows
            ),
            "graph_topology.json": canonical_json_bytes(self.topology),
        }


def _source_review_rows(
    cumulative_source_audit: Mapping[str, Any],
) -> dict[int, Mapping[str, Any]]:
    batches = cumulative_source_audit.get("batches")
    if type(batches) is not list:
        raise ValueError("cumulative source audit batches differ")
    result: dict[int, Mapping[str, Any]] = {}
    for batch in batches:
        if type(batch) is not dict:
            raise ValueError("cumulative source audit batch differs")
        batch_audit = batch.get("batch_source_audit")
        if (
            type(batch_audit) is not dict
            or type(batch_audit.get("rows")) is not list
            or type(batch_audit.get("exclusions")) is not list
        ):
            raise ValueError("cumulative source review audit differs")
        for row in (*batch_audit["rows"], *batch_audit["exclusions"]):
            if type(row) is not dict or type(row.get("train_index")) is not int:
                raise ValueError("source review row identity differs")
            train_index = int(row["train_index"])
            if train_index in result:
                raise ValueError("source review row repeats a train index")
            result[train_index] = row
    return result


def _topology(
    source: SectionGraphSource,
    graph: ExperienceGraph,
    partition: CanonicalPartition,
) -> dict[str, Any]:
    directed = nx.DiGraph()
    directed.add_nodes_from(node.canonical_id for node in graph.nodes)
    directed.add_edges_from((edge.source, edge.target) for edge in graph.edges)
    undirected = directed.to_undirected()
    weak_sizes = sorted(
        (len(component) for component in nx.weakly_connected_components(directed)),
        reverse=True,
    )
    strong_sizes = sorted(
        (len(component) for component in nx.strongly_connected_components(directed)),
        reverse=True,
    )
    articulation = sorted(nx.articulation_points(undirected)) if undirected else []
    degrees = dict(undirected.degree())
    hub = min(
        (node_id for node_id, degree in degrees.items() if degree == max(degrees.values())),
        default=None,
    ) if degrees else None
    after_hub = undirected.copy()
    if hub is not None:
        after_hub.remove_node(hub)
    after_sizes = sorted(
        (len(component) for component in nx.connected_components(after_hub)),
        reverse=True,
    ) if after_hub else []
    cross_workflow = [
        _canonical_id(group.members)
        for group in partition.groups
        if len({train_index for train_index, _node_index in group.members}) > 1
    ]
    source_node_count = sum(len(group.members) for group in partition.groups)
    cross_workflow_source_node_count = sum(
        len(group.members)
        for group in partition.groups
        if len({train_index for train_index, _node_index in group.members}) > 1
    )
    workflows_by_canonical = {
        _canonical_id(group.members): {
            train_index for train_index, _node_index in group.members
        }
        for group in partition.groups
    }
    workflow_ids = sorted(
        {
            train_index
            for group in partition.groups
            for train_index, _node_index in group.members
        }
    )
    workflow_overlap = nx.Graph()
    workflow_overlap.add_nodes_from(workflow_ids)
    workflows_with_cross_canonical: set[int] = set()
    for member_workflows in workflows_by_canonical.values():
        ordered = sorted(member_workflows)
        if len(ordered) > 1:
            workflows_with_cross_canonical.update(ordered)
        workflow_overlap.add_edges_from(
            (left, right)
            for index, left in enumerate(ordered)
            for right in ordered[index + 1 :]
        )
    workflow_component_sizes = sorted(
        (
            len(component)
            for component in nx.connected_components(workflow_overlap)
        ),
        reverse=True,
    )
    canonical_component_workflow_counts = sorted(
        (
            len(
                set().union(
                    *(workflows_by_canonical[node_id] for node_id in component)
                )
            )
            for component in nx.weakly_connected_components(directed)
        ),
        reverse=True,
    )
    edge_support = Counter(
        len(edge.supporting_workflow_ids) for edge in graph.edges
    )
    source_nodes_per_workflow = Counter(
        len(workflow.experience_nodes) for workflow in source.workflows
    )
    canonical_group_sizes = Counter(len(group.members) for group in partition.groups)
    largest_component_node_count = weak_sizes[0] if weak_sizes else 0
    largest_component_workflow_count = (
        canonical_component_workflow_counts[0]
        if canonical_component_workflow_counts
        else 0
    )
    return {
        "format": "degs_graph_topology_v2",
        "canonical_node_count": len(graph.nodes),
        "projected_edge_count": len(graph.edges),
        "weak_component_count": len(weak_sizes),
        "weak_component_sizes": weak_sizes,
        "strong_component_count": len(strong_sizes),
        "strong_component_sizes": strong_sizes,
        "isolated_canonical_ids": sorted(nx.isolates(directed)),
        "singleton_canonical_count": sum(
            len(group.members) == 1 for group in partition.groups
        ),
        "cross_workflow_canonical_count": len(cross_workflow),
        "cross_workflow_canonical_ids": sorted(cross_workflow),
        "cross_workflow_source_node_count": cross_workflow_source_node_count,
        "cross_workflow_source_node_rate": (
            cross_workflow_source_node_count / source_node_count
            if source_node_count
            else 0.0
        ),
        "workflow_count": len(workflow_ids),
        "workflows_with_cross_workflow_canonical_count": len(
            workflows_with_cross_canonical
        ),
        "workflows_with_cross_workflow_canonical_rate": (
            len(workflows_with_cross_canonical) / len(workflow_ids)
            if workflow_ids
            else 0.0
        ),
        "workflow_overlap_component_count": len(workflow_component_sizes),
        "workflow_overlap_component_sizes": workflow_component_sizes,
        "canonical_component_distinct_workflow_counts": (
            canonical_component_workflow_counts
        ),
        "source_nodes_per_workflow": {
            str(key): source_nodes_per_workflow[key]
            for key in sorted(source_nodes_per_workflow)
        },
        "canonical_group_sizes": {
            str(key): canonical_group_sizes[key]
            for key in sorted(canonical_group_sizes)
        },
        "largest_weak_component_node_rate": (
            largest_component_node_count / len(graph.nodes) if graph.nodes else 0.0
        ),
        "largest_weak_component_workflow_count": largest_component_workflow_count,
        "largest_weak_component_workflow_rate": (
            largest_component_workflow_count / len(workflow_ids)
            if workflow_ids
            else 0.0
        ),
        "edge_provenance_workflow_count_distribution": {
            str(key): edge_support[key] for key in sorted(edge_support)
        },
        "edge_provenance_multiplicity_is_quality_signal": False,
        "maximum_undirected_degree": max(degrees.values(), default=0),
        "articulation_canonical_ids": articulation,
        "hub_removal": {
            "removed_canonical_id": hub,
            "largest_weak_component_before": weak_sizes[0] if weak_sizes else 0,
            "largest_component_after": after_sizes[0] if after_sizes else 0,
        },
    }


def build_graph_quality_artifacts(
    *, snapshot_id: str, source: SectionGraphSource, partition: CanonicalPartition,
    graph: ExperienceGraph, canonical_audit: Mapping[str, Any],
    cumulative_source_audit: Mapping[str, Any], state: IncrementalStateStore,
) -> GraphQualityArtifacts:
    """Audit evidence and connectivity without vetoing retrieval on quality scores."""
    hard_violations: list[dict[str, Any]] = []
    def violation(check: str, detail: str) -> None:
        hard_violations.append({"check": check, "detail": detail})
    workflow_by_index = source.workflow_by_index
    known = {(w.train_index, i) for w in source.workflows for i, _ in enumerate(w.experience_nodes)}
    covered = [member for group in partition.groups for member in group.members]
    if len(covered) != len(set(covered)) or set(covered) != known:
        violation("complete_exact_partition", "source coverage differs")
    else:
        if compile_experience_graph(source, partition) != graph:
            violation("traceable_source_edge_projection", "projected graph differs")
    coordinate_to_leaf = {
        (w.train_index, i): experience_leaf_id(w.train_index, i, node)
        for w in source.workflows for i, node in enumerate(w.experience_nodes)
    }
    leaf_to_canonical = {coordinate_to_leaf[m]: _canonical_id(group.members)
                         for group in partition.groups for m in group.members}
    heads = {m: _canonical_id(group.members) for group in partition.groups for m in group.members}
    splits = 0
    for prior in canonical_audit.get("prior_groups", []):
        destinations = {heads.get(tuple(m)) for m in prior["members"]}
        if None in destinations or len(destinations) != 1:
            splits += 1
    if splits:
        violation("monotonic_prior_groups", f"{splits} previously committed groups were split")

    prior_units = {row["canonical_id"]: {tuple(m) for m in row["members"]}
                   for row in canonical_audit.get("prior_groups", [])}
    prior_members = set().union(*prior_units.values()) if prior_units else set()
    units = {**prior_units, **{_canonical_id((m,)): {m} for m in known - prior_members}}
    for event in canonical_audit.get("merge_events", []):
        left = units.get(event["left_canonical_id"])
        right = units.get(event["right_canonical_id"])
        if (left is None or right is None or left & right
            or _canonical_id(sorted(left | right)) != event["child_canonical_id"]):
            violation("one_call_merge_evidence", "event does not merge two complete current units")
            continue
        del units[event["left_canonical_id"]], units[event["right_canonical_id"]]
        units[event["child_canonical_id"]] = left | right
    if units != {_canonical_id(g.members): set(g.members) for g in partition.groups}:
        violation("one_call_merge_evidence", "final partition is not the monotonic event projection")

    # Historical DIFFERENT/UNCERTAIN responses are observations, never cannot-links.
    merge_ledger = []
    current = list(canonical_audit.get("merge_events", []))
    historical = state.connection.execute(
        """SELECT snapshot_id, left_canonical_id, right_canonical_id,
                  child_canonical_id, request_sha256, apply_order
           FROM canonical_merge_events WHERE snapshot_id != ?
           ORDER BY rowid""", (snapshot_id,),
    ).fetchall()
    events = [{"snapshot_id": row[0], "left_canonical_id": row[1], "right_canonical_id": row[2],
               "child_canonical_id": row[3], "request_sha256": row[4], "apply_order": row[5]}
              for row in historical] + [{"snapshot_id": snapshot_id, **row} for row in current]
    for event in events:
        job = state.connection.execute(
            "SELECT stage, validated_response_json FROM canonical_jobs WHERE request_sha256 = ?",
            (event["request_sha256"],),
        ).fetchone()
        response = json.loads(job[1]) if job else None
        try:
            decision = parse_canonical_merge(response) if job and job[0] == "MERGE" else None
        except (TypeError, ValueError):
            decision = None
        valid = bool(
            decision is not None
            and decision.relation is TemplateRelation.SAME_TEMPLATE
            and decision.canonical_experience is not None
        )
        if not valid:
            violation("one_call_merge_evidence", f"missing SAME output for {event['child_canonical_id']}")
        merge_ledger.append({**event, "event_kind": "MERGE", "response": response, "evidence_valid": valid})
    review_by_train = _source_review_rows(cumulative_source_audit)
    for w in source.workflows:
        if not review_by_train.get(w.train_index, {}).get("source_review_status"):
            violation("explicit_source_review_status", f"missing review for train index {w.train_index}")
    views = {row["leaf_id"]: row for row in canonical_audit.get("views", [])}
    aliases = {_canonical_id(g.members): tuple(coordinate_to_leaf[m] for m in g.members) for g in partition.groups}
    cache = state.embedding_cache()
    vectors, texts = {}, {}
    for leaf_id, row in views.items():
        view = CanonicalizationView(**row["view"])
        text = normalize_embedding_text(canonicalization_view_embedding_text(view))
        key = _sha(text.encode("utf-8"))
        if key in cache:
            vectors[leaf_id], texts[leaf_id] = cache[key].vector, text
    neighbors, recall_rows, high_similarity_unmerged = {}, [], []
    for canonical_id in sorted(aliases):
        if not all(leaf in vectors for leaves in aliases.values() for leaf in leaves):
            break  # Seed import has no new View generation; explicitly no recall audit yet.
        ranked = canonical_unit_candidates(canonical_id, aliases, vectors, exact_text_by_leaf=texts,
                                           k=CANONICAL_RECALL_AUDIT_K)
        neighbors[canonical_id] = [row.to_dict() for row in ranked if row.rank <= CANONICAL_CANDIDATE_K or row.exact_text_match]
        for row in ranked:
            item = {**row.to_dict(), "diagnostic_only": True}
            if CANONICAL_CANDIDATE_K < row.rank <= CANONICAL_RECALL_AUDIT_K:
                recall_rows.append(item)
            if row.similarity >= 0.5:
                high_similarity_unmerged.append(item)
    # Local failures remain inspectable after later successful batches, without
    # being replayed as semantic restrictions or dropped from the audit.
    resolution_rows = [
        {"stage": stage, "status": status, "subject": json.loads(subject), "evidence": json.loads(evidence)}
        for stage, status, subject, evidence in state.connection.execute(
            "SELECT stage, status, subject_json, evidence_json FROM canonical_resolution_events WHERE created_snapshot_id != ? ORDER BY rowid",
            (snapshot_id,),
        )
    ] + list(canonical_audit.get("resolution_events", []))
    high_similarity_unmerged.extend({"diagnostic_only": True, "row_kind": "RESOLUTION_EVENT", **row}
                                    for row in resolution_rows)
    source_node_rows: list[dict[str, Any]] = []
    for train_index, review in sorted(review_by_train.items()):
        workflow = workflow_by_index.get(train_index)
        if "source_review_status" not in review:
            exclusion_status = review.get("status")
            if type(exclusion_status) is not str or not exclusion_status:
                raise ValueError("source exclusion audit status differs")
            source_node_rows.append(
                {
                    "train_index": train_index,
                    "task_id": review.get("task_id"),
                    "draft_node_index": None,
                    "draft_source_node": None,
                    "draft_graph_sha256": None,
                    "review_status": None,
                    "review_ledger_status": None,
                    "review_outcome": exclusion_status,
                    "review_disposition": None,
                    "final_nodes": [],
                    "removal_or_fallback_reason": str(
                        review.get("error", exclusion_status)
                    ),
                    "source_exclusion": dict(review),
                }
            )
            continue
        draft_graph = review.get("draft_graph", {})
        decisions = review.get("source_review_decisions", [])
        draft_nodes = draft_graph.get("experience_nodes", [])
        if type(draft_nodes) is not list or type(decisions) is not list:
            raise ValueError("source review draft-node audit differs")
        decisions_by_draft: dict[int, dict[str, Any]] = {}
        for decision in decisions:
            if type(decision) is not dict or type(decision.get("draft_node")) is not int:
                continue
            decisions_by_draft[int(decision["draft_node"])] = decision

        def final_record(node_index: int) -> dict[str, Any]:
            if workflow is None or not 0 <= node_index < len(workflow.experience_nodes):
                raise ValueError("source review final-node audit differs")
            node = workflow.experience_nodes[node_index]
            leaf_id = coordinate_to_leaf[(train_index, node_index)]
            canonical_id = leaf_to_canonical[leaf_id]
            return {
                "final_node_index": node_index,
                "leaf_id": leaf_id,
                "final_source_node": node.to_dict(),
                "canonical_id": canonical_id,
                "candidate_neighbors": neighbors.get(canonical_id, []),
                "view_generation_state": views.get(leaf_id, {}).get("status", "NOT_REQUESTED"),
                "merge_evidence": [row for row in merge_ledger if row.get("child_canonical_id") == canonical_id],

            }

        referenced_final_indices: set[int] = set()
        for draft_node_index, draft_node in enumerate(draft_nodes):
            decision = decisions_by_draft.get(draft_node_index)
            fallback = review.get("source_review_status") in {
                "REVIEW_DRAFT_FALLBACK",
                SOURCE_REVIEW_RETRY_STATUS,
            }
            if decision is not None:
                final_indices = decision.get("final_nodes", [])
                if type(final_indices) is not list or any(
                    type(index) is not int for index in final_indices
                ):
                    raise ValueError("source review disposition audit differs")
                outcome = str(decision.get("decision"))
                reason = str(decision.get("basis", ""))
            elif fallback and workflow is not None and draft_node_index < len(
                workflow.experience_nodes
            ):
                final_indices = [draft_node_index]
                outcome = "FALLBACK_RETAINED"
                reason = "semantic review unavailable; schema-valid draft retained"
            else:
                final_indices = []
                outcome = "REVIEW_LEDGER_INCOMPLETE"
                reason = "no valid review disposition maps this draft node"
            referenced_final_indices.update(final_indices)
            source_node_rows.append(
                {
                    "train_index": train_index,
                    "task_id": review.get("task_id"),
                    "draft_node_index": draft_node_index,
                    "draft_source_node": draft_node,
                    "draft_graph_sha256": review.get("draft_graph_sha256"),
                    "review_status": review.get("source_review_status"),
                    "review_ledger_status": review.get(
                        "source_review_ledger_status"
                    ),
                    "review_outcome": outcome,
                    "review_disposition": decision,
                    "final_nodes": [
                        final_record(index) for index in final_indices
                    ],
                    "removal_or_fallback_reason": reason,
                }
            )
        if workflow is not None:
            for node_index in sorted(
                set(range(len(workflow.experience_nodes)))
                - referenced_final_indices
            ):
                source_node_rows.append(
                    {
                        "train_index": train_index,
                        "task_id": workflow.task_id,
                        "draft_node_index": None,
                        "draft_source_node": None,
                        "draft_graph_sha256": review.get("draft_graph_sha256"),
                        "review_status": review.get("source_review_status"),
                        "review_ledger_status": review.get(
                            "source_review_ledger_status"
                        ),
                        "review_outcome": "UNMAPPED_FINAL_REVIEW_NODE",
                        "review_disposition": None,
                        "final_nodes": [final_record(node_index)],
                        "removal_or_fallback_reason": (
                            "final node has no valid draft disposition"
                        ),
                    }
                )

    topology = _topology(source, graph, partition)
    final_members = {_canonical_id(g.members): set(g.members) for g in partition.groups}
    prior_heads_by_final = Counter(heads[next(iter(members))] for members in prior_units.values() if members and members <= set(heads))
    changes = Counter()
    for prior_id, members in prior_units.items():
        if not members or not members <= set(heads):
            continue
        destination = heads[next(iter(members))]
        if final_members[destination] == members:
            changes["unchanged_prior_group_count"] += 1
        elif prior_heads_by_final[destination] > 1:
            changes["whole_merged_prior_group_count"] += 1
        else:
            changes["new_occurrence_absorbing_prior_group_count"] += 1
    old_graph, new_graph = nx.Graph(), nx.Graph()
    old_graph.add_nodes_from(prior_units)
    new_graph.add_nodes_from(final_members)
    new_graph.add_edges_from((edge.source, edge.target) for edge in graph.edges)
    prior_by_member = {m: key for key, members in prior_units.items() for m in members}
    internal_edges = []
    source_edge_count = 0
    for workflow in source.workflows:
        for edge in workflow.edges:
            source_edge_count += 1
            left, right = (workflow.train_index, edge.source), (workflow.train_index, edge.target)
            if left in prior_by_member and right in prior_by_member:
                old_graph.add_edge(prior_by_member[left], prior_by_member[right])
            if heads.get(left) == heads.get(right):
                internal_edges.append({"train_index": workflow.train_index, "source": edge.source,
                    "target": edge.target, "canonical_id": heads.get(left)})
    component_by_node = {node: i for i, component in enumerate(nx.connected_components(new_graph)) for node in component}
    broken_components = sum(len({component_by_node[heads[next(iter(prior_units[key]))]] for key in component}) > 1
                            for component in nx.connected_components(old_graph)) if not splits else None
    topology.update({"source_occurrence_edge_count": source_edge_count,
                     "internal_source_edge_count": len(internal_edges),
                     "self_loop_projected_edge_count": sum(edge.source == edge.target for edge in graph.edges),
                     "self_loop_occurrence_evidence": internal_edges,
                     "prior_connected_component_break_count": broken_components})
    hashes = {
        "source_node_audit_sha256": _sha(_jsonl_bytes(source_node_rows)),
        "merge_ledger_sha256": _sha(_jsonl_bytes(merge_ledger)),
        "unresolved_candidates_sha256": _sha(_jsonl_bytes(high_similarity_unmerged)),
        "candidate_recall_audit_sha256": _sha(_jsonl_bytes(recall_rows)),
        "topology_sha256": _sha(canonical_json_bytes(topology)),
    }
    body = {
        "format": GRAPH_QUALITY_AUDIT_FORMAT, "protocol": GRAPH_QUALITY_PROTOCOL,
        "protocol_sha256": GRAPH_QUALITY_PROTOCOL_SHA256, "snapshot_id": snapshot_id,
        "section_graphs_sha256": source.sha256, "canonical_partition_sha256": partition.sha256,
        "experience_graph_sha256": graph.experience_graph_sha256,
        "status": "NOT_READY" if hard_violations else "READY", "hard_violations": hard_violations,
        "retrieval_allowed_by_quality_policy": True,
        "metrics": {**topology, "source_node_count": len(known),
                    "singleton_rate": topology["singleton_canonical_count"] / len(partition.groups) if partition.groups else 0.0,
                    "prior_group_split_count": splits, "merge_event_count": len(events),
                    **{key: changes[key] for key in ("unchanged_prior_group_count", "whole_merged_prior_group_count", "new_occurrence_absorbing_prior_group_count")},
                    "high_similarity_unmerged_count": sum(row.get("row_kind") != "RESOLUTION_EVENT" for row in high_similarity_unmerged),
                    "high_similarity_pairs_are_directed": True,
                    "resolution_event_count": len(resolution_rows),
                    "resolution_status_counts": dict(sorted(Counter(row["status"] for row in resolution_rows).items())),
                    "source_review_fallback_count": sum(row.get("source_review_status") in {"REVIEW_DRAFT_FALLBACK", SOURCE_REVIEW_RETRY_STATUS} for row in review_by_train.values()),
                    "source_exclusion_count": sum("source_review_status" not in row for row in review_by_train.values()),
                    "source_exclusion_status_counts": dict(sorted(Counter(str(row.get("status")) for row in review_by_train.values() if "source_review_status" not in row).items()))},
        "artifact_hashes": hashes,
    }
    return GraphQualityArtifacts({**body, "self_sha256": _sha(canonical_json_bytes(body))},
        tuple(source_node_rows), tuple(high_similarity_unmerged), tuple(recall_rows), tuple(merge_ledger), topology)


def _canonical_json_file(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if type(value) is not dict or canonical_json_bytes(value) != raw:
        raise ValueError(f"{path.name} is not canonical JSON")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Recompute and print a DEGS graph-quality audit."
    )
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--state-db", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest_path = args.snapshot_manifest.expanduser().absolute()
    manifest = _canonical_json_file(manifest_path)
    artifacts = manifest.get("artifacts")
    if type(artifacts) is not dict:
        raise ValueError("snapshot artifact manifest differs")
    root = manifest_path.parent
    source = load_section_graphs(root / str(artifacts["accumulated_section_graphs"]))
    partition = load_canonical_partition(
        root / str(artifacts["canonical_partition"]), source=source
    )
    graph = compile_experience_graph(source, partition)
    with IncrementalStateStore(args.state_db) as state:
        result = build_graph_quality_artifacts(
            snapshot_id=str(manifest["snapshot_id"]),
            source=source,
            partition=partition,
            graph=graph,
            canonical_audit=_canonical_json_file(
                root / str(artifacts["canonical_audit"])
            ),
            cumulative_source_audit=_canonical_json_file(
                root / str(artifacts["cumulative_source_audit"])
            ),
            state=state,
        )
    print(json.dumps(result.audit, ensure_ascii=False, sort_keys=True))
    return 0


__all__ = [
    "GRAPH_QUALITY_AUDIT_FORMAT",
    "GRAPH_QUALITY_PROTOCOL",
    "GRAPH_QUALITY_PROTOCOL_SHA256",
    "GraphQualityArtifacts",
    "build_graph_quality_artifacts",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
