from __future__ import annotations

import hashlib
import json
from pathlib import Path

from degs.core import canonical_json_bytes
from degs.section_graph import ExperienceEdge, ExperienceNode, IOContract
from degs.source_rebuild import _load_saved_review_response_attempts
from degs.source_review import (
    ReviewDecisionKind,
    SOURCE_REVIEW_KIND,
    SOURCE_REVIEW_PROMPT_SHA256,
    parse_source_review_response,
    source_review_payload,
    source_review_response_schema,
)
from degs.validated_repair import SOURCE_RAW_RESPONSE_FORMAT


def _node(operation: str) -> ExperienceNode:
    contract = IOContract("artifact", "task state")
    return ExperienceNode(
        operation,
        ("When the observed task state requires this transformation.",),
        (contract,),
        (contract,),
    )


def test_source_review_keeps_final_graph_schema_and_audits_draft_disposition() -> None:
    response = {
        "experience_nodes": [_node("Apply the task-specific transformation").to_dict()],
        "edges": [],
        "review_decisions": [
            {
                "draft_node": 0,
                "decision": "FOLD",
                "final_nodes": [0],
                "basis": "The routine persistence step belongs to the substantive output contract.",
            },
            {
                "draft_node": 1,
                "decision": "REWRITE",
                "final_nodes": [0],
                "basis": "The retained node expresses the evidence-supported operation.",
            },
        ],
    }
    nodes, edges, discarded, decisions, errors = parse_source_review_response(
        response,
        draft_count=2,
    )
    assert len(nodes) == 1
    assert edges == ()
    assert discarded == ()
    assert errors == ()
    assert decisions[0].decision is ReviewDecisionKind.FOLD
    assert set(source_review_response_schema()["required"]) == {
        "experience_nodes",
        "edges",
        "review_decisions",
    }


def test_bad_review_ledger_does_not_discard_valid_nodes_or_edges() -> None:
    response = {
        "experience_nodes": [_node("Observe a controlling condition").to_dict(), _node("Act on it").to_dict()],
        "edges": [{"source": 0, "target": 1}, {"source": 0, "target": 99}],
        "review_decisions": [
            {
                "draft_node": 99,
                "decision": "KEEP",
                "final_nodes": [0],
                "basis": "Bad index must remain an audit-only defect.",
            }
        ],
    }
    nodes, edges, discarded, decisions, errors = parse_source_review_response(
        response,
        draft_count=1,
    )
    assert len(nodes) == 2
    assert edges == (ExperienceEdge(0, 1),)
    assert discarded
    assert decisions == ()
    assert errors


def test_keep_must_map_exactly_one_node_without_discarding_final_graph() -> None:
    response = {
        "experience_nodes": [_node("Observe").to_dict(), _node("Act").to_dict()],
        "edges": [{"source": 0, "target": 1}],
        "review_decisions": [
            {
                "draft_node": 0,
                "decision": "KEEP",
                "final_nodes": [0, 1],
                "basis": "This invalid ledger must not erase valid experience.",
            }
        ],
    }
    nodes, edges, discarded, decisions, errors = parse_source_review_response(
        response, draft_count=1
    )
    assert len(nodes) == 2
    assert edges == (ExperienceEdge(0, 1),)
    assert discarded == ()
    assert decisions == ()
    assert errors == (
        "review_decisions[0] mapping differs",
        "draft node 0 has no valid disposition",
    )


def test_complete_review_ledger_contracts_unmapped_final_nodes() -> None:
    response = {
        "experience_nodes": [
            _node("Identify matching rows").to_dict(),
            _node("Save the modified artifact").to_dict(),
            _node("Check that no matching row remains").to_dict(),
        ],
        "edges": [
            {"source": 0, "target": 1},
            {"source": 1, "target": 2},
        ],
        "review_decisions": [
            {
                "draft_node": 0,
                "decision": "KEEP",
                "final_nodes": [0],
                "basis": "The matching operation is substantive.",
            },
            {
                "draft_node": 1,
                "decision": "REMOVE",
                "final_nodes": [],
                "basis": "Ordinary persistence is execution mechanics.",
            },
            {
                "draft_node": 2,
                "decision": "KEEP",
                "final_nodes": [2],
                "basis": "The task-specific invariant is substantive.",
            },
        ],
    }
    nodes, edges, discarded, decisions, errors = parse_source_review_response(
        response, draft_count=3
    )
    assert [node.operation for node in nodes] == [
        "Identify matching rows",
        "Check that no matching row remains",
    ]
    assert edges == (ExperienceEdge(0, 1),)
    assert discarded == ()
    assert decisions[2].final_nodes == (1,)
    assert errors == (
        "final node 1 had no review disposition and was contracted out",
    )


def test_source_review_payload_preserves_complete_evidence_and_draft() -> None:
    evidence = {"successful_trajectory": {"large_observation": "x" * 20_000}}
    draft = (_node("Transform the artifact"),)
    payload = source_review_payload(
        evidence_mode="ORIGINAL_SUCCESS",
        evidence=evidence,
        draft_nodes=draft,
        draft_edges=(),
    )
    assert payload["evidence"] == evidence
    assert len(payload["evidence"]["successful_trajectory"]["large_observation"]) == 20_000
    assert payload["draft_graph"]["experience_nodes"] == [draft[0].to_dict()]


def test_saved_complete_review_response_is_reusable_after_process_exit(
    tmp_path: Path,
) -> None:
    response = {
        "experience_nodes": [_node("Apply the supported transformation").to_dict()],
        "edges": [],
        "review_decisions": [
            {
                "draft_node": 0,
                "decision": "KEEP",
                "final_nodes": [0],
                "basis": "The operation is both causally supported and reusable.",
            }
        ],
    }
    protocol = {
        "format": "degs_experience_source_review_protocol_v1",
        "request_kind": SOURCE_REVIEW_KIND,
        "prompt_sha256": SOURCE_REVIEW_PROMPT_SHA256,
    }
    protocol_sha256 = hashlib.sha256(
        canonical_json_bytes(protocol)
    ).hexdigest()
    raw = {
        "format": SOURCE_RAW_RESPONSE_FORMAT,
        "outcome": "COMPLETE",
        "request_kind": SOURCE_REVIEW_KIND,
        "request_id": "source-review-trajectory-1",
        "system_prompt_sha256": SOURCE_REVIEW_PROMPT_SHA256,
        "payload_sha256": "a" * 64,
        "source_protocol": protocol,
        "source_protocol_sha256": protocol_sha256,
        "response": json.dumps(response),
    }
    (tmp_path / "raw_review_response_attempt_01.json").write_text(
        json.dumps(raw), encoding="utf-8"
    )

    invalid, reviewed = _load_saved_review_response_attempts(
        tmp_path,
        expected_protocol=protocol,
        expected_request_id="source-review-trajectory-1",
        expected_payload_sha256="a" * 64,
        draft_count=1,
    )

    assert invalid == ()
    assert reviewed is not None
    assert reviewed.graph_dict() == {
        "experience_nodes": response["experience_nodes"],
        "edges": [],
    }
    assert reviewed.request_payload_sha256 == "a" * 64
