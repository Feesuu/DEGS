from __future__ import annotations

import hashlib

from degs.core import canonical_json_bytes
from degs.section_graph import SECTION_GRAPH_FORMAT, SOURCE_SPLIT
from degs.source_rebuild import (
    ExperienceSourceBuild,
    fixed_incremental_source_batches,
)


def _workflow(train_index: int) -> dict:
    return {
        "train_index": train_index,
        "task_id": f"task-{train_index}",
        "query_text": f"query {train_index}",
        "experience_nodes": [
            {
                "operation": "inspect one target property",
                "applicability": ["a target property controls a later action"],
                "inputs": [{"type": "artifact", "description": "target"}],
                "outputs": [{"type": "evidence", "description": "property"}],
            }
        ],
        "edges": [],
    }


def test_global_source_build_splits_into_exact_fixed_25x8_batches() -> None:
    rows = [
        {
            "train_index": index,
            "source_review_status": "REVIEW_ACCEPTED",
            "origin": "ORIGINAL_SUCCESS",
            "discarded_edge_reasons": [],
        }
        for index in (0, 9)
    ]
    exclusions = [
        {
            "train_index": index,
            "status": "SOURCE_EXCLUDED_NO_VALIDATED_SUCCESS",
        }
        for index in range(200)
        if index not in {0, 9}
    ]
    section_graphs = {
        "format": SECTION_GRAPH_FORMAT,
        "source_split": SOURCE_SPLIT,
        "workflows": [_workflow(0), _workflow(9)],
    }
    build = ExperienceSourceBuild(section_graphs, tuple(rows), tuple(exclusions))
    full_audit = {
        "format": "test",
        "source_split": SOURCE_SPLIT,
        "section_graphs_sha256": "full",
        "source_workflow_count": 2,
        "source_extraction_workers": 16,
        "source_review_workers": 16,
        "incremental_batch_size": None,
        "batch_train_indices": None,
        "source_extraction_semantic_attempt_limit": 3,
        "source_review_semantic_attempt_limit": 3,
        "producer_transport_failure_policy": {},
        "review_retry_mode": False,
        "review_retry_queue_count": 0,
        "review_retry_queue": [],
        "review_status_counts": {"REVIEW_ACCEPTED": 2},
        "origin_counts": {"ORIGINAL_SUCCESS": 2},
        "discarded_edge_count": 0,
        "excluded_source_workflow_count": 198,
        "exclusion_counts": {"SOURCE_EXCLUDED_NO_VALIDATED_SUCCESS": 198},
        "exclusions": exclusions,
        "rows": rows,
    }

    batches = fixed_incremental_source_batches(build, full_audit)

    assert len(batches) == 25
    all_indices = []
    for number, (source, audit) in enumerate(batches):
        expected = list(range(number * 8, number * 8 + 8))
        all_indices.extend(audit["batch_train_indices"])
        assert audit["batch_train_indices"] == expected
        assert audit["incremental_batch_size"] == 8
        assert audit["source_workflow_count"] == len(source["workflows"])
        assert audit["section_graphs_sha256"] == hashlib.sha256(
            canonical_json_bytes(source)
        ).hexdigest()
        assert len(audit["rows"]) + len(audit["exclusions"]) == 8
    assert all_indices == list(range(200))
    assert [row["train_index"] for row in batches[0][0]["workflows"]] == [0]
    assert [row["train_index"] for row in batches[1][0]["workflows"]] == [9]
