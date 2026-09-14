from __future__ import annotations

import argparse
import asyncio
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable, Mapping, Sequence, cast

from openai import APIError
from react_agent.models import (
    OpenAIClient,
    RequestCompletionLengthExceeded,
    RequestContextLengthExceeded,
    RequestRuntimeTimeout,
)
from sb_adapter.transport import validate_service_url

from .canonicalize import (
    CANONICAL_CANDIDATE_K, CANONICAL_CANDIDATE_POLICY,
    CANONICAL_MERGE_KIND, CANONICAL_MERGE_PROMPT_SHA256, CANONICAL_MERGE_PROTOCOL_FORMAT, CANONICAL_MERGE_SYSTEM_PROMPT,
    CANONICAL_LLM_WORKERS, CANONICAL_PARTITION_POLICY, CANONICAL_RECALL_AUDIT_K, CANONICAL_SEMANTIC_ATTEMPTS,
    CANONICAL_VIEW_KIND, CANONICAL_VIEW_PROMPT_SHA256, CANONICAL_VIEW_PROTOCOL_FORMAT, CANONICAL_VIEW_SYSTEM_PROMPT,
    CanonicalizationView, TemplateRelation, _run_canonical_jobs,
    canonical_merge_response_schema, canonical_structured_array_policy, canonical_unit_candidates,
    canonicalization_view_embedding_text, canonicalization_view_response_schema,
    openai_canonical_merge_llm, openai_canonical_view_llm, parse_canonicalization_view, parse_canonical_merge,
)
from .core import (
    EMBEDDING_ASYNC_WORKERS,
    EMBEDDING_MODEL,
    StrictEmbeddingAdapter,
    canonical_json_bytes,
    normalize_embedding_text,
)
from .graph_quality import (
    GRAPH_QUALITY_PROTOCOL_SHA256,
    GraphQualityArtifacts,
    build_graph_quality_artifacts,
)
from .graph_dataset_contract import (
    GraphDatasetContract,
    SPREADSHEETBENCH_GRAPH_CONTRACT,
)
from .section_graph import (
    CANONICAL_PARTITION_FORMAT,
    SECTION_GRAPH_FORMAT,
    SOURCE_SPLIT,
    CanonicalExperience,
    CanonicalGroup,
    CanonicalPartition,
    ExperienceGraph,
    ExperienceNode,
    SectionGraphSource,
    _canonical_document,
    _canonical_id,
    compile_experience_graph,
    experience_leaf_id,
    load_canonical_partition,
    load_section_graphs,
)
from .state_store import (
    EMBEDDING_CACHE_NAMESPACE,
    INCREMENTAL_METHOD_ID,
    IncrementalStateStore,
)
from .source_rebuild import (
    SOURCE_EXTRACTION_SEMANTIC_ATTEMPTS,
    SOURCE_REBUILD_AUDIT_FORMAT,
    SOURCE_REBUILD_WORKERS,
    SOURCE_REVIEW_RETRY_STATUS,
)
from .source_review import (
    SOURCE_REVIEW_KIND,
    SOURCE_REVIEW_PROMPT_SHA256,
    SOURCE_REVIEW_PROTOCOL_FORMAT,
    source_review_response_schema,
)
from .source_replay import (
    SOURCE_REPLAY_OUTCOME_FORMAT,
    validate_source_replay_outcome_protocol,
)
from .successful_source import (
    SUCCESS_EXTRACTION_KIND,
    SUCCESS_PROMPT_SHA256,
    SUCCESS_SOURCE_PROTOCOL_FORMAT,
)
from .transport import QwenEmbeddingHTTPTransport
from .validated_repair import (
    JsonObjectLLM,
    REPAIR_EXTRACTION_KIND,
    REPAIR_PROMPT_SHA256,
    REPAIR_SOURCE_MAX_TOKENS,
    REPAIR_SOURCE_MODEL,
    REPAIR_SOURCE_PROTOCOL_FORMAT,
    REPAIR_SOURCE_TEMPERATURE,
    REPAIR_SOURCE_THINKING,
    REPAIR_SOURCE_TIMEOUT_SECONDS,
    PRODUCER_RUNTIME_TIMEOUT_RETRIES,
    PRODUCER_TRANSPORT_RETRY_WAITS,
    ProducerTransportGuard,
    SystemicProducerTransportFailure,
    producer_transport_failure_policy,
    _source_generation_config,
    _write_json_output,
    repair_response_schema,
)


INCREMENTAL_BATCH_SIZE = 8
INCREMENTAL_SNAPSHOT_FORMAT = "degs_incremental_experience_graph_snapshot_v7"
INCREMENTAL_CANONICAL_PROTOCOL_FORMAT = (
    "degs_monotonic_operation_canonical_protocol_v1"
)
CANONICAL_CONTEXT_LENGTH_POLICY = (
    "context_overflow_item_local_no_merge_no_repeat_v3"
)
INCREMENTAL_SOURCE_AUDIT_LEDGER_FORMAT = (
    "degs_incremental_source_audit_ledger_v2"
)
_SOURCE_EXCLUSION_STATUSES = frozenset(
    {
        "SOURCE_EXCLUDED_NO_VALIDATED_SUCCESS",
        "SOURCE_EXCLUDED_CONTEXT_LENGTH",
        "SOURCE_EXCLUDED_GENERATION_FAILURE",
        "SOURCE_EXCLUDED_NO_REUSABLE_EXPERIENCE",
        "SOURCE_EXCLUDED_NO_STEP",
        "SOURCE_EXCLUDED_EMPTY_PUBLIC_QUESTION",
        "SOURCE_EXCLUDED_REVIEW_FAILURE",
    }
)
_SOURCE_AUDIT_FIELDS = {
    "format", "source_split", "section_graphs_sha256", "source_workflow_count",
    "source_extraction_workers", "incremental_batch_size", "batch_train_indices",
    "source_extraction_semantic_attempt_limit", "source_review_workers",
    "source_review_semantic_attempt_limit", "review_status_counts",
    "producer_transport_failure_policy",
    "review_retry_mode", "review_retry_queue_count", "review_retry_queue",
    "origin_counts", "discarded_edge_count", "excluded_source_workflow_count",
    "exclusion_counts", "exclusions", "rows",
}
_SOURCE_LLM_PROTOCOL_FIELDS = {
    "format", "request_kind", "model", "temperature", "thinking", "max_tokens",
    "timeout_seconds", "generation_config", "retry_waits_seconds",
    "runtime_timeout_retries", "prompt_sha256", "service_url",
}


def _validate_source_llm_protocol(
    value: Any,
    *,
    origin: str,
    expected_service_url: str | None = None,
) -> None:
    if origin == "ORIGINAL_SUCCESS":
        expected_format = SUCCESS_SOURCE_PROTOCOL_FORMAT
        expected_kind = SUCCESS_EXTRACTION_KIND
        expected_prompt = SUCCESS_PROMPT_SHA256
    elif origin == "REPLAY_VALIDATED_SUCCESS":
        expected_format = REPAIR_SOURCE_PROTOCOL_FORMAT
        expected_kind = REPAIR_EXTRACTION_KIND
        expected_prompt = REPAIR_PROMPT_SHA256
    else:
        raise ValueError("source extraction origin differs")
    if type(value) is not dict or set(value) != _SOURCE_LLM_PROTOCOL_FIELDS:
        raise ValueError("source extraction LLM protocol fields differ")
    service_url = value.get("service_url")
    expected = {
        "format": expected_format,
        "request_kind": expected_kind,
        "model": REPAIR_SOURCE_MODEL,
        "temperature": REPAIR_SOURCE_TEMPERATURE,
        "thinking": REPAIR_SOURCE_THINKING,
        "max_tokens": REPAIR_SOURCE_MAX_TOKENS,
        "timeout_seconds": REPAIR_SOURCE_TIMEOUT_SECONDS,
        "generation_config": _source_generation_config(),
        "retry_waits_seconds": list(PRODUCER_TRANSPORT_RETRY_WAITS),
        "runtime_timeout_retries": PRODUCER_RUNTIME_TIMEOUT_RETRIES,
        "prompt_sha256": expected_prompt,
        "service_url": service_url,
    }
    if (
        value != expected
        or type(service_url) is not str
        or (
            expected_service_url is not None
            and service_url.rstrip("/") != expected_service_url.rstrip("/")
        )
    ):
        raise ValueError("source extraction LLM protocol differs")
    validate_service_url(service_url)


def _validate_source_review_llm_protocol(
    value: Any,
    *,
    expected_service_url: str | None = None,
) -> None:
    if type(value) is not dict or set(value) != _SOURCE_LLM_PROTOCOL_FIELDS:
        raise ValueError("source review LLM protocol fields differ")
    service_url = value.get("service_url")
    expected = {
        "format": SOURCE_REVIEW_PROTOCOL_FORMAT,
        "request_kind": SOURCE_REVIEW_KIND,
        "model": REPAIR_SOURCE_MODEL,
        "temperature": REPAIR_SOURCE_TEMPERATURE,
        "thinking": REPAIR_SOURCE_THINKING,
        "max_tokens": REPAIR_SOURCE_MAX_TOKENS,
        "timeout_seconds": REPAIR_SOURCE_TIMEOUT_SECONDS,
        "generation_config": _source_generation_config(),
        "retry_waits_seconds": list(PRODUCER_TRANSPORT_RETRY_WAITS),
        "runtime_timeout_retries": PRODUCER_RUNTIME_TIMEOUT_RETRIES,
        "prompt_sha256": SOURCE_REVIEW_PROMPT_SHA256,
        "service_url": service_url,
    }
    if (
        value != expected
        or type(service_url) is not str
        or (
            expected_service_url is not None
            and service_url.rstrip("/") != expected_service_url.rstrip("/")
        )
    ):
        raise ValueError("source review LLM protocol differs")
    validate_service_url(service_url)


@dataclass(frozen=True)
class IncrementalCanonicalBuild:
    partition: CanonicalPartition
    partition_payload: dict[str, Any]
    audit: dict[str, Any]
    changed_canonical_ids: frozenset[str]
    retired_canonical_ids: frozenset[str]
    requested_embedding_texts: tuple[str, ...]
    view_state_rows: tuple[dict[str, Any], ...] = ()
    neighbor_state_rows: tuple[dict[str, Any], ...] = ()
    merge_event_rows: tuple[dict[str, Any], ...] = ()
    created_groups: tuple[CanonicalGroup, ...] = ()


@dataclass(frozen=True)
class IncrementalSnapshotBuild:
    snapshot_id: str
    output_dir: Path
    manifest: Mapping[str, Any]


class _CanonicalSemanticAttemptsExhausted(Exception):
    def __init__(
        self,
        *,
        stage: str,
        request_id: str,
        request_sha256: str,
        request_payload_sha256: str,
        attempt_count: int,
        last_error_type: str,
        last_error_message: str,
    ) -> None:
        super().__init__(
            f"Canonical {stage} producer exhausted semantic attempts; "
            f"request_id={request_id}; request_sha256={request_sha256}"
        )
        self.stage = stage
        self.request_id = request_id
        self.request_sha256 = request_sha256
        self.request_payload_sha256 = request_payload_sha256
        self.attempt_count = attempt_count
        self.last_error_type = last_error_type
        self.last_error_message = last_error_message


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _validate_batch_source_audit(
    *,
    batch_source: SectionGraphSource,
    batch_train_indices: Sequence[int],
    audit: Mapping[str, Any],
    expected_generation_endpoint: str | None = None,
) -> tuple[str, dict[int, str]]:
    if type(audit) is not dict:
        raise ValueError("incremental batch source audit must be an object")
    indices = tuple(batch_train_indices)
    rows = audit.get("rows")
    exclusions = audit.get("exclusions")
    review_retry_queue = audit.get("review_retry_queue")
    if (
        set(audit) != _SOURCE_AUDIT_FIELDS
        or audit.get("format") != SOURCE_REBUILD_AUDIT_FORMAT
        or audit.get("source_split") != SOURCE_SPLIT
        or audit.get("section_graphs_sha256") != batch_source.sha256
        or audit.get("source_workflow_count") != len(batch_source.workflows)
        or audit.get("source_extraction_workers") != SOURCE_REBUILD_WORKERS
        or audit.get("source_review_workers") != SOURCE_REBUILD_WORKERS
        or audit.get("incremental_batch_size") != INCREMENTAL_BATCH_SIZE
        or audit.get("batch_train_indices") != list(indices)
        or audit.get("source_extraction_semantic_attempt_limit")
        != SOURCE_EXTRACTION_SEMANTIC_ATTEMPTS
        or audit.get("source_review_semantic_attempt_limit")
        != SOURCE_EXTRACTION_SEMANTIC_ATTEMPTS
        or audit.get("producer_transport_failure_policy")
        != producer_transport_failure_policy()
        or type(audit.get("review_retry_mode")) is not bool
        or type(review_retry_queue) is not list
        or audit.get("review_retry_queue_count") != len(review_retry_queue)
        or type(rows) is not list
        or type(exclusions) is not list
        or audit.get("excluded_source_workflow_count") != len(exclusions)
    ):
        raise ValueError("incremental batch source audit identity differs")
    workflow_by_index = batch_source.workflow_by_index
    status_by_index: dict[int, str] = {}
    origin_counts: Counter[str] = Counter()
    review_status_counts: Counter[str] = Counter()
    discarded_edge_count = 0
    response_schema_sha256 = _sha256_bytes(
        canonical_json_bytes(repair_response_schema())
    )
    for row in rows:
        if type(row) is not dict:
            raise ValueError("incremental batch source audit row differs")
        train_index = row.get("train_index")
        workflow = (
            workflow_by_index.get(train_index)
            if type(train_index) is int
            else None
        )
        origin = row.get("origin")
        common_fields = {
            "train_index", "task_id", "trajectory_id", "origin", "status",
            "prompt_sha256", "source_protocol", "source_protocol_sha256",
            "request_payload_sha256", "response_schema_sha256", "query_text",
            "query_text_sha256", "discarded_edge_reasons",
            "source_extraction_attempt_index", "invalid_response_attempts",
            "draft_graph", "draft_discarded_edge_reasons", "source_review_status",
            "source_review_prompt_sha256", "source_review_protocol",
            "source_review_protocol_sha256",
            "source_review_request_payload_sha256",
            "source_review_response_schema_sha256",
            "source_review_attempt_index",
            "source_review_invalid_response_attempts",
            "source_review_decisions", "source_review_ledger_status",
            "source_review_ledger_errors",
            "source_review_discarded_edge_reasons", "draft_graph_sha256",
            "reviewed_graph_sha256",
        }
        replay_fields = {
            "accepted_patch_id", "accepted_attempt_index", "accepted_patch_sha256",
            "successful_replay_sha256", "repair_memory_sha256",
            "source_replay_protocol", "source_replay_protocol_sha256",
            "replay_evidence_mode",
        }
        expected_fields = common_fields | (
            replay_fields if origin == "REPLAY_VALIDATED_SUCCESS" else set()
        )
        protocol = row.get("source_protocol")
        discarded = row.get("discarded_edge_reasons")
        invalid_attempts = row.get("invalid_response_attempts")
        review_status = row.get("source_review_status")
        review_protocol = row.get("source_review_protocol")
        review_invalid_attempts = row.get(
            "source_review_invalid_response_attempts"
        )
        review_attempt_index = row.get("source_review_attempt_index")
        reviewed_graph_sha256 = _sha256_bytes(
            canonical_json_bytes(
                {
                    "experience_nodes": [
                        node.to_dict() for node in workflow.experience_nodes
                    ] if workflow is not None else [],
                    "edges": [
                        edge.to_dict() for edge in workflow.edges
                    ] if workflow is not None else [],
                }
            )
        )
        draft_graph = row.get("draft_graph")
        if (
            set(row) != expected_fields
            or type(train_index) is not int
            or train_index in status_by_index
            or workflow is None
            or row.get("task_id") != workflow.task_id
            or row.get("query_text") != workflow.query_text
            or row.get("query_text_sha256")
            != _sha256_bytes(workflow.query_text.encode("utf-8"))
            or row.get("status") != "INGESTED"
            or origin not in {"ORIGINAL_SUCCESS", "REPLAY_VALIDATED_SUCCESS"}
            or type(row.get("trajectory_id")) is not str
            or not row["trajectory_id"]
            or row.get("prompt_sha256")
            != (
                SUCCESS_PROMPT_SHA256
                if origin == "ORIGINAL_SUCCESS"
                else REPAIR_PROMPT_SHA256
            )
            or type(protocol) is not dict
            or row.get("source_protocol_sha256")
            != _sha256_bytes(canonical_json_bytes(protocol))
            or row.get("response_schema_sha256") != response_schema_sha256
            or type(row.get("request_payload_sha256")) is not str
            or len(row["request_payload_sha256"]) != 64
            or type(discarded) is not list
            or any(type(reason) is not str or not reason for reason in discarded)
            or type(row.get("source_extraction_attempt_index")) is not int
            or not 1 <= row["source_extraction_attempt_index"] <= SOURCE_EXTRACTION_SEMANTIC_ATTEMPTS
            or type(invalid_attempts) is not list
            or len(invalid_attempts)
            != row["source_extraction_attempt_index"] - 1
            or any(type(reason) is not str or not reason for reason in invalid_attempts)
            or review_status
            not in {
                "REVIEW_ACCEPTED",
                "REVIEW_DRAFT_FALLBACK",
                SOURCE_REVIEW_RETRY_STATUS,
            }
            or row.get("source_review_prompt_sha256")
            != SOURCE_REVIEW_PROMPT_SHA256
            or type(review_protocol) is not dict
            or row.get("source_review_protocol_sha256")
            != _sha256_bytes(canonical_json_bytes(review_protocol))
            or row.get("source_review_response_schema_sha256")
            != _sha256_bytes(
                canonical_json_bytes(source_review_response_schema())
            )
            or any(
                type(row.get(field)) is not str
                or len(row[field]) != 64
                for field in (
                    "source_review_request_payload_sha256",
                    "draft_graph_sha256",
                    "reviewed_graph_sha256",
                )
            )
            or row.get("reviewed_graph_sha256") != reviewed_graph_sha256
            or type(review_attempt_index) is not int
            or not 1
            <= review_attempt_index
            <= 2 * SOURCE_EXTRACTION_SEMANTIC_ATTEMPTS
            or type(review_invalid_attempts) is not list
            or any(
                type(reason) is not str or not reason
                for reason in review_invalid_attempts
            )
            or (
                review_status == "REVIEW_ACCEPTED"
                and review_attempt_index != len(review_invalid_attempts) + 1
            )
            or (
                review_status
                in {"REVIEW_DRAFT_FALLBACK", SOURCE_REVIEW_RETRY_STATUS}
                and review_attempt_index != len(review_invalid_attempts)
            )
            or type(row.get("source_review_decisions")) is not list
            or row.get("source_review_ledger_status")
            not in {
                "REVIEW_LEDGER_COMPLETE",
                "REVIEW_LEDGER_INCOMPLETE",
                "REVIEW_LEDGER_UNAVAILABLE",
            }
            or (
                review_status == "REVIEW_ACCEPTED"
                and (
                    row["source_review_ledger_status"]
                    == "REVIEW_LEDGER_UNAVAILABLE"
                )
            )
            or (
                review_status
                in {"REVIEW_DRAFT_FALLBACK", SOURCE_REVIEW_RETRY_STATUS}
                and row["source_review_ledger_status"]
                != "REVIEW_LEDGER_UNAVAILABLE"
            )
            or type(row.get("source_review_ledger_errors")) is not list
            or any(
                type(reason) is not str or not reason
                for reason in row.get("source_review_ledger_errors", [])
            )
            or type(row.get("draft_discarded_edge_reasons")) is not list
            or type(draft_graph) is not dict
            or set(draft_graph) != {"experience_nodes", "edges"}
            or row.get("draft_graph_sha256")
            != _sha256_bytes(canonical_json_bytes(draft_graph))
            or row.get("source_review_discarded_edge_reasons") != discarded
            or (
                origin == "REPLAY_VALIDATED_SUCCESS"
                and (
                    type(row.get("accepted_patch_id")) is not str
                    or not row["accepted_patch_id"]
                    or type(row.get("accepted_attempt_index")) is not int
                    or not 1 <= row["accepted_attempt_index"] <= 3
                    or any(
                        type(row.get(field)) is not str or len(row[field]) != 64
                        for field in (
                            "accepted_patch_sha256",
                            "successful_replay_sha256",
                            "repair_memory_sha256",
                        )
                    )
                    or type(row.get("source_replay_protocol")) is not dict
                    or row.get("source_replay_protocol_sha256")
                    != _sha256_bytes(
                        canonical_json_bytes(row.get("source_replay_protocol"))
                    )
                    or row.get("replay_evidence_mode") != "CURRENT_NO_TRUNCATION"
                )
            )
        ):
            raise ValueError("incremental batch ingested source audit differs")
        _validate_source_llm_protocol(
            protocol,
            origin=str(origin),
            expected_service_url=expected_generation_endpoint,
        )
        _validate_source_review_llm_protocol(
            review_protocol,
            expected_service_url=expected_generation_endpoint,
        )
        if origin == "REPLAY_VALIDATED_SUCCESS":
            validate_source_replay_outcome_protocol(
                {
                    "format": SOURCE_REPLAY_OUTCOME_FORMAT,
                    "source_replay_protocol": row["source_replay_protocol"],
                    "source_replay_protocol_sha256": row[
                        "source_replay_protocol_sha256"
                    ],
                }
            )
        origin_counts[str(origin)] += 1
        review_status_counts[str(review_status)] += 1
        discarded_edge_count += len(discarded)
        status_by_index[train_index] = "INGESTED"
    for row in exclusions:
        if type(row) is not dict:
            raise ValueError("incremental batch exclusion audit differs")
        train_index = row.get("train_index")
        status = row.get("status")
        expected_fields = {
            "train_index", "task_id", "trajectory_id", "origin", "status"
        }
        if status == "SOURCE_EXCLUDED_NO_VALIDATED_SUCCESS":
            expected_fields.add("replay_terminal_status")
        elif status == "SOURCE_EXCLUDED_CONTEXT_LENGTH":
            expected_fields.add("error")
        elif status == "SOURCE_EXCLUDED_GENERATION_FAILURE":
            expected_fields.update({"semantic_attempt_count", "invalid_response_attempts"})
        if status in {
            "SOURCE_EXCLUDED_CONTEXT_LENGTH",
            "SOURCE_EXCLUDED_GENERATION_FAILURE",
            "SOURCE_EXCLUDED_NO_REUSABLE_EXPERIENCE",
        }:
            expected_fields.update(
                {
                    "prompt_sha256",
                    "source_protocol",
                    "source_protocol_sha256",
                    "request_payload_sha256",
                    "response_schema_sha256",
                }
            )
        if "source_review_status" in row:
            expected_fields.update(
                {
                    "source_review_status",
                    "source_review_prompt_sha256",
                    "source_review_protocol",
                    "source_review_protocol_sha256",
                    "source_review_request_payload_sha256",
                    "source_review_response_schema_sha256",
                    "source_review_attempt_index",
                    "source_review_invalid_response_attempts",
                    "source_review_decisions",
                    "source_review_ledger_status",
                    "source_review_ledger_errors",
                    "source_review_discarded_edge_reasons",
                    "draft_graph",
                    "draft_discarded_edge_reasons",
                    "draft_graph_sha256",
                    "reviewed_graph_sha256",
                }
            )
        if (
            set(row) != expected_fields
            or type(train_index) is not int
            or train_index in status_by_index
            or train_index in workflow_by_index
            or status not in _SOURCE_EXCLUSION_STATUSES
            or type(row.get("task_id")) is not str
            or not row["task_id"]
            or type(row.get("trajectory_id")) is not str
            or not row["trajectory_id"]
            or row.get("origin") not in {
                "ORIGINAL_FAILURE", "ORIGINAL_SUCCESS", "REPLAY_VALIDATED_SUCCESS"
            }
            or (
                status == "SOURCE_EXCLUDED_NO_VALIDATED_SUCCESS"
                and (
                    row.get("origin") != "ORIGINAL_FAILURE"
                    or row.get("replay_terminal_status")
                    not in {"REPLAY_EXHAUSTED", "REPLAY_RUNTIME_FAILURE"}
                )
            )
            or (
                status == "SOURCE_EXCLUDED_CONTEXT_LENGTH"
                and (type(row.get("error")) is not str or not row["error"])
            )
            or (
                status == "SOURCE_EXCLUDED_GENERATION_FAILURE"
                and (
                    row.get("semantic_attempt_count") != SOURCE_EXTRACTION_SEMANTIC_ATTEMPTS
                    or type(row.get("invalid_response_attempts")) is not list
                    or len(row["invalid_response_attempts"])
                    != SOURCE_EXTRACTION_SEMANTIC_ATTEMPTS
                    or any(
                        type(reason) is not str or not reason
                        for reason in row["invalid_response_attempts"]
                    )
                )
            )
        ):
            raise ValueError("incremental batch source exclusion differs")
        if status != "SOURCE_EXCLUDED_NO_VALIDATED_SUCCESS":
            protocol = row.get("source_protocol")
            if (
                type(protocol) is not dict
                or row.get("prompt_sha256") != protocol.get("prompt_sha256")
                or row.get("source_protocol_sha256")
                != _sha256_bytes(canonical_json_bytes(protocol))
                or type(row.get("request_payload_sha256")) is not str
                or len(row["request_payload_sha256"]) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in row["request_payload_sha256"]
                )
                or row.get("response_schema_sha256")
                != _sha256_bytes(canonical_json_bytes(repair_response_schema()))
            ):
                raise ValueError("incremental active source exclusion identity differs")
            _validate_source_llm_protocol(
                protocol,
                origin=str(row["origin"]),
                expected_service_url=expected_generation_endpoint,
            )
        if "source_review_status" in row:
            review_protocol = row.get("source_review_protocol")
            draft_graph = row.get("draft_graph")
            if (
                row.get("source_review_prompt_sha256")
                != SOURCE_REVIEW_PROMPT_SHA256
                or type(review_protocol) is not dict
                or row.get("source_review_protocol_sha256")
                != _sha256_bytes(canonical_json_bytes(review_protocol))
                or row.get("source_review_response_schema_sha256")
                != _sha256_bytes(
                    canonical_json_bytes(source_review_response_schema())
                )
                or row.get("reviewed_graph_sha256")
                != _sha256_bytes(
                    canonical_json_bytes(
                        {"experience_nodes": [], "edges": []}
                    )
                )
                or type(draft_graph) is not dict
                or set(draft_graph) != {"experience_nodes", "edges"}
                or row.get("draft_graph_sha256")
                != _sha256_bytes(canonical_json_bytes(draft_graph))
                or type(row.get("draft_discarded_edge_reasons")) is not list
                or row.get("source_review_status") != "REVIEW_ACCEPTED"
                or row.get("source_review_ledger_status")
                not in {
                    "REVIEW_LEDGER_COMPLETE",
                    "REVIEW_LEDGER_INCOMPLETE",
                }
                or type(row.get("source_review_decisions")) is not list
                or type(row.get("source_review_ledger_errors")) is not list
                or type(row.get("source_review_attempt_index")) is not int
                or not 1
                <= row["source_review_attempt_index"]
                <= 2 * SOURCE_EXTRACTION_SEMANTIC_ATTEMPTS
                or type(
                    row.get("source_review_invalid_response_attempts")
                )
                is not list
                or row["source_review_attempt_index"]
                != len(row["source_review_invalid_response_attempts"]) + 1
            ):
                raise ValueError("incremental source review exclusion differs")
            _validate_source_review_llm_protocol(
                review_protocol,
                expected_service_url=expected_generation_endpoint,
            )
        status_by_index[train_index] = str(status)
    exclusion_counts = Counter(row["status"] for row in exclusions)
    expected_retry_queue = [
        {
            "train_index": row["train_index"],
            "task_id": row["task_id"],
            "trajectory_id": row["trajectory_id"],
            "status": (
                "PENDING"
                if row["source_review_status"] == "REVIEW_DRAFT_FALLBACK"
                else "EXHAUSTED"
            ),
            "review_request_payload_sha256": row[
                "source_review_request_payload_sha256"
            ],
        }
        for row in rows
        if row.get("source_review_status")
        in {"REVIEW_DRAFT_FALLBACK", SOURCE_REVIEW_RETRY_STATUS}
    ]
    if (
        set(status_by_index) != set(indices)
        or audit.get("origin_counts") != dict(sorted(origin_counts.items()))
        or audit.get("discarded_edge_count") != discarded_edge_count
        or audit.get("review_status_counts")
        != dict(sorted(review_status_counts.items()))
        or audit.get("exclusion_counts") != dict(sorted(exclusion_counts.items()))
        or review_retry_queue != expected_retry_queue
        or (
            audit["review_retry_mode"]
            and any(row["status"] == "PENDING" for row in review_retry_queue)
        )
    ):
        raise ValueError("incremental batch source audit must cover all 8 indices")
    return _sha256_bytes(canonical_json_bytes(dict(audit))), status_by_index


def _source_payload(source: SectionGraphSource) -> dict[str, Any]:
    return {
        "format": SECTION_GRAPH_FORMAT,
        "source_split": source.source_split,
        "workflows": [
            {
                "train_index": workflow.train_index,
                "task_id": workflow.task_id,
                "query_text": workflow.query_text,
                "experience_nodes": [
                    node.to_dict() for node in workflow.experience_nodes
                ],
                "edges": [edge.to_dict() for edge in workflow.edges],
            }
            for workflow in source.workflows
        ],
    }


def merge_source_batch(
    previous: SectionGraphSource | None,
    batch: SectionGraphSource,
) -> SectionGraphSource:
    if type(batch) is not SectionGraphSource:
        raise TypeError("validated source batch is required")
    if previous is not None and previous.source_split != batch.source_split:
        raise ValueError("incremental source split differs")
    rows = {} if previous is None else dict(previous.workflow_by_index)
    task_ids = {row.task_id for row in rows.values()}
    for workflow in batch.workflows:
        if workflow.train_index in rows:
            if rows[workflow.train_index] != workflow:
                raise ValueError("incremental workflow conflicts with existing source")
            continue
        if workflow.task_id in task_ids:
            raise ValueError("incremental workflow task ID is repeated")
        rows[workflow.train_index] = workflow
        task_ids.add(workflow.task_id)
    workflows = tuple(rows[index] for index in sorted(rows))
    provisional = SectionGraphSource(
        workflows,
        "0" * 64,
        batch.source_split,
    )
    payload = _source_payload(provisional)
    return SectionGraphSource(
        workflows,
        _sha256_bytes(canonical_json_bytes(payload)),
        batch.source_split,
    )


def _validate_committing_snapshot_state(
    *,
    state: IncrementalStateStore,
    snapshot_id: str,
    parent_snapshot_id: str | None,
    source: SectionGraphSource,
    partition: CanonicalPartition,
    graph: ExperienceGraph,
    manifest: Mapping[str, Any],
    source_status_by_index: Mapping[int, str],
    expected_snapshot_status: str = "COMMITTED",
) -> None:
    """Compare the full live SQLite projection with pending snapshot artifacts."""

    expected_workflows: list[tuple[Any, ...]] = []
    expected_nodes: list[tuple[Any, ...]] = []
    expected_source_edges: list[tuple[Any, ...]] = []
    node_id_by_coordinate: dict[tuple[int, int], str] = {}
    for workflow in source.workflows:
        workflow_payload = {
            "train_index": workflow.train_index,
            "task_id": workflow.task_id,
            "query_text": workflow.query_text,
            "experience_nodes": [
                node.to_dict() for node in workflow.experience_nodes
            ],
            "edges": [edge.to_dict() for edge in workflow.edges],
        }
        workflow_bytes = canonical_json_bytes(workflow_payload)
        expected_workflows.append(
            (
                workflow.train_index,
                workflow.task_id,
                workflow.query_text,
                _sha256_bytes(workflow.query_text.encode("utf-8")),
                _sha256_bytes(workflow_bytes),
                workflow_bytes,
            )
        )
        for node_index, node in enumerate(workflow.experience_nodes):
            node_id = experience_leaf_id(
                workflow.train_index, node_index, node
            )
            node_id_by_coordinate[(workflow.train_index, node_index)] = node_id
            node_bytes = canonical_json_bytes(node.to_dict())
            expected_nodes.append(
                (
                    node_id,
                    workflow.train_index,
                    node_index,
                    _sha256_bytes(node_bytes),
                    node_bytes,
                )
            )
        expected_source_edges.extend(
            (
                workflow.train_index,
                node_id_by_coordinate[(workflow.train_index, edge.source)],
                node_id_by_coordinate[(workflow.train_index, edge.target)],
            )
            for edge in workflow.edges
        )

    actual_workflows = state.connection.execute(
        """
        SELECT train_index, task_id, query_text, query_text_sha256,
               workflow_sha256, workflow_json
        FROM workflows ORDER BY train_index
        """
    ).fetchall()
    actual_nodes = state.connection.execute(
        """
        SELECT node_id, train_index, node_index, node_sha256, node_json
        FROM experience_nodes ORDER BY train_index, node_index
        """
    ).fetchall()
    actual_source_edges = state.connection.execute(
        """
        SELECT train_index, source_node_id, target_node_id
        FROM source_edges ORDER BY train_index, source_node_id, target_node_id
        """
    ).fetchall()
    if (
        actual_workflows != sorted(expected_workflows)
        or actual_nodes != sorted(expected_nodes, key=lambda row: (row[1], row[2]))
        or actual_source_edges != sorted(expected_source_edges)
    ):
        raise ValueError("committing snapshot source state differs")

    expected_active_nodes: list[tuple[Any, ...]] = []
    expected_heads: list[tuple[str, str]] = []
    expected_members: list[tuple[str, str]] = []
    for group in partition.groups:
        canonical_id = _canonical_id(group.members)
        canonical_bytes = canonical_json_bytes(
            group.canonical_experience.to_dict()
        )
        document = _canonical_document(group.canonical_experience)
        expected_active_nodes.append(
            (
                canonical_id,
                _sha256_bytes(canonical_bytes),
                canonical_bytes,
                _sha256_bytes(document.encode("utf-8")),
                document,
                len(group.members),
            )
        )
        for coordinate in group.members:
            node_id = node_id_by_coordinate[coordinate]
            expected_heads.append((node_id, canonical_id))
            expected_members.append((canonical_id, node_id))
    active_ids = [row[0] for row in expected_active_nodes]
    actual_active_nodes = state.connection.execute(
        """
        SELECT canonical_id, canonical_sha256, canonical_json,
               document_sha256, document, member_count
        FROM canonical_nodes
        WHERE retired_snapshot_id IS NULL
        ORDER BY canonical_id
        """
    ).fetchall()
    actual_heads = state.connection.execute(
        "SELECT node_id, canonical_id FROM canonical_heads ORDER BY node_id"
    ).fetchall()
    actual_members = (
        state.connection.execute(
            f"""
            SELECT canonical_id, node_id FROM canonical_leaf_members
            WHERE canonical_id IN ({','.join('?' for _ in active_ids)})
            ORDER BY canonical_id, node_id
            """,
            tuple(active_ids),
        ).fetchall()
        if active_ids
        else []
    )
    incomplete_history = state.connection.execute(
        """SELECT n.canonical_id FROM canonical_nodes n
           LEFT JOIN canonical_leaf_members m USING(canonical_id)
           GROUP BY n.canonical_id HAVING COUNT(m.node_id) != n.member_count LIMIT 1"""
    ).fetchone()
    if (
        incomplete_history is not None
        or actual_active_nodes != sorted(expected_active_nodes)
        or actual_heads != sorted(expected_heads)
        or actual_members != sorted(expected_members)
    ):
        raise ValueError("committing snapshot Canonical membership differs")

    expected_projection = sorted(
        (
            train_index,
            edge.source,
            edge.target,
        )
        for edge in graph.edges
        for train_index in edge.supporting_workflow_ids
    )
    actual_projection = state.connection.execute(
        """
        SELECT train_index, source_canonical_id, target_canonical_id
        FROM source_edge_projection
        ORDER BY train_index, source_canonical_id, target_canonical_id
        """
    ).fetchall()
    if actual_projection != expected_projection:
        raise ValueError("committing snapshot edge projection differs")

    snapshot_row = state.connection.execute(
        """
        SELECT parent_snapshot_id, batch_source_sha256,
               batch_source_audit_sha256, status, manifest_sha256,
               source_workflow_count, canonical_count,
               experience_node_count, experience_edge_count
        FROM snapshots WHERE snapshot_id = ?
        """,
        (snapshot_id,),
    ).fetchone()
    if expected_snapshot_status not in {"BUILDING", "COMMITTED"}:
        raise ValueError("snapshot agreement status differs")
    expected_snapshot_row = (
        parent_snapshot_id,
        manifest["batch_source_sha256"],
        manifest["batch_source_audit_sha256"],
        expected_snapshot_status,
        manifest["self_sha256"],
        len(source.workflows),
        len(partition.groups),
        len(graph.nodes),
        len(graph.edges),
    )
    actual_batch_items = state.connection.execute(
        """
        SELECT train_index, status FROM snapshot_batch_items
        WHERE snapshot_id = ? ORDER BY train_index
        """,
        (snapshot_id,),
    ).fetchall()
    if (
        snapshot_row != expected_snapshot_row
        or actual_batch_items != sorted(source_status_by_index.items())
    ):
        raise ValueError("committing snapshot manifest state differs")


def _partition_from_groups(
    source: SectionGraphSource,
    groups: Sequence[CanonicalGroup],
) -> tuple[CanonicalPartition, dict[str, Any]]:
    ordered = tuple(sorted(groups, key=lambda group: group.members[0]))
    payload = {
        "format": CANONICAL_PARTITION_FORMAT,
        "section_graphs_sha256": source.sha256,
        "groups": [
            {
                "members": [list(member) for member in group.members],
                "canonical_experience": group.canonical_experience.to_dict(),
            }
            for group in ordered
        ],
    }
    partition_sha = _sha256_bytes(canonical_json_bytes(payload))
    return CanonicalPartition(ordered, source.sha256, partition_sha), payload


def _source_occurrences(source: SectionGraphSource) -> dict[str, tuple[tuple[int, int], ExperienceNode]]:
    return {
        experience_leaf_id(workflow.train_index, i, node): ((workflow.train_index, i), node)
        for workflow in source.workflows for i, node in enumerate(workflow.experience_nodes)
    }


def _canonical_view_payload(node: ExperienceNode) -> dict[str, Any]:
    return {"experience_node": node.to_dict()}


def _canonical_merge_payload(left: CanonicalExperience, right: CanonicalExperience) -> dict[str, Any]:
    ordered = sorted((left.to_dict(), right.to_dict()), key=canonical_json_bytes)
    return {"left": ordered[0], "right": ordered[1]}


def _validate_monotonic_partition(previous: CanonicalPartition, current: CanonicalPartition) -> None:
    heads = {member: group for group in current.groups for member in group.members}
    for group in previous.groups:
        destinations = {_canonical_id(heads[member].members) for member in group.members}
        if len(destinations) != 1:
            raise ValueError("committed Canonical group was split")
        destination = heads[group.members[0]]
        if group.members == destination.members and group.canonical_experience != destination.canonical_experience:
            raise ValueError("unchanged Canonical experience was overwritten")


def canonical_protocol(
    view_protocol: Mapping[str, Any],
    merge_protocol: Mapping[str, Any],
    dataset_contract: GraphDatasetContract = SPREADSHEETBENCH_GRAPH_CONTRACT,
) -> dict[str, Any]:
    protocol = {
        "format": INCREMENTAL_CANONICAL_PROTOCOL_FORMAT,
        "method": INCREMENTAL_METHOD_ID, "batch_size": dataset_contract.batch_size,
        "incremental_unit": "indivisible_canonical_plus_new_source_singletons",
        "canonicalization_view": {
            "format": CANONICAL_VIEW_PROTOCOL_FORMAT,
            "prompt_sha256": CANONICAL_VIEW_PROMPT_SHA256,
            "response_schema_sha256": _sha256_bytes(canonical_json_bytes(canonicalization_view_response_schema())),
            "llm_protocol": dict(view_protocol),
        },
        "merge": {
            "format": CANONICAL_MERGE_PROTOCOL_FORMAT,
            "prompt_sha256": CANONICAL_MERGE_PROMPT_SHA256,
            "response_schema_sha256": _sha256_bytes(canonical_json_bytes(canonical_merge_response_schema())),
            "llm_protocol": dict(merge_protocol),
        },
        "candidate_graph": {
            "policy": CANONICAL_CANDIDATE_POLICY, "embedding_model": EMBEDDING_MODEL,
            "embedding_document": "node_only_view_aliases", "distinct_canonical_top_k": CANONICAL_CANDIDATE_K,
            "exact_normalized_view_always_included": True, "similarity_is_merge_threshold": False,
            "recall_audit_k": CANONICAL_RECALL_AUDIT_K,
        },
        "partition": {
            "policy": CANONICAL_PARTITION_POLICY, "previous_canonical_atomic": True,
            "node_only_identity": True, "old_group_split_allowed": False,
            "schedule": "max_alias_similarity_then_canonical_id;async_non_overlapping_rounds",
        },
        "structured_array_policy": canonical_structured_array_policy(),
        "semantic_attempt_limit": CANONICAL_SEMANTIC_ATTEMPTS,
        "producer_transport_failure_policy": producer_transport_failure_policy(),
        "context_length_policy": CANONICAL_CONTEXT_LENGTH_POLICY,
        "llm_workers": CANONICAL_LLM_WORKERS, "input_truncation": False, "ontology": None,
    }
    if dataset_contract != SPREADSHEETBENCH_GRAPH_CONTRACT:
        protocol["dataset_contract"] = dataset_contract.to_dict()
    return protocol


async def build_incremental_canonical_partition(
    *, source: SectionGraphSource, batch: SectionGraphSource,
    previous_partition: CanonicalPartition, view_llm: JsonObjectLLM,
    merge_llm: JsonObjectLLM, embedder: StrictEmbeddingAdapter,
    state: IncrementalStateStore, snapshot_id: str, refresh: bool = False,
    published_audit: Mapping[str, Any] | None = None,
    dataset_contract: GraphDatasetContract = SPREADSHEETBENCH_GRAPH_CONTRACT,
) -> IncrementalCanonicalBuild:
    protocol = canonical_protocol(
        view_llm.protocol_identity,
        merge_llm.protocol_identity,
        dataset_contract,
    )
    protocol_sha = _sha256_bytes(canonical_json_bytes(protocol))
    llm_semaphore = asyncio.Semaphore(CANONICAL_LLM_WORKERS)
    transport_guards = {
        stage: ProducerTransportGuard(stage=stage)
        for stage in ("VIEW", "MERGE")
    }
    pending_transport_attempts: dict[str, list[dict[str, Any]]] = {
        stage: []
        for stage in ("VIEW", "MERGE")
    }

    def persist_transport_wave(
        *,
        stage: str,
        status: str,
        attempts: Sequence[Mapping[str, Any]],
    ) -> None:
        if not attempts:
            return
        state.put_canonical_transport_wave(
            stage=stage,
            status=status,
            policy=producer_transport_failure_policy(),
            failed_request_ids=[str(row["request_id"]) for row in attempts],
            attempts=attempts,
            created_snapshot_id=snapshot_id,
        )

    async def run_stage_wave(
        stages: Sequence[str],
        jobs: Sequence[Any],
        worker: Callable[[Any], Any],
    ) -> tuple[Any, ...]:
        waves = {
            stage: await transport_guards[stage].begin_wave()
            for stage in stages
        }
        attempt_offsets = {
            stage: len(pending_transport_attempts[stage])
            for stage in stages
        }
        try:
            results = await _run_canonical_jobs(jobs, worker)
        except SystemicProducerTransportFailure as exc:
            for stage in stages:
                attempts = pending_transport_attempts[stage][
                    attempt_offsets[stage] :
                ]
                persist_transport_wave(
                    stage=stage,
                    status=(
                        "SYSTEMIC"
                        if stage == exc.stage
                        else "ABORTED_BY_SYSTEMIC"
                    ),
                    attempts=attempts,
                )
            raise
        for stage in stages:
            attempts = pending_transport_attempts[stage][
                attempt_offsets[stage] :
            ]
            try:
                await transport_guards[stage].raise_if_systemic(waves[stage])
            except SystemicProducerTransportFailure:
                persist_transport_wave(
                    stage=stage,
                    status="SYSTEMIC",
                    attempts=attempts,
                )
                raise
            persist_transport_wave(
                stage=stage,
                status="ITEM_LOCAL",
                attempts=attempts,
            )
        return results

    async def run_canonical_request(
        *,
        stage: str,
        llm: JsonObjectLLM,
        kind: str,
        prompt_sha256: str,
        system_prompt: str,
        request_id: str,
        payload: Mapping[str, Any],
        response_schema: Mapping[str, Any],
        parser: Callable[[Mapping[str, Any]], tuple[Any, dict[str, Any]]],
    ) -> tuple[Any, dict[str, Any]]:
        payload_sha256 = _sha256_bytes(canonical_json_bytes(dict(payload)))
        request_sha256 = _sha256_bytes(
            canonical_json_bytes(
                {
                    "stage": stage,
                    "kind": kind,
                    "request_id": request_id,
                    "prompt_sha256": prompt_sha256,
                    "protocol_sha256": protocol_sha,
                    "payload": dict(payload),
                }
            )
        )
        cached = state.get_canonical_job(
            request_sha256,
            stage=stage,
            prompt_sha256=prompt_sha256,
            producer_protocol_sha256=protocol_sha,
        )
        if cached is not None:
            response, stored_audit = cached
            parsed, base_audit = parser(response)
            expected = {
                **base_audit,
                "request_sha256": request_sha256,
                "request_payload_sha256": payload_sha256,
                "semantic_attempt_index": stored_audit.get(
                    "semantic_attempt_index"
                ),
            }
            if (
                stored_audit != expected
                or type(stored_audit.get("semantic_attempt_index")) is not int
                or not 1
                <= stored_audit["semantic_attempt_index"]
                <= CANONICAL_SEMANTIC_ATTEMPTS
            ):
                raise ValueError("Canonical cache audit differs")
            resumed = dict(stored_audit)
            resumed["status"] = f"{resumed['status']}_RESUMED"
            return parsed, resumed

        prior_failures = state.canonical_attempt_failures(
            request_sha256=request_sha256,
            stage=stage,
            created_snapshot_id=snapshot_id,
        )
        if len(prior_failures) > CANONICAL_SEMANTIC_ATTEMPTS:
            raise ValueError("Canonical attempt history exceeds protocol limit")
        last_semantic_error: Exception | None = None
        last_error_type = prior_failures[-1][1] if prior_failures else ""
        last_error_message = prior_failures[-1][2] if prior_failures else ""
        if last_error_type == "RequestContextLengthExceeded":
            raise _CanonicalSemanticAttemptsExhausted(
                stage=stage,
                request_id=request_id,
                request_sha256=request_sha256,
                request_payload_sha256=payload_sha256,
                attempt_count=len(prior_failures),
                last_error_type=last_error_type,
                last_error_message=last_error_message,
            )
        semantic_attempt = len(prior_failures) + 1
        transport_attempt = 0
        while (
            semantic_attempt <= CANONICAL_SEMANTIC_ATTEMPTS
            and transport_attempt < CANONICAL_SEMANTIC_ATTEMPTS
        ):
            response: Mapping[str, Any] | None = None
            try:
                async with llm_semaphore:
                    response = await llm.complete_json_async(
                        kind=kind,
                        request_id=request_id,
                        system_prompt=system_prompt,
                        payload=dict(payload),
                        response_schema=dict(response_schema),
                    )
                await transport_guards[stage].record_success()
                parsed, audit = parser(response)
            except RequestContextLengthExceeded as exc:
                await transport_guards[stage].record_success()
                state.record_canonical_attempt(
                    request_sha256=request_sha256,
                    stage=stage,
                    semantic_attempt_index=semantic_attempt,
                    response=None,
                    validation_error=exc,
                    created_snapshot_id=snapshot_id,
                )
                raise _CanonicalSemanticAttemptsExhausted(
                    stage=stage,
                    request_id=request_id,
                    request_sha256=request_sha256,
                    request_payload_sha256=payload_sha256,
                    attempt_count=semantic_attempt,
                    last_error_type=type(exc).__name__,
                    last_error_message=str(exc),
                ) from exc
            except (RequestRuntimeTimeout, APIError) as exc:
                transport_attempt += 1
                pending_transport_attempts[stage].append(
                    {
                        "request_sha256": request_sha256,
                        "request_id": request_id,
                        "attempt_index": transport_attempt,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                )
                last_semantic_error = exc
                last_error_type = type(exc).__name__
                last_error_message = str(exc)
                await transport_guards[stage].record_failure(
                    request_id=request_id, error=exc
                )
            except (ValueError, RequestCompletionLengthExceeded) as exc:
                await transport_guards[stage].record_success()
                if isinstance(exc, RequestCompletionLengthExceeded):
                    response = {
                        "finish_reason": "length",
                        "incomplete_response": exc.partial_content,
                        "incomplete_reasoning": exc.reasoning_content,
                    }
                state.record_canonical_attempt(
                    request_sha256=request_sha256,
                    stage=stage,
                    semantic_attempt_index=semantic_attempt,
                    response=response,
                    validation_error=exc,
                    created_snapshot_id=snapshot_id,
                )
                last_semantic_error = exc
                last_error_type = type(exc).__name__
                last_error_message = str(exc)
                semantic_attempt += 1
            else:
                stored_audit = {
                    **audit,
                    "request_sha256": request_sha256,
                    "request_payload_sha256": payload_sha256,
                    "semantic_attempt_index": semantic_attempt,
                }
                state.put_canonical_job(
                    request_sha256,
                    stage=stage,
                    semantic_attempt_index=semantic_attempt,
                    prompt_sha256=prompt_sha256,
                    producer_protocol_sha256=protocol_sha,
                    response=dict(response),
                    audit=stored_audit,
                    created_snapshot_id=snapshot_id,
                )
                return parsed, stored_audit
        exhausted = _CanonicalSemanticAttemptsExhausted(
            stage=stage,
            request_id=request_id,
            request_sha256=request_sha256,
            request_payload_sha256=payload_sha256,
            attempt_count=semantic_attempt - 1,
            last_error_type=last_error_type,
            last_error_message=last_error_message,
        )
        if last_semantic_error is None:
            raise exhausted
        raise exhausted from last_semantic_error


    occurrences = _source_occurrences(source)
    published_view_fallbacks = {}
    published_merge_failures = {}
    if published_audit is not None:
        committed_view_ids = {row[0] for row in state.connection.execute("SELECT node_id FROM canonical_views")}
        for row in published_audit["views"]:
            if row["status"] == "VIEW_SOURCE_FALLBACK" and row["leaf_id"] not in committed_view_ids:
                node = occurrences[row["leaf_id"]][1]
                key = _sha256_bytes(canonical_json_bytes(node.to_dict()))
                published_view_fallbacks[key] = row
        for row in published_audit["merge_decisions"]:
            if row["decision"] is None:
                if row["status"] != "MERGE_EXHAUSTED_NO_MERGE":
                    raise ValueError("published no-merge terminal status differs")
                published_merge_failures[(row["left_canonical_id"], row["right_canonical_id"])] = row
    new_occurrences = _source_occurrences(batch)
    prior_units = {_canonical_id(group.members): group for group in previous_partition.groups}
    units = dict(prior_units)
    created_groups: dict[str, CanonicalGroup] = {}
    leaf_by_coordinate = {coordinate: leaf_id for leaf_id, (coordinate, _node) in occurrences.items()}
    for leaf_id, (coordinate, node) in new_occurrences.items():
        group = CanonicalGroup((coordinate,), CanonicalExperience(node.operation, node.applicability, node.inputs, node.outputs))
        key = _canonical_id(group.members)
        if any(coordinate in prior.members for prior in previous_partition.groups):
            raise ValueError("new source occurrence already belongs to a committed group")
        units[key] = group
        created_groups[key] = group

    stored_views = {}
    for row in state.connection.execute(
        "SELECT node_id, source_node_sha256, request_sha256, view_json, normalized_text_sha256, status FROM canonical_views"
    ):
        leaf_id, node_sha, request_sha, view_bytes, text_sha, status = row
        if leaf_id not in occurrences:
            raise ValueError("stored View has no source occurrence")
        if node_sha != _sha256_bytes(canonical_json_bytes(occurrences[leaf_id][1].to_dict())):
            raise ValueError("stored View source differs")
        view = parse_canonicalization_view(json.loads(view_bytes))
        text = normalize_embedding_text(canonicalization_view_embedding_text(view))
        if _sha256_bytes(text.encode()) != text_sha:
            raise ValueError("stored View text differs")
        stored_views[leaf_id] = (view, {"status": status, "request_sha256": request_sha})

    async def produce_view(node: ExperienceNode) -> tuple[CanonicalizationView, dict[str, Any]]:
        frozen = published_view_fallbacks.get(_sha256_bytes(canonical_json_bytes(node.to_dict())))
        if frozen is not None:
            return parse_canonicalization_view(frozen["view"]), {
                key: value for key, value in frozen.items() if key not in {"leaf_id", "view"}
            }
        payload = _canonical_view_payload(node)
        content_id = _sha256_bytes(canonical_json_bytes(payload))
        def parse(response):
            return parse_canonicalization_view(response), {"status": "VIEW_ACCEPTED"}
        try:
            return await run_canonical_request(
                stage="VIEW", llm=view_llm, kind=CANONICAL_VIEW_KIND,
                prompt_sha256=CANONICAL_VIEW_PROMPT_SHA256, system_prompt=CANONICAL_VIEW_SYSTEM_PROMPT,
                request_id=f"canonical-view-{content_id}", payload=payload,
                response_schema=canonicalization_view_response_schema(), parser=parse,
            )
        except _CanonicalSemanticAttemptsExhausted as exc:
            return CanonicalizationView(node.operation, " ".join(node.applicability)), {
                "status": "VIEW_SOURCE_FALLBACK", "request_sha256": None,
                "failed_request_sha256": exc.request_sha256, "last_error_type": exc.last_error_type,
                "last_error_message": exc.last_error_message, "semantic_attempt_count": exc.attempt_count,
            }

    missing_nodes = {}
    missing_keys = {}
    for leaf_id, (_coordinate, node) in occurrences.items():
        if leaf_id not in stored_views:
            key = _sha256_bytes(canonical_json_bytes(node.to_dict()))
            missing_nodes[key] = node
            missing_keys[leaf_id] = key
    ordered_keys = sorted(missing_nodes)
    produced = await run_stage_wave(("VIEW",), tuple(missing_nodes[key] for key in ordered_keys), produce_view)
    by_content = dict(zip(ordered_keys, produced, strict=True))
    views_and_audits = {**stored_views, **{leaf_id: by_content[key] for leaf_id, key in missing_keys.items()}}
    view_texts = {
        leaf_id: normalize_embedding_text(canonicalization_view_embedding_text(view))
        for leaf_id, (view, _audit) in views_and_audits.items()
    }
    cache_before_views = len(embedder.cache)
    embedding_texts = tuple(sorted(set(view_texts.values())))
    embedded = await embedder.embed_async(embedding_texts, workers=EMBEDDING_ASYNC_WORKERS) if embedding_texts else ()
    vectors_by_text = {row.normalized_text: row.vector for row in embedded}
    vectors = {leaf_id: vectors_by_text[text] for leaf_id, text in view_texts.items()}
    view_rows = tuple({
        "node_id": leaf_id,
        "source_node_sha256": _sha256_bytes(canonical_json_bytes(occurrences[leaf_id][1].to_dict())),
        "request_sha256": audit.get("request_sha256"),
        "view": view.to_dict(),
        "normalized_text_sha256": _sha256_bytes(view_texts[leaf_id].encode()),
        "status": "VIEW_SOURCE_FALLBACK" if "FALLBACK" in audit["status"] else "VIEW_ACCEPTED",
    } for leaf_id, (view, audit) in sorted(views_and_audits.items()))
    resolutions = [{
        "stage": "VIEW", "status": "VIEW_SOURCE_FALLBACK",
        "subject": {"leaf_id": leaf_id}, "evidence": dict(audit),
    } for leaf_id, (_view, audit) in sorted(views_and_audits.items()) if leaf_id in missing_keys and "FALLBACK" in audit["status"]]

    frontier = set(units) if refresh else set(created_groups)
    pending: dict[tuple[str, str], float] = {}
    evaluated: set[tuple[str, str]] = set()
    neighbor_map: dict[str, tuple[dict[str, Any], ...]] = {}
    merge_events: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []

    async def judge_merge(pair: tuple[str, str]):
        if pair in published_merge_failures:
            frozen = published_merge_failures[pair]
            return pair, None, {key: value for key, value in frozen.items()
                                if key not in {"left_canonical_id", "right_canonical_id", "decision"}}
        left_id, right_id = pair
        payload = _canonical_merge_payload(units[left_id].canonical_experience, units[right_id].canonical_experience)
        content_id = _sha256_bytes(canonical_json_bytes(payload))
        def parse(response):
            return parse_canonical_merge(response), {"status": "MERGE_ACCEPTED"}
        try:
            decision, audit = await run_canonical_request(
                stage="MERGE", llm=merge_llm, kind=CANONICAL_MERGE_KIND,
                prompt_sha256=CANONICAL_MERGE_PROMPT_SHA256, system_prompt=CANONICAL_MERGE_SYSTEM_PROMPT,
                request_id=f"canonical-merge-{content_id}", payload=payload,
                response_schema=canonical_merge_response_schema(), parser=parse,
            )
            return pair, decision, audit
        except _CanonicalSemanticAttemptsExhausted as exc:
            audit = {
                "status": "MERGE_EXHAUSTED_NO_MERGE", "request_sha256": None,
                "failed_request_sha256": exc.request_sha256, "last_error_type": exc.last_error_type,
                "last_error_message": exc.last_error_message, "semantic_attempt_count": exc.attempt_count,
            }
            return pair, None, audit

    while frontier or pending:
        aliases = {key: tuple(leaf_by_coordinate[m] for m in group.members) for key, group in units.items()}
        for key in sorted(frontier):
            if key not in units:
                continue
            rows = canonical_unit_candidates(key, aliases, vectors, exact_text_by_leaf=view_texts)
            neighbor_map[key] = tuple(row.to_dict() for row in rows)
            for row in rows:
                pair = tuple(sorted((key, row.target_canonical_id)))
                if pair not in evaluated:
                    pending[pair] = max(pending.get(pair, -1.0), row.similarity)
        frontier.clear()
        pending = {pair: score for pair, score in pending.items() if all(key in units for key in pair) and pair not in evaluated}
        wave: list[tuple[str, str]] = []
        occupied: set[str] = set()
        for pair, _score in sorted(pending.items(), key=lambda row: (-row[1], row[0])):
            if not occupied.intersection(pair):
                wave.append(pair)
                occupied.update(pair)
                if len(wave) == CANONICAL_LLM_WORKERS:
                    break
        if not wave:
            break
        # Coalesce identical content requests in a wave; fan out their result to independent unit pairs.
        unique_pairs = {}
        pair_contents = {}
        for pair in wave:
            payload_sha = _sha256_bytes(canonical_json_bytes(_canonical_merge_payload(
                units[pair[0]].canonical_experience, units[pair[1]].canonical_experience)))
            unique_pairs.setdefault(payload_sha, pair)
            pair_contents[pair] = payload_sha
        completed = await run_stage_wave(("MERGE",), tuple(unique_pairs.values()), judge_merge)
        by_payload = {pair_contents[pair]: (decision, audit) for pair, decision, audit in completed}
        for pair in wave:
            evaluated.add(pair)
            pending.pop(pair, None)
            decision, audit = by_payload[pair_contents[pair]]
            left_id, right_id = pair
            record = {**audit, "left_canonical_id": left_id, "right_canonical_id": right_id,
                      "decision": decision.to_dict() if decision is not None else None}
            decisions.append(record)
            if decision is None:
                resolutions.append({"stage": "MERGE", "status": "MERGE_EXHAUSTED_NO_MERGE",
                    "subject": {"left_canonical_id": left_id, "right_canonical_id": right_id}, "evidence": record})
                continue
            if decision.relation is not TemplateRelation.SAME_TEMPLATE:
                continue
            assert decision.canonical_experience is not None
            left, right = units[left_id], units[right_id]
            if set(left.members).intersection(right.members):
                raise ValueError("fusion parents overlap")
            group = CanonicalGroup(tuple(sorted((*left.members, *right.members))), decision.canonical_experience)
            child = _canonical_id(group.members)
            created_groups[child] = group
            del units[left_id], units[right_id]
            units[child] = group
            frontier.add(child)
            merge_events.append({
                "left_canonical_id": left_id, "right_canonical_id": right_id, "child_canonical_id": child,
                "request_sha256": audit["request_sha256"], "apply_order": len(merge_events),
            })

    partition, payload = _partition_from_groups(source, tuple(sorted(units.values(), key=lambda group: group.members[0])))
    _validate_monotonic_partition(previous_partition, partition)
    support_texts = tuple(sorted({
        normalize_embedding_text(group.canonical_experience.operation)
        for group in partition.groups
    }))
    cache_after_views = len(embedder.cache)
    if support_texts:
        await embedder.embed_async(support_texts, workers=EMBEDDING_ASYNC_WORKERS)
    audit = {
        "format": "degs_monotonic_operation_canonical_audit_v1",
        "snapshot_id": snapshot_id,
        "protocol": protocol, "protocol_sha256": protocol_sha,
        "section_graphs_sha256": source.sha256,
        "views": [{"leaf_id": leaf_id, "view": view.to_dict(), **item} for leaf_id, (view, item) in sorted(views_and_audits.items())],
        "merge_decisions": decisions, "merge_events": merge_events,
        "resolution_events": resolutions,
        "prior_groups": [{"canonical_id": key, "members": [list(m) for m in group.members]} for key, group in sorted(prior_units.items())],
        "prior_group_split_count": 0,
        "refresh": refresh, "canonical_group_count": len(partition.groups),
        "embedding": {"api_request_count": (cache_after_views - cache_before_views + 31) // 32 + (len(embedder.cache) - cache_after_views + 31) // 32},
    }
    return IncrementalCanonicalBuild(
        partition=partition, partition_payload=payload, audit=audit,
        changed_canonical_ids=frozenset(set(units) - set(prior_units)),
        retired_canonical_ids=frozenset(set(prior_units) - set(units)),
        requested_embedding_texts=tuple(sorted(set((*embedding_texts, *support_texts)))),
        view_state_rows=view_rows,
        neighbor_state_rows=tuple(row for key in sorted(neighbor_map) for row in neighbor_map[key]),
        merge_event_rows=tuple(merge_events),
        created_groups=tuple(created_groups.values()),
    )


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _write_bytes(path: Path, value: bytes) -> None:
    output = Path(path).expanduser().absolute()
    if output.exists():
        raise FileExistsError("graph output must be fresh")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        try:
            remaining = memoryview(value)
            while remaining:
                written = os.write(descriptor, remaining)
                if written <= 0:
                    raise OSError("graph output write made no progress")
                remaining = remaining[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.link(temporary, output, follow_symlinks=False)
    finally:
        temporary.unlink(missing_ok=True)


def _train_input_sha(source_sha: str, audit_sha: str, indices: Sequence[int], protocol_sha: str) -> str:
    return _sha256_bytes(canonical_json_bytes({
        "batch_source_sha256": source_sha, "batch_source_audit_sha256": audit_sha,
        "batch_train_indices": list(indices), "canonical_protocol_sha256": protocol_sha,
    }))


def _operation_id(parent: str | None, kind: str, input_sha: str) -> str:
    if kind != "TRAIN_BATCH" or len(input_sha) != 64:
        raise ValueError("snapshot operation identity differs")
    digest = _sha256_bytes(canonical_json_bytes({
        "format": INCREMENTAL_SNAPSHOT_FORMAT, "method": INCREMENTAL_METHOD_ID,
        "parent_snapshot_id": parent, "operation_kind": kind, "operation_input_sha256": input_sha,
    }))
    return f"snapshot_{digest[:24]}"


def _snapshot_id(
    *, parent_snapshot_id: str | None, batch_source_sha256: str,
    batch_source_audit_sha256: str, batch_train_indices: Sequence[int], canonical_protocol_sha256: str,
) -> str:
    return _operation_id(parent_snapshot_id, "TRAIN_BATCH",
        _train_input_sha(batch_source_sha256, batch_source_audit_sha256, batch_train_indices, canonical_protocol_sha256))


def _source_batch_from_accumulated(
    source: SectionGraphSource,
    batch_train_indices: Sequence[int],
) -> SectionGraphSource:
    selected = set(batch_train_indices)
    workflows = tuple(
        workflow
        for workflow in source.workflows
        if workflow.train_index in selected
    )
    provisional = SectionGraphSource(
        workflows,
        "0" * 64,
        source.source_split,
    )
    return SectionGraphSource(
        workflows,
        _sha256_bytes(canonical_json_bytes(_source_payload(provisional))),
        source.source_split,
    )


def validate_cumulative_source_audit(
    *, source: SectionGraphSource, ledger: Mapping[str, Any],
    expected_snapshot_id: str, expected_generation_endpoint: str | None = None,
    dataset_contract: GraphDatasetContract = SPREADSHEETBENCH_GRAPH_CONTRACT,
    source_audit_validator: Callable[..., tuple[str, dict[int, str]]] = (
        _validate_batch_source_audit
    ),
) -> dict[int, str]:
    fields = {"format", "method", "batches"}
    if (type(ledger) is not dict or set(ledger) != fields
        or ledger["format"] != INCREMENTAL_SOURCE_AUDIT_LEDGER_FORMAT or ledger["method"] != INCREMENTAL_METHOD_ID
        or type(ledger["batches"]) is not list
        or not 1 <= len(ledger["batches"]) <= dataset_contract.batch_count):
        raise ValueError("incremental cumulative source audit differs")
    statuses: dict[int, str] = {}
    operations: dict[str, str | None] = {}
    for number, record in enumerate(ledger["batches"]):
        indices = dataset_contract.batch_indices(number)
        if (type(record) is not dict or set(record) != {
            "snapshot_id", "parent_snapshot_id", "batch_train_indices",
            "batch_source_sha256", "batch_source_audit_sha256", "batch_source_audit", "canonical_protocol_sha256"}
            or record["batch_train_indices"] != list(indices)):
            raise ValueError("incremental cumulative source audit chain differs")
        batch = _source_batch_from_accumulated(source, indices)
        audit_sha, batch_status = source_audit_validator(
            batch_source=batch, batch_train_indices=indices, audit=record["batch_source_audit"],
            expected_generation_endpoint=expected_generation_endpoint)
        if record["batch_source_sha256"] != batch.sha256 or record["batch_source_audit_sha256"] != audit_sha:
            raise ValueError("incremental cumulative source audit identity differs")
        identity = _snapshot_id(parent_snapshot_id=record["parent_snapshot_id"],
            batch_source_sha256=batch.sha256, batch_source_audit_sha256=audit_sha, batch_train_indices=indices,
            canonical_protocol_sha256=record["canonical_protocol_sha256"])
        if record["snapshot_id"] != identity or identity in operations:
            raise ValueError("incremental source operation identity differs")
        operations[identity] = record["parent_snapshot_id"]
        statuses.update(batch_status)
    cursor, visited = expected_snapshot_id, set()
    while cursor is not None:
        if cursor not in operations or cursor in visited:
            raise ValueError("incremental cumulative source audit head differs")
        visited.add(cursor)
        cursor = operations[cursor]
    if visited != set(operations):
        raise ValueError("cumulative source audit has detached operations")
    return statuses


class IncrementalGraphBuilder:
    def __init__(
        self,
        *,
        state: IncrementalStateStore,
        snapshot_root: Path,
        embedder: StrictEmbeddingAdapter,
        view_llm: JsonObjectLLM,
        merge_llm: JsonObjectLLM,
        dataset_contract: GraphDatasetContract = SPREADSHEETBENCH_GRAPH_CONTRACT,
        source_audit_validator: Callable[
            ..., tuple[str, dict[int, str]]
        ] = _validate_batch_source_audit,
    ) -> None:
        if type(state) is not IncrementalStateStore:
            raise TypeError("incremental state store is required")
        if not isinstance(embedder, StrictEmbeddingAdapter):
            raise TypeError("strict embedding adapter is required")
        if state.dataset_contract != dataset_contract:
            raise ValueError("incremental state graph dataset contract differs")
        if not callable(source_audit_validator):
            raise TypeError("source audit validator is required")
        self.state = state
        self.snapshot_root = Path(snapshot_root).expanduser().absolute()
        self.embedder = embedder
        self.view_llm = view_llm
        self.merge_llm = merge_llm
        self.dataset_contract = dataset_contract
        self.source_audit_validator = source_audit_validator
        endpoint = getattr(embedder.transport, "endpoint", None)
        if type(endpoint) is not str or not endpoint:
            raise ValueError("incremental embedding endpoint identity differs")
        self.state.bind_embedding_endpoint(endpoint)
        generation_endpoints = {
            getattr(llm, "protocol_identity", {}).get("service_url")
            for llm in (view_llm, merge_llm)
        }
        if (
            len(generation_endpoints) != 1
            or None in generation_endpoints
            or "" in generation_endpoints
        ):
            raise ValueError("incremental generation endpoint identity differs")
        generation_endpoint = next(iter(generation_endpoints))
        if type(generation_endpoint) is not str:
            raise ValueError("incremental generation endpoint identity differs")
        validate_service_url(generation_endpoint)
        self.state.bind_generation_endpoint(generation_endpoint)
    @property
    def canonical_protocol_sha256(self) -> str:
        return _sha256_bytes(canonical_json_bytes(canonical_protocol(
            self.view_llm.protocol_identity,
            self.merge_llm.protocol_identity,
            self.dataset_contract,
        )))

    def _previous_artifacts(
        self,
    ) -> tuple[
        SectionGraphSource | None,
        CanonicalPartition | None,
        Mapping[str, Any] | None,
    ]:
        head = self.state.head_snapshot_id
        if head is None:
            return None, None, None
        row = self.state.connection.execute(
            """
            SELECT status, manifest_sha256, source_workflow_count,
                   canonical_count, experience_node_count, experience_edge_count
            FROM snapshots WHERE snapshot_id = ?
            """,
            (head,),
        ).fetchone()
        root = self.snapshot_root / head
        manifest_path = root / "snapshot_manifest.json"
        manifest_bytes = manifest_path.read_bytes()
        manifest = _load_json(manifest_path)
        unsigned = {
            key: value for key, value in manifest.items() if key != "self_sha256"
        }
        artifacts = manifest.get("artifacts")
        if (
            row is None
            or row[0] != "COMMITTED"
            or manifest_bytes != canonical_json_bytes(manifest)
            or manifest.get("snapshot_id") != head
            or manifest.get("self_sha256")
            != _sha256_bytes(canonical_json_bytes(unsigned))
            or row[1] != manifest.get("self_sha256")
            or artifacts
            != {
                "accumulated_section_graphs": "accumulated_section_graphs.json",
                "batch_source_audit": "batch_source_audit.json",
                "cumulative_source_audit": "cumulative_source_audit.json",
                "canonical_partition": "canonical_partition.json",
                "canonical_audit": "canonical_audit.json",
                "experience_graph": "experience_graph.json",
                "graph_quality_audit": "graph_quality_audit.json",
                "source_node_audit": "source_node_audit.jsonl",
                "high_similarity_unmerged": "high_similarity_unmerged.jsonl",
                "candidate_recall_audit": "candidate_recall_audit.jsonl",
                "canonical_merge_ledger": "canonical_merge_ledger.jsonl",
                "graph_topology": "graph_topology.json",
            }
        ):
            raise ValueError("committed incremental snapshot identity differs")
        artifacts = cast(dict[str, str], artifacts)
        source = load_section_graphs(
            root / artifacts["accumulated_section_graphs"],
            allow_empty=True,
            dataset_contract=self.dataset_contract,
        )
        partition = load_canonical_partition(
            root / artifacts["canonical_partition"], source=source, allow_empty=True
        )
        experience_graph = compile_experience_graph(source, partition)
        experience_graph_path = root / artifacts["experience_graph"]
        experience_graph_payload = _load_json(experience_graph_path)
        canonical_audit_path = root / artifacts["canonical_audit"]
        canonical_audit_bytes = canonical_audit_path.read_bytes()
        canonical_audit = _load_json(canonical_audit_path)
        batch_audit_path = root / artifacts["batch_source_audit"]
        batch_audit_bytes = batch_audit_path.read_bytes()
        batch_audit = _load_json(batch_audit_path)
        cumulative_audit_path = root / artifacts["cumulative_source_audit"]
        cumulative_audit_bytes = cumulative_audit_path.read_bytes()
        cumulative_audit = _load_json(cumulative_audit_path)
        quality_path = root / artifacts["graph_quality_audit"]
        quality_bytes = quality_path.read_bytes()
        quality = _load_json(quality_path)
        quality_hashes = quality.get("artifact_hashes")
        quality_state = self.state.connection.execute(
            """
            SELECT status, audit_protocol_sha256, summary_sha256,
                   source_node_audit_sha256, merge_ledger_sha256,
                   unresolved_candidates_sha256, topology_sha256
            FROM graph_quality_audits WHERE snapshot_id = ?
            """,
            (head,),
        ).fetchone()
        cumulative_statuses = validate_cumulative_source_audit(
            source=source,
            ledger=cumulative_audit,
            expected_snapshot_id=head,
            expected_generation_endpoint=self.state.generation_endpoint,
            dataset_contract=self.dataset_contract,
            source_audit_validator=self.source_audit_validator,
        )
        stored_statuses = {
            int(index): str(status)
            for index, status in self.state.connection.execute(
                """
                SELECT item.train_index, item.status
                FROM snapshot_batch_items AS item
                JOIN snapshots AS snapshot USING(snapshot_id)
                WHERE snapshot.status = 'COMMITTED'
                ORDER BY item.train_index
                """
            ).fetchall()
        }
        if (
            source.sha256 != manifest.get("accumulated_source_sha256")
            or partition.sha256 != manifest.get("canonical_partition_sha256")
            or experience_graph.to_dict() != experience_graph_payload
            or experience_graph.experience_graph_sha256
            != manifest.get("experience_graph_sha256")
            or canonical_audit_bytes != canonical_json_bytes(canonical_audit)
            or _sha256_bytes(canonical_audit_bytes)
            != manifest.get("canonical_audit_sha256")
            or canonical_audit.get("protocol_sha256")
            != manifest.get("canonical_protocol_sha256")
            or batch_audit_bytes != canonical_json_bytes(batch_audit)
            or _sha256_bytes(batch_audit_bytes)
            != manifest.get("batch_source_audit_sha256")
            or cumulative_audit_bytes != canonical_json_bytes(cumulative_audit)
            or _sha256_bytes(cumulative_audit_bytes)
            != manifest.get("cumulative_source_audit_sha256")
            or cumulative_statuses != stored_statuses
            or type(quality_hashes) is not dict
            or quality_bytes != canonical_json_bytes(quality)
            or _sha256_bytes(quality_bytes)
            != manifest.get("graph_quality_audit_sha256")
            or quality.get("protocol_sha256")
            != manifest.get("graph_quality_protocol_sha256")
            or quality.get("status") != manifest.get("graph_quality_status")
            or quality_state
            != (
                quality.get("status"),
                quality.get("protocol_sha256"),
                _sha256_bytes(quality_bytes),
                quality_hashes.get("source_node_audit_sha256"),
                quality_hashes.get("merge_ledger_sha256"),
                quality_hashes.get("unresolved_candidates_sha256"),
                quality_hashes.get("topology_sha256"),
            )
            or _sha256_bytes((root / artifacts["source_node_audit"]).read_bytes())
            != quality_hashes.get("source_node_audit_sha256")
            or _sha256_bytes(
                (root / artifacts["canonical_merge_ledger"]).read_bytes()
            ) != quality_hashes.get("merge_ledger_sha256")
            or _sha256_bytes(
                (root / artifacts["high_similarity_unmerged"]).read_bytes()
            ) != quality_hashes.get("unresolved_candidates_sha256")
            or _sha256_bytes(
                (root / artifacts["candidate_recall_audit"]).read_bytes()
            ) != quality_hashes.get("candidate_recall_audit_sha256")
            or _sha256_bytes((root / artifacts["graph_topology"]).read_bytes())
            != quality_hashes.get("topology_sha256")
            or tuple(row[2:])
            != (
                len(source.workflows),
                len(partition.groups),
                len(experience_graph.nodes),
                len(experience_graph.edges),
            )
        ):
            raise ValueError("committed incremental snapshot artifact differs")
        return source, partition, cumulative_audit

    def _prepare_snapshot(
        self,
        *,
        snapshot_id: str,
        parent_snapshot_id: str | None,
        batch_source_sha256: str,
        batch_source_audit_sha256: str,
        batch_train_indices: Sequence[int],
        operation_kind: str = "TRAIN_BATCH",
        operation_input_sha256: str | None = None,
    ) -> None:
        operation_input_sha256 = operation_input_sha256 or _train_input_sha(batch_source_sha256, batch_source_audit_sha256, batch_train_indices, self.canonical_protocol_sha256)
        with self.state.transaction():
            processed = {
                int(row[0])
                for row in self.state.connection.execute(
                    """
                    SELECT item.train_index
                    FROM snapshot_batch_items AS item
                    JOIN snapshots AS snapshot USING(snapshot_id)
                    WHERE snapshot.status = 'COMMITTED'
                    """
                ).fetchall()
            }
            overlap = processed.intersection(batch_train_indices)
            if overlap:
                raise ValueError(
                    f"incremental batch repeats processed train indices: {sorted(overlap)}"
                )
            other_build = self.state.connection.execute(
                """
                SELECT snapshot_id FROM snapshots
                WHERE status = 'BUILDING' AND snapshot_id != ?
                """,
                (snapshot_id,),
            ).fetchone()
            if other_build is not None:
                raise RuntimeError(
                    f"incremental snapshot {other_build[0]} is already building"
                )
            row = self.state.connection.execute(
                """
                SELECT parent_snapshot_id, batch_source_sha256,
                       batch_source_audit_sha256, status, operation_kind, operation_input_sha256
                FROM snapshots WHERE snapshot_id = ?
                """,
                (snapshot_id,),
            ).fetchone()
            expected = (
                parent_snapshot_id,
                batch_source_sha256,
                batch_source_audit_sha256,
                "BUILDING", operation_kind, operation_input_sha256,
            )
            if row is None:
                self.state.connection.execute(
                    """
                    INSERT INTO snapshots(
                        snapshot_id, parent_snapshot_id, batch_source_sha256,
                        batch_source_audit_sha256, status, operation_kind, operation_input_sha256
                    ) VALUES (?, ?, ?, ?, 'BUILDING', ?, ?)
                    """,
                    (
                        snapshot_id,
                        parent_snapshot_id,
                        batch_source_sha256,
                        batch_source_audit_sha256, operation_kind, operation_input_sha256,
                    ),
                )
            elif tuple(row) != expected:
                raise ValueError("incremental snapshot resume identity differs")

    async def update_async(
        self,
        *,
        batch_source: SectionGraphSource,
        batch_train_indices: Sequence[int],
        batch_source_audit: Mapping[str, Any],
    ) -> IncrementalSnapshotBuild:
        protocol_sha = self.canonical_protocol_sha256
        indices = tuple(sorted(batch_train_indices))
        if (
            not indices
            or len(indices) != len(set(indices))
            or any(
                type(index) is not int
                or not 0 <= index < self.dataset_contract.train_count
                for index in indices
            )
        ):
            raise ValueError("incremental graph batch indices differ")
        if batch_source.source_split != self.dataset_contract.source_split:
            raise ValueError("incremental batch source split differs")
        batch_workflow_ids = {
            workflow.train_index for workflow in batch_source.workflows
        }
        if not batch_workflow_ids <= set(indices):
            raise ValueError(
                "batch source contains a workflow outside its 8-task batch"
            )
        batch_source_audit_sha256, source_status_by_index = (
            self.source_audit_validator(
                batch_source=batch_source,
                batch_train_indices=indices,
                audit=batch_source_audit,
                expected_generation_endpoint=self.state.generation_endpoint,
            )
        )

        existing_batches = self.state.connection.execute(
            """
            SELECT snapshot.snapshot_id, snapshot.batch_source_sha256,
                   snapshot.batch_source_audit_sha256, item.train_index
            FROM snapshots AS snapshot
            JOIN snapshot_batch_items AS item USING(snapshot_id)
            WHERE snapshot.status = 'COMMITTED'
            ORDER BY snapshot.snapshot_id, item.train_index
            """
        ).fetchall()
        by_snapshot: dict[str, tuple[str, str, list[int]]] = {}
        for existing_snapshot_id, source_sha, audit_sha, train_index in existing_batches:
            if existing_snapshot_id not in by_snapshot:
                by_snapshot[str(existing_snapshot_id)] = (
                    str(source_sha),
                    str(audit_sha),
                    [],
                )
            by_snapshot[str(existing_snapshot_id)][2].append(int(train_index))
        for existing_snapshot_id, (
            source_sha,
            audit_sha,
            existing_indices,
        ) in by_snapshot.items():
            if tuple(existing_indices) != indices:
                continue
            if (
                source_sha != batch_source.sha256
                or audit_sha != batch_source_audit_sha256
            ):
                raise ValueError(
                    "incremental batch replay changes its source or audit"
                )
            result_snapshot_id = (
                self.state.head_snapshot_id or existing_snapshot_id
            )
            output = self.snapshot_root / result_snapshot_id
            self._previous_artifacts()
            if _load_json(output / "snapshot_manifest.json")["canonical_protocol_sha256"] != protocol_sha:
                raise ValueError("committed batch Canonical protocol differs")
            return IncrementalSnapshotBuild(
                result_snapshot_id,
                output,
                _load_json(output / "snapshot_manifest.json"),
            )

        processed_indices = tuple(
            int(row[0])
            for row in self.state.connection.execute(
                """
                SELECT item.train_index
                FROM snapshot_batch_items AS item
                JOIN snapshots AS snapshot USING(snapshot_id)
                WHERE snapshot.status = 'COMMITTED'
                ORDER BY item.train_index
                """
            ).fetchall()
        )
        if processed_indices != tuple(range(len(processed_indices))):
            raise ValueError(
                "committed incremental train-index prefix differs"
            )
        expected_indices = self.dataset_contract.next_batch_indices(
            len(processed_indices)
        )
        if indices != expected_indices:
            raise ValueError(
                f"next incremental batch must be train indices {list(expected_indices)}"
            )

        parent_snapshot_id = self.state.head_snapshot_id
        snapshot_id = _snapshot_id(
            parent_snapshot_id=parent_snapshot_id,
            batch_source_sha256=batch_source.sha256,
            batch_source_audit_sha256=batch_source_audit_sha256,
            batch_train_indices=indices,
            canonical_protocol_sha256=protocol_sha,
        )
        (
            previous_source,
            previous_partition,
            previous_source_audit_ledger,
        ) = self._previous_artifacts()
        if parent_snapshot_id is not None:
            parent_manifest = _load_json(self.snapshot_root / parent_snapshot_id / "snapshot_manifest.json")
            if parent_manifest["canonical_protocol_sha256"] != protocol_sha:
                raise ValueError("incremental Canonical protocol changes across batches")
        self._prepare_snapshot(
            snapshot_id=snapshot_id,
            parent_snapshot_id=parent_snapshot_id,
            batch_source_sha256=batch_source.sha256,
            batch_source_audit_sha256=batch_source_audit_sha256,
            batch_train_indices=indices,
        )
        source = merge_source_batch(previous_source, batch_source)
        cumulative_source_audit = {
            "format": INCREMENTAL_SOURCE_AUDIT_LEDGER_FORMAT,
            "method": INCREMENTAL_METHOD_ID,
            "batches": [
                *(
                    []
                    if previous_source_audit_ledger is None
                    else previous_source_audit_ledger["batches"]
                ),
                {
                    "snapshot_id": snapshot_id,
                    "parent_snapshot_id": parent_snapshot_id,
                    "batch_train_indices": list(indices),
                    "canonical_protocol_sha256": protocol_sha,
                    "batch_source_sha256": batch_source.sha256,
                    "batch_source_audit_sha256": (
                        batch_source_audit_sha256
                    ),
                    "batch_source_audit": dict(batch_source_audit),
                },
            ],
        }
        cumulative_statuses = validate_cumulative_source_audit(
            source=source,
            ledger=cumulative_source_audit,
            expected_snapshot_id=snapshot_id,
            expected_generation_endpoint=self.state.generation_endpoint,
            dataset_contract=self.dataset_contract,
            source_audit_validator=self.source_audit_validator,
        )
        committed_statuses = {
            int(index): str(status)
            for index, status in self.state.connection.execute(
                """
                SELECT item.train_index, item.status
                FROM snapshot_batch_items AS item
                JOIN snapshots AS snapshot USING(snapshot_id)
                WHERE snapshot.status = 'COMMITTED'
                """
            ).fetchall()
        }
        if cumulative_statuses != {
            **committed_statuses,
            **source_status_by_index,
        }:
            raise ValueError(
                "incremental cumulative source status differs"
            )

        return await self._finish_operation(
            snapshot_id=snapshot_id, parent_snapshot_id=parent_snapshot_id,
            operation_kind="TRAIN_BATCH", operation_input_sha256=_train_input_sha(batch_source.sha256, batch_source_audit_sha256, indices, protocol_sha),
            source=source, batch_source=batch_source, indices=indices,
            batch_source_audit=batch_source_audit, batch_source_audit_sha256=batch_source_audit_sha256,
            cumulative_source_audit=cumulative_source_audit, source_status_by_index=source_status_by_index,
            previous_partition=previous_partition,
            all_processed_indices=tuple(sorted(set(processed_indices) | set(indices))),
        )

    async def _finish_operation(
        self, *, snapshot_id: str, parent_snapshot_id: str | None,
        operation_kind: str, operation_input_sha256: str, source: SectionGraphSource,
        batch_source: SectionGraphSource, indices: Sequence[int],
        batch_source_audit: Mapping[str, Any], batch_source_audit_sha256: str,
        cumulative_source_audit: Mapping[str, Any], source_status_by_index: Mapping[int, str],
        previous_partition: CanonicalPartition | None, all_processed_indices: Sequence[int],
    ) -> IncrementalSnapshotBuild:
        if previous_partition is None:
            previous_partition = CanonicalPartition(
                (), "0" * 64, "0" * 64
            )
        # A published operation has already decided item-local failures. Replay its
        # terminal outcomes only for this operation, never as historical cannot-links.
        output = self.snapshot_root / snapshot_id
        published_audit = None
        if output.exists():
            published = _load_json(output / "snapshot_manifest.json")
            published_audit = _load_json(output / "canonical_audit.json")
            expected_identity = {
                "snapshot_id": snapshot_id, "parent_snapshot_id": parent_snapshot_id,
                "operation_kind": operation_kind, "operation_input_sha256": operation_input_sha256,
                "accumulated_source_sha256": source.sha256,
                "canonical_protocol_sha256": self.canonical_protocol_sha256,
            }
            if (any(published.get(key) != value for key, value in expected_identity.items())
                or published.get("self_sha256") != _sha256_bytes(canonical_json_bytes({k: v for k, v in published.items() if k != "self_sha256"}))
                or published.get("canonical_audit_sha256") != _sha256_bytes(canonical_json_bytes(published_audit))
                or published_audit.get("protocol_sha256") != self.canonical_protocol_sha256
                or published_audit.get("section_graphs_sha256") != source.sha256
                or published_audit.get("snapshot_id") != snapshot_id):
                raise ValueError("published operation resume identity differs")
        cache_before_canonical = len(self.embedder.cache)
        canonical = await build_incremental_canonical_partition(
            source=source,
            batch=batch_source,
            previous_partition=previous_partition,
            view_llm=self.view_llm,
            merge_llm=self.merge_llm,
            embedder=self.embedder,
            state=self.state,
            snapshot_id=snapshot_id,
            refresh=False,
            published_audit=published_audit,
            dataset_contract=self.dataset_contract,
        )
        cache_after_canonical = len(self.embedder.cache)
        if parent_snapshot_id is not None:
            parent_manifest = _load_json(
                self.snapshot_root
                / parent_snapshot_id
                / "snapshot_manifest.json"
            )
            if parent_manifest.get("canonical_protocol_sha256") != (
                canonical.audit["protocol_sha256"]
            ):
                raise ValueError(
                    "incremental Canonical protocol changes across batches"
                )

        experience_graph = compile_experience_graph(
            source, canonical.partition
        )
        canonical_document_texts = tuple(
            dict.fromkeys(
                normalize_embedding_text(node.document)
                for node in experience_graph.nodes
            )
        )
        cache_before_documents = len(self.embedder.cache)
        if canonical_document_texts:
            await self.embedder.embed_async(
                canonical_document_texts,
                workers=EMBEDDING_ASYNC_WORKERS,
            )
        cache_after_documents = len(self.embedder.cache)
        document_cache_misses = (
            cache_after_documents - cache_before_documents
        )
        canonical_cache_misses = (
            cache_after_canonical - cache_before_canonical
        )
        canonical_embedding = canonical.audit["embedding"]
        unique_requested_text_count = len(
            {
                *canonical.requested_embedding_texts,
                *canonical_document_texts,
            }
        )
        cache_misses = canonical_cache_misses + document_cache_misses
        canonical_audit_sha256 = _sha256_bytes(
            canonical_json_bytes(canonical.audit)
        )
        cumulative_source_audit_sha256 = _sha256_bytes(
            canonical_json_bytes(cumulative_source_audit)
        )
        graph_quality = build_graph_quality_artifacts(
            snapshot_id=snapshot_id,
            source=source,
            partition=canonical.partition,
            graph=experience_graph,
            canonical_audit=canonical.audit,
            cumulative_source_audit=cumulative_source_audit,
            state=self.state,
        )
        graph_quality_audit_sha256 = _sha256_bytes(
            canonical_json_bytes(graph_quality.audit)
        )

        snapshot_body = {
            "format": INCREMENTAL_SNAPSHOT_FORMAT,
            "method": INCREMENTAL_METHOD_ID,
            "snapshot_id": snapshot_id,
            "parent_snapshot_id": parent_snapshot_id,
            "operation_kind": operation_kind,
            "operation_input_sha256": operation_input_sha256,
            "batch_size": len(indices),
            "batch_train_indices": list(indices),
            "batch_ingested_workflow_ids": sorted(w.train_index for w in batch_source.workflows),
            "batch_no_source_indices": sorted(
                set(indices) - {w.train_index for w in batch_source.workflows}
            ),
            "batch_source_sha256": batch_source.sha256,
            "batch_source_audit_sha256": batch_source_audit_sha256,
            "cumulative_source_audit_sha256": (
                cumulative_source_audit_sha256
            ),
            "processed_train_index_count": len(all_processed_indices),
            "processed_train_indices_sha256": _sha256_bytes(
                canonical_json_bytes(list(all_processed_indices))
            ),
            "accumulated_source_sha256": source.sha256,
            "canonical_partition_sha256": canonical.partition.sha256,
            "experience_graph_sha256": (
                experience_graph.experience_graph_sha256
            ),
            "canonical_audit_sha256": canonical_audit_sha256,
            "canonical_protocol_sha256": (
                canonical.audit["protocol_sha256"]
            ),
            "graph_quality_status": graph_quality.status,
            "graph_quality_audit_sha256": graph_quality_audit_sha256,
            "graph_quality_protocol_sha256": GRAPH_QUALITY_PROTOCOL_SHA256,
            "source_workflow_count": len(source.workflows),
            "canonical_count": len(canonical.partition.groups),
            "experience_node_count": len(experience_graph.nodes),
            "experience_edge_count": len(experience_graph.edges),
            "embedding": {
                "endpoint": self.state.embedding_endpoint,
                "model": EMBEDDING_MODEL,
                "cache_namespace_sha256": EMBEDDING_CACHE_NAMESPACE,
                "unique_requested_text_count": unique_requested_text_count,
                "cache_hit_count": (
                    unique_requested_text_count - cache_misses
                ),
                "cache_miss_count": cache_misses,
                "api_request_count": (
                    canonical_embedding["api_request_count"]
                    + (document_cache_misses + 31) // 32
                ),
                "async_workers": EMBEDDING_ASYNC_WORKERS,
            },
            "generation_endpoint": self.state.generation_endpoint,
            "coverage": {
                "native_train_batch_count": len(cumulative_source_audit["batches"]),
            },
            "canonical": {
                "method": "monotonic_operation_canonicalization",
                "changed_canonical_ids": sorted(
                    canonical.changed_canonical_ids
                ),
                "retired_canonical_ids": sorted(
                    canonical.retired_canonical_ids
                ),
            },
            "artifacts": {
                "accumulated_section_graphs": (
                    "accumulated_section_graphs.json"
                ),
                "batch_source_audit": "batch_source_audit.json",
                "cumulative_source_audit": (
                    "cumulative_source_audit.json"
                ),
                "canonical_partition": "canonical_partition.json",
                "canonical_audit": "canonical_audit.json",
                "experience_graph": "experience_graph.json",
                "graph_quality_audit": "graph_quality_audit.json",
                "source_node_audit": "source_node_audit.jsonl",
                "high_similarity_unmerged": (
                    "high_similarity_unmerged.jsonl"
                ),
                "candidate_recall_audit": "candidate_recall_audit.jsonl",
                "canonical_merge_ledger": "canonical_merge_ledger.jsonl",
                "graph_topology": "graph_topology.json",
            },
        }
        snapshot_manifest = {
            **snapshot_body,
            "self_sha256": _sha256_bytes(
                canonical_json_bytes(snapshot_body)
            ),
        }
        output = self.snapshot_root / snapshot_id
        if output.exists():
            published = _load_json(output / "snapshot_manifest.json")
            original_audit = _load_json(output / "canonical_audit.json")
            runtime_fields = {"embedding", "canonical_audit_sha256", "graph_quality_audit_sha256", "self_sha256"}
            if ({k: v for k, v in published.items() if k not in runtime_fields}
                != {k: v for k, v in snapshot_manifest.items() if k not in runtime_fields}
                or published.get("self_sha256") != _sha256_bytes(canonical_json_bytes({k: v for k, v in published.items() if k != "self_sha256"}))
                or published["canonical_audit_sha256"] != _sha256_bytes(canonical_json_bytes(original_audit))):
                raise ValueError("published snapshot resume identity differs")
            def semantic_audit(value):
                return {
                    "protocol_sha256": value["protocol_sha256"],
                    "views": [(row["leaf_id"], row["view"]) for row in value["views"]],
                    "merge_events": value["merge_events"],
                    "merge_decisions": [(row["left_canonical_id"], row["right_canonical_id"], row["decision"]) for row in value["merge_decisions"]],
                    "prior_groups": value["prior_groups"],
                }
            if semantic_audit(original_audit) != semantic_audit(canonical.audit):
                raise ValueError("published Canonical decisions differ on replay")
            quality_audit = _load_json(output / "graph_quality_audit.json")
            if published["graph_quality_audit_sha256"] != _sha256_bytes(canonical_json_bytes(quality_audit)):
                raise ValueError("published graph quality hash differs")
            def jsonl(filename):
                return tuple(json.loads(line) for line in (output / filename).read_bytes().splitlines())
            graph_quality = GraphQualityArtifacts(quality_audit, jsonl("source_node_audit.jsonl"),
                jsonl("high_similarity_unmerged.jsonl"), jsonl("candidate_recall_audit.jsonl"),
                jsonl("canonical_merge_ledger.jsonl"), _load_json(output / "graph_topology.json"))
            for field, filename in (("source_node_audit_sha256", "source_node_audit.jsonl"),
                ("merge_ledger_sha256", "canonical_merge_ledger.jsonl"),
                ("unresolved_candidates_sha256", "high_similarity_unmerged.jsonl"),
                ("candidate_recall_audit_sha256", "candidate_recall_audit.jsonl"),
                ("topology_sha256", "graph_topology.json")):
                if _sha256_bytes((output / filename).read_bytes()) != quality_audit["artifact_hashes"][field]:
                    raise ValueError("published graph quality artifact differs")
            canonical = replace(canonical, audit=original_audit)
            snapshot_manifest = published
        self._publish_snapshot(
            output=output,
            source=source,
            batch_source_audit=batch_source_audit,
            cumulative_source_audit=cumulative_source_audit,
            canonical=canonical,
            experience_graph=experience_graph,
            graph_quality=graph_quality,
            manifest=snapshot_manifest,
            dataset_contract=self.dataset_contract,
        )
        self._commit_state(
            state=self.state,
            snapshot_id=snapshot_id,
            parent_snapshot_id=parent_snapshot_id,
            batch_source=batch_source,
            batch_train_indices=indices,
            source_status_by_index=source_status_by_index,
            source=source,
            canonical=canonical,
            experience_graph=experience_graph,
            graph_quality=graph_quality,
            manifest=snapshot_manifest,
        )
        return IncrementalSnapshotBuild(
            snapshot_id, output, snapshot_manifest
        )
    @staticmethod
    def _publish_snapshot(
        *,
        output: Path,
        source: SectionGraphSource,
        batch_source_audit: Mapping[str, Any],
        cumulative_source_audit: Mapping[str, Any],
        canonical: IncrementalCanonicalBuild,
        experience_graph: ExperienceGraph,
        graph_quality: GraphQualityArtifacts,
        manifest: Mapping[str, Any],
        dataset_contract: GraphDatasetContract,
    ) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            existing = _load_json(output / "snapshot_manifest.json")
            if existing != dict(manifest):
                raise FileExistsError("snapshot output conflicts with resumed build")
            checked_source = load_section_graphs(
                output / "accumulated_section_graphs.json",
                allow_empty=True,
                dataset_contract=dataset_contract,
            )
            checked_partition = load_canonical_partition(
                output / "canonical_partition.json",
                source=checked_source,
                allow_empty=True,
            )
            if (
                checked_source != source
                or checked_partition != canonical.partition
                or compile_experience_graph(checked_source, checked_partition)
                != experience_graph
                or _load_json(output / "experience_graph.json")
                != experience_graph.to_dict()
                or _load_json(output / "batch_source_audit.json")
                != dict(batch_source_audit)
                or _load_json(output / "cumulative_source_audit.json")
                != dict(cumulative_source_audit)
                or _load_json(output / "canonical_audit.json") != canonical.audit
                or _load_json(output / "graph_quality_audit.json")
                != graph_quality.audit
                or any(
                    (output / name).read_bytes() != value
                    for name, value in graph_quality.serialized_files().items()
                )
            ):
                raise ValueError("resumed incremental snapshot artifact differs")
            return
        staging = Path(
            tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
        )
        published = False
        try:
            _write_json_output(
                staging / "accumulated_section_graphs.json", _source_payload(source)
            )
            _write_json_output(
                staging / "batch_source_audit.json", dict(batch_source_audit)
            )
            _write_json_output(
                staging / "cumulative_source_audit.json",
                dict(cumulative_source_audit),
            )
            _write_json_output(
                staging / "canonical_partition.json", canonical.partition_payload
            )
            _write_json_output(staging / "canonical_audit.json", canonical.audit)
            _write_json_output(
                staging / "experience_graph.json", experience_graph.to_dict()
            )
            for name, value in graph_quality.serialized_files().items():
                _write_bytes(staging / name, value)
            checked_source = load_section_graphs(
                staging / "accumulated_section_graphs.json",
                allow_empty=True,
                dataset_contract=dataset_contract,
            )
            checked_partition = load_canonical_partition(
                staging / "canonical_partition.json",
                source=checked_source,
                allow_empty=True,
            )
            checked_graph = compile_experience_graph(checked_source, checked_partition)
            if checked_graph != experience_graph:
                raise ValueError("published incremental experience graph differs")
            _write_json_output(staging / "snapshot_manifest.json", dict(manifest))
            os.replace(staging, output)
            published = True
        finally:
            if not published and staging.exists():
                shutil.rmtree(staging)

    @staticmethod
    def _commit_state(
        *,
        state: IncrementalStateStore,
        snapshot_id: str,
        parent_snapshot_id: str | None,
        batch_source: SectionGraphSource,
        batch_train_indices: Sequence[int],
        source_status_by_index: Mapping[int, str],
        source: SectionGraphSource,
        canonical: IncrementalCanonicalBuild,
        experience_graph: ExperienceGraph,
        graph_quality: GraphQualityArtifacts,
        manifest: Mapping[str, Any],
    ) -> None:
        with state.transaction():
            if state.head_snapshot_id != parent_snapshot_id:
                raise RuntimeError("incremental graph head changed during build")
            for workflow in batch_source.workflows:
                workflow_payload = {
                    "train_index": workflow.train_index,
                    "task_id": workflow.task_id,
                    "query_text": workflow.query_text,
                    "experience_nodes": [node.to_dict() for node in workflow.experience_nodes],
                    "edges": [edge.to_dict() for edge in workflow.edges],
                }
                workflow_bytes = canonical_json_bytes(workflow_payload)
                state.connection.execute(
                    """
                    INSERT INTO workflows(
                        train_index, task_id, query_text, query_text_sha256,
                        workflow_sha256, workflow_json, added_snapshot_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        workflow.train_index,
                        workflow.task_id,
                        workflow.query_text,
                        _sha256_bytes(workflow.query_text.encode("utf-8")),
                        _sha256_bytes(workflow_bytes),
                        workflow_bytes,
                        snapshot_id,
                    ),
                )
                node_ids: list[str] = []
                for node_index, node in enumerate(workflow.experience_nodes):
                    node_id = experience_leaf_id(
                        workflow.train_index, node_index, node
                    )
                    node_ids.append(node_id)
                    node_bytes = canonical_json_bytes(node.to_dict())
                    state.connection.execute(
                        """
                        INSERT INTO experience_nodes(
                            node_id, train_index, node_index, node_sha256,
                            node_json, added_snapshot_id
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            node_id,
                            workflow.train_index,
                            node_index,
                            _sha256_bytes(node_bytes),
                            node_bytes,
                            snapshot_id,
                        ),
                    )
                for edge in workflow.edges:
                    state.connection.execute(
                        """
                        INSERT INTO source_edges(
                            train_index, source_node_id, target_node_id
                        ) VALUES (?, ?, ?)
                        """,
                        (
                            workflow.train_index,
                            node_ids[edge.source],
                            node_ids[edge.target],
                        ),
                    )
            canonical_by_member: dict[tuple[int, int], str] = {}
            all_groups = {_canonical_id(group.members): group for group in (*canonical.created_groups, *canonical.partition.groups)}
            final_ids = {_canonical_id(group.members) for group in canonical.partition.groups}
            for group in all_groups.values():
                canonical_id = _canonical_id(group.members)
                canonical_bytes = canonical_json_bytes(group.canonical_experience.to_dict())
                document = _canonical_document(group.canonical_experience)
                existing = state.connection.execute(
                    "SELECT canonical_sha256, document_sha256 FROM canonical_nodes WHERE canonical_id = ?",
                    (canonical_id,),
                ).fetchone()
                identity = (
                    _sha256_bytes(canonical_bytes),
                    _sha256_bytes(document.encode("utf-8")),
                )
                if existing is None:
                    state.connection.execute(
                        """
                        INSERT INTO canonical_nodes(
                            canonical_id, canonical_sha256, canonical_json,
                            document_sha256, document, member_count,
                            created_snapshot_id, retired_snapshot_id
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL)
                        """,
                        (
                            canonical_id,
                            identity[0],
                            canonical_bytes,
                            identity[1],
                            document,
                            len(group.members),
                            snapshot_id,
                        ),
                    )
                elif tuple(existing) != identity:
                    raise ValueError("stable Canonical identity changed content")
                else:
                    state.connection.execute(
                        "UPDATE canonical_nodes SET retired_snapshot_id = NULL WHERE canonical_id = ?",
                        (canonical_id,),
                    )
                for member in group.members:
                    workflow = source.workflow_by_index[member[0]]
                    node_id = experience_leaf_id(
                        member[0], member[1], workflow.experience_nodes[member[1]]
                    )
                    if canonical_id in final_ids:
                        canonical_by_member[member] = canonical_id
                    state.connection.execute(
                        "INSERT OR IGNORE INTO canonical_leaf_members(canonical_id, node_id) VALUES (?, ?)",
                        (canonical_id, node_id),
                    )
                    if canonical_id not in final_ids:
                        continue
                    state.connection.execute(
                        """
                        INSERT INTO canonical_heads(node_id, canonical_id) VALUES (?, ?)
                        ON CONFLICT(node_id) DO UPDATE SET canonical_id = excluded.canonical_id
                        """,
                        (node_id, canonical_id),
                    )
            for retired_id in (set(all_groups) | set(canonical.retired_canonical_ids)) - final_ids:
                state.connection.execute(
                    "UPDATE canonical_nodes SET retired_snapshot_id = ? WHERE canonical_id = ? AND retired_snapshot_id IS NULL",
                    (snapshot_id, retired_id),
                )
            for row in canonical.view_state_rows:
                state.put_canonical_view(
                    node_id=str(row["node_id"]),
                    source_node_sha256=str(row["source_node_sha256"]),
                    request_sha256=cast(str | None, row["request_sha256"]),
                    view=cast(Mapping[str, Any], row["view"]),
                    normalized_text_sha256=str(row["normalized_text_sha256"]),
                    status=str(row["status"]),
                    updated_snapshot_id=snapshot_id,
                )
            state.replace_snapshot_neighbors(
                snapshot_id=snapshot_id,
                rows=canonical.neighbor_state_rows,
            )
            for event in canonical.merge_event_rows:
                state.put_canonical_merge_event(snapshot_id=snapshot_id, event=event)
            for event in canonical.audit["resolution_events"]:
                state.put_canonical_resolution_event(
                    stage=str(event["stage"]),
                    status=str(event["status"]),
                    subject=cast(Mapping[str, Any], event["subject"]),
                    evidence=cast(Mapping[str, Any], event["evidence"]),
                    created_snapshot_id=snapshot_id,
                )
            touched_workflows = {
                workflow.train_index for workflow in batch_source.workflows
            }
            for group in canonical.partition.groups:
                if _canonical_id(group.members) in canonical.changed_canonical_ids:
                    touched_workflows.update(member[0] for member in group.members)
            for train_index in touched_workflows:
                state.connection.execute(
                    "DELETE FROM source_edge_projection WHERE train_index = ?",
                    (train_index,),
                )
                workflow = source.workflow_by_index[train_index]
                pairs = {
                    (
                        canonical_by_member[(train_index, edge.source)],
                        canonical_by_member[(train_index, edge.target)],
                    )
                    for edge in workflow.edges
                }
                for source_id, target_id in sorted(pairs):
                    state.connection.execute(
                        """
                        INSERT INTO source_edge_projection(
                            train_index, source_canonical_id, target_canonical_id
                        ) VALUES (?, ?, ?)
                        """,
                        (train_index, source_id, target_id),
                    )
            stored_projection: dict[tuple[str, str], set[int]] = defaultdict(set)
            for train_index, source_id, target_id in state.connection.execute(
                """
                SELECT train_index, source_canonical_id, target_canonical_id
                FROM source_edge_projection
                ORDER BY train_index, source_canonical_id, target_canonical_id
                """
            ).fetchall():
                stored_projection[(str(source_id), str(target_id))].add(
                    int(train_index)
                )
            expected_projection = {
                (edge.source, edge.target): set(edge.supporting_workflow_ids)
                for edge in experience_graph.edges
            }
            if stored_projection != expected_projection:
                raise ValueError("incremental source edge projection differs")
            for train_index in batch_train_indices:
                state.connection.execute(
                    """
                    INSERT INTO snapshot_batch_items(snapshot_id, train_index, status)
                    VALUES (?, ?, ?)
                    """,
                    (
                        snapshot_id,
                        train_index,
                        source_status_by_index[train_index],
                    ),
                )
            state.connection.execute(
                """
                UPDATE snapshots SET
                    manifest_sha256 = ?,
                    source_workflow_count = ?,
                    canonical_count = ?,
                    experience_node_count = ?,
                    experience_edge_count = ?
                WHERE snapshot_id = ? AND status = 'BUILDING'
                """,
                (
                    manifest["self_sha256"],
                    len(source.workflows),
                    len(canonical.partition.groups),
                    len(experience_graph.nodes),
                    len(experience_graph.edges),
                    snapshot_id,
                ),
            )
            _validate_committing_snapshot_state(
                state=state,
                snapshot_id=snapshot_id,
                parent_snapshot_id=parent_snapshot_id,
                source=source,
                partition=canonical.partition,
                graph=experience_graph,
                manifest=manifest,
                source_status_by_index=source_status_by_index,
                expected_snapshot_status="BUILDING",
            )
            artifact_hashes = graph_quality.audit["artifact_hashes"]
            state.put_graph_quality_audit(
                snapshot_id=snapshot_id,
                status=graph_quality.status,
                audit_protocol_sha256=GRAPH_QUALITY_PROTOCOL_SHA256,
                summary_sha256=_sha256_bytes(
                    canonical_json_bytes(graph_quality.audit)
                ),
                source_node_audit_sha256=str(
                    artifact_hashes["source_node_audit_sha256"]
                ),
                merge_ledger_sha256=str(
                    artifact_hashes["merge_ledger_sha256"]
                ),
                unresolved_candidates_sha256=str(
                    artifact_hashes["unresolved_candidates_sha256"]
                ),
                topology_sha256=str(artifact_hashes["topology_sha256"]),
                hard_violations=cast(
                    Sequence[Mapping[str, Any]],
                    graph_quality.audit["hard_violations"],
                ),
            )
            updated = state.connection.execute(
                """
                UPDATE snapshots SET status = 'COMMITTED'
                WHERE snapshot_id = ? AND status = 'BUILDING'
                """,
                (snapshot_id,),
            ).rowcount
            if updated != 1:
                raise ValueError("committing snapshot status differs")
            state.set_head_snapshot_id(snapshot_id)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply one 8-task batch to the DEGS ExperienceGraph."
    )
    parser.add_argument("--state-db", type=Path, required=True)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--batch-section-graphs", type=Path)
    parser.add_argument("--batch-source-audit", type=Path)
    parser.add_argument(
        "--batch-train-index",
        type=int,
        action="append",
        help="Exactly the next contiguous aligned block of eight train indices.",
    )
    parser.add_argument("--llm-base-url", required=True)
    parser.add_argument("--embedding-base-url", required=True)
    parser.add_argument("--llm-api-key-env", default="DEGS_API_KEY")
    parser.add_argument(
        "--embedding-api-key-env", default="DEGS_EMBEDDING_API_KEY"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    validate_service_url(args.llm_base_url)
    validate_service_url(args.embedding_base_url)
    llm_key = os.environ.get(args.llm_api_key_env)
    embedding_key = os.environ.get(args.embedding_api_key_env)
    if not llm_key or not embedding_key:
        raise ValueError("generation and embedding API keys are required")
    batch_arguments = (args.batch_section_graphs, args.batch_source_audit, args.batch_train_index)
    if any(value is None for value in batch_arguments):
        raise ValueError("train update requires source, audit and eight indices")
    with IncrementalStateStore(args.state_db) as state:
        embedder = StrictEmbeddingAdapter(
            QwenEmbeddingHTTPTransport(
                base_url=args.embedding_base_url,
                api_key=embedding_key,
            ),
            cache=state.embedding_cache(),
        )
        llm_client = OpenAIClient(
            model=REPAIR_SOURCE_MODEL,
            api_key=llm_key,
            base_url=args.llm_base_url,
            generation_config=_source_generation_config(),
            retry_times=PRODUCER_TRANSPORT_RETRY_WAITS,
            runtime_timeout_retries=PRODUCER_RUNTIME_TIMEOUT_RETRIES,
            timeout=REPAIR_SOURCE_TIMEOUT_SECONDS,
            trust_env=False,
        )
        builder = IncrementalGraphBuilder(
                state=state,
                snapshot_root=args.snapshot_root,
                embedder=embedder,
                view_llm=openai_canonical_view_llm(llm_client),
                merge_llm=openai_canonical_merge_llm(llm_client),
            )
        batch_source = load_section_graphs(args.batch_section_graphs, allow_empty=True)
        batch_audit_bytes = args.batch_source_audit.read_bytes()
        batch_source_audit = _load_json(args.batch_source_audit)
        if batch_audit_bytes != canonical_json_bytes(batch_source_audit):
            raise ValueError("batch source audit must use canonical JSON")
        operation = builder.update_async(
            batch_source=batch_source,
            batch_train_indices=args.batch_train_index,
            batch_source_audit=batch_source_audit,
        )
        result = asyncio.run(operation)
    print(json.dumps(dict(result.manifest), sort_keys=True))
    return 0


__all__ = [
    "INCREMENTAL_BATCH_SIZE",
    "INCREMENTAL_SNAPSHOT_FORMAT",
    "INCREMENTAL_SOURCE_AUDIT_LEDGER_FORMAT",
    "IncrementalCanonicalBuild",
    "IncrementalGraphBuilder",
    "IncrementalSnapshotBuild",
    "build_incremental_canonical_partition",
    "main",
    "merge_source_batch",
    "validate_cumulative_source_audit",
]


if __name__ == "__main__":
    raise SystemExit(main())
