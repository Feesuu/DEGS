#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import copy
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import tempfile
from typing import Any, Mapping, Sequence, cast

from openai import APIError
from react_agent.models import (
    OpenAIClient,
    RequestCompletionLengthExceeded,
    RequestContextLengthExceeded,
    RequestRuntimeTimeout,
)
from sb_adapter.transport import validate_service_url

from .canonicalize import (
    CANONICAL_VIEW_KIND, CANONICAL_VIEW_PROMPT_SHA256, CANONICAL_VIEW_PROTOCOL_FORMAT,
    CANONICAL_MERGE_KIND, CANONICAL_MERGE_PROMPT_SHA256, CANONICAL_MERGE_PROTOCOL_FORMAT,
)

from .core import (
    EMBEDDING_ASYNC_WORKERS,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL,
    EMBEDDING_TIMEOUT_S,
    StrictEmbeddingAdapter,
    canonical_json_bytes,
    normalize_embedding_text,
)
from .dataset import (
    DEVELOPMENT_COUNT,
    DEVELOPMENT_START,
    EXPECTED_DEVELOPMENT_QUERY_PROJECTION_SHA256,
    EXPECTED_QUERY_PROJECTION_SHA256,
    development_query_projection_sha256,
    _load_development_harness_records,
    _load_train_harness_records,
    load_train_queries,
    query_projection_sha256,
)
from .incremental_graph import (
    INCREMENTAL_SNAPSHOT_FORMAT,
    validate_cumulative_source_audit,
    canonical_protocol,
)
from .experience_simgrag import (
    RECALL_SCORE_FORMAT,
    SEMANTIC_RECALL_DOCUMENT_FORMAT,
    build_retrieval_index,
    canonical_retrieval_document,
    need_retrieval_document,
    retrieve_experience_subgraphs,
)
from .section_graph import (
    CanonicalPartition,
    ExperienceGraph,
    SectionGraphSource,
    compile_experience_graph,
    load_canonical_partition,
    load_section_graphs,
)
from .state_store import INCREMENTAL_METHOD_ID, IncrementalStateStore
from .runtime_identity import METHOD_CONTRACT, METHOD_VERSION
from .target_context import (
    TargetContextUnavailable,
    TargetEvidenceCard,
    WORKFLOW_CONTEXT_DOCUMENT_FORMAT,
    build_target_evidence_card,
    unavailable_target_evidence_card,
)
from .retrieval_clarification import (
    CLARIFICATION_KIND,
    CLARIFICATION_PROMPT_SHA256,
    CLARIFICATION_PROTOCOL_FORMAT,
    CLARIFICATION_SYSTEM_PROMPT,
    RetrievalClarificationDecision,
    clarifiable_need_indices,
    clarification_response_schema,
    openai_retrieval_clarification_llm,
    parse_retrieval_clarification,
)
from .transport import QwenEmbeddingHTTPTransport, seal_train_instruction_authority
from .validated_repair import (
    PRODUCER_RUNTIME_TIMEOUT_RETRIES,
    PRODUCER_TRANSPORT_RETRY_WAITS,
    REPAIR_SOURCE_MAX_TOKENS,
    REPAIR_SOURCE_MODEL,
    REPAIR_SOURCE_TEMPERATURE,
    REPAIR_SOURCE_THINKING,
    REPAIR_SOURCE_TIMEOUT_SECONDS,
)
from .workflow_retrieval import (
    NEED_GRAPH_KIND,
    NEED_GRAPH_PROTOCOL_FORMAT,
    NEED_GRAPH_PROMPT_SHA256,
    NEED_GRAPH_SYSTEM_PROMPT,
    SELECTOR_KIND,
    SELECTOR_PROTOCOL_FORMAT,
    SELECTOR_PROMPT_SHA256,
    SELECTOR_SYSTEM_PROMPT,
    WORKFLOW_RECALL_K,
    NeedGraph,
    NeedGuidedRetrievalResult,
    SubgraphSelection,
    finalize_subgraph_selection,
    need_graph_response_schema,
    openai_need_graph_llm,
    openai_selector_llm,
    parse_need_graph,
    parse_selector_response,
    recall_source_workflows,
    render_workflow_fallback,
    selector_response_schema,
    workflow_fallback_candidates,
    workflow_fallback_selector_payload,
)


FORMAT = "degs_retrieval_bundle_v1"
EXPERIENCE_FORMAT = "degs_experience_v1"
METHOD_NAME = "DEGS 0.77.41 Stable R1"
METHOD_FAMILY = "EXPERIENCE_SIMGRAG_RETRIEVAL"
CLAIM_SCOPE = (
    "train[0,200) ExperienceGraph with development[200,400) query plus symmetric "
    "input-workbook context for parent-workflow late fusion; "
    "no development outcome, gold, verifier result, or Agent trace during bundle build"
)
LLM_ASYNC_WORKERS = 16
SEMANTIC_ATTEMPTS = 3
BUNDLE_TRANSPORT_RETRY_WAITS = PRODUCER_TRANSPORT_RETRY_WAITS
BUNDLE_RUNTIME_TIMEOUT_RETRIES = PRODUCER_RUNTIME_TIMEOUT_RETRIES
_RESULT_STATUSES = frozenset(
    {
        "OK",
        "SELECTOR_FALLBACK_C0",
        "NEED_GRAPH_FALLBACK_WORKFLOW",
        "SEARCH_FALLBACK_WORKFLOW",
    }
)
_TASK_AUDIT_FIELDS = {
    "query_index",
    "dataset_index",
    "task_id",
    "status",
    "row_sha256",
    "retrieval_audit_sha256",
}
_EXPERIENCE_METADATA_FIELDS = {
    "format",
    "method_family",
    "status",
    "query_index",
    "dataset_index",
    "authority_sha256",
    "experience_sha256",
    "retrieval_audit_sha256",
    "retrieval_audit",
}
_EMBEDDING_PROTOCOL_FIELDS = {
    "endpoint",
    "model",
    "batch_size",
    "timeout_s",
    "cache",
    "unique_normalized_text_count",
    "cache_hit_count",
    "cache_miss_count",
    "api_request_count",
    "async_workers",
}
_LLM_PRODUCER_FIELDS = {
    "format",
    "request_kind",
    "model",
    "temperature",
    "thinking",
    "max_tokens",
    "timeout_seconds",
    "generation_config",
    "retry_waits_seconds",
    "runtime_timeout_retries",
    "prompt_sha256",
    "service_url",
}
EXPECTED_BUNDLE_RUNTIME = {
    "python": "3.12.7",
    "dependencies": {
        "networkx": "3.6.1",
        "numpy": "2.5.2",
        "openpyxl": "3.1.5",
    },
}


@dataclass(frozen=True)
class _Task:
    query_index: int
    dataset_index: int
    task_id: str
    instruction: str
    spreadsheet_path: str
    answer_position: str


@dataclass(frozen=True)
class _SnapshotContext:
    source: SectionGraphSource
    partition: CanonicalPartition
    graph: ExperienceGraph
    identity: Mapping[str, Any]
    source_statuses: Mapping[int, str]


_VERIFIED_TOKEN = object()


class _VerifiedExperienceBundle:
    __slots__ = ("_root", "_manifest")

    def __init__(self, token: object, root: Path, manifest: Mapping[str, Any]) -> None:
        if token is not _VERIFIED_TOKEN:
            raise TypeError("verified bundle capabilities cannot be caller-constructed")
        self._root = root
        self._manifest = copy.deepcopy(dict(manifest))

    @property
    def root(self) -> Path:
        return self._root

    @property
    def manifest(self) -> dict[str, Any]:
        return copy.deepcopy(self._manifest)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _strict_json_bytes(value: bytes, *, label: str) -> Any:
    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate key {key!r}")
            result[key] = item
        return result

    try:
        return json.loads(
            value.decode("utf-8"),
            object_pairs_hook=no_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} is not strict JSON") from exc


def _read_canonical_json(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    value = path.read_bytes()
    payload = _strict_json_bytes(value, label=label)
    if type(payload) is not dict or value != canonical_json_bytes(payload):
        raise ValueError(f"{label} is not canonical JSON")
    return payload, value


def _write_exclusive(path: Path, value: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(value)


def _runtime_identity() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in EXPECTED_BUNDLE_RUNTIME["dependencies"]
        },
    }


def _source_code_identity() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    files = (
        "bundle.py",
        "canonicalize.py",
        "core.py",
        "dataset.py",
        "experience_simgrag.py",
        "graph_quality.py",
        "incremental_graph.py",
        "section_graph.py",
        "source_replay.py",
        "source_replay_executor.py",
        "replay_overflow_probe.py",
        "source_rebuild.py",
        "successful_source.py",
        "state_store.py",
        "source_review.py",
        "transport.py",
        "validated_repair.py",
        "workflow_retrieval.py",
        "retrieval_clarification.py",
        "target_context.py",
        "runtime_identity.py",
        "resources/QUERY_NEED_GRAPH_PROMPT_V2.txt",
        "resources/CANONICALIZATION_VIEW_PROMPT_V4.txt",
        "resources/CANONICAL_OPERATION_MERGE_PROMPT_V2.txt",
        "resources/EXPERIENCE_SOURCE_REVIEW_PROMPT_V2.txt",
        "resources/SOURCE_REPLAY_PATCH_PROMPT_V2.txt",
        "resources/SUCCESSFUL_TRAJECTORY_EXPERIENCE_PROMPT_V4.txt",
        "resources/VALIDATED_REPAIR_EXPERIENCE_PROMPT_V4.txt",
        "resources/EXPERIENCE_SIMGRAG_SELECTOR_PROMPT_V4.txt",
        "resources/EVIDENCE_RETRIEVAL_CLARIFICATION_PROMPT_V1.txt",
    )
    return {name: _sha((root / name).read_bytes()) for name in files}


def _validated_llm_protocol(
    value: Mapping[str, Any],
    *,
    expected_format: str,
    expected_kind: str,
    expected_prompt_sha256: str,
    expected_retry_waits: Sequence[int] = (),
    expected_runtime_timeout_retries: int = 0,
) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError("retrieval LLM protocol differs")
    protocol = dict(value)
    service_url = protocol.get("service_url")
    expected = {
        "format": expected_format,
        "request_kind": expected_kind,
        "model": REPAIR_SOURCE_MODEL,
        "temperature": REPAIR_SOURCE_TEMPERATURE,
        "thinking": REPAIR_SOURCE_THINKING,
        "max_tokens": REPAIR_SOURCE_MAX_TOKENS,
        "timeout_seconds": REPAIR_SOURCE_TIMEOUT_SECONDS,
        "generation_config": {
            "temperature": REPAIR_SOURCE_TEMPERATURE,
            "max_tokens": REPAIR_SOURCE_MAX_TOKENS,
            "extra_body": {
                "chat_template_kwargs": {
                    "enable_thinking": REPAIR_SOURCE_THINKING
                }
            },
        },
        "retry_waits_seconds": list(expected_retry_waits),
        "runtime_timeout_retries": expected_runtime_timeout_retries,
        "prompt_sha256": expected_prompt_sha256,
        "service_url": service_url,
    }
    if (
        set(protocol) != _LLM_PRODUCER_FIELDS
        or protocol != expected
        or type(service_url) is not str
        or not service_url.rstrip("/").endswith("/v1")
    ):
        raise ValueError("retrieval LLM protocol differs")
    validate_service_url(service_url)
    return protocol


def _validated_canonical_protocol(
    audit: Mapping[str, Any], *, snapshot_id: str, source_sha256: str,
) -> dict[str, Any]:
    if (type(audit) is not dict or audit.get("format") != "degs_monotonic_operation_canonical_audit_v1"
        or audit.get("snapshot_id") != snapshot_id or audit.get("section_graphs_sha256") != source_sha256):
        raise ValueError("incremental Canonical audit identity differs")
    protocol = audit.get("protocol")
    if type(protocol) is not dict:
        raise ValueError("incremental Canonical protocol differs")
    producers = []
    for name, format_id, kind, prompt_sha in (
        ("canonicalization_view", CANONICAL_VIEW_PROTOCOL_FORMAT, CANONICAL_VIEW_KIND, CANONICAL_VIEW_PROMPT_SHA256),
        ("merge", CANONICAL_MERGE_PROTOCOL_FORMAT, CANONICAL_MERGE_KIND, CANONICAL_MERGE_PROMPT_SHA256),
    ):
        record = protocol.get(name)
        if type(record) is not dict or type(record.get("llm_protocol")) is not dict:
            raise ValueError("incremental Canonical producer protocol differs")
        producers.append(_validated_llm_protocol(
            record["llm_protocol"], expected_format=format_id, expected_kind=kind,
            expected_prompt_sha256=prompt_sha,
            expected_retry_waits=PRODUCER_TRANSPORT_RETRY_WAITS,
            expected_runtime_timeout_retries=PRODUCER_RUNTIME_TIMEOUT_RETRIES,
        ))
    if (protocol != canonical_protocol(*producers)
        or len({p["service_url"].rstrip("/") for p in producers}) != 1
        or audit.get("protocol_sha256") != _sha(canonical_json_bytes(protocol))
        or any(type(audit.get(key)) is not list for key in ("views", "merge_decisions", "merge_events", "prior_groups", "resolution_events"))):
        raise ValueError("incremental Canonical protocol differs")
    return protocol


def _embedding_protocol_matches(value: Any) -> bool:
    if type(value) is not dict or set(value) != _EMBEDDING_PROTOCOL_FIELDS:
        return False
    endpoint = value["endpoint"]
    try:
        validate_service_url(str(endpoint))
    except ValueError:
        return False
    misses = value["cache_miss_count"]
    hits = value["cache_hit_count"]
    unique = value["unique_normalized_text_count"]
    return (
        type(endpoint) is str
        and endpoint.rstrip("/").endswith("/v1/embeddings")
        and value["model"] == EMBEDDING_MODEL
        and value["batch_size"] == EMBEDDING_BATCH_SIZE
        and value["timeout_s"] == EMBEDDING_TIMEOUT_S
        and value["cache"] == "normalized_text_sha256"
        and type(unique) is int
        and unique > 0
        and type(hits) is int
        and hits >= 0
        and type(misses) is int
        and misses >= 0
        and hits + misses == unique
        and value["api_request_count"]
        == (misses + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE
        and value["async_workers"] == EMBEDDING_ASYNC_WORKERS
    )


def _task_plan(rows: Sequence[Mapping[str, Any]]) -> tuple[_Task, ...]:
    tasks = tuple(
        _Task(
            int(row["development_index"]),
            int(row["dataset_index"]),
            str(row["task_id"]),
            str(row["instruction"]),
            str(row["spreadsheet_path"]),
            str(row["answer_position"]),
        )
        for row in rows
    )
    if len(tasks) != DEVELOPMENT_COUNT or any(
        task.query_index != index
        or task.dataset_index != DEVELOPMENT_START + index
        for index, task in enumerate(tasks)
    ):
        raise ValueError("development query order differs")
    return tasks


def _query_fields(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "development_index": row["development_index"],
            "dataset_index": row["dataset_index"],
            "task_id": row["task_id"],
            "instruction": row["instruction"],
        }
        for row in rows
    ]


def _target_evidence_cards(
    tasks: Sequence[_Task], *, dataset_path: Path
) -> dict[int, TargetEvidenceCard]:
    cards: dict[int, TargetEvidenceCard] = {}
    for task in tasks:
        try:
            card = build_target_evidence_card(
                dataset_path=dataset_path,
                spreadsheet_path=task.spreadsheet_path,
                answer_position=task.answer_position,
                instruction=task.instruction,
            )
        except TargetContextUnavailable as exc:
            card = unavailable_target_evidence_card(exc)
        cards[task.query_index] = card
    if len(cards) >= 3 and not any(card.status == "OK" for card in cards.values()):
        raise RuntimeError(
            "systemic target-evidence source failure: no input workbook was readable"
        )
    return cards


def _train_evidence_cards(
    rows: Sequence[Mapping[str, Any]], *, dataset_path: Path
) -> dict[int, TargetEvidenceCard]:
    cards: dict[int, TargetEvidenceCard] = {}
    for row in rows:
        train_index = int(row["train_index"])
        try:
            card = build_target_evidence_card(
                dataset_path=dataset_path,
                spreadsheet_path=str(row["spreadsheet_path"]),
                answer_position=str(row["answer_position"]),
                instruction=str(row["instruction"]),
            )
        except TargetContextUnavailable as exc:
            card = unavailable_target_evidence_card(exc)
        cards[train_index] = card
    if len(cards) != 200 or any(index not in cards for index in range(200)):
        raise ValueError("train workflow-context evidence population differs")
    if not any(card.status == "OK" for card in cards.values()):
        raise RuntimeError(
            "systemic train workflow-context source failure: no input workbook was readable"
        )
    return cards


def _evidence_cards_sha256(cards: Mapping[int, TargetEvidenceCard]) -> str:
    return _sha(
        canonical_json_bytes(
            [
                {"index": index, "audit": cards[index].audit_payload()}
                for index in sorted(cards)
            ]
        )
    )


def _workflow_document_for_kinds(
    *,
    card: TargetEvidenceCard,
    instruction: str,
    kinds: set[str],
) -> str:
    if card.status != "OK":
        return instruction
    evidence_ids = tuple(
        row["evidence_id"]
        for row in card.observations
        if row["kind"] in kinds
    )
    if not evidence_ids:
        return instruction
    return card.workflow_context_document(
        instruction,
        evidence_ids=evidence_ids,
    )


def _load_snapshot(
    snapshot_manifest_path: Path,
    *,
    state_db_path: Path,
    train_queries: Sequence[Mapping[str, Any]],
) -> _SnapshotContext:
    manifest_path = snapshot_manifest_path.expanduser().absolute()
    root = manifest_path.parent
    manifest, manifest_bytes = _read_canonical_json(
        manifest_path, label="incremental snapshot manifest"
    )
    artifacts = manifest.get("artifacts")
    expected_artifacts = {
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
    unsigned = {key: value for key, value in manifest.items() if key != "self_sha256"}
    expected_processed_sha = _sha(canonical_json_bytes(list(range(200))))
    snapshot_embedding = manifest.get("embedding")
    if (
        manifest.get("format") != INCREMENTAL_SNAPSHOT_FORMAT
        or manifest.get("method") != INCREMENTAL_METHOD_ID
        or manifest.get("self_sha256") != _sha(canonical_json_bytes(unsigned))
        or manifest_bytes != canonical_json_bytes(manifest)
        or manifest.get("processed_train_index_count") != 200
        or manifest.get("graph_quality_status") not in {"READY", "NOT_READY"}
        or manifest.get("processed_train_indices_sha256") != expected_processed_sha
        or artifacts != expected_artifacts
        or type(snapshot_embedding) is not dict
        or set(snapshot_embedding)
        != {
            "endpoint", "model", "cache_namespace_sha256",
            "unique_requested_text_count", "cache_hit_count",
            "cache_miss_count", "api_request_count", "async_workers",
        }
        or snapshot_embedding.get("model") != EMBEDDING_MODEL
        or snapshot_embedding.get("async_workers") != EMBEDDING_ASYNC_WORKERS
    ):
        raise ValueError("bundle requires a complete audited DEGS graph snapshot")
    artifacts = cast(dict[str, str], artifacts)
    snapshot_embedding = cast(dict[str, Any], snapshot_embedding)
    validate_service_url(snapshot_embedding["endpoint"])
    source = load_section_graphs(root / artifacts["accumulated_section_graphs"])
    partition = load_canonical_partition(
        root / artifacts["canonical_partition"], source=source
    )
    graph = compile_experience_graph(source, partition)
    graph_payload, graph_bytes = _read_canonical_json(
        root / artifacts["experience_graph"], label="ExperienceGraph"
    )
    ledger, ledger_bytes = _read_canonical_json(
        root / artifacts["cumulative_source_audit"],
        label="cumulative source audit",
    )
    query_by_index = {int(row["train_index"]): row for row in train_queries}
    if any(
        query_by_index[workflow.train_index]["task_id"] != workflow.task_id
        or query_by_index[workflow.train_index]["instruction"] != workflow.query_text
        for workflow in source.workflows
    ):
        raise ValueError("source workflow query identity differs from train[0,200)")
    audited_source_rows = [
        row
        for batch in ledger["batches"]
        for key in ("rows", "exclusions")
        for row in batch["batch_source_audit"][key]
    ]
    if any(
        query_by_index[row["train_index"]]["task_id"] != row["task_id"]
        for row in audited_source_rows
    ):
        raise ValueError("source audit task identity differs from train[0,200)")
    canonical_audit_path = root / artifacts["canonical_audit"]
    canonical_audit, canonical_audit_bytes = _read_canonical_json(
        canonical_audit_path, label="incremental Canonical audit"
    )
    canonical_protocol = _validated_canonical_protocol(
        canonical_audit,
        snapshot_id=manifest["snapshot_id"],
        source_sha256=source.sha256,
    )
    graph_quality, graph_quality_bytes = _read_canonical_json(
        root / artifacts["graph_quality_audit"],
        label="graph quality audit",
    )
    quality_hashes = graph_quality.get("artifact_hashes")
    generation_endpoint = canonical_protocol["canonicalization_view"][
        "llm_protocol"
    ]["service_url"].rstrip("/")
    statuses = validate_cumulative_source_audit(
        source=source,
        ledger=ledger,
        expected_snapshot_id=manifest["snapshot_id"],
        expected_generation_endpoint=generation_endpoint,
    )
    if (
        source.sha256 != manifest.get("accumulated_source_sha256")
        or partition.sha256 != manifest.get("canonical_partition_sha256")
        or graph.to_dict() != graph_payload
        or graph_bytes != canonical_json_bytes(graph_payload)
        or graph.experience_graph_sha256 != manifest.get("experience_graph_sha256")
        or _sha(ledger_bytes) != manifest.get("cumulative_source_audit_sha256")
        or _sha(canonical_audit_bytes) != manifest.get("canonical_audit_sha256")
        or _sha(canonical_json_bytes(canonical_protocol))
        != manifest.get("canonical_protocol_sha256")
        or manifest.get("generation_endpoint") != generation_endpoint
        or graph_quality.get("status") != manifest.get("graph_quality_status")
        or _sha(graph_quality_bytes)
        != manifest.get("graph_quality_audit_sha256")
        or graph_quality.get("protocol_sha256")
        != manifest.get("graph_quality_protocol_sha256")
        or type(quality_hashes) is not dict
        or _sha((root / artifacts["source_node_audit"]).read_bytes())
        != quality_hashes.get("source_node_audit_sha256")
        or _sha((root / artifacts["canonical_merge_ledger"]).read_bytes())
        != quality_hashes.get("merge_ledger_sha256")
        or _sha((root / artifacts["high_similarity_unmerged"]).read_bytes())
        != quality_hashes.get("unresolved_candidates_sha256")
        or _sha((root / artifacts["candidate_recall_audit"]).read_bytes())
        != quality_hashes.get("candidate_recall_audit_sha256")
        or _sha((root / artifacts["graph_topology"]).read_bytes())
        != quality_hashes.get("topology_sha256")
        or set(statuses) != set(range(200))
    ):
        raise ValueError("incremental snapshot artifact identity differs")
    with IncrementalStateStore(state_db_path) as state:
        snapshot_row = state.connection.execute(
            """
            SELECT status, manifest_sha256, source_workflow_count,
                   canonical_count, experience_node_count, experience_edge_count
            FROM snapshots WHERE snapshot_id = ?
            """,
            (manifest["snapshot_id"],),
        ).fetchone()
        stored = {
            int(index): str(status)
            for index, status in state.connection.execute(
                """
                SELECT item.train_index, item.status
                FROM snapshot_batch_items AS item
                JOIN snapshots AS snapshot USING(snapshot_id)
                WHERE snapshot.status = 'COMMITTED'
                ORDER BY item.train_index
                """
            ).fetchall()
        }
        quality_state = state.connection.execute(
            """
            SELECT status, audit_protocol_sha256, summary_sha256
            FROM graph_quality_audits WHERE snapshot_id = ?
            """,
            (manifest["snapshot_id"],),
        ).fetchone()
        if (
            state.head_snapshot_id != manifest["snapshot_id"]
            or state.embedding_endpoint != snapshot_embedding["endpoint"]
            or state.generation_endpoint != generation_endpoint
            or snapshot_row
            != (
                "COMMITTED",
                manifest["self_sha256"],
                len(source.workflows),
                len(partition.groups),
                len(graph.nodes),
                len(graph.edges),
            )
            or stored != statuses
            or quality_state
            != (
                graph_quality["status"],
                graph_quality["protocol_sha256"],
                _sha(graph_quality_bytes),
            )
        ):
            raise ValueError("state database is not bound to the final snapshot")
    identity = {
        "format": manifest["format"],
        "method": manifest["method"],
        "snapshot_id": manifest["snapshot_id"],
        "manifest_sha256": manifest["self_sha256"],
        "accumulated_source_sha256": source.sha256,
        "canonical_partition_sha256": partition.sha256,
        "experience_graph_sha256": graph.experience_graph_sha256,
        "cumulative_source_audit_sha256": manifest["cumulative_source_audit_sha256"],
        "canonical_audit_sha256": manifest["canonical_audit_sha256"],
        "graph_quality_status": graph_quality["status"],
        "graph_quality_audit_sha256": manifest["graph_quality_audit_sha256"],
        "graph_quality_protocol_sha256": manifest[
            "graph_quality_protocol_sha256"
        ],
        "embedding_endpoint": snapshot_embedding["endpoint"],
        "generation_endpoint": generation_endpoint,
    }
    return _SnapshotContext(source, partition, graph, identity, statuses)


def _request_identity(
    *, stage: str,
    prompt_sha256: str,
    protocol: Mapping[str, Any],
    payload: Mapping[str, Any],
    response_schema: Mapping[str, Any],
) -> tuple[str, str]:
    protocol_sha = _sha(canonical_json_bytes(dict(protocol)))
    request_sha = _sha(
        canonical_json_bytes(
            {
                "stage": stage,
                "prompt_sha256": prompt_sha256,
                "producer_protocol_sha256": protocol_sha,
                "payload": dict(payload),
                "response_schema": dict(response_schema),
            }
        )
    )
    return request_sha, protocol_sha


async def _validated_llm_job(
    *,
    state: IncrementalStateStore,
    llm: Any,
    stage: str,
    prompt_sha256: str,
    kind: str,
    request_id: str,
    system_prompt: str,
    payload: Mapping[str, Any],
    response_schema: Mapping[str, Any],
    parser: Any,
) -> tuple[Any | None, dict[str, Any]]:
    request_sha, protocol_sha = _request_identity(
        stage=stage,
        prompt_sha256=prompt_sha256,
        protocol=llm.protocol_identity,
        payload=payload,
        response_schema=response_schema,
    )
    cached = state.get_retrieval_job(
        request_sha,
        stage=stage,
        prompt_sha256=prompt_sha256,
        producer_protocol_sha256=protocol_sha,
    )
    if cached is not None:
        response, stored_audit = cached
        return parser(response), {**stored_audit, "cache_hit": True}
    errors: list[str] = []
    for attempt in range(1, SEMANTIC_ATTEMPTS + 1):
        try:
            response = await llm.complete_json_async(
                kind=kind,
                request_id=request_id,
                system_prompt=system_prompt,
                payload=payload,
                response_schema=response_schema,
            )
        except RequestContextLengthExceeded as exc:
            return None, {
                "request_sha256": request_sha,
                "producer_protocol_sha256": protocol_sha,
                "attempts": attempt,
                "cache_hit": False,
                "failure": "CONTEXT_LENGTH_EXCEEDED",
                "errors": [f"{type(exc).__name__}: {exc}"],
            }
        except (RequestRuntimeTimeout, APIError) as exc:
            return None, {
                "request_sha256": request_sha,
                "producer_protocol_sha256": protocol_sha,
                "attempts": attempt - 1,
                "cache_hit": False,
                "failure": f"{stage}_FAILED",
                "failure_kind": "TRANSPORT_EXHAUSTED",
                "errors": [f"{type(exc).__name__}: {exc}"],
            }
        except RequestCompletionLengthExceeded as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            continue
        try:
            parsed = parser(response)
        except ValueError as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            continue
        audit = {
            "request_sha256": request_sha,
            "producer_protocol_sha256": protocol_sha,
            "attempts": attempt,
            "cache_hit": False,
        }
        state.put_retrieval_job(
            request_sha,
            stage=stage,
            prompt_sha256=prompt_sha256,
            producer_protocol_sha256=protocol_sha,
            response=response,
            audit=audit,
        )
        return parsed, audit
    return None, {
        "request_sha256": request_sha,
        "producer_protocol_sha256": protocol_sha,
        "attempts": SEMANTIC_ATTEMPTS,
        "cache_hit": False,
        "failure": f"{stage}_FAILED",
        "errors": errors,
    }


def _embedding_texts(
    *,
    train_queries: Sequence[Mapping[str, Any]],
    tasks: Sequence[_Task],
    needs: Mapping[int, NeedGraph],
    need_documents_by_query: Mapping[int, Mapping[int, str]],
    train_workflow_documents: Mapping[int, Mapping[int, str]],
    target_workflow_documents: Mapping[int, str],
    context: _SnapshotContext,
) -> tuple[str, ...]:
    texts: set[str] = {
        *(str(row["instruction"]) for row in train_queries),
        *(task.instruction for task in tasks),
        *(
            document
            for documents in train_workflow_documents.values()
            for document in documents.values()
        ),
        *target_workflow_documents.values(),
    }
    for query_index, need_graph in needs.items():
        documents = need_documents_by_query.get(query_index, {})
        for need_index, need in enumerate(need_graph.nodes):
            texts.add(documents.get(need_index, need_retrieval_document(need)))
    for node in context.graph.nodes:
        texts.add(canonical_retrieval_document(node.experience))
    return tuple(sorted({normalize_embedding_text(text) for text in texts}))


def _candidate_audit(
    *,
    need_graph: NeedGraph,
    workflow_recalls: Sequence[Any],
    search_result: Any,
) -> dict[str, Any]:
    return {
        "need_graph": need_graph.to_dict(),
        "workflow_recalls": [row.to_dict() for row in workflow_recalls],
        "need_candidates": {
            str(index): [row.to_dict() for row in rows]
            for index, rows in sorted(search_result.need_candidates.items())
        },
        "candidate_region_node_ids": list(search_result.region_node_ids),
        "subgraph_candidates": [row.to_dict() for row in search_result.candidates],
        "search_metrics": dict(search_result.metrics),
    }


def _validate_clarification_wave(
    rows: Sequence[tuple[int, RetrievalClarificationDecision, Mapping[str, Any]]],
) -> None:
    """Fail only when the whole clarification producer is unreachable."""
    attempted = [
        audit
        for _index, _decision, audit in rows
        if int(audit.get("attempts", 0)) > 0
        or audit.get("failure_kind") == "TRANSPORT_EXHAUSTED"
    ]
    if attempted and all(
        audit.get("failure_kind") == "TRANSPORT_EXHAUSTED"
        for audit in attempted
    ):
        raise RuntimeError(
            "systemic retrieval-clarification transport outage: "
            "no attempted item reached the producer"
        )


async def _build_tasks_clean_async(
    *,
    tasks: Sequence[_Task],
    train_queries: Sequence[Mapping[str, Any]],
    context: _SnapshotContext,
    state: IncrementalStateStore,
    embedder: StrictEmbeddingAdapter,
    need_llm: Any,
    clarification_llm: Any,
    selector_llm: Any,
    target_evidence_cards: Mapping[int, TargetEvidenceCard],
    train_evidence_cards: Mapping[int, TargetEvidenceCard],
) -> tuple[list[NeedGuidedRetrievalResult], dict[str, Any]]:
    """Build the complete DEGS retrieval result directly from the graph.

    NeedGraph production, evidence-constrained clarification, and workbook-role
    late fusion are implementation stages of one method.  They deliberately use
    the original request identities so verified cache rows remain reusable.
    """
    semaphore = asyncio.Semaphore(LLM_ASYNC_WORKERS)

    async def need_job(task: _Task) -> tuple[int, NeedGraph | None, dict[str, Any]]:
        async with semaphore:
            graph, audit = await _validated_llm_job(
                state=state,
                llm=need_llm,
                stage="NEED_GRAPH",
                prompt_sha256=NEED_GRAPH_PROMPT_SHA256,
                kind=NEED_GRAPH_KIND,
                request_id=f"need-{task.task_id}",
                system_prompt=NEED_GRAPH_SYSTEM_PROMPT,
                payload={"query": task.instruction},
                response_schema=need_graph_response_schema(),
                parser=parse_need_graph,
            )
        return task.query_index, graph, audit

    need_rows = await asyncio.gather(*(need_job(task) for task in tasks))
    needs = {
        index: graph for index, graph, _audit in need_rows if graph is not None
    }
    need_audits = {index: audit for index, _graph, audit in need_rows}

    async def clarification_job(
        task: _Task,
    ) -> tuple[int, RetrievalClarificationDecision, dict[str, Any]]:
        need_graph = needs.get(task.query_index)
        card = target_evidence_cards[task.query_index]
        if need_graph is None or card.status != "OK":
            reason = (
                "NeedGraph unavailable"
                if need_graph is None
                else "target evidence unavailable"
            )
            return (
                task.query_index,
                RetrievalClarificationDecision("KEEP", ()),
                {
                    "attempts": 0,
                    "cache_hit": False,
                    "failure": reason,
                    "decision": "KEEP",
                    "clarifications": [],
                },
            )
        if not clarifiable_need_indices(need_graph):
            return (
                task.query_index,
                RetrievalClarificationDecision("KEEP", ()),
                {
                    "attempts": 0,
                    "cache_hit": False,
                    "failure": "no representation-ambiguous NeedNode",
                    "decision": "KEEP",
                    "clarifications": [],
                },
            )
        parser = lambda value: parse_retrieval_clarification(
            value,
            need_graph=need_graph,
            evidence_ids=card.evidence_ids,
        )
        async with semaphore:
            decision, audit = await _validated_llm_job(
                state=state,
                llm=clarification_llm,
                # The cache key also contains prompt, protocol, and payload
                # hashes, so sharing the historical stage name is unambiguous.
                stage="NEED_GRAPH",
                prompt_sha256=CLARIFICATION_PROMPT_SHA256,
                kind=CLARIFICATION_KIND,
                request_id=f"retrieval-clarification-{task.task_id}",
                system_prompt=CLARIFICATION_SYSTEM_PROMPT,
                payload={
                    "query": task.instruction,
                    "frozen_need_graph": need_graph.to_dict(),
                    **card.llm_payload(),
                },
                response_schema=clarification_response_schema(
                    need_graph, evidence_ids=card.evidence_ids
                ),
                parser=parser,
            )
        if decision is None:
            decision = RetrievalClarificationDecision("KEEP", ())
        if audit.get("failure") == "NEED_GRAPH_FAILED":
            audit = {**audit, "failure": "RETRIEVAL_CLARIFICATION_FAILED"}
        return (
            task.query_index,
            decision,
            {
                **audit,
                "decision": decision.decision,
                "clarifications": [
                    row.to_dict() for row in decision.clarifications
                ],
            },
        )

    clarification_rows = await asyncio.gather(
        *(clarification_job(task) for task in tasks)
    )
    _validate_clarification_wave(clarification_rows)
    clarification_decisions = {
        index: decision for index, decision, _audit in clarification_rows
    }
    clarification_audits = {
        index: audit for index, _decision, audit in clarification_rows
    }

    need_documents_by_query: dict[int, dict[int, str]] = {}
    for query_index, decision in clarification_decisions.items():
        need_graph = needs.get(query_index)
        if need_graph is None:
            continue
        need_documents_by_query[query_index] = {
            need_index: need_retrieval_document(
                need_graph.nodes[need_index],
                target_evidence_cards[query_index].retrieval_clarification(
                    evidence_ids
                ),
            )
            for need_index, evidence_ids in decision.by_need_node.items()
        }

    target_workflow_evidence_ids = {
        task.query_index: tuple(
            sorted(
                {
                    evidence_id
                    for row in clarification_decisions[
                        task.query_index
                    ].clarifications
                    for evidence_id in row.evidence_ids
                }
            )
        )
        for task in tasks
        if clarification_decisions[task.query_index].decision == "CLARIFY"
    }
    target_workflow_documents = {
        task.query_index: target_evidence_cards[
            task.query_index
        ].workflow_context_document(
            task.instruction,
            evidence_ids=target_workflow_evidence_ids[task.query_index],
        )
        for task in tasks
        if task.query_index in target_workflow_evidence_ids
    }
    target_workflow_kinds = {
        query_index: {
            row["kind"]
            for row in target_evidence_cards[query_index].observations
            if row["evidence_id"] in evidence_ids
        }
        for query_index, evidence_ids in target_workflow_evidence_ids.items()
    }
    train_workflow_documents = {
        query_index: {
            int(row["train_index"]): _workflow_document_for_kinds(
                card=train_evidence_cards[int(row["train_index"])],
                instruction=str(row["instruction"]),
                kinds=kinds,
            )
            for row in train_queries
        }
        for query_index, kinds in target_workflow_kinds.items()
    }

    texts = _embedding_texts(
        train_queries=train_queries,
        tasks=tasks,
        needs=needs,
        need_documents_by_query=need_documents_by_query,
        train_workflow_documents=train_workflow_documents,
        target_workflow_documents=target_workflow_documents,
        context=context,
    )
    cache_before = len(state.embedding_cache())
    embedded = await embedder.embed_async(texts, workers=EMBEDDING_ASYNC_WORKERS)
    cache_after = len(state.embedding_cache())
    vectors = {row.normalized_text: row.vector for row in embedded}
    train_vectors = {
        int(row["train_index"]): vectors[
            normalize_embedding_text(str(row["instruction"]))
        ]
        for row in train_queries
    }
    train_workflow_vectors = {
        query_index: {
            train_index: vectors[normalize_embedding_text(document)]
            for train_index, document in documents.items()
        }
        for query_index, documents in train_workflow_documents.items()
    }
    retrieval_index = build_retrieval_index(
        context.graph, context.partition, context.source, vectors=vectors
    )

    normal_prepared: dict[int, tuple[Any, ...]] = {}
    fallback_prepared: dict[int, tuple[Any, ...]] = {}
    results: list[NeedGuidedRetrievalResult | None] = [None] * len(tasks)
    train_evidence_sha256 = _evidence_cards_sha256(train_evidence_cards)

    def workflow_context_audit(index: int) -> dict[str, Any]:
        document = target_workflow_documents.get(index)
        if target_evidence_cards[index].status != "OK":
            mode = "TARGET_EVIDENCE_UNAVAILABLE_QUERY_ONLY"
        elif document is not None:
            mode = "WORKBOOK_CONDITIONED_LATE_FUSION"
        else:
            mode = "QUERY_ONLY"
        return {
            "format": WORKFLOW_CONTEXT_DOCUMENT_FORMAT,
            "mode": mode,
            "target_document_sha256": (
                _sha(normalize_embedding_text(document).encode("utf-8"))
                if document is not None
                else None
            ),
            "train_evidence_sha256": train_evidence_sha256,
        }

    for task in tasks:
        all_recalls = recall_source_workflows(
            vectors[normalize_embedding_text(task.instruction)],
            train_vectors,
            top_k=len(train_vectors),
        )
        raw_recalls = all_recalls[:WORKFLOW_RECALL_K]
        if task.query_index in target_workflow_documents:
            context_recalls = recall_source_workflows(
                vectors[
                    normalize_embedding_text(
                        target_workflow_documents[task.query_index]
                    )
                ],
                train_workflow_vectors[task.query_index],
                top_k=len(train_vectors),
            )
            workflow_context_similarities = {
                row.train_index: row.similarity for row in context_recalls
            }
        else:
            workflow_context_similarities = {
                row.train_index: row.similarity for row in all_recalls
            }
        need_graph = needs.get(task.query_index)
        if need_graph is None:
            fallback_recalls = tuple(
                row
                for row in all_recalls
                if row.train_index in retrieval_index.canonical_ids_by_workflow
            )[:WORKFLOW_RECALL_K]
            candidates = workflow_fallback_candidates(
                fallback_recalls,
                canonical_ids_by_workflow=(
                    retrieval_index.canonical_ids_by_workflow
                ),
            )
            payload = workflow_fallback_selector_payload(
                query_text=task.instruction,
                candidates=candidates,
                graph=context.graph,
                source=context.source,
                fallback_mode="NEED_GRAPH_UNAVAILABLE",
            )
            fallback_prepared[task.query_index] = (
                task,
                candidates,
                payload,
                "NEED_GRAPH_FALLBACK_WORKFLOW",
                "NeedGraph generation was unavailable",
                {},
            )
            continue
        search_result = retrieve_experience_subgraphs(
            need_graph,
            raw_recalls,
            index=retrieval_index,
            vectors=vectors,
            workflow_context_similarities=workflow_context_similarities,
            need_documents=need_documents_by_query.get(task.query_index),
        )
        candidate_audit = _candidate_audit(
            need_graph=need_graph,
            workflow_recalls=raw_recalls,
            search_result=search_result,
        )
        if not search_result.candidates:
            fallback_recalls = tuple(
                row
                for row in all_recalls
                if row.train_index in retrieval_index.canonical_ids_by_workflow
            )[:WORKFLOW_RECALL_K]
            candidates = workflow_fallback_candidates(
                fallback_recalls,
                canonical_ids_by_workflow=(
                    retrieval_index.canonical_ids_by_workflow
                ),
            )
            payload = workflow_fallback_selector_payload(
                query_text=task.instruction,
                candidates=candidates,
                graph=context.graph,
                source=context.source,
                fallback_mode="SOFT_SUBGRAPH_SEARCH_EMPTY",
            )
            fallback_prepared[task.query_index] = (
                task,
                candidates,
                payload,
                "SEARCH_FALLBACK_WORKFLOW",
                "soft subgraph search returned no candidate",
                candidate_audit,
            )
            continue
        normal_prepared[task.query_index] = (
            task,
            need_graph,
            raw_recalls,
            search_result,
            candidate_audit,
        )

    async def normal_selector_job(
        index: int, row: tuple[Any, ...]
    ) -> tuple[int, NeedGuidedRetrievalResult]:
        task, need_graph, recalls, search_result, candidate_audit = row
        selection = SubgraphSelection(search_result.candidates[0].candidate_id)
        selected_result = finalize_subgraph_selection(
            selection,
            query_text=task.instruction,
            need_graph=need_graph,
            workflow_recalls=recalls,
            search_result=search_result,
            graph=context.graph,
            source=context.source,
        )
        status = selected_result.status
        return index, NeedGuidedRetrievalResult(
            status,
            selected_result.experience,
            {
                **dict(selected_result.audit),
                **candidate_audit,
                "status": status,
                "need_generation": need_audits[index],
                "target_evidence": target_evidence_cards[index].audit_payload(),
                "retrieval_clarification": clarification_audits[index],
                "workflow_context": workflow_context_audit(index),
                "selector": {
                    "llm_called": False,
                    "policy": "DETERMINISTIC_TOP_RANKED_C0",
                    "selected_candidate_id": selection.selected_candidate_id,
                },
                "selector_fallback_candidate_id": None,
            },
        )

    async def fallback_selector_job(
        index: int, row: tuple[Any, ...]
    ) -> tuple[int, NeedGuidedRetrievalResult]:
        task, candidates, payload, status, reason, extra_audit = row
        parser = lambda value: parse_selector_response(value, candidates=candidates)
        async with semaphore:
            selection, selector_audit = await _validated_llm_job(
                state=state,
                llm=selector_llm,
                stage="SELECTOR",
                prompt_sha256=SELECTOR_PROMPT_SHA256,
                kind=SELECTOR_KIND,
                request_id=f"workflow-fallback-selector-{task.task_id}",
                system_prompt=SELECTOR_SYSTEM_PROMPT,
                payload=payload,
                response_schema=selector_response_schema(candidates),
                parser=parser,
            )
        fallback_used = selection is None
        if selection is None:
            selection = SubgraphSelection(candidates[0].candidate_id)
        result = render_workflow_fallback(
            selection=selection,
            query_text=task.instruction,
            candidates=candidates,
            source=context.source,
            graph=context.graph,
            status=status,
            reason=reason,
        )
        return index, NeedGuidedRetrievalResult(
            result.status,
            result.experience,
            {
                **dict(result.audit),
                **extra_audit,
                "need_generation": need_audits[index],
                "target_evidence": target_evidence_cards[index].audit_payload(),
                "retrieval_clarification": clarification_audits[index],
                "workflow_context": workflow_context_audit(index),
                "selector": selector_audit,
                "selector_fallback_candidate_id": (
                    selection.selected_candidate_id if fallback_used else None
                ),
            },
        )

    normal_selected, fallback_selected = await asyncio.gather(
        asyncio.gather(
            *(
                normal_selector_job(index, row)
                for index, row in sorted(normal_prepared.items())
            )
        ),
        asyncio.gather(
            *(
                fallback_selector_job(index, row)
                for index, row in sorted(fallback_prepared.items())
            )
        ),
    )
    for index, result in (*normal_selected, *fallback_selected):
        results[index] = result
    if any(result is None for result in results):
        raise RuntimeError("bundle task orchestration left an unresolved task")
    embedding_audit = {
        "endpoint": getattr(embedder.transport, "endpoint", "injected"),
        "model": EMBEDDING_MODEL,
        "batch_size": EMBEDDING_BATCH_SIZE,
        "timeout_s": EMBEDDING_TIMEOUT_S,
        "cache": "normalized_text_sha256",
        "unique_normalized_text_count": len(texts),
        "cache_hit_count": len(texts) - (cache_after - cache_before),
        "cache_miss_count": cache_after - cache_before,
        "api_request_count": (
            cache_after - cache_before + EMBEDDING_BATCH_SIZE - 1
        )
        // EMBEDDING_BATCH_SIZE,
        "async_workers": EMBEDDING_ASYNC_WORKERS,
    }
    return [result for result in results if result is not None], embedding_audit


def _rows(
    *,
    tasks: Sequence[_Task],
    results: Sequence[NeedGuidedRetrievalResult],
    authority_sha256: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for task, result in zip(tasks, results, strict=True):
        if result.status not in _RESULT_STATUSES:
            raise ValueError("retrieval result status differs")
        if not result.experience:
            raise ValueError("every retrieval status must inject experience")
        audit = dict(result.audit)
        audit_sha = _sha(canonical_json_bytes(audit))
        metadata = {
            "format": EXPERIENCE_FORMAT,
            "method_family": METHOD_FAMILY,
            "status": result.status,
            "query_index": task.query_index,
            "dataset_index": task.dataset_index,
            "authority_sha256": authority_sha256,
            "experience_sha256": _sha(result.experience.encode("utf-8")),
            "retrieval_audit_sha256": audit_sha,
            "retrieval_audit": audit,
        }
        row = {
            "instance_id": task.task_id,
            "experience": result.experience,
            "metadata": metadata,
        }
        row_sha = _sha(canonical_json_bytes(row))
        rows.append(row)
        audits.append(
            {
                "query_index": task.query_index,
                "dataset_index": task.dataset_index,
                "task_id": task.task_id,
                "status": result.status,
                "row_sha256": row_sha,
                "retrieval_audit_sha256": audit_sha,
            }
        )
    return rows, audits


def _valid_clean_retrieval_audit(
    audit: Mapping[str, Any],
    *,
    card: TargetEvidenceCard,
    train_evidence_sha256: str,
    instruction: str,
) -> bool:
    if (
        audit.get("target_evidence") != card.audit_payload()
        or type(audit.get("need_generation")) is not dict
    ):
        return False
    raw_need = audit.get("need_graph")
    clarification = audit.get("retrieval_clarification")
    workflow_context = audit.get("workflow_context")
    if type(clarification) is not dict or type(workflow_context) is not dict:
        return False
    if (
        set(workflow_context)
        != {
            "format",
            "mode",
            "target_document_sha256",
            "train_evidence_sha256",
        }
        or workflow_context.get("format") != WORKFLOW_CONTEXT_DOCUMENT_FORMAT
        or workflow_context.get("train_evidence_sha256")
        != train_evidence_sha256
    ):
        return False
    decision = clarification.get("decision")
    rows = clarification.get("clarifications")
    if type(raw_need) is not dict:
        return (
            decision == "KEEP"
            and rows == []
            and workflow_context.get("target_document_sha256") is None
            and workflow_context.get("mode")
            in {"QUERY_ONLY", "TARGET_EVIDENCE_UNAVAILABLE_QUERY_ONLY"}
        )
    try:
        need_graph = parse_need_graph(raw_need)
        if card.status == "OK":
            parsed = parse_retrieval_clarification(
                {"clarifications": rows},
                need_graph=need_graph,
                evidence_ids=card.evidence_ids,
            )
        else:
            if decision != "KEEP" or rows != []:
                return False
            parsed = RetrievalClarificationDecision("KEEP", ())
    except (TypeError, ValueError):
        return False
    if parsed.decision != decision or [
        row.to_dict() for row in parsed.clarifications
    ] != rows:
        return False
    if parsed.decision == "KEEP":
        expected_mode = (
            "QUERY_ONLY"
            if card.status == "OK"
            else "TARGET_EVIDENCE_UNAVAILABLE_QUERY_ONLY"
        )
        return (
            workflow_context.get("mode") == expected_mode
            and workflow_context.get("target_document_sha256") is None
        )
    evidence_ids = tuple(
        sorted(
            {
                evidence_id
                for row in parsed.clarifications
                for evidence_id in row.evidence_ids
            }
        )
    )
    document = card.workflow_context_document(
        instruction,
        evidence_ids=evidence_ids,
    )
    return (
        workflow_context.get("mode") == "WORKBOOK_CONDITIONED_LATE_FUSION"
        and workflow_context.get("target_document_sha256")
        == _sha(normalize_embedding_text(document).encode("utf-8"))
    )


async def _build_clean_async(
    *,
    dataset_path: Path,
    snapshot_manifest_path: Path,
    state_db_path: Path,
    output_dir: Path,
    embedding_transport: Any,
    need_llm: Any,
    clarification_llm: Any,
    selector_llm: Any,
) -> _VerifiedExperienceBundle:
    producer_specs = (
        (
            "need_producer",
            need_llm,
            NEED_GRAPH_PROTOCOL_FORMAT,
            NEED_GRAPH_KIND,
            NEED_GRAPH_PROMPT_SHA256,
        ),
        (
            "clarification_producer",
            clarification_llm,
            CLARIFICATION_PROTOCOL_FORMAT,
            CLARIFICATION_KIND,
            CLARIFICATION_PROMPT_SHA256,
        ),
        (
            "selector_producer",
            selector_llm,
            SELECTOR_PROTOCOL_FORMAT,
            SELECTOR_KIND,
            SELECTOR_PROMPT_SHA256,
        ),
    )
    protocols = {
        name: _validated_llm_protocol(
            llm.protocol_identity,
            expected_format=format_id,
            expected_kind=kind,
            expected_prompt_sha256=prompt_sha,
            expected_retry_waits=BUNDLE_TRANSPORT_RETRY_WAITS,
            expected_runtime_timeout_retries=BUNDLE_RUNTIME_TIMEOUT_RETRIES,
        )
        for name, llm, format_id, kind, prompt_sha in producer_specs
    }
    if len({row["service_url"] for row in protocols.values()}) != 1:
        raise ValueError("retrieval producers must use the same LLM service")

    train_queries = load_train_queries(dataset_path)
    train_records = _load_train_harness_records(dataset_path)
    development = _load_development_harness_records(dataset_path)
    if query_projection_sha256(train_queries) != EXPECTED_QUERY_PROJECTION_SHA256:
        raise ValueError("train query projection differs")
    if (
        development_query_projection_sha256(_query_fields(development))
        != EXPECTED_DEVELOPMENT_QUERY_PROJECTION_SHA256
    ):
        raise ValueError("development query projection differs")
    authority = seal_train_instruction_authority(
        [
            {"id": row["task_id"], "instruction": row["instruction"]}
            for row in train_queries
        ]
    ).authority
    tasks = _task_plan(development)
    target_evidence_cards = _target_evidence_cards(tasks, dataset_path=dataset_path)
    train_evidence_cards = _train_evidence_cards(
        train_records, dataset_path=dataset_path
    )
    context = _load_snapshot(
        snapshot_manifest_path,
        state_db_path=state_db_path,
        train_queries=train_queries,
    )
    generation_endpoint = next(iter(protocols.values()))["service_url"].rstrip("/")
    if generation_endpoint != context.identity["generation_endpoint"]:
        raise ValueError("retrieval generation endpoint differs from graph producers")

    output = output_dir.expanduser().absolute()
    if output.exists() and not output.is_dir():
        raise FileExistsError("bundle output directory must be fresh")
    if output.exists():
        try:
            output.rmdir()
        except OSError as exc:
            raise FileExistsError(
                "bundle output directory must be fresh or empty"
            ) from exc
    output.parent.mkdir(parents=True, exist_ok=True)
    with IncrementalStateStore(state_db_path) as state:
        state.bind_embedding_endpoint(getattr(embedding_transport, "endpoint", ""))
        embedder = StrictEmbeddingAdapter(
            embedding_transport, cache=state.embedding_cache()
        )
        if context.identity["embedding_endpoint"] != state.embedding_endpoint:
            raise ValueError("bundle embedding endpoint differs from the graph")
        results, embedding_audit = await _build_tasks_clean_async(
            tasks=tasks,
            train_queries=train_queries,
            context=context,
            state=state,
            embedder=embedder,
            need_llm=need_llm,
            clarification_llm=clarification_llm,
            selector_llm=selector_llm,
            target_evidence_cards=target_evidence_cards,
            train_evidence_cards=train_evidence_cards,
        )

    rows, task_audits = _rows(
        tasks=tasks,
        results=results,
        authority_sha256=authority.authority_sha256,
    )
    experience_bytes = b"".join(
        canonical_json_bytes(row) + b"\n" for row in rows
    )
    llm_protocol = {
        "model": REPAIR_SOURCE_MODEL,
        "temperature": REPAIR_SOURCE_TEMPERATURE,
        "thinking": REPAIR_SOURCE_THINKING,
        "max_tokens": REPAIR_SOURCE_MAX_TOKENS,
        "timeout_seconds": REPAIR_SOURCE_TIMEOUT_SECONDS,
        "semantic_attempts": SEMANTIC_ATTEMPTS,
        "async_workers": LLM_ASYNC_WORKERS,
        "need_prompt_sha256": NEED_GRAPH_PROMPT_SHA256,
        "clarification_prompt_sha256": CLARIFICATION_PROMPT_SHA256,
        "selector_prompt_sha256": SELECTOR_PROMPT_SHA256,
        **protocols,
    }
    body = {
        "format": FORMAT,
        "method_family": METHOD_FAMILY,
        "method_name": METHOD_NAME,
        "method_version": METHOD_VERSION,
        "method_contract": copy.deepcopy(METHOD_CONTRACT),
        "graph_source_method": INCREMENTAL_METHOD_ID,
        "semantic_recall_document_format": SEMANTIC_RECALL_DOCUMENT_FORMAT,
        "recall_score_format": RECALL_SCORE_FORMAT,
        "claim_scope": CLAIM_SCOPE,
        "fixed_denominator": DEVELOPMENT_COUNT,
        "authority_sha256": authority.authority_sha256,
        "train_query_projection_sha256": EXPECTED_QUERY_PROJECTION_SHA256,
        "development_query_projection_sha256": (
            EXPECTED_DEVELOPMENT_QUERY_PROJECTION_SHA256
        ),
        "workflow_context_document_format": WORKFLOW_CONTEXT_DOCUMENT_FORMAT,
        "train_workflow_evidence_sha256": _evidence_cards_sha256(
            train_evidence_cards
        ),
        "snapshot": dict(context.identity),
        "experience_graph": context.graph.identity(),
        "source_status_counts": dict(
            sorted(Counter(context.source_statuses.values()).items())
        ),
        "embedding_protocol": embedding_audit,
        "llm_protocol": llm_protocol,
        "tasks": task_audits,
        "experience_file": "experience.jsonl",
        "experience_sha256": _sha(experience_bytes),
        "row_count": len(rows),
        "status_counts": dict(
            sorted(Counter(result.status for result in results).items())
        ),
        "runtime": _runtime_identity(),
    }
    manifest = {**body, "self_sha256": _sha(canonical_json_bytes(body))}
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    published = False
    try:
        _write_exclusive(staging / "experience.jsonl", experience_bytes)
        _write_exclusive(
            staging / "bundle_manifest.json", canonical_json_bytes(manifest)
        )
        os.replace(staging, output)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)
    return verify_from_paths(
        dataset_path=dataset_path,
        snapshot_manifest_path=snapshot_manifest_path,
        state_db_path=state_db_path,
        output_dir=output,
    )


def build_from_paths(
    *,
    dataset_path: Path,
    snapshot_manifest_path: Path,
    state_db_path: Path,
    output_dir: Path,
    embedding_transport: Any,
    need_llm: Any,
    clarification_llm: Any,
    selector_llm: Any,
) -> _VerifiedExperienceBundle:
    return asyncio.run(
        _build_clean_async(
            dataset_path=dataset_path,
            snapshot_manifest_path=snapshot_manifest_path,
            state_db_path=state_db_path,
            output_dir=output_dir,
            embedding_transport=embedding_transport,
            need_llm=need_llm,
            clarification_llm=clarification_llm,
            selector_llm=selector_llm,
        )
    )


def verify_from_paths(
    *,
    dataset_path: Path,
    snapshot_manifest_path: Path,
    state_db_path: Path,
    output_dir: Path,
) -> _VerifiedExperienceBundle:
    train_queries = load_train_queries(dataset_path)
    train_records = _load_train_harness_records(dataset_path)
    development = _load_development_harness_records(dataset_path)
    if query_projection_sha256(train_queries) != EXPECTED_QUERY_PROJECTION_SHA256:
        raise ValueError("train query projection differs")
    if (
        development_query_projection_sha256(_query_fields(development))
        != EXPECTED_DEVELOPMENT_QUERY_PROJECTION_SHA256
    ):
        raise ValueError("development query projection differs")
    authority = seal_train_instruction_authority(
        [
            {"id": row["task_id"], "instruction": row["instruction"]}
            for row in train_queries
        ]
    ).authority
    tasks = _task_plan(development)
    target_evidence_cards = _target_evidence_cards(tasks, dataset_path=dataset_path)
    train_evidence_cards = _train_evidence_cards(
        train_records, dataset_path=dataset_path
    )
    context = _load_snapshot(
        snapshot_manifest_path,
        state_db_path=state_db_path,
        train_queries=train_queries,
    )
    root = output_dir.expanduser().absolute()
    if (
        not root.is_dir()
        or {path.name for path in root.iterdir()}
        != {"bundle_manifest.json", "experience.jsonl"}
    ):
        raise ValueError("bundle paths differ")
    manifest, manifest_bytes = _read_canonical_json(
        root / "bundle_manifest.json", label="bundle manifest"
    )
    unsigned = {key: value for key, value in manifest.items() if key != "self_sha256"}
    experience_bytes = (root / "experience.jsonl").read_bytes()
    lines = experience_bytes.splitlines()
    llm_protocol = manifest.get("llm_protocol")
    if type(llm_protocol) is not dict:
        raise ValueError("bundle retrieval LLM protocol differs")
    try:
        need_protocol = _validated_llm_protocol(
            llm_protocol["need_producer"],
            expected_format=NEED_GRAPH_PROTOCOL_FORMAT,
            expected_kind=NEED_GRAPH_KIND,
            expected_prompt_sha256=NEED_GRAPH_PROMPT_SHA256,
            expected_retry_waits=BUNDLE_TRANSPORT_RETRY_WAITS,
            expected_runtime_timeout_retries=BUNDLE_RUNTIME_TIMEOUT_RETRIES,
        )
        clarification_protocol = _validated_llm_protocol(
            llm_protocol["clarification_producer"],
            expected_format=CLARIFICATION_PROTOCOL_FORMAT,
            expected_kind=CLARIFICATION_KIND,
            expected_prompt_sha256=CLARIFICATION_PROMPT_SHA256,
            expected_retry_waits=BUNDLE_TRANSPORT_RETRY_WAITS,
            expected_runtime_timeout_retries=BUNDLE_RUNTIME_TIMEOUT_RETRIES,
        )
        selector_protocol = _validated_llm_protocol(
            llm_protocol["selector_producer"],
            expected_format=SELECTOR_PROTOCOL_FORMAT,
            expected_kind=SELECTOR_KIND,
            expected_prompt_sha256=SELECTOR_PROMPT_SHA256,
            expected_retry_waits=BUNDLE_TRANSPORT_RETRY_WAITS,
            expected_runtime_timeout_retries=BUNDLE_RUNTIME_TIMEOUT_RETRIES,
        )
    except (AttributeError, KeyError, TypeError, ValueError):
        raise ValueError("bundle retrieval LLM protocol differs") from None
    expected_llm_protocol = {
        "model": REPAIR_SOURCE_MODEL,
        "temperature": REPAIR_SOURCE_TEMPERATURE,
        "thinking": REPAIR_SOURCE_THINKING,
        "max_tokens": REPAIR_SOURCE_MAX_TOKENS,
        "timeout_seconds": REPAIR_SOURCE_TIMEOUT_SECONDS,
        "semantic_attempts": SEMANTIC_ATTEMPTS,
        "async_workers": LLM_ASYNC_WORKERS,
        "need_prompt_sha256": NEED_GRAPH_PROMPT_SHA256,
        "clarification_prompt_sha256": CLARIFICATION_PROMPT_SHA256,
        "selector_prompt_sha256": SELECTOR_PROMPT_SHA256,
        "need_producer": need_protocol,
        "clarification_producer": clarification_protocol,
        "selector_producer": selector_protocol,
    }
    expected_fields = {
        "format",
        "method_family",
        "method_name",
        "method_version",
        "method_contract",
        "graph_source_method",
        "semantic_recall_document_format",
        "recall_score_format",
        "claim_scope",
        "fixed_denominator",
        "authority_sha256",
        "train_query_projection_sha256",
        "development_query_projection_sha256",
        "workflow_context_document_format",
        "train_workflow_evidence_sha256",
        "snapshot",
        "experience_graph",
        "source_status_counts",
        "embedding_protocol",
        "llm_protocol",
        "tasks",
        "experience_file",
        "experience_sha256",
        "row_count",
        "status_counts",
        "runtime",
        "self_sha256",
    }
    if (
        set(manifest) != expected_fields
        or manifest.get("format") != FORMAT
        or manifest.get("method_family") != METHOD_FAMILY
        or manifest.get("method_name") != METHOD_NAME
        or manifest.get("method_version") != METHOD_VERSION
        or manifest.get("method_contract") != METHOD_CONTRACT
        or manifest.get("graph_source_method") != INCREMENTAL_METHOD_ID
        or manifest.get("semantic_recall_document_format")
        != SEMANTIC_RECALL_DOCUMENT_FORMAT
        or manifest.get("recall_score_format") != RECALL_SCORE_FORMAT
        or manifest.get("claim_scope") != CLAIM_SCOPE
        or manifest.get("fixed_denominator") != DEVELOPMENT_COUNT
        or manifest.get("authority_sha256") != authority.authority_sha256
        or manifest.get("train_query_projection_sha256")
        != EXPECTED_QUERY_PROJECTION_SHA256
        or manifest.get("development_query_projection_sha256")
        != EXPECTED_DEVELOPMENT_QUERY_PROJECTION_SHA256
        or manifest.get("workflow_context_document_format")
        != WORKFLOW_CONTEXT_DOCUMENT_FORMAT
        or manifest.get("train_workflow_evidence_sha256")
        != _evidence_cards_sha256(train_evidence_cards)
        or manifest.get("snapshot") != dict(context.identity)
        or manifest.get("experience_graph") != context.graph.identity()
        or manifest.get("source_status_counts")
        != dict(sorted(Counter(context.source_statuses.values()).items()))
        or not _embedding_protocol_matches(manifest.get("embedding_protocol"))
        or manifest["embedding_protocol"].get("endpoint")
        != context.identity["embedding_endpoint"]
        or llm_protocol != expected_llm_protocol
        or len(
            {
                need_protocol["service_url"],
                clarification_protocol["service_url"],
                selector_protocol["service_url"],
            }
        )
        != 1
        or need_protocol["service_url"].rstrip("/")
        != context.identity["generation_endpoint"]
        or manifest.get("experience_file") != "experience.jsonl"
        or manifest.get("experience_sha256") != _sha(experience_bytes)
        or manifest.get("row_count") != DEVELOPMENT_COUNT
        or manifest.get("self_sha256") != _sha(canonical_json_bytes(unsigned))
        or manifest_bytes != canonical_json_bytes(manifest)
        or experience_bytes != b"".join(line + b"\n" for line in lines)
        or len(lines) != DEVELOPMENT_COUNT
        or type(manifest.get("runtime")) is not dict
    ):
        raise ValueError("bundle identity differs")
    task_audits = manifest.get("tasks")
    train_evidence_sha256 = _evidence_cards_sha256(train_evidence_cards)
    if type(task_audits) is not list or len(task_audits) != DEVELOPMENT_COUNT:
        raise ValueError("bundle task audit differs")
    rows: list[dict[str, Any]] = []
    for index, (line, task, task_audit) in enumerate(
        zip(lines, tasks, task_audits, strict=True)
    ):
        row = _strict_json_bytes(line, label=f"experience row {index}")
        metadata = row.get("metadata") if type(row) is dict else None
        audit = metadata.get("retrieval_audit") if type(metadata) is dict else None
        if (
            type(row) is not dict
            or set(row) != {"instance_id", "experience", "metadata"}
            or line != canonical_json_bytes(row)
            or type(task_audit) is not dict
            or set(task_audit) != _TASK_AUDIT_FIELDS
            or row.get("instance_id") != task.task_id
            or type(row.get("experience")) is not str
            or not row["experience"]
            or type(metadata) is not dict
            or set(metadata) != _EXPERIENCE_METADATA_FIELDS
            or metadata.get("format") != EXPERIENCE_FORMAT
            or metadata.get("method_family") != METHOD_FAMILY
            or metadata.get("query_index") != index
            or metadata.get("dataset_index") != DEVELOPMENT_START + index
            or metadata.get("authority_sha256") != authority.authority_sha256
            or metadata.get("experience_sha256")
            != _sha(row["experience"].encode("utf-8"))
            or type(audit) is not dict
            or metadata.get("retrieval_audit_sha256")
            != _sha(canonical_json_bytes(audit))
            or metadata.get("status") != task_audit.get("status")
            or metadata.get("status") not in _RESULT_STATUSES
            or task_audit.get("task_id") != task.task_id
            or task_audit.get("query_index") != index
            or task_audit.get("dataset_index") != DEVELOPMENT_START + index
            or task_audit.get("row_sha256") != _sha(canonical_json_bytes(row))
            or task_audit.get("retrieval_audit_sha256")
            != metadata.get("retrieval_audit_sha256")
            or not _valid_clean_retrieval_audit(
                audit,
                card=target_evidence_cards[index],
                train_evidence_sha256=train_evidence_sha256,
                instruction=task.instruction,
            )
        ):
            raise ValueError(f"experience row {index} identity differs")
        rows.append(row)
    if dict(
        sorted(Counter(row["metadata"]["status"] for row in rows).items())
    ) != manifest.get("status_counts"):
        raise ValueError("bundle status summary differs")
    return _VerifiedExperienceBundle(_VERIFIED_TOKEN, root, manifest)


def _client(*, api_key: str, base_url: str) -> OpenAIClient:
    return OpenAIClient(
        model=REPAIR_SOURCE_MODEL,
        api_key=api_key,
        base_url=base_url,
        generation_config={
            "temperature": REPAIR_SOURCE_TEMPERATURE,
            "max_tokens": REPAIR_SOURCE_MAX_TOKENS,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": REPAIR_SOURCE_THINKING}},
        },
        retry_times=BUNDLE_TRANSPORT_RETRY_WAITS,
        runtime_timeout_retries=BUNDLE_RUNTIME_TIMEOUT_RETRIES,
        timeout=REPAIR_SOURCE_TIMEOUT_SECONDS,
        trust_env=False,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build or verify the DEGS 0.77.41 Stable R1 bundle."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("build", "verify"):
        child = subparsers.add_parser(command)
        child.add_argument("--dataset-path", type=Path, required=True)
        child.add_argument("--snapshot-manifest-path", type=Path, required=True)
        child.add_argument("--state-db", type=Path, required=True)
        child.add_argument("--output-dir", type=Path, required=True)
        if command == "build":
            child.add_argument("--llm-base-url", required=True)
            child.add_argument("--embedding-base-url", required=True)
            child.add_argument("--llm-api-key-env", default="DEGS_API_KEY")
            child.add_argument("--embedding-api-key-env", default="DEGS_EMBEDDING_API_KEY")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    common = {
        "dataset_path": args.dataset_path,
        "snapshot_manifest_path": args.snapshot_manifest_path,
        "state_db_path": args.state_db,
        "output_dir": args.output_dir,
    }
    if args.command == "build":
        validate_service_url(args.llm_base_url)
        validate_service_url(args.embedding_base_url)
        llm_key = os.environ.get(args.llm_api_key_env)
        embedding_key = os.environ.get(args.embedding_api_key_env)
        if not llm_key or not embedding_key:
            raise ValueError("generation and embedding API keys are required")
        client = _client(api_key=llm_key, base_url=args.llm_base_url)
        verified = build_from_paths(
            **common,
            embedding_transport=QwenEmbeddingHTTPTransport(
                base_url=args.embedding_base_url, api_key=embedding_key
            ),
            need_llm=openai_need_graph_llm(
                client,
                expected_retry_times=BUNDLE_TRANSPORT_RETRY_WAITS,
                expected_runtime_timeout_retries=BUNDLE_RUNTIME_TIMEOUT_RETRIES,
            ),
            clarification_llm=openai_retrieval_clarification_llm(
                client,
                expected_retry_times=BUNDLE_TRANSPORT_RETRY_WAITS,
                expected_runtime_timeout_retries=BUNDLE_RUNTIME_TIMEOUT_RETRIES,
            ),
            selector_llm=openai_selector_llm(
                client,
                expected_retry_times=BUNDLE_TRANSPORT_RETRY_WAITS,
                expected_runtime_timeout_retries=BUNDLE_RUNTIME_TIMEOUT_RETRIES,
            ),
        )
    else:
        verified = verify_from_paths(**common)
    manifest = verified.manifest
    print(json.dumps({"output_dir": str(verified.root), "self_sha256": manifest["self_sha256"], "status_counts": manifest["status_counts"]}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CLAIM_SCOPE",
    "EXPECTED_BUNDLE_RUNTIME",
    "EXPERIENCE_FORMAT",
    "FORMAT",
    "LLM_ASYNC_WORKERS",
    "METHOD_FAMILY",
    "build_from_paths",
    "main",
    "verify_from_paths",
]
