from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Mapping

import pytest
from react_agent.models import (
    RequestCompletionLengthExceeded,
    RequestContextLengthExceeded,
    RequestRuntimeTimeout,
)

import degs
import degs.bundle as bundle_module
import degs.canonicalize as canonicalize_module
import degs.core as core_module
import degs.incremental_graph as incremental_graph_module
import degs.successful_source as successful_source_module
import degs.source_review as source_review_module
import degs.workflow_retrieval as workflow_retrieval_module
import degs.retrieval_clarification as retrieval_clarification_module
from degs.bundle import (
    BUNDLE_RUNTIME_TIMEOUT_RETRIES,
    BUNDLE_TRANSPORT_RETRY_WAITS,
    _load_snapshot,
    build_from_paths,
    verify_from_paths,
)
from degs.dataset import (
    development_query_projection_sha256,
    query_projection_sha256,
)
from degs.provider import DEGSExperienceProvider
from degs.canonicalize import (
    CANONICAL_MERGE_KIND, CANONICAL_MERGE_PROMPT_SHA256, CANONICAL_MERGE_PROTOCOL_FORMAT,
    CANONICAL_VIEW_KIND, CANONICAL_VIEW_PROMPT_SHA256, CANONICAL_VIEW_PROTOCOL_FORMAT,
)

from degs.core import (
    EMBEDDING_MODEL,
    StrictEmbeddingAdapter,
    canonical_json_bytes,
)
from degs.incremental_graph import (
    INCREMENTAL_BATCH_SIZE,
    IncrementalGraphBuilder,
    _validate_committing_snapshot_state,
)
from degs.section_graph import (
    CanonicalPartition,
    CanonicalExperience,
    ExperienceNode,
    IOContract,
    SectionGraphSource,
    WorkflowGraph,
    compile_experience_graph,
    load_canonical_partition,
    load_section_graphs,
)
from degs.source_rebuild import SOURCE_REBUILD_AUDIT_FORMAT, SOURCE_REBUILD_WORKERS
from degs.source_review import (
    SOURCE_REVIEW_KIND,
    SOURCE_REVIEW_PROMPT_SHA256,
    SOURCE_REVIEW_PROTOCOL_FORMAT,
    source_review_response_schema,
)
from degs.state_store import INCREMENTAL_METHOD_ID, IncrementalStateStore
from degs.target_context import TargetEvidenceCard
from degs.retrieval_clarification import (
    CLARIFICATION_KIND,
    CLARIFICATION_PROMPT_SHA256,
    CLARIFICATION_PROTOCOL_FORMAT,
)
from degs.successful_source import (
    SUCCESS_EXTRACTION_KIND,
    SUCCESS_PROMPT_SHA256,
    SUCCESS_SOURCE_PROTOCOL_FORMAT,
)
from degs.validated_repair import (
    PRODUCER_RUNTIME_TIMEOUT_RETRIES,
    PRODUCER_TRANSPORT_RETRY_WAITS,
    REPAIR_SOURCE_MAX_TOKENS,
    REPAIR_SOURCE_MODEL,
    REPAIR_PROMPT_SHA256,
    REPAIR_SYSTEM_PROMPT,
    REPAIR_SOURCE_TEMPERATURE,
    REPAIR_SOURCE_THINKING,
    REPAIR_SOURCE_TIMEOUT_SECONDS,
    SystemicProducerTransportFailure,
    _source_generation_config,
    producer_transport_failure_policy,
    repair_response_schema,
)
from degs.workflow_retrieval import RETRIEVAL_METHOD_ID
from degs.workflow_retrieval import (
    NEED_GRAPH_KIND,
    NEED_GRAPH_PROMPT_SHA256,
    NEED_GRAPH_PROTOCOL_FORMAT,
    SELECTOR_KIND,
    SELECTOR_PROMPT_SHA256,
    SELECTOR_PROTOCOL_FORMAT,
)


def _contract(label: str) -> IOContract:
    return IOContract("artifact", label)


def _node(index: int) -> ExperienceNode:
    return ExperienceNode(
        f"Inspect spreadsheet structure for task {index}",
        ("The task operates on a spreadsheet whose structure must be inspected.",),
        (_contract("input spreadsheet"),),
        (_contract("observed spreadsheet structure"),),
    )


def _source_for_workflows(
    workflows: tuple[WorkflowGraph, ...],
) -> SectionGraphSource:
    payload = {
        "format": "degs_experience_workflows_v4",
        "source_split": "train[0,200)",
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
            for workflow in workflows
        ],
    }
    return SectionGraphSource(
        workflows,
        hashlib.sha256(canonical_json_bytes(payload)).hexdigest(),
    )


def _batch(start: int, *, source_count: int = 8) -> SectionGraphSource:
    return _source_for_workflows(
        tuple(
            WorkflowGraph(
                index,
                f"task-{index:03d}",
                f"Inspect synthetic spreadsheet {index:03d}",
                (_node(index),),
                (),
            )
            for index in range(start, start + source_count)
        )
    )


def _empty_batch() -> SectionGraphSource:
    return _source_for_workflows(())


def _batch_audit(
    source: SectionGraphSource,
    indices: tuple[int, ...],
) -> dict:
    protocol = {
        "format": SUCCESS_SOURCE_PROTOCOL_FORMAT,
        "request_kind": SUCCESS_EXTRACTION_KIND,
        "model": REPAIR_SOURCE_MODEL,
        "temperature": REPAIR_SOURCE_TEMPERATURE,
        "thinking": REPAIR_SOURCE_THINKING,
        "max_tokens": REPAIR_SOURCE_MAX_TOKENS,
        "timeout_seconds": REPAIR_SOURCE_TIMEOUT_SECONDS,
        "generation_config": _source_generation_config(),
        "retry_waits_seconds": list(PRODUCER_TRANSPORT_RETRY_WAITS),
        "runtime_timeout_retries": PRODUCER_RUNTIME_TIMEOUT_RETRIES,
        "prompt_sha256": SUCCESS_PROMPT_SHA256,
        "service_url": "http://127.0.0.1:9999/v1",
    }
    protocol_sha = hashlib.sha256(
        canonical_json_bytes(protocol)
    ).hexdigest()
    response_schema_sha = hashlib.sha256(
        canonical_json_bytes(repair_response_schema())
    ).hexdigest()
    review_protocol = {
        **protocol,
        "format": SOURCE_REVIEW_PROTOCOL_FORMAT,
        "request_kind": SOURCE_REVIEW_KIND,
        "prompt_sha256": SOURCE_REVIEW_PROMPT_SHA256,
    }
    review_protocol_sha = hashlib.sha256(
        canonical_json_bytes(review_protocol)
    ).hexdigest()
    review_schema_sha = hashlib.sha256(
        canonical_json_bytes(source_review_response_schema())
    ).hexdigest()
    present = set(source.workflow_by_index)
    exclusions = [
        {
            "train_index": index,
            "task_id": f"task-{index:03d}",
            "trajectory_id": f"trajectory-{index:03d}",
            "origin": "ORIGINAL_FAILURE",
            "status": "SOURCE_EXCLUDED_NO_VALIDATED_SUCCESS",
            "replay_terminal_status": "REPLAY_EXHAUSTED",
        }
        for index in indices
        if index not in present
    ]
    return {
        "format": SOURCE_REBUILD_AUDIT_FORMAT,
        "source_split": "train[0,200)",
        "section_graphs_sha256": source.sha256,
        "source_workflow_count": len(source.workflows),
        "source_extraction_workers": SOURCE_REBUILD_WORKERS,
        "source_review_workers": SOURCE_REBUILD_WORKERS,
        "incremental_batch_size": INCREMENTAL_BATCH_SIZE,
        "batch_train_indices": list(indices),
        "source_extraction_semantic_attempt_limit": 3,
        "source_review_semantic_attempt_limit": 3,
        "producer_transport_failure_policy": (
            producer_transport_failure_policy()
        ),
        "review_retry_mode": False,
        "review_retry_queue_count": 0,
        "review_retry_queue": [],
        "review_status_counts": (
            {"REVIEW_ACCEPTED": len(source.workflows)}
            if source.workflows else {}
        ),
        "origin_counts": (
            {"ORIGINAL_SUCCESS": len(source.workflows)}
            if source.workflows else {}
        ),
        "discarded_edge_count": 0,
        "excluded_source_workflow_count": len(exclusions),
        "exclusion_counts": (
            {"SOURCE_EXCLUDED_NO_VALIDATED_SUCCESS": len(exclusions)}
            if exclusions else {}
        ),
        "exclusions": exclusions,
        "rows": [
            {
                "train_index": workflow.train_index,
                "task_id": workflow.task_id,
                "trajectory_id": f"trajectory-{workflow.train_index:03d}",
                "origin": "ORIGINAL_SUCCESS",
                "query_text": workflow.query_text,
                "query_text_sha256": hashlib.sha256(
                    workflow.query_text.encode("utf-8")
                ).hexdigest(),
                "status": "INGESTED",
                "prompt_sha256": SUCCESS_PROMPT_SHA256,
                "source_protocol": protocol,
                "source_protocol_sha256": protocol_sha,
                "request_payload_sha256": "1" * 64,
                "response_schema_sha256": response_schema_sha,
                "discarded_edge_reasons": [],
                "draft_graph": {
                    "experience_nodes": [
                        node.to_dict()
                        for node in workflow.experience_nodes
                    ],
                    "edges": [
                        edge.to_dict() for edge in workflow.edges
                    ],
                },
                "draft_discarded_edge_reasons": [],
                "source_extraction_attempt_index": 1,
                "invalid_response_attempts": [],
                "source_review_status": "REVIEW_ACCEPTED",
                "source_review_prompt_sha256": SOURCE_REVIEW_PROMPT_SHA256,
                "source_review_protocol": review_protocol,
                "source_review_protocol_sha256": review_protocol_sha,
                "source_review_request_payload_sha256": "2" * 64,
                "source_review_response_schema_sha256": review_schema_sha,
                "source_review_attempt_index": 1,
                "source_review_invalid_response_attempts": [],
                "source_review_decisions": [
                    {
                        "draft_node": node_index,
                        "decision": "KEEP",
                        "final_nodes": [node_index],
                        "basis": "The node is supported and reusable.",
                    }
                    for node_index, _node in enumerate(
                        workflow.experience_nodes
                    )
                ],
                "source_review_ledger_status": "REVIEW_LEDGER_COMPLETE",
                "source_review_ledger_errors": [],
                "source_review_discarded_edge_reasons": [],
                "draft_graph_sha256": hashlib.sha256(
                    canonical_json_bytes(
                        {
                            "experience_nodes": [
                                node.to_dict()
                                for node in workflow.experience_nodes
                            ],
                            "edges": [
                                edge.to_dict() for edge in workflow.edges
                            ],
                        }
                    )
                ).hexdigest(),
                "reviewed_graph_sha256": hashlib.sha256(
                    canonical_json_bytes(
                        {
                            "experience_nodes": [
                                node.to_dict()
                                for node in workflow.experience_nodes
                            ],
                            "edges": [
                                edge.to_dict() for edge in workflow.edges
                            ],
                        }
                    )
                ).hexdigest(),
            }
            for workflow in source.workflows
        ],
    }


def _stored_node_id(
    state: IncrementalStateStore, coordinate: tuple[int, int]
) -> str:
    row = state.connection.execute(
        """
        SELECT node_id FROM experience_nodes
        WHERE train_index = ? AND node_index = ?
        """,
        coordinate,
    ).fetchone()
    assert row is not None
    return str(row[0])


def _stored_member_ids_by_group(
    state: IncrementalStateStore, partition: CanonicalPartition
) -> tuple[frozenset[str], ...]:
    return tuple(
        frozenset(_stored_node_id(state, coordinate) for coordinate in group.members)
        for group in partition.groups
    )


class _EmbeddingTransport:
    endpoint = "http://127.0.0.1:9999/v1/embeddings"

    def __init__(self) -> None:
        self.calls = 0

    async def embed_async(self, request):
        self.calls += 1
        await asyncio.sleep(0)
        return {
            "model": EMBEDDING_MODEL,
            "data": [
                {"index": index, "embedding": [1.0, 0.0]}
                for index, _text in enumerate(request["input"])
            ],
        }

    def embed(self, _request):
        raise AssertionError("formal graph construction must use async embedding")


class _CanonicalLLM:
    def __init__(
        self,
        stage: str,
        *,
        relation: str | None = None,
        failure: Exception | None = None,
    ) -> None:
        self.stage = stage
        self.relation = relation or "SAME_TEMPLATE"
        self.failure = failure
        self.calls: list[dict] = []

    @property
    def protocol_identity(self):
        values = {
            "view": (
                CANONICAL_VIEW_PROTOCOL_FORMAT,
                CANONICAL_VIEW_KIND,
                CANONICAL_VIEW_PROMPT_SHA256,
            ),
            "merge": (CANONICAL_MERGE_PROTOCOL_FORMAT, CANONICAL_MERGE_KIND, CANONICAL_MERGE_PROMPT_SHA256),
        }
        protocol_format, kind, prompt_sha = values[self.stage]
        return {
            "format": protocol_format,
            "request_kind": kind,
            "model": REPAIR_SOURCE_MODEL,
            "temperature": REPAIR_SOURCE_TEMPERATURE,
            "thinking": REPAIR_SOURCE_THINKING,
            "max_tokens": REPAIR_SOURCE_MAX_TOKENS,
            "timeout_seconds": REPAIR_SOURCE_TIMEOUT_SECONDS,
            "generation_config": _source_generation_config(),
            "retry_waits_seconds": list(PRODUCER_TRANSPORT_RETRY_WAITS),
            "runtime_timeout_retries": PRODUCER_RUNTIME_TIMEOUT_RETRIES,
            "prompt_sha256": prompt_sha,
            "service_url": "http://127.0.0.1:9999/v1",
        }

    async def complete_json_async(self, **request):
        self.calls.append(request)
        await asyncio.sleep(0)
        if self.failure is not None:
            raise self.failure
        if self.stage == "view":
            return {
                "reusable_identity": (
                    "Inspect the structural organization of a spreadsheet."
                ),
                "applicability_boundary": (
                    "Use when later operations depend on knowing workbook structure."
                ),
            }
        return {
            "relation": self.relation,
            "basis": "Parameter substitution preserves the same reusable operation.",
            "canonical_experience": None if self.relation != "SAME_TEMPLATE" else {
                "operation": (
                    "Inspect the spreadsheet structure required by downstream work."
                ),
                "applicability": [
                    "Use when downstream operations require structural workbook evidence."
                ],
                "inputs": [_contract("input spreadsheet").to_dict()],
                "outputs": [
                    _contract("observed spreadsheet structure").to_dict()
                ],
            }
        }


class _InvalidThenValidViewLLM(_CanonicalLLM):
    def __init__(self) -> None:
        super().__init__("view")
        self.attempts_by_request: dict[str, int] = {}

    async def complete_json_async(self, **request):
        self.calls.append(request)
        await asyncio.sleep(0)
        request_id = request["request_id"]
        attempt = self.attempts_by_request.get(request_id, 0) + 1
        self.attempts_by_request[request_id] = attempt
        response = {
            "reusable_identity": "Inspect a structured artifact.",
            "applicability_boundary": (
                "Use when later work depends on structural evidence."
            ),
        }
        if attempt == 1:
            response["unexpected_field"] = "invalid"
        return response


class _OneRequestTransportFailureViewLLM(_CanonicalLLM):
    def __init__(self, failure: Exception) -> None:
        super().__init__("view")
        self.transport_failure = failure
        self.failed_request_id: str | None = None

    async def complete_json_async(self, **request):
        request_id = request["request_id"]
        if self.failed_request_id is None:
            self.failed_request_id = request_id
        if request_id == self.failed_request_id:
            self.calls.append(request)
            await asyncio.sleep(0)
            raise self.transport_failure
        return await super().complete_json_async(**request)


class _TransportThenSchemaThenCrashViewLLM(_CanonicalLLM):
    def __init__(self) -> None:
        super().__init__("view")
        self.target_request_id: str | None = None
        self.target_attempts = 0

    async def complete_json_async(self, **request):
        request_id = request["request_id"]
        if self.target_request_id is None:
            self.target_request_id = request_id
        if request_id != self.target_request_id:
            return await super().complete_json_async(**request)
        self.calls.append(request)
        self.target_attempts += 1
        await asyncio.sleep(0)
        if self.target_attempts == 1:
            raise RequestRuntimeTimeout("synthetic transient timeout")
        if self.target_attempts == 2:
            return {
                "reusable_identity": "Inspect a structured artifact.",
                "applicability_boundary": (
                    "Use when later work depends on structural evidence."
                ),
                "unexpected_field": "invalid",
            }
        raise RuntimeError("synthetic process interruption")


class _FailingEmbeddingTransport(_EmbeddingTransport):
    async def embed_async(self, _request):
        raise RuntimeError("synthetic embedding interruption")


class _FormalRetrievalLLM:
    def __init__(
        self,
        stage: str,
        *,
        fail_request_id: str | None = None,
        clarify_request_id: str | None = None,
    ) -> None:
        self.stage = stage
        self.fail_request_id = fail_request_id
        self.clarify_request_id = clarify_request_id
        self.calls = []

    @property
    def protocol_identity(self):
        if self.stage == "need":
            protocol_format = NEED_GRAPH_PROTOCOL_FORMAT
            kind = NEED_GRAPH_KIND
            prompt_sha = NEED_GRAPH_PROMPT_SHA256
        elif self.stage == "clarification":
            protocol_format = CLARIFICATION_PROTOCOL_FORMAT
            kind = CLARIFICATION_KIND
            prompt_sha = CLARIFICATION_PROMPT_SHA256
        else:
            protocol_format = SELECTOR_PROTOCOL_FORMAT
            kind = SELECTOR_KIND
            prompt_sha = SELECTOR_PROMPT_SHA256
        return {
            "format": protocol_format,
            "request_kind": kind,
            "model": REPAIR_SOURCE_MODEL,
            "temperature": REPAIR_SOURCE_TEMPERATURE,
            "thinking": REPAIR_SOURCE_THINKING,
            "max_tokens": REPAIR_SOURCE_MAX_TOKENS,
            "timeout_seconds": REPAIR_SOURCE_TIMEOUT_SECONDS,
            "generation_config": _source_generation_config(),
            "retry_waits_seconds": list(BUNDLE_TRANSPORT_RETRY_WAITS),
            "runtime_timeout_retries": BUNDLE_RUNTIME_TIMEOUT_RETRIES,
            "prompt_sha256": prompt_sha,
            "service_url": "http://127.0.0.1:9999/v1",
        }

    async def complete_json_async(self, **request):
        await asyncio.sleep(0)
        self.calls.append(request)
        if self.stage == "need":
            return {
                "need_nodes": [
                    {
                        "description": "Inspect spreadsheet structure",
                        "applicability_context": [
                            "The task requires structural workbook evidence."
                        ],
                        "inputs": [_contract("input spreadsheet").to_dict()],
                        "outputs": [
                            _contract("observed target value representation").to_dict()
                        ],
                    }
                ],
                "edges": [],
            }
        if self.stage == "clarification":
            if request["request_id"] == self.clarify_request_id:
                return {
                    "clarifications": [
                        {"need_node": 0, "evidence_ids": ["O0"]}
                    ]
                }
            return {"clarifications": []}
        if request["request_id"] == self.fail_request_id:
            return {"selected_candidate_id": "not-a-supplied-candidate"}
        candidates = request["payload"]["candidates"]
        return {
            "selected_candidate_id": (
                candidates[0]["candidate_id"] if candidates else None
            )
        }


def _builder(
    tmp_path: Path, state: IncrementalStateStore, *,
    view_failure: Exception | None = None, merge_failure: Exception | None = None,
    view_llm: _CanonicalLLM | None = None, merge_llm: _CanonicalLLM | None = None,
):
    view = view_llm or _CanonicalLLM("view", failure=view_failure)
    merge = merge_llm or _CanonicalLLM("merge", failure=merge_failure)
    return (IncrementalGraphBuilder(
        state=state, snapshot_root=tmp_path / "snapshots",
        embedder=StrictEmbeddingAdapter(_EmbeddingTransport(), cache=state.embedding_cache()),
        view_llm=view, merge_llm=merge,
    ), view, merge)


def test_method_identity_and_removed_old_canonical_surface() -> None:
    assert "0.78.0" in degs.METHOD_NAME
    assert "DEGS" in RETRIEVAL_METHOD_ID
    assert degs.__version__ == "0.78.0"
    assert "DEGS 0.77.41 Stable R1" in INCREMENTAL_METHOD_ID
    for module in (
        "degs.query",
        "degs.retrieval",
        "degs.section_bundle",
        "degs.experience",
    ):
        assert importlib.util.find_spec(module) is None
    for symbol in (
        "CanonicalPartitionProducer",
        "match_contract_sets",
        "score_experience_node_pair",
        "WORKFLOW_TOP_K",
        "WORKFLOW_LEIDEN_RESOLUTION",
        "ALIGN_LEIDEN_RESOLUTION",
    ):
        assert not hasattr(canonicalize_module, symbol)
    assert not hasattr(workflow_retrieval_module, "NeedGraphProducer")
    assert not hasattr(core_module, "sanitize_member_content")


def test_accepted_parameterized_micro_operation_prompt_is_restored() -> None:
    prompt = canonicalize_module.CANONICAL_MERGE_SYSTEM_PROMPT
    assert "causal micro-operation" in prompt
    assert "locally executable procedure" in prompt
    assert "Parameter substitution may change" in prompt
    assert "predecessor" in prompt
    assert canonicalize_module.CANONICAL_MERGE_KIND.endswith("_v3")
    assert canonicalize_module.CANONICAL_MERGE_PROTOCOL_FORMAT.endswith("_v3")


def test_causal_source_prompts_replace_the_old_extraction_identity() -> None:
    assert SUCCESS_PROMPT_SHA256 != (
        "0774e0902a464dfdeaa194ae9f2c37c7507dcdcb86a242a2e0e7b93d2d1230f0"
    )
    assert REPAIR_PROMPT_SHA256 != (
        "c033c7c5e5f1cd95a637a0ee53781d7a17763da403510586bd6ec9d7787a3e2e"
    )
    success_prompt = successful_source_module.SUCCESS_SYSTEM_PROMPT
    assert "counterfactual" in success_prompt
    assert "Routine setup, loading, opening, saving" in success_prompt
    assert "plausible incorrect result" in success_prompt
    assert "jointly supported" in REPAIR_SYSTEM_PROMPT
    assert "Do not emit the union" in REPAIR_SYSTEM_PROMPT


def test_canonical_semantic_retry_audit_and_cache_resume(tmp_path: Path) -> None:
    database = tmp_path / "state.sqlite3"
    batch = _batch(0)
    audit = _batch_audit(batch, tuple(range(8)))
    with IncrementalStateStore(database) as state:
        first_view = _InvalidThenValidViewLLM()
        interrupted = IncrementalGraphBuilder(
            state=state,
            snapshot_root=tmp_path / "snapshots",
            embedder=StrictEmbeddingAdapter(
                _FailingEmbeddingTransport(), cache=state.embedding_cache()
            ),
            view_llm=first_view,
            merge_llm=_CanonicalLLM("merge"),
        )
        with pytest.raises(RuntimeError, match="synthetic embedding interruption"):
            asyncio.run(
                interrupted.update_async(
                    batch_source=batch,
                    batch_train_indices=range(8),
                    batch_source_audit=audit,
                )
            )
        assert state.head_snapshot_id is None
        statuses = state.connection.execute(
            """
            SELECT status, COUNT(*)
            FROM canonical_attempts
            WHERE stage = 'VIEW'
            GROUP BY status
            """
        ).fetchall()
        assert statuses == [("ACCEPTED", 8), ("REJECTED", 8)]
        assert len(first_view.calls) == 16

        resumed_view = _CanonicalLLM("view")
        resumed = IncrementalGraphBuilder(
            state=state,
            snapshot_root=tmp_path / "snapshots",
            embedder=StrictEmbeddingAdapter(
                _EmbeddingTransport(), cache=state.embedding_cache()
            ),
            view_llm=resumed_view,
            merge_llm=_CanonicalLLM("merge"),
        )
        result = asyncio.run(
            resumed.update_async(
                batch_source=batch,
                batch_train_indices=range(8),
                batch_source_audit=audit,
            )
        )
        assert state.head_snapshot_id == result.snapshot_id
        assert resumed_view.calls == []

    canonical_audit = json.loads(
        (result.output_dir / "canonical_audit.json").read_text()
    )
    requests = canonical_audit["views"]
    assert all(row["status"] == "VIEW_ACCEPTED_RESUMED" for row in requests)
    assert all(row["semantic_attempt_index"] == 2 for row in requests)


def test_canonical_acceptance_and_cached_job_commit_atomically(
    tmp_path: Path,
) -> None:
    request_sha256 = "a" * 64
    snapshot_id = "synthetic-building-snapshot"
    with IncrementalStateStore(tmp_path / "state.sqlite3") as state:
        state.connection.execute(
            """
            INSERT INTO snapshots(
                snapshot_id, parent_snapshot_id, batch_source_sha256,
                batch_source_audit_sha256, status
            ) VALUES (?, NULL, ?, ?, 'BUILDING')
            """,
            (snapshot_id, "d" * 64, "e" * 64),
        )
        state.connection.execute(
            """
            CREATE TRIGGER synthetic_abort_canonical_job
            BEFORE INSERT ON canonical_jobs
            BEGIN
                SELECT RAISE(ABORT, 'synthetic canonical cache interruption');
            END
            """
        )
        with pytest.raises(ValueError, match="Canonical cache contains conflicting output"):
            state.put_canonical_job(
                request_sha256,
                stage="VIEW",
                semantic_attempt_index=1,
                prompt_sha256="b" * 64,
                producer_protocol_sha256="c" * 64,
                response={"identity": "operation", "boundary": "condition"},
                audit={"status": "VIEW_ACCEPTED"},
                created_snapshot_id=snapshot_id,
            )
        assert state.connection.execute(
            "SELECT COUNT(*) FROM canonical_attempts WHERE request_sha256 = ?",
            (request_sha256,),
        ).fetchone() == (0,)
        assert state.connection.execute(
            "SELECT COUNT(*) FROM canonical_jobs WHERE request_sha256 = ?",
            (request_sha256,),
        ).fetchone() == (0,)


@pytest.mark.parametrize(
    "table,delete_sql",
    [
        (
            "canonical_heads",
            "DELETE FROM canonical_heads WHERE node_id = "
            "(SELECT node_id FROM canonical_heads ORDER BY node_id LIMIT 1)",
        ),
        (
            "canonical_leaf_members",
            "DELETE FROM canonical_leaf_members WHERE rowid = "
            "(SELECT rowid FROM canonical_leaf_members ORDER BY rowid LIMIT 1)",
        ),
    ],
)
def test_snapshot_state_agreement_rejects_membership_corruption(
    tmp_path: Path,
    table: str,
    delete_sql: str,
) -> None:
    with IncrementalStateStore(tmp_path / f"{table}.sqlite3") as state:
        builder, _view, _merge = _builder(
            tmp_path / table, state
        )
        batch = _batch(0)
        result = asyncio.run(
            builder.update_async(
                batch_source=batch,
                batch_train_indices=range(8),
                batch_source_audit=_batch_audit(batch, tuple(range(8))),
            )
        )
        source = load_section_graphs(
            result.output_dir / "accumulated_section_graphs.json"
        )
        partition = load_canonical_partition(
            result.output_dir / "canonical_partition.json", source=source
        )
        graph = compile_experience_graph(source, partition)
        state.connection.execute(delete_sql)
        with pytest.raises(
            ValueError, match="committing snapshot Canonical membership differs"
        ):
            _validate_committing_snapshot_state(
                state=state,
                snapshot_id=result.snapshot_id,
                parent_snapshot_id=None,
                source=source,
                partition=partition,
                graph=graph,
                manifest=result.manifest,
                source_status_by_index={index: "INGESTED" for index in range(8)},
            )


def test_canonical_transport_failure_is_item_local_and_commits_fallback(
    tmp_path: Path,
) -> None:
    database = tmp_path / "state.sqlite3"
    batch = _batch(0)
    audit = _batch_audit(batch, tuple(range(8)))
    with IncrementalStateStore(database) as state:
        failing, _view, _merge = _builder(
            tmp_path,
            state,
            view_llm=_OneRequestTransportFailureViewLLM(
                RequestRuntimeTimeout("one request unavailable")
            ),
        )
        result = asyncio.run(
            failing.update_async(
                batch_source=batch,
                batch_train_indices=range(8),
                batch_source_audit=audit,
            )
        )
        assert state.head_snapshot_id == result.snapshot_id
        assert state.connection.execute(
            "SELECT status FROM snapshots"
        ).fetchone() == ("COMMITTED",)
        assert state.connection.execute(
            """
            SELECT COUNT(*) FROM canonical_resolution_events
            WHERE status = 'VIEW_SOURCE_FALLBACK'
            """
        ).fetchone() == (1,)
        assert state.connection.execute(
            """
            SELECT COUNT(*) FROM canonical_attempts
            WHERE stage = 'VIEW' AND error_type = 'RequestRuntimeTimeout'
            """
        ).fetchone() == (0,)
        transport_status, attempts_json = state.connection.execute(
            """
            SELECT status, attempts_json FROM canonical_transport_waves
            WHERE stage = 'VIEW'
            """
        ).fetchone()
        assert transport_status == "ITEM_LOCAL"
        assert len(json.loads(attempts_json)) == 3


def test_canonical_systemic_transport_outage_fails_after_the_wave(
    tmp_path: Path,
) -> None:
    batch = _batch(0)
    audit = _batch_audit(batch, tuple(range(8)))
    with IncrementalStateStore(tmp_path / "state.sqlite3") as state:
        failing, view, _merge = _builder(
            tmp_path,
            state,
            view_failure=RequestRuntimeTimeout("provider unavailable"),
        )
        with pytest.raises(
            SystemicProducerTransportFailure,
            match="8 distinct requests in one stage wave",
        ):
            asyncio.run(
                failing.update_async(
                    batch_source=batch,
                    batch_train_indices=range(8),
                    batch_source_audit=audit,
                )
            )
        assert len(view.calls) == 24
        assert state.head_snapshot_id is None
        assert state.connection.execute(
            "SELECT status FROM snapshots"
        ).fetchone() == ("BUILDING",)
        assert state.connection.execute(
            "SELECT status FROM canonical_transport_waves"
        ).fetchone() == ("SYSTEMIC",)

        still_down, still_down_view, *_ = _builder(
            tmp_path,
            state,
            view_failure=RequestRuntimeTimeout("provider unavailable"),
        )
        with pytest.raises(SystemicProducerTransportFailure):
            asyncio.run(
                still_down.update_async(
                    batch_source=batch,
                    batch_train_indices=range(8),
                    batch_source_audit=audit,
                )
            )
        assert len(still_down_view.calls) == 24
        assert state.head_snapshot_id is None

        restored, restored_view, *_ = _builder(tmp_path, state)
        result = asyncio.run(
            restored.update_async(
                batch_source=batch,
                batch_train_indices=range(8),
                batch_source_audit=audit,
            )
        )
        assert len(restored_view.calls) == 8
        assert state.head_snapshot_id == result.snapshot_id
        assert state.connection.execute(
            "SELECT status FROM snapshots"
        ).fetchone() == ("COMMITTED",)


def test_canonical_transport_does_not_create_a_gap_in_semantic_resume(
    tmp_path: Path,
) -> None:
    batch = _batch(0)
    audit = _batch_audit(batch, tuple(range(8)))
    with IncrementalStateStore(tmp_path / "state.sqlite3") as state:
        interrupted_llm = _TransportThenSchemaThenCrashViewLLM()
        interrupted, *_ = _builder(
            tmp_path,
            state,
            view_llm=interrupted_llm,
        )
        with pytest.raises(RuntimeError, match="process interruption"):
            asyncio.run(
                interrupted.update_async(
                    batch_source=batch,
                    batch_train_indices=range(8),
                    batch_source_audit=audit,
                )
            )
        assert interrupted_llm.target_request_id is not None
        attempt_rows = state.connection.execute(
            """
            SELECT semantic_attempt_index, error_type
            FROM canonical_attempts
            WHERE stage = 'VIEW' AND error_type IS NOT NULL
            ORDER BY semantic_attempt_index
            """
        ).fetchall()
        assert attempt_rows == [(1, "ValueError")]
        assert state.connection.execute(
            "SELECT status FROM snapshots"
        ).fetchone() == ("BUILDING",)

        restored, restored_view, *_ = _builder(tmp_path, state)
        result = asyncio.run(
            restored.update_async(
                batch_source=batch,
                batch_train_indices=range(8),
                batch_source_audit=audit,
            )
        )
        assert state.head_snapshot_id == result.snapshot_id
        resumed_target_calls = [
            request
            for request in restored_view.calls
            if request["request_id"] == interrupted_llm.target_request_id
        ]
        assert len(resumed_target_calls) == 1
        accepted_attempt = state.connection.execute(
            """
            SELECT semantic_attempt_index, status
            FROM canonical_attempts
            WHERE stage = 'VIEW' AND status = 'ACCEPTED'
              AND semantic_attempt_index = 2
            """
        ).fetchall()
        assert accepted_attempt == [(2, "ACCEPTED")]


def test_canonical_job_wave_cancels_and_drains_on_immediate_failure() -> None:
    async def exercise() -> tuple[list[int], list[int]]:
        ready = asyncio.Event()
        started = 0
        cancelled: list[int] = []
        completed: list[int] = []

        async def worker(job: int) -> int:
            nonlocal started
            if job == 0:
                await ready.wait()
                raise SystemicProducerTransportFailure(
                    "synthetic immediate auth failure",
                    stage="VIEW",
                    failed_request_ids=("request-0",),
                )
            started += 1
            if started == 7:
                ready.set()
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                cancelled.append(job)
                raise
            completed.append(job)
            return job

        with pytest.raises(SystemicProducerTransportFailure):
            await canonicalize_module._run_canonical_jobs(
                tuple(range(8)), worker
            )
        return cancelled, completed

    cancelled, completed = asyncio.run(exercise())
    assert sorted(cancelled) == list(range(1, 8))
    assert completed == []


def test_embedding_and_generation_endpoints_are_runtime_metadata(
    tmp_path: Path,
) -> None:
    with IncrementalStateStore(tmp_path / "state.sqlite3") as state:
        state.bind_embedding_endpoint(
            "http://127.0.0.1:9000/v1/embeddings"
        )
        state.bind_embedding_endpoint(
            "http://127.0.0.1:9001/v1/embeddings"
        )
        state.bind_generation_endpoint("http://127.0.0.1:9000/v1")
        state.bind_generation_endpoint("http://127.0.0.1:9001/v1")
        assert state.embedding_endpoint == "http://127.0.0.1:9001/v1/embeddings"
        assert state.generation_endpoint == "http://127.0.0.1:9001/v1"


def test_final_snapshot_verifier_accepts_only_the_new_canonical_protocol(
    tmp_path: Path,
) -> None:
    database = tmp_path / "state.sqlite3"
    final = None
    with IncrementalStateStore(database) as state:
        builder, _view, _merge = _builder(tmp_path, state)
        for start in range(0, 200, INCREMENTAL_BATCH_SIZE):
            indices = tuple(range(start, start + INCREMENTAL_BATCH_SIZE))
            source = _batch(0, source_count=1) if start == 0 else _empty_batch()
            final = asyncio.run(
                builder.update_async(
                    batch_source=source,
                    batch_train_indices=indices,
                    batch_source_audit=_batch_audit(source, indices),
                )
            )
    assert final is not None
    context = _load_snapshot(
        final.output_dir / "snapshot_manifest.json",
        state_db_path=database,
        train_queries=tuple(
            {
                "train_index": index,
                "task_id": f"task-{index:03d}",
                "instruction": f"Inspect synthetic spreadsheet {index:03d}",
            }
            for index in range(200)
        ),
    )
    assert context.identity["method"] == INCREMENTAL_METHOD_ID
    assert context.identity["snapshot_id"] == final.snapshot_id
    canonical_audit = json.loads(
        (final.output_dir / "canonical_audit.json").read_text()
    )
    assert canonical_audit["protocol"]["context_length_policy"] == (
        "context_overflow_item_local_no_merge_no_repeat_v3"
    )
    legacy_audit = json.loads(json.dumps(canonical_audit))
    legacy_audit["protocol"].pop("context_length_policy")
    legacy_audit["protocol_sha256"] = hashlib.sha256(
        canonical_json_bytes(legacy_audit["protocol"])
    ).hexdigest()
    with pytest.raises(ValueError, match="incremental Canonical protocol differs"):
        bundle_module._validated_canonical_protocol(
            legacy_audit,
            snapshot_id=final.snapshot_id,
            source_sha256=canonical_audit["section_graphs_sha256"],
        )


def test_formal_bundle_build_verify_and_provider_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        bundle_module,
        "_runtime_identity",
        lambda: bundle_module.EXPECTED_BUNDLE_RUNTIME,
    )
    real_quality_builder = incremental_graph_module.build_graph_quality_artifacts

    def diagnostic_not_ready(**kwargs):
        artifacts = real_quality_builder(**kwargs)
        audit_body = {
            key: value
            for key, value in artifacts.audit.items()
            if key != "self_sha256"
        }
        audit_body["status"] = "NOT_READY"
        audit_body["hard_violations"] = [
            {
                "check": "one_call_merge_evidence",
                "detail": "synthetic diagnostic status",
            }
        ]
        audit = {
            **audit_body,
            "self_sha256": hashlib.sha256(
                canonical_json_bytes(audit_body)
            ).hexdigest(),
        }
        return type(artifacts)(
            audit,
            artifacts.source_node_rows,
            artifacts.high_similarity_unmerged_rows,
            artifacts.candidate_recall_rows,
            artifacts.merge_ledger_rows,
            artifacts.topology,
        )

    monkeypatch.setattr(
        incremental_graph_module,
        "build_graph_quality_artifacts",
        diagnostic_not_ready,
    )
    database = tmp_path / "state.sqlite3"
    final = None
    with IncrementalStateStore(database) as state:
        builder, _view, _merge = _builder(tmp_path, state)
        for start in range(0, 200, INCREMENTAL_BATCH_SIZE):
            indices = tuple(range(start, start + INCREMENTAL_BATCH_SIZE))
            source = _batch(0, source_count=1) if start == 0 else _empty_batch()
            final = asyncio.run(
                builder.update_async(
                    batch_source=source,
                    batch_train_indices=indices,
                    batch_source_audit=_batch_audit(source, indices),
                )
            )
    assert final is not None
    train_queries = [
        {
            "train_index": index,
            "task_id": f"task-{index:03d}",
            "instruction": f"Inspect synthetic spreadsheet {index:03d}",
        }
        for index in range(200)
    ]
    development_queries = [
        {
            "development_index": index,
            "dataset_index": 200 + index,
            "task_id": f"development-{index:03d}",
            "instruction": f"Inspect development spreadsheet {index:03d}",
            "spreadsheet_path": f"spreadsheet/development-{index:03d}",
            "instruction_type": "Cell-Level Manipulation",
            "answer_position": "A1",
        }
        for index in range(200)
    ]
    monkeypatch.setattr(
        bundle_module, "load_train_queries", lambda _path: train_queries
    )
    monkeypatch.setattr(
        bundle_module,
        "_load_train_harness_records",
        lambda _path: [
            {
                **row,
                "spreadsheet_path": f"spreadsheet/{row['task_id']}",
                "answer_position": "A1",
            }
            for row in train_queries
        ],
    )
    monkeypatch.setattr(
        bundle_module,
        "_load_development_harness_records",
        lambda _path: development_queries,
    )
    card = TargetEvidenceCard(
        status="OK",
        observations=(
            {
                "evidence_id": "O0",
                "kind": "target_range_landmarks",
                "content": json.dumps(
                        {
                            "sheet": "Sheet1",
                            "cells": [
                                {
                                    "coordinate": "A1",
                                    "value": 1,
                                    "data_type": "n",
                                    "number_format": "0",
                                }
                            ],
                            "merged_ranges": [],
                        }
                ),
            },
        ),
        input_sha256="a" * 64,
        input_relative_path="spreadsheet/development/input.xlsx",
    )
    monkeypatch.setattr(
        bundle_module,
        "_target_evidence_cards",
        lambda tasks, dataset_path: {task.query_index: card for task in tasks},
    )
    monkeypatch.setattr(
        bundle_module,
        "_train_evidence_cards",
        lambda rows, dataset_path: {index: card for index in range(200)},
    )
    monkeypatch.setattr(
        bundle_module,
        "EXPECTED_QUERY_PROJECTION_SHA256",
        query_projection_sha256(train_queries),
    )
    monkeypatch.setattr(
        bundle_module,
        "EXPECTED_DEVELOPMENT_QUERY_PROJECTION_SHA256",
        development_query_projection_sha256(
            bundle_module._query_fields(development_queries)
        ),
    )
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text("[]\n", encoding="utf-8")
    output = tmp_path / "bundle"
    output.mkdir()
    need_llm = _FormalRetrievalLLM("need")
    clarification_llm = _FormalRetrievalLLM(
        "clarification",
        clarify_request_id="retrieval-clarification-development-000",
    )
    selector_llm = _FormalRetrievalLLM("selector")
    built = build_from_paths(
        dataset_path=dataset_path,
        snapshot_manifest_path=(
            final.output_dir / "snapshot_manifest.json"
        ),
        state_db_path=database,
        output_dir=output,
        embedding_transport=_EmbeddingTransport(),
        need_llm=need_llm,
        clarification_llm=clarification_llm,
        selector_llm=selector_llm,
    )
    verified = verify_from_paths(
        dataset_path=dataset_path,
        snapshot_manifest_path=(
            final.output_dir / "snapshot_manifest.json"
        ),
        state_db_path=database,
        output_dir=output,
    )
    provider = DEGSExperienceProvider(verified)
    assert final.manifest["graph_quality_status"] == "NOT_READY"
    assert built.manifest == verified.manifest
    assert verified.manifest["row_count"] == 200
    assert verified.manifest["status_counts"] == {"OK": 200}
    first_row = json.loads(
        (output / "experience.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert "input_observations" not in first_row["experience"]
    assert "declared_answer_position" not in first_row["experience"]
    first_row = json.loads(
        (output / "experience.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert first_row["metadata"]["retrieval_audit"]["selector"] == {
        "llm_called": False,
        "policy": "DETERMINISTIC_TOP_RANKED_C0",
        "selected_candidate_id": "C0",
    }
    assert first_row["metadata"]["retrieval_audit"][
        "retrieval_clarification"
    ]["attempts"] == 1
    assert first_row["experience"]
    second_row = json.loads(
        (output / "experience.jsonl").read_text(encoding="utf-8").splitlines()[1]
    )
    assert second_row["metadata"]["retrieval_audit"]["selector"]["llm_called"] is False
    assert second_row["experience"]
    assert len(need_llm.calls) == 200
    assert len(clarification_llm.calls) == 200
    assert len(selector_llm.calls) == 0
    assert provider.for_instance("development-000").experience
    with pytest.raises(FileExistsError, match="fresh or empty"):
        build_from_paths(
            dataset_path=dataset_path,
            snapshot_manifest_path=(
                final.output_dir / "snapshot_manifest.json"
            ),
            state_db_path=database,
            output_dir=output,
            embedding_transport=_EmbeddingTransport(),
            need_llm=_FormalRetrievalLLM("need"),
            clarification_llm=_FormalRetrievalLLM("clarification"),
            selector_llm=_FormalRetrievalLLM("selector"),
        )
    changed_card = TargetEvidenceCard(
        status=card.status,
        observations=card.observations,
        input_sha256="b" * 64,
        input_relative_path=card.input_relative_path,
    )
    monkeypatch.setattr(
        bundle_module,
        "_target_evidence_cards",
        lambda tasks, dataset_path: {
            task.query_index: changed_card for task in tasks
        },
    )
    changed = build_from_paths(
        dataset_path=dataset_path,
        snapshot_manifest_path=final.output_dir / "snapshot_manifest.json",
        state_db_path=database,
        output_dir=tmp_path / "changed-input-bundle",
        embedding_transport=_EmbeddingTransport(),
        need_llm=_FormalRetrievalLLM("need"),
        clarification_llm=_FormalRetrievalLLM("clarification"),
        selector_llm=_FormalRetrievalLLM("selector"),
    )
    assert changed.manifest["row_count"] == 200
    unavailable_card = TargetEvidenceCard(
        status="UNAVAILABLE",
        observations=(),
        input_sha256=None,
        input_relative_path=None,
        failure="synthetic item-local input failure",
    )
    monkeypatch.setattr(
        bundle_module,
        "_target_evidence_cards",
        lambda tasks, dataset_path: {
            task.query_index: (
                unavailable_card if task.query_index == 0 else card
            )
            for task in tasks
        },
    )
    unavailable_output = tmp_path / "unavailable-input-bundle"
    fallback = build_from_paths(
        dataset_path=dataset_path,
        snapshot_manifest_path=final.output_dir / "snapshot_manifest.json",
        state_db_path=database,
        output_dir=unavailable_output,
        embedding_transport=_EmbeddingTransport(),
        need_llm=_FormalRetrievalLLM("need"),
        clarification_llm=_FormalRetrievalLLM("clarification"),
        selector_llm=_FormalRetrievalLLM("selector"),
    )
    fallback_first = json.loads(
        (unavailable_output / "experience.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert fallback.manifest["row_count"] == 200
    assert fallback_first["experience"]
    assert fallback_first["metadata"]["retrieval_audit"][
        "workflow_context"
    ]["mode"] == "TARGET_EVIDENCE_UNAVAILABLE_QUERY_ONLY"
    verified_fallback = verify_from_paths(
        dataset_path=dataset_path,
        snapshot_manifest_path=final.output_dir / "snapshot_manifest.json",
        state_db_path=database,
        output_dir=unavailable_output,
    )
    assert verified_fallback.manifest["row_count"] == 200

def test_monotonic_groups_survive_dynamic_insert_and_only_node_payloads_are_sent(tmp_path):
    with IncrementalStateStore(tmp_path / "state.sqlite3") as state:
        builder, view, merge = _builder(tmp_path, state)
        first = _batch(0)
        a = asyncio.run(builder.update_async(batch_source=first, batch_train_indices=range(8),
                        batch_source_audit=_batch_audit(first, tuple(range(8)))))
        assert a.manifest["canonical_count"] == 1
        first_id = state.connection.execute("SELECT DISTINCT canonical_id FROM canonical_heads").fetchone()[0]
        second = _batch(8)
        b = asyncio.run(builder.update_async(batch_source=second, batch_train_indices=range(8, 16),
                        batch_source_audit=_batch_audit(second, tuple(range(8, 16)))))
        assert b.manifest["canonical_count"] == 1
        assert state.connection.execute("SELECT COUNT(*) FROM canonical_leaf_members WHERE canonical_id = ?", (first_id,)).fetchone() == (8,)
        assert state.connection.execute("SELECT COUNT(DISTINCT canonical_id) FROM canonical_heads").fetchone() == (1,)
        assert all(set(r["payload"]) == {"experience_node"} for r in view.calls)
        assert all(set(r["payload"]) == {"left", "right"} for r in merge.calls)
        for r in merge.calls:
            assert all(set(node) == {"operation", "applicability", "inputs", "outputs"} for node in r["payload"].values())
        assert state.connection.execute("SELECT DISTINCT stage FROM canonical_jobs ORDER BY stage").fetchall() == [("MERGE",), ("VIEW",)]
        counts = (len(view.calls), len(merge.calls), builder.embedder.transport.calls)
        resumed = asyncio.run(builder.update_async(batch_source=second, batch_train_indices=range(8, 16),
                              batch_source_audit=_batch_audit(second, tuple(range(8, 16)))))
        assert resumed.snapshot_id == b.snapshot_id
        assert counts == (len(view.calls), len(merge.calls), builder.embedder.transport.calls)


@pytest.mark.parametrize("failure", [RequestCompletionLengthExceeded("too long"), RequestContextLengthExceeded("context full")])
def test_merge_exhaustion_keeps_units_and_commits_without_retry_on_resume(tmp_path, failure):
    with IncrementalStateStore(tmp_path / "state.sqlite3") as state:
        builder, _, merge = _builder(tmp_path, state, merge_failure=failure)
        source = _batch(0)
        result = asyncio.run(builder.update_async(batch_source=source, batch_train_indices=range(8),
                             batch_source_audit=_batch_audit(source, tuple(range(8)))))
        assert result.manifest["canonical_count"] == 8
        assert state.head_snapshot_id == result.snapshot_id
        assert state.connection.execute("SELECT COUNT(*) FROM canonical_merge_events").fetchone() == (0,)
        assert state.connection.execute("SELECT COUNT(*) FROM canonical_resolution_events WHERE status = 'MERGE_EXHAUSTED_NO_MERGE'").fetchone()[0] > 0
        calls = len(merge.calls)
        asyncio.run(builder.update_async(batch_source=source, batch_train_indices=range(8),
                    batch_source_audit=_batch_audit(source, tuple(range(8)))))
        assert len(merge.calls) == calls

@pytest.mark.parametrize("during_commit", [False, True])
@pytest.mark.parametrize("local_failure_stage", [None, "view", "merge"])
def test_publish_then_commit_interruption_resumes_original_artifacts(tmp_path, monkeypatch, during_commit, local_failure_stage):
    with IncrementalStateStore(tmp_path / "state.sqlite3") as state:
        failing = _OneRequestTransportFailureViewLLM(RequestRuntimeTimeout("temporary item outage"))
        if local_failure_stage == "merge":
            failing.stage = "merge"
        builder, view, merge = _builder(tmp_path, state,
            view_llm=failing if local_failure_stage == "view" else None,
            merge_llm=failing if local_failure_stage == "merge" else None)
        batch = _batch(0)
        kwargs = dict(batch_source=batch, batch_train_indices=range(8),
                      batch_source_audit=_batch_audit(batch, tuple(range(8))))
        original = IncrementalGraphBuilder._commit_state
        if during_commit:
            state.connection.execute("""CREATE TRIGGER interrupt_commit BEFORE INSERT ON canonical_heads
                BEGIN SELECT RAISE(ABORT, 'synthetic transaction interruption'); END""")
        else:
            def stop(**kwargs):
                raise RuntimeError("synthetic publish interruption")
            monkeypatch.setattr(IncrementalGraphBuilder, "_commit_state", staticmethod(stop))
        with pytest.raises(Exception, match="synthetic .*interruption"):
            asyncio.run(builder.update_async(**kwargs))
        assert state.head_snapshot_id is None
        published = {p: p.read_bytes() for p in (tmp_path / "snapshots").glob("*/*") if p.is_file()}
        assert published
        if local_failure_stage is not None:
            audit_path = next(p for p in published if p.name == "canonical_audit.json")
            assert any(row["stage"] == local_failure_stage.upper() for row in json.loads(published[audit_path])["resolution_events"])
            # The service is now healthy: recovery must NOT reconsider a published
            # item-local terminal result, even when an identical later request succeeded.
            failing.failed_request_id = "service-recovered"
        llm_calls = len(view.calls), len(merge.calls)
        if during_commit:
            state.connection.execute("DROP TRIGGER interrupt_commit")
        else:
            monkeypatch.setattr(IncrementalGraphBuilder, "_commit_state", staticmethod(original))
        result = asyncio.run(builder.update_async(**kwargs))
        assert result.snapshot_id == state.head_snapshot_id
        assert {p: p.read_bytes() for p in published} == published
        assert (len(view.calls), len(merge.calls)) == llm_calls


def test_published_resume_does_not_apply_old_fallback_to_new_identical_occurrence(tmp_path, monkeypatch):
    with IncrementalStateStore(tmp_path / "state.sqlite3") as state:
        initial, _, _ = _builder(tmp_path, state, view_llm=_OneRequestTransportFailureViewLLM(
            RequestRuntimeTimeout("old local outage")))
        batch = _batch(0)
        asyncio.run(initial.update_async(batch_source=batch, batch_train_indices=range(8),
            batch_source_audit=_batch_audit(batch, tuple(range(8)))))
        next_batch = _source_for_workflows(tuple(
            WorkflowGraph(i, f"task-{i:03d}", f"Inspect synthetic spreadsheet {i:03d}", (_node(i - 8),), ())
            for i in range(8, 16)))
        builder, view, merge = _builder(tmp_path, state)
        kwargs = dict(batch_source=next_batch, batch_train_indices=range(8,16),
            batch_source_audit=_batch_audit(next_batch, tuple(range(8,16))))
        original = IncrementalGraphBuilder._commit_state
        def stop(**kwargs):
            raise RuntimeError("synthetic interruption")
        monkeypatch.setattr(IncrementalGraphBuilder, "_commit_state", staticmethod(stop))
        with pytest.raises(RuntimeError, match="synthetic interruption"):
            asyncio.run(builder.update_async(**kwargs))
        published = {p: p.read_bytes() for p in (tmp_path / "snapshots").glob("*/*") if p.is_file()}
        calls = len(view.calls), len(merge.calls)
        monkeypatch.setattr(IncrementalGraphBuilder, "_commit_state", staticmethod(original))
        result = asyncio.run(builder.update_async(**kwargs))
        assert state.head_snapshot_id == result.snapshot_id
        assert (len(view.calls), len(merge.calls)) == calls
        assert {p: p.read_bytes() for p in published} == published


def test_ram_union_interruption_replays_cached_decisions_without_duplicate_events(tmp_path, monkeypatch):
    batch = _batch(0)
    kwargs = dict(batch_source=batch, batch_train_indices=range(8),
                  batch_source_audit=_batch_audit(batch, tuple(range(8))))
    with IncrementalStateStore(tmp_path / "control.sqlite3") as control:
        clean, _, _ = _builder(tmp_path / "control", control)
        expected = asyncio.run(clean.update_async(**kwargs))
        expected_events = control.connection.execute(
            "SELECT left_canonical_id, right_canonical_id, child_canonical_id, apply_order "
            "FROM canonical_merge_events ORDER BY apply_order").fetchall()

    with IncrementalStateStore(tmp_path / "interrupted.sqlite3") as state:
        builder, view, merge = _builder(tmp_path / "interrupted", state)
        original = incremental_graph_module.canonical_unit_candidates

        def stop_after_union(source_id, aliases, vectors, **options):
            # The initial eight inputs are singletons. A multi-alias head proves
            # a union was applied in RAM, before any partition publication.
            if any(len(members) > 1 for members in aliases.values()):
                raise RuntimeError("synthetic RAM union interruption")
            return original(source_id, aliases, vectors, **options)

        monkeypatch.setattr(incremental_graph_module, "canonical_unit_candidates", stop_after_union)
        with pytest.raises(RuntimeError, match="synthetic RAM union interruption"):
            asyncio.run(builder.update_async(**kwargs))
        assert state.head_snapshot_id is None
        assert state.connection.execute("SELECT COUNT(*) FROM canonical_merge_events").fetchone() == (0,)
        assert merge.calls
        view_calls = len(view.calls)
        accepted_requests = [call["payload"] for call in merge.calls]
        monkeypatch.setattr(incremental_graph_module, "canonical_unit_candidates", original)
        resumed = asyncio.run(builder.update_async(**kwargs))
        assert resumed.manifest["canonical_partition_sha256"] == expected.manifest["canonical_partition_sha256"]
        assert state.connection.execute(
            "SELECT left_canonical_id, right_canonical_id, child_canonical_id, apply_order "
            "FROM canonical_merge_events ORDER BY apply_order").fetchall() == expected_events
        assert len(view.calls) == view_calls
        assert not any(call["payload"] in accepted_requests for call in merge.calls[len(accepted_requests):])
