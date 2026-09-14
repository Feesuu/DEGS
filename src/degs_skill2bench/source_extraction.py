from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib
import importlib.resources
from typing import Any, Mapping, Sequence

from openai import APIError
from react_agent.models import (
    RequestCompletionLengthExceeded,
    RequestContextLengthExceeded,
    RequestRuntimeTimeout,
)

from degs.core import canonical_json_bytes
from degs.section_graph import (
    SECTION_GRAPH_FORMAT,
    ExperienceEdge,
    ExperienceNode,
    SectionGraphSource,
    WorkflowGraph,
)
from degs.source_review import (
    ExperienceSourceReviewer,
    SOURCE_REVIEW_SYSTEM_PROMPT,
)
from degs.validated_repair import (
    JsonObjectLLM,
    OpenAIJsonObjectLLM,
    ProducerTransportGuard,
    _parse_llm_experience_graph,
    experience_node_schema,
    gather_cancel_on_error,
    producer_transport_failure_policy,
)

from .contract import SKILL2BENCH_MAX_STEPS, SKILL2BENCH_SOURCE_SPLIT
from .step_evidence import validate_step_evidence
from .step_units import (
    public_step_view,
    step_task_id,
    step_workflow_coordinates,
    step_workflow_ids_for_task_batch,
)


SOURCE_EXTRACTION_KIND = "extract_skill2bench_step_experience_v1"
SOURCE_AUDIT_FORMAT = "degs_skill2bench_step_source_audit_v1"
SOURCE_EXTRACTION_ATTEMPTS = 3
SOURCE_SYSTEM_PROMPT = (
    importlib.resources.files("degs_skill2bench")
    .joinpath("resources", "SKILL2BENCH_STEP_SOURCE_PROMPT_V1.txt")
    .read_text(encoding="utf-8")
    .strip()
)
SOURCE_PROMPT_SHA256 = hashlib.sha256(SOURCE_SYSTEM_PROMPT.encode()).hexdigest()
SOURCE_REVIEW_KIND = "review_skill2bench_step_experience_v1"
SOURCE_REVIEW_PROTOCOL_FORMAT = "degs_skill2bench_step_source_review_protocol_v1"
SOURCE_REVIEW_PROFILE = (
    importlib.resources.files("degs_skill2bench")
    .joinpath("resources", "SKILL2BENCH_STEP_REVIEW_PROFILE_V1.txt")
    .read_text(encoding="utf-8")
    .strip()
)
SOURCE_REVIEW_PROMPT = f"{SOURCE_REVIEW_SYSTEM_PROMPT}\n\n{SOURCE_REVIEW_PROFILE}"
SOURCE_REVIEW_PROMPT_SHA256 = hashlib.sha256(
    SOURCE_REVIEW_PROMPT.encode()
).hexdigest()


def step_source_reviewer(client: Any) -> ExperienceSourceReviewer:
    llm = OpenAIJsonObjectLLM(
        client,
        request_kind=SOURCE_REVIEW_KIND,
        source_protocol_format=SOURCE_REVIEW_PROTOCOL_FORMAT,
        prompt_sha256=SOURCE_REVIEW_PROMPT_SHA256,
        response_schema_name="degs_skill2bench_step_source_review_v1",
    )
    return ExperienceSourceReviewer(
        llm,
        request_kind=SOURCE_REVIEW_KIND,
        system_prompt=SOURCE_REVIEW_PROMPT,
    )


def source_response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "experience_nodes": {
                "type": "array",
                "items": experience_node_schema(),
            },
            "edges": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "integer", "minimum": 0},
                        "target": {"type": "integer", "minimum": 0},
                    },
                    "required": ["source", "target"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["experience_nodes", "edges"],
        "additionalProperties": False,
    }


@dataclass(frozen=True)
class StepExtraction:
    nodes: tuple[ExperienceNode, ...]
    edges: tuple[ExperienceEdge, ...]
    discarded_edge_reasons: tuple[str, ...]
    request_payload_sha256: str


class StepExperienceExtractor:
    def __init__(self, llm: JsonObjectLLM) -> None:
        self.llm = llm

    async def extract_async(
        self,
        *,
        task_id: str,
        public_step: Mapping[str, Any],
        evidence: Sequence[Mapping[str, Any]],
    ) -> StepExtraction:
        step_number = int(public_step["target_step"]["number"])
        checked = [
            validate_step_evidence(dict(row), expected_step=step_number)
            for row in evidence
        ]
        payload = {
            "scenario_background": public_step["scenario_background"],
            "target_step": dict(public_step["target_step"]),
            "accepted_evidence": checked,
        }
        payload_sha = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
        raw = await self.llm.complete_json_async(
            kind=SOURCE_EXTRACTION_KIND,
            request_id=task_id,
            system_prompt=SOURCE_SYSTEM_PROMPT,
            payload=payload,
            response_schema=source_response_schema(),
        )
        nodes, edges, discarded = _parse_llm_experience_graph(raw)
        return StepExtraction(nodes, edges, discarded, payload_sha)


def _review_evidence(
    public_step: Mapping[str, Any], evidence: Sequence[Mapping[str, Any]]
) -> tuple[str, dict[str, Any]]:
    step_number = int(public_step["target_step"]["number"])
    checked = [
        validate_step_evidence(dict(row), expected_step=step_number)
        for row in evidence
    ]
    mode = (
        "REPLAY_VALIDATED_SUCCESS"
        if any(row["origin"] == "VALIDATED_REPAIR" for row in checked)
        else "ORIGINAL_SUCCESS"
    )
    return mode, {
        "scenario_background": public_step["scenario_background"],
        "target_step": dict(public_step["target_step"]),
        "accepted_evidence": checked,
    }


def _workflow_sha256(workflow: WorkflowGraph) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "train_index": workflow.train_index,
                "task_id": workflow.task_id,
                "query_text": workflow.query_text,
                "experience_nodes": [row.to_dict() for row in workflow.experience_nodes],
                "edges": [row.to_dict() for row in workflow.edges],
            }
        )
    ).hexdigest()


def _source_payload(workflows: Sequence[WorkflowGraph]) -> dict[str, Any]:
    return {
        "format": SECTION_GRAPH_FORMAT,
        "source_split": SKILL2BENCH_SOURCE_SPLIT,
        "workflows": [
            {
                "train_index": row.train_index,
                "task_id": row.task_id,
                "query_text": row.query_text,
                "experience_nodes": [node.to_dict() for node in row.experience_nodes],
                "edges": [edge.to_dict() for edge in row.edges],
            }
            for row in workflows
        ],
    }


async def build_source_batch(
    *,
    batch_task_indices: Sequence[int],
    public_tasks: Mapping[int, Mapping[str, Any]],
    evidence_by_index: Mapping[int, Sequence[Mapping[str, Any]]],
    extractor: StepExperienceExtractor,
    reviewer: ExperienceSourceReviewer,
    workers: int = 16,
) -> tuple[SectionGraphSource, dict[str, Any]]:
    task_indices = tuple(batch_task_indices)
    if (
        not task_indices
        or tuple(sorted(task_indices)) != task_indices
        or len(task_indices) != len(set(task_indices))
        or set(public_tasks) != set(task_indices)
        or set(evidence_by_index) != set(task_indices)
        or workers <= 0
    ):
        raise ValueError("Skill2Bench source batch identity differs")
    workflow_ids = step_workflow_ids_for_task_batch(task_indices)
    semaphore = asyncio.Semaphore(workers)
    extraction_transport_guard = ProducerTransportGuard(
        stage="Skill2Bench Step source extraction"
    )
    review_transport_guard = ProducerTransportGuard(
        stage="Skill2Bench Step source review"
    )
    extraction_wave = await extraction_transport_guard.begin_wave()
    review_wave = await review_transport_guard.begin_wave()

    async def one(workflow_id: int) -> tuple[WorkflowGraph | None, dict[str, Any]]:
        task_index, step_number = step_workflow_coordinates(workflow_id)
        task = public_tasks[task_index]
        questions = task["questions"]
        task_id = step_task_id(task_index, step_number)
        base = {
            "train_index": workflow_id,
            "task_index": task_index,
            "step_number": step_number,
            "task_id": task_id,
        }
        if step_number > len(questions):
            return None, {**base, "status": "SOURCE_EXCLUDED_NO_STEP", "evidence_ids": [], "attempts": []}
        step_evidence = [
            validate_step_evidence(dict(row), expected_step=step_number)
            for row in evidence_by_index[task_index]
            if row.get("step_index") == step_number
        ]
        evidence_ids = [row["evidence_id"] for row in step_evidence]
        if not step_evidence:
            return None, {**base, "status": "SOURCE_EXCLUDED_NO_VALIDATED_SUCCESS", "evidence_ids": [], "attempts": []}
        question = str(questions[step_number - 1])
        if not question.strip():
            return None, {**base, "status": "SOURCE_EXCLUDED_EMPTY_PUBLIC_QUESTION", "evidence_ids": evidence_ids, "attempts": []}
        public = public_step_view(task, step_number=step_number, instance_id=task_id)
        attempts: list[dict[str, Any]] = []
        result: StepExtraction | None = None
        for attempt in range(1, SOURCE_EXTRACTION_ATTEMPTS + 1):
            try:
                async with semaphore:
                    result = await extractor.extract_async(
                        task_id=task_id,
                        public_step=public,
                        evidence=step_evidence,
                    )
                await extraction_transport_guard.record_success()
                attempts.append(
                    {"stage": "EXTRACTION", "attempt": attempt, "status": "ACCEPTED"}
                )
                if not result.nodes:
                    return None, {
                        **base,
                        "status": "SOURCE_EXCLUDED_NO_REUSABLE_EXPERIENCE",
                        "evidence_ids": evidence_ids,
                        "attempts": attempts,
                    }
                break
            except (APIError, RequestRuntimeTimeout) as exc:
                await extraction_transport_guard.record_failure(
                    request_id=f"extract-{task_id}", error=exc
                )
                attempts.append(
                    {
                        "stage": "EXTRACTION",
                        "attempt": attempt,
                        "status": "REJECTED",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
            except (
                RequestCompletionLengthExceeded,
                RequestContextLengthExceeded,
                ValueError,
            ) as exc:
                await extraction_transport_guard.record_success()
                attempts.append(
                    {
                        "stage": "EXTRACTION",
                        "attempt": attempt,
                        "status": "REJECTED",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
        if result is None:
            return None, {
                **base,
                "status": "SOURCE_EXCLUDED_GENERATION_FAILURE",
                "evidence_ids": evidence_ids,
                "attempts": attempts,
            }

        draft_graph = {
            "experience_nodes": [node.to_dict() for node in result.nodes],
            "edges": [edge.to_dict() for edge in result.edges],
        }
        nodes: tuple[ExperienceNode, ...] = ()
        edges: tuple[ExperienceEdge, ...] = ()
        review_status = "REVIEW_PENDING"
        review_ledger_status = "PENDING"
        review_decisions: list[dict[str, Any]] = []
        review_discarded: tuple[str, ...] = ()
        evidence_mode, review_evidence = _review_evidence(public, step_evidence)
        for attempt in range(1, SOURCE_EXTRACTION_ATTEMPTS + 1):
            try:
                async with semaphore:
                    reviewed = await reviewer.review_async(
                        request_id=f"review-{task_id}",
                        evidence_mode=evidence_mode,
                        evidence=review_evidence,
                        draft_nodes=result.nodes,
                        draft_edges=result.edges,
                    )
                await review_transport_guard.record_success()
                if reviewed.review_ledger_errors:
                    raise ValueError("; ".join(reviewed.review_ledger_errors))
                nodes, edges = reviewed.experience_nodes, reviewed.edges
                review_decisions = [
                    row.to_dict() for row in reviewed.review_decisions
                ]
                review_discarded = reviewed.discarded_edge_reasons
                review_status = "REVIEW_ACCEPTED"
                review_ledger_status = "ACCEPTED"
                attempts.append(
                    {"stage": "REVIEW", "attempt": attempt, "status": "ACCEPTED"}
                )
                break
            except (APIError, RequestRuntimeTimeout) as exc:
                await review_transport_guard.record_failure(
                    request_id=f"review-{task_id}", error=exc
                )
                attempts.append(
                    {
                        "stage": "REVIEW",
                        "attempt": attempt,
                        "status": "REJECTED",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
            except (
                RequestCompletionLengthExceeded,
                RequestContextLengthExceeded,
                ValueError,
            ) as exc:
                await review_transport_guard.record_success()
                attempts.append(
                    {
                        "stage": "REVIEW",
                        "attempt": attempt,
                        "status": "REJECTED",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
        if review_status != "REVIEW_ACCEPTED":
            return None, {
                **base,
                "status": "SOURCE_EXCLUDED_REVIEW_FAILURE",
                "evidence_ids": evidence_ids,
                "attempts": attempts,
            }
        if not nodes:
            return None, {
                **base,
                "status": "SOURCE_EXCLUDED_NO_REUSABLE_EXPERIENCE",
                "evidence_ids": evidence_ids,
                "attempts": attempts,
            }
        workflow = WorkflowGraph(workflow_id, task_id, question, nodes, edges)
        return workflow, {
            **base,
            "status": "INGESTED",
            "evidence_ids": evidence_ids,
            "attempts": attempts,
            "request_payload_sha256": result.request_payload_sha256,
            "workflow_sha256": _workflow_sha256(workflow),
            "discarded_edge_reasons": list(
                (*result.discarded_edge_reasons, *review_discarded)
            ),
            "draft_graph": draft_graph,
            "draft_graph_sha256": hashlib.sha256(
                canonical_json_bytes(draft_graph)
            ).hexdigest(),
            "source_review_status": review_status,
            "source_review_ledger_status": review_ledger_status,
            "source_review_decisions": review_decisions,
        }

    results = await gather_cancel_on_error(
        tuple(one(workflow_id) for workflow_id in workflow_ids)
    )
    await extraction_transport_guard.raise_if_systemic(extraction_wave)
    await review_transport_guard.raise_if_systemic(review_wave)
    workflows = tuple(row for row, _audit in results if row is not None)
    payload = _source_payload(workflows)
    source = SectionGraphSource(
        workflows,
        hashlib.sha256(canonical_json_bytes(payload)).hexdigest(),
        SKILL2BENCH_SOURCE_SPLIT,
    )
    row_audits = [row for _workflow, row in results]
    audit = {
        "format": SOURCE_AUDIT_FORMAT,
        "source_split": SKILL2BENCH_SOURCE_SPLIT,
        "batch_train_indices": list(workflow_ids),
        "batch_task_indices": list(task_indices),
        "section_graphs_sha256": source.sha256,
        "source_workflow_count": len(workflows),
        "source_extraction_workers": workers,
        "source_extraction_attempt_limit": SOURCE_EXTRACTION_ATTEMPTS,
        "transport_failure_policy": producer_transport_failure_policy(),
        "source_review_prompt_sha256": SOURCE_REVIEW_PROMPT_SHA256,
        "prompt_sha256": SOURCE_PROMPT_SHA256,
        "source_protocol": dict(extractor.llm.protocol_identity),
        "source_review_protocol": dict(reviewer.llm.protocol_identity),
        "rows": [row for row in row_audits if row["status"] == "INGESTED"],
        "exclusions": [row for row in row_audits if row["status"] != "INGESTED"],
    }
    return source, audit


def validate_batch_source_audit(
    *,
    batch_source: SectionGraphSource,
    batch_train_indices: Sequence[int],
    audit: Mapping[str, Any],
    expected_generation_endpoint: str | None = None,
) -> tuple[str, dict[int, str]]:
    indices = tuple(batch_train_indices)
    rows = audit.get("rows")
    exclusions = audit.get("exclusions")
    protocol = audit.get("source_protocol")
    if (
        type(audit) is not dict
        or set(audit) != {
            "format",
            "source_split",
            "batch_train_indices",
            "batch_task_indices",
            "section_graphs_sha256",
            "source_workflow_count",
            "source_extraction_workers",
            "source_extraction_attempt_limit",
            "transport_failure_policy",
            "source_review_prompt_sha256",
            "prompt_sha256",
            "source_protocol",
            "source_review_protocol",
            "rows",
            "exclusions",
        }
        or audit["format"] != SOURCE_AUDIT_FORMAT
        or audit["source_split"] != SKILL2BENCH_SOURCE_SPLIT
        or batch_source.source_split != SKILL2BENCH_SOURCE_SPLIT
        or audit["batch_train_indices"] != list(indices)
        or audit["batch_task_indices"]
        != sorted({index // SKILL2BENCH_MAX_STEPS for index in indices})
        or audit["section_graphs_sha256"] != batch_source.sha256
        or audit["source_workflow_count"] != len(batch_source.workflows)
        or audit["source_extraction_attempt_limit"] != SOURCE_EXTRACTION_ATTEMPTS
        or audit["transport_failure_policy"] != producer_transport_failure_policy()
        or audit["prompt_sha256"] != SOURCE_PROMPT_SHA256
        or audit["source_review_prompt_sha256"] != SOURCE_REVIEW_PROMPT_SHA256
        or type(protocol) is not dict
        or type(audit["source_review_protocol"]) is not dict
        or type(rows) is not list
        or type(exclusions) is not list
        or len(rows) + len(exclusions) != len(indices)
        or (
            expected_generation_endpoint is not None
            and protocol.get("service_url", "").rstrip("/")
            != expected_generation_endpoint.rstrip("/")
        )
        or (
            expected_generation_endpoint is not None
            and audit["source_review_protocol"].get("service_url", "").rstrip("/")
            != expected_generation_endpoint.rstrip("/")
        )
    ):
        raise ValueError("Skill2Bench batch source audit identity differs")
    statuses: dict[int, str] = {}
    workflows = batch_source.workflow_by_index
    by_index = {
        int(row["train_index"]): row
        for row in (*rows, *exclusions)
        if type(row) is dict and type(row.get("train_index")) is int
    }
    if set(by_index) != set(indices):
        raise ValueError("Skill2Bench source audit coverage differs")
    for index in indices:
        row = by_index[index]
        if type(row) is not dict or row.get("train_index") != index:
            raise ValueError("Skill2Bench source audit row differs")
        status = row.get("status")
        if status not in {
            "INGESTED",
            "SOURCE_EXCLUDED_NO_VALIDATED_SUCCESS",
            "SOURCE_EXCLUDED_GENERATION_FAILURE",
            "SOURCE_EXCLUDED_NO_REUSABLE_EXPERIENCE",
            "SOURCE_EXCLUDED_NO_STEP",
            "SOURCE_EXCLUDED_EMPTY_PUBLIC_QUESTION",
            "SOURCE_EXCLUDED_REVIEW_FAILURE",
        }:
            raise ValueError("Skill2Bench source status differs")
        if (status == "INGESTED") != (index in workflows):
            raise ValueError("Skill2Bench source workflow status differs")
        if status == "INGESTED" and (
            row.get("source_review_status") != "REVIEW_ACCEPTED"
            or row.get("source_review_ledger_status") != "ACCEPTED"
        ):
            raise ValueError("Skill2Bench source review acceptance differs")
        statuses[index] = status
    if set(workflows) - set(indices):
        raise ValueError("Skill2Bench source contains an out-of-batch workflow")
    return hashlib.sha256(canonical_json_bytes(dict(audit))).hexdigest(), statuses


__all__ = [
    "SOURCE_PROMPT_SHA256",
    "SOURCE_REVIEW_PROMPT_SHA256",
    "SOURCE_SYSTEM_PROMPT",
    "StepExperienceExtractor",
    "build_source_batch",
    "source_response_schema",
    "step_source_reviewer",
    "validate_batch_source_audit",
]
