from __future__ import annotations

import hashlib
import importlib.resources
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from react_agent.models import OpenAIClient

from degs import bundle as shared
from degs.core import StrictEmbeddingAdapter, canonical_json_bytes
from degs.retrieval_store import RetrievalStore
from degs.section_graph import load_section_graphs
from degs.target_context import TargetEvidenceCard
from degs.validated_repair import OpenAIJsonObjectLLM
from degs.workflow_retrieval import (
    NEED_GRAPH_SYSTEM_PROMPT,
)

from .contract import Skill2BenchProtocol
from .dataset import public_task_view
from .source_extraction import validate_batch_source_audit
from .step_units import public_step_view


BUNDLE_FORMAT = "degs_skill2bench_step_retrieval_bundle_v1"
STEP_NEED_KIND = "degs_skill2bench_step_need_graph_v1"
STEP_NEED_PROTOCOL_FORMAT = "degs_skill2bench_step_need_graph_protocol_v1"
STEP_NEED_PROFILE = (
    importlib.resources.files("degs_skill2bench")
    .joinpath("resources", "SKILL2BENCH_STEP_NEED_PROFILE_V1.txt")
    .read_text(encoding="utf-8")
    .strip()
)
STEP_NEED_SYSTEM_PROMPT = f"{NEED_GRAPH_SYSTEM_PROMPT}\n\n{STEP_NEED_PROFILE}"
STEP_NEED_PROMPT_SHA256 = hashlib.sha256(STEP_NEED_SYSTEM_PROMPT.encode()).hexdigest()


def openai_step_need_llm(client: OpenAIClient) -> OpenAIJsonObjectLLM:
    return OpenAIJsonObjectLLM(
        client,
        request_kind=STEP_NEED_KIND,
        source_protocol_format=STEP_NEED_PROTOCOL_FORMAT,
        prompt_sha256=STEP_NEED_PROMPT_SHA256,
        response_schema_name="degs_skill2bench_step_need_graph_v1",
        expected_retry_times=shared.BUNDLE_TRANSPORT_RETRY_WAITS,
        expected_runtime_timeout_retries=shared.BUNDLE_RUNTIME_TIMEOUT_RETRIES,
    )


def _unavailable_card() -> TargetEvidenceCard:
    return TargetEvidenceCard(
        status="UNAVAILABLE",
        observations=(),
        input_sha256=None,
        input_relative_path=None,
        failure="Skill2Bench has no workbook input artifact",
    )


def _read_source_queries(
    snapshot_manifest_path: Path,
    protocol: Skill2BenchProtocol,
) -> tuple[dict[str, Any], ...]:
    manifest = json.loads(snapshot_manifest_path.read_text())
    graph_path = snapshot_manifest_path.parent / manifest["artifacts"][
        "accumulated_section_graphs"
    ]
    source = load_section_graphs(
        graph_path,
        dataset_contract=protocol.graph_contract,
        allow_empty=True,
    )
    return tuple(
        {
            "train_index": workflow.train_index,
            "task_id": workflow.task_id,
            "instruction": workflow.query_text,
        }
        for workflow in source.workflows
    )


def verify_step_retrieval_bundle(
    *,
    test_tasks: Sequence[Mapping[str, Any]],
    snapshot_manifest_path: Path,
    state_db_path: Path,
    output_dir: Path,
    protocol: Skill2BenchProtocol,
) -> tuple[dict[str, Any], ...]:
    if len(test_tasks) != protocol.test_count:
        raise ValueError("Skill2Bench test population differs")
    train_queries = _read_source_queries(snapshot_manifest_path, protocol)
    context = shared._load_snapshot(
        snapshot_manifest_path,
        state_db_path=state_db_path,
        train_queries=train_queries,
        dataset_contract=protocol.graph_contract,
        source_audit_validator=validate_batch_source_audit,
    )
    root = output_dir.expanduser().resolve()
    if not root.is_dir() or {path.name for path in root.iterdir()} != {
        "bundle_manifest.json",
        "experience.jsonl",
    }:
        raise ValueError("Skill2Bench retrieval bundle files differ")
    manifest_bytes = (root / "bundle_manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    unsigned = {key: value for key, value in manifest.items() if key != "self_sha256"}
    experience_bytes = (root / "experience.jsonl").read_bytes()
    lines = experience_bytes.splitlines()
    if (
        type(manifest) is not dict
        or manifest_bytes != canonical_json_bytes(manifest)
        or manifest.get("format") != BUNDLE_FORMAT
        or manifest.get("profile") != protocol.profile
        or manifest.get("model") != protocol.model
        or manifest.get("row_count") != protocol.test_count
        or manifest.get("source_snapshot") != dict(context.identity)
        or manifest.get("experience_sha256")
        != hashlib.sha256(experience_bytes).hexdigest()
        or manifest.get("selection_policy") != "DETERMINISTIC_TOP_RANKED_C0"
        or manifest.get("step_need_prompt_sha256") != STEP_NEED_PROMPT_SHA256
        or manifest.get("self_sha256")
        != hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
        or len(lines) != protocol.test_count
    ):
        raise ValueError("Skill2Bench retrieval bundle identity differs")
    rows: list[dict[str, Any]] = []
    step_count = 0
    for index, (line, raw_task) in enumerate(zip(lines, test_tasks, strict=True)):
        row = json.loads(line)
        public = public_task_view(raw_task)
        steps = row.get("steps") if type(row) is dict else None
        if (
            type(row) is not dict
            or line != canonical_json_bytes(row)
            or set(row) != {"test_index", "instance_id", "steps", "experience"}
            or row["test_index"] != index
            or row["instance_id"] != public["instance_id"]
            or type(steps) is not list
            or len(steps) != len(public["questions"])
            or type(row["experience"]) is not str
        ):
            raise ValueError(f"Skill2Bench retrieval row {index} differs")
        expected_parts = []
        for step_number, (step, question) in enumerate(
            zip(steps, public["questions"], strict=True), 1
        ):
            if (
                type(step) is not dict
                or set(step)
                != {
                    "step_number",
                    "question_sha256",
                    "status",
                    "experience",
                    "retrieval_audit",
                }
                or step["step_number"] != step_number
                or step["question_sha256"]
                != hashlib.sha256(question.encode()).hexdigest()
                or type(step["status"]) is not str
                or type(step["experience"]) is not str
                or type(step["retrieval_audit"]) is not dict
            ):
                raise ValueError(
                    f"Skill2Bench retrieval row {index} Step {step_number} differs"
                )
            if step["experience"]:
                expected_parts.append(
                    f"Step {step_number}:\n{step['experience']}"
                )
        if row["experience"] != "\n\n".join(expected_parts):
            raise ValueError(f"Skill2Bench retrieval row {index} assembly differs")
        step_count += len(steps)
        rows.append(row)
    if manifest.get("step_count") != step_count:
        raise ValueError("Skill2Bench retrieval Step count differs")
    return tuple(rows)


async def build_step_retrieval_bundle(
    *,
    test_tasks: Sequence[Mapping[str, Any]],
    snapshot_manifest_path: Path,
    state_db_path: Path,
    retrieval_cache_path: Path,
    output_dir: Path,
    embedding_transport: Any,
    need_llm: Any,
    clarification_llm: Any,
    selector_llm: Any,
    protocol: Skill2BenchProtocol,
) -> dict[str, Any]:
    if len(test_tasks) != protocol.test_count:
        raise ValueError("Skill2Bench test population differs")
    train_queries = _read_source_queries(snapshot_manifest_path, protocol)
    context = shared._load_snapshot(
        snapshot_manifest_path,
        state_db_path=state_db_path,
        train_queries=train_queries,
        dataset_contract=protocol.graph_contract,
        source_audit_validator=validate_batch_source_audit,
    )
    active: list[tuple[int, int, shared._Task, dict[str, Any]]] = []
    query_index = 0
    for task_index, raw_task in enumerate(test_tasks):
        public = public_task_view(raw_task)
        for step_number, question in enumerate(public["questions"], 1):
            if question.strip():
                step = public_step_view(raw_task, step_number=step_number)
                active.append(
                    (
                        task_index,
                        step_number,
                        shared._Task(
                            query_index,
                            task_index,
                            f"{public['instance_id']}::step-{step_number:02d}",
                            question,
                            "",
                            "",
                        ),
                        {
                            "scenario_background": step["scenario_background"],
                            "target_step": dict(step["target_step"]),
                        },
                    )
                )
                query_index += 1
    tasks = tuple(row[2] for row in active)
    payloads = {row[2].query_index: row[3] for row in active}
    target_cards = {task.query_index: _unavailable_card() for task in tasks}
    train_cards = {
        int(row["train_index"]): _unavailable_card() for row in train_queries
    }
    with RetrievalStore(retrieval_cache_path) as store:
        store.bind_embedding_endpoint(getattr(embedding_transport, "endpoint", ""))
        results, embedding_audit = await shared._build_tasks_clean_async(
            tasks=tasks,
            train_queries=train_queries,
            context=context,
            state=store,
            embedder=StrictEmbeddingAdapter(
                embedding_transport,
                cache=store.embedding_cache(),
            ),
            need_llm=need_llm,
            clarification_llm=clarification_llm,
            selector_llm=selector_llm,
            target_evidence_cards=target_cards,
            train_evidence_cards=train_cards,
            need_system_prompt=STEP_NEED_SYSTEM_PROMPT,
            need_prompt_sha256=STEP_NEED_PROMPT_SHA256,
            need_kind=STEP_NEED_KIND,
            need_payloads=payloads,
        )
    by_coordinate = {
        (task_index, step_number): result
        for (task_index, step_number, _task, _payload), result in zip(
            active, results, strict=True
        )
    }
    rows = []
    for task_index, raw_task in enumerate(test_tasks):
        public = public_task_view(raw_task)
        steps = []
        for step_number, question in enumerate(public["questions"], 1):
            result = by_coordinate.get((task_index, step_number))
            steps.append(
                {
                    "step_number": step_number,
                    "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
                    "status": "EMPTY_PUBLIC_QUESTION" if result is None else result.status,
                    "experience": "" if result is None else result.experience,
                    "retrieval_audit": {} if result is None else dict(result.audit),
                }
            )
        experience = "\n\n".join(
            f"Step {step['step_number']}:\n{step['experience']}"
            for step in steps
            if step["experience"]
        )
        rows.append(
            {
                "test_index": task_index,
                "instance_id": public["instance_id"],
                "steps": steps,
                "experience": experience,
            }
        )
    output = output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    payload = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    (output / "experience.jsonl").write_bytes(payload)
    body = {
        "format": BUNDLE_FORMAT,
        "profile": protocol.profile,
        "model": protocol.model,
        "row_count": len(rows),
        "step_count": sum(len(row["steps"]) for row in rows),
        "source_snapshot": dict(context.identity),
        "experience_sha256": hashlib.sha256(payload).hexdigest(),
        "embedding": embedding_audit,
        "selection_policy": "DETERMINISTIC_TOP_RANKED_C0",
        "step_need_prompt_sha256": STEP_NEED_PROMPT_SHA256,
    }
    manifest = {
        **body,
        "self_sha256": hashlib.sha256(canonical_json_bytes(body)).hexdigest(),
    }
    (output / "bundle_manifest.json").write_bytes(canonical_json_bytes(manifest))
    return manifest


__all__ = [
    "STEP_NEED_PROMPT_SHA256",
    "STEP_NEED_SYSTEM_PROMPT",
    "build_step_retrieval_bundle",
    "openai_step_need_llm",
    "verify_step_retrieval_bundle",
]
