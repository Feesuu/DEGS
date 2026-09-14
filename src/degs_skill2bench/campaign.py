from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import hashlib
import os
from pathlib import Path
import time
from typing import Any, Iterator, Mapping, Sequence

from openai import APIError
from react_agent.models import OpenAIClient

from degs.canonicalize import openai_canonical_merge_llm, openai_canonical_view_llm
from degs.core import StrictEmbeddingAdapter, canonical_json_bytes
from degs.incremental_graph import IncrementalGraphBuilder
from degs.retrieval_clarification import openai_retrieval_clarification_llm
from degs.state_store import IncrementalStateStore
from degs.transport import QwenEmbeddingHTTPTransport
from degs.validated_repair import (
    OpenAIJsonObjectLLM,
    ProducerTransportGuard,
    REPAIR_SOURCE_MODEL,
    SystemicProducerTransportFailure,
    _source_generation_config,
    gather_cancel_on_error,
    producer_transport_failure_policy,
)
from degs.workflow_retrieval import openai_selector_llm

from .contract import (
    BASELINE_PYTHON_SHA256,
    OFFICIAL_EVALUATOR_SHA256,
    Skill2BenchProtocol,
    skill2bench_protocol,
)
from .dataset import load_split, public_task_view
from .repair import (
    PATCH_KIND,
    PATCH_PROMPT_SHA256,
    PATCH_PROTOCOL_FORMAT,
    failed_step_payload,
    produce_patch,
    render_repair_skill,
    step_outcome,
)
from .retrieval import (
    STEP_NEED_PROMPT_SHA256,
    build_step_retrieval_bundle,
    openai_step_need_llm,
    verify_step_retrieval_bundle,
)
from .runtime import (
    AGENT_DATASET_PROFILE_SHA256,
    aggregate_metrics,
    render_agent_skill,
    run_and_evaluate_task,
)
from .source_extraction import (
    SOURCE_EXTRACTION_KIND,
    SOURCE_PROMPT_SHA256,
    SOURCE_REVIEW_PROMPT_SHA256,
    StepExperienceExtractor,
    _source_payload,
    build_source_batch,
    validate_batch_source_audit,
    step_source_reviewer,
)
from .step_evidence import (
    original_success_record,
    validate_step_evidence,
    validated_repair_record,
)


CAMPAIGN_FORMAT = "degs_07741_skill2bench_campaign_v1"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _python_tree_sha256(root: Path) -> str:
    source = root.expanduser().resolve()
    rows = [
        {
            "path": path.relative_to(source).as_posix(),
            "sha256": _sha256_file(path),
        }
        for path in sorted(source.rglob("*.py"))
        if "__pycache__" not in path.parts
    ]
    if not rows:
        raise ValueError(f"Python runtime tree is empty: {source}")
    return hashlib.sha256(canonical_json_bytes(rows)).hexdigest()


def _official_evaluator_sha256(root: Path) -> str:
    source = root.expanduser().resolve()
    math_root = source / "evaluation/math_evaluation"
    paths = [
        source / "calculate_skill_entropy/calculate_entropy_w_outputs.py",
        *(
            math_root / name
            for name in ("parser.py", "grader.py", "utils.py", "examples.py")
        ),
        *(
            path
            for path in sorted((math_root / "latex2sympy").rglob("*.py"))
            if "tests" not in path.parts
            and "sandbox" not in path.parts
            and path.name != "setup.py"
        ),
    ]
    if any(not path.is_file() for path in paths):
        raise ValueError("Skill2Bench official evaluator dependency closure differs")
    rows = [
        {
            "path": path.relative_to(source).as_posix(),
            "sha256": _sha256_file(path),
        }
        for path in paths
    ]
    return hashlib.sha256(canonical_json_bytes(rows)).hexdigest()


def _runtime_identity(baseline_root: Path, official_evaluator_root: Path) -> dict[str, str]:
    baseline = baseline_root.expanduser().resolve()
    evaluator = official_evaluator_root.expanduser().resolve()
    required = (
        baseline / "skill2bench/agent.py",
        baseline / "skill2bench/evaluator.py",
        baseline / "skill2bench/metrics.py",
        evaluator / "calculate_skill_entropy/calculate_entropy_w_outputs.py",
    )
    if any(not path.is_file() for path in required):
        raise ValueError("Skill2Bench runtime or official evaluator differs")
    identity = {
        "baseline_python_sha256": _python_tree_sha256(baseline),
        "official_evaluator_sha256": _official_evaluator_sha256(evaluator),
    }
    if identity != {
        "baseline_python_sha256": BASELINE_PYTHON_SHA256,
        "official_evaluator_sha256": OFFICIAL_EVALUATOR_SHA256,
    }:
        raise ValueError("Skill2Bench pinned runtime identity differs")
    return identity


def _population_output_sha256(
    rollout: Mapping[str, Any], evaluation: Mapping[str, Any]
) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            {"rollout": dict(rollout), "evaluation": dict(evaluation)}
        )
    ).hexdigest()


def _load_population_item(
    artifact: Path, receipt: Path, *, request_sha256: str, index: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not artifact.is_file() or not receipt.is_file():
        raise ValueError(f"Skill2Bench cached item {index} is incomplete")
    cached = json.loads(artifact.read_text())
    recorded = json.loads(receipt.read_text())
    if set(cached) != {"request_sha256", "output_sha256", "rollout", "evaluation"}:
        raise ValueError(f"Skill2Bench cached item {index} fields differ")
    output_sha256 = _population_output_sha256(
        cached["rollout"], cached["evaluation"]
    )
    expected = {
        "request_sha256": request_sha256,
        "output_sha256": output_sha256,
    }
    if (
        cached["request_sha256"] != request_sha256
        or cached["output_sha256"] != output_sha256
        or recorded != expected
    ):
        raise ValueError(f"Skill2Bench cached item {index} identity differs")
    return dict(cached["rollout"]), dict(cached["evaluation"])


def _write_population_item(
    artifact: Path,
    receipt: Path,
    *,
    request_sha256: str,
    rollout: Mapping[str, Any],
    evaluation: Mapping[str, Any],
) -> None:
    output_sha256 = _population_output_sha256(rollout, evaluation)
    _write_json(
        receipt,
        {"request_sha256": request_sha256, "output_sha256": output_sha256},
    )
    _write_json(
        artifact,
        {
            "request_sha256": request_sha256,
            "output_sha256": output_sha256,
            "rollout": dict(rollout),
            "evaluation": dict(evaluation),
        },
    )


def _population_failure(task: Mapping[str, Any], exc: Exception) -> tuple[dict[str, Any], dict[str, Any]]:
    instance_id = str(task.get("instance_id") or "")
    error = f"{type(exc).__name__}: {exc}"
    rollout = {
        "instance_id": instance_id,
        "answer": "",
        "agent_success": False,
        "error": error,
        "turns": 0,
        "react_steps": [],
        "tool_calls": {},
        "model_calls": 0,
        "tokens_estimated": 0,
        "input_tokens_estimated": 0,
        "output_tokens_estimated": 0,
        "item_local_failure": True,
    }
    steps = []
    for number, step in enumerate(task.get("steps", ()), 1):
        domain = str(step.get("domain") or "")
        steps.append(
            {
                "step": number,
                "correct": False,
                "score": 0.0,
                "status": "runtime_failure",
                "domain": domain,
                "skill": str(step.get("skill") or ""),
                "is_open_ended": bool(step.get("is_open_ended")),
                "prediction": "",
                "reference": str(step.get("solution") or ""),
            }
        )
    evaluation = {
        "instance_id": instance_id,
        "entropy_level": task.get("entropy_level"),
        "num_steps": len(steps),
        "success": False,
        "score": 0.0,
        "parse_failure": True,
        "partial_parse_failure": True,
        "parsed_step_ids": [],
        "expected_step_ids": list(range(1, len(steps) + 1)),
        "duplicate_step_ids": [],
        "evaluator_backend": "runtime_failure",
        "steps": steps,
        "item_local_failure": error,
    }
    return rollout, evaluation


_TRANSPORT_ERROR_TYPES = frozenset(
    {
        "APIConnectionError",
        "APITimeoutError",
        "RequestRuntimeTimeout",
    }
)


def _transport_failure(exc: Exception) -> Exception | None:
    status_code = getattr(exc, "status_code", None)
    error_type = getattr(exc, "worker_error_type", type(exc).__name__)
    if (
        isinstance(exc, APIError)
        or getattr(exc, "transport_failure", False) is True
        or status_code in {401, 403, 404}
        or error_type in _TRANSPORT_ERROR_TYPES
    ):
        return exc
    if _looks_systemic_text(f"{error_type}: {exc}"):
        return exc
    return None


def _looks_systemic_text(value: str) -> bool:
    text = value.lower()
    return any(
        marker in text
        for marker in (
            "apiconnectionerror",
            "apitimeouterror",
            "requestruntimetimeout",
            "connection refused",
            "connection error",
            "http 401",
            "http 403",
            "http 404",
            "error code: 401",
            "error code: 403",
            "error code: 404",
            "engine is dead",
        )
    )


@contextmanager
def _stage_timer(run_root: Path, stage: str) -> Iterator[None]:
    started = time.perf_counter()
    wall_started = datetime.now(timezone.utc).isoformat()
    try:
        yield
    finally:
        record = {
            "stage": stage,
            "started_at": wall_started,
            "elapsed_seconds": time.perf_counter() - started,
        }
        path = run_root / "usage/stages.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("ab") as stream:
            stream.write(canonical_json_bytes(record) + b"\n")


async def _run_population(
    *,
    tasks: Sequence[Mapping[str, Any]],
    output_root: Path,
    skill_by_index: Mapping[int, Path | None],
    baseline_root: Path,
    official_evaluator_root: Path,
    base_url: str,
    api_key: str,
    protocol: Skill2BenchProtocol,
    campaign_sha256: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    semaphore = asyncio.Semaphore(protocol.agent_workers)
    transport_guard = ProducerTransportGuard(stage="Skill2Bench Agent/evaluator")
    transport_wave = await transport_guard.begin_wave()

    async def one(index: int, task: Mapping[str, Any]):
        artifact = output_root / "items" / f"{index:03d}.json"
        receipt = output_root / "receipts" / f"{index:03d}.json"
        skill_path = skill_by_index.get(index)
        request = {
            "campaign_sha256": campaign_sha256,
            "index": index,
            "task_sha256": hashlib.sha256(canonical_json_bytes(dict(task))).hexdigest(),
            "skill_sha256": (
                None if skill_path is None else _sha256_file(skill_path)
            ),
        }
        request_sha = hashlib.sha256(canonical_json_bytes(request)).hexdigest()
        if artifact.is_file():
            rollout, evaluation = _load_population_item(
                artifact, receipt, request_sha256=request_sha, index=index
            )
            return (
                index,
                rollout,
                evaluation,
                None,
                artifact,
                receipt,
                request_sha,
                True,
            )
        try:
            async with semaphore:
                rollout, evaluation = await asyncio.to_thread(
                    run_and_evaluate_task,
                    task=task,
                    baseline_root=baseline_root,
                    official_evaluator_root=official_evaluator_root,
                    base_url=base_url,
                    api_key=api_key,
                    working_dir=output_root / "workspaces" / f"{index:03d}",
                    skill_path=skill_path,
                    protocol=protocol,
                )
            failure = None
        except Exception as exc:
            rollout, evaluation = _population_failure(task, exc)
            failure = exc
        return index, rollout, evaluation, failure, artifact, receipt, request_sha, False

    completed = await asyncio.gather(
        *(one(index, task) for index, task in enumerate(tasks))
    )
    completed = sorted(completed)
    transport_failures: list[Exception] = []
    systemic_failure: SystemicProducerTransportFailure | None = None
    for index, _rollout, _evaluation, failure, *_paths, from_cache in completed:
        if from_cache:
            continue
        if failure is None:
            await transport_guard.record_success()
        elif (classified_failure := _transport_failure(failure)) is not None:
            transport_failures.append(failure)
            try:
                await transport_guard.record_failure(
                    request_id=f"population-{index:03d}",
                    error=classified_failure,
                )
            except SystemicProducerTransportFailure as exc:
                systemic_failure = exc
    if systemic_failure is None:
        try:
            await transport_guard.raise_if_systemic(transport_wave)
        except SystemicProducerTransportFailure as exc:
            systemic_failure = exc
    for (
        index,
        rollout,
        evaluation,
        failure,
        artifact,
        receipt,
        request_sha,
        from_cache,
    ) in completed:
        if from_cache:
            continue
        if systemic_failure is not None and failure in transport_failures:
            continue
        _write_population_item(
            artifact,
            receipt,
            request_sha256=request_sha,
            rollout=rollout,
            evaluation=evaluation,
        )
    if systemic_failure is not None:
        raise systemic_failure from transport_failures[0]
    rollouts = [row[1] for row in completed]
    evaluations = [row[2] for row in completed]
    _write_jsonl(output_root / "rollouts.jsonl", rollouts)
    _write_jsonl(output_root / "evaluation.jsonl", evaluations)
    return rollouts, evaluations


def _producer_llm(
    client: OpenAIClient,
    *,
    kind: str,
    protocol_format: str,
    prompt_sha256: str,
    schema_name: str,
) -> OpenAIJsonObjectLLM:
    return OpenAIJsonObjectLLM(
        client,
        request_kind=kind,
        source_protocol_format=protocol_format,
        prompt_sha256=prompt_sha256,
        response_schema_name=schema_name,
    )


async def _collect_train_evidence(
    *,
    tasks: Sequence[Mapping[str, Any]],
    rollouts: Sequence[Mapping[str, Any]],
    evaluations: Sequence[Mapping[str, Any]],
    run_root: Path,
    patch_llm: Any,
    baseline_root: Path,
    official_evaluator_root: Path,
    base_url: str,
    api_key: str,
    protocol: Skill2BenchProtocol,
    campaign_sha256: str,
) -> dict[int, list[dict[str, Any]]]:
    evidence_by_index: dict[int, list[dict[str, Any]]] = {
        index: [] for index in range(len(tasks))
    }
    repair_targets = []
    for index, (task, rollout, evaluation) in enumerate(
        zip(tasks, rollouts, evaluations, strict=True)
    ):
        expected_steps = len(task["steps"])
        for evaluated_step in evaluation["steps"]:
            step_number = int(evaluated_step["step"])
            outcome = step_outcome(evaluated_step)
            if outcome == "SUCCESS":
                record = original_success_record(
                    evidence_id=f"train-{index:03d}::step-{step_number:02d}::original",
                    step_number=step_number,
                    rollout=rollout,
                    expected_steps=expected_steps,
                )
                if record is not None:
                    evidence_by_index[index].append(record)
            elif outcome == "FAILURE":
                repair_targets.append((index, task, rollout, evaluated_step))

    producer_semaphore = asyncio.Semaphore(protocol.producer_workers)
    agent_semaphore = asyncio.Semaphore(protocol.agent_workers)
    patch_transport_guard = ProducerTransportGuard(
        stage="Skill2Bench repair patch"
    )
    replay_transport_guard = ProducerTransportGuard(
        stage="Skill2Bench repair replay"
    )
    patch_transport_wave = await patch_transport_guard.begin_wave()
    replay_transport_wave = await replay_transport_guard.begin_wave()

    async def repair_one(index, task, failed_rollout, evaluated_step):
        step_number = int(evaluated_step["step"])
        target_root = run_root / "train/repairs" / f"{index:03d}" / f"step-{step_number:02d}"
        accepted_path = target_root / "accepted.json"
        repair_request = {
            "campaign_sha256": campaign_sha256,
            "task_sha256": hashlib.sha256(canonical_json_bytes(dict(task))).hexdigest(),
            "failed_rollout_sha256": hashlib.sha256(
                canonical_json_bytes(dict(failed_rollout))
            ).hexdigest(),
            "evaluated_step_sha256": hashlib.sha256(
                canonical_json_bytes(dict(evaluated_step))
            ).hexdigest(),
            "step_number": step_number,
        }
        repair_request_sha = hashlib.sha256(
            canonical_json_bytes(repair_request)
        ).hexdigest()
        if accepted_path.is_file():
            accepted_record = json.loads(accepted_path.read_text())
            if set(accepted_record) != {
                "request_sha256",
                "attempt",
                "result_sha256",
                "evidence_sha256",
            } or accepted_record["request_sha256"] != repair_request_sha:
                raise ValueError("Skill2Bench accepted repair identity differs")
            accepted_attempt = int(accepted_record["attempt"])
            result_path = target_root / f"attempt-{accepted_attempt:02d}/result.json"
            if (
                not result_path.is_file()
                or _sha256_file(result_path) != accepted_record["result_sha256"]
            ):
                raise ValueError("Skill2Bench accepted repair result differs")
            result = json.loads(result_path.read_text())
            memory = result["patch"]
            if (
                result.get("request_sha256") != repair_request_sha
                or result.get("attempt") != accepted_attempt
                or result.get("accepted") is not True
                or result.get("patch_sha256")
                != hashlib.sha256(canonical_json_bytes(memory)).hexdigest()
                or step_outcome(result["evaluation"]["steps"][step_number - 1])
                != "SUCCESS"
            ):
                raise ValueError("Skill2Bench accepted repair provenance differs")
            evidence = validated_repair_record(
                evidence_id=(
                    f"train-{index:03d}::step-{step_number:02d}::"
                    f"repair-{accepted_attempt:02d}"
                ),
                step_number=step_number,
                memory=memory,
                successful_replay=result["replay"],
            )
            if (
                hashlib.sha256(canonical_json_bytes(evidence)).hexdigest()
                != accepted_record["evidence_sha256"]
            ):
                raise ValueError("Skill2Bench accepted repair evidence differs")
            return index, validate_step_evidence(evidence, expected_step=step_number)
        attempts = []
        for attempt in range(1, 4):
            active_transport_guard = patch_transport_guard
            active_request_id = (
                f"patch-{index:03d}-step-{step_number:02d}"
            )
            try:
                patch_payload = failed_step_payload(
                    task=task,
                    rollout=failed_rollout,
                    evaluated_step=evaluated_step,
                    attempt_index=attempt,
                )
                async with producer_semaphore:
                    memory, patch_sha = await produce_patch(
                        llm=patch_llm,
                        task_id=f"train-{index:03d}-step-{step_number:02d}-attempt-{attempt:02d}",
                        payload=patch_payload,
                    )
                await patch_transport_guard.record_success()
                skill_path = target_root / f"attempt-{attempt:02d}/skill/SKILL.md"
                skill_path.parent.mkdir(parents=True, exist_ok=True)
                skill_path.write_text(render_repair_skill(step_number, memory))
                active_transport_guard = replay_transport_guard
                active_request_id = (
                    f"replay-{index:03d}-step-{step_number:02d}"
                )
                async with agent_semaphore:
                    replay, replay_evaluation = await asyncio.to_thread(
                        run_and_evaluate_task,
                        task=task,
                        baseline_root=baseline_root,
                        official_evaluator_root=official_evaluator_root,
                        base_url=base_url,
                        api_key=api_key,
                        working_dir=target_root / f"attempt-{attempt:02d}/workspace",
                        skill_path=skill_path,
                        protocol=protocol,
                    )
                await replay_transport_guard.record_success()
                replay_step = replay_evaluation["steps"][step_number - 1]
                accepted = step_outcome(replay_step) == "SUCCESS"
                attempt_record = {
                    "request_sha256": repair_request_sha,
                    "attempt": attempt,
                    "patch_sha256": patch_sha,
                    "patch": memory.to_dict(),
                    "accepted": accepted,
                    "replay": replay,
                    "evaluation": replay_evaluation,
                }
                _write_json(target_root / f"attempt-{attempt:02d}/result.json", attempt_record)
                attempts.append({"attempt": attempt, "accepted": accepted, "patch_sha256": patch_sha})
                if accepted:
                    evidence = validated_repair_record(
                        evidence_id=f"train-{index:03d}::step-{step_number:02d}::repair-{attempt:02d}",
                        step_number=step_number,
                        memory=memory,
                        successful_replay=replay,
                    )
                    _write_json(
                        accepted_path,
                        {
                            "request_sha256": repair_request_sha,
                            "attempt": attempt,
                            "result_sha256": _sha256_file(
                                target_root / f"attempt-{attempt:02d}/result.json"
                            ),
                            "evidence_sha256": hashlib.sha256(
                                canonical_json_bytes(evidence)
                            ).hexdigest(),
                        },
                    )
                    return index, evidence
            except Exception as exc:
                attempts.append(
                    {
                        "attempt": attempt,
                        "accepted": False,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                classified_failure = _transport_failure(exc)
                if classified_failure is not None:
                    await active_transport_guard.record_failure(
                        request_id=active_request_id,
                        error=classified_failure,
                    )
        _write_json(target_root / "failed.json", {"attempts": attempts})
        return index, None

    repaired = await gather_cancel_on_error(
        tuple(repair_one(*row) for row in repair_targets)
    )
    await patch_transport_guard.raise_if_systemic(patch_transport_wave)
    await replay_transport_guard.raise_if_systemic(replay_transport_wave)
    for index, evidence in repaired:
        if evidence is not None:
            evidence_by_index[index].append(evidence)
    _write_json(
        run_root / "train/evidence.json",
        {
            "tasks": [
                {"train_index": index, "evidence": evidence_by_index[index]}
                for index in range(len(tasks))
            ]
        },
    )
    return evidence_by_index


async def run_campaign(
    *,
    train_path: Path,
    test_path: Path,
    run_root: Path,
    baseline_root: Path,
    official_evaluator_root: Path,
    generation_base_url: str,
    generation_api_key_file: Path,
    embedding_base_url: str,
    embedding_api_key_file: Path,
    protocol: Skill2BenchProtocol,
) -> dict[str, Any]:
    if REPAIR_SOURCE_MODEL != protocol.model:
        raise ValueError(
            f"start through the Skill2Bench CLI so model-dependent modules use {protocol.model}"
        )
    root = run_root.expanduser().resolve()
    manifest_path = root / "manifest.json"
    if root.exists() and not manifest_path.is_file():
        if not root.is_dir() or any(root.iterdir()):
            raise FileExistsError(
                "Skill2Bench run root must be fresh or contain its manifest"
            )
    root.mkdir(parents=True, exist_ok=True)
    manifest_body = {
        "format": CAMPAIGN_FORMAT,
        "profile": protocol.profile,
        "model": protocol.model,
        "train_sha256": protocol.train_sha256,
        "test_sha256": protocol.test_sha256,
        "generation_base_url": generation_base_url.rstrip("/"),
        "embedding_base_url": embedding_base_url.rstrip("/"),
        "agent_workers": protocol.agent_workers,
        "producer_workers": protocol.producer_workers,
        "agent_max_tokens": None,
        "max_turns": protocol.max_turns,
        "thinking": protocol.thinking,
        "selection_policy": "DETERMINISTIC_TOP_RANKED_C0",
        "agent_dataset_profile_sha256": AGENT_DATASET_PROFILE_SHA256,
        "repair_patch_prompt_sha256": PATCH_PROMPT_SHA256,
        "source_prompt_sha256": SOURCE_PROMPT_SHA256,
        "source_review_prompt_sha256": SOURCE_REVIEW_PROMPT_SHA256,
        "step_need_prompt_sha256": STEP_NEED_PROMPT_SHA256,
        "graph_dataset_contract": protocol.graph_contract.to_dict(),
        "runtime": _runtime_identity(baseline_root, official_evaluator_root),
    }
    campaign_sha256 = hashlib.sha256(canonical_json_bytes(manifest_body)).hexdigest()
    manifest = {**manifest_body, "self_sha256": campaign_sha256}
    if manifest_path.is_file():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Skill2Bench campaign manifest differs")
    else:
        _write_json(manifest_path, manifest)
    train_tasks = load_split(train_path, split="train")
    test_tasks = load_split(test_path, split="test")
    api_key = generation_api_key_file.read_text().strip()
    embedding_key = embedding_api_key_file.read_text().strip()
    os.environ["REACT_AGENT_USAGE_LOG"] = str(root / "usage/llm_calls.jsonl")

    train_profile = root / "train/profile/SKILL.md"
    train_profile.parent.mkdir(parents=True, exist_ok=True)
    train_profile.write_text(render_agent_skill(""))

    with _stage_timer(root, "train_rollout_and_evaluation"):
        train_rollouts, train_evaluations = await _run_population(
            tasks=train_tasks,
            output_root=root / "train/original",
            skill_by_index={index: train_profile for index in range(len(train_tasks))},
            baseline_root=baseline_root,
            official_evaluator_root=official_evaluator_root,
            base_url=generation_base_url,
            api_key=api_key,
            protocol=protocol,
            campaign_sha256=campaign_sha256,
        )

    producer_client = OpenAIClient(
        model=protocol.model,
        api_key=api_key,
        base_url=generation_base_url,
        generation_config=_source_generation_config(32_000),
        retry_times=(5, 10, 30),
        runtime_timeout_retries=1,
        timeout=600,
        trust_env=False,
    )
    patch_llm = _producer_llm(
        producer_client,
        kind=PATCH_KIND,
        protocol_format=PATCH_PROTOCOL_FORMAT,
        prompt_sha256=PATCH_PROMPT_SHA256,
        schema_name="degs_skill2bench_repair_patch_v1",
    )
    with _stage_timer(root, "repair_and_evidence"):
        evidence_by_index = await _collect_train_evidence(
            tasks=train_tasks,
            rollouts=train_rollouts,
            evaluations=train_evaluations,
            run_root=root,
            patch_llm=patch_llm,
            baseline_root=baseline_root,
            official_evaluator_root=official_evaluator_root,
            base_url=generation_base_url,
            api_key=api_key,
            protocol=protocol,
            campaign_sha256=campaign_sha256,
        )

    source_llm = _producer_llm(
        producer_client,
        kind=SOURCE_EXTRACTION_KIND,
        protocol_format="degs_skill2bench_step_source_protocol_v1",
        prompt_sha256=SOURCE_PROMPT_SHA256,
        schema_name="degs_skill2bench_step_source_v1",
    )
    state_path = root / "graph/state.sqlite3"
    snapshot_root = root / "graph/snapshots"
    public_tasks = {
        index: public_task_view(task) for index, task in enumerate(train_tasks)
    }
    with _stage_timer(root, "dynamic_graph"):
        with IncrementalStateStore(
            state_path,
            dataset_contract=protocol.graph_contract,
        ) as state:
            builder = IncrementalGraphBuilder(
                state=state,
                snapshot_root=snapshot_root,
                embedder=StrictEmbeddingAdapter(
                    QwenEmbeddingHTTPTransport(
                        base_url=embedding_base_url,
                        api_key=embedding_key,
                    ),
                    cache=state.embedding_cache(),
                ),
                view_llm=openai_canonical_view_llm(producer_client),
                merge_llm=openai_canonical_merge_llm(producer_client),
                dataset_contract=protocol.graph_contract,
                source_audit_validator=validate_batch_source_audit,
            )
            for batch_index, task_indices in enumerate(protocol.train_batches()):
                batch_root = root / "graph/source_batches" / f"batch-{batch_index:02d}"
                source_path = batch_root / "source.json"
                audit_path = batch_root / "audit.json"
                reuse_cached_batch = source_path.is_file() and audit_path.is_file()
                if reuse_cached_batch:
                    from degs.section_graph import load_section_graphs

                    source = load_section_graphs(
                        source_path,
                        allow_empty=True,
                        dataset_contract=protocol.graph_contract,
                    )
                    audit = json.loads(audit_path.read_text())
                    reuse_cached_batch = (
                        audit.get("transport_failure_policy")
                        == producer_transport_failure_policy()
                    )
                if not reuse_cached_batch:
                    source, audit = await build_source_batch(
                        batch_task_indices=task_indices,
                        public_tasks={index: public_tasks[index] for index in task_indices},
                        evidence_by_index={index: evidence_by_index[index] for index in task_indices},
                        extractor=StepExperienceExtractor(source_llm),
                        reviewer=step_source_reviewer(producer_client),
                        workers=protocol.producer_workers,
                    )
                    batch_root.mkdir(parents=True, exist_ok=True)
                    source_path.write_bytes(
                        canonical_json_bytes(_source_payload(source.workflows))
                    )
                    audit_path.write_bytes(canonical_json_bytes(audit))
                await builder.update_async(
                    batch_source=source,
                    batch_train_indices=protocol.graph_contract.batch_indices(batch_index),
                    batch_source_audit=audit,
                )
            head = state.head_snapshot_id
            if head is None:
                raise RuntimeError("Skill2Bench graph has no committed snapshot")
    snapshot_manifest = snapshot_root / head / "snapshot_manifest.json"

    with _stage_timer(root, "step_retrieval"):
        bundle_dir = root / "retrieval/bundle"
        if not (bundle_dir / "bundle_manifest.json").is_file():
            await build_step_retrieval_bundle(
                test_tasks=test_tasks,
                snapshot_manifest_path=snapshot_manifest,
                state_db_path=state_path,
                retrieval_cache_path=root / "retrieval/cache.sqlite3",
                output_dir=bundle_dir,
                embedding_transport=QwenEmbeddingHTTPTransport(
                    base_url=embedding_base_url,
                    api_key=embedding_key,
                ),
                need_llm=openai_step_need_llm(producer_client),
                clarification_llm=openai_retrieval_clarification_llm(producer_client),
                selector_llm=openai_selector_llm(producer_client),
                protocol=protocol,
            )
    bundle_rows = verify_step_retrieval_bundle(
        test_tasks=test_tasks,
        snapshot_manifest_path=snapshot_manifest,
        state_db_path=state_path,
        output_dir=bundle_dir,
        protocol=protocol,
    )
    skill_by_index: dict[int, Path | None] = {}
    for index, row in enumerate(bundle_rows):
        experience = str(row["experience"] or "")
        skill_path = root / "retrieval/skills" / f"{index:03d}/SKILL.md"
        skill_path.parent.mkdir(parents=True, exist_ok=True)
        skill_path.write_text(render_agent_skill(experience))
        skill_by_index[index] = skill_path

    with _stage_timer(root, "test_rollout_and_evaluation"):
        test_rollouts, test_evaluations = await _run_population(
            tasks=test_tasks,
            output_root=root / "test",
            skill_by_index=skill_by_index,
            baseline_root=baseline_root,
            official_evaluator_root=official_evaluator_root,
            base_url=generation_base_url,
            api_key=api_key,
            protocol=protocol,
            campaign_sha256=campaign_sha256,
        )
    metrics = aggregate_metrics(
        test_evaluations,
        test_rollouts,
        baseline_root=baseline_root,
    )
    summary = {
        "format": "degs_07741_skill2bench_result_v1",
        "profile": protocol.profile,
        "model": protocol.model,
        "snapshot_id": head,
        "retrieval_bundle_sha256": json.loads(
            (bundle_dir / "bundle_manifest.json").read_text()
        )["self_sha256"],
        "metrics": metrics,
    }
    _write_json(root / "result.json", summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the complete DEGS Skill2Bench pipeline.")
    parser.add_argument("--profile", choices=("9b", "27b"), required=True)
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--test-path", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--official-evaluator-root", type=Path, required=True)
    parser.add_argument("--generation-base-url", required=True)
    parser.add_argument("--generation-api-key-file", type=Path, required=True)
    parser.add_argument("--embedding-base-url", required=True)
    parser.add_argument("--embedding-api-key-file", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = asyncio.run(
        run_campaign(
            train_path=args.train_path,
            test_path=args.test_path,
            run_root=args.run_root,
            baseline_root=args.baseline_root,
            official_evaluator_root=args.official_evaluator_root,
            generation_base_url=args.generation_base_url,
            generation_api_key_file=args.generation_api_key_file,
            embedding_base_url=args.embedding_base_url,
            embedding_api_key_file=args.embedding_api_key_file,
            protocol=skill2bench_protocol(args.profile),
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
