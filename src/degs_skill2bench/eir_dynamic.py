"""Skill2Bench adapter for the shared evidence-bounded EIR learning loop."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
import os
import time
from typing import Any, Awaitable, Callable, Mapping, Sequence
from datetime import datetime, timezone

from degs.contextual_binding import (
    ContextualBindingProducer,
    ExperienceExpectation,
    render_bound_guidance,
)
from degs.contextual_runtime import retrieve_and_bind
from degs.core import StrictEmbeddingAdapter, canonical_json_bytes
from degs.dynamic_train import (
    EMPTY_GRAPH_SNAPSHOT_ID,
    PreparedEpisode,
    _empty_graph,
    _usage_summary,
)
from degs.eir_graph import (
    CanonicalResolver,
    LearningDeltaApplier,
    compile_eir_experience_graph,
)
from degs.episode_evidence import EpisodeEvidence, EpisodeOutcome, EvidenceItem
from degs.episode_learning import EpisodeReflectionProducer
from degs.state_store import EIRStateStore

from .contract import SKILL2BENCH_MAX_STEPS, Skill2BenchProtocol
from .dataset import public_task_view
from .repair import step_outcome
from .runtime import render_agent_skill
from .step_evidence import source_step_fragments
from .step_units import public_step_view, step_task_id


RunPopulation = Callable[..., Awaitable[tuple[list[dict[str, Any]], list[dict[str, Any]]]]]
CollectEvidence = Callable[..., Awaitable[dict[int, list[dict[str, Any]]]]]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(canonical_json_bytes(value) + b"\n")
    temporary.replace(path)


def _prepared_steps(
    tasks: Sequence[Mapping[str, Any]], *, task_offset: int
) -> tuple[tuple[PreparedEpisode, ...], tuple[tuple[int, int], ...]]:
    items: list[PreparedEpisode] = []
    coordinates: list[tuple[int, int]] = []
    for local_index, raw_task in enumerate(tasks):
        task_index = task_offset + local_index
        public = public_task_view(raw_task)
        for step_number, question in enumerate(public["questions"], 1):
            if not question.strip():
                continue
            step = public_step_view(raw_task, step_number=step_number)
            workflow_index = task_index * SKILL2BENCH_MAX_STEPS + step_number - 1
            items.append(
                PreparedEpisode(
                    workflow_index,
                    step_task_id(task_index, step_number),
                    question,
                    (
                        EvidenceItem(
                            "context:scenario",
                            "scenario_background",
                            step["scenario_background"],
                        ),
                        EvidenceItem(
                            "context:target_step",
                            "independent_target_step",
                            dict(step["target_step"]),
                        ),
                    ),
                    {
                        "local_task_index": local_index,
                        "task_index": task_index,
                        "step_number": step_number,
                    },
                )
            )
            coordinates.append((local_index, step_number))
    return tuple(items), tuple(coordinates)


def _trace_items(
    events: Sequence[Mapping[str, Any]], *, prefix: str
) -> tuple[EvidenceItem, ...]:
    return tuple(
        EvidenceItem(f"trace:{prefix}:event:{index}", "agent_event", dict(event))
        for index, event in enumerate(events)
    )


def _episode(
    *,
    prepared: PreparedEpisode,
    read_snapshot_id: str,
    retrieval: Mapping[str, Any],
    expectations: Sequence[ExperienceExpectation],
    rollout: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    accepted_evidence: Sequence[Mapping[str, Any]],
    dataset_contract_id: str,
) -> EpisodeEvidence:
    step_number = int(prepared.task["step_number"])
    evaluated = next(
        (
            row
            for row in evaluation.get("steps", ())
            if isinstance(row, Mapping) and row.get("step") == step_number
        ),
        None,
    )
    episode_id = "episode_" + hashlib.sha256(
        canonical_json_bytes(
            {
                "train_index": prepared.train_index,
                "task_id": prepared.task_id,
                "read_snapshot_id": read_snapshot_id,
            }
        )
    ).hexdigest()[:24]
    if not isinstance(evaluated, Mapping):
        return EpisodeEvidence(
            episode_id,
            dataset_contract_id,
            prepared.train_index,
            prepared.task_id,
            read_snapshot_id,
            prepared.query_text,
            prepared.observable_context,
            dict(retrieval),
            tuple(expectations),
            (),
            (),
            EpisodeOutcome.ITEM_LOCAL_RUNTIME_FAILURE,
        )
    fragments = source_step_fragments(
        rollout, expected_steps=int(evaluation.get("num_steps") or len(evaluation["steps"]))
    ).get(step_number, ())
    if not fragments:
        raw_events = rollout.get("react_steps")
        fragments = (
            tuple(row for row in raw_events if isinstance(row, Mapping))
            if isinstance(raw_events, list)
            else ()
        )
    original_trace = _trace_items(fragments, prefix="original")
    verifier = (
        EvidenceItem(
            "verifier:original:step",
            "verifier_success" if step_outcome(evaluated) == "SUCCESS" else "verifier_failure",
            dict(evaluated),
        ),
    )
    if step_outcome(evaluated) == "SUCCESS" and original_trace:
        outcome = EpisodeOutcome.ORIGINAL_SUCCESS
        repair = None
    else:
        repair = next(
            (
                row
                for row in accepted_evidence
                if row.get("origin") == "VALIDATED_REPAIR"
                and row.get("step_index") == step_number
            ),
            None,
        )
        outcome = (
            EpisodeOutcome.REPAIR_SUCCESS
            if repair is not None and original_trace
            else EpisodeOutcome.UNRESOLVED_TASK_FAILURE
            if original_trace
            else EpisodeOutcome.ITEM_LOCAL_RUNTIME_FAILURE
        )
    if outcome is EpisodeOutcome.ITEM_LOCAL_RUNTIME_FAILURE:
        return EpisodeEvidence(
            episode_id,
            dataset_contract_id,
            prepared.train_index,
            prepared.task_id,
            read_snapshot_id,
            prepared.query_text,
            prepared.observable_context,
            dict(retrieval),
            tuple(expectations),
            (),
            (),
            outcome,
        )
    if outcome is EpisodeOutcome.REPAIR_SUCCESS:
        assert repair is not None
        replay = tuple(
            row
            for row in repair["successful_trajectory"]
            if isinstance(row, Mapping)
        )
        return EpisodeEvidence(
            episode_id,
            dataset_contract_id,
            prepared.train_index,
            prepared.task_id,
            read_snapshot_id,
            prepared.query_text,
            prepared.observable_context,
            dict(retrieval),
            tuple(expectations),
            original_trace,
            verifier,
            outcome,
            (
                EvidenceItem(
                    "patch:final",
                    "validated_repair_memory",
                    dict(repair["validated_repair_memory"]),
                ),
            ),
            _trace_items(replay, prefix="replay"),
            (
                EvidenceItem(
                    "verifier:replay:step",
                    "verifier_success",
                    {"step": step_number, "status": "SUCCESS"},
                ),
            ),
        )
    return EpisodeEvidence(
        episode_id,
        dataset_contract_id,
        prepared.train_index,
        prepared.task_id,
        read_snapshot_id,
        prepared.query_text,
        prepared.observable_context,
        dict(retrieval),
        tuple(expectations),
        original_trace,
        verifier,
        outcome,
    )


async def run_dynamic_training(
    *,
    train_tasks: Sequence[Mapping[str, Any]],
    root: Path,
    state: EIRStateStore,
    embedding: StrictEmbeddingAdapter,
    binding: ContextualBindingProducer,
    reflection: EpisodeReflectionProducer,
    resolver: CanonicalResolver,
    run_population: RunPopulation,
    collect_evidence: CollectEvidence,
    population_kwargs: Mapping[str, Any],
    evidence_kwargs: Mapping[str, Any],
    protocol: Skill2BenchProtocol,
) -> str:
    if len(train_tasks) != protocol.train_count:
        raise ValueError("Skill2Bench train population differs")
    expected_parent: str | None = None
    for batch_index, task_indices in enumerate(protocol.train_batches()):
        committed = state.committed_snapshot_for_batch(batch_index)
        if committed is not None:
            snapshot_id, parent_snapshot_id, stored = committed
            if parent_snapshot_id != expected_parent:
                raise ValueError("Skill2Bench committed snapshots are not contiguous")
            batch_root = root / "train/dynamic" / f"batch-{batch_index:02d}"
            artifact = batch_root / "manifest.json"
            expected_artifact = {**dict(stored), "snapshot_id": snapshot_id}
            _write_json(artifact, expected_artifact)
            graph_artifact = batch_root / "experience_graph.json"
            if not graph_artifact.is_file():
                committed_graph, _ = compile_eir_experience_graph(
                    state, snapshot_id=snapshot_id
                )
                _write_json(graph_artifact, committed_graph.to_dict())
            expected_parent = snapshot_id
            continue
        if state.head_snapshot_id != expected_parent:
            raise ValueError("Skill2Bench EIR HEAD differs from committed prefix")
        started_at = datetime.now(timezone.utc).isoformat()
        started = time.monotonic()
        usage_value = os.environ.get("REACT_AGENT_USAGE_LOG")
        usage_path = Path(usage_value) if usage_value else None
        usage_before = dict(_usage_summary(usage_path)) if usage_path else {}
        parent = state.head_snapshot_id
        read_snapshot_id = parent or EMPTY_GRAPH_SNAPSHOT_ID
        if parent is None:
            graph, versions = _empty_graph(), {}
        else:
            graph, versions = compile_eir_experience_graph(
                state, snapshot_id=parent
            )
        batch_tasks = tuple(train_tasks[index] for index in task_indices)
        prepared, coordinates = _prepared_steps(
            batch_tasks, task_offset=task_indices[0]
        )
        retrievals, expectations, binding_failures = await retrieve_and_bind(
            items=prepared,
            graph=graph,
            snapshot_id=read_snapshot_id,
            canonical_versions=versions,
            embedding=embedding,
            binding=binding,
            workers=protocol.producer_workers,
            request_prefix=f"skill2bench-train-batch-{batch_index:02d}",
        )
        by_coordinate = {
            coordinate: render_bound_guidance(expectations[index])
            for index, coordinate in enumerate(coordinates)
        }
        batch_root = root / "train/dynamic" / f"batch-{batch_index:02d}"
        skill_by_index: dict[int, Path] = {}
        for local_index, task in enumerate(batch_tasks):
            public = public_task_view(task)
            parts = [
                f"Step {step_number}:\n{guidance}"
                for step_number in range(1, len(public["questions"]) + 1)
                if (guidance := by_coordinate.get((local_index, step_number), ""))
            ]
            path = batch_root / "guidance" / f"{local_index:03d}/SKILL.md"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(render_agent_skill("\n\n".join(parts)))
            skill_by_index[local_index] = path
        rollouts, evaluations = await run_population(
            tasks=batch_tasks,
            output_root=batch_root / "original",
            skill_by_index=skill_by_index,
            **dict(population_kwargs),
        )
        evidence = await collect_evidence(
            tasks=batch_tasks,
            rollouts=rollouts,
            evaluations=evaluations,
            run_root=batch_root,
            **dict(evidence_kwargs),
        )
        episodes = tuple(
            _episode(
                prepared=item,
                read_snapshot_id=read_snapshot_id,
                retrieval=retrievals[index].to_dict(),
                expectations=expectations[index],
                rollout=rollouts[int(item.task["local_task_index"])],
                evaluation=evaluations[int(item.task["local_task_index"])],
                accepted_evidence=evidence[int(item.task["local_task_index"])],
                dataset_contract_id=protocol.graph_contract.identity,
            )
            for index, item in enumerate(prepared)
        )
        active = {
            row.canonical_id: row.experience
            for row in state.active_canonicals(snapshot_id=parent)
        } if parent is not None else {}
        semaphore = asyncio.Semaphore(protocol.producer_workers)
        reflection_failures: dict[str, str] = {}

        async def reflect(episode: EpisodeEvidence):
            if episode.outcome is EpisodeOutcome.ITEM_LOCAL_RUNTIME_FAILURE:
                return None
            try:
                async with semaphore:
                    return await reflection.produce(
                        episode=episode, active_experiences=active
                    )
            except Exception as exc:
                from degs.validated_repair import SystemicProducerTransportFailure

                if isinstance(exc, SystemicProducerTransportFailure):
                    raise
                reflection_failures[episode.episode_id] = (
                    f"{type(exc).__name__}: {exc}"
                )
                return None

        deltas = tuple(await asyncio.gather(*(reflect(row) for row in episodes)))
        operation_sha = hashlib.sha256(canonical_json_bytes({
            "batch_index": batch_index,
            "parent_snapshot_id": parent,
            "task_indices": list(task_indices),
            "step_ids": [row.task_id for row in prepared],
        })).hexdigest()
        snapshot_id = state.begin_snapshot(
            batch_index=batch_index,
            parent_snapshot_id=parent,
            operation_input_sha256=operation_sha,
        )
        runtime_metrics: dict[str, Any] = {}

        def commit_manifest(audits):
            usage_after = dict(_usage_summary(usage_path)) if usage_path else {}
            runtime_metrics.update(
                {
                    "started_at": started_at,
                    "ended_at": datetime.now(timezone.utc).isoformat(),
                    "wall_seconds": time.monotonic() - started,
                    "usage": {
                        key: usage_after.get(key, 0) - usage_before.get(key, 0)
                        for key in usage_after
                    },
                }
            )
            return {
                "format": "degs_skill2bench_eir_dynamic_batch_v1",
                "batch_index": batch_index,
                "task_indices": list(task_indices),
                "parent_snapshot_id": parent,
                "read_snapshot_id": read_snapshot_id,
                "binding_failures": dict(binding_failures),
                "reflection_failures": reflection_failures,
                "commit_audits": [row.to_dict() for row in audits],
                "runtime_metrics": dict(runtime_metrics),
            }

        audits = await LearningDeltaApplier(state, resolver).apply_batch(
            snapshot_id=snapshot_id,
            episodes=tuple(zip(episodes, deltas, strict=True)),
            learning_errors=reflection_failures,
            commit_manifest_factory=commit_manifest,
        )
        manifest = commit_manifest(audits)
        next_graph, _next_versions = compile_eir_experience_graph(
            state, snapshot_id=snapshot_id
        )
        _write_json(batch_root / "manifest.json", {**manifest, "snapshot_id": snapshot_id})
        _write_json(batch_root / "experience_graph.json", next_graph.to_dict())
        expected_parent = snapshot_id
    head = state.head_snapshot_id
    if head is None or head != expected_parent:
        raise RuntimeError("Skill2Bench EIR graph HEAD differs from completed batches")
    return head


__all__ = ["run_dynamic_training"]
