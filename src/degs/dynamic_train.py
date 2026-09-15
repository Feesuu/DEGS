from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Protocol, Sequence
from datetime import datetime, timezone

from .contextual_binding import (
    ContextualBindingProducer,
    ExperienceExpectation,
    experience_expectation_from_dict,
    render_bound_guidance,
)
from .contextual_retrieval import ContextualRetrieval
from .contextual_runtime import retrieve_and_bind
from .core import EMBEDDING_MODEL, StrictEmbeddingAdapter, canonical_json_bytes
from .eir_graph import (
    CanonicalResolver,
    EpisodeCommitAudit,
    LearningDeltaApplier,
    compile_eir_experience_graph,
)
from .episode_evidence import EpisodeEvidence, EpisodeOutcome, EvidenceItem
from .episode_learning import (
    EpisodeReflectionProducer,
    LearningDelta,
    learning_delta_from_dict,
)
from .graph_dataset_contract import GraphDatasetContract
from .section_graph import ExperienceGraph
from .state_store import EIRStateStore
from .runtime_config import worker_count
from .validated_repair import SystemicProducerTransportFailure


EMPTY_GRAPH_SNAPSHOT_ID = "G0"
DYNAMIC_TRAIN_FORMAT = "degs_eir_dynamic_train_v1"
AGENT_WORKERS = worker_count("DEGS_AGENT_WORKERS", 8)
PRODUCER_WORKERS = worker_count("DEGS_PRODUCER_WORKERS", 32)


@dataclass(frozen=True)
class PreparedEpisode:
    train_index: int
    task_id: str
    query_text: str
    observable_context: tuple[EvidenceItem, ...]
    task: Any

    def __post_init__(self) -> None:
        if (
            type(self.train_index) is not int
            or self.train_index < 0
            or type(self.task_id) is not str
            or not self.task_id
            or type(self.query_text) is not str
            or not self.query_text.strip()
        ):
            raise ValueError("prepared episode differs")
        evidence_ids = [row.evidence_id for row in self.observable_context]
        if len(evidence_ids) != len(set(evidence_ids)) or "query:0" in evidence_ids:
            raise ValueError("prepared observable evidence differs")

    @property
    def retrieval_document(self) -> str:
        return canonical_json_bytes(
            {
                "query": {"evidence_id": "query:0", "content": self.query_text},
                "observable_context": [row.to_dict() for row in self.observable_context],
            }
        ).decode("utf-8")


class DatasetEpisodeAdapter(Protocol):
    async def prepare(self, task: Any) -> PreparedEpisode: ...

    async def execute_batch(
        self,
        *,
        prepared: Sequence[PreparedEpisode],
        read_snapshot_id: str,
        retrievals: Sequence[ContextualRetrieval],
        expectations: Sequence[Sequence[ExperienceExpectation]],
        guidance: Sequence[str],
    ) -> Sequence[EpisodeEvidence]: ...


@dataclass(frozen=True)
class DynamicBatchResult:
    batch_index: int
    snapshot_id: str
    parent_snapshot_id: str | None
    commit_audits: tuple[EpisodeCommitAudit, ...]
    binding_failures: Mapping[str, str]
    reflection_failures: Mapping[str, str]
    graph: ExperienceGraph
    canonical_versions: Mapping[str, int]
    runtime_metrics: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": DYNAMIC_TRAIN_FORMAT,
            "batch_index": self.batch_index,
            "snapshot_id": self.snapshot_id,
            "parent_snapshot_id": self.parent_snapshot_id,
            "commit_audits": [row.to_dict() for row in self.commit_audits],
            "binding_failures": dict(self.binding_failures),
            "reflection_failures": dict(self.reflection_failures),
            "graph_identity": self.graph.identity(),
            "canonical_versions": dict(self.canonical_versions),
            "runtime_metrics": dict(self.runtime_metrics),
        }


def _empty_graph() -> ExperienceGraph:
    source_digest = hashlib.sha256(
        canonical_json_bytes({"format": "degs_eir_empty_source_v1"})
    ).hexdigest()
    partition_digest = hashlib.sha256(
        canonical_json_bytes({"format": "degs_eir_empty_partition_v1"})
    ).hexdigest()
    body = {
        "format": "degs_eir_experience_graph_v1",
        "snapshot_id": EMPTY_GRAPH_SNAPSHOT_ID,
        "section_graphs_sha256": source_digest,
        "canonical_partition_sha256": partition_digest,
        "canonical_versions": {},
        "nodes": [],
        "edges": [],
    }
    graph_digest = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    return ExperienceGraph(
        (),
        (),
        source_digest,
        partition_digest,
        graph_digest,
        graph_format="degs_eir_experience_graph_v1",
        snapshot_id=EMPTY_GRAPH_SNAPSHOT_ID,
        canonical_versions={},
    )


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(canonical_json_bytes(dict(value)) + b"\n")
    temporary.replace(path)


def _bind_protocol(path: Path, value: Mapping[str, Any]) -> None:
    payload = dict(value)
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        operational = {
            "generation_base_url",
            "embedding_base_url",
            "agent_workers",
            "producer_workers",
            "canonical_view_workers",
            "canonical_merge_workers",
        }
        if type(existing) is not dict or {
            key: value for key, value in existing.items() if key not in operational
        } != {
            key: value for key, value in payload.items() if key not in operational
        }:
            raise ValueError("dynamic train method/data boundary differs")
    _write_json(path, payload)


def _usage_summary(path: Path) -> Mapping[str, int]:
    totals = {
        "request_count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    if not path.is_file():
        return totals
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid batch usage JSONL line {line_number}") from exc
        usage = row.get("usage") if type(row) is dict else None
        usage = usage if type(usage) is dict else {}
        prompt = usage.get("prompt_tokens", usage.get("input_tokens", 0))
        completion = usage.get("completion_tokens", usage.get("output_tokens", 0))
        total = usage.get("total_tokens", 0)
        prompt = prompt if type(prompt) is int and prompt >= 0 else 0
        completion = completion if type(completion) is int and completion >= 0 else 0
        total = total if type(total) is int and total >= 0 else prompt + completion
        totals["request_count"] += 1
        totals["prompt_tokens"] += prompt
        totals["completion_tokens"] += completion
        totals["total_tokens"] += total or prompt + completion
    return totals


async def _bounded_map(
    values: Sequence[Any],
    *,
    workers: int,
    worker: Any,
) -> tuple[Any, ...]:
    semaphore = asyncio.Semaphore(workers)

    async def one(value: Any) -> Any:
        async with semaphore:
            return await worker(value)

    return tuple(await asyncio.gather(*(one(value) for value in values)))


class DynamicTrainCampaign:
    def __init__(
        self,
        *,
        store: EIRStateStore,
        contract: GraphDatasetContract,
        embedding: StrictEmbeddingAdapter,
        binding: ContextualBindingProducer,
        reflection: EpisodeReflectionProducer,
        resolver: CanonicalResolver,
        adapter: DatasetEpisodeAdapter,
        output_dir: Path | str,
    ) -> None:
        if contract.batch_size != 8:
            raise ValueError("formal EIR train batch size differs")
        self.store = store
        self.contract = contract
        self.embedding = embedding
        self.binding = binding
        self.reflection = reflection
        self.resolver = resolver
        self.adapter = adapter
        self.output_dir = Path(output_dir).expanduser().absolute()

    def _episode_dir(self, *, batch_index: int, row: PreparedEpisode) -> Path:
        return (
            self.output_dir
            / "batches"
            / f"batch_{batch_index:02d}"
            / "episodes"
            / f"{row.train_index:04d}"
        )

    async def run(self, tasks: Sequence[Any]) -> tuple[DynamicBatchResult, ...]:
        if len(tasks) != self.contract.train_count:
            raise ValueError("dynamic train population differs from dataset contract")
        results: list[DynamicBatchResult] = []
        expected_parent: str | None = None
        for batch_index in range(self.contract.batch_count):
            committed = self.store.committed_snapshot_for_batch(batch_index)
            if committed is not None:
                snapshot_id, parent_snapshot_id, stored = committed
                if parent_snapshot_id != expected_parent:
                    raise ValueError("committed EIR snapshots are not a contiguous prefix")
                graph, canonical_versions = compile_eir_experience_graph(
                    self.store, snapshot_id=snapshot_id
                )
                audits = tuple(
                    EpisodeCommitAudit(
                        episode_id=str(row["episode_id"]),
                        train_index=int(row["train_index"]),
                        status=str(row["status"]),
                        update_counts=dict(row["update_counts"]),
                        new_source_node_count=int(row["new_source_node_count"]),
                        new_canonical_count=int(row["new_canonical_count"]),
                        absorbed_node_count=int(row["absorbed_node_count"]),
                        procedure_edge_count=int(row["procedure_edge_count"]),
                        discarded=tuple(map(str, row["discarded"])),
                        error=None if row.get("error") is None else str(row["error"]),
                    )
                    for row in stored.get("commit_audits", ())
                )
                result = DynamicBatchResult(
                    batch_index,
                    snapshot_id,
                    parent_snapshot_id,
                    audits,
                    dict(stored.get("binding_failures", {})),
                    dict(stored.get("reflection_failures", {})),
                    graph,
                    canonical_versions,
                    dict(stored.get("runtime_metrics", {})),
                )
                batch_dir = self.output_dir / "batches" / f"batch_{batch_index:02d}"
                _write_json(batch_dir / "manifest.json", result.to_dict())
                _write_json(batch_dir / "experience_graph.json", graph.to_dict())
                results.append(result)
                expected_parent = snapshot_id
                continue
            if self.store.head_snapshot_id != expected_parent:
                raise ValueError("EIR HEAD differs from the committed batch prefix")
            indices = self.contract.batch_indices(batch_index)
            batch_tasks = tuple(tasks[index] for index in indices)
            prepared = await _bounded_map(
                batch_tasks,
                workers=PRODUCER_WORKERS,
                worker=self.adapter.prepare,
            )
            result = await self.run_batch(
                batch_index=batch_index,
                prepared=prepared,
            )
            results.append(result)
            expected_parent = result.snapshot_id
        if self.store.head_snapshot_id != expected_parent:
            raise ValueError("final EIR HEAD differs from completed batches")
        return tuple(results)

    async def run_batch(
        self,
        *,
        batch_index: int,
        prepared: Sequence[PreparedEpisode],
    ) -> DynamicBatchResult:
        started_at = datetime.now(timezone.utc).isoformat()
        started = time.monotonic()
        expected_indices = self.contract.batch_indices(batch_index)
        if tuple(row.train_index for row in prepared) != expected_indices:
            raise ValueError("dynamic batch population or order differs")
        parent = self.store.head_snapshot_id
        read_snapshot_id = parent or EMPTY_GRAPH_SNAPSHOT_ID
        if parent is None:
            graph, canonical_versions = _empty_graph(), {}
        else:
            graph, canonical_versions = compile_eir_experience_graph(
                self.store, snapshot_id=parent
            )
        runtime_dir = self.output_dir / "batches" / f"batch_{batch_index:02d}" / "runtime"
        usage_path = runtime_dir / "usage.jsonl"
        os.environ["REACT_AGENT_USAGE_LOG"] = str(usage_path)
        os.environ["REACT_AGENT_RUNTIME_EVENT_LOG"] = str(
            runtime_dir / "runtime_events.jsonl"
        )

        cached_expectations: dict[str, tuple[ExperienceExpectation, ...]] = {}
        for row in prepared:
            expectation_path = self._episode_dir(
                batch_index=batch_index, row=row
            ) / "expectation.json"
            if not expectation_path.is_file():
                continue
            try:
                stored = json.loads(expectation_path.read_text(encoding="utf-8"))
                if type(stored) is not dict or set(stored) != {"expectations", "error"}:
                    raise ValueError("stored contextual expectation artifact differs")
                if stored["error"] is not None:
                    continue
                raw_rows = stored["expectations"]
                if type(raw_rows) is not list:
                    raise ValueError("stored contextual expectations differ")
                cached_expectations[row.task_id] = tuple(
                    experience_expectation_from_dict(item) for item in raw_rows
                )
            except (OSError, TypeError, ValueError):
                continue

        prepared_by_task_id = {row.task_id: row for row in prepared}

        async def checkpoint_binding(
            item: Any,
            retrieval: ContextualRetrieval,
            decision: tuple[ExperienceExpectation, ...],
        ) -> None:
            row = prepared_by_task_id[item.task_id]
            episode_dir = self._episode_dir(batch_index=batch_index, row=row)
            retrieval_path = episode_dir / "retrieval.json"
            _write_json(retrieval_path, retrieval.to_dict())
            _write_json(
                episode_dir / "expectation.json",
                {
                    "expectations": [entry.to_dict() for entry in decision],
                    "error": None,
                },
            )

        retrievals, expectations, failure_rows = await retrieve_and_bind(
            items=prepared,
            graph=graph,
            snapshot_id=read_snapshot_id,
            canonical_versions=canonical_versions,
            embedding=self.embedding,
            binding=self.binding,
            workers=PRODUCER_WORKERS,
            request_prefix="train",
            cached_expectations=cached_expectations,
            decision_callback=checkpoint_binding,
        )
        binding_failures = dict(failure_rows)
        for index, row in enumerate(prepared):
            episode_dir = self._episode_dir(batch_index=batch_index, row=row)
            retrieval_path = episode_dir / "retrieval.json"
            _write_json(retrieval_path, retrievals[index].to_dict())
            _write_json(
                episode_dir / "expectation.json",
                {
                    "expectations": [item.to_dict() for item in expectations[index]],
                    "error": binding_failures.get(row.task_id),
                },
            )

        executed = await self.adapter.execute_batch(
            prepared=prepared,
            read_snapshot_id=read_snapshot_id,
            retrievals=retrievals,
            expectations=expectations,
            guidance=tuple(render_bound_guidance(rows) for rows in expectations),
        )
        episodes = tuple(executed)
        if len(episodes) != len(prepared):
            raise ValueError("dataset adapter batch result count differs")
        by_train_index = {row.train_index: row for row in episodes}
        if len(by_train_index) != len(episodes):
            raise ValueError("dataset adapter returned duplicate episode indices")
        episodes = tuple(by_train_index[row.train_index] for row in prepared)
        for row, episode in zip(prepared, episodes, strict=True):
            if (
                episode.train_index != row.train_index
                or episode.task_id != row.task_id
                or episode.read_snapshot_id != read_snapshot_id
            ):
                raise ValueError("dataset adapter changed episode identity")
            _write_json(
                self._episode_dir(batch_index=batch_index, row=row)
                / "episode_evidence.json",
                episode.to_learning_payload(),
            )
        reflection_failures: dict[str, str] = {}
        active_experiences = {
            row.canonical_id: row.experience
            for row in self.store.active_canonicals(snapshot_id=parent)
        } if parent is not None else {}

        async def reflect(index: int) -> LearningDelta | None:
            episode = episodes[index]
            delta_path = self._episode_dir(
                batch_index=batch_index, row=prepared[index]
            ) / "learning_delta.json"
            try:
                if delta_path.is_file():
                    try:
                        return learning_delta_from_dict(
                            json.loads(delta_path.read_text(encoding="utf-8")),
                            episode=episode,
                            active_experiences=active_experiences,
                        )
                    except (OSError, TypeError, ValueError):
                        pass
                delta = await self.reflection.produce(
                    episode=episode, active_experiences=active_experiences
                )
                if delta is not None:
                    _write_json(delta_path, delta.to_dict())
                return delta
            except SystemicProducerTransportFailure:
                raise
            except Exception as exc:
                reflection_failures[episode.episode_id] = f"{type(exc).__name__}: {exc}"
                return None

        deltas = await _bounded_map(
            tuple(range(len(episodes))), workers=PRODUCER_WORKERS, worker=reflect
        )
        for row, episode, delta in zip(prepared, episodes, deltas, strict=True):
            episode_dir = self._episode_dir(batch_index=batch_index, row=row)
            if delta is not None:
                _write_json(episode_dir / "learning_delta.json", delta.to_dict())
            _write_json(
                episode_dir / "status.json",
                {
                    "episode_id": episode.episode_id,
                    "outcome": episode.outcome.value,
                    "binding_error": binding_failures.get(row.task_id),
                    "reflection_error": reflection_failures.get(episode.episode_id),
                    "learning_delta": delta is not None,
                },
            )
        operation_sha = hashlib.sha256(
            canonical_json_bytes(
                {
                    "format": DYNAMIC_TRAIN_FORMAT,
                    "batch_index": batch_index,
                    "parent_snapshot_id": parent,
                    "tasks": [
                        {"train_index": row.train_index, "task_id": row.task_id}
                        for row in prepared
                    ],
                }
            )
        ).hexdigest()
        snapshot_id = self.store.begin_snapshot(
            batch_index=batch_index,
            parent_snapshot_id=parent,
            operation_input_sha256=operation_sha,
        )
        applier = LearningDeltaApplier(self.store, self.resolver)
        runtime_metrics: dict[str, Any] = {}

        def commit_manifest(
            audits: tuple[EpisodeCommitAudit, ...],
        ) -> Mapping[str, Any]:
            runtime_metrics.update(
                {
                    "started_at": started_at,
                    "ended_at": datetime.now(timezone.utc).isoformat(),
                    "wall_seconds": time.monotonic() - started,
                    "usage": dict(_usage_summary(usage_path)),
                    "episode_outcome_counts": {
                        outcome.value: sum(
                            row.outcome is outcome for row in episodes
                        )
                        for outcome in EpisodeOutcome
                    },
                }
            )
            return {
                "format": DYNAMIC_TRAIN_FORMAT,
                "batch_index": batch_index,
                "parent_snapshot_id": parent,
                "read_snapshot_id": read_snapshot_id,
                "task_indices": list(expected_indices),
                "binding_failures": binding_failures,
                "reflection_failures": reflection_failures,
                "commit_audits": [row.to_dict() for row in audits],
                "runtime_metrics": dict(runtime_metrics),
            }

        commit_audits = await applier.apply_batch(
            snapshot_id=snapshot_id,
            episodes=tuple(zip(episodes, deltas, strict=True)),
            learning_errors=reflection_failures,
            commit_manifest_factory=commit_manifest,
        )
        next_graph, next_versions = compile_eir_experience_graph(
            self.store, snapshot_id=snapshot_id
        )
        result = DynamicBatchResult(
            batch_index,
            snapshot_id,
            parent,
            commit_audits,
            dict(binding_failures),
            dict(reflection_failures),
            next_graph,
            next_versions,
            runtime_metrics,
        )
        batch_dir = self.output_dir / "batches" / f"batch_{batch_index:02d}"
        _write_json(batch_dir / "manifest.json", result.to_dict())
        _write_json(batch_dir / "experience_graph.json", next_graph.to_dict())
        return result


__all__ = [
    "AGENT_WORKERS",
    "DYNAMIC_TRAIN_FORMAT",
    "DatasetEpisodeAdapter",
    "DynamicBatchResult",
    "DynamicTrainCampaign",
    "EMPTY_GRAPH_SNAPSHOT_ID",
    "PRODUCER_WORKERS",
    "PreparedEpisode",
]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run DEGS 0.78.0 batch-online SpreadsheetBench train learning."
    )
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--generation-base-url", required=True)
    parser.add_argument("--embedding-base-url", required=True)
    parser.add_argument("--model", choices=("Qwen3.5-9B-AWQ", "Qwen3.5-27B-AWQ"), required=True)
    parser.add_argument("--generation-api-key-env", default="DEGS_API_KEY")
    parser.add_argument("--embedding-api-key-env", default="DEGS_EMBEDDING_API_KEY")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    # Model-dependent producer/replay modules read DEGS_MODEL at import time.
    # Bind the explicit CLI profile before importing any of them.
    os.environ["DEGS_MODEL"] = args.model
    from react_agent.models import OpenAIClient

    from . import __version__
    from .canonicalize import (
        CANONICAL_CANDIDATE_K,
        CANONICAL_LLM_WORKERS,
        canonical_merge_response_schema,
        canonicalization_view_response_schema,
        openai_canonical_merge_llm,
        openai_canonical_view_llm,
    )
    from .contextual_binding import (
        BINDING_KIND,
        BINDING_PROMPT_SHA256,
        BINDING_PROTOCOL_FORMAT,
        contextual_binding_response_schema,
    )
    from .contextual_retrieval import (
        CONTEXTUAL_RETRIEVAL_METHOD,
        CONTEXTUAL_TOP_K,
        CONTEXT_NEIGHBORS_PER_ANCHOR,
    )
    from .eir_canonical import EIR_CANONICAL_VIEW_WORKERS, EIRCanonicalResolver
    from .episode_learning import (
        REFLECTION_KIND,
        REFLECTION_PROMPT_SHA256,
        REFLECTION_PROTOCOL_FORMAT,
        learning_delta_response_schema,
    )
    from .graph_dataset_contract import SPREADSHEETBENCH_GRAPH_CONTRACT
    from .spreadsheet_episode import SpreadsheetEpisodeAdapter, _tree_sha256
    from .transport import QwenEmbeddingHTTPTransport
    from .validated_repair import (
        OpenAIJsonObjectLLM,
        PRODUCER_RUNTIME_TIMEOUT_RETRIES,
        PRODUCER_TRANSPORT_RETRY_WAITS,
        REPAIR_SOURCE_TIMEOUT_SECONDS,
        _source_generation_config,
    )

    generation_key = os.environ.get(args.generation_api_key_env)
    embedding_key = os.environ.get(args.embedding_api_key_env)
    if not generation_key or not embedding_key:
        raise ValueError("generation and embedding API keys are required")
    run_dir = args.run_dir.expanduser().absolute()
    run_dir.mkdir(parents=True, exist_ok=True)
    with EIRStateStore(
        run_dir / "state" / "eir_state.sqlite3",
        dataset_contract=SPREADSHEETBENCH_GRAPH_CONTRACT,
    ) as store:
        embedder = StrictEmbeddingAdapter(
            QwenEmbeddingHTTPTransport(
                base_url=args.embedding_base_url,
                api_key=embedding_key,
            ),
            cache=store.embedding_cache(),
        )
        client = OpenAIClient(
            model=args.model,
            api_key=generation_key,
            base_url=args.generation_base_url,
            generation_config=_source_generation_config(),
            retry_times=PRODUCER_TRANSPORT_RETRY_WAITS,
            runtime_timeout_retries=PRODUCER_RUNTIME_TIMEOUT_RETRIES,
            timeout=REPAIR_SOURCE_TIMEOUT_SECONDS,
            trust_env=False,
        )
        binding_llm = OpenAIJsonObjectLLM(
            client,
            request_kind=BINDING_KIND,
            source_protocol_format=BINDING_PROTOCOL_FORMAT,
            prompt_sha256=BINDING_PROMPT_SHA256,
            response_schema_name="degs_contextual_binding_v1",
        )
        reflection_llm = OpenAIJsonObjectLLM(
            client,
            request_kind=REFLECTION_KIND,
            source_protocol_format=REFLECTION_PROTOCOL_FORMAT,
            prompt_sha256=REFLECTION_PROMPT_SHA256,
            response_schema_name="degs_episode_reflection_v1",
        )
        canonical_view_llm = openai_canonical_view_llm(client)
        canonical_merge_llm = openai_canonical_merge_llm(client)
        _bind_protocol(
            run_dir / "dynamic_protocol.json",
            {
                "format": "degs_eir_dynamic_protocol_v1",
                "method_version": __version__,
                "dataset_contract_id": SPREADSHEETBENCH_GRAPH_CONTRACT.identity,
                "dataset_json_sha256": hashlib.sha256(
                    (args.dataset_path / "dataset.json").read_bytes()
                ).hexdigest(),
                "dataset_tree_sha256": _tree_sha256(args.dataset_path),
                "model": args.model,
                "generation_base_url": args.generation_base_url.rstrip("/"),
                "embedding_base_url": args.embedding_base_url.rstrip("/"),
                "embedding_model": EMBEDDING_MODEL,
                "batch_size": SPREADSHEETBENCH_GRAPH_CONTRACT.batch_size,
                "agent_workers": AGENT_WORKERS,
                "producer_workers": PRODUCER_WORKERS,
                "canonical_view_workers": EIR_CANONICAL_VIEW_WORKERS,
                "canonical_merge_workers": CANONICAL_LLM_WORKERS,
                "canonical_candidate_k": CANONICAL_CANDIDATE_K,
                "retrieval": {
                    "method": CONTEXTUAL_RETRIEVAL_METHOD,
                    "top_k": CONTEXTUAL_TOP_K,
                    "neighbors_per_anchor": CONTEXT_NEIGHBORS_PER_ANCHOR,
                },
                "binding_producer": dict(binding_llm.protocol_identity),
                "binding_response_schema_sha256": hashlib.sha256(
                    canonical_json_bytes(contextual_binding_response_schema())
                ).hexdigest(),
                "reflection_producer": dict(reflection_llm.protocol_identity),
                "reflection_response_schema_sha256": hashlib.sha256(
                    canonical_json_bytes(learning_delta_response_schema())
                ).hexdigest(),
                "canonical_view_producer": dict(
                    canonical_view_llm.protocol_identity
                ),
                "canonical_view_response_schema_sha256": hashlib.sha256(
                    canonical_json_bytes(canonicalization_view_response_schema())
                ).hexdigest(),
                "canonical_merge_producer": dict(
                    canonical_merge_llm.protocol_identity
                ),
                "canonical_merge_response_schema_sha256": hashlib.sha256(
                    canonical_json_bytes(canonical_merge_response_schema())
                ).hexdigest(),
            },
        )
        adapter = SpreadsheetEpisodeAdapter(
            dataset_path=args.dataset_path,
            run_dir=run_dir,
            runtime_root=args.runtime_root,
            generation_base_url=args.generation_base_url,
            api_key=generation_key,
            model=args.model,
            dataset_contract_id=SPREADSHEETBENCH_GRAPH_CONTRACT.identity,
        )
        campaign = DynamicTrainCampaign(
            store=store,
            contract=SPREADSHEETBENCH_GRAPH_CONTRACT,
            embedding=embedder,
            binding=ContextualBindingProducer(binding_llm),
            reflection=EpisodeReflectionProducer(reflection_llm),
            resolver=EIRCanonicalResolver(
                view_llm=canonical_view_llm,
                merge_llm=canonical_merge_llm,
                embedding=embedder,
            ),
            adapter=adapter,
            output_dir=run_dir,
        )
        results = asyncio.run(campaign.run(adapter.load_tasks()))
    print(
        json.dumps(
            {
                "batch_count": len(results),
                "head_snapshot_id": results[-1].snapshot_id if results else None,
                "active_canonical_count": len(results[-1].graph.nodes) if results else 0,
                "edge_count": len(results[-1].graph.edges) if results else 0,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
