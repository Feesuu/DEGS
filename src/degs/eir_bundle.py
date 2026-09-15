from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from react_agent.models import OpenAIClient

from .contextual_binding import (
    BINDING_KIND,
    BINDING_PROMPT_SHA256,
    BINDING_PROTOCOL_FORMAT,
    ContextualBindingProducer,
    ExperienceExpectation,
    contextual_binding_response_schema,
    experience_expectation_from_dict,
    render_bound_guidance,
)
from .contextual_retrieval import (
    CONTEXTUAL_RETRIEVAL_METHOD,
    CONTEXTUAL_TOP_K,
    CONTEXT_NEIGHBORS_PER_ANCHOR,
    ContextualRetrieval,
)
from .contextual_runtime import retrieve_and_bind
from .core import StrictEmbeddingAdapter, canonical_json_bytes
from .dataset import _load_development_harness_records
from .dynamic_train import PRODUCER_WORKERS, PreparedEpisode, _bounded_map
from .eir_graph import compile_eir_experience_graph
from .episode_evidence import EvidenceItem
from .graph_dataset_contract import GraphDatasetContract, SPREADSHEETBENCH_GRAPH_CONTRACT
from .state_store import EIRStateStore
from .target_context import (
    TargetContextUnavailable,
    build_target_evidence_card,
    unavailable_target_evidence_card,
)
from .transport import QwenEmbeddingHTTPTransport
from .validated_repair import (
    OpenAIJsonObjectLLM,
    PRODUCER_RUNTIME_TIMEOUT_RETRIES,
    PRODUCER_TRANSPORT_RETRY_WAITS,
    REPAIR_SOURCE_TIMEOUT_SECONDS,
    _source_generation_config,
)


EIR_BUNDLE_FORMAT = "degs_eir_contextual_guidance_bundle_v1"
EIR_GUIDANCE_ROW_FORMAT = "degs_eir_guidance_row_v1"


@dataclass(frozen=True)
class VerifiedEIRGuidanceBundle:
    root: Path
    manifest: Mapping[str, Any]


def verify_contextual_bundle(
    *,
    output_dir: Path,
    expected_instance_ids: Sequence[str],
    expected_state_db: Path,
    expected_dataset: str,
) -> VerifiedEIRGuidanceBundle:
    root = output_dir.expanduser().resolve()
    if not root.is_dir() or {path.name for path in root.iterdir()} != {
        "bundle_manifest.json",
        "experience.jsonl",
    }:
        raise ValueError("EIR guidance bundle files differ")
    manifest_bytes = (root / "bundle_manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    unsigned = {key: value for key, value in manifest.items() if key != "self_sha256"}
    payload = (root / "experience.jsonl").read_bytes()
    lines = payload.splitlines()
    if (
        type(manifest) is not dict
        or manifest_bytes != canonical_json_bytes(manifest) + b"\n"
        or manifest.get("format") != EIR_BUNDLE_FORMAT
        or manifest.get("dataset") != expected_dataset
        or manifest.get("state_db") != str(expected_state_db.expanduser().absolute())
        or manifest.get("experience_file") != "experience.jsonl"
        or manifest.get("experience_sha256") != hashlib.sha256(payload).hexdigest()
        or manifest.get("row_count") != len(expected_instance_ids)
        or manifest.get("fixed_denominator") != len(expected_instance_ids)
        or manifest.get("self_sha256")
        != hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
        or len(lines) != len(expected_instance_ids)
    ):
        raise ValueError("EIR guidance bundle identity differs")
    actual_ids: list[str] = []
    for index, line in enumerate(lines):
        row = json.loads(line)
        if (
            type(row) is not dict
            or line != canonical_json_bytes(row)
            or row.get("format") != EIR_GUIDANCE_ROW_FORMAT
            or type(row.get("dataset_index")) is not int
            or type(row.get("instance_id")) is not str
            or type(row.get("experience")) is not str
            or type(row.get("retrieval")) is not dict
            or type(row.get("expectations")) is not list
            or row.get("status") not in {"COMPLETE", "BINDING_FAILURE"}
        ):
            raise ValueError(f"EIR guidance row {index} differs")
        actual_ids.append(row["instance_id"])
    if actual_ids != list(expected_instance_ids) or len(actual_ids) != len(set(actual_ids)):
        raise ValueError("EIR guidance population differs")
    return VerifiedEIRGuidanceBundle(root, manifest)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(canonical_json_bytes(value) + b"\n")
    temporary.replace(path)


async def _prepare(
    record: Mapping[str, Any], *, dataset_path: Path
) -> PreparedEpisode:
    try:
        card = await asyncio.to_thread(
            build_target_evidence_card,
            dataset_path=dataset_path / "dataset.json",
            spreadsheet_path=str(record["spreadsheet_path"]),
            answer_position=str(record["answer_position"]),
            instruction=str(record["instruction"]),
        )
    except TargetContextUnavailable as exc:
        card = unavailable_target_evidence_card(exc)
    return PreparedEpisode(
        int(record["dataset_index"]),
        str(record["task_id"]),
        str(record["instruction"]),
        tuple(
            EvidenceItem(
                f"context:{row['evidence_id']}",
                str(row["kind"]),
                json.loads(str(row["content"])),
            )
            for row in card.observations
        ),
        dict(record),
    )


async def build_contextual_bundle(
    *,
    prepared: Sequence[PreparedEpisode],
    dataset_label: str,
    population_identity: Mapping[str, Any],
    dataset_contract: GraphDatasetContract,
    state_db: Path,
    output_dir: Path,
    generation_base_url: str,
    embedding_base_url: str,
    model: str,
    generation_key: str,
    embedding_key: str,
    finalize_rows: Callable[[Sequence[Mapping[str, Any]]], Sequence[Mapping[str, Any]]] | None = None,
    fixed_denominator: int | None = None,
) -> Mapping[str, Any]:
    prepared = tuple(prepared)
    if not prepared or len({row.task_id for row in prepared}) != len(prepared):
        raise ValueError("EIR retrieval population differs")
    if len({row.train_index for row in prepared}) != len(prepared):
        raise ValueError("EIR retrieval indices differ")
    output_dir = output_dir.expanduser().absolute()
    if output_dir.exists():
        raise FileExistsError("EIR bundle output must be fresh")
    checkpoint_root = output_dir.parent / f".{output_dir.name}.checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    with EIRStateStore(
        state_db,
        dataset_contract=dataset_contract,
        readonly=True,
    ) as store, EIRStateStore(
        checkpoint_root / "retrieval_cache.sqlite3",
        dataset_contract=dataset_contract,
    ) as cache_store:
        snapshot_id = store.head_snapshot_id
        if snapshot_id is None:
            raise ValueError("EIR development retrieval requires a committed graph")
        graph, canonical_versions = compile_eir_experience_graph(
            store, snapshot_id=snapshot_id
        )
        embedder = StrictEmbeddingAdapter(
            QwenEmbeddingHTTPTransport(
                base_url=embedding_base_url, api_key=embedding_key
            ),
            cache=cache_store.embedding_cache(),
        )
        client = OpenAIClient(
            model=model,
            api_key=generation_key,
            base_url=generation_base_url,
            generation_config=_source_generation_config(),
            retry_times=PRODUCER_TRANSPORT_RETRY_WAITS,
            runtime_timeout_retries=PRODUCER_RUNTIME_TIMEOUT_RETRIES,
            timeout=REPAIR_SOURCE_TIMEOUT_SECONDS,
            trust_env=False,
        )
        producer = ContextualBindingProducer(
            OpenAIJsonObjectLLM(
                client,
                request_kind=BINDING_KIND,
                source_protocol_format=BINDING_PROTOCOL_FORMAT,
                prompt_sha256=BINDING_PROMPT_SHA256,
                response_schema_name="degs_contextual_binding_v1",
            )
        )

        base_identity = {
            "format": "degs_eir_binding_checkpoint_identity_v1",
            "dataset": dataset_label,
            "population_identity_sha256": hashlib.sha256(
                canonical_json_bytes(dict(population_identity))
            ).hexdigest(),
            "model": model,
            "generation_base_url": generation_base_url.rstrip("/"),
            "embedding_base_url": embedding_base_url.rstrip("/"),
            "snapshot_id": snapshot_id,
            "graph_identity": graph.identity(),
            "retrieval_method": CONTEXTUAL_RETRIEVAL_METHOD,
            "top_k": CONTEXTUAL_TOP_K,
            "neighbors_per_anchor": CONTEXT_NEIGHBORS_PER_ANCHOR,
            "binding_prompt_sha256": BINDING_PROMPT_SHA256,
            "binding_response_schema_sha256": hashlib.sha256(
                canonical_json_bytes(contextual_binding_response_schema())
            ).hexdigest(),
            "binding_producer": dict(producer.llm.protocol_identity),
        }
        checkpoint_paths: dict[str, Path] = {}
        checkpoint_identities: dict[str, Mapping[str, Any]] = {}
        cached_expectations: dict[str, tuple[ExperienceExpectation, ...]] = {}
        for row in prepared:
            identity = {
                **base_identity,
                "task_id": row.task_id,
                "dataset_index": row.train_index,
                "retrieval_document_sha256": hashlib.sha256(
                    row.retrieval_document.encode("utf-8")
                ).hexdigest(),
            }
            path = checkpoint_root / (
                hashlib.sha256(
                    f"{row.train_index}:{row.task_id}".encode("utf-8")
                ).hexdigest()
                + ".json"
            )
            checkpoint_paths[row.task_id] = path
            checkpoint_identities[row.task_id] = identity
            if not path.is_file():
                continue
            stored = json.loads(path.read_text(encoding="utf-8"))
            if (
                type(stored) is not dict
                or set(stored) != {"identity", "retrieval", "expectations"}
                or stored["identity"] != identity
                or type(stored["expectations"]) is not list
            ):
                raise ValueError("EIR binding checkpoint identity differs")
            cached_expectations[row.task_id] = tuple(
                experience_expectation_from_dict(item)
                for item in stored["expectations"]
            )

        async def checkpoint_binding(
            row: Any,
            retrieval: ContextualRetrieval,
            decision: tuple[ExperienceExpectation, ...],
        ) -> None:
            path = checkpoint_paths[row.task_id]
            value = {
                "identity": checkpoint_identities[row.task_id],
                "retrieval": retrieval.to_dict(),
                "expectations": [item.to_dict() for item in decision],
            }
            if path.is_file():
                stored = json.loads(path.read_text(encoding="utf-8"))
                if stored != value:
                    raise ValueError("EIR binding checkpoint content differs")
                return
            _write_json(path, value)

        retrievals, decisions, failures = await retrieve_and_bind(
            items=prepared,
            graph=graph,
            snapshot_id=snapshot_id,
            canonical_versions=canonical_versions,
            embedding=embedder,
            binding=producer,
            workers=PRODUCER_WORKERS,
            request_prefix="retrieval",
            cached_expectations=cached_expectations,
            decision_callback=checkpoint_binding,
        )

    rows = []
    for index, row in enumerate(prepared):
        retrieval = retrievals[index]
        decision = decisions[index]
        guidance = render_bound_guidance(decision)
        rows.append(
            {
                "format": EIR_GUIDANCE_ROW_FORMAT,
                "instance_id": row.task_id,
                "dataset_index": row.train_index,
                "experience": guidance,
                "snapshot_id": snapshot_id,
                "retrieval": retrieval.to_dict(),
                "expectations": [item.to_dict() for item in decision],
                "status": "BINDING_FAILURE" if row.task_id in failures else "COMPLETE",
                "error": failures.get(row.task_id),
            }
        )
    if finalize_rows is not None:
        rows = [dict(row) for row in finalize_rows(tuple(rows))]
    if not rows or len({str(row.get("instance_id")) for row in rows}) != len(rows):
        raise ValueError("final EIR guidance population differs")
    denominator = len(rows) if fixed_denominator is None else fixed_denominator
    if type(denominator) is not int or denominator != len(rows):
        raise ValueError("EIR guidance denominator differs")
    output_dir.mkdir(parents=True)
    experience_path = output_dir / "experience.jsonl"
    experience_payload = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    experience_path.write_bytes(experience_payload)
    body = {
        "format": EIR_BUNDLE_FORMAT,
        "dataset": dataset_label,
        "fixed_denominator": denominator,
        "population_identity": dict(population_identity),
        "model": model,
        "generation_base_url": generation_base_url,
        "embedding_base_url": embedding_base_url,
        "snapshot_id": snapshot_id,
        "state_db": str(state_db.expanduser().absolute()),
        "experience_file": "experience.jsonl",
        "experience_sha256": hashlib.sha256(experience_payload).hexdigest(),
        "row_count": len(rows),
        "binding_failure_count": len(failures),
        "graph_identity": graph.identity(),
    }
    manifest = {
        **body,
        "self_sha256": hashlib.sha256(canonical_json_bytes(body)).hexdigest(),
    }
    _write_json(output_dir / "bundle_manifest.json", manifest)
    return manifest


async def build_development_bundle(
    *,
    dataset_path: Path,
    state_db: Path,
    output_dir: Path,
    generation_base_url: str,
    embedding_base_url: str,
    model: str,
    generation_key: str,
    embedding_key: str,
) -> Mapping[str, Any]:
    dataset_path = dataset_path.expanduser().resolve()
    records = _load_development_harness_records(dataset_path / "dataset.json")
    prepared = await _bounded_map(
        tuple(records),
        workers=PRODUCER_WORKERS,
        worker=lambda row: _prepare(row, dataset_path=dataset_path),
    )
    return await build_contextual_bundle(
        prepared=prepared,
        dataset_label="SpreadsheetBench development[200,400)",
        population_identity={
            "dataset_json_sha256": hashlib.sha256(
                (dataset_path / "dataset.json").read_bytes()
            ).hexdigest(),
            "split": "development[200,400)",
        },
        dataset_contract=SPREADSHEETBENCH_GRAPH_CONTRACT,
        state_db=state_db,
        output_dir=output_dir,
        generation_base_url=generation_base_url,
        embedding_base_url=embedding_base_url,
        model=model,
        generation_key=generation_key,
        embedding_key=embedding_key,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an EIR contextual development bundle.")
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--state-db", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--generation-base-url", required=True)
    parser.add_argument("--embedding-base-url", required=True)
    parser.add_argument("--model", choices=("Qwen3.5-9B-AWQ", "Qwen3.5-27B-AWQ"), required=True)
    parser.add_argument("--generation-api-key-env", default="DEGS_API_KEY")
    parser.add_argument("--embedding-api-key-env", default="DEGS_EMBEDDING_API_KEY")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    generation_key = os.environ.get(args.generation_api_key_env)
    embedding_key = os.environ.get(args.embedding_api_key_env)
    if not generation_key or not embedding_key:
        raise ValueError("generation and embedding API keys are required")
    os.environ["DEGS_MODEL"] = args.model
    manifest = asyncio.run(
        build_development_bundle(
            dataset_path=args.dataset_path,
            state_db=args.state_db,
            output_dir=args.output_dir,
            generation_base_url=args.generation_base_url,
            embedding_base_url=args.embedding_base_url,
            model=args.model,
            generation_key=generation_key,
            embedding_key=embedding_key,
        )
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
