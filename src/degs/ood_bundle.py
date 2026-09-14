"""Build WikiTQ/HiTab bundles with the single DEGS online retrieval runtime."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from sb_adapter.transport import validate_service_url

from . import bundle as retrieval
from .ood_dataset import verify_population
from .population_bundle import build_bundle, verify_bundle
from .provider import ExperiencePayload
from .retrieval_clarification import openai_retrieval_clarification_llm
from .transport import QwenEmbeddingHTTPTransport
from .workflow_retrieval import openai_need_graph_llm, openai_selector_llm


FORMAT = "degs_tableqa_ood_retrieval_bundle_v1"
CLAIM_SCOPE = (
    "SpreadsheetBench train[0,200) ExperienceGraph transferred to input-only "
    "WikiTQ/HiTab queries; target answers, outcomes, verifier results, and Agent "
    "traces are unavailable"
)


def _population_identity(population: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "dataset": population["dataset"],
        "manifest_sha256": population["self_sha256"],
        "query_projection_sha256": population["query_projection_sha256"],
        "input_tree_sha256": population["input_tree_sha256"],
        "task_count": population["task_count"],
    }


def _tasks(population: Mapping[str, Any]) -> tuple[Any, ...]:
    tasks = tuple(
        retrieval._Task(
            int(row["query_index"]),
            int(row["dataset_index"]),
            str(row["task_id"]),
            str(row["semantic_query"]),
            str(row["spreadsheet_path"]),
            str(row["answer_position"]),
        )
        for row in population["tasks"]
    )
    if len(tasks) != population["task_count"] or any(
        task.query_index != index or task.dataset_index != index
        for index, task in enumerate(tasks)
    ):
        raise ValueError("OOD retrieval population differs")
    return tasks


def build_from_paths(
    *,
    source_dataset_path: Path,
    prepared_data_path: Path,
    snapshot_manifest_path: Path,
    state_db_path: Path,
    output_dir: Path,
    retrieval_cache_path: Path | None = None,
    embedding_transport: Any,
    need_llm: Any,
    clarification_llm: Any,
    selector_llm: Any,
) -> Any:
    population = verify_population(prepared_data_path)
    return build_bundle(
        format_id=FORMAT,
        claim_scope=CLAIM_SCOPE,
        population_identity=_population_identity(population),
        tasks=_tasks(population),
        target_dataset_path=prepared_data_path / "dataset.json",
        source_dataset_path=source_dataset_path,
        snapshot_manifest_path=snapshot_manifest_path,
        state_db_path=state_db_path,
        output_dir=output_dir,
        retrieval_cache_path=retrieval_cache_path,
        embedding_transport=embedding_transport,
        need_llm=need_llm,
        clarification_llm=clarification_llm,
        selector_llm=selector_llm,
    )


def verify_from_paths(
    *,
    source_dataset_path: Path,
    prepared_data_path: Path,
    snapshot_manifest_path: Path,
    state_db_path: Path,
    output_dir: Path,
) -> Any:
    population = verify_population(prepared_data_path)
    return verify_bundle(
        format_id=FORMAT,
        claim_scope=CLAIM_SCOPE,
        population_identity=_population_identity(population),
        tasks=_tasks(population),
        target_dataset_path=prepared_data_path / "dataset.json",
        source_dataset_path=source_dataset_path,
        snapshot_manifest_path=snapshot_manifest_path,
        state_db_path=state_db_path,
        output_dir=output_dir,
    )


class OODExperienceProvider:
    def __init__(self, verified_bundle: Any) -> None:
        self.path = verified_bundle.root / "experience.jsonl"
        self.manifest = verified_bundle.manifest
        payloads: dict[str, ExperiencePayload] = {}
        for line in self.path.read_bytes().splitlines():
            row = json.loads(line)
            instance_id = str(row["instance_id"])
            if instance_id in payloads:
                raise ValueError("OOD experience identity is duplicated")
            payloads[instance_id] = ExperiencePayload(
                str(row["experience"]), dict(row["metadata"])
            )
        if len(payloads) != self.manifest["row_count"]:
            raise ValueError("OOD experience population differs")
        self._payloads = payloads

    def for_instance(self, instance_id: str) -> ExperiencePayload:
        try:
            return self._payloads[str(instance_id)]
        except KeyError as exc:
            raise KeyError(f"experience is absent for OOD task {instance_id}") from exc

    def identity(self) -> Mapping[str, Any]:
        return {
            "provider": "degs_ood_exact_task_lookup",
            "path": str(self.path),
            "sha256": self.manifest["experience_sha256"],
            "bundle_self_sha256": self.manifest["self_sha256"],
            "row_count": len(self._payloads),
        }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("build", "verify"):
        child = commands.add_parser(command)
        child.add_argument("--source-dataset-path", type=Path, required=True)
        child.add_argument("--prepared-data-path", type=Path, required=True)
        child.add_argument("--snapshot-manifest-path", type=Path, required=True)
        child.add_argument("--state-db", type=Path, required=True)
        child.add_argument("--output-dir", type=Path, required=True)
        if command == "build":
            child.add_argument("--retrieval-cache", type=Path)
            child.add_argument("--llm-base-url", required=True)
            child.add_argument("--embedding-base-url", required=True)
            child.add_argument("--llm-api-key-env", default="DEGS_API_KEY")
            child.add_argument(
                "--embedding-api-key-env", default="DEGS_EMBEDDING_API_KEY"
            )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    common = {
        "source_dataset_path": args.source_dataset_path,
        "prepared_data_path": args.prepared_data_path,
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
        client = retrieval._client(api_key=llm_key, base_url=args.llm_base_url)
        verified = build_from_paths(
            **common,
            retrieval_cache_path=args.retrieval_cache,
            embedding_transport=QwenEmbeddingHTTPTransport(
                base_url=args.embedding_base_url, api_key=embedding_key
            ),
            need_llm=openai_need_graph_llm(
                client,
                expected_retry_times=retrieval.BUNDLE_TRANSPORT_RETRY_WAITS,
                expected_runtime_timeout_retries=retrieval.BUNDLE_RUNTIME_TIMEOUT_RETRIES,
            ),
            clarification_llm=openai_retrieval_clarification_llm(
                client,
                expected_retry_times=retrieval.BUNDLE_TRANSPORT_RETRY_WAITS,
                expected_runtime_timeout_retries=retrieval.BUNDLE_RUNTIME_TIMEOUT_RETRIES,
            ),
            selector_llm=openai_selector_llm(
                client,
                expected_retry_times=retrieval.BUNDLE_TRANSPORT_RETRY_WAITS,
                expected_runtime_timeout_retries=retrieval.BUNDLE_RUNTIME_TIMEOUT_RETRIES,
            ),
        )
    else:
        verified = verify_from_paths(**common)
    print(
        json.dumps(
            {
                "method": retrieval.METHOD_NAME,
                "dataset": verified.manifest["population"]["dataset"],
                "row_count": verified.manifest["row_count"],
                "self_sha256": verified.manifest["self_sha256"],
                "status_counts": verified.manifest["status_counts"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
