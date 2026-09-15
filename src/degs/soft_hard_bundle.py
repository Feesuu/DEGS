"""Build SpreadsheetBench Soft/Hard guidance through the formal EIR retriever."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .dynamic_train import PRODUCER_WORKERS, PreparedEpisode, _bounded_map
from .eir_bundle import (
    EIR_BUNDLE_FORMAT,
    VerifiedEIRGuidanceBundle,
    _prepare,
    build_contextual_bundle,
    verify_contextual_bundle,
)
from .graph_dataset_contract import SPREADSHEETBENCH_GRAPH_CONTRACT
from .provider import EIRGuidanceProvider
from .soft_hard_dataset import (
    TESTCASE_COUNT,
    load_prepared_retrieval_population,
    retrieval_query_rows,
)


FORMAT = EIR_BUNDLE_FORMAT
DATASET_LABEL = "SpreadsheetBench Soft/Hard fixed 2,529-case population"


def _population_identity(population: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "manifest_sha256": population["self_sha256"],
        "retrieval_projection_sha256": population["retrieval_projection_sha256"],
        "input_tree_sha256": population["prepared_input_tree_sha256"],
        "task_count": population["task_count"],
        "case_count": population["testcase_count"],
    }


def _snapshot_id(path: Path) -> str:
    value = json.loads(path.read_text(encoding="utf-8"))
    snapshot_id = value.get("snapshot_id") if isinstance(value, Mapping) else None
    if type(snapshot_id) is not str or not snapshot_id:
        raise ValueError("Soft/Hard frozen snapshot manifest differs")
    return snapshot_id


async def _prepared_rows(
    *, population: Mapping[str, Any], prepared_data_path: Path
) -> tuple[PreparedEpisode, ...]:
    records = tuple(
        {
            "dataset_index": int(row["query_index"]),
            "task_id": str(row["case_id"]),
            "instruction": str(row["instruction"]),
            "spreadsheet_path": str(row["spreadsheet_path"]),
            "answer_position": str(row["answer_position"]),
        }
        for row in retrieval_query_rows(population)
    )
    if len(records) != TESTCASE_COUNT:
        raise ValueError("Soft/Hard retrieval population differs")
    return await _bounded_map(
        records,
        workers=PRODUCER_WORKERS,
        worker=lambda row: _prepare(row, dataset_path=prepared_data_path),
    )


async def build_from_paths(
    *,
    source_dataset_path: Path,
    prepared_data_path: Path,
    retrieval_manifest_path: Path,
    snapshot_manifest_path: Path,
    state_db_path: Path,
    output_dir: Path,
    generation_base_url: str,
    embedding_base_url: str,
    model: str,
    generation_key: str,
    embedding_key: str,
) -> Mapping[str, Any]:
    del source_dataset_path
    population = load_prepared_retrieval_population(
        data_path=prepared_data_path, manifest_path=retrieval_manifest_path
    )
    expected_snapshot = _snapshot_id(snapshot_manifest_path)
    prepared = await _prepared_rows(
        population=population, prepared_data_path=prepared_data_path
    )
    manifest = await build_contextual_bundle(
        prepared=prepared,
        dataset_label=DATASET_LABEL,
        population_identity=_population_identity(population),
        dataset_contract=SPREADSHEETBENCH_GRAPH_CONTRACT,
        state_db=state_db_path,
        output_dir=output_dir,
        generation_base_url=generation_base_url,
        embedding_base_url=embedding_base_url,
        model=model,
        generation_key=generation_key,
        embedding_key=embedding_key,
    )
    if manifest["snapshot_id"] != expected_snapshot:
        raise ValueError("Soft/Hard retrieval did not read the declared frozen snapshot")
    return manifest


def verify_from_paths(
    *,
    source_dataset_path: Path,
    prepared_data_path: Path,
    retrieval_manifest_path: Path,
    snapshot_manifest_path: Path,
    state_db_path: Path,
    output_dir: Path,
) -> VerifiedEIRGuidanceBundle:
    del source_dataset_path
    population = load_prepared_retrieval_population(
        data_path=prepared_data_path, manifest_path=retrieval_manifest_path
    )
    instance_ids = tuple(str(row["case_id"]) for row in retrieval_query_rows(population))
    verified = verify_contextual_bundle(
        output_dir=output_dir,
        expected_instance_ids=instance_ids,
        expected_state_db=state_db_path,
        expected_dataset=DATASET_LABEL,
    )
    if (
        verified.manifest.get("snapshot_id") != _snapshot_id(snapshot_manifest_path)
        or verified.manifest.get("population_identity") != _population_identity(population)
    ):
        raise ValueError("Soft/Hard EIR bundle provenance differs")
    return verified


class SoftHardExperienceProvider(EIRGuidanceProvider):
    """Exact case lookup over a verified EIR Soft/Hard bundle."""

    def __init__(self, verified_bundle: VerifiedEIRGuidanceBundle) -> None:
        if type(verified_bundle) is not VerifiedEIRGuidanceBundle:
            raise TypeError("Soft/Hard provider requires a verified EIR bundle")
        loaded = EIRGuidanceProvider.from_bundle(verified_bundle.root)
        self.__dict__.update(loaded.__dict__)
        self.path = verified_bundle.root / "experience.jsonl"
        self.manifest = verified_bundle.manifest
        if self.identity()["row_count"] != TESTCASE_COUNT:
            raise ValueError("Soft/Hard experience population differs")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("build", "verify"):
        child = commands.add_parser(command)
        child.add_argument("--source-dataset-path", type=Path, required=True)
        child.add_argument("--prepared-data-path", type=Path, required=True)
        child.add_argument("--retrieval-manifest-path", type=Path, required=True)
        child.add_argument("--snapshot-manifest-path", type=Path, required=True)
        child.add_argument("--state-db", type=Path, required=True)
        child.add_argument("--output-dir", type=Path, required=True)
        if command == "build":
            child.add_argument("--llm-base-url", required=True)
            child.add_argument("--embedding-base-url", required=True)
            child.add_argument("--model", choices=("Qwen3.5-9B-AWQ", "Qwen3.5-27B-AWQ"), required=True)
            child.add_argument("--llm-api-key-env", default="DEGS_API_KEY")
            child.add_argument("--embedding-api-key-env", default="DEGS_EMBEDDING_API_KEY")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    common = {
        "source_dataset_path": args.source_dataset_path,
        "prepared_data_path": args.prepared_data_path,
        "retrieval_manifest_path": args.retrieval_manifest_path,
        "snapshot_manifest_path": args.snapshot_manifest_path,
        "state_db_path": args.state_db,
        "output_dir": args.output_dir,
    }
    if args.command == "build":
        generation_key = os.environ.get(args.llm_api_key_env)
        embedding_key = os.environ.get(args.embedding_api_key_env)
        if not generation_key or not embedding_key:
            raise ValueError("generation and embedding API keys are required")
        manifest = asyncio.run(
            build_from_paths(
                **common,
                generation_base_url=args.llm_base_url,
                embedding_base_url=args.embedding_base_url,
                model=args.model,
                generation_key=generation_key,
                embedding_key=embedding_key,
            )
        )
    else:
        manifest = verify_from_paths(**common).manifest
    print(json.dumps({
        "format": FORMAT,
        "row_count": manifest["row_count"],
        "self_sha256": manifest["self_sha256"],
        "binding_failure_count": manifest["binding_failure_count"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
