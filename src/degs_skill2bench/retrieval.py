"""Step-scoped Skill2Bench retrieval through the shared EIR Top-5 binder."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

from degs.core import canonical_json_bytes
from degs.dynamic_train import PreparedEpisode
from degs.eir_bundle import (
    EIR_GUIDANCE_ROW_FORMAT,
    build_contextual_bundle,
    verify_contextual_bundle,
)
from degs.episode_evidence import EvidenceItem

from .contract import Skill2BenchProtocol
from .dataset import public_task_view
from .step_units import public_step_view


BUNDLE_FORMAT = "degs_eir_contextual_guidance_bundle_v1"


def _dataset_label(protocol: Skill2BenchProtocol) -> str:
    return f"Skill2Bench {protocol.profile} step-scoped test"


def _population_identity(
    test_tasks: Sequence[Mapping[str, Any]], protocol: Skill2BenchProtocol
) -> dict[str, Any]:
    public = [public_task_view(row) for row in test_tasks]
    return {
        "profile": protocol.profile,
        "model": protocol.model,
        "test_sha256": protocol.test_sha256,
        "task_count": protocol.test_count,
        "public_projection_sha256": hashlib.sha256(
            canonical_json_bytes(public)
        ).hexdigest(),
        "retrieval_scope": "ONE_INDEPENDENT_RETRIEVAL_PER_STEP",
    }


def _step_items(
    test_tasks: Sequence[Mapping[str, Any]],
) -> tuple[tuple[PreparedEpisode, ...], tuple[tuple[int, int], ...]]:
    items: list[PreparedEpisode] = []
    coordinates: list[tuple[int, int]] = []
    for task_index, raw_task in enumerate(test_tasks):
        public = public_task_view(raw_task)
        for step_number, question in enumerate(public["questions"], 1):
            if not question.strip():
                continue
            step = public_step_view(raw_task, step_number=step_number)
            index = len(items)
            items.append(
                PreparedEpisode(
                    index,
                    f"{public['instance_id']}::step-{step_number:02d}",
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
                    {"task_index": task_index, "step_number": step_number},
                )
            )
            coordinates.append((task_index, step_number))
    return tuple(items), tuple(coordinates)


def _finalizer(
    *,
    test_tasks: Sequence[Mapping[str, Any]],
    coordinates: Sequence[tuple[int, int]],
):
    def finalize(step_rows: Sequence[Mapping[str, Any]]) -> Sequence[Mapping[str, Any]]:
        by_coordinate = {
            coordinate: dict(row)
            for coordinate, row in zip(coordinates, step_rows, strict=True)
        }
        rows: list[dict[str, Any]] = []
        for task_index, raw_task in enumerate(test_tasks):
            public = public_task_view(raw_task)
            steps = []
            experience_parts = []
            retrieval_steps = []
            all_expectations = []
            errors = []
            for step_number, question in enumerate(public["questions"], 1):
                source = by_coordinate.get((task_index, step_number))
                if source is None:
                    step = {
                        "step_number": step_number,
                        "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
                        "status": "EMPTY_PUBLIC_QUESTION",
                        "experience": "",
                        "retrieval": {},
                        "expectations": [],
                        "error": None,
                    }
                else:
                    step = {
                        "step_number": step_number,
                        "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
                        "status": source["status"],
                        "experience": source["experience"],
                        "retrieval": source["retrieval"],
                        "expectations": source["expectations"],
                        "error": source["error"],
                    }
                    if source["experience"]:
                        experience_parts.append(
                            f"Step {step_number}:\n{source['experience']}"
                        )
                    retrieval_steps.append(
                        {"step_number": step_number, **dict(source["retrieval"])}
                    )
                    all_expectations.append(
                        {"step_number": step_number, "items": source["expectations"]}
                    )
                    if source["error"]:
                        errors.append(f"Step {step_number}: {source['error']}")
                steps.append(step)
            rows.append(
                {
                    "format": EIR_GUIDANCE_ROW_FORMAT,
                    "instance_id": public["instance_id"],
                    "dataset_index": task_index,
                    "experience": "\n\n".join(experience_parts),
                    "snapshot_id": next(
                        (
                            source["snapshot_id"]
                            for coordinate, source in by_coordinate.items()
                            if coordinate[0] == task_index
                        ),
                        "",
                    ),
                    "retrieval": {
                        "format": "degs_skill2bench_eir_step_retrieval_v1",
                        "steps": retrieval_steps,
                    },
                    "expectations": all_expectations,
                    "steps": steps,
                    "status": "BINDING_FAILURE" if errors else "COMPLETE",
                    "error": "\n".join(errors) or None,
                }
            )
        return rows

    return finalize


async def build_step_retrieval_bundle(
    *,
    test_tasks: Sequence[Mapping[str, Any]],
    state_db_path: Path,
    output_dir: Path,
    generation_base_url: str,
    embedding_base_url: str,
    generation_key: str,
    embedding_key: str,
    protocol: Skill2BenchProtocol,
) -> dict[str, Any]:
    if len(test_tasks) != protocol.test_count:
        raise ValueError("Skill2Bench test population differs")
    items, coordinates = _step_items(test_tasks)
    return dict(
        await build_contextual_bundle(
            prepared=items,
            dataset_label=_dataset_label(protocol),
            population_identity=_population_identity(test_tasks, protocol),
            dataset_contract=protocol.graph_contract,
            state_db=state_db_path,
            output_dir=output_dir,
            generation_base_url=generation_base_url,
            embedding_base_url=embedding_base_url,
            model=protocol.model,
            generation_key=generation_key,
            embedding_key=embedding_key,
            finalize_rows=_finalizer(
                test_tasks=test_tasks, coordinates=coordinates
            ),
            fixed_denominator=protocol.test_count,
        )
    )


def verify_step_retrieval_bundle(
    *,
    test_tasks: Sequence[Mapping[str, Any]],
    state_db_path: Path,
    output_dir: Path,
    protocol: Skill2BenchProtocol,
) -> tuple[dict[str, Any], ...]:
    public = [public_task_view(row) for row in test_tasks]
    verified = verify_contextual_bundle(
        output_dir=output_dir,
        expected_instance_ids=tuple(row["instance_id"] for row in public),
        expected_state_db=state_db_path,
        expected_dataset=_dataset_label(protocol),
    )
    if verified.manifest.get("population_identity") != _population_identity(
        test_tasks, protocol
    ):
        raise ValueError("Skill2Bench EIR retrieval provenance differs")
    rows = tuple(
        __import__("json").loads(line)
        for line in (verified.root / "experience.jsonl").read_bytes().splitlines()
    )
    for index, (row, task) in enumerate(zip(rows, public, strict=True)):
        steps = row.get("steps")
        if type(steps) is not list or len(steps) != len(task["questions"]):
            raise ValueError(f"Skill2Bench retrieval row {index} Steps differ")
    return rows


__all__ = ["build_step_retrieval_bundle", "verify_step_retrieval_bundle"]
