"""Prepare the fixed SpreadsheetBench Soft/Hard evaluation population."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from .core import canonical_json_bytes


FORMAT = "degs_spreadsheetbench_soft_hard_population_v2"
RETRIEVAL_FORMAT = "degs_spreadsheetbench_soft_hard_retrieval_population_v1"
FULL_DATASET_SHA256 = (
    "e5137ecbec4273d91344a0c8feb2aff2d4a93d5881ac40e490250dfd8db227de"
)
VERIFIED_DATASET_SHA256 = (
    "bcecaa89a005bd4e3bbe98da150a86e8062c27f262e575d5e47bd9861b3525e7"
)
TASK_COUNT = 912
TESTCASE_COUNT = 2529
EXCLUDED_TRAIN_CASE_COUNT = 200
EXPECTED_QUERY_PROJECTION_SHA256 = (
    "b2207734d670c623cbcd5862aaf42c67f058ef9541d71a7cfa42949826088938"
)
EXPECTED_EVALUATION_STRATA = {
    "train_task_overlap": {"task_count": 200, "testcase_count": 400},
    "development_task_overlap": {"task_count": 200, "testcase_count": 597},
    "untouched_task": {"task_count": 512, "testcase_count": 1532},
}
EXPECTED_BYTE_IDENTICAL_EXCLUSIONS = {
    "input": 186,
    "answer": 165,
}
EXPECTED_EXCLUSION_PROJECTION_SHA256 = (
    "15bf53552444a48ebc909a15000472ff2f350ab080f9841624c770ae0d9894f0"
)
EXPECTED_CASE_PROJECTION_SHA256 = (
    "9c7ffc7db3b799049ec12c827040d384c7449d38399ae53f3cd07a80a50b528f"
)
EXPECTED_PREPARED_TREE_SHA256 = (
    "72e22331d7f5933cdc3db30443819276306f32f91b083389622e10e6a4999a4a"
)
EXPECTED_STRATUM_PROJECTION_SHA256 = (
    "2d4196e62d4c0c179eeb0a92d99c904074abd8e686da84df85f8fc8b61592fa1"
)
EXPECTED_TASK_METADATA_PROJECTION_SHA256 = (
    "9e4c9ec4b3e7d2a427ee55d55d43a17f07d60bf46023927647fdd7cf143a11ce"
)
EXPECTED_RETRIEVAL_PROJECTION_SHA256 = (
    "ff8de2602294bb70ade6e84e98b9dc3f89ddffa053a8a08a0066d42402609fc7"
)
EXPECTED_INPUT_TREE_PROJECTION_SHA256 = (
    "a75bb6973297fa7d8fc764474232c0bb7892a07a08158579e92c091e9758c105"
)
EXCLUSION_FIELDS = {
    "train_index",
    "task_id",
    "input_file",
    "answer_file",
    "full_input_sha256",
    "verified_input_sha256",
    "input_bytes_equal",
    "full_answer_sha256",
    "verified_answer_sha256",
    "answer_bytes_equal",
    "verified_instruction_sha256",
    "full_instruction_sha256",
    "instruction_text_equal",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        row: dict[str, Any] = {}
        for key, value in pairs:
            if key in row:
                raise ValueError(f"duplicate JSON key in {path}: {key}")
            row[key] = value
        return row

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=no_duplicates)


def _safe_relative(value: Any, *, label: str) -> Path:
    if type(value) is not str or not value:
        raise ValueError(f"{label} must be a non-empty string")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or value != path.as_posix():
        raise ValueError(f"unsafe {label}: {value}")
    return path


def _spreadsheet_dir(root: Path, row: Mapping[str, Any]) -> Path:
    relative = _safe_relative(
        row.get("spreadsheet_path") or f"spreadsheet/{row['id']}",
        label=f"spreadsheet_path for {row.get('id')}",
    )
    path = root / relative
    if path.is_symlink() or not path.is_dir():
        raise FileNotFoundError(f"spreadsheet directory differs: {path}")
    return path


def _normalized_name(name: str) -> str:
    if name.endswith("_input .xlsx"):
        return name[: -len("_input .xlsx")] + "_input.xlsx"
    return name


def _tree_sha256(root: Path) -> str:
    rows: list[dict[str, str]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink():
            raise ValueError(f"prepared dataset contains a symlink: {path}")
        if path.is_file():
            rows.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": _sha256(path),
                }
            )
    return hashlib.sha256(canonical_json_bytes(rows)).hexdigest()


def _link_or_copy_regular(source: Path, target: Path) -> str:
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"source workbook must be a regular file: {source}")
    target.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"prepared dataset collision: {target}")
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def _recover_interrupted_publication(
    marker: Path,
    *,
    output: Path,
    manifest: Path,
    retrieval_manifest: Path,
    manifest_staging: Path,
    retrieval_manifest_staging: Path,
    marker_staging: Path,
) -> None:
    finals_absent = all(
        not path.exists() and not path.is_symlink()
        for path in (output, manifest, retrieval_manifest)
    )
    staging_paths = (
        manifest_staging,
        retrieval_manifest_staging,
        marker_staging,
    )
    if not marker.exists():
        if finals_absent:
            for path in staging_paths:
                if path.is_symlink():
                    raise ValueError("orphan Soft/Hard publication staging is a symlink")
                if path.is_file():
                    path.unlink()
                elif path.exists():
                    raise ValueError("orphan Soft/Hard publication staging differs")
        return
    if marker.is_symlink() or not marker.is_file():
        raise ValueError("Soft/Hard publication marker differs")
    try:
        actual_marker = _read_json(marker)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        if not finals_absent:
            raise ValueError("Soft/Hard publication marker is incomplete") from error
        marker.unlink()
        for path in staging_paths:
            if path.is_symlink():
                raise ValueError("orphan Soft/Hard publication staging is a symlink")
            if path.is_file():
                path.unlink()
            elif path.exists():
                raise ValueError("orphan Soft/Hard publication staging differs")
        return
    expected = {
        "format": "degs_soft_hard_population_publication_v1",
        "output": str(output),
        "manifest": str(manifest),
        "retrieval_manifest": str(retrieval_manifest),
        "manifest_staging": str(manifest_staging),
        "retrieval_manifest_staging": str(retrieval_manifest_staging),
        "marker_staging": str(marker_staging),
    }
    if actual_marker != expected:
        raise ValueError("Soft/Hard publication marker identity differs")
    for path in (
        manifest,
        retrieval_manifest,
        manifest_staging,
        retrieval_manifest_staging,
        marker_staging,
    ):
        if path.is_symlink():
            raise ValueError("interrupted Soft/Hard publication contains a symlink")
        if path.is_file():
            path.unlink()
        elif path.exists():
            raise ValueError("interrupted Soft/Hard publication path differs")
    if output.is_symlink():
        raise ValueError("interrupted Soft/Hard dataset is a symlink")
    if output.is_dir():
        shutil.rmtree(output)
    elif output.exists():
        raise ValueError("interrupted Soft/Hard dataset path differs")
    marker.unlink()


def _dataset_rows(
    root: Path, *, expected_sha256: str | None, label: str
) -> tuple[list[dict[str, Any]], str]:
    dataset_file = root / "dataset.json"
    if dataset_file.is_symlink() or not dataset_file.is_file():
        raise FileNotFoundError(f"{label} dataset.json differs: {dataset_file}")
    actual_sha256 = _sha256(dataset_file)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError(f"{label} dataset identity differs")
    value = _read_json(dataset_file)
    if type(value) is not list or any(type(row) is not dict for row in value):
        raise ValueError(f"{label} dataset rows differ")
    return value, actual_sha256


def _query_projection(tasks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "query_index": row["query_index"],
            "dataset_index": row["dataset_index"],
            "task_id": row["task_id"],
            "instruction": row["instruction"],
        }
        for row in tasks
    ]


def _case_projection(tasks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "task_id": task["task_id"],
            "spreadsheet_path": task["spreadsheet_path"],
            "cases": [
                {
                    key: case[key]
                    for key in (
                        "case_id",
                        "input_file",
                        "answer_file",
                        "output_file",
                        "retrieval_spreadsheet_path",
                    )
                }
                for case in task["cases"]
            ],
        }
        for task in tasks
    ]


def _retrieval_projection(
    tasks: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in tasks:
        for case in task["cases"]:
            rows.append(
                {
                    "query_index": len(rows),
                    "dataset_index": len(rows),
                    "case_id": case["case_id"],
                    "task_id": task["task_id"],
                    "instruction": task["instruction"],
                    "instruction_type": task["instruction_type"],
                    "execution_spreadsheet_path": task["spreadsheet_path"],
                    "spreadsheet_path": case["retrieval_spreadsheet_path"],
                    "answer_position": task["answer_position"],
                    "input_file": case["input_file"],
                    "output_file": case["output_file"],
                    "input_sha256": case["input_sha256"],
                    "input_size": case["input_size"],
                }
            )
    return rows


def _input_only_tasks(
    tasks: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "query_index": task["query_index"],
            "dataset_index": task["dataset_index"],
            "task_id": task["task_id"],
            "instruction": task["instruction"],
            "instruction_type": task["instruction_type"],
            "answer_position": task["answer_position"],
            "spreadsheet_path": task["spreadsheet_path"],
            "cases": [
                {
                    key: case[key]
                    for key in (
                        "case_id",
                        "input_file",
                        "output_file",
                        "retrieval_spreadsheet_path",
                        "input_sha256",
                        "input_size",
                    )
                }
                for case in task["cases"]
            ],
        }
        for task in tasks
    ]


def _input_tree_projection(
    tasks: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in tasks:
        for case in task["cases"]:
            for path in (
                f"{task['spreadsheet_path']}/{case['input_file']}",
                f"{case['retrieval_spreadsheet_path']}/initial.xlsx",
            ):
                rows.append(
                    {
                        "path": path,
                        "sha256": case["input_sha256"],
                        "size": case["input_size"],
                    }
                )
    return rows


def _stratum_projection(tasks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "task_id": task["task_id"],
            "evaluation_stratum": task["evaluation_stratum"],
        }
        for task in tasks
    ]


def _task_metadata_projection(
    tasks: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "task_id": task["task_id"],
            "instruction_type": task["instruction_type"],
            "answer_position": task["answer_position"],
        }
        for task in tasks
    ]


def _validate_population(
    tasks: Sequence[Mapping[str, Any]],
    *,
    expected_tasks: int,
    expected_testcases: int,
) -> None:
    if len(tasks) != expected_tasks:
        raise ValueError(
            f"prepared task count differs: {len(tasks)} != {expected_tasks}"
        )
    task_ids: set[str] = set()
    case_ids: set[str] = set()
    testcase_count = 0
    for index, row in enumerate(tasks):
        if (
            row.get("query_index") != index
            or row.get("dataset_index") != index
            or type(row.get("task_id")) is not str
            or not row["task_id"]
            or row["task_id"] in task_ids
            or type(row.get("instruction")) is not str
            or type(row.get("instruction_type")) is not str
            or type(row.get("answer_position")) is not str
            or type(row.get("instruction_missing")) is not bool
            or row["instruction_missing"] != (not row["instruction"].strip())
            or row.get("evaluation_stratum")
            not in {"train_task_overlap", "development_task_overlap", "untouched_task"}
            or _safe_relative(
                row.get("spreadsheet_path"),
                label=f"spreadsheet_path for {row.get('task_id')}",
            ).as_posix()
            != row.get("spreadsheet_path")
            or type(row.get("cases")) is not list
            or not row["cases"]
        ):
            raise ValueError(f"prepared task row {index} differs")
        task_ids.add(row["task_id"])
        for case in row["cases"]:
            if (
                type(case) is not dict
                or set(case)
                != {
                    "case_id",
                    "input_file",
                    "answer_file",
                    "output_file",
                    "retrieval_spreadsheet_path",
                    "input_sha256",
                    "input_size",
                    "answer_sha256",
                    "answer_size",
                }
                or type(case["case_id"]) is not str
                or not case["case_id"]
                or case["case_id"] in case_ids
                or type(case.get("input_file")) is not str
                or type(case.get("answer_file")) is not str
                or type(case.get("output_file")) is not str
                or Path(case["input_file"]).name != case["input_file"]
                or Path(case["answer_file"]).name != case["answer_file"]
                or Path(case["output_file"]).name != case["output_file"]
                or not case["input_file"].endswith("_input.xlsx")
                or not case["answer_file"].endswith("_answer.xlsx")
                or not case["output_file"].endswith("_output.xlsx")
                or case["output_file"]
                != case["input_file"][: -len("_input.xlsx")] + "_output.xlsx"
                or _safe_relative(
                    case.get("retrieval_spreadsheet_path"),
                    label=f"retrieval path for {case.get('case_id')}",
                ).parts[:1]
                != ("retrieval",)
                or any(
                    type(case.get(key)) is not str
                    or len(case[key]) != 64
                    or any(character not in "0123456789abcdef" for character in case[key])
                    for key in ("input_sha256", "answer_sha256")
                )
                or type(case.get("input_size")) is not int
                or case["input_size"] <= 0
                or type(case.get("answer_size")) is not int
                or case["answer_size"] <= 0
            ):
                raise ValueError(f"prepared case row differs for task {row['task_id']}")
            case_ids.add(case["case_id"])
            testcase_count += 1
    if testcase_count != expected_testcases:
        raise ValueError(
            f"prepared testcase count differs: {testcase_count} != {expected_testcases}"
        )


def prepare_soft_hard_population(
    *,
    full_data_path: Path,
    verified_data_path: Path,
    output_dir: Path,
    manifest_path: Path,
    retrieval_manifest_path: Path,
    train_start: int = 0,
    train_end: int = EXCLUDED_TRAIN_CASE_COUNT,
    expected_tasks: int = TASK_COUNT,
    expected_testcases: int = TESTCASE_COUNT,
    expected_excluded_cases: int = EXCLUDED_TRAIN_CASE_COUNT,
    expected_full_dataset_sha256: str | None = FULL_DATASET_SHA256,
    expected_verified_dataset_sha256: str | None = VERIFIED_DATASET_SHA256,
) -> dict[str, Any]:
    """Materialize and seal the full-data view without the train workbook cases."""

    full_root = full_data_path.expanduser().resolve()
    verified_root = verified_data_path.expanduser().resolve()
    output = output_dir.expanduser().absolute()
    manifest_output = manifest_path.expanduser().absolute()
    retrieval_manifest_output = retrieval_manifest_path.expanduser().absolute()
    manifest_staging = manifest_output.with_name(f".{manifest_output.name}.tmp")
    retrieval_manifest_staging = retrieval_manifest_output.with_name(
        f".{retrieval_manifest_output.name}.tmp"
    )
    publication_marker = manifest_output.parent / (
        f".{manifest_output.name}.{retrieval_manifest_output.name}.publication.json"
    )
    publication_marker_staging = publication_marker.with_name(
        f".{publication_marker.name}.tmp"
    )
    _recover_interrupted_publication(
        publication_marker,
        output=output,
        manifest=manifest_output,
        retrieval_manifest=retrieval_manifest_output,
        manifest_staging=manifest_staging,
        retrieval_manifest_staging=retrieval_manifest_staging,
        marker_staging=publication_marker_staging,
    )
    if output.exists() or output.is_symlink():
        raise FileExistsError("prepared dataset output must be fresh")
    if manifest_output.exists() or manifest_output.is_symlink():
        raise FileExistsError("prepared population manifest must be fresh")
    if retrieval_manifest_output.exists() or retrieval_manifest_output.is_symlink():
        raise FileExistsError("prepared retrieval manifest must be fresh")
    if retrieval_manifest_output == manifest_output:
        raise ValueError("evaluation and retrieval manifests must be separate")
    full_rows, full_dataset_sha = _dataset_rows(
        full_root,
        expected_sha256=expected_full_dataset_sha256,
        label="full",
    )
    verified_rows, verified_dataset_sha = _dataset_rows(
        verified_root,
        expected_sha256=expected_verified_dataset_sha256,
        label="verified",
    )
    if (
        type(train_start) is not int
        or type(train_end) is not int
        or train_start != 0
        or train_end - train_start != expected_excluded_cases
        or train_end > len(verified_rows)
    ):
        raise ValueError("train exclusion slice differs from fixed [0,200)")
    full_by_id: dict[str, dict[str, Any]] = {}
    for row in full_rows:
        task_id = str(row.get("id", ""))
        if not task_id or task_id in full_by_id:
            raise ValueError("full dataset task identity differs")
        full_by_id[task_id] = row
    verified_index_by_id: dict[str, int] = {}
    for verified_index, row in enumerate(verified_rows):
        task_id = str(row.get("id", ""))
        if not task_id or task_id in verified_index_by_id:
            raise ValueError("verified dataset task identity differs")
        verified_index_by_id[task_id] = verified_index

    exclusions: dict[str, set[str]] = {}
    exclusion_rows: list[dict[str, Any]] = []
    for train_index, verified_row in enumerate(
        verified_rows[train_start:train_end], start=train_start
    ):
        task_id = str(verified_row.get("id", ""))
        full_row = full_by_id.get(task_id)
        if full_row is None:
            raise ValueError(f"verified train task is absent from full data: {task_id}")
        source_dir = _spreadsheet_dir(full_root, full_row)
        verified_dir = _spreadsheet_dir(verified_root, verified_row)
        answer_name = f"1_{task_id}_answer.xlsx"
        verified_input_name = f"1_{task_id}_init.xlsx"
        verified_answer_name = f"1_{task_id}_golden.xlsx"
        input_candidates = (f"1_{task_id}_input.xlsx", f"1_{task_id}_input .xlsx")
        present_inputs = [name for name in input_candidates if (source_dir / name).is_file()]
        if (
            len(present_inputs) != 1
            or not (source_dir / answer_name).is_file()
            or not (verified_dir / verified_input_name).is_file()
            or not (verified_dir / verified_answer_name).is_file()
        ):
            raise ValueError(f"fixed train workbook pair differs for {task_id}")
        full_input_sha = _sha256(source_dir / present_inputs[0])
        full_answer_sha = _sha256(source_dir / answer_name)
        verified_input_sha = _sha256(verified_dir / verified_input_name)
        verified_answer_sha = _sha256(verified_dir / verified_answer_name)
        exclusions[task_id] = {present_inputs[0], answer_name}
        exclusion_rows.append(
            {
                "train_index": train_index,
                "task_id": task_id,
                "input_file": present_inputs[0],
                "answer_file": answer_name,
                "full_input_sha256": full_input_sha,
                "verified_input_sha256": verified_input_sha,
                "input_bytes_equal": full_input_sha == verified_input_sha,
                "full_answer_sha256": full_answer_sha,
                "verified_answer_sha256": verified_answer_sha,
                "answer_bytes_equal": full_answer_sha == verified_answer_sha,
                "verified_instruction_sha256": hashlib.sha256(
                    str(verified_row.get("instruction", "")).encode("utf-8")
                ).hexdigest(),
                "full_instruction_sha256": hashlib.sha256(
                    str(full_row.get("instruction", "")).encode("utf-8")
                ).hexdigest(),
                "instruction_text_equal": (
                    verified_row.get("instruction") == full_row.get("instruction")
                ),
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    link_counts = {"hardlink": 0, "copy": 0}
    normalization_rows: list[dict[str, str]] = []
    tasks: list[dict[str, Any]] = []
    published = False
    try:
        for dataset_index, row in enumerate(full_rows):
            task_id = str(row["id"])
            relative_dir = _safe_relative(
                row.get("spreadsheet_path") or f"spreadsheet/{task_id}",
                label=f"spreadsheet_path for {task_id}",
            )
            source_dir = _spreadsheet_dir(full_root, row)
            target_dir = staging / relative_dir
            target_dir.mkdir(parents=True)
            for source in sorted(source_dir.iterdir(), key=lambda item: item.name):
                if source.name in exclusions.get(task_id, set()):
                    continue
                if source.suffix.lower() != ".xlsx":
                    continue
                target_name = _normalized_name(source.name)
                if target_name != source.name:
                    normalization_rows.append(
                        {
                            "task_id": task_id,
                            "source_name": source.name,
                            "target_name": target_name,
                        }
                    )
                mode = _link_or_copy_regular(source, target_dir / target_name)
                link_counts[mode] += 1

            names = {path.name for path in target_dir.iterdir() if path.is_file()}
            inputs = sorted(name for name in names if name.endswith("_input.xlsx"))
            answers = {name for name in names if name.endswith("_answer.xlsx")}
            cases: list[dict[str, str]] = []
            for input_name in inputs:
                stem = input_name[: -len("_input.xlsx")]
                answer_name = f"{stem}_answer.xlsx"
                if answer_name not in answers:
                    raise ValueError(f"answer is absent for {task_id}/{input_name}")
                case_id = f"{task_id}__{stem}"
                retrieval_relative = Path(
                    "retrieval", hashlib.sha256(case_id.encode()).hexdigest()
                )
                retrieval_dir = staging / retrieval_relative
                retrieval_dir.mkdir(parents=True, mode=0o700)
                mode = _link_or_copy_regular(
                    target_dir / input_name, retrieval_dir / "initial.xlsx"
                )
                link_counts[mode] += 1
                cases.append(
                    {
                        "case_id": case_id,
                        "input_file": input_name,
                        "answer_file": answer_name,
                        "output_file": f"{stem}_output.xlsx",
                        "retrieval_spreadsheet_path": retrieval_relative.as_posix(),
                        "input_sha256": _sha256(target_dir / input_name),
                        "input_size": (target_dir / input_name).stat().st_size,
                        "answer_sha256": _sha256(target_dir / answer_name),
                        "answer_size": (target_dir / answer_name).stat().st_size,
                    }
                )
            if len(cases) != len(answers):
                raise ValueError(f"input/answer pairing differs for {task_id}")
            instruction = row.get("instruction")
            if type(instruction) is not str:
                raise ValueError(f"instruction differs for {task_id}")
            instruction_missing = not instruction.strip()
            verified_index = verified_index_by_id.get(task_id)
            if verified_index is None:
                evaluation_stratum = "untouched_task"
            elif verified_index < EXCLUDED_TRAIN_CASE_COUNT:
                evaluation_stratum = "train_task_overlap"
            else:
                evaluation_stratum = "development_task_overlap"
            tasks.append(
                {
                    "query_index": dataset_index,
                    "dataset_index": dataset_index,
                    "task_id": task_id,
                    "instruction": instruction,
                    "instruction_missing": instruction_missing,
                    "evaluation_stratum": evaluation_stratum,
                    "spreadsheet_path": relative_dir.as_posix(),
                    "instruction_type": row.get("instruction_type", ""),
                    "answer_position": row.get("answer_position", ""),
                    "cases": cases,
                }
            )

        _validate_population(
            tasks,
            expected_tasks=expected_tasks,
            expected_testcases=expected_testcases,
        )
        dataset_bytes = canonical_json_bytes(full_rows)
        (staging / "dataset.json").write_bytes(dataset_bytes)
        prepared_tree_sha = _tree_sha256(staging)
        projection = _query_projection(tasks)
        input_only_tasks = _input_only_tasks(tasks)
        retrieval_projection = _retrieval_projection(input_only_tasks)
        input_tree_projection_sha = hashlib.sha256(
            canonical_json_bytes(_input_tree_projection(input_only_tasks))
        ).hexdigest()
        retrieval_body = {
            "format": RETRIEVAL_FORMAT,
            "claim_scope": (
                "query instruction and target input workbook only; no answer path, "
                "answer hash, verifier result, target outcome, or Agent trace"
            ),
            "task_count": len(tasks),
            "testcase_count": len(retrieval_projection),
            "prepared_input_tree_sha256": input_tree_projection_sha,
            "retrieval_projection_sha256": hashlib.sha256(
                canonical_json_bytes(retrieval_projection)
            ).hexdigest(),
            "tasks": input_only_tasks,
        }
        retrieval_manifest = {
            **retrieval_body,
            "self_sha256": hashlib.sha256(
                canonical_json_bytes(retrieval_body)
            ).hexdigest(),
        }
        body = {
            "format": FORMAT,
            "source_full_dataset_sha256": full_dataset_sha,
            "source_verified_dataset_sha256": verified_dataset_sha,
            "prepared_dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
            "prepared_dataset_tree_sha256": prepared_tree_sha,
            "train_start": train_start,
            "train_end": train_end,
            "excluded_train_case_count": len(exclusion_rows),
            "exclusion_workbook_identity": {
                "input_byte_identical_count": sum(
                    row["input_bytes_equal"] for row in exclusion_rows
                ),
                "answer_byte_identical_count": sum(
                    row["answer_bytes_equal"] for row in exclusion_rows
                ),
            },
            "task_count": len(tasks),
            "testcase_count": sum(len(row["cases"]) for row in tasks),
            "missing_instruction_task_count": sum(
                row["instruction_missing"] for row in tasks
            ),
            "evaluation_strata": {
                name: {
                    "task_count": sum(
                        row["evaluation_stratum"] == name for row in tasks
                    ),
                    "testcase_count": sum(
                        len(row["cases"])
                        for row in tasks
                        if row["evaluation_stratum"] == name
                    ),
                }
                for name in (
                    "train_task_overlap",
                    "development_task_overlap",
                    "untouched_task",
                )
            },
            "query_projection_sha256": hashlib.sha256(
                canonical_json_bytes(projection)
            ).hexdigest(),
            "case_projection_sha256": hashlib.sha256(
                canonical_json_bytes(_case_projection(tasks))
            ).hexdigest(),
            "exclusion_projection_sha256": hashlib.sha256(
                canonical_json_bytes(exclusion_rows)
            ).hexdigest(),
            "stratum_projection_sha256": hashlib.sha256(
                canonical_json_bytes(_stratum_projection(tasks))
            ).hexdigest(),
            "task_metadata_projection_sha256": hashlib.sha256(
                canonical_json_bytes(_task_metadata_projection(tasks))
            ).hexdigest(),
            "retrieval_manifest_sha256": retrieval_manifest["self_sha256"],
            "materialization": link_counts,
            "exclusions": exclusion_rows,
            "normalizations": normalization_rows,
            "tasks": tasks,
        }
        manifest = {
            **body,
            "self_sha256": hashlib.sha256(canonical_json_bytes(body)).hexdigest(),
        }
        manifest_output.parent.mkdir(parents=True, exist_ok=True)
        if manifest_staging.exists() or manifest_staging.is_symlink():
            raise FileExistsError("prepared population manifest staging path differs")
        if retrieval_manifest_staging.exists() or retrieval_manifest_staging.is_symlink():
            raise FileExistsError("prepared retrieval manifest staging path differs")
        manifest_staging.write_bytes(canonical_json_bytes(manifest))
        retrieval_manifest_staging.write_bytes(
            canonical_json_bytes(retrieval_manifest)
        )
        publication = {
            "format": "degs_soft_hard_population_publication_v1",
            "output": str(output),
            "manifest": str(manifest_output),
            "retrieval_manifest": str(retrieval_manifest_output),
            "manifest_staging": str(manifest_staging),
            "retrieval_manifest_staging": str(retrieval_manifest_staging),
            "marker_staging": str(publication_marker_staging),
        }
        publication_marker_staging.write_bytes(canonical_json_bytes(publication))
        os.replace(publication_marker_staging, publication_marker)
        os.replace(staging, output)
        os.replace(manifest_staging, manifest_output)
        os.replace(retrieval_manifest_staging, retrieval_manifest_output)
        publication_marker.unlink()
        published = True
        return manifest
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)


def load_prepared_population(
    *,
    data_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    root = data_path.expanduser().absolute()
    manifest_file = manifest_path.expanduser().absolute()
    if root.is_symlink() or not root.is_dir():
        raise FileNotFoundError("prepared dataset directory differs")
    if manifest_file.is_symlink() or not manifest_file.is_file():
        raise FileNotFoundError("prepared population manifest differs")
    root = root.resolve()
    manifest_file = manifest_file.resolve()
    manifest = _read_json(manifest_file)
    if type(manifest) is not dict:
        raise ValueError("prepared population manifest differs")
    unsigned = {key: value for key, value in manifest.items() if key != "self_sha256"}
    tasks = manifest.get("tasks")
    if (
        manifest.get("format") != FORMAT
        or manifest.get("self_sha256")
        != hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
        or manifest.get("task_count") != TASK_COUNT
        or manifest.get("testcase_count") != TESTCASE_COUNT
        or manifest.get("excluded_train_case_count")
        != EXCLUDED_TRAIN_CASE_COUNT
        or manifest.get("source_full_dataset_sha256") != FULL_DATASET_SHA256
        or manifest.get("source_verified_dataset_sha256")
        != VERIFIED_DATASET_SHA256
        or manifest.get("train_start") != 0
        or manifest.get("train_end") != EXCLUDED_TRAIN_CASE_COUNT
        or manifest.get("query_projection_sha256")
        != EXPECTED_QUERY_PROJECTION_SHA256
        or manifest.get("case_projection_sha256")
        != EXPECTED_CASE_PROJECTION_SHA256
        or manifest.get("exclusion_projection_sha256")
        != EXPECTED_EXCLUSION_PROJECTION_SHA256
        or manifest.get("prepared_dataset_tree_sha256")
        != EXPECTED_PREPARED_TREE_SHA256
        or manifest.get("stratum_projection_sha256")
        != EXPECTED_STRATUM_PROJECTION_SHA256
        or manifest.get("task_metadata_projection_sha256")
        != EXPECTED_TASK_METADATA_PROJECTION_SHA256
        or manifest.get("evaluation_strata") != EXPECTED_EVALUATION_STRATA
        or manifest.get("exclusion_workbook_identity")
        != {
            "input_byte_identical_count": EXPECTED_BYTE_IDENTICAL_EXCLUSIONS["input"],
            "answer_byte_identical_count": EXPECTED_BYTE_IDENTICAL_EXCLUSIONS["answer"],
        }
        or type(tasks) is not list
    ):
        raise ValueError("prepared population identity differs")
    _validate_population(
        tasks,
        expected_tasks=TASK_COUNT,
        expected_testcases=TESTCASE_COUNT,
    )
    exclusions = manifest.get("exclusions")
    if (
        type(exclusions) is not list
        or len(exclusions) != EXCLUDED_TRAIN_CASE_COUNT
        or [row.get("train_index") for row in exclusions]
        != list(range(EXCLUDED_TRAIN_CASE_COUNT))
        or len({row.get("task_id") for row in exclusions})
        != EXCLUDED_TRAIN_CASE_COUNT
        or hashlib.sha256(canonical_json_bytes(exclusions)).hexdigest()
        != EXPECTED_EXCLUSION_PROJECTION_SHA256
        or any(
            type(row) is not dict
            or set(row) != EXCLUSION_FIELDS
            or type(row.get("task_id")) is not str
            or not row["task_id"]
            or Path(str(row.get("input_file", ""))).name != row.get("input_file")
            or Path(str(row.get("answer_file", ""))).name != row.get("answer_file")
            or row.get("input_file")
            not in {
                f"1_{row['task_id']}_input.xlsx",
                f"1_{row['task_id']}_input .xlsx",
            }
            or row.get("answer_file") != f"1_{row['task_id']}_answer.xlsx"
            or any(
                type(row.get(key)) is not str
                or len(row[key]) != 64
                or any(character not in "0123456789abcdef" for character in row[key])
                for key in (
                    "full_input_sha256",
                    "verified_input_sha256",
                    "full_answer_sha256",
                    "verified_answer_sha256",
                )
            )
            or row.get("input_bytes_equal")
            != (row.get("full_input_sha256") == row.get("verified_input_sha256"))
            or row.get("answer_bytes_equal")
            != (row.get("full_answer_sha256") == row.get("verified_answer_sha256"))
            for row in exclusions
        )
    ):
        raise ValueError("fixed train exclusion projection differs")
    dataset_file = root / "dataset.json"
    if (
        dataset_file.is_symlink()
        or not dataset_file.is_file()
        or _sha256(dataset_file) != manifest.get("prepared_dataset_sha256")
        or _tree_sha256(root) != manifest.get("prepared_dataset_tree_sha256")
        or hashlib.sha256(canonical_json_bytes(_case_projection(tasks))).hexdigest()
        != EXPECTED_CASE_PROJECTION_SHA256
        or hashlib.sha256(canonical_json_bytes(_stratum_projection(tasks))).hexdigest()
        != EXPECTED_STRATUM_PROJECTION_SHA256
        or hashlib.sha256(
            canonical_json_bytes(_task_metadata_projection(tasks))
        ).hexdigest()
        != EXPECTED_TASK_METADATA_PROJECTION_SHA256
        or hashlib.sha256(canonical_json_bytes(_query_projection(tasks))).hexdigest()
        != manifest.get("query_projection_sha256")
    ):
        raise ValueError("prepared dataset identity differs")
    for task in tasks:
        directory = root / task["spreadsheet_path"]
        for case in task["cases"]:
            input_path = directory / case["input_file"]
            answer_path = directory / case["answer_file"]
            retrieval_path = root / case["retrieval_spreadsheet_path"] / "initial.xlsx"
            if (
                _sha256(input_path) != case["input_sha256"]
                or input_path.stat().st_size != case["input_size"]
                or _sha256(answer_path) != case["answer_sha256"]
                or answer_path.stat().st_size != case["answer_size"]
                or _sha256(retrieval_path) != case["input_sha256"]
                or retrieval_path.stat().st_size != case["input_size"]
            ):
                raise ValueError("prepared case workbook identity differs")
    return manifest


def load_prepared_retrieval_population(
    *, data_path: Path, manifest_path: Path
) -> dict[str, Any]:
    """Validate the input-only retrieval view without opening evaluation gold."""

    root = data_path.expanduser().resolve()
    manifest_file = manifest_path.expanduser().resolve()
    if root.is_symlink() or not root.is_dir():
        raise FileNotFoundError("prepared retrieval dataset differs")
    if manifest_file.is_symlink() or not manifest_file.is_file():
        raise FileNotFoundError("prepared retrieval manifest differs")
    manifest = _read_json(manifest_file)
    if type(manifest) is not dict:
        raise ValueError("prepared retrieval manifest differs")
    unsigned = {key: value for key, value in manifest.items() if key != "self_sha256"}
    tasks = manifest.get("tasks")
    if (
        manifest.get("format") != RETRIEVAL_FORMAT
        or manifest.get("self_sha256")
        != hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
        or manifest.get("task_count") != TASK_COUNT
        or manifest.get("testcase_count") != TESTCASE_COUNT
        or manifest.get("retrieval_projection_sha256")
        != EXPECTED_RETRIEVAL_PROJECTION_SHA256
        or manifest.get("prepared_input_tree_sha256")
        != EXPECTED_INPUT_TREE_PROJECTION_SHA256
        or type(tasks) is not list
        or len(tasks) != TASK_COUNT
    ):
        raise ValueError("prepared retrieval population identity differs")
    task_ids: set[str] = set()
    case_ids: set[str] = set()
    case_count = 0
    for task_index, task in enumerate(tasks):
        if (
            type(task) is not dict
            or set(task)
            != {
                "query_index",
                "dataset_index",
                "task_id",
                "instruction",
                "instruction_type",
                "answer_position",
                "spreadsheet_path",
                "cases",
            }
            or task.get("query_index") != task_index
            or task.get("dataset_index") != task_index
            or type(task.get("task_id")) is not str
            or not task["task_id"]
            or task["task_id"] in task_ids
            or type(task.get("instruction")) is not str
            or type(task.get("instruction_type")) is not str
            or type(task.get("answer_position")) is not str
            or _safe_relative(
                task.get("spreadsheet_path"),
                label=f"execution spreadsheet path {task_index}",
            ).as_posix()
            != task.get("spreadsheet_path")
            or type(task.get("cases")) is not list
            or not task["cases"]
        ):
            raise ValueError(f"prepared retrieval task {task_index} differs")
        task_ids.add(task["task_id"])
        execution_dir = root / task["spreadsheet_path"]
        for case in task["cases"]:
            path = _safe_relative(
                case.get("retrieval_spreadsheet_path")
                if type(case) is dict
                else None,
                label=f"retrieval spreadsheet path {case_count}",
            )
            if (
                type(case) is not dict
                or set(case)
                != {
                    "case_id",
                    "input_file",
                    "output_file",
                    "retrieval_spreadsheet_path",
                    "input_sha256",
                    "input_size",
                }
                or type(case.get("case_id")) is not str
                or not case["case_id"]
                or case["case_id"] in case_ids
                or Path(str(case.get("input_file", ""))).name
                != case.get("input_file")
                or Path(str(case.get("output_file", ""))).name
                != case.get("output_file")
                or path.parts[:1] != ("retrieval",)
                or type(case.get("input_sha256")) is not str
                or len(case["input_sha256"]) != 64
                or type(case.get("input_size")) is not int
                or case["input_size"] <= 0
            ):
                raise ValueError(f"prepared retrieval case {case_count} differs")
            execution_input = execution_dir / case["input_file"]
            directory = root / path
            input_path = directory / "initial.xlsx"
            if (
                execution_dir.is_symlink()
                or not execution_dir.is_dir()
                or execution_input.is_symlink()
                or not execution_input.is_file()
                or directory.is_symlink()
                or not directory.is_dir()
                or {item.name for item in directory.iterdir()} != {"initial.xlsx"}
                or input_path.is_symlink()
                or not input_path.is_file()
                or _sha256(execution_input) != case["input_sha256"]
                or execution_input.stat().st_size != case["input_size"]
                or _sha256(input_path) != case["input_sha256"]
                or input_path.stat().st_size != case["input_size"]
            ):
                raise ValueError(f"prepared retrieval workbook {case_count} differs")
            case_ids.add(case["case_id"])
            case_count += 1
    if len(task_ids) != TASK_COUNT or case_count != TESTCASE_COUNT:
        raise ValueError("prepared retrieval task population differs")
    if (
        hashlib.sha256(
            canonical_json_bytes(_retrieval_projection(tasks))
        ).hexdigest()
        != EXPECTED_RETRIEVAL_PROJECTION_SHA256
        or hashlib.sha256(
            canonical_json_bytes(_input_tree_projection(tasks))
        ).hexdigest()
        != EXPECTED_INPUT_TREE_PROJECTION_SHA256
    ):
        raise ValueError("prepared retrieval query projection differs")
    return manifest


def query_rows(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    tasks = manifest.get("tasks")
    if type(tasks) is not list:
        raise ValueError("prepared population tasks differ")
    return _query_projection(tasks)


def retrieval_query_rows(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    tasks = manifest.get("tasks")
    if type(tasks) is not list:
        raise ValueError("prepared retrieval tasks differ")
    return _retrieval_projection(tasks)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare the fixed SpreadsheetBench Soft/Hard population."
    )
    parser.add_argument("--full-data-path", type=Path, required=True)
    parser.add_argument("--verified-data-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--retrieval-manifest-path", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = prepare_soft_hard_population(
        full_data_path=args.full_data_path,
        verified_data_path=args.verified_data_path,
        output_dir=args.output_dir,
        manifest_path=args.manifest_path,
        retrieval_manifest_path=args.retrieval_manifest_path,
    )
    print(json.dumps({key: manifest[key] for key in ("task_count", "testcase_count", "self_sha256")}, sort_keys=True))
    return 0


__all__ = [
    "EXCLUDED_TRAIN_CASE_COUNT",
    "FORMAT",
    "RETRIEVAL_FORMAT",
    "FULL_DATASET_SHA256",
    "EXPECTED_QUERY_PROJECTION_SHA256",
    "EXPECTED_EVALUATION_STRATA",
    "EXPECTED_BYTE_IDENTICAL_EXCLUSIONS",
    "EXPECTED_CASE_PROJECTION_SHA256",
    "EXPECTED_EXCLUSION_PROJECTION_SHA256",
    "EXPECTED_PREPARED_TREE_SHA256",
    "EXPECTED_STRATUM_PROJECTION_SHA256",
    "EXPECTED_TASK_METADATA_PROJECTION_SHA256",
    "TASK_COUNT",
    "TESTCASE_COUNT",
    "VERIFIED_DATASET_SHA256",
    "canonical_json_bytes",
    "load_prepared_population",
    "load_prepared_retrieval_population",
    "prepare_soft_hard_population",
    "retrieval_query_rows",
    "query_rows",
]


if __name__ == "__main__":
    raise SystemExit(main())
