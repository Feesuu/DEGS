from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import degs.soft_hard_dataset as soft_hard_dataset
from degs.core import canonical_json_bytes
from degs.soft_hard_dataset import (
    load_prepared_population,
    load_prepared_retrieval_population,
    prepare_soft_hard_population,
    query_rows,
    retrieval_query_rows,
)


def _write(path: Path, payload: bytes = b"xlsx") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _dataset_rows() -> list[dict[str, str]]:
    return [
        {
            "id": "13-1",
            "instruction": "Transform the first workbook.",
            "spreadsheet_path": "spreadsheet/13-1",
            "instruction_type": "Cell-Level Manipulation",
            "answer_position": "A1",
        },
        {
            "id": "371-33",
            "instruction": "Transform the second workbook.",
            "spreadsheet_path": "spreadsheet/371-33",
            "instruction_type": "Sheet-Level Manipulation",
            "answer_position": "B1",
        },
    ]


def test_prepare_excludes_exact_train_case_and_keeps_task_group(
    tmp_path: Path, monkeypatch
) -> None:
    full = tmp_path / "full"
    verified = tmp_path / "verified"
    rows = _dataset_rows()
    _write(full / "dataset.json", json.dumps(rows).encode())
    verified_row = {**rows[0], "instruction": "Verified wording differs."}
    _write(verified / "dataset.json", json.dumps([verified_row]).encode())
    _write(verified / "spreadsheet/13-1/1_13-1_init.xlsx")
    _write(verified / "spreadsheet/13-1/1_13-1_golden.xlsx")
    for index in (1, 2):
        _write(full / "spreadsheet/13-1" / f"{index}_13-1_input.xlsx")
        _write(full / "spreadsheet/13-1" / f"{index}_13-1_answer.xlsx")
    _write(full / "spreadsheet/371-33/1_371-33_input.xlsx")
    _write(full / "spreadsheet/371-33/1_371-33_answer.xlsx")
    _write(full / "spreadsheet/371-33/2_371-33_input .xlsx")
    _write(full / "spreadsheet/371-33/2_371-33_answer.xlsx")

    prepared = tmp_path / "prepared"
    manifest_path = tmp_path / "population.json"
    retrieval_manifest_path = tmp_path / "input.json"
    manifest = prepare_soft_hard_population(
        full_data_path=full,
        verified_data_path=verified,
        output_dir=prepared,
        manifest_path=manifest_path,
        retrieval_manifest_path=retrieval_manifest_path,
        train_end=1,
        expected_tasks=2,
        expected_testcases=3,
        expected_excluded_cases=1,
        expected_full_dataset_sha256=None,
        expected_verified_dataset_sha256=None,
    )

    assert not (prepared / "spreadsheet/13-1/1_13-1_input.xlsx").exists()
    assert not (prepared / "spreadsheet/13-1/1_13-1_answer.xlsx").exists()
    assert (prepared / "spreadsheet/13-1/2_13-1_input.xlsx").is_file()
    assert (prepared / "spreadsheet/371-33/2_371-33_input.xlsx").is_file()
    assert not (prepared / "spreadsheet/371-33/2_371-33_input .xlsx").exists()
    assert all(not path.is_symlink() for path in prepared.rglob("*"))
    assert manifest["task_count"] == 2
    assert manifest["testcase_count"] == 3
    assert len(manifest["normalizations"]) == 1
    assert manifest["exclusions"][0]["instruction_text_equal"] is False
    monkeypatch.setattr(soft_hard_dataset, "TASK_COUNT", 2)
    monkeypatch.setattr(soft_hard_dataset, "TESTCASE_COUNT", 3)
    monkeypatch.setattr(soft_hard_dataset, "EXCLUDED_TRAIN_CASE_COUNT", 1)
    monkeypatch.setattr(
        soft_hard_dataset,
        "FULL_DATASET_SHA256",
        manifest["source_full_dataset_sha256"],
    )
    monkeypatch.setattr(
        soft_hard_dataset,
        "VERIFIED_DATASET_SHA256",
        manifest["source_verified_dataset_sha256"],
    )
    monkeypatch.setattr(
        soft_hard_dataset,
        "EXPECTED_QUERY_PROJECTION_SHA256",
        manifest["query_projection_sha256"],
    )
    monkeypatch.setattr(
        soft_hard_dataset,
        "EXPECTED_CASE_PROJECTION_SHA256",
        manifest["case_projection_sha256"],
    )
    monkeypatch.setattr(
        soft_hard_dataset,
        "EXPECTED_EXCLUSION_PROJECTION_SHA256",
        manifest["exclusion_projection_sha256"],
    )
    monkeypatch.setattr(
        soft_hard_dataset,
        "EXPECTED_PREPARED_TREE_SHA256",
        manifest["prepared_dataset_tree_sha256"],
    )
    monkeypatch.setattr(
        soft_hard_dataset,
        "EXPECTED_STRATUM_PROJECTION_SHA256",
        manifest["stratum_projection_sha256"],
    )
    monkeypatch.setattr(
        soft_hard_dataset,
        "EXPECTED_TASK_METADATA_PROJECTION_SHA256",
        manifest["task_metadata_projection_sha256"],
    )
    monkeypatch.setattr(
        soft_hard_dataset,
        "EXPECTED_EVALUATION_STRATA",
        manifest["evaluation_strata"],
    )
    monkeypatch.setattr(
        soft_hard_dataset,
        "EXPECTED_BYTE_IDENTICAL_EXCLUSIONS",
        {"input": 1, "answer": 1},
    )
    retrieval_manifest = json.loads(
        retrieval_manifest_path.read_text(encoding="utf-8")
    )
    monkeypatch.setattr(
        soft_hard_dataset,
        "EXPECTED_RETRIEVAL_PROJECTION_SHA256",
        retrieval_manifest["retrieval_projection_sha256"],
    )
    monkeypatch.setattr(
        soft_hard_dataset,
        "EXPECTED_INPUT_TREE_PROJECTION_SHA256",
        retrieval_manifest["prepared_input_tree_sha256"],
    )
    loaded = load_prepared_population(
        data_path=prepared,
        manifest_path=manifest_path,
    )
    assert query_rows(loaded) == [
        {
            "query_index": 0,
            "dataset_index": 0,
            "task_id": "13-1",
            "instruction": "Transform the first workbook.",
        },
        {
            "query_index": 1,
            "dataset_index": 1,
            "task_id": "371-33",
            "instruction": "Transform the second workbook.",
        },
    ]
    retrieval = load_prepared_retrieval_population(
        data_path=prepared,
        manifest_path=retrieval_manifest_path,
    )
    retrieval_rows = retrieval_query_rows(retrieval)
    assert len(retrieval_rows) == 3
    assert retrieval_rows[0]["case_id"].startswith("13-1__2_")
    assert retrieval_rows[0]["spreadsheet_path"].startswith("retrieval/")
    assert "answer_file" not in retrieval_rows[0]
    assert b"answer_file" not in retrieval_manifest_path.read_bytes()

    tampered = json.loads(manifest_path.read_text(encoding="utf-8"))
    tampered["tasks"][0]["answer_position"] = "ZZZ999"
    unsigned = {key: value for key, value in tampered.items() if key != "self_sha256"}
    tampered["self_sha256"] = hashlib.sha256(
        canonical_json_bytes(unsigned)
    ).hexdigest()
    manifest_path.write_bytes(canonical_json_bytes(tampered))
    with pytest.raises(ValueError, match="prepared dataset identity differs"):
        load_prepared_population(
            data_path=prepared,
            manifest_path=manifest_path,
        )


def test_missing_instruction_is_retained_without_rewriting_the_retrieval_query(
    tmp_path: Path,
) -> None:
    full = tmp_path / "full"
    verified = tmp_path / "verified"
    row = {
        "id": "blank",
        "instruction": "\u00a0",
        "spreadsheet_path": "spreadsheet/blank",
        "instruction_type": "Cell-Level Manipulation",
        "answer_position": "A1",
    }
    _write(full / "dataset.json", json.dumps([row]).encode())
    _write(verified / "dataset.json", json.dumps([]).encode())
    _write(full / "spreadsheet/blank/1_blank_input.xlsx")
    _write(full / "spreadsheet/blank/1_blank_answer.xlsx")

    manifest = prepare_soft_hard_population(
        full_data_path=full,
        verified_data_path=verified,
        output_dir=tmp_path / "prepared",
        manifest_path=tmp_path / "population.json",
        retrieval_manifest_path=tmp_path / "input.json",
        train_end=0,
        expected_tasks=1,
        expected_testcases=1,
        expected_excluded_cases=0,
        expected_full_dataset_sha256=None,
        expected_verified_dataset_sha256=None,
    )

    assert manifest["tasks"][0]["instruction"] == "\u00a0"
    assert manifest["tasks"][0]["instruction_missing"] is True
    assert query_rows(manifest)[0]["instruction"] == "\u00a0"
    assert manifest["missing_instruction_task_count"] == 1


def test_prepare_rejects_unsafe_spreadsheet_path(tmp_path: Path) -> None:
    full = tmp_path / "full"
    verified = tmp_path / "verified"
    row = {
        "id": "bad",
        "instruction": "bad",
        "spreadsheet_path": "../escape",
    }
    _write(full / "dataset.json", json.dumps([row]).encode())
    _write(verified / "dataset.json", json.dumps([row]).encode())

    with pytest.raises(ValueError, match="unsafe spreadsheet_path"):
        prepare_soft_hard_population(
            full_data_path=full,
            verified_data_path=verified,
            output_dir=tmp_path / "prepared",
            manifest_path=tmp_path / "population.json",
            retrieval_manifest_path=tmp_path / "input.json",
            train_end=1,
            expected_tasks=1,
            expected_testcases=0,
            expected_excluded_cases=1,
            expected_full_dataset_sha256=None,
            expected_verified_dataset_sha256=None,
        )


def test_interrupted_population_publication_is_recoverable(tmp_path: Path) -> None:
    output = tmp_path / "prepared"
    output.mkdir()
    (output / "partial").write_text("partial", encoding="utf-8")
    manifest = tmp_path / "population.json"
    retrieval = tmp_path / "input.json"
    manifest_staging = tmp_path / ".population.json.tmp"
    retrieval_staging = tmp_path / ".input.json.tmp"
    marker_staging = tmp_path / "..population.json.input.json.publication.json.tmp"
    manifest.write_text("{}", encoding="utf-8")
    manifest_staging.write_text("staged", encoding="utf-8")
    retrieval_staging.write_text("staged", encoding="utf-8")
    marker = tmp_path / ".population.json.input.json.publication.json"
    marker.write_bytes(
        canonical_json_bytes(
            {
                "format": "degs_soft_hard_population_publication_v1",
                "output": str(output),
                "manifest": str(manifest),
                "retrieval_manifest": str(retrieval),
                "manifest_staging": str(manifest_staging),
                "retrieval_manifest_staging": str(retrieval_staging),
                "marker_staging": str(marker_staging),
            }
        )
    )

    soft_hard_dataset._recover_interrupted_publication(
        marker,
        output=output,
        manifest=manifest,
        retrieval_manifest=retrieval,
        manifest_staging=manifest_staging,
        retrieval_manifest_staging=retrieval_staging,
        marker_staging=marker_staging,
    )

    assert not output.exists()
    assert not manifest.exists()
    assert not manifest_staging.exists()
    assert not retrieval_staging.exists()
    assert not marker.exists()


def test_incomplete_publication_marker_is_recoverable_before_any_final(
    tmp_path: Path,
) -> None:
    output = tmp_path / "prepared"
    manifest = tmp_path / "population.json"
    retrieval = tmp_path / "input.json"
    manifest_staging = tmp_path / ".population.json.tmp"
    retrieval_staging = tmp_path / ".input.json.tmp"
    marker = tmp_path / ".population.json.input.json.publication.json"
    marker_staging = tmp_path / "..population.json.input.json.publication.json.tmp"
    manifest_staging.write_text("staged", encoding="utf-8")
    retrieval_staging.write_text("staged", encoding="utf-8")
    marker.write_text('{"format":', encoding="utf-8")

    soft_hard_dataset._recover_interrupted_publication(
        marker,
        output=output,
        manifest=manifest,
        retrieval_manifest=retrieval,
        manifest_staging=manifest_staging,
        retrieval_manifest_staging=retrieval_staging,
        marker_staging=marker_staging,
    )

    assert not marker.exists()
    assert not manifest_staging.exists()
    assert not retrieval_staging.exists()


@pytest.mark.parametrize("fail_on_replace", [1, 2, 3, 4])
def test_prepare_recovers_each_publication_rename_boundary(
    tmp_path: Path, monkeypatch, fail_on_replace: int
) -> None:
    full = tmp_path / "full"
    verified = tmp_path / "verified"
    row = {
        "id": "one",
        "instruction": "Transform the workbook.",
        "spreadsheet_path": "spreadsheet/one",
        "instruction_type": "Cell-Level Manipulation",
        "answer_position": "A1",
    }
    _write(full / "dataset.json", json.dumps([row]).encode())
    _write(verified / "dataset.json", b"[]")
    _write(full / "spreadsheet/one/1_one_input.xlsx")
    _write(full / "spreadsheet/one/1_one_answer.xlsx")
    output = tmp_path / "prepared"
    manifest = tmp_path / "population.json"
    retrieval = tmp_path / "input.json"
    real_replace = soft_hard_dataset.os.replace
    replace_count = 0

    def interrupted_replace(source: Path, target: Path) -> None:
        nonlocal replace_count
        replace_count += 1
        if replace_count == fail_on_replace:
            raise RuntimeError("simulated publication interruption")
        real_replace(source, target)

    with monkeypatch.context() as patcher:
        patcher.setattr(soft_hard_dataset.os, "replace", interrupted_replace)
        with pytest.raises(RuntimeError, match="simulated publication interruption"):
            prepare_soft_hard_population(
                full_data_path=full,
                verified_data_path=verified,
                output_dir=output,
                manifest_path=manifest,
                retrieval_manifest_path=retrieval,
                train_end=0,
                expected_tasks=1,
                expected_testcases=1,
                expected_excluded_cases=0,
                expected_full_dataset_sha256=None,
                expected_verified_dataset_sha256=None,
            )

    result = prepare_soft_hard_population(
        full_data_path=full,
        verified_data_path=verified,
        output_dir=output,
        manifest_path=manifest,
        retrieval_manifest_path=retrieval,
        train_end=0,
        expected_tasks=1,
        expected_testcases=1,
        expected_excluded_cases=0,
        expected_full_dataset_sha256=None,
        expected_verified_dataset_sha256=None,
    )

    assert result["task_count"] == 1
    assert output.is_dir()
    assert manifest.is_file()
    assert retrieval.is_file()
