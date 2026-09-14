from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from degs.soft_hard_evaluate import (
    _snapshot_workbook,
    aggregate_results,
    evaluate_run,
)


def test_soft_hard_are_task_macro_averages_not_testcase_micro_average() -> None:
    population = {
        "tasks": [
            {
                "task_id": "a",
                "instruction_type": "Cell-Level",
                "cases": [{"case_id": "a1"}, {"case_id": "a2"}],
            },
            {
                "task_id": "b",
                "instruction_type": "Sheet-Level",
                "cases": [{"case_id": "b1"}],
            },
        ]
    }
    rows = [
        {"case_id": "a1", "passed": True, "failure_type": ""},
        {"case_id": "a2", "passed": False, "failure_type": "verifier_failure"},
        {"case_id": "b1", "passed": True, "failure_type": ""},
    ]

    result = aggregate_results(population, rows)

    assert result["avg_soft_score"] == pytest.approx(0.75)
    assert result["avg_hard_score"] == pytest.approx(0.5)
    assert result["testcase_accuracy"] == pytest.approx(2 / 3)
    assert result["fully_correct_tasks"] == 1


def test_missing_and_recalc_failures_remain_in_denominator() -> None:
    population = {
        "tasks": [
            {
                "task_id": "a",
                "instruction_type": "Cell-Level",
                "cases": [{"case_id": "a1"}, {"case_id": "a2"}],
            }
        ]
    }
    rows = [
        {"case_id": "a1", "passed": False, "failure_type": "missing_output"},
        {
            "case_id": "a2",
            "passed": False,
            "failure_type": "libreoffice_recalc_error",
        },
    ]

    result = aggregate_results(population, rows)

    assert result["testcase_count"] == 2
    assert result["avg_soft_score"] == 0
    assert result["avg_hard_score"] == 0
    assert result["missing_output_cases"] == 1
    assert result["recalc_error_cases"] == 1


def test_evaluation_snapshot_is_bound_to_expected_bytes(tmp_path: Path) -> None:
    source = tmp_path / "source.xlsx"
    source.write_bytes(b"workbook")
    destination = tmp_path / "snapshot/case/output.xlsx"

    result = _snapshot_workbook(
        source,
        destination,
        expected_sha256=hashlib.sha256(b"workbook").hexdigest(),
        expected_size=len(b"workbook"),
    )

    assert result.read_bytes() == b"workbook"
    with pytest.raises(ValueError, match="changed"):
        _snapshot_workbook(
            source,
            tmp_path / "snapshot/case/changed.xlsx",
            expected_sha256="0" * 64,
            expected_size=len(b"workbook"),
        )


def test_formal_evaluator_requires_sixteen_workers(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be 16"):
        evaluate_run(
            prepared_data_path=tmp_path,
            population_manifest_path=tmp_path / "population.json",
            input_manifest_path=tmp_path / "input.json",
            run_dir=tmp_path / "run",
            workers=1,
        )
