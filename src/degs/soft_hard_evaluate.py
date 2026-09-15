"""LibreOffice-recalculated Soft/Hard evaluation for the fixed population."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

from sb_adapter.evaluate import (
    _compare_workbooks,
    _recalculate_workbook,
    _resolve_comparator,
    evaluation_runtime_identity,
)

from .benchmark import _write_json_atomic
from .soft_hard_benchmark import (
    _write_bytes_atomic,
    verify_completed_run,
)
from .soft_hard_dataset import (
    TASK_COUNT,
    TESTCASE_COUNT,
    canonical_json_bytes,
    load_prepared_population,
    load_prepared_retrieval_population,
)
from .runtime_config import worker_count


FORMAT = "degs_spreadsheetbench_soft_hard_evaluation_v1"
LIBREOFFICE_WORKERS = worker_count("DEGS_LIBREOFFICE_WORKERS", 16)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_workbook(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str,
    expected_size: int,
) -> Path:
    if not source.is_file():
        raise ValueError(f"evaluation source workbook differs: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        payload = source.read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected_sha256 or len(payload) != expected_size:
            raise ValueError(f"evaluation source workbook changed: {source}")
        with os.fdopen(temporary_descriptor, "wb") as output_handle:
            output_handle.write(payload)
        os.replace(temporary, destination)
        return destination
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate_results(
    population: Mapping[str, Any],
    case_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_case = {str(row["case_id"]): row for row in case_results}
    if len(by_case) != len(case_results):
        raise ValueError("evaluation case identities are not unique")
    task_results: list[dict[str, Any]] = []
    by_type: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"soft": [], "hard": []}
    )
    by_stratum: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"soft": [], "hard": [], "passed": [], "cases": []}
    )
    expected_case_ids: list[str] = []
    for task in population["tasks"]:
        rows = []
        for case in task["cases"]:
            case_id = case["case_id"]
            expected_case_ids.append(case_id)
            row = by_case.get(case_id)
            if row is None:
                raise ValueError(f"evaluation result is absent for {case_id}")
            rows.append(row)
        passed_count = sum(bool(row["passed"]) for row in rows)
        total_count = len(rows)
        soft_score = passed_count / total_count
        hard_score = int(passed_count == total_count)
        instruction_type = str(task["instruction_type"])
        by_type[instruction_type]["soft"].append(soft_score)
        by_type[instruction_type]["hard"].append(float(hard_score))
        stratum = str(task.get("evaluation_stratum", "unspecified"))
        by_stratum[stratum]["soft"].append(soft_score)
        by_stratum[stratum]["hard"].append(float(hard_score))
        by_stratum[stratum]["passed"].append(float(passed_count))
        by_stratum[stratum]["cases"].append(float(total_count))
        task_results.append(
            {
                "task_id": task["task_id"],
                "instruction_type": instruction_type,
                "passed_count": passed_count,
                "total_count": total_count,
                "soft_score": soft_score,
                "hard_score": hard_score,
                "cases": rows,
            }
        )
    if set(expected_case_ids) != set(by_case):
        raise ValueError("evaluation result population differs")
    passed_testcases = sum(bool(row["passed"]) for row in case_results)
    task_count = len(task_results)
    testcase_count = len(case_results)
    return {
        "task_count": task_count,
        "testcase_count": testcase_count,
        "passed_testcases": passed_testcases,
        "testcase_accuracy": passed_testcases / testcase_count,
        "fully_correct_tasks": sum(row["hard_score"] for row in task_results),
        "avg_soft_score": sum(row["soft_score"] for row in task_results)
        / task_count,
        "avg_hard_score": sum(row["hard_score"] for row in task_results)
        / task_count,
        "missing_output_cases": sum(
            row.get("failure_type") == "missing_output" for row in case_results
        ),
        "recalc_error_cases": sum(
            row.get("failure_type") == "libreoffice_recalc_error"
            for row in case_results
        ),
        "comparator_error_cases": sum(
            row.get("failure_type") == "comparator_error"
            for row in case_results
        ),
        "by_instruction_type": {
            name: {
                "task_count": len(values["soft"]),
                "avg_soft_score": sum(values["soft"]) / len(values["soft"]),
                "avg_hard_score": sum(values["hard"]) / len(values["hard"]),
            }
            for name, values in sorted(by_type.items())
        },
        "by_evaluation_stratum": {
            name: {
                "task_count": len(values["soft"]),
                "testcase_count": int(sum(values["cases"])),
                "passed_testcases": int(sum(values["passed"])),
                "testcase_accuracy": sum(values["passed"])
                / sum(values["cases"]),
                "avg_soft_score": sum(values["soft"]) / len(values["soft"]),
                "avg_hard_score": sum(values["hard"]) / len(values["hard"]),
            }
            for name, values in sorted(by_stratum.items())
        },
        "tasks": task_results,
    }


def evaluate_run(
    *,
    prepared_data_path: Path,
    population_manifest_path: Path,
    input_manifest_path: Path,
    run_dir: Path,
    workers: int = LIBREOFFICE_WORKERS,
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("LibreOffice workers must be positive")
    evaluation_adapter_sha256 = _sha256(Path(__file__))
    prepared_data_path = prepared_data_path.expanduser().resolve()
    population_manifest_path = population_manifest_path.expanduser().resolve()
    input_manifest_path = input_manifest_path.expanduser().resolve()
    run_dir = run_dir.expanduser().resolve()
    population = load_prepared_population(
        data_path=prepared_data_path,
        manifest_path=population_manifest_path,
    )
    input_population = load_prepared_retrieval_population(
        data_path=prepared_data_path,
        manifest_path=input_manifest_path,
    )
    run_manifest, _completion, generation_rows, ledger_bytes = (
        verify_completed_run(run_dir=run_dir, population=input_population)
    )
    if (
        run_manifest.get("input_manifest_sha256")
        != input_population["self_sha256"]
        or population.get("retrieval_manifest_sha256")
        != input_population["self_sha256"]
        or run_manifest.get("prepared_input_tree_sha256")
        != input_population["prepared_input_tree_sha256"]
        or run_manifest.get("case_ids")
        != [
            case["case_id"]
            for task in population["tasks"]
            for case in task["cases"]
        ]
    ):
        raise ValueError("evaluation population differs from completed run")
    generation_by_case = {row["case_id"]: row for row in generation_rows}
    if len(generation_by_case) != TESTCASE_COUNT:
        raise ValueError("generation result case identities differ")
    comparator, comparator_identity = _resolve_comparator("local")
    runtime = evaluation_runtime_identity(
        "local", comparator_identity=comparator_identity
    )
    soffice = runtime["libreoffice"]["executable"]
    recalc_dir = run_dir / "libreoffice_recalculated_outputs"
    recalc_dir.mkdir(parents=True, exist_ok=True)
    if recalc_dir.resolve().parent != run_dir.resolve():
        raise ValueError("LibreOffice recalculation directory escapes run root")
    snapshot_dir = run_dir / "evaluation_workbook_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    if snapshot_dir.resolve().parent != run_dir.resolve():
        raise ValueError("evaluation workbook snapshot escapes run root")
    case_plan = [
        (task, case)
        for task in population["tasks"]
        for case in task["cases"]
    ]

    def evaluate_case(
        task: Mapping[str, Any], case: Mapping[str, Any]
    ) -> dict[str, Any]:
        generated = generation_by_case[case["case_id"]]
        spreadsheet_dir = prepared_data_path / task["spreadsheet_path"]
        output_path = run_dir / "outputs" / task["spreadsheet_path"] / case["output_file"]
        answer_path = spreadsheet_dir / case["answer_file"]
        base = {
            "case_id": case["case_id"],
            "task_id": task["task_id"],
            "input_file": case["input_file"],
            "answer_file": case["answer_file"],
            "output_file": case["output_file"],
            "agent_success": generated["agent_success"],
            "agent_completed": generated["agent_completed"],
            "output_preserved": generated["output_preserved"],
            "turns": generated["turns"],
            "generated_output_sha256": generated["output_sha256"],
            "generated_output_size": generated["output_size"],
        }
        if output_path.is_symlink() or not output_path.is_file():
            return {
                **base,
                "passed": False,
                "failure_type": "missing_output",
                "message": "Agent output workbook is absent",
                "recalculated_output_path": "",
                "recalculated_output_sha256": "",
                "recalculated_output_size": 0,
            }
        case_snapshot_dir = snapshot_dir / hashlib.sha256(
            case["case_id"].encode("utf-8")
        ).hexdigest()
        output_snapshot = _snapshot_workbook(
            output_path,
            case_snapshot_dir / "output.xlsx",
            expected_sha256=generated["output_sha256"],
            expected_size=generated["output_size"],
        )
        answer_snapshot = _snapshot_workbook(
            answer_path,
            case_snapshot_dir / "answer.xlsx",
            expected_sha256=case["answer_sha256"],
            expected_size=case["answer_size"],
        )
        try:
            recalculated = _recalculate_workbook(
                str(output_snapshot),
                str(recalc_dir),
                case["case_id"],
                soffice=soffice,
            )
        except Exception as exc:
            return {
                **base,
                "passed": False,
                "failure_type": "libreoffice_recalc_error",
                "message": f"{type(exc).__name__}:{exc}",
                "recalculated_output_path": "",
                "recalculated_output_sha256": "",
                "recalculated_output_size": 0,
            }
        try:
            passed, message = _compare_workbooks(
                comparator,
                "local",
                str(answer_snapshot),
                recalculated,
                task["instruction_type"],
                task["answer_position"],
            )
        except Exception as exc:
            recalculated_path = Path(recalculated)
            return {
                **base,
                "passed": False,
                "failure_type": "comparator_error",
                "message": f"{type(exc).__name__}:{exc}",
                "recalculated_output_path": recalculated,
                "recalculated_output_sha256": _sha256(recalculated_path),
                "recalculated_output_size": recalculated_path.stat().st_size,
            }
        recalculated_path = Path(recalculated)
        return {
            **base,
            "passed": bool(passed),
            "failure_type": "" if passed else "verifier_failure",
            "message": str(message),
            "recalculated_output_path": recalculated,
            "recalculated_output_sha256": _sha256(recalculated_path),
            "recalculated_output_size": recalculated_path.stat().st_size,
        }

    rows_by_case: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(evaluate_case, task, case): case["case_id"]
            for task, case in case_plan
        }
        for ordinal, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            rows_by_case[row["case_id"]] = row
            if ordinal % 100 == 0 or ordinal == TESTCASE_COUNT:
                print(
                    f"LibreOffice evaluated {ordinal}/{TESTCASE_COUNT} cases",
                    flush=True,
                )
    ordered = [rows_by_case[case["case_id"]] for _task, case in case_plan]
    aggregated = aggregate_results(population, ordered)
    if (
        aggregated["task_count"] != TASK_COUNT
        or aggregated["testcase_count"] != TESTCASE_COUNT
    ):
        raise ValueError("Soft/Hard evaluation denominator differs")
    result = {
        "format": FORMAT,
        "evaluation_mode": "libreoffice_recalculation_primary",
        "libreoffice_workers": workers,
        "population_manifest_sha256": population["self_sha256"],
        "input_manifest_sha256": input_population["self_sha256"],
        "method_version": run_manifest["method_version"],
        "model": run_manifest["model"],
        "bundle_self_sha256": run_manifest["bundle_self_sha256"],
        "run_protocol_sha256": run_manifest["protocol_sha256"],
        "generation_results_sha256": hashlib.sha256(ledger_bytes).hexdigest(),
        "evaluation_adapter_sha256": evaluation_adapter_sha256,
        "workbook_comparator": comparator_identity,
        "libreoffice": runtime["libreoffice"],
        "summary": {
            key: value for key, value in aggregated.items() if key != "tasks"
        },
        "tasks": aggregated["tasks"],
    }
    result["self_sha256"] = hashlib.sha256(
        canonical_json_bytes(result)
    ).hexdigest()
    _write_json_atomic(run_dir / "eval_details.json", result)
    summary = {
        "format": FORMAT,
        "evaluation_mode": result["evaluation_mode"],
        "libreoffice_workers": workers,
        "population_manifest_sha256": population["self_sha256"],
        "input_manifest_sha256": input_population["self_sha256"],
        "method_version": run_manifest["method_version"],
        "model": run_manifest["model"],
        "bundle_self_sha256": run_manifest["bundle_self_sha256"],
        "run_protocol_sha256": run_manifest["protocol_sha256"],
        "generation_results_sha256": result["generation_results_sha256"],
        "evaluation_adapter_sha256": result["evaluation_adapter_sha256"],
        "workbook_comparator": comparator_identity,
        "libreoffice": runtime["libreoffice"],
        **result["summary"],
    }
    summary["self_sha256"] = hashlib.sha256(
        canonical_json_bytes(summary)
    ).hexdigest()
    _write_json_atomic(run_dir / "eval_summary.json", summary)
    report = "\n".join(
        (
            f"# DEGS {summary['method_version']} SpreadsheetBench Soft/Hard",
            "",
            f"- Tasks: {summary['task_count']}",
            f"- Testcases: {summary['testcase_count']}",
            f"- Passed testcases: {summary['passed_testcases']}",
            f"- Soft: {summary['avg_soft_score'] * 100:.2f}%",
            f"- Hard: {summary['avg_hard_score'] * 100:.2f}%",
            f"- Testcase micro accuracy: {summary['testcase_accuracy'] * 100:.2f}%",
            f"- Missing outputs: {summary['missing_output_cases']}",
            f"- LibreOffice errors: {summary['recalc_error_cases']}",
            f"- Comparator errors: {summary['comparator_error_cases']}",
            "- Evaluation mode: LibreOffice recalculation primary",
            "- Claim boundary: descriptive case-excluded/transductive full-population score; not untouched held-out generalization",
            "- Stratified metrics: see by_evaluation_stratum in eval_summary.json",
            "",
        )
    )
    _write_bytes_atomic(run_dir / "soft_hard_report.md", report.encode("utf-8"))
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the completed Soft/Hard run after LibreOffice recalc."
    )
    parser.add_argument("--prepared-data-path", type=Path, required=True)
    parser.add_argument("--population-manifest-path", type=Path, required=True)
    parser.add_argument("--input-manifest-path", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=LIBREOFFICE_WORKERS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = evaluate_run(
        prepared_data_path=args.prepared_data_path,
        population_manifest_path=args.population_manifest_path,
        input_manifest_path=args.input_manifest_path,
        run_dir=args.run_dir,
        workers=args.workers,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


__all__ = ["FORMAT", "LIBREOFFICE_WORKERS", "aggregate_results", "evaluate_run"]


if __name__ == "__main__":
    raise SystemExit(main())
