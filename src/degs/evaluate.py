from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence

from sb_adapter.evaluate import evaluate as spreadsheetbench_evaluate

from .dataset import DEVELOPMENT_END, DEVELOPMENT_START


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def evaluate_run(
    *, data_path: Path, run_dir: Path, base_url: str, model: str,
) -> dict[str, Any]:
    data_path = data_path.expanduser().resolve()
    run_dir = run_dir.expanduser().resolve()
    manifest = run_dir / "outputs/run_manifest.json"
    result = spreadsheetbench_evaluate(
        data_path=str(data_path),
        output_dir=str(run_dir / "outputs"),
        start_idx=DEVELOPMENT_START,
        end_idx=DEVELOPMENT_END,
        verbose=False,
        recalc_dir=str(run_dir / "recalculated_outputs"),
        evaluator_backend="local",
        run_manifest=str(manifest),
        run_completion=str(run_dir / "results.json"),
        expected_base_url=base_url,
        expected_model=model,
    )
    raw_summary = result.get("summary")
    if (
        not isinstance(raw_summary, dict)
        or raw_summary.get("total_instances")
        != DEVELOPMENT_END - DEVELOPMENT_START
    ):
        raise ValueError("evaluator denominator differs from the fixed 200")
    _write_json(run_dir / "eval_details.json", result)

    summary_fields = (
        "total_instances",
        "fully_correct_instances",
        "instance_accuracy",
        "total_test_cases",
        "passed_test_cases",
        "test_case_accuracy",
        "raw_passed_test_cases",
        "raw_test_case_accuracy",
        "raw_evaluation_mode",
        "avg_soft_score",
        "avg_hard_score",
        "by_instruction_type",
        "evaluation_mode",
        "workbook_comparator",
        "libreoffice",
        "run_manifest",
        "run_completion",
    )
    summary = {
        "format": "degs_evaluation_summary_v1",
        "fixed_denominator": DEVELOPMENT_END - DEVELOPMENT_START,
        **{field: raw_summary[field] for field in summary_fields},
    }
    summary_output = run_dir / "eval_summary.json"
    _write_json(summary_output, summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate one fixed DEGS development[200,400) run."
    )
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--base-url",
        default=os.getenv("DEGS_CHAT_BASE_URL"),
        help="Generation service URL recorded by the matching run manifest.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("DEGS_MODEL", "Qwen3.5-9B-AWQ"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.base_url:
        raise SystemExit("evaluate requires --base-url or DEGS_CHAT_BASE_URL")
    result = evaluate_run(
        data_path=args.data_path,
        run_dir=args.run_dir,
        base_url=args.base_url,
        model=args.model,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
