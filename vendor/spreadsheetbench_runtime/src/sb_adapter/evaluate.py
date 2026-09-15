#!/usr/bin/env python3
"""
SpreadsheetBench workbook evaluation with mandatory LibreOffice recalculation
and the packaged, source-hashed comparator.
"""

import argparse
import hashlib
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .spreadsheetbench_support import (
    compare_workbooks as local_compare_workbooks,
    find_output_dir,
    find_spreadsheet_dir,
    load_dataset,
)


ROOT = Path(__file__).resolve().parents[2]


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _source_identity(function) -> dict[str, str | None]:
    source_path = inspect.getsourcefile(function)
    resolved = Path(source_path).resolve() if source_path else None
    module = sys.modules.get(function.__module__)
    return {
        "module": function.__module__,
        "qualname": function.__qualname__,
        "module_version": (
            str(getattr(module, "__version__"))
            if module is not None and getattr(module, "__version__", None)
            else None
        ),
        "source_path": str(resolved) if resolved else None,
        "source_sha256": (
            hashlib.sha256(resolved.read_bytes()).hexdigest()
            if resolved and resolved.is_file()
            else None
        ),
    }


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    files = (
        item
        for item in root.rglob("*")
        if item.is_file()
        and "__pycache__" not in item.parts
        and item.suffix != ".pyc"
    )
    for path in sorted(files):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _validate_run_manifest(
    path: str | Path,
    *,
    data_path: str | Path,
    start_idx: int,
    end_idx: int,
) -> dict[str, Any]:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("run manifest must be a JSON object")
    dataset_file = Path(data_path) / "dataset.json"
    dataset = json.loads(dataset_file.read_text(encoding="utf-8"))
    expected_ids = [str(item["id"]) for item in dataset[start_idx:end_idx]]
    dataset_sha = hashlib.sha256(dataset_file.read_bytes()).hexdigest()
    protocol = {
        key: value
        for key, value in payload.items()
        if key not in {"protocol_sha256", "created_at"}
    }
    protocol_text = json.dumps(
        protocol, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    protocol_sha = hashlib.sha256(protocol_text.encode("utf-8")).hexdigest()
    checks = {
        "format": payload.get("format") == "spreadsheetbench_adapter_run_v1",
        "dataset_sha256": payload.get("dataset_sha256") == dataset_sha,
        "dataset_tree_sha256": payload.get("dataset_tree_sha256")
        == _tree_sha256(Path(data_path)),
        "start_idx": payload.get("start_idx") == start_idx,
        "end_idx": payload.get("end_idx") == end_idx,
        "instance_ids": payload.get("instance_ids") == expected_ids,
        "model": payload.get("model") == "Qwen3.5-9B-AWQ",
        "base_url_recorded": isinstance(payload.get("base_url"), str)
        and bool(payload["base_url"]),
        "temperature": payload.get("temperature") == 0.0,
        "max_tokens_recorded": payload.get("max_tokens") is None
        or isinstance(payload.get("max_tokens"), int),
        "thinking": payload.get("thinking") == "false",
        "max_turns": payload.get("max_turns") == 30,
        "bash_timeout": payload.get("bash_timeout") == 120,
        "bash_sandbox": payload.get("bash_sandbox") == "required",
        "workers_recorded": isinstance(payload.get("workers"), int)
        and payload["workers"] > 0,
        "llm_timeout": payload.get("llm_timeout") == 600.0,
        "retry_waits": payload.get("retry_waits") == [5, 10, 30],
        "response_cache_disabled": payload.get("response_cache_enabled") is False,
        "protocol_sha256": payload.get("protocol_sha256") == protocol_sha,
    }
    required = {
        "format",
        "dataset_sha256",
        "dataset_tree_sha256",
        "start_idx",
        "end_idx",
        "instance_ids",
        "model",
    }
    failed = [name for name in required if not checks[name]]
    if failed:
        raise ValueError(f"run manifest does not match evaluation protocol: {failed}")
    return {
        "file_name": manifest_path.name,
        "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "protocol_sha256": protocol_sha,
        "checks": checks,
        "advisory_mismatches": [
            name for name, passed in checks.items() if not passed and name not in required
        ],
    }


def _resolve_comparator(
    requested_backend: str,
):
    if requested_backend != "local":
        raise ValueError("this package supports only its source-hashed local comparator")
    comparator = local_compare_workbooks
    identity = {
        "requested_backend": requested_backend,
        "resolved_backend": "local",
        **_source_identity(comparator),
    }
    return comparator, identity


def _compare_workbooks(
    comparator,
    resolved_backend: str,
    gt_path,
    output_path,
    instruction_type,
    answer_position,
):
    return comparator(gt_path, output_path, answer_position)


def _soffice_executable() -> str:
    soffice = shutil.which("soffice")
    if not soffice:
        raise RuntimeError("LibreOffice executable `soffice` not found on PATH")
    return soffice


def _safe_instance_dir_name(instance_id: str) -> str:
    raw = str(instance_id)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._-")
    if not safe:
        safe = "instance"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10]
    return f"{safe[:80]}_{digest}"


def _ensure_within(parent: Path, child: Path) -> None:
    try:
        child.relative_to(parent)
    except ValueError as exc:
        raise RuntimeError(f"path escapes recalc audit root: {child}") from exc


def _preflight_libreoffice(
    timeout_seconds: int = 30,
) -> tuple[str, str]:
    soffice = _soffice_executable()
    try:
        proc = subprocess.run(
            [soffice, "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"LibreOffice preflight timed out after {timeout_seconds}s") from exc
    if proc.returncode != 0:
        raise RuntimeError(f"LibreOffice preflight failed with exit {proc.returncode}: {proc.stdout.strip()}")
    return soffice, proc.stdout.strip()


def evaluation_runtime_identity(
    evaluator_backend: str,
    *,
    libreoffice_timeout_seconds: int = 30,
    comparator_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve the exact comparator and LibreOffice identities used by eval."""

    if comparator_identity is None:
        _, comparator_identity = _resolve_comparator(evaluator_backend)
    soffice, soffice_version = _preflight_libreoffice(
        libreoffice_timeout_seconds
    )
    return {
        "workbook_comparator": comparator_identity,
        "libreoffice": {
            "executable": str(Path(soffice).resolve()),
            "version": soffice_version,
        },
    }


def _recalculate_workbook(
    input_path: str,
    recalc_dir: str,
    instance_id: str,
    *,
    soffice: str | None = None,
    timeout_seconds: int = 180,
) -> str:
    """Recalculate a workbook via LibreOffice and return the copied output path."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(input_path)

    soffice = soffice or _soffice_executable()
    input_file = Path(input_path).resolve()
    recalc_root = Path(recalc_dir)
    recalc_root.mkdir(parents=True, exist_ok=True)
    recalc_root = recalc_root.resolve()
    try:
        input_file.relative_to(recalc_root)
    except ValueError:
        pass
    else:
        raise RuntimeError(
            f"recalc_dir must not contain source workbook; recalc_dir={recalc_root}, source={input_file}"
        )

    instance_root = recalc_root / _safe_instance_dir_name(instance_id)
    instance_root.mkdir(parents=True, exist_ok=True)
    instance_root = instance_root.resolve()
    _ensure_within(recalc_root, instance_root)

    out_dir = Path(tempfile.mkdtemp(prefix="recalc_", dir=str(instance_root))).resolve()
    _ensure_within(recalc_root, out_dir)
    output_file = out_dir / input_file.name

    with tempfile.TemporaryDirectory(prefix="sb_adapter_recalc_") as tmp:
        runtime_root = Path(tmp)
        profile = runtime_root / "profile"
        home = runtime_root / "home"
        config_home = runtime_root / "config"
        cache_home = runtime_root / "cache"
        runtime_dir = runtime_root / "runtime"
        for directory in (home, config_home, cache_home, runtime_dir):
            directory.mkdir()
        bwrap = shutil.which("bwrap")
        if not bwrap:
            raise RuntimeError("bubblewrap (`bwrap`) is required for LibreOffice evaluation")
        sandbox_input = f"/input/{input_file.name}"
        cmd = [
            bwrap,
            "--die-with-parent",
            "--new-session",
            "--unshare-net",
            "--unshare-pid",
            "--clearenv",
        ]
        for root in ("/usr", "/bin", "/lib", "/lib64", "/etc", "/var/cache/fontconfig"):
            if Path(root).exists():
                cmd.extend(("--ro-bind", root, root))
        cmd.extend([
            "--dir",
            "/input",
            "--ro-bind",
            str(input_file),
            sandbox_input,
            "--bind",
            str(out_dir),
            "/output",
            "--bind",
            str(runtime_root),
            "/runtime",
            "--tmpfs",
            "/tmp",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--setenv",
            "HOME",
            "/runtime/home",
            "--setenv",
            "XDG_CONFIG_HOME",
            "/runtime/config",
            "--setenv",
            "XDG_CACHE_HOME",
            "/runtime/cache",
            "--setenv",
            "XDG_RUNTIME_DIR",
            "/runtime/runtime",
            "--setenv",
            "PATH",
            "/usr/bin:/bin",
            soffice,
            "-env:UserInstallation=file:///runtime/profile",
            "--headless",
            "--invisible",
            "--norestore",
            "--nodefault",
            "--nolockcheck",
            "--nofirststartwizard",
            "--convert-to",
            "xlsx",
            "--outdir",
            "/output",
            sandbox_input,
        ])
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"LibreOffice recalc timed out after {timeout_seconds}s for {input_file}") from exc
    if proc.returncode != 0:
        raise RuntimeError(f"LibreOffice recalc failed with exit {proc.returncode}: {proc.stdout.strip()}")
    if not output_file.exists():
        raise RuntimeError(f"LibreOffice did not produce recalculated workbook: {output_file}; output={proc.stdout.strip()}")
    return str(output_file)


def evaluate(
    data_path,
    output_dir,
    start_idx=0,
    end_idx=None,
    verbose=False,
    recalc_dir=None,
    evaluator_backend="local",
    run_manifest=None,
):
    """
    Evaluate outputs with an explicit SpreadsheetBench comparator backend.

    Returns:
        dict with evaluation results
    """
    full_dataset = load_dataset(data_path)

    if end_idx is None:
        end_idx = len(full_dataset)
    if not run_manifest:
        raise ValueError("evaluation requires the matching --run-manifest")
    manifest_identity = _validate_run_manifest(
        run_manifest,
        data_path=data_path,
        start_idx=start_idx,
        end_idx=end_idx,
    )
    dataset = full_dataset[start_idx:end_idx]

    if recalc_dir is None:
        recalc_dir = os.path.join(output_dir, "eval_artifacts", "libreoffice_recalculated_outputs")
    comparator, comparator_identity = _resolve_comparator(evaluator_backend)
    resolved_backend = str(comparator_identity["resolved_backend"])
    runtime_identity = evaluation_runtime_identity(
        evaluator_backend,
        comparator_identity=comparator_identity,
    )
    libreoffice_identity = dict(runtime_identity["libreoffice"])
    soffice = str(libreoffice_identity["executable"])
    soffice_version = str(libreoffice_identity["version"])
    print(
        f"Evaluating {len(dataset)} instances using {resolved_backend} "
        "SpreadsheetBench comparison with LibreOffice recalc..."
    )
    print(f"LibreOffice executable: {soffice}")
    print(f"LibreOffice version: {soffice_version}")

    results = []
    total_test_cases = 0
    passed_test_cases = 0
    fully_correct = 0
    raw_passed_test_cases = 0

    # Track by instruction type (like official eval)
    type_results = defaultdict(lambda: {"soft": [], "hard": []})

    for instance in tqdm(dataset):
        instance_id = str(instance["id"])
        spreadsheet_path = str(instance.get("spreadsheet_path", instance_id))
        instruction_type = instance.get("instruction_type", "")
        answer_position = instance.get("answer_position", "")

        if not answer_position:
            message = "Missing answer_position in dataset metadata"
            results.append({
                "id": instance_id,
                "instruction_type": instruction_type,
                "success": False,
                "error": message,
                "test_cases": [],
                "passed_count": 0,
                "total_count": 0,
                "soft_score": 0.0,
                "hard_score": 0,
            })
            type_results[instruction_type]["soft"].append(0.0)
            type_results[instruction_type]["hard"].append(0)
            continue

        # Find spreadsheet directory (contains ground truth)
        spreadsheet_dir = find_spreadsheet_dir(data_path, instance)
        if spreadsheet_dir is None:
            results.append({
                "id": instance_id,
                "instruction_type": instruction_type,
                "success": False,
                "error": "Spreadsheet directory not found",
                "test_cases": [],
                "passed_count": 0,
                "total_count": 0,
                "soft_score": 0.0,
                "hard_score": 0,
            })
            type_results[instruction_type]["soft"].append(0.0)
            type_results[instruction_type]["hard"].append(0)
            continue

        # Find output directory for this instance
        output_instance_dir = find_output_dir(output_dir, instance)

        # Find all test cases (ground truth files)
        # Standard format: *_answer.xlsx, Verified format: *_golden.xlsx
        try:
            all_files = os.listdir(spreadsheet_dir)
        except FileNotFoundError:
            results.append({
                "id": instance_id,
                "instruction_type": instruction_type,
                "success": False,
                "error": f"Cannot list spreadsheet directory: {spreadsheet_dir}",
                "test_cases": [],
                "passed_count": 0,
                "total_count": 0,
                "soft_score": 0.0,
                "hard_score": 0,
            })
            type_results[instruction_type]["soft"].append(0.0)
            type_results[instruction_type]["hard"].append(0)
            continue

        gt_files = sorted([f for f in all_files if f.endswith("_answer.xlsx")])

        if not gt_files:
            # Try verified dataset format
            gt_files = sorted([f for f in all_files if f.endswith("_golden.xlsx")])

        if not gt_files:
            # Try exact match for simple naming: golden.xlsx
            if "golden.xlsx" in all_files:
                gt_files = ["golden.xlsx"]

        if not gt_files:
            results.append({
                "id": instance_id,
                "instruction_type": instruction_type,
                "success": False,
                "error": "No ground truth files found (expected *_answer.xlsx or *_golden.xlsx)",
                "test_cases": [],
                "passed_count": 0,
                "total_count": 0,
                "soft_score": 0.0,
                "hard_score": 0,
            })
            type_results[instruction_type]["soft"].append(0.0)
            type_results[instruction_type]["hard"].append(0)
            continue

        test_case_results = []

        for gt_file in gt_files:
            # Derive output filename from ground truth filename
            if gt_file.endswith("_answer.xlsx"):
                output_file = gt_file.replace("_answer.xlsx", "_output.xlsx")
            elif gt_file == "golden.xlsx":
                # Simple naming: golden.xlsx -> initial_output.xlsx
                output_file = "initial_output.xlsx"
            else:  # _golden.xlsx
                output_file = gt_file.replace("_golden.xlsx", "_output.xlsx")

            gt_path = os.path.join(spreadsheet_dir, gt_file)
            output_path = os.path.join(output_instance_dir, output_file)

            total_test_cases += 1

            raw_result = False
            raw_msg = ""
            try:
                raw_result, raw_msg = _compare_workbooks(
                    comparator,
                    resolved_backend,
                    gt_path,
                    output_path,
                    instruction_type,
                    answer_position,
                )
            except Exception as e:
                raw_msg = str(e)

            if raw_result:
                raw_passed_test_cases += 1

            recalculated_output_path = ""
            recalc_error = ""
            try:
                recalculated_output_path = _recalculate_workbook(output_path, recalc_dir, instance_id, soffice=soffice)
                result, msg = _compare_workbooks(
                    comparator,
                    resolved_backend,
                    gt_path,
                    recalculated_output_path,
                    instruction_type,
                    answer_position,
                )
            except FileNotFoundError as e:
                result = False
                recalc_error = f"Output file not found for LibreOffice recalc: {e}"
                msg = raw_msg or recalc_error
            except Exception as e:
                result = False
                recalc_error = f"LibreOffice recalc/eval failed: {e}"
                msg = recalc_error

            test_case_results.append({
                "gt_file": gt_file,
                "output_file": output_file,
                "output_path": output_path,
                "recalculated_output_path": recalculated_output_path,
                "evaluation_mode": "libreoffice_recalc",
                "raw_evaluation_mode": "audit_only_no_recalc",
                "raw_passed": raw_result,
                "raw_message": raw_msg,
                "recalc_error": recalc_error,
                "passed": result,
                "message": msg,
            })

            if result:
                passed_test_cases += 1
            elif verbose:
                print(f"  {instance_id}/{output_file}: {msg}")

        # Calculate metrics for this instance (matching official eval)
        passed_count = sum(1 for tc in test_case_results if tc["passed"])
        total_count = len(test_case_results)
        soft_score = passed_count / total_count if total_count > 0 else 0
        hard_score = 1 if passed_count == total_count else 0

        if hard_score == 1:
            fully_correct += 1

        # Track by instruction type
        type_results[instruction_type]["soft"].append(soft_score)
        type_results[instruction_type]["hard"].append(hard_score)

        results.append({
            "id": instance_id,
            "instruction_type": instruction_type,
            "success": hard_score == 1,
            "test_cases": test_case_results,
            "passed_count": passed_count,
            "total_count": total_count,
            "soft_score": soft_score,
            "hard_score": hard_score,
        })

    # Calculate overall metrics
    total_instances = len(results)

    soft_scores = [r.get("soft_score", 0) for r in results if "soft_score" in r]
    hard_scores = [r.get("hard_score", 0) for r in results if "hard_score" in r]

    avg_soft_score = sum(soft_scores) / len(soft_scores) if soft_scores else 0
    avg_hard_score = sum(hard_scores) / len(hard_scores) if hard_scores else 0

    # Calculate per-type metrics
    type_metrics = {}
    for inst_type, scores in type_results.items():
        type_metrics[inst_type] = {
            "count": len(scores["soft"]),
            "avg_soft_score": sum(scores["soft"]) / len(scores["soft"]) if scores["soft"] else 0,
            "avg_hard_score": sum(scores["hard"]) / len(scores["hard"]) if scores["hard"] else 0,
        }

    summary = {
        "total_instances": total_instances,
        "fully_correct_instances": fully_correct,
        "instance_accuracy": fully_correct / total_instances if total_instances > 0 else 0,
        "total_test_cases": total_test_cases,
        "passed_test_cases": passed_test_cases,
        "test_case_accuracy": passed_test_cases / total_test_cases if total_test_cases > 0 else 0,
        "raw_passed_test_cases": raw_passed_test_cases,
        "raw_test_case_accuracy": raw_passed_test_cases / total_test_cases if total_test_cases > 0 else 0,
        "raw_evaluation_mode": "audit_only_no_recalc",
        "avg_soft_score": avg_soft_score,
        "avg_hard_score": avg_hard_score,
        "by_instruction_type": type_metrics,
        "evaluation_mode": "libreoffice_recalc",
        "recalculated_output_dir": recalc_dir,
        "workbook_comparator": comparator_identity,
        "libreoffice": {
            "executable": str(Path(soffice).resolve()),
            "version": soffice_version,
        },
        "run_manifest": manifest_identity,
    }

    return {
        "format": "spreadsheetbench_adapter_evaluation_v1",
        "summary": summary,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SpreadsheetBench outputs using official evaluation logic"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to SpreadsheetBench data directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory containing agent outputs",
    )
    parser.add_argument(
        "--results_file",
        type=str,
        default=None,
        help="Path to save evaluation results JSON (default: output_dir/eval_official_results.json)",
    )
    parser.add_argument(
        "--recalc_dir",
        type=str,
        default=None,
        help="Directory for LibreOffice-recalculated workbook audit copies",
    )
    parser.add_argument(
        "--run-manifest",
        required=True,
        help="Run manifest that must match the evaluated dataset slice",
    )
    parser.add_argument(
        "--evaluator-backend",
        choices=["local"],
        default="local",
        help="Use the packaged, source-hashed SpreadsheetBench comparator.",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Start index for evaluation",
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=None,
        help="End index for evaluation (exclusive)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed error messages",
    )
    args = parser.parse_args()

    # Run evaluation
    eval_result = evaluate(
        data_path=args.data_path,
        output_dir=args.output_dir,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        verbose=args.verbose,
        recalc_dir=args.recalc_dir,
        evaluator_backend=args.evaluator_backend,
        run_manifest=args.run_manifest,
    )

    # Print summary
    _print_summary(eval_result["summary"])

    # Save results
    results_file = args.results_file or os.path.join(args.output_dir, "eval_official_results.json")
    _write_json_atomic(Path(results_file), eval_result)
    print(f"Results saved to: {results_file}")


def _print_summary(summary: dict, label: str = "") -> None:
    """Print a formatted evaluation summary."""
    comparator = summary.get("workbook_comparator", {})
    resolved_backend = comparator.get("resolved_backend", "unknown")
    header = (
        f"EVALUATION RESULTS{' (' + label + ')' if label else ''} "
        f"(SpreadsheetBench comparator: {resolved_backend})"
    )
    print("\n" + "=" * 60)
    print(header)
    print("=" * 60)
    print(f"Total Instances:        {summary['total_instances']}")
    print(f"Fully Correct:          {summary['fully_correct_instances']}")
    print(f"Instance Accuracy:      {summary['instance_accuracy']*100:.1f}%")
    print(f"Total Test Cases:       {summary['total_test_cases']}")
    print(f"Passed Test Cases:      {summary['passed_test_cases']}")
    print(f"Test Case Accuracy:     {summary['test_case_accuracy']*100:.1f}%")
    print(f"Raw Audit Test Accuracy: {summary.get('raw_test_case_accuracy', 0)*100:.1f}% (audit-only, not official)")
    print(f"Avg Soft Score:         {summary['avg_soft_score']*100:.1f}%")
    print(f"Avg Hard Score:         {summary['avg_hard_score']*100:.1f}%")
    print(f"Evaluation Mode:        {summary.get('evaluation_mode', 'unknown')}")
    print(f"Comparator Backend:     {resolved_backend}")
    if summary.get("recalculated_output_dir"):
        print(f"Recalculated Outputs:   {summary['recalculated_output_dir']}")

    if summary["by_instruction_type"]:
        print("-" * 60)
        print("By Instruction Type:")
        for inst_type, metrics in sorted(summary["by_instruction_type"].items()):
            print(f"  {inst_type or '(unknown)'}:")
            print(f"    Count: {metrics['count']}")
            print(f"    Soft:  {metrics['avg_soft_score']*100:.1f}%")
            print(f"    Hard:  {metrics['avg_hard_score']*100:.1f}%")

    print("=" * 60)


if __name__ == "__main__":
    main()
