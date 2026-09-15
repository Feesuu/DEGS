#!/usr/bin/env python3
"""
SpreadsheetBench workbook evaluation with mandatory LibreOffice recalculation
and the packaged, source-hashed comparator.
"""

import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm
import spreadsheet_agent.system_prompts as runtime_prompts

from degs import __version__
from degs.dataset import (
    DEVELOPMENT_END,
    DEVELOPMENT_START,
    EXPECTED_DEVELOPMENT_QUERY_PROJECTION_SHA256,
    load_development_queries,
)
from degs.eir_bundle import EIR_BUNDLE_FORMAT, verify_contextual_bundle
from degs.provider import EIRGuidanceProvider

from .spreadsheetbench_support import (
    compare_workbooks as local_compare_workbooks,
    find_output_dir,
    find_spreadsheet_dir,
    load_dataset,
)


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
    files = []
    for item in root.rglob("*"):
        if item.is_symlink():
            raise ValueError(f"dataset tree must not contain symlinks: {item}")
        if item.is_file() and "__pycache__" not in item.parts and item.suffix != ".pyc":
            files.append(item)
    for path in sorted(files):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


_RUN_MANIFEST_FIELDS = {
    "base_url",
    "bash_sandbox",
    "bash_timeout",
    "bundle_dir",
    "bundle_self_sha256",
    "snapshot_manifest_path",
    "state_db_path",
    "claim_scope",
    "created_at",
    "dataset_name",
    "dataset_sha256",
    "dataset_tree_sha256",
    "dependency_versions",
    "end_idx",
    "experience_provider",
    "fixed_denominator",
    "format",
    "instance_ids",
    "llm_timeout",
    "max_tokens",
    "completion_recovery_attempt_limit",
    "max_consecutive_format_errors",
    "truncate_observations",
    "max_turns",
    "method",
    "method_version",
    "model",
    "protocol_sha256",
    "python_version",
    "response_cache_enabled",
    "runtime_event_log",
    "retry_waits",
    "runtime_timeout_retries",
    "stagnation_repeat_limit",
    "stagnation_recovery_attempt_limit",
    "start_idx",
    "system_prompt",
    "temperature",
    "thinking",
    "workers",
}

_RUN_COMPLETION_FIELDS = {
    "format",
    "run_manifest",
    "protocol_sha256",
    "started_at",
    "ended_at",
    "total_instances",
    "completed_instances",
    "agent_completed_instances",
    "results",
}


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _read_json_snapshot(path: str | Path, *, label: str) -> tuple[Any, bytes]:
    content = Path(path).read_bytes()
    try:
        payload = json.loads(
            content,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} is not strict JSON") from exc
    return payload, content


def _validate_run_completion(
    run_manifest: str | Path,
    run_completion: str | Path,
) -> dict[str, Any]:
    manifest_path = Path(run_manifest).expanduser().absolute()
    completion_path = Path(run_completion).expanduser().absolute()
    manifest, _ = _read_json_snapshot(manifest_path, label="run manifest")
    completion, completion_bytes = _read_json_snapshot(
        completion_path,
        label="run completion",
    )
    if type(manifest) is not dict or type(completion) is not dict:
        raise ValueError("run completion identity must contain JSON objects")
    rows = completion.get("results")
    instance_ids = manifest.get("instance_ids")
    if (
        set(completion) != _RUN_COMPLETION_FIELDS
        or completion.get("format") != "degs_spreadsheetbench_results_v1"
        or completion.get("total_instances") != 200
        or completion.get("completed_instances") != 200
        or type(instance_ids) is not list
        or len(instance_ids) != 200
        or len(set(instance_ids)) != 200
        or type(rows) is not list
        or len(rows) != 200
        or [row.get("id") if type(row) is dict else None for row in rows]
        != instance_ids
        or any(type(row.get("success")) is not bool for row in rows)
        or completion.get("agent_completed_instances")
        != sum(row["success"] for row in rows)
        or type(completion.get("started_at")) is not str
        or not completion["started_at"]
        or type(completion.get("ended_at")) is not str
        or not completion["ended_at"]
    ):
        raise ValueError("run completion does not prove an exact finished 200-task run")
    return {
        "file_name": completion_path.name,
        "sha256": hashlib.sha256(completion_bytes).hexdigest(),
        "completed_instances": 200,
    }


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _is_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _dependency_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for distribution in ("openai", "openpyxl", "networkx"):
        try:
            result[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            result[distribution] = "missing"
    return result


def _bundle_link_matches(
    payload: dict[str, Any],
    *,
    expected_task_ids: list[str],
) -> bool:
    try:
        root = Path(payload["bundle_dir"]).expanduser().absolute()
        raw_manifest = json.loads((root / "bundle_manifest.json").read_text())
        if raw_manifest.get("format") != EIR_BUNDLE_FORMAT:
            return False
        verified_eir = verify_contextual_bundle(
            output_dir=root,
            expected_instance_ids=expected_task_ids,
            expected_state_db=Path(payload["state_db_path"]),
            expected_dataset="SpreadsheetBench development[200,400)",
        )
        EIRGuidanceProvider.from_bundle(root)
        snapshot_manifest = json.loads(
            Path(payload["snapshot_manifest_path"]).read_text()
        )
        return (
            verified_eir.manifest.get("fixed_denominator") == 200
            and verified_eir.manifest.get("snapshot_id")
            == snapshot_manifest.get("snapshot_id")
        )
    except (
        AttributeError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ):
        return False

def _validate_run_manifest(
    path: str | Path,
    *,
    data_path: str | Path,
    start_idx: int,
    end_idx: int,
    expected_base_url: str = "http://127.0.0.1:8000/v1",
    expected_model: str = "Qwen3.5-9B-AWQ",
) -> dict[str, Any]:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if type(payload) is not dict:
        raise ValueError("run manifest does not match the DEGS schema")
    dataset_file = Path(data_path) / "dataset.json"
    development_queries = load_development_queries(dataset_file)
    expected_ids = [row["task_id"] for row in development_queries]
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
    prompt_identity = payload.get("system_prompt")
    checks = {
        "format": payload.get("format") == "spreadsheetbench_adapter_run_v1",
        "claim_scope": payload.get("claim_scope")
        == "development[200,400) diagnostic only; fixed denominator 200",
        "method": payload.get("method") == "DEGS_EXPERIENCE_GRAPH_RETRIEVAL",
        "method_version": payload.get("method_version") == __version__,
        "fixed_denominator": payload.get("fixed_denominator") == 200,
        "dataset_name": payload.get("dataset_name") == Path(data_path).name,
        "dataset_sha256": payload.get("dataset_sha256") == dataset_sha,
        "dataset_tree_sha256": payload.get("dataset_tree_sha256")
        == _tree_sha256(Path(data_path)),
        "bundle_provider": _bundle_link_matches(
            payload,
            expected_task_ids=expected_ids,
        ),
        "system_prompt": type(prompt_identity) is dict
        and prompt_identity
        == {
            "file": "preloaded_experience_full_system_v1.txt",
            "sha256": hashlib.sha256(
                (
                    Path(runtime_prompts.__file__).resolve().parent
                    / "system_prompt/preloaded_experience_full_system_v1.txt"
                ).read_bytes()
            ).hexdigest(),
        },
        "start_idx": payload.get("start_idx") == start_idx,
        "end_idx": payload.get("end_idx") == end_idx,
        "instance_ids": payload.get("instance_ids") == expected_ids,
        "model": payload.get("model") == expected_model,
        "base_url": payload.get("base_url") == expected_base_url,
        "temperature": payload.get("temperature") == 0.0,
        "max_tokens": payload.get("max_tokens") == 32_000,
        "completion_recovery_attempt_limit": payload.get(
            "completion_recovery_attempt_limit"
        )
        == 1,
        "max_consecutive_format_errors": payload.get(
            "max_consecutive_format_errors"
        )
        == 2,
        "truncate_observations": payload.get("truncate_observations") is False,
        "thinking": payload.get("thinking") == "false",
        "max_turns": payload.get("max_turns") == 30,
        "bash_timeout": payload.get("bash_timeout") == 120,
        "bash_sandbox": payload.get("bash_sandbox") == "required",
        "workers_recorded": isinstance(payload.get("workers"), int)
        and payload["workers"] > 0,
        "llm_timeout": payload.get("llm_timeout") == 600.0,
        "retry_waits": payload.get("retry_waits") == [5, 10, 30],
        "runtime_timeout_retries": payload.get("runtime_timeout_retries") == 1,
        "stagnation_repeat_limit": payload.get("stagnation_repeat_limit") == 2,
        "stagnation_recovery_attempt_limit": payload.get(
            "stagnation_recovery_attempt_limit"
        )
        == 1,
        "response_cache_disabled": payload.get("response_cache_enabled") is False,
        "runtime_event_log": payload.get("runtime_event_log")
        == "runtime_events.jsonl",
        "python_version": payload.get("python_version") == platform.python_version(),
        "dependency_versions": payload.get("dependency_versions")
        == _dependency_versions(),
        "created_at": type(payload.get("created_at")) is str
        and bool(payload.get("created_at")),
        "protocol_sha256": payload.get("protocol_sha256") == protocol_sha,
    }
    required = {
        "format",
        "claim_scope",
        "method",
        "fixed_denominator",
        "dataset_name",
        "dataset_sha256",
        "dataset_tree_sha256",
        "bundle_provider",
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
    start_idx=DEVELOPMENT_START,
    end_idx=DEVELOPMENT_END,
    verbose=False,
    recalc_dir=None,
    evaluator_backend="local",
    run_manifest=None,
    run_completion=None,
    expected_base_url="http://127.0.0.1:8000/v1",
    expected_model="Qwen3.5-9B-AWQ",
):
    """
    Evaluate outputs with an explicit SpreadsheetBench comparator backend.

    Returns:
        dict with evaluation results
    """
    if not run_manifest:
        raise ValueError("evaluation requires the matching --run-manifest")
    if not run_completion:
        raise ValueError("evaluation requires the atomic completed-run artifact")
    if start_idx != DEVELOPMENT_START or end_idx != DEVELOPMENT_END:
        raise ValueError("evaluation is fixed to development[200,400)")
    completion_identity = _validate_run_completion(run_manifest, run_completion)
    manifest_identity = _validate_run_manifest(
        run_manifest,
        data_path=data_path,
        start_idx=start_idx,
        end_idx=end_idx,
        expected_base_url=expected_base_url,
        expected_model=expected_model,
    )
    full_dataset = load_dataset(data_path)
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
        "run_completion": completion_identity,
    }

    return {
        "format": "spreadsheetbench_adapter_evaluation_v1",
        "summary": summary,
        "results": results,
    }


if __name__ == "__main__":
    raise SystemExit(
        "generic evaluator CLI is disabled; use b2-experience-evaluate-development"
    )
