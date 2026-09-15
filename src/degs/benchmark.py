from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import tempfile
import threading
from typing import Any, Mapping, Sequence

from react_agent import OpenAIClient
from sb_adapter.transport import validate_service_url
from spreadsheet_agent.runner import BenchmarkInstance, SpreadsheetBenchRunner
import spreadsheet_agent.system_prompts as runtime_prompts

from .agent import DEGSExperienceAgent
from . import __version__
from .dataset import (
    DEVELOPMENT_END,
    DEVELOPMENT_START,
    _load_development_harness_records,
)
from .dynamic_train import DYNAMIC_TRAIN_FORMAT
from .eir_bundle import verify_contextual_bundle
from .graph_dataset_contract import SPREADSHEETBENCH_GRAPH_CONTRACT
from .provider import EIRGuidanceProvider
from .state_store import EIRStateStore


WORKERS = 8
MAX_TURNS = 30
MAX_COMPLETION_TOKENS = 32_000
COMPLETION_RECOVERY_ATTEMPT_LIMIT = 1
MAX_CONSECUTIVE_FORMAT_ERRORS = 2
BASH_TIMEOUT_S = 120
LLM_TIMEOUT_S = 600.0
RETRY_WAITS_S = (5, 10, 30)
RUNTIME_TIMEOUT_RETRIES = 1
STAGNATION_REPEAT_LIMIT = 2
STAGNATION_RECOVERY_ATTEMPT_LIMIT = 1
MODEL = os.getenv("DEGS_MODEL", "Qwen3.5-9B-AWQ")
API_KEY_ENV = "DEGS_API_KEY"
TEMPERATURE = 0.0
THINKING = False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    files: list[Path] = []
    for item in root.rglob("*"):
        if item.is_dir():
            continue
        if item.is_file() and "__pycache__" not in item.parts and item.suffix != ".pyc":
            files.append(item)
    for path in sorted(files):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _dependency_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for distribution in ("openai", "openpyxl", "networkx"):
        try:
            result[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            result[distribution] = "missing"
    return result


def _write_json_atomic(path: Path, payload: Any) -> None:
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


def _load_resume_rows(
    path: Path, *, instructions: Mapping[str, str]
) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = path.read_bytes()
    if payload and not payload.endswith(b"\n"):
        payload = payload[: payload.rfind(b"\n") + 1]
        path.write_bytes(payload)
    rows: dict[str, dict[str, Any]] = {}
    for line in payload.splitlines():
        row = json.loads(line)
        task_id = str(row.get("id")) if type(row) is dict else ""
        if (
            task_id not in instructions
            or task_id in rows
            or row.get("instruction") != instructions[task_id]
        ):
            raise ValueError("resumed result ledger differs")
        rows[task_id] = row
    return rows


def _serialize_result(result: Any, output_dir: Path, *, spreadsheet_path: str) -> dict[str, Any]:
    payload = asdict(result)
    payload["id"] = str(result.id)
    for test_case in payload.get("test_cases", []):
        test_case["output_path"] = str(
            output_dir / spreadsheet_path / test_case["output_file"]
        )
    return payload


def _manifest(
    *,
    data_path: Path,
    bundle_dir: Path,
    bundle_manifest: Mapping[str, Any],
    provider: EIRGuidanceProvider,
    snapshot_manifest_path: Path,
    state_db_path: Path,
    instance_ids: Sequence[str],
    base_url: str,
) -> dict[str, Any]:
    prompt = (
        Path(runtime_prompts.__file__).resolve().parent
        / "system_prompt/preloaded_experience_full_system_v1.txt"
    )
    body = {
        "format": "spreadsheetbench_adapter_run_v1",
        "claim_scope": "development[200,400) diagnostic only; fixed denominator 200",
        "method": "DEGS_EXPERIENCE_GRAPH_RETRIEVAL",
        "method_version": __version__,
        "dataset_name": data_path.name,
        "dataset_sha256": _sha256(data_path / "dataset.json"),
        "dataset_tree_sha256": _tree_sha256(data_path),
        "bundle_dir": str(bundle_dir),
        "snapshot_manifest_path": str(snapshot_manifest_path),
        "state_db_path": str(state_db_path),
        "bundle_self_sha256": bundle_manifest["self_sha256"],
        "experience_provider": dict(provider.identity()),
        "system_prompt": {
            "file": "preloaded_experience_full_system_v1.txt",
            "sha256": _sha256(prompt),
        },
        "start_idx": DEVELOPMENT_START,
        "end_idx": DEVELOPMENT_END,
        "fixed_denominator": DEVELOPMENT_END - DEVELOPMENT_START,
        "instance_ids": list(instance_ids),
        "model": MODEL,
        "base_url": base_url,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_COMPLETION_TOKENS,
        "completion_recovery_attempt_limit": COMPLETION_RECOVERY_ATTEMPT_LIMIT,
        "max_consecutive_format_errors": MAX_CONSECUTIVE_FORMAT_ERRORS,
        "truncate_observations": False,
        "thinking": "false",
        "max_turns": MAX_TURNS,
        "bash_timeout": BASH_TIMEOUT_S,
        "bash_sandbox": "required",
        "workers": WORKERS,
        "llm_timeout": LLM_TIMEOUT_S,
        "retry_waits": list(RETRY_WAITS_S),
        "runtime_timeout_retries": RUNTIME_TIMEOUT_RETRIES,
        "stagnation_repeat_limit": STAGNATION_REPEAT_LIMIT,
        "stagnation_recovery_attempt_limit": STAGNATION_RECOVERY_ATTEMPT_LIMIT,
        "response_cache_enabled": False,
        "runtime_event_log": "runtime_events.jsonl",
        "python_version": platform.python_version(),
        "dependency_versions": _dependency_versions(),
    }
    encoded = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        **body,
        "protocol_sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _client(*, api_key: str, base_url: str) -> OpenAIClient:
    return OpenAIClient(
        model=MODEL,
        api_key=api_key,
        base_url=base_url,
        generation_config={
            "temperature": TEMPERATURE,
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": THINKING},
            },
        },
        retry_times=RETRY_WAITS_S,
        runtime_timeout_retries=RUNTIME_TIMEOUT_RETRIES,
        timeout=LLM_TIMEOUT_S,
        trust_env=False,
    )


def run(
    *,
    data_path: Path,
    snapshot_manifest_path: Path,
    state_db_path: Path,
    bundle_dir: Path,
    run_dir: Path,
    base_url: str,
    dry_run: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    validate_service_url(base_url)
    data_path = data_path.expanduser().resolve()
    snapshot_manifest_path = snapshot_manifest_path.expanduser().resolve()
    state_db_path = state_db_path.expanduser().resolve()
    bundle_dir = bundle_dir.expanduser().resolve()
    run_dir = run_dir.expanduser().absolute()
    if run_dir.exists() and not resume:
        raise FileExistsError("DEGS run directory must be fresh")
    if resume and not run_dir.is_dir():
        raise FileNotFoundError("resumed DEGS run directory must already exist")
    if dry_run and resume:
        raise ValueError("dry-run and resume are mutually exclusive")

    instances = [
        BenchmarkInstance(
            id=row["task_id"],
            instruction=row["instruction"],
            spreadsheet_path=row["spreadsheet_path"],
            instruction_type=row["instruction_type"],
            answer_position=row["answer_position"],
            metadata={},
        )
        for row in _load_development_harness_records(data_path / "dataset.json")
    ]
    instance_ids = [str(instance.id) for instance in instances]
    if len(instance_ids) != DEVELOPMENT_END - DEVELOPMENT_START or len(set(instance_ids)) != len(
        instance_ids
    ):
        raise ValueError("selected development slice is not an exact unique 200")
    verified_bundle = verify_contextual_bundle(
        output_dir=bundle_dir,
        expected_instance_ids=instance_ids,
        expected_state_db=state_db_path,
        expected_dataset="SpreadsheetBench development[200,400)",
    )
    bundle_manifest = verified_bundle.manifest
    if (
        bundle_manifest.get("model") != MODEL
        or base_url.rstrip("/")
        != str(bundle_manifest.get("generation_base_url", "")).rstrip("/")
    ):
        raise ValueError("EIR heldout bundle protocol differs")
    snapshot_manifest = json.loads(
        snapshot_manifest_path.read_text(encoding="utf-8")
    )
    if (
        type(snapshot_manifest) is not dict
        or snapshot_manifest.get("format") != DYNAMIC_TRAIN_FORMAT
        or snapshot_manifest.get("batch_index") != 24
        or snapshot_manifest.get("snapshot_id") != bundle_manifest.get("snapshot_id")
    ):
        raise ValueError("EIR heldout snapshot manifest differs")
    with EIRStateStore(
        state_db_path, dataset_contract=SPREADSHEETBENCH_GRAPH_CONTRACT
    ) as store:
        if store.head_snapshot_id != bundle_manifest.get("snapshot_id"):
            raise ValueError("EIR heldout graph HEAD differs")
    provider = EIRGuidanceProvider.from_bundle(bundle_dir)
    for instance_id in instance_ids:
        provider.for_instance(instance_id)

    manifest = _manifest(
        data_path=data_path,
        bundle_dir=bundle_dir,
        bundle_manifest=bundle_manifest,
        provider=provider,
        snapshot_manifest_path=snapshot_manifest_path,
        state_db_path=state_db_path,
        instance_ids=instance_ids,
        base_url=base_url,
    )
    if dry_run:
        return {"manifest": manifest, "results": []}

    api_key = os.getenv(API_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"missing API key: set {API_KEY_ENV}")
    if not resume:
        run_dir.mkdir(parents=True)
    output_dir = run_dir / "outputs"
    log_dir = run_dir / "logs"
    if resume:
        if not output_dir.is_dir() or not log_dir.is_dir():
            raise ValueError("resumed DEGS run is missing its output or log directory")
        stored_manifest = json.loads(
            (output_dir / "run_manifest.json").read_bytes()
        )
        if (
            {key: value for key, value in stored_manifest.items() if key != "created_at"}
            != {key: value for key, value in manifest.items() if key != "created_at"}
        ):
            raise ValueError("resumed DEGS run identity differs")
        manifest = stored_manifest
    else:
        output_dir.mkdir()
        log_dir.mkdir()
        _write_json_atomic(output_dir / "run_manifest.json", manifest)
    os.environ["SB_RUN_PROTOCOL_SHA256"] = manifest["protocol_sha256"]
    os.environ["REACT_AGENT_USAGE_LOG"] = str(run_dir / "react_usage.jsonl")
    os.environ["REACT_AGENT_RUNTIME_EVENT_LOG"] = str(
        run_dir / "runtime_events.jsonl"
    )

    working_directory = tempfile.TemporaryDirectory(prefix="degs_benchmark_")
    working_dir = Path(working_directory.name)
    thread_state = threading.local()

    def process(instance: Any) -> Any:
        if not hasattr(thread_state, "runner"):
            agent = DEGSExperienceAgent(
                client=_client(api_key=api_key, base_url=base_url),
                max_turns=MAX_TURNS,
                verbose=False,
                timeout=BASH_TIMEOUT_S,
                sandbox_mode="required",
                log_dir=str(log_dir),
                stagnation_repeat_limit=STAGNATION_REPEAT_LIMIT,
                stagnation_recovery_attempt_limit=(
                    STAGNATION_RECOVERY_ATTEMPT_LIMIT
                ),
                max_completion_tokens=MAX_COMPLETION_TOKENS,
                completion_recovery_attempt_limit=(
                    COMPLETION_RECOVERY_ATTEMPT_LIMIT
                ),
                max_consecutive_format_errors=MAX_CONSECUTIVE_FORMAT_ERRORS,
                truncate_observations=False,
                libreoffice_output_feedback=False,
                structured_process_contract=False,
                experience_provider=provider,
            )
            thread_state.runner = SpreadsheetBenchRunner(
                agent=agent,
                data_path=str(data_path),
                output_dir=str(output_dir),
                working_dir=str(working_dir),
                workbook_structure_preflight=False,
            )
        return thread_state.runner.run_instance(instance)

    ledger_path = run_dir / "results.jsonl"
    rows_by_id: dict[str, dict[str, Any]] = {}
    instructions = {
        str(instance.id): instance.instruction for instance in instances
    }
    if resume:
        rows_by_id = _load_resume_rows(
            ledger_path, instructions=instructions
        )
    pending_instances = [
        instance for instance in instances if str(instance.id) not in rows_by_id
    ]
    started_at = datetime.now(timezone.utc)
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {
            pool.submit(process, instance): instance for instance in pending_instances
        }
        with ledger_path.open("a" if resume else "x", encoding="utf-8") as ledger:
            for future in as_completed(futures):
                instance = futures[future]
                try:
                    row = _serialize_result(
                        future.result(),
                        output_dir,
                        spreadsheet_path=str(instance.spreadsheet_path),
                    )
                except Exception as exc:
                    row = {
                        "id": str(instance.id),
                        "instruction": instance.instruction,
                        "success": False,
                        "test_cases": [],
                        "error": f"worker_internal_error:{type(exc).__name__}:{exc}",
                        "failure_type": "runtime_invalid_exception",
                    }
                rows_by_id[str(instance.id)] = row
                ledger.write(json.dumps(row, ensure_ascii=False) + "\n")
                ledger.flush()
                os.fsync(ledger.fileno())

    ordered = [rows_by_id[instance_id] for instance_id in instance_ids]
    payload = {
        "format": "degs_spreadsheetbench_results_v1",
        "run_manifest": str(output_dir / "run_manifest.json"),
        "protocol_sha256": manifest["protocol_sha256"],
        "started_at": started_at.isoformat(),
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "total_instances": len(instance_ids),
        "completed_instances": len(ordered),
        "agent_completed_instances": sum(bool(row.get("success")) for row in ordered),
        "results": ordered,
    }
    _write_json_atomic(run_dir / "results.json", payload)
    working_directory.cleanup()
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run DEGS on fixed development[200,400)."
    )
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--snapshot-manifest-path", type=Path, required=True)
    parser.add_argument("--state-db", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--base-url",
        default=os.getenv("DEGS_CHAT_BASE_URL"),
        help="OpenAI-compatible chat service base URL ending in /v1.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.base_url:
        raise SystemExit("run requires --base-url or DEGS_CHAT_BASE_URL")
    result = run(
        data_path=args.data_path,
        snapshot_manifest_path=args.snapshot_manifest_path,
        state_db_path=args.state_db,
        bundle_dir=args.bundle_dir,
        run_dir=args.run_dir,
        base_url=args.base_url,
        dry_run=args.dry_run,
        resume=args.resume,
    )
    print(json.dumps(result["manifest"] if args.dry_run else {
        "total_instances": result["total_instances"],
        "completed_instances": result["completed_instances"],
        "agent_completed_instances": result["agent_completed_instances"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
