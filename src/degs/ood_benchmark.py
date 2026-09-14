"""Run DEGS over a verified table-QA OOD population."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Any, Mapping, Sequence

from sb_adapter.transport import validate_service_url

from . import __version__
from . import bundle as retrieval
from .benchmark import (
    API_KEY_ENV,
    BASH_TIMEOUT_S,
    COMPLETION_RECOVERY_ATTEMPT_LIMIT,
    LLM_TIMEOUT_S,
    MAX_COMPLETION_TOKENS,
    MAX_CONSECUTIVE_FORMAT_ERRORS,
    MAX_TURNS,
    MODEL,
    RETRY_WAITS_S,
    RUNTIME_TIMEOUT_RETRIES,
    STAGNATION_RECOVERY_ATTEMPT_LIMIT,
    STAGNATION_REPEAT_LIMIT,
    TEMPERATURE,
    THINKING,
    _client,
)
from .core import canonical_json_bytes
from .ood_bundle import OODExperienceProvider, verify_from_paths as verify_bundle
from .ood_dataset import verify_population
from .soft_hard_benchmark import _GenerationTransportTracker, _TrackedAgentClient
from spreadsheet_agent.agents.cli_only_agent import CLIOnlyAgent
from spreadsheet_agent.runner import SpreadsheetBenchRunner
from spreadsheet_agent.system_prompts import render_full_system_prompt


FORMAT = "degs_tableqa_ood_agent_run_v1"
RESULT_FORMAT = "degs_tableqa_ood_case_result_v1"
COMPLETION_FORMAT = "degs_tableqa_ood_agent_completion_v1"
WORKERS = 8


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with staging.open("xb") as handle:
        handle.write(payload)
    os.replace(staging, path)


def _case_result_path(root: Path, task_id: str) -> Path:
    return root / f"{_sha(task_id.encode('utf-8'))}.json"


def _output_identity(path: Path) -> tuple[str | None, int]:
    if path.is_symlink() or not path.is_file():
        return None, 0
    payload = path.read_bytes()
    return _sha(payload), len(payload)


class OODExperienceAgent(CLIOnlyAgent):
    def __init__(self, *args: Any, experience_provider: OODExperienceProvider, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if type(experience_provider) is not OODExperienceProvider:
            raise ValueError("OOD Agent requires the verified OOD experience provider")
        self.experience_provider = experience_provider
        self._experience_content = ""

    @property
    def name(self) -> str:
        return "degs_ood_experience_agent"

    def get_system_template(self) -> str:
        return render_full_system_prompt(
            "preloaded_experience_full_system_v1.txt",
            experience_content=self._experience_content,
        )

    def run(self, context: Any) -> dict[str, Any]:
        retrieval_id = getattr(context, "retrieval_id", "") or context.instance_id
        payload = self.experience_provider.for_instance(retrieval_id)
        if (
            payload.metadata.get("format") != retrieval.EXPERIENCE_FORMAT
            or payload.metadata.get("method_family") != retrieval.METHOD_FAMILY
        ):
            raise ValueError("OOD task experience is outside the DEGS method boundary")
        self._experience_content = payload.experience
        self._agent = None
        return super().run(context)


def _manifest(
    *,
    population: Mapping[str, Any],
    prepared_data_path: Path,
    bundle: Any,
    provider: OODExperienceProvider,
    source_dataset_path: Path,
    snapshot_manifest_path: Path,
    state_db_path: Path,
    base_url: str,
) -> dict[str, Any]:
    body = {
        "format": FORMAT,
        "method_version": __version__,
        "method_name": retrieval.METHOD_NAME,
        "dataset": population["dataset"],
        "task_count": population["task_count"],
        "workers": WORKERS,
        "model": MODEL,
        "temperature": TEMPERATURE,
        "thinking": THINKING,
        "max_turns": MAX_TURNS,
        "max_tokens": MAX_COMPLETION_TOKENS,
        "bash_timeout_s": BASH_TIMEOUT_S,
        "llm_timeout_s": LLM_TIMEOUT_S,
        "retry_waits_s": list(RETRY_WAITS_S),
        "runtime_timeout_retries": RUNTIME_TIMEOUT_RETRIES,
        "stagnation_repeat_limit": STAGNATION_REPEAT_LIMIT,
        "stagnation_recovery_attempt_limit": STAGNATION_RECOVERY_ATTEMPT_LIMIT,
        "completion_recovery_attempt_limit": COMPLETION_RECOVERY_ATTEMPT_LIMIT,
        "max_consecutive_format_errors": MAX_CONSECUTIVE_FORMAT_ERRORS,
        "generation_base_url": base_url.rstrip("/"),
        "population_manifest_sha256": population["self_sha256"],
        "query_projection_sha256": population["query_projection_sha256"],
        "input_tree_sha256": population["input_tree_sha256"],
        "prepared_data_path": str(prepared_data_path),
        "source_dataset_path": str(source_dataset_path),
        "snapshot_manifest_path": str(snapshot_manifest_path),
        "state_db_path": str(state_db_path),
        "bundle": {
            "self_sha256": bundle.manifest["self_sha256"],
            "experience_sha256": bundle.manifest["experience_sha256"],
            "row_count": bundle.manifest["row_count"],
        },
        "provider": dict(provider.identity()),
        "agent_prompt_sha256": _sha(
            (
                Path(__file__).resolve().parents[1]
                / "spreadsheet_agent/system_prompt/preloaded_experience_full_system_v1.txt"
            ).read_bytes()
        ),
        "gold_available_to_agent": False,
    }
    return {**body, "protocol_sha256": _sha(canonical_json_bytes(body))}


def _instances(prepared_data_path: Path) -> dict[str, Any]:
    runner = SpreadsheetBenchRunner(
        agent=None,  # type: ignore[arg-type] -- load_data does not access the Agent.
        data_path=str(prepared_data_path),
        output_dir=str(prepared_data_path),
        working_dir=str(prepared_data_path),
    )
    return {instance.id: instance for instance in runner.load_data()}


def _load_completed(
    *,
    tasks: Sequence[Mapping[str, Any]],
    case_results_dir: Path,
    output_dir: Path,
    protocol_sha256: str,
) -> tuple[dict[str, dict[str, Any]], list[Mapping[str, Any]]]:
    completed: dict[str, dict[str, Any]] = {}
    pending = []
    for task in tasks:
        path = _case_result_path(case_results_dir, str(task["task_id"]))
        if not path.is_file():
            pending.append(task)
            continue
        row = _read_object(path)
        output_path = output_dir / task["spreadsheet_path"] / task["output_file"]
        output_sha, output_size = _output_identity(output_path)
        if (
            row.get("format") != RESULT_FORMAT
            or row.get("protocol_sha256") != protocol_sha256
            or row.get("task_id") != task["task_id"]
            or row.get("query_index") != task["query_index"]
            or row.get("output_sha256") != output_sha
            or row.get("output_size") != output_size
        ):
            raise ValueError(f"OOD resume result differs for {task['task_id']}")
        completed[str(task["task_id"])] = row
    return completed, pending


def _run_locked(
    *,
    source_dataset_path: Path,
    prepared_data_path: Path,
    snapshot_manifest_path: Path,
    state_db_path: Path,
    bundle_dir: Path,
    run_dir: Path,
    base_url: str,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    validate_service_url(base_url)
    source_dataset_path = source_dataset_path.expanduser().resolve()
    prepared_data_path = prepared_data_path.expanduser().resolve()
    snapshot_manifest_path = snapshot_manifest_path.expanduser().resolve()
    state_db_path = state_db_path.expanduser().resolve()
    bundle_dir = bundle_dir.expanduser().resolve()
    run_dir = run_dir.expanduser().absolute()
    population = verify_population(prepared_data_path)
    bundle = verify_bundle(
        source_dataset_path=source_dataset_path,
        prepared_data_path=prepared_data_path,
        snapshot_manifest_path=snapshot_manifest_path,
        state_db_path=state_db_path,
        output_dir=bundle_dir,
    )
    provider = OODExperienceProvider(bundle)
    expected_manifest = _manifest(
        population=population,
        prepared_data_path=prepared_data_path,
        bundle=bundle,
        provider=provider,
        source_dataset_path=source_dataset_path,
        snapshot_manifest_path=snapshot_manifest_path,
        state_db_path=state_db_path,
        base_url=base_url,
    )
    if dry_run:
        return {"manifest": expected_manifest, "completed_tasks": 0}
    manifest_path = run_dir / "run_manifest.json"
    if run_dir.exists() or run_dir.is_symlink():
        if not resume or run_dir.is_symlink() or not manifest_path.is_file():
            raise FileExistsError("OOD run directory must be fresh, or use --resume")
        if _read_object(manifest_path) != expected_manifest:
            raise ValueError("OOD resume protocol identity differs")
    else:
        for directory in (run_dir, run_dir / "outputs", run_dir / "logs", run_dir / "case_results"):
            directory.mkdir(mode=0o700, parents=directory == run_dir)
        _write_atomic(manifest_path, json.dumps(expected_manifest, ensure_ascii=False, indent=2).encode("utf-8") + b"\n")
    output_dir = run_dir / "outputs"
    log_dir = run_dir / "logs"
    case_results_dir = run_dir / "case_results"
    tasks = population["tasks"]
    instances = _instances(prepared_data_path)
    completed, pending = _load_completed(
        tasks=tasks,
        case_results_dir=case_results_dir,
        output_dir=output_dir,
        protocol_sha256=expected_manifest["protocol_sha256"],
    )
    print(
        f"OOD {population['dataset']}: completed={len(completed)} pending={len(pending)} "
        f"total={len(tasks)} workers={WORKERS}",
        flush=True,
    )
    api_key = os.getenv(API_KEY_ENV, "")
    if pending and not api_key:
        raise RuntimeError(f"missing API key: set {API_KEY_ENV}")
    if pending:
        preflight = _client(api_key=api_key, base_url=base_url)
        models = {str(item.id) for item in preflight._client.models.list().data}
        if MODEL not in models:
            raise RuntimeError(f"configured model is unavailable: {MODEL}")
    os.environ["SB_RUN_PROTOCOL_SHA256"] = expected_manifest["protocol_sha256"]
    os.environ["REACT_AGENT_USAGE_LOG"] = str(run_dir / "react_usage.jsonl")
    os.environ["REACT_AGENT_RUNTIME_EVENT_LOG"] = str(run_dir / "runtime_events.jsonl")
    temp = tempfile.TemporaryDirectory(prefix=f"degs_ood_{population['dataset']}_")
    working_dir = Path(temp.name)
    thread_state = threading.local()
    tracker = _GenerationTransportTracker()

    def process(task: Mapping[str, Any]) -> dict[str, Any]:
        task_started_at = datetime.now(timezone.utc)
        task_started = time.monotonic()
        if not hasattr(thread_state, "runner"):
            agent = OODExperienceAgent(
                client=_TrackedAgentClient(_client(api_key=api_key, base_url=base_url), tracker),
                max_turns=MAX_TURNS,
                verbose=False,
                timeout=BASH_TIMEOUT_S,
                sandbox_mode="required",
                log_dir=str(log_dir),
                stagnation_repeat_limit=STAGNATION_REPEAT_LIMIT,
                stagnation_recovery_attempt_limit=STAGNATION_RECOVERY_ATTEMPT_LIMIT,
                max_completion_tokens=MAX_COMPLETION_TOKENS,
                completion_recovery_attempt_limit=COMPLETION_RECOVERY_ATTEMPT_LIMIT,
                max_consecutive_format_errors=MAX_CONSECUTIVE_FORMAT_ERRORS,
                truncate_observations=False,
                libreoffice_output_feedback=False,
                structured_process_contract=False,
                experience_provider=provider,
            )
            thread_state.runner = SpreadsheetBenchRunner(
                agent=agent,
                data_path=str(prepared_data_path),
                output_dir=str(output_dir),
                working_dir=str(working_dir),
                workbook_structure_preflight=False,
            )
        result = thread_state.runner.run_test_case(
            instances[str(task["task_id"])],
            str(task["input_file"]),
            runtime_instance_id=str(task["task_id"]),
            retrieval_id=str(task["task_id"]),
        )
        payload = asdict(result)
        output_path = output_dir / task["spreadsheet_path"] / task["output_file"]
        output_sha, output_size = _output_identity(output_path)
        return {
            "format": RESULT_FORMAT,
            "protocol_sha256": expected_manifest["protocol_sha256"],
            "dataset": population["dataset"],
            "task_id": task["task_id"],
            "source_id": task["source_id"],
            "query_index": task["query_index"],
            "output_path": str(output_path),
            "output_sha256": output_sha,
            "output_size": output_size,
            "agent_success": bool(payload["success"]),
            "agent_completed": bool(payload["agent_completed"]),
            "output_preserved": bool(payload["output_preserved"]),
            "turns": int(payload["turns"]),
            "answer": str(payload["agent_answer"]),
            "error": str(payload["error"]),
            "failure_kind": str(payload.get("failure_kind", "")),
            "started_at": task_started_at.isoformat(),
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "wall_seconds": time.monotonic() - task_started,
        }

    started = datetime.now(timezone.utc)
    try:
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            processed = 0
            for start in range(0, len(pending), WORKERS):
                wave = pending[start : start + WORKERS]
                transport_wave = tracker.begin_wave()
                futures = {pool.submit(process, task): task for task in wave}
                wave_rows = []
                for future in as_completed(futures):
                    task = futures[future]
                    try:
                        row = future.result()
                    except Exception as exc:
                        output_path = output_dir / task["spreadsheet_path"] / task["output_file"]
                        output_sha, output_size = _output_identity(output_path)
                        row = {
                            "format": RESULT_FORMAT,
                            "protocol_sha256": expected_manifest["protocol_sha256"],
                            "dataset": population["dataset"],
                            "task_id": task["task_id"],
                            "source_id": task["source_id"],
                            "query_index": task["query_index"],
                            "output_path": str(output_path),
                            "output_sha256": output_sha,
                            "output_size": output_size,
                            "agent_success": False,
                            "agent_completed": False,
                            "output_preserved": output_sha is not None,
                            "turns": 0,
                            "answer": "",
                            "error": f"{type(exc).__name__}: {exc}",
                            "failure_kind": "worker_exception",
                            "started_at": "",
                            "ended_at": datetime.now(timezone.utc).isoformat(),
                            "wall_seconds": 0.0,
                        }
                    wave_rows.append((task, row))
                tracker.raise_if_systemic(transport_wave)
                for task, row in wave_rows:
                    _write_atomic(
                        _case_result_path(case_results_dir, str(task["task_id"])),
                        json.dumps(row, ensure_ascii=False, indent=2).encode("utf-8") + b"\n",
                    )
                    completed[str(task["task_id"])] = row
                    processed += 1
                    if processed % 25 == 0 or processed == len(pending):
                        print(f"completed {len(completed)}/{len(tasks)} OOD tasks", flush=True)
    finally:
        temp.cleanup()
    if len(completed) != len(tasks):
        raise RuntimeError("OOD run ended before every task was recorded")
    ordered = [completed[str(task["task_id"])] for task in tasks]
    ledger = b"".join(canonical_json_bytes(row) + b"\n" for row in ordered)
    _write_atomic(run_dir / "results.jsonl", ledger)
    result = {
        "format": COMPLETION_FORMAT,
        "protocol_sha256": expected_manifest["protocol_sha256"],
        "dataset": population["dataset"],
        "started_at": started.isoformat(),
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "task_denominator": len(tasks),
        "completed_tasks": len(ordered),
        "agent_success_tasks": sum(row["agent_success"] for row in ordered),
        "agent_completed_tasks": sum(row["agent_completed"] for row in ordered),
        "output_preserved_tasks": sum(row["output_preserved"] for row in ordered),
        "results_jsonl_sha256": _sha(ledger),
    }
    _write_atomic(
        run_dir / "results.json",
        json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8") + b"\n",
    )
    return result


def run(
    *,
    source_dataset_path: Path,
    prepared_data_path: Path,
    snapshot_manifest_path: Path,
    state_db_path: Path,
    bundle_dir: Path,
    run_dir: Path,
    base_url: str,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    target = run_dir.expanduser().absolute()
    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target.parent / f".{target.name}.run.lock"
    descriptor = os.open(
        lock_path,
        os.O_CREAT | os.O_RDWR,
    )
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another OOD runner owns this run") from exc
        return _run_locked(
            source_dataset_path=source_dataset_path,
            prepared_data_path=prepared_data_path,
            snapshot_manifest_path=snapshot_manifest_path,
            state_db_path=state_db_path,
            bundle_dir=bundle_dir,
            run_dir=target,
            base_url=base_url,
            resume=resume,
            dry_run=dry_run,
        )
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset-path", type=Path, required=True)
    parser.add_argument("--prepared-data-path", type=Path, required=True)
    parser.add_argument("--snapshot-manifest-path", type=Path, required=True)
    parser.add_argument("--state-db", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run(
        source_dataset_path=args.source_dataset_path,
        prepared_data_path=args.prepared_data_path,
        snapshot_manifest_path=args.snapshot_manifest_path,
        state_db_path=args.state_db,
        bundle_dir=args.bundle_dir,
        run_dir=args.run_dir,
        base_url=args.base_url,
        resume=args.resume,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
