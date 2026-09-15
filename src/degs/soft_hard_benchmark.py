"""Run DEGS over the fixed Soft/Hard case plan."""

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
from typing import Any, Mapping, Sequence

from openai import APIError
from react_agent.models import (
    RequestCompletionLengthExceeded,
    RequestContextLengthExceeded,
    RequestRuntimeTimeout,
)
from sb_adapter.transport import validate_service_url
from spreadsheet_agent.agents.cli_only_agent import CLIOnlyAgent
from spreadsheet_agent.runner import BenchmarkInstance, SpreadsheetBenchRunner
import spreadsheet_agent.system_prompts as runtime_prompts
from spreadsheet_agent.system_prompts import render_full_system_prompt

from . import __version__
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
    WORKERS,
    _client,
    _dependency_versions,
    _sha256,
    _write_json_atomic,
)
from .soft_hard_bundle import FORMAT as BUNDLE_FORMAT
from .soft_hard_bundle import SoftHardExperienceProvider
from .soft_hard_bundle import verify_from_paths as verify_bundle
from .soft_hard_dataset import (
    TASK_COUNT,
    TESTCASE_COUNT,
    canonical_json_bytes,
    load_prepared_retrieval_population,
)
from .provider import EIR_GUIDANCE_FORMAT, EIR_METHOD_FAMILY


FORMAT = "degs_spreadsheetbench_soft_hard_run_v1"
RESULT_FORMAT = "degs_spreadsheetbench_soft_hard_case_result_v1"
COMPLETION_FORMAT = "degs_spreadsheetbench_soft_hard_completion_v1"
CASE_RESULT_FIELDS = {
    "format",
    "protocol_sha256",
    "case_id",
    "task_id",
    "query_index",
    "input_file",
    "output_file",
    "output_path",
    "output_sha256",
    "output_size",
    "agent_success",
    "agent_completed",
    "output_preserved",
    "turns",
    "answer",
    "error",
    "failure_kind",
}
COMPLETION_FIELDS = {
    "format",
    "run_manifest",
    "protocol_sha256",
    "started_at",
    "ended_at",
    "task_denominator",
    "testcase_denominator",
    "completed_cases",
    "agent_success_cases",
    "agent_completed_cases",
    "output_preserved_cases",
    "results_jsonl",
    "results_jsonl_sha256",
}


def _protocol_body(
    *,
    prepared_data_path: Path,
    input_manifest_path: Path,
    population: Mapping[str, Any],
    source_dataset_path: Path,
    bundle_dir: Path,
    bundle_manifest: Mapping[str, Any],
    provider: SoftHardExperienceProvider,
    snapshot_manifest_path: Path,
    state_db_path: Path,
    base_url: str,
) -> dict[str, Any]:
    prompt = (
        Path(runtime_prompts.__file__).resolve().parent
        / "system_prompt/preloaded_experience_full_system_v1.txt"
    )
    case_ids = [
        case["case_id"]
        for task in population["tasks"]
        for case in task["cases"]
    ]
    task_ids = [task["task_id"] for task in population["tasks"]]
    return {
        "format": FORMAT,
        "claim_scope": (
            "single-run SpreadsheetBench Soft/Hard after task-ID/prefix-1 aligned "
            "train-case exclusion; final scores require LibreOffice recalculation"
        ),
        "method": "DEGS_EXPERIENCE_GRAPH_RETRIEVAL",
        "method_version": __version__,
        "source_dataset_path": str(source_dataset_path),
        "prepared_data_path": str(prepared_data_path),
        "input_manifest_path": str(input_manifest_path),
        "input_manifest_sha256": population["self_sha256"],
        "prepared_input_tree_sha256": population["prepared_input_tree_sha256"],
        "bundle_dir": str(bundle_dir),
        "bundle_format": BUNDLE_FORMAT,
        "bundle_self_sha256": bundle_manifest["self_sha256"],
        "snapshot_manifest_path": str(snapshot_manifest_path),
        "state_db_path": str(state_db_path),
        "experience_provider": dict(provider.identity()),
        "system_prompt": {
            "file": "preloaded_experience_full_system_v1.txt",
            "sha256": _sha256(prompt),
        },
        "task_denominator": TASK_COUNT,
        "testcase_denominator": TESTCASE_COUNT,
        "task_ids": task_ids,
        "case_ids": case_ids,
        "model": MODEL,
        "base_url": base_url,
        "temperature": TEMPERATURE,
        "thinking": THINKING,
        "seed_policy": "current_vrf_no_explicit_seed",
        "max_tokens": MAX_COMPLETION_TOKENS,
        "completion_recovery_attempt_limit": COMPLETION_RECOVERY_ATTEMPT_LIMIT,
        "max_consecutive_format_errors": MAX_CONSECUTIVE_FORMAT_ERRORS,
        "truncate_observations": False,
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
        "dependency_versions": _dependency_versions(),
    }


def _manifest(**kwargs: Any) -> dict[str, Any]:
    body = _protocol_body(**kwargs)
    return {
        **body,
        "protocol_sha256": hashlib.sha256(canonical_json_bytes(body)).hexdigest(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


class SoftHardExperienceAgent(CLIOnlyAgent):
    """Frozen Agent behavior with case IDs separated from task retrieval IDs."""

    def __init__(
        self,
        *args: Any,
        experience_provider: SoftHardExperienceProvider,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if type(experience_provider) is not SoftHardExperienceProvider:
            raise ValueError("Soft/Hard Agent requires its verified task provider")
        self.experience_provider = experience_provider
        self._experience_content = ""

    @property
    def name(self) -> str:
        return "degs_eir_contextual_guidance_agent"

    def get_system_template(self) -> str:
        return render_full_system_prompt(
            "preloaded_experience_full_system_v1.txt",
            experience_content=self._experience_content,
        )

    def run(self, context: Any) -> dict[str, Any]:
        retrieval_id = getattr(context, "retrieval_id", "") or context.instance_id
        payload = self.experience_provider.for_instance(retrieval_id)
        if (
            payload.metadata.get("format") != EIR_GUIDANCE_FORMAT
            or payload.metadata.get("method_family") != EIR_METHOD_FAMILY
        ):
            raise ValueError("task experience is outside the method boundary")
        self._experience_content = payload.experience
        self._agent = None
        return super().run(context)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _case_result_path(case_results_dir: Path, case_id: str) -> Path:
    digest = hashlib.sha256(case_id.encode("utf-8")).hexdigest()
    return case_results_dir / f"{digest}.json"


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _require_run_directory(run_dir: Path, name: str) -> Path:
    child = run_dir / name
    if child.is_symlink() or not child.is_dir():
        raise ValueError(f"run {name} directory differs")
    if child.resolve().parent != run_dir.resolve():
        raise ValueError(f"run {name} directory escapes run root")
    return child


def _output_identity(path: Path) -> tuple[str, int]:
    if path.is_symlink() or not path.is_file():
        return "", 0
    return _sha256(path), path.stat().st_size


class _GenerationTransportTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._fatal_event = threading.Event()
        self._fatal_message = ""
        self._success_events: list[str] = []
        self._failure_events: list[tuple[str, str, int | None, bool]] = []

    def begin_wave(self) -> tuple[int, int]:
        with self._lock:
            return len(self._success_events), len(self._failure_events)

    def record_success(self, request_id: str) -> None:
        with self._lock:
            self._success_events.append(request_id)

    def record_failure(
        self,
        request_id: str,
        error: Exception,
        *,
        case_id: str | None = None,
        fatal: bool = False,
    ) -> None:
        status_code = getattr(error, "status_code", None)
        is_fatal = fatal or status_code in {401, 403, 404}
        with self._lock:
            self._failure_events.append(
                (
                    request_id,
                    case_id or request_id,
                    status_code,
                    is_fatal,
                )
            )
        if is_fatal:
            self.signal_fatal(
                "generation endpoint configuration failed"
                if status_code in {401, 403, 404}
                else "unknown generation client failure"
            )

    def signal_fatal(self, message: str) -> None:
        with self._lock:
            if not self._fatal_event.is_set():
                self._fatal_message = message
                self._fatal_event.set()

    def raise_if_any_fatal(self) -> None:
        if self._fatal_event.is_set():
            with self._lock:
                message = self._fatal_message
            raise RuntimeError(f"{message}; current wave remains pending")

    def raise_if_fatal(self, wave: tuple[int, int]) -> None:
        with self._lock:
            failures = self._failure_events[wave[1] :]
        fatal = [row for row in failures if row[3]]
        if fatal:
            status_codes = {row[2] for row in fatal}
            message = (
                "generation endpoint configuration failed"
                if status_codes & {401, 403, 404}
                else "unknown generation client failure"
            )
            raise RuntimeError(f"{message}; current wave remains pending")

    def has_item_local_transport_failure(
        self, case_id: str, wave: tuple[int, int]
    ) -> bool:
        with self._lock:
            failures = self._failure_events[wave[1] :]
        return any(row[1] == case_id and not row[3] for row in failures)

    def raise_if_systemic(self, wave: tuple[int, int]) -> None:
        self.raise_if_fatal(wave)
        with self._lock:
            successes = self._success_events[wave[0] :]
            failures = self._failure_events[wave[1] :]
        if not successes and len({row[0] for row in failures}) >= 3:
            raise RuntimeError(
                "systemic generation transport failure; current wave remains pending"
            )


class _TrackedAgentClient:
    def __init__(self, client: Any, tracker: _GenerationTransportTracker) -> None:
        self._client = client
        self._tracker = tracker
        self._case_id = ""
        self._request_index = 0

    def set_runtime_context(self, *, instance_id: str, protocol_sha256: str) -> None:
        self._case_id = instance_id
        self._request_index = 0
        self._client.set_runtime_context(
            instance_id=instance_id,
            protocol_sha256=protocol_sha256,
        )

    async def chat_async(self, *args: Any, **kwargs: Any) -> str:
        self._tracker.raise_if_any_fatal()
        self._request_index += 1
        request_id = f"{self._case_id}:{self._request_index}"
        try:
            response = await self._client.chat_async(*args, **kwargs)
        except (RequestRuntimeTimeout, APIError) as exc:
            self._tracker.record_failure(
                request_id, exc, case_id=self._case_id
            )
            raise
        except (RequestCompletionLengthExceeded, RequestContextLengthExceeded):
            self._tracker.record_success(request_id)
            raise
        except Exception as exc:
            self._tracker.record_failure(
                request_id, exc, case_id=self._case_id, fatal=True
            )
            raise
        self._tracker.record_success(request_id)
        self._tracker.raise_if_any_fatal()
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


def _validate_case_row(
    row: Any,
    *,
    case: Mapping[str, Any],
    task: Mapping[str, Any],
    protocol_sha256: str,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    expected_output_path = (
        str(output_dir / task["spreadsheet_path"] / case["output_file"])
        if output_dir is not None
        else row.get("output_path") if type(row) is dict else None
    )
    if (
        type(row) is not dict
        or set(row) != CASE_RESULT_FIELDS
        or row.get("format") != RESULT_FORMAT
        or row.get("protocol_sha256") != protocol_sha256
        or row.get("case_id") != case["case_id"]
        or row.get("task_id") != task["task_id"]
        or row.get("query_index") != task["query_index"]
        or row.get("input_file") != case["input_file"]
        or row.get("output_file") != case["output_file"]
        or row.get("output_path") != expected_output_path
        or type(row.get("agent_success")) is not bool
        or type(row.get("agent_completed")) is not bool
        or type(row.get("output_preserved")) is not bool
        or type(row.get("turns")) is not int
        or type(row.get("output_sha256")) is not str
        or type(row.get("output_size")) is not int
        or type(row.get("error")) is not str
        or type(row.get("failure_kind")) is not str
    ):
        raise ValueError(f"completed case identity differs: {case['case_id']}")
    output_path = Path(row["output_path"])
    actual_sha, actual_size = _output_identity(output_path)
    if row["output_preserved"]:
        if (
            len(row["output_sha256"]) != 64
            or row["output_size"] <= 0
            or (actual_sha, actual_size)
            != (row["output_sha256"], row["output_size"])
        ):
            raise ValueError(f"completed output identity differs: {case['case_id']}")
    elif row["output_sha256"] or row["output_size"] != 0 or actual_sha:
        raise ValueError(f"absent output identity differs: {case['case_id']}")
    return row


def _task_instances(population: Mapping[str, Any]) -> dict[str, BenchmarkInstance]:
    return {
        task["task_id"]: BenchmarkInstance(
            id=task["task_id"],
            instruction=task["instruction"],
            spreadsheet_path=task["spreadsheet_path"],
            instruction_type=task["instruction_type"],
            answer_position=task["answer_position"],
            metadata={},
        )
        for task in population["tasks"]
    }


def _load_completed_case_results(
    *,
    case_plan: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    case_results_dir: Path,
    protocol_sha256: str,
    output_dir: Path | None = None,
) -> tuple[dict[str, dict[str, Any]], list[tuple[Mapping[str, Any], Mapping[str, Any]]]]:
    completed: dict[str, dict[str, Any]] = {}
    pending: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for task, case in case_plan:
        path = _case_result_path(case_results_dir, case["case_id"])
        if path.is_symlink():
            raise ValueError(f"completed case journal is a symlink: {case['case_id']}")
        if path.exists():
            completed[case["case_id"]] = _validate_case_row(
                _read_json(path),
                case=case,
                task=task,
                protocol_sha256=protocol_sha256,
                output_dir=output_dir,
            )
        else:
            pending.append((task, case))
    return completed, pending


def verify_completed_run(
    *, run_dir: Path, population: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], bytes]:
    root = run_dir.expanduser().resolve()
    output_dir = _require_run_directory(root, "outputs")
    _require_run_directory(root, "logs")
    _require_run_directory(root, "case_results")
    manifest_path = output_dir / "run_manifest.json"
    completion_path = root / "results.json"
    ledger_path = root / "results.jsonl"
    if any(
        path.is_symlink() or not path.is_file()
        for path in (manifest_path, completion_path, ledger_path)
    ):
        raise ValueError("completed Soft/Hard run files differ")
    manifest = _read_json(manifest_path)
    completion = _read_json(completion_path)
    body = {
        key: value
        for key, value in manifest.items()
        if key not in {"protocol_sha256", "created_at"}
    }
    expected_task_ids = [task["task_id"] for task in population["tasks"]]
    case_plan = [
        (task, case)
        for task in population["tasks"]
        for case in task["cases"]
    ]
    expected_case_ids = [case["case_id"] for _task, case in case_plan]
    ledger_bytes = ledger_path.read_bytes()
    lines = ledger_bytes.splitlines()
    rows = [json.loads(line) for line in lines]
    if (
        manifest.get("format") != FORMAT
        or manifest.get("protocol_sha256")
        != hashlib.sha256(canonical_json_bytes(body)).hexdigest()
        or manifest.get("method") != "DEGS_EXPERIENCE_GRAPH_RETRIEVAL"
        or manifest.get("method_version") != __version__
        or manifest.get("bundle_format") != BUNDLE_FORMAT
        or manifest.get("input_manifest_sha256")
        != population["self_sha256"]
        or manifest.get("prepared_input_tree_sha256")
        != population["prepared_input_tree_sha256"]
        or manifest.get("task_denominator") != TASK_COUNT
        or manifest.get("testcase_denominator") != TESTCASE_COUNT
        or manifest.get("task_ids") != expected_task_ids
        or manifest.get("case_ids") != expected_case_ids
        or manifest.get("model") != MODEL
        or manifest.get("temperature") != TEMPERATURE
        or manifest.get("thinking") is not THINKING
        or manifest.get("seed_policy") != "current_vrf_no_explicit_seed"
        or manifest.get("max_tokens") != MAX_COMPLETION_TOKENS
        or manifest.get("max_turns") != MAX_TURNS
        or manifest.get("workers") != WORKERS
        or manifest.get("response_cache_enabled") is not False
        or set(completion) != COMPLETION_FIELDS
        or completion.get("format") != COMPLETION_FORMAT
        or completion.get("run_manifest") != str(manifest_path)
        or completion.get("results_jsonl") != str(ledger_path)
        or completion.get("protocol_sha256") != manifest.get("protocol_sha256")
        or completion.get("task_denominator") != TASK_COUNT
        or completion.get("testcase_denominator") != TESTCASE_COUNT
        or completion.get("completed_cases") != TESTCASE_COUNT
        or completion.get("results_jsonl_sha256")
        != hashlib.sha256(ledger_bytes).hexdigest()
        or len(rows) != TESTCASE_COUNT
        or ledger_bytes != b"".join(line + b"\n" for line in lines)
    ):
        raise ValueError("completed Soft/Hard run identity differs")
    validated = [
        _validate_case_row(
            row,
            case=case,
            task=task,
            protocol_sha256=manifest["protocol_sha256"],
            output_dir=output_dir,
        )
        for row, (task, case) in zip(rows, case_plan, strict=True)
    ]
    if (
        completion.get("agent_success_cases")
        != sum(row["agent_success"] for row in validated)
        or completion.get("agent_completed_cases")
        != sum(row["agent_completed"] for row in validated)
        or completion.get("output_preserved_cases")
        != sum(row["output_preserved"] for row in validated)
    ):
        raise ValueError("completed Soft/Hard run summary differs")
    return manifest, completion, validated, ledger_bytes


def _run_locked(
    *,
    source_dataset_path: Path,
    prepared_data_path: Path,
    input_manifest_path: Path,
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
    input_manifest_path = input_manifest_path.expanduser().resolve()
    snapshot_manifest_path = snapshot_manifest_path.expanduser().resolve()
    state_db_path = state_db_path.expanduser().resolve()
    bundle_dir = bundle_dir.expanduser().resolve()
    run_dir = run_dir.expanduser().absolute()
    population = load_prepared_retrieval_population(
        data_path=prepared_data_path,
        manifest_path=input_manifest_path,
    )
    verified_bundle = verify_bundle(
        source_dataset_path=source_dataset_path,
        prepared_data_path=prepared_data_path,
        retrieval_manifest_path=input_manifest_path,
        snapshot_manifest_path=snapshot_manifest_path,
        state_db_path=state_db_path,
        output_dir=bundle_dir,
    )
    bundle_manifest = verified_bundle.manifest
    provider = SoftHardExperienceProvider(verified_bundle)
    expected_manifest = _manifest(
        prepared_data_path=prepared_data_path,
        input_manifest_path=input_manifest_path,
        population=population,
        source_dataset_path=source_dataset_path,
        bundle_dir=bundle_dir,
        bundle_manifest=bundle_manifest,
        provider=provider,
        snapshot_manifest_path=snapshot_manifest_path,
        state_db_path=state_db_path,
        base_url=base_url,
    )
    if dry_run:
        return {"manifest": expected_manifest, "completed_cases": 0}

    manifest_path = run_dir / "outputs/run_manifest.json"
    if run_dir.exists() or run_dir.is_symlink():
        if (
            not resume
            or run_dir.is_symlink()
            or manifest_path.is_symlink()
            or not manifest_path.is_file()
        ):
            raise FileExistsError(
                "Soft/Hard run directory must be fresh, or use --resume"
            )
        stored = _read_json(manifest_path)
        expected_body = {
            key: value
            for key, value in expected_manifest.items()
            if key not in {"protocol_sha256", "created_at"}
        }
        stored_body = {
            key: value
            for key, value in stored.items()
            if key not in {"protocol_sha256", "created_at"}
        }
        if (
            stored_body != expected_body
            or stored.get("protocol_sha256")
            != hashlib.sha256(canonical_json_bytes(stored_body)).hexdigest()
        ):
            raise ValueError("resume protocol identity differs")
        manifest = stored
    else:
        run_dir.mkdir(parents=True)
        (run_dir / "outputs").mkdir()
        (run_dir / "logs").mkdir()
        (run_dir / "case_results").mkdir()
        _write_json_atomic(manifest_path, expected_manifest)
        manifest = expected_manifest

    output_dir = _require_run_directory(run_dir, "outputs")
    log_dir = _require_run_directory(run_dir, "logs")
    case_results_dir = _require_run_directory(run_dir, "case_results")

    completion_path = run_dir / "results.json"
    ledger_path = run_dir / "results.jsonl"
    if completion_path.is_file() and ledger_path.is_file():
        _stored_manifest, completion, _rows, _ledger = verify_completed_run(
            run_dir=run_dir, population=population
        )
        return completion

    os.environ["SB_RUN_PROTOCOL_SHA256"] = manifest["protocol_sha256"]
    os.environ["REACT_AGENT_USAGE_LOG"] = str(run_dir / "react_usage.jsonl")
    os.environ["REACT_AGENT_RUNTIME_EVENT_LOG"] = str(
        run_dir / "runtime_events.jsonl"
    )
    instances = _task_instances(population)
    case_plan = [
        (task, case)
        for task in population["tasks"]
        for case in task["cases"]
    ]
    completed, pending = _load_completed_case_results(
        case_plan=case_plan,
        case_results_dir=case_results_dir,
        protocol_sha256=manifest["protocol_sha256"],
        output_dir=output_dir,
    )
    print(
        f"Soft/Hard cases: completed={len(completed)} pending={len(pending)} "
        f"total={len(case_plan)} workers={WORKERS}",
        flush=True,
    )
    api_key = os.getenv(API_KEY_ENV, "")
    if pending:
        if not api_key:
            raise RuntimeError(f"missing API key: set {API_KEY_ENV}")
        preflight_client = _client(api_key=api_key, base_url=base_url)
        available_models = {
            str(row.id) for row in preflight_client._client.models.list().data
        }
        if MODEL not in available_models:
            raise RuntimeError(
                f"configured model is unavailable at endpoint: {MODEL}"
            )

    working_directory = tempfile.TemporaryDirectory(prefix="degs_soft_hard_")
    working_dir = Path(working_directory.name)
    thread_state = threading.local()
    transport_tracker = _GenerationTransportTracker()

    def process(task: Mapping[str, Any], case: Mapping[str, Any]) -> dict[str, Any]:
        if not hasattr(thread_state, "runner"):
            agent = SoftHardExperienceAgent(
                client=_TrackedAgentClient(
                    _client(api_key=api_key, base_url=base_url),
                    transport_tracker,
                ),
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
            instances[task["task_id"]],
            case["input_file"],
            runtime_instance_id=case["case_id"],
            retrieval_id=case["case_id"],
        )
        payload = asdict(result)
        output_path = output_dir / task["spreadsheet_path"] / case["output_file"]
        output_sha256, output_size = _output_identity(output_path)
        return {
            "format": RESULT_FORMAT,
            "protocol_sha256": manifest["protocol_sha256"],
            "case_id": case["case_id"],
            "task_id": task["task_id"],
            "query_index": task["query_index"],
            "input_file": case["input_file"],
            "output_file": case["output_file"],
            "output_path": str(output_path),
            "output_sha256": output_sha256,
            "output_size": output_size,
            "agent_success": bool(payload["success"]),
            "agent_completed": bool(payload["agent_completed"]),
            "output_preserved": bool(payload["output_preserved"]),
            "turns": int(payload["turns"]),
            "answer": str(payload["agent_answer"]),
            "error": str(payload["error"]),
            "failure_kind": str(payload.get("failure_kind", "")),
        }

    started_at = datetime.now(timezone.utc)
    try:
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            processed = 0
            for start in range(0, len(pending), WORKERS):
                wave = pending[start : start + WORKERS]
                transport_wave = transport_tracker.begin_wave()
                futures = {
                    pool.submit(process, task, case): (task, case)
                    for task, case in wave
                }
                wave_rows: list[
                    tuple[Mapping[str, Any], Mapping[str, Any], dict[str, Any]]
                ] = []
                for future in as_completed(futures):
                    task, case = futures[future]
                    try:
                        row = future.result()
                    except Exception as exc:
                        transport_tracker.signal_fatal(
                            "worker failed outside the item-local result boundary"
                        )
                        for sibling in futures:
                            if sibling is not future:
                                sibling.cancel()
                        raise RuntimeError(
                            "worker failed outside the item-local result boundary; "
                            "current wave remains pending"
                        ) from exc
                    try:
                        transport_tracker.raise_if_fatal(transport_wave)
                    except RuntimeError:
                        for sibling in futures:
                            if sibling is not future:
                                sibling.cancel()
                        raise
                    if row.get("failure_kind") in {
                        "agent_exception",
                        "runner_exception",
                        "worker_internal_error",
                    } and not transport_tracker.has_item_local_transport_failure(
                        case["case_id"], transport_wave
                    ):
                        transport_tracker.signal_fatal(
                            "unknown Agent/runner failure"
                        )
                        for sibling in futures:
                            if sibling is not future:
                                sibling.cancel()
                        raise RuntimeError(
                            "unknown Agent/runner failure; current wave remains pending"
                        )
                    wave_rows.append((task, case, row))
                transport_tracker.raise_if_systemic(transport_wave)
                for task, case, row in wave_rows:
                    row = _validate_case_row(
                        row,
                        case=case,
                        task=task,
                        protocol_sha256=manifest["protocol_sha256"],
                        output_dir=output_dir,
                    )
                    _write_json_atomic(
                        _case_result_path(case_results_dir, case["case_id"]), row
                    )
                    completed[case["case_id"]] = row
                    processed += 1
                    if processed % 25 == 0 or processed == len(pending):
                        print(
                            f"completed {len(completed)}/{TESTCASE_COUNT} cases",
                            flush=True,
                        )
    finally:
        working_directory.cleanup()

    if len(completed) != TESTCASE_COUNT:
        raise RuntimeError("Soft/Hard run ended before every case was recorded")
    ordered = [completed[case["case_id"]] for _task, case in case_plan]
    results_bytes = b"".join(canonical_json_bytes(row) + b"\n" for row in ordered)
    _write_bytes_atomic(ledger_path, results_bytes)
    payload = {
        "format": COMPLETION_FORMAT,
        "run_manifest": str(manifest_path),
        "protocol_sha256": manifest["protocol_sha256"],
        "started_at": started_at.isoformat(),
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "task_denominator": TASK_COUNT,
        "testcase_denominator": TESTCASE_COUNT,
        "completed_cases": len(ordered),
        "agent_success_cases": sum(row["agent_success"] for row in ordered),
        "agent_completed_cases": sum(row["agent_completed"] for row in ordered),
        "output_preserved_cases": sum(row["output_preserved"] for row in ordered),
        "results_jsonl": str(ledger_path),
        "results_jsonl_sha256": hashlib.sha256(results_bytes).hexdigest(),
    }
    _write_json_atomic(completion_path, payload)
    return payload


def run(
    *,
    source_dataset_path: Path,
    prepared_data_path: Path,
    input_manifest_path: Path,
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
            raise RuntimeError("another Soft/Hard runner owns this run") from exc
        return _run_locked(
            source_dataset_path=source_dataset_path,
            prepared_data_path=prepared_data_path,
            input_manifest_path=input_manifest_path,
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
    parser = argparse.ArgumentParser(
        description="Run DEGS on SpreadsheetBench Soft/Hard."
    )
    parser.add_argument("--source-dataset-path", type=Path, required=True)
    parser.add_argument("--prepared-data-path", type=Path, required=True)
    parser.add_argument("--input-manifest-path", type=Path, required=True)
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
        input_manifest_path=args.input_manifest_path,
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


__all__ = [
    "COMPLETION_FORMAT",
    "FORMAT",
    "RESULT_FORMAT",
    "run",
    "verify_completed_run",
]


if __name__ == "__main__":
    raise SystemExit(main())
