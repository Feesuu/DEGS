from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from typing import Any, Callable, Mapping, Sequence

from react_agent.models import OpenAIClient
from sb_adapter.transport import validate_service_url

from .core import TRAIN_INSTRUCTION_COUNT
from .source_replay import (
    SOURCE_REPLAY_PATCH_KIND,
    SOURCE_REPLAY_PATCH_PROMPT_SHA256,
    SOURCE_REPLAY_PROTOCOL_FORMAT,
    SOURCE_REPLAY_BASH_TIMEOUT_SECONDS,
    SOURCE_REPLAY_EXECUTOR_RETRY_WAITS,
    SOURCE_REPLAY_MAX_TURNS,
    SOURCE_REPLAY_MODEL,
    SOURCE_REPLAY_PATCH_MAX_TOKENS,
    SOURCE_REPLAY_TIMEOUT_SECONDS,
    SOURCE_REPLAY_TASK_WORKERS,
    ReplayExecution,
    SourceReplayController,
)
from .transport import _task_id
from .validated_repair import (
    OpenAIJsonObjectLLM,
    REPAIR_SOURCE_MODEL,
    REPAIR_SOURCE_TIMEOUT_SECONDS,
    PRODUCER_RUNTIME_TIMEOUT_RETRIES,
    PRODUCER_TRANSPORT_RETRY_WAITS,
    _source_generation_config,
    _write_json_output,
)


_CONTEXT_OVERFLOW_PROBE = Path(__file__).with_name("replay_overflow_probe.py")
_UPSTREAM_RUNTIME_PATHS = (
    "src/sb_adapter/run_benchmark.py",
    "src/sb_adapter/evaluate.py",
    "src/sb_adapter/export_trajectories.py",
    "src/sb_adapter/log_parser.py",
    "src/sb_adapter/schemas.py",
    "src/sb_adapter/transport.py",
    "src/sb_adapter/experience.py",
    "src/sb_adapter/spreadsheetbench_support.py",
    "src/spreadsheet_agent/agents/source_replay_patch_agent.py",
    "src/spreadsheet_agent/agents/cli_only_agent.py",
    "src/spreadsheet_agent/agents/base.py",
    "src/spreadsheet_agent/system_prompt/source_replay_patch_full_system_v1.txt",
    "src/spreadsheet_agent/system_prompts.py",
    "src/spreadsheet_agent/tools/bash.py",
    "src/spreadsheet_agent/output_feedback.py",
    "src/spreadsheet_agent/runner.py",
    "src/react_agent/agent.py",
    "src/react_agent/converter.py",
    "src/react_agent/models.py",
    "src/react_agent/tools.py",
)
def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _strict_json(path: Path) -> Any:
    def without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=without_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {token}")
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"artifact is not readable strict JSON: {path}") from exc


def _write_json_value(path: Path, payload: Any) -> None:
    output = path.expanduser().absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )


@dataclass(frozen=True)
class ReplayRuntime:
    upstream_root: Path
    dataset_path: Path
    base_url: str
    api_key_env: str
    python_executable: str
    model: str = SOURCE_REPLAY_MODEL
    max_turns: int = SOURCE_REPLAY_MAX_TURNS
    max_completion_tokens: int = SOURCE_REPLAY_PATCH_MAX_TOKENS
    bash_timeout: int = SOURCE_REPLAY_BASH_TIMEOUT_SECONDS
    llm_timeout: float = SOURCE_REPLAY_TIMEOUT_SECONDS
    retry_waits: tuple[int, ...] = SOURCE_REPLAY_EXECUTOR_RETRY_WAITS

    def __post_init__(self) -> None:
        if (
            self.model != SOURCE_REPLAY_MODEL
            or self.max_turns != SOURCE_REPLAY_MAX_TURNS
            or self.max_completion_tokens != SOURCE_REPLAY_PATCH_MAX_TOKENS
            or self.bash_timeout != SOURCE_REPLAY_BASH_TIMEOUT_SECONDS
            or self.llm_timeout != SOURCE_REPLAY_TIMEOUT_SECONDS
            or self.retry_waits != SOURCE_REPLAY_EXECUTOR_RETRY_WAITS
        ):
            raise ValueError("source replay runtime differs from the fixed protocol")
        if not isinstance(self.api_key_env, str) or not self.api_key_env:
            raise ValueError("source replay API key environment name differs")
        if not isinstance(self.python_executable, str) or not self.python_executable:
            raise ValueError("source replay Python executable differs")


class SubprocessReplayExecutor:
    """Execute fresh task attempts with the bundled benchmark harness."""

    def __init__(self, runtime: ReplayRuntime) -> None:
        if not isinstance(runtime, ReplayRuntime):
            raise TypeError("replay runtime differs")
        self.runtime = runtime
        root = runtime.upstream_root.expanduser().absolute()
        dataset_path = runtime.dataset_path.expanduser().absolute()
        for relative in _UPSTREAM_RUNTIME_PATHS:
            if not (root / relative).is_file():
                raise ValueError(f"replay runtime file is missing: {relative}")
        dataset = _strict_json(dataset_path / "dataset.json")
        if type(dataset) is not list or len(dataset) < TRAIN_INSTRUCTION_COUNT:
            raise ValueError("SpreadsheetBench dataset cannot provide train[0,200)")
        try:
            task_ids = [
                _task_id(row.get("id") if isinstance(row, Mapping) else None)
                for row in dataset
            ]
        except ValueError as exc:
            raise ValueError("SpreadsheetBench task identity differs") from exc
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("SpreadsheetBench dataset contains duplicate task IDs")
        self.root = root
        self.dataset_path = dataset_path
        self.task_indices = {task_id: index for index, task_id in enumerate(task_ids)}
        self.train_task_ids = tuple(task_ids[:TRAIN_INSTRUCTION_COUNT])
        self._protocol_identity = {
            "format": "degs_fresh_replay_executor_v1",
            "dataset_json_sha256": _sha256_file(dataset_path / "dataset.json"),
            "model": runtime.model,
            "base_url": runtime.base_url,
            "max_turns": runtime.max_turns,
            "max_completion_tokens": runtime.max_completion_tokens,
            "bash_timeout": runtime.bash_timeout,
            "llm_timeout": runtime.llm_timeout,
            "retry_waits": list(runtime.retry_waits),
            "workers": 1,
            "outer_task_workers": SOURCE_REPLAY_TASK_WORKERS,
            "temperature": 0,
            "thinking": False,
            "fresh_input_each_attempt": True,
            "context_overflow_reporting": "machine_readable_marker_v1",
            "observation_policy": "full_no_truncation",
        }

    @property
    def protocol_identity(self) -> Mapping[str, Any]:
        return self._protocol_identity

    def execute(
        self,
        *,
        failed_record: Mapping[str, Any],
        rendered_patch: str,
        patch_id: str,
        attempt_index: int,
        attempt_dir: Path,
        source_replay_protocol_sha256: str,
    ) -> ReplayExecution:
        task_id = failed_record.get("task_id")
        if not isinstance(task_id, str) or task_id not in self.task_indices:
            raise ValueError("source task is absent from the fixed dataset")
        dataset_index = self.task_indices[task_id]
        if not 0 <= dataset_index < TRAIN_INSTRUCTION_COUNT:
            raise ValueError("source replay is restricted to train[0,200)")
        runner_root = attempt_dir / "runner"
        output_dir = runner_root / "outputs"
        log_dir = runner_root / "logs"
        results_path = runner_root / "results.json"
        experience_path = attempt_dir / "patch_experience.jsonl"
        replay_trajectory_id = (
            f"{task_id}::source_replay::{attempt_index}::{patch_id.rsplit('_', 1)[-1]}"
        )
        experience_row = {
            "instance_id": task_id,
            "experience": rendered_patch,
            "metadata": {
                "format": "degs_source_replay_patch_v1",
                "task_id": task_id,
                "patch_id": patch_id,
                "attempt_index": attempt_index,
                "degs_source_replay_protocol_sha256": source_replay_protocol_sha256,
            },
        }
        descriptor = os.open(
            experience_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
        try:
            os.write(
                descriptor,
                (json.dumps(experience_row, ensure_ascii=False) + "\n").encode("utf-8"),
            )
        finally:
            os.close(descriptor)

        context_overflow_marker = attempt_dir / "context_length_exceeded.marker"
        self._run_command(
            [
                self.runtime.python_executable,
                str(_CONTEXT_OVERFLOW_PROBE),
                "sb_adapter.run_benchmark",
                "--data-path",
                str(self.dataset_path),
                "--output-dir",
                str(output_dir),
                "--working-dir",
                str(runner_root / "working"),
                "--log-dir",
                str(log_dir),
                "--results-file",
                str(results_path),
                "--usage-log",
                str(runner_root / "usage.jsonl"),
                "--start-idx",
                str(dataset_index),
                "--end-idx",
                str(dataset_index + 1),
                "--workers",
                "1",
                "--max-turns",
                str(self.runtime.max_turns),
                "--max-tokens",
                str(self.runtime.max_completion_tokens),
                "--bash-timeout",
                str(self.runtime.bash_timeout),
                "--model",
                self.runtime.model,
                "--base-url",
                self.runtime.base_url,
                "--api-key-env",
                self.runtime.api_key_env,
                "--temperature",
                "0",
                "--thinking",
                "false",
                "--llm-timeout",
                str(self.runtime.llm_timeout),
                "--retry-waits",
                ",".join(map(str, self.runtime.retry_waits)),
                "--agent",
                "source_replay_patch",
                "--experience-file",
                str(experience_path),
                "--require-experience-for-all",
            ],
            attempt_dir / "runner_subprocess.json",
            context_overflow_marker=context_overflow_marker,
        )
        if context_overflow_marker.is_file():
            return ReplayExecution(
                task_id=task_id,
                trajectory_id=replay_trajectory_id,
                success=False,
                verifier_score=0.0,
                verifier_feedback="CONTEXT_LENGTH_EXCEEDED",
                trajectory={
                    "task_id": task_id,
                    "trajectory_id": replay_trajectory_id,
                    "instruction": failed_record.get("instruction", ""),
                    "success": False,
                    "steps": [],
                    "final_response": "CONTEXT_LENGTH_EXCEEDED",
                },
                error="CONTEXT_LENGTH_EXCEEDED",
            )
        evaluation_path = attempt_dir / "eval.json"
        manifest = _strict_json(output_dir / "run_manifest.json")
        if (
            type(manifest) is not dict
            or manifest.get("model") != self.runtime.model
            or manifest.get("max_tokens") != self.runtime.max_completion_tokens
            or manifest.get("max_turns") != self.runtime.max_turns
            or manifest.get("temperature") != 0.0
            or manifest.get("thinking") != "false"
            or manifest.get("start_idx") != dataset_index
            or manifest.get("end_idx") != dataset_index + 1
            or manifest.get("workers") != 1
        ):
            raise ValueError("source replay child run protocol differs")
        self._run_command(
            [
                self.runtime.python_executable,
                "-m",
                "sb_adapter.evaluate",
                "--data_path",
                str(self.dataset_path),
                "--output_dir",
                str(output_dir),
                "--run-manifest",
                str(output_dir / "run_manifest.json"),
                "--results_file",
                str(evaluation_path),
                "--recalc_dir",
                str(attempt_dir / "recalc"),
                "--expected-base-url",
                self.runtime.base_url,
                "--expected-workers",
                "1",
                "--start_idx",
                str(dataset_index),
                "--end_idx",
                str(dataset_index + 1),
            ],
            attempt_dir / "evaluate_subprocess.json",
        )
        exported_path = attempt_dir / "replay_records.json"
        self._run_command(
            [
                self.runtime.python_executable,
                "-m",
                "sb_adapter.export_trajectories",
                "--data-path",
                str(self.dataset_path),
                "--log-dir",
                str(log_dir),
                "--eval-file",
                str(evaluation_path),
                "--run-manifest",
                str(output_dir / "run_manifest.json"),
                "--output",
                str(exported_path),
                "--start-idx",
                str(dataset_index),
                "--end-idx",
                str(dataset_index + 1),
            ],
            attempt_dir / "export_subprocess.json",
        )
        records = _strict_json(exported_path)
        if (
            type(records) is not list
            or len(records) != 1
            or type(records[0]) is not dict
            or records[0].get("task_id") != task_id
        ):
            raise ValueError("source replay exported the wrong trajectory")
        record = records[0]
        success = record.get("success") is True
        score = record.get("verifier_score")
        score = float(score) if type(score) in (int, float) else 0.0
        feedback = record.get("verifier_feedback")
        feedback = feedback if isinstance(feedback, str) else ""
        record["trajectory_id"] = replay_trajectory_id
        record["success"] = success
        record["verifier_score"] = score
        record["verifier_feedback"] = feedback
        executor_protocol_sha256 = (
            manifest.get("protocol_sha256") if isinstance(manifest, Mapping) else None
        )
        return ReplayExecution(
            task_id=task_id,
            trajectory_id=replay_trajectory_id,
            success=success,
            verifier_score=score,
            verifier_feedback=feedback,
            trajectory=record,
            error=None if success else feedback,
            executor_run_protocol_sha256=(
                str(executor_protocol_sha256)
                if isinstance(executor_protocol_sha256, str)
                else None
            ),
        )

    def _run_command(
        self,
        command: list[str],
        audit_path: Path,
        *,
        context_overflow_marker: Path | None = None,
    ) -> str:
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(self.root / "src") + (
            os.pathsep + environment["PYTHONPATH"]
            if environment.get("PYTHONPATH")
            else ""
        )
        if context_overflow_marker is not None:
            environment["DEGS_CONTEXT_OVERFLOW_MARKER"] = str(
                context_overflow_marker
            )
        completed = subprocess.run(
            command,
            cwd=self.root,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        _write_json_output(
            audit_path,
            {
                "command": command,
                "returncode": completed.returncode,
                "output": completed.stdout,
            },
        )
        if completed.returncode and not (
            context_overflow_marker is not None
            and context_overflow_marker.is_file()
        ):
            raise RuntimeError(
                f"subprocess failed with exit {completed.returncode}: {completed.stdout[-4000:]}"
            )
        return completed.stdout


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Regenerate no-truncation repair patches and fresh replay outcomes."
    )
    parser.add_argument("--original-records", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--outcomes-output", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key-env", default="DEGS_API_KEY")
    parser.add_argument("--python-executable", default=sys.executable)
    return parser


def _recover_in_train_order(
    failed_records: Sequence[Mapping[str, Any]],
    recover: Callable[[Mapping[str, Any]], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Run independent source tasks concurrently and retain train order."""

    with ThreadPoolExecutor(
        max_workers=SOURCE_REPLAY_TASK_WORKERS,
        thread_name_prefix="degs-source-replay",
    ) as pool:
        return list(pool.map(recover, failed_records))


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    validate_service_url(args.base_url)
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise ValueError(f"missing generation API key: set {args.api_key_env}")
    originals = _strict_json(args.original_records.expanduser().absolute())
    if type(originals) is not list or len(originals) != TRAIN_INSTRUCTION_COUNT:
        raise ValueError("source replay requires exactly train[0,200) original records")
    if any(
        type(record) is not dict
        or not isinstance(record.get("task_id"), str)
        or not isinstance(record.get("trajectory_id"), str)
        or type(record.get("success")) is not bool
        for record in originals
    ):
        raise ValueError("original train trajectory identity differs")
    runtime = ReplayRuntime(
        upstream_root=args.upstream_root,
        dataset_path=args.data_path,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        python_executable=args.python_executable,
    )
    executor = SubprocessReplayExecutor(runtime)
    task_ids = [row.get("task_id") if isinstance(row, Mapping) else None for row in originals]
    if tuple(task_ids) != executor.train_task_ids:
        raise ValueError("original records do not match ordered train[0,200)")
    failed_records = [
        record
        for record in originals
        if isinstance(record, Mapping) and record.get("success") is False
    ]
    thread_state = threading.local()

    def recover(record: Mapping[str, Any]) -> dict[str, Any]:
        controller = getattr(thread_state, "controller", None)
        if controller is None:
            client = OpenAIClient(
                model=REPAIR_SOURCE_MODEL,
                api_key=api_key,
                base_url=args.base_url,
                generation_config=_source_generation_config(
                    SOURCE_REPLAY_PATCH_MAX_TOKENS
                ),
                retry_times=PRODUCER_TRANSPORT_RETRY_WAITS,
                runtime_timeout_retries=PRODUCER_RUNTIME_TIMEOUT_RETRIES,
                timeout=REPAIR_SOURCE_TIMEOUT_SECONDS,
                trust_env=False,
            )
            patch_llm = OpenAIJsonObjectLLM(
                client,
                request_kind=SOURCE_REPLAY_PATCH_KIND,
                source_protocol_format=SOURCE_REPLAY_PROTOCOL_FORMAT,
                prompt_sha256=SOURCE_REPLAY_PATCH_PROMPT_SHA256,
                response_schema_name="degs_source_replay_patch_v2",
                expected_max_tokens=SOURCE_REPLAY_PATCH_MAX_TOKENS,
            )
            controller = SourceReplayController(
                patch_llm=patch_llm,
                attempt_executor=executor,
                run_root=args.run_root,
            )
            thread_state.controller = controller
        return controller.recover(record)

    outcomes = _recover_in_train_order(failed_records, recover)
    if len(outcomes) != sum(
        isinstance(record, Mapping) and record.get("success") is False
        for record in originals
    ):
        raise ValueError("original record success identity differs")
    _write_json_value(args.outcomes_output, outcomes)
    print(
        json.dumps(
            {
                "outcome_count": len(outcomes),
                "validated_success_count": sum(
                    row["status"] == "REPLAY_VALIDATED_SUCCESS" for row in outcomes
                ),
                "outcomes_output": str(args.outcomes_output.expanduser().absolute()),
            },
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "ReplayRuntime",
    "SubprocessReplayExecutor",
    "SOURCE_REPLAY_TASK_WORKERS",
]


if __name__ == "__main__":
    raise SystemExit(main())
