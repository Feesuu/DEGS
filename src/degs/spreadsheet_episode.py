from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import threading
from typing import Any, Mapping, Sequence

from react_agent.models import OpenAIClient
from spreadsheet_agent.runner import BenchmarkInstance, SpreadsheetBenchRunner

from .agent import DEGSExperienceAgent
from .contextual_binding import ExperienceExpectation
from .contextual_retrieval import ContextualRetrieval
from .core import canonical_json_bytes
from .dataset import _load_train_harness_records
from .dynamic_train import AGENT_WORKERS, PreparedEpisode
from .episode_evidence import (
    EpisodeEvidence,
    EpisodeOutcome,
    EvidenceItem,
    episode_evidence_from_payload,
)
from .provider import EIRGuidanceProvider
from .source_replay import (
    SOURCE_REPLAY_PATCH_KIND,
    SOURCE_REPLAY_PATCH_PROMPT_SHA256,
    SOURCE_REPLAY_PATCH_MAX_TOKENS,
    SOURCE_REPLAY_PROTOCOL_FORMAT,
    SourceReplayController,
    validate_source_replay_outcome_protocol,
)
from .source_replay_executor import (
    ReplayRuntime,
    SubprocessReplayExecutor,
    _recover_in_train_order,
)
from .target_context import (
    TargetContextUnavailable,
    build_target_evidence_card,
    unavailable_target_evidence_card,
)
from .validated_repair import (
    OpenAIJsonObjectLLM,
    PRODUCER_RUNTIME_TIMEOUT_RETRIES,
    PRODUCER_TRANSPORT_RETRY_WAITS,
    REPAIR_SOURCE_TIMEOUT_SECONDS,
    _source_generation_config,
)


SPREADSHEET_EPISODE_FORMAT = "degs_spreadsheet_episode_adapter_v1"
MAX_TURNS = 30
MAX_COMPLETION_TOKENS = 32_000
BASH_TIMEOUT_S = 120
LLM_TIMEOUT_S = 600.0
RETRY_WAITS_S = (5, 10, 30)


class _ModelAwareReplayExecutor(SubprocessReplayExecutor):
    """Keep replay evaluation identical while accepting the selected 9B/27B model."""

    def _run_command(
        self,
        command: list[str],
        audit_path: Path,
        *,
        context_overflow_marker: Path | None = None,
    ) -> str:
        child = list(command)
        if child[:3] == [self.runtime.python_executable, "-m", "sb_adapter.evaluate"]:
            child = [
                self.runtime.python_executable,
                str(Path(__file__).with_name("fresh_train_evaluate.py")),
                "--expected-model",
                self.runtime.model,
                *child[3:],
                "--expected-max-tokens",
                str(self.runtime.max_completion_tokens),
            ]
        return super()._run_command(
            child,
            audit_path,
            context_overflow_marker=context_overflow_marker,
        )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(canonical_json_bytes(value) + b"\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in ("openai", "openpyxl", "tqdm", "transformers"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "missing"
    return versions


def _agent_client(*, model: str, api_key: str, base_url: str) -> OpenAIClient:
    return OpenAIClient(
        model=model,
        api_key=api_key,
        base_url=base_url,
        generation_config={
            "temperature": 0.0,
            "max_tokens": MAX_COMPLETION_TOKENS,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        },
        retry_times=RETRY_WAITS_S,
        runtime_timeout_retries=1,
        timeout=LLM_TIMEOUT_S,
        trust_env=False,
    )


def _trace_items(record: Mapping[str, Any], *, prefix: str) -> tuple[EvidenceItem, ...]:
    items: list[EvidenceItem] = []
    steps = record.get("steps")
    if type(steps) is list:
        for ordinal, step in enumerate(steps):
            if not isinstance(step, Mapping):
                continue
            step_id = step.get("step_id")
            stable = step_id if type(step_id) is int and step_id >= 0 else ordinal
            action = step.get("action")
            observation = step.get("observation")
            if type(action) is str and action:
                items.append(
                    EvidenceItem(f"trace:{prefix}:turn:{stable}:action", "action", action)
                )
            if type(observation) is str and observation:
                items.append(
                    EvidenceItem(
                        f"trace:{prefix}:turn:{stable}:observation",
                        "observation",
                        observation,
                    )
                )
    final_response = record.get("final_response")
    if type(final_response) is str and final_response:
        items.append(
            EvidenceItem(f"trace:{prefix}:final", "final_response", final_response)
        )
    return tuple(items)


class SpreadsheetEpisodeAdapter:
    """SpreadsheetBench execution/verifier/replay boundary for dynamic EIR train."""

    def __init__(
        self,
        *,
        dataset_path: Path | str,
        run_dir: Path | str,
        runtime_root: Path | str,
        generation_base_url: str,
        api_key: str,
        model: str,
        dataset_contract_id: str,
    ) -> None:
        self.dataset_path = Path(dataset_path).expanduser().resolve()
        self.run_dir = Path(run_dir).expanduser().absolute()
        self.runtime_root = Path(runtime_root).expanduser().resolve()
        self.generation_base_url = generation_base_url
        self.api_key = api_key
        self.model = model
        self.dataset_contract_id = dataset_contract_id
        if model not in {"Qwen3.5-9B-AWQ", "Qwen3.5-27B-AWQ"}:
            raise ValueError("Spreadsheet episode model differs")
        if not (self.dataset_path / "dataset.json").is_file():
            raise FileNotFoundError("SpreadsheetBench dataset is unavailable")

    def load_tasks(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(_load_train_harness_records(self.dataset_path / "dataset.json"))

    async def prepare(self, task: Any) -> PreparedEpisode:
        if not isinstance(task, Mapping):
            raise TypeError("Spreadsheet task differs")
        try:
            card = await asyncio.to_thread(
                build_target_evidence_card,
                dataset_path=self.dataset_path / "dataset.json",
                spreadsheet_path=str(task["spreadsheet_path"]),
                answer_position=str(task["answer_position"]),
                instruction=str(task["instruction"]),
            )
        except TargetContextUnavailable as exc:
            card = unavailable_target_evidence_card(exc)
        observations = tuple(
            EvidenceItem(
                f"context:{row['evidence_id']}",
                str(row["kind"]),
                json.loads(str(row["content"])),
            )
            for row in card.observations
        )
        return PreparedEpisode(
            int(task["train_index"]),
            str(task["task_id"]),
            str(task["instruction"]),
            observations,
            dict(task),
        )

    async def execute_batch(
        self,
        *,
        prepared: Sequence[PreparedEpisode],
        read_snapshot_id: str,
        retrievals: Sequence[ContextualRetrieval],
        expectations: Sequence[Sequence[ExperienceExpectation]],
        guidance: Sequence[str],
    ) -> Sequence[EpisodeEvidence]:
        return await asyncio.to_thread(
            self._execute_batch_sync,
            tuple(prepared),
            read_snapshot_id,
            tuple(retrievals),
            tuple(tuple(row) for row in expectations),
            tuple(guidance),
        )

    def _execute_batch_sync(
        self,
        prepared: tuple[PreparedEpisode, ...],
        read_snapshot_id: str,
        retrievals: tuple[ContextualRetrieval, ...],
        expectations: tuple[tuple[ExperienceExpectation, ...], ...],
        guidance: tuple[str, ...],
    ) -> tuple[EpisodeEvidence, ...]:
        if not (
            len(prepared) == len(retrievals) == len(expectations) == len(guidance)
        ):
            raise ValueError("Spreadsheet episode batch inputs differ")
        batch_index = prepared[0].train_index // 8
        batch_dir = self.run_dir / "batches" / f"batch_{batch_index:02d}" / "runtime"
        completed_path = batch_dir / "episodes.json"
        if completed_path.is_file():
            stored = json.loads(completed_path.read_text(encoding="utf-8"))
            if type(stored) is not list:
                raise ValueError("stored Spreadsheet episode batch differs")
            completed = tuple(episode_evidence_from_payload(row) for row in stored)
            self._validate_completed_episodes(
                completed=completed,
                prepared=prepared,
                read_snapshot_id=read_snapshot_id,
                retrievals=retrievals,
                expectations=expectations,
            )
            return completed
        output_dir = batch_dir / "outputs"
        log_dir = batch_dir / "logs"
        working_dir = batch_dir / "working"
        output_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(exist_ok=True)
        working_dir.mkdir(exist_ok=True)
        retrieval_hashes = {
            row.task_id: hashlib.sha256(
                canonical_json_bytes(retrievals[index].to_dict())
            ).hexdigest()
            for index, row in enumerate(prepared)
        }
        provider = EIRGuidanceProvider(
            {row.task_id: guidance[index] for index, row in enumerate(prepared)},
            snapshot_id=read_snapshot_id,
            retrieval_sha256_by_id=retrieval_hashes,
        )
        instances = [
            BenchmarkInstance(
                id=row.task_id,
                instruction=row.query_text,
                spreadsheet_path=str(row.task["spreadsheet_path"]),
                instruction_type=str(row.task["instruction_type"]),
                answer_position=str(row.task["answer_position"]),
                metadata={},
            )
            for row in prepared
        ]
        start_idx = prepared[0].train_index
        end_idx = prepared[-1].train_index + 1
        manifest = self._run_manifest(
            provider=provider,
            instance_ids=[row.task_id for row in prepared],
            start_idx=start_idx,
            end_idx=end_idx,
        )
        manifest_path = output_dir / "run_manifest.json"
        if manifest_path.is_file():
            stored_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if {
                key: value
                for key, value in stored_manifest.items()
                if key != "created_at"
            } != {key: value for key, value in manifest.items() if key != "created_at"}:
                raise ValueError("stored Spreadsheet batch protocol differs")
            manifest = stored_manifest
        else:
            _write_json(manifest_path, manifest)
        os.environ["SB_RUN_PROTOCOL_SHA256"] = manifest["protocol_sha256"]
        os.environ["REACT_AGENT_USAGE_LOG"] = str(batch_dir / "usage.jsonl")
        os.environ["REACT_AGENT_RUNTIME_EVENT_LOG"] = str(
            batch_dir / "runtime_events.jsonl"
        )
        thread_state = threading.local()

        def process(instance: BenchmarkInstance) -> Any:
            if not hasattr(thread_state, "runner"):
                agent = DEGSExperienceAgent(
                    client=_agent_client(
                        model=self.model,
                        api_key=self.api_key,
                        base_url=self.generation_base_url,
                    ),
                    max_turns=MAX_TURNS,
                    verbose=False,
                    timeout=BASH_TIMEOUT_S,
                    sandbox_mode="required",
                    log_dir=str(log_dir),
                    stagnation_repeat_limit=2,
                    stagnation_recovery_attempt_limit=1,
                    max_completion_tokens=MAX_COMPLETION_TOKENS,
                    completion_recovery_attempt_limit=1,
                    max_consecutive_format_errors=2,
                    truncate_observations=False,
                    libreoffice_output_feedback=False,
                    structured_process_contract=False,
                    experience_provider=provider,
                )
                thread_state.runner = SpreadsheetBenchRunner(
                    agent=agent,
                    data_path=str(self.dataset_path),
                    output_dir=str(output_dir),
                    working_dir=str(working_dir),
                    workbook_structure_preflight=False,
                )
            return thread_state.runner.run_instance(instance)

        item_dir = batch_dir / "agent_items"
        item_dir.mkdir(exist_ok=True)
        runner_rows: dict[str, dict[str, Any]] = {}
        missing: list[BenchmarkInstance] = []
        for instance in instances:
            receipt = self._agent_item_path(item_dir, str(instance.id))
            if receipt.is_file():
                row = json.loads(receipt.read_text(encoding="utf-8"))
                if type(row) is not dict or row.get("id") != str(instance.id):
                    raise ValueError("stored Spreadsheet Agent item differs")
                runner_rows[str(instance.id)] = row
            else:
                missing.append(instance)
        with ThreadPoolExecutor(max_workers=AGENT_WORKERS) as pool:
            futures = {pool.submit(process, instance): instance for instance in missing}
            for future in as_completed(futures):
                instance = futures[future]
                try:
                    value = asdict(future.result())
                    value["id"] = str(instance.id)
                except Exception as exc:
                    value = {
                        "id": str(instance.id),
                        "instruction": instance.instruction,
                        "success": False,
                        "test_cases": [],
                        "error": f"worker_internal_error:{type(exc).__name__}:{exc}",
                    }
                runner_rows[str(instance.id)] = value
                _write_json(
                    self._agent_item_path(item_dir, str(instance.id)),
                    value,
                )
        _write_json(
            batch_dir / "runner_results.json",
            {"results": [runner_rows[str(instance.id)] for instance in instances]},
        )
        evaluation_path = batch_dir / "evaluation.json"
        if evaluation_path.is_file():
            evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        else:
            evaluation = self._evaluate(
                output_dir=output_dir,
                manifest_path=manifest_path,
                start_idx=start_idx,
                end_idx=end_idx,
                recalc_dir=batch_dir / "recalculated",
            )
            _write_json(evaluation_path, evaluation)
        records_path = batch_dir / "records.json"
        if records_path.is_file():
            records = json.loads(records_path.read_text(encoding="utf-8"))
        else:
            records = self._export_records(
                log_dir=log_dir,
                evaluation_path=evaluation_path,
                manifest_path=manifest_path,
                output_path=records_path,
                start_idx=start_idx,
                end_idx=end_idx,
            )
        self._validate_records_artifact(
            records_path=records_path,
            evaluation_path=evaluation_path,
            run_manifest_path=manifest_path,
            start_idx=start_idx,
            end_idx=end_idx,
            expected_task_ids=[row.task_id for row in prepared],
        )
        if type(records) is not list:
            raise ValueError("stored Spreadsheet trajectory records differ")
        original_by_id = {
            str(record["task_id"]): record
            for record in records
            if isinstance(record, Mapping) and record.get("task_id") is not None
        }
        if len(original_by_id) != len(records):
            raise ValueError("Spreadsheet trajectory record identities differ")
        failed_records = [
            original_by_id[row.task_id]
            for row in prepared
            if row.task_id in original_by_id
            and original_by_id[row.task_id].get("success") is False
        ]
        replay_path = batch_dir / "replay_outcomes.json"
        if replay_path.is_file():
            replay_rows = json.loads(replay_path.read_text(encoding="utf-8"))
            if type(replay_rows) is not list:
                raise ValueError("stored Spreadsheet replay outcomes differ")
        else:
            replay_by_id = self._replay(failed_records, batch_dir=batch_dir)
            replay_rows = list(replay_by_id.values())
            if not failed_records:
                _write_json(replay_path, [])
        for row in replay_rows:
            if type(row) is not dict:
                raise ValueError("stored Spreadsheet replay outcome differs")
            validate_source_replay_outcome_protocol(row)
        replay_task_ids = [str(row.get("task_id")) for row in replay_rows]
        expected_replay_task_ids = [str(row["task_id"]) for row in failed_records]
        if replay_task_ids != expected_replay_task_ids:
            raise ValueError("stored Spreadsheet replay population differs")
        replay_by_id = {str(row["task_id"]): row for row in replay_rows}
        evaluation_by_id = {
            str(row["id"]): row for row in evaluation.get("results", [])
        }
        episode_rows: list[EpisodeEvidence] = []
        for index, row in enumerate(prepared):
            try:
                episode = self._episode_from_artifacts(
                    prepared=row,
                    read_snapshot_id=read_snapshot_id,
                    retrieval=retrievals[index],
                    expectations=expectations[index],
                    original=original_by_id.get(row.task_id),
                    verifier=evaluation_by_id.get(row.task_id),
                    replay_outcome=replay_by_id.get(row.task_id),
                )
            except Exception as exc:
                episode = self._runtime_failure_episode(
                    prepared=row,
                    read_snapshot_id=read_snapshot_id,
                    retrieval=retrievals[index],
                    expectations=expectations[index],
                    error=f"evidence_projection:{type(exc).__name__}: {exc}",
                )
            episode_rows.append(episode)
        episodes = tuple(episode_rows)
        _write_json(
            batch_dir / "episodes.json",
            [episode.to_learning_payload() for episode in episodes],
        )
        return episodes

    @staticmethod
    def _agent_item_path(item_dir: Path, task_id: str) -> Path:
        digest = hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:24]
        return item_dir / f"{digest}.json"

    def _validate_completed_episodes(
        self,
        *,
        completed: tuple[EpisodeEvidence, ...],
        prepared: tuple[PreparedEpisode, ...],
        read_snapshot_id: str,
        retrievals: tuple[ContextualRetrieval, ...],
        expectations: tuple[tuple[ExperienceExpectation, ...], ...],
    ) -> None:
        if len(completed) != len(prepared):
            raise ValueError("stored Spreadsheet episode population differs")
        for index, (episode, source) in enumerate(zip(completed, prepared, strict=True)):
            if (
                episode.dataset_contract_id != self.dataset_contract_id
                or episode.train_index != source.train_index
                or episode.task_id != source.task_id
                or episode.read_snapshot_id != read_snapshot_id
                or episode.query_text != source.query_text
                or episode.observable_context != source.observable_context
                or dict(episode.retrieval_context) != retrievals[index].to_dict()
                or episode.expectations != expectations[index]
            ):
                raise ValueError("stored Spreadsheet episode identity differs")

    def _runtime_failure_episode(
        self,
        *,
        prepared: PreparedEpisode,
        read_snapshot_id: str,
        retrieval: ContextualRetrieval,
        expectations: Sequence[ExperienceExpectation],
        error: str,
    ) -> EpisodeEvidence:
        retrieval_context = {**retrieval.to_dict(), "adapter_error": error}
        episode_id = "episode_" + hashlib.sha256(
            canonical_json_bytes(
                {
                    "train_index": prepared.train_index,
                    "task_id": prepared.task_id,
                    "read_snapshot_id": read_snapshot_id,
                }
            )
        ).hexdigest()[:24]
        return EpisodeEvidence(
            episode_id,
            self.dataset_contract_id,
            prepared.train_index,
            prepared.task_id,
            read_snapshot_id,
            prepared.query_text,
            prepared.observable_context,
            retrieval_context,
            tuple(expectations),
            (),
            (),
            EpisodeOutcome.ITEM_LOCAL_RUNTIME_FAILURE,
        )

    def _run_manifest(
        self,
        *,
        provider: EIRGuidanceProvider,
        instance_ids: list[str],
        start_idx: int,
        end_idx: int,
    ) -> dict[str, Any]:
        dataset_file = self.dataset_path / "dataset.json"
        protocol = {
            "format": "spreadsheetbench_adapter_run_v1",
            "dataset_name": self.dataset_path.name,
            "dataset_sha256": _sha256(dataset_file),
            "dataset_tree_sha256": _tree_sha256(self.dataset_path),
            "start_idx": start_idx,
            "end_idx": end_idx,
            "instance_ids": instance_ids,
            "agent": "degs_eir_contextual_guidance",
            "experience_provider": dict(provider.identity()),
            "system_prompt_templates": [
                {"file_name": "preloaded_experience_full_system_v1.txt"}
            ],
            "model": self.model,
            "base_url": self.generation_base_url,
            "temperature": 0.0,
            "max_tokens": MAX_COMPLETION_TOKENS,
            "thinking": "false",
            "max_turns": MAX_TURNS,
            "bash_timeout": BASH_TIMEOUT_S,
            "bash_sandbox": "required",
            "workers": AGENT_WORKERS,
            "llm_timeout": LLM_TIMEOUT_S,
            "retry_waits": list(RETRY_WAITS_S),
            "runtime_timeout_retries": 1,
            "stagnation_repeat_limit": 2,
            "runtime_event_log_enabled": True,
            "runtime_event_log": "runtime_events.jsonl",
            "response_cache_enabled": False,
            "python_executable": Path(sys.executable).name,
            "python_version": platform.python_version(),
            "dependency_versions": _dependency_versions(),
        }
        identity = hashlib.sha256(canonical_json_bytes(protocol)).hexdigest()
        return {
            **protocol,
            "protocol_sha256": identity,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    def _evaluate(
        self,
        *,
        output_dir: Path,
        manifest_path: Path,
        start_idx: int,
        end_idx: int,
        recalc_dir: Path,
    ) -> Mapping[str, Any]:
        recalc_dir.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".train-evaluate-", dir=recalc_dir.parent
        ) as temporary:
            result_path = Path(temporary) / "evaluation.json"
            command = [
                sys.executable,
                str(Path(__file__).with_name("fresh_train_evaluate.py")),
                "--expected-model",
                self.model,
                "--data_path",
                str(self.dataset_path),
                "--output_dir",
                str(output_dir),
                "--results_file",
                str(result_path),
                "--recalc_dir",
                str(recalc_dir),
                "--run-manifest",
                str(manifest_path),
                "--expected-base-url",
                self.generation_base_url,
                "--expected-workers",
                str(AGENT_WORKERS),
                "--expected-thinking",
                "false",
                "--expected-max-tokens",
                str(MAX_COMPLETION_TOKENS),
                "--start_idx",
                str(start_idx),
                "--end_idx",
                str(end_idx),
            ]
            completed = subprocess.run(
                command,
                cwd=Path(__file__).resolve().parents[2],
                env=self._vendor_environment(),
                capture_output=True,
                text=True,
                check=False,
            )
            (recalc_dir.parent / "verifier.log").write_text(
                completed.stdout + completed.stderr,
                encoding="utf-8",
            )
            if completed.returncode:
                raise RuntimeError(
                    "Spreadsheet train verifier failed; inspect "
                    f"{recalc_dir.parent / 'verifier.log'}"
                )
            return json.loads(result_path.read_text(encoding="utf-8"))

    def _export_records(
        self,
        *,
        log_dir: Path,
        evaluation_path: Path,
        manifest_path: Path,
        output_path: Path,
        start_idx: int,
        end_idx: int,
    ) -> list[Mapping[str, Any]]:
        command = [
            sys.executable,
            "-m",
            "sb_adapter.export_trajectories",
            "--data-path",
            str(self.dataset_path),
            "--log-dir",
            str(log_dir),
            "--eval-file",
            str(evaluation_path),
            "--run-manifest",
            str(manifest_path),
            "--output",
            str(output_path),
            "--start-idx",
            str(start_idx),
            "--end-idx",
            str(end_idx),
        ]
        completed = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[2],
            env=self._vendor_environment(),
            capture_output=True,
            text=True,
            check=False,
        )
        (output_path.parent / "trajectory_export.log").write_text(
            completed.stdout + completed.stderr,
            encoding="utf-8",
        )
        if completed.returncode:
            raise RuntimeError(
                "Spreadsheet trajectory export failed; inspect "
                f"{output_path.parent / 'trajectory_export.log'}"
            )
        records = json.loads(output_path.read_text(encoding="utf-8"))
        if type(records) is not list:
            raise ValueError("Spreadsheet trajectory export differs")
        return records

    def _validate_records_artifact(
        self,
        *,
        records_path: Path,
        evaluation_path: Path,
        run_manifest_path: Path,
        start_idx: int,
        end_idx: int,
        expected_task_ids: Sequence[str],
    ) -> None:
        export_manifest_path = records_path.with_suffix(".manifest.json")
        if not export_manifest_path.is_file():
            raise ValueError("Spreadsheet trajectory export manifest is unavailable")
        export_manifest = json.loads(
            export_manifest_path.read_text(encoding="utf-8")
        )
        run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        if type(export_manifest) is not dict or type(run_manifest) is not dict:
            raise ValueError("Spreadsheet trajectory export identity differs")
        expected = {
            "format": "ordered_spreadsheetbench_trajectories_v1",
            "dataset_sha256": _sha256(self.dataset_path / "dataset.json"),
            "evaluation_sha256": _sha256(evaluation_path),
            "source_run_manifest_sha256": _sha256(run_manifest_path),
            "source_protocol_sha256": run_manifest.get("protocol_sha256"),
            "start_idx": start_idx,
            "end_idx": end_idx,
            "record_count": len(expected_task_ids),
            "task_ids": list(expected_task_ids),
            "records_sha256": _sha256(records_path),
        }
        differences = [
            key for key, value in expected.items() if export_manifest.get(key) != value
        ]
        if differences:
            raise ValueError(
                "Spreadsheet trajectory export identity differs: "
                + ", ".join(differences)
            )

    def _vendor_environment(self) -> dict[str, str]:
        environment = dict(os.environ)
        roots = [
            str(self.runtime_root / "src"),
            str(Path(__file__).resolve().parents[1]),
        ]
        existing = environment.get("PYTHONPATH")
        if existing:
            roots.append(existing)
        environment["PYTHONPATH"] = os.pathsep.join(roots)
        return environment

    def _replay(
        self,
        failed_records: Sequence[Mapping[str, Any]],
        *,
        batch_dir: Path,
    ) -> Mapping[str, Mapping[str, Any]]:
        if not failed_records:
            return {}
        runtime = ReplayRuntime(
            upstream_root=self.runtime_root,
            dataset_path=self.dataset_path,
            base_url=self.generation_base_url,
            api_key_env="DEGS_API_KEY",
            python_executable=sys.executable,
            model=self.model,
        )
        os.environ["DEGS_API_KEY"] = self.api_key
        executor = _ModelAwareReplayExecutor(runtime)
        thread_state = threading.local()

        def recover(record: Mapping[str, Any]) -> dict[str, Any]:
            controller = getattr(thread_state, "controller", None)
            if controller is None:
                client = OpenAIClient(
                    model=self.model,
                    api_key=self.api_key,
                    base_url=self.generation_base_url,
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
                    run_root=batch_dir / "replay",
                )
                thread_state.controller = controller
            return controller.recover(record)

        outcomes = _recover_in_train_order(failed_records, recover)
        _write_json(batch_dir / "replay_outcomes.json", outcomes)
        return {str(row["task_id"]): row for row in outcomes}

    def _episode_from_artifacts(
        self,
        *,
        prepared: PreparedEpisode,
        read_snapshot_id: str,
        retrieval: ContextualRetrieval,
        expectations: Sequence[ExperienceExpectation],
        original: Mapping[str, Any] | None,
        verifier: Mapping[str, Any] | None,
        replay_outcome: Mapping[str, Any] | None,
    ) -> EpisodeEvidence:
        episode_id = "episode_" + hashlib.sha256(
            canonical_json_bytes(
                {
                    "train_index": prepared.train_index,
                    "task_id": prepared.task_id,
                    "read_snapshot_id": read_snapshot_id,
                }
            )
        ).hexdigest()[:24]
        if original is None or verifier is None:
            return EpisodeEvidence(
                episode_id,
                self.dataset_contract_id,
                prepared.train_index,
                prepared.task_id,
                read_snapshot_id,
                prepared.query_text,
                prepared.observable_context,
                retrieval.to_dict(),
                tuple(expectations),
                (),
                (),
                EpisodeOutcome.ITEM_LOCAL_RUNTIME_FAILURE,
            )
        original_trace = _trace_items(original, prefix="original")
        original_success = verifier.get("success") is True
        original_verifier = (
            EvidenceItem(
                "verifier:original:overall",
                "verifier_success" if original_success else "verifier_failure",
                dict(verifier),
            ),
        )
        if original_success and original_trace:
            return EpisodeEvidence(
                episode_id,
                self.dataset_contract_id,
                prepared.train_index,
                prepared.task_id,
                read_snapshot_id,
                prepared.query_text,
                prepared.observable_context,
                retrieval.to_dict(),
                tuple(expectations),
                original_trace,
                original_verifier,
                EpisodeOutcome.ORIGINAL_SUCCESS,
            )
        if (
            replay_outcome is not None
            and replay_outcome.get("status") == "REPLAY_VALIDATED_SUCCESS"
        ):
            replay_trace: tuple[EvidenceItem, ...] = ()
            accepted_index = replay_outcome.get("accepted_attempt_index")
            attempts = replay_outcome.get("attempts")
            accepted = next(
                (
                    row
                    for row in attempts
                    if isinstance(row, Mapping)
                    and row.get("attempt_index") == accepted_index
                ),
                None,
            ) if isinstance(attempts, list) else None
            if isinstance(accepted, Mapping):
                replay_path = Path(str(accepted["replay_trajectory_path"]))
                replay = json.loads(replay_path.read_text(encoding="utf-8"))
                replay_trace = _trace_items(replay, prefix="replay")
                if not replay_trace:
                    accepted = None
            if isinstance(accepted, Mapping):
                return EpisodeEvidence(
                    episode_id,
                    self.dataset_contract_id,
                    prepared.train_index,
                    prepared.task_id,
                    read_snapshot_id,
                    prepared.query_text,
                    prepared.observable_context,
                    retrieval.to_dict(),
                    tuple(expectations),
                    original_trace,
                    original_verifier,
                    EpisodeOutcome.REPAIR_SUCCESS,
                    (
                        EvidenceItem("patch:final", "patch", accepted.get("patch")),
                    ),
                    replay_trace,
                    (
                        EvidenceItem(
                            "verifier:replay:overall",
                            "verifier_success",
                            {
                                "score": accepted.get("verifier_score"),
                                "feedback": accepted.get("verifier_feedback"),
                            },
                        ),
                    ),
                )
        outcome = (
            EpisodeOutcome.UNRESOLVED_TASK_FAILURE
            if original_trace
            else EpisodeOutcome.ITEM_LOCAL_RUNTIME_FAILURE
        )
        return EpisodeEvidence(
            episode_id,
            self.dataset_contract_id,
            prepared.train_index,
            prepared.task_id,
            read_snapshot_id,
            prepared.query_text,
            prepared.observable_context,
            retrieval.to_dict(),
            tuple(expectations),
            original_trace if outcome is EpisodeOutcome.UNRESOLVED_TASK_FAILURE else (),
            original_verifier if outcome is EpisodeOutcome.UNRESOLVED_TASK_FAILURE else (),
            outcome,
        )


__all__ = ["SPREADSHEET_EPISODE_FORMAT", "SpreadsheetEpisodeAdapter"]
