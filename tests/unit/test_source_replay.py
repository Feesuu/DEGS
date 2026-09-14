from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from degs.core import canonical_json_bytes
from degs.source_replay import (
    SOURCE_REPLAY_OUTCOME_FORMAT,
    SOURCE_REPLAY_PATCH_KIND,
    SOURCE_REPLAY_PATCH_PROMPT,
    SOURCE_REPLAY_PATCH_PROMPT_SHA256,
    SOURCE_REPLAY_PROTOCOL_FORMAT,
    ReplayExecution,
    SourceReplayController,
    parse_source_replay_patch,
    render_failed_trajectory,
    render_patch_for_executor,
    source_replay_patch_response_schema,
)
from degs.source_replay_executor import (
    ReplayRuntime,
    SubprocessReplayExecutor,
    SOURCE_REPLAY_TASK_WORKERS,
    _UPSTREAM_RUNTIME_PATHS,
    _recover_in_train_order,
)
from degs import replay_overflow_probe


def _failed_record() -> dict:
    return {
        "task_id": "synthetic-failure",
        "trajectory_id": "synthetic-failure::original",
        "instruction": "Repair the complete artifact without changing valid state.",
        "instruction_type": "Sheet-Level Manipulation",
        "answer_position": "Sheet1!A1:Z999",
        "success": False,
        "steps": [
            {
                "step_id": 1,
                "raw_model_output": "raw-start\n" + "r" * 31_000 + "\nraw-end",
                "action": "action-start\n" + "a" * 25_000 + "\naction-end",
                "observation": "observation-start\n" + "o" * 160_000 + "\nobservation-end",
                "action_valid": True,
                "tool_name": "bash",
            }
        ],
        "final_response": "final-start\n" + "f" * 25_000 + "\nfinal-end",
        "verifier_score": 0.0,
        "verifier_feedback": "verifier-start\n" + "v" * 30_000 + "\nverifier-end",
    }


class ScriptedPatchLLM:
    def __init__(self, response: dict):
        self.response = response
        self.calls: list[dict] = []

    @property
    def protocol_identity(self):
        return {"format": "synthetic_patch_llm_v1", "model": "synthetic"}

    def complete_json(self, **request):
        self.calls.append(request)
        return json.loads(json.dumps(self.response))


class SuccessfulExecutor:
    def __init__(self):
        self.calls: list[dict] = []

    @property
    def protocol_identity(self):
        return {"format": "synthetic_fresh_replay_executor_v1"}

    def execute(self, **request):
        self.calls.append(request)
        failed = request["failed_record"]
        return ReplayExecution(
            task_id=failed["task_id"],
            trajectory_id=f'{failed["task_id"]}::replay::{request["attempt_index"]}',
            success=True,
            verifier_score=1.0,
            verifier_feedback="All task constraints passed.",
            trajectory={
                "task_id": failed["task_id"],
                "trajectory_id": f'{failed["task_id"]}::replay::{request["attempt_index"]}',
                "instruction": failed["instruction"],
                "success": True,
                "steps": [
                    {
                        "step_id": 1,
                        "action": "Apply every required correction.",
                        "observation": "The complete correction was applied.",
                        "action_valid": True,
                        "tool_name": "bash",
                    }
                ],
                "final_response": "Completed and verified.",
            },
        )


class ContextOverflowExecutor(SuccessfulExecutor):
    def execute(self, **request):
        self.calls.append(request)
        failed = request["failed_record"]
        trajectory_id = f'{failed["task_id"]}::overflow'
        return ReplayExecution(
            task_id=failed["task_id"],
            trajectory_id=trajectory_id,
            success=False,
            verifier_score=0.0,
            verifier_feedback="CONTEXT_LENGTH_EXCEEDED",
            trajectory={
                "task_id": failed["task_id"],
                "trajectory_id": trajectory_id,
                "instruction": failed["instruction"],
                "success": False,
                "steps": [],
                "final_response": "CONTEXT_LENGTH_EXCEEDED",
            },
            error="CONTEXT_LENGTH_EXCEEDED",
        )


def _patch_response() -> dict:
    return {
        "diagnosis": "The previous attempt changed state before collecting complete evidence.",
        "instructions": [
            "Collect the complete target set before any mutation.",
            "Preserve every valid region while applying all required changes.",
        ],
        "checks": [
            "Reopen the saved artifact and verify every requested target and invariant."
        ],
    }


def _keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _keys(item)


def test_failed_trajectory_renderer_preserves_every_character() -> None:
    rendered = render_failed_trajectory(_failed_record())

    serialized = json.dumps(rendered.payload, ensure_ascii=False)
    for marker in (
        "action-start",
        "action-end",
        "raw-start",
        "raw-end",
        "observation-start",
        "observation-end",
        "final-start",
        "final-end",
        "verifier-start",
        "verifier-end",
    ):
        assert marker in serialized
    assert "input_truncated" not in set(_keys(rendered.payload))
    assert rendered.payload["turns"][1]["raw_model_output"].endswith("raw-end")


def test_failed_renderer_keeps_raw_model_output_when_parsed_action_is_empty() -> None:
    record = _failed_record()
    record["steps"][0]["action"] = ""

    rendered = render_failed_trajectory(record)

    assert rendered.payload["turns"][1]["action"] == ""
    assert rendered.payload["turns"][1]["raw_model_output"].startswith("raw-start")


def test_patch_schema_has_no_host_count_or_character_caps_and_prompt_is_not_minimalist() -> None:
    schema = source_replay_patch_response_schema()

    assert "maxItems" not in set(_keys(schema))
    assert "maxLength" not in set(_keys(schema))
    assert set(schema["properties"]) == {"diagnosis", "instructions", "checks"}
    lowered = SOURCE_REPLAY_PATCH_PROMPT.lower()
    for forbidden in ("smallest", "minimal", "concise"):
        assert forbidden not in lowered
    assert "all evidence-supported corrective" in lowered
    assert "context_length_exceeded" in lowered
    assert "source_turn_ids" not in SOURCE_REPLAY_PATCH_PROMPT


def test_patch_accepts_unbounded_rows_and_contains_no_turn_evidence_field() -> None:
    payload = {
        "diagnosis": "A complete evidence-grounded diagnosis.",
        "instructions": [
            f"Instruction {index} with its condition and scope."
            for index in range(12)
        ],
        "checks": [f"Check {index} against observable output state." for index in range(9)],
    }

    patch = parse_source_replay_patch(payload)
    rendered = render_patch_for_executor(patch)

    assert len(patch.instructions) == 12
    assert len(patch.checks) == 9
    assert "A complete evidence-grounded diagnosis" not in rendered
    assert "source_turn_ids" not in patch.to_dict()
    assert all(row in rendered for row in patch.instructions)
    assert all(row in rendered for row in patch.checks)


def test_controller_builds_new_identity_from_full_failure_and_fresh_success(
    tmp_path: Path,
) -> None:
    llm = ScriptedPatchLLM(_patch_response())
    executor = SuccessfulExecutor()
    controller = SourceReplayController(
        patch_llm=llm,
        attempt_executor=executor,
        run_root=tmp_path / "replay",
    )

    outcome = controller.recover(_failed_record())

    assert outcome["format"] == SOURCE_REPLAY_OUTCOME_FORMAT
    assert outcome["status"] == "REPLAY_VALIDATED_SUCCESS"
    assert outcome["source_replay_protocol"]["format"] == SOURCE_REPLAY_PROTOCOL_FORMAT
    assert outcome["source_replay_protocol"]["patch_request_kind"] == SOURCE_REPLAY_PATCH_KIND
    assert (
        outcome["source_replay_protocol"]["patch_prompt_sha256"]
        == SOURCE_REPLAY_PATCH_PROMPT_SHA256
    )
    assert outcome["source_replay_protocol"]["no_input_truncation"] is True
    assert hashlib.sha256(
        canonical_json_bytes(outcome["source_replay_protocol"])
    ).hexdigest() == outcome["source_replay_protocol_sha256"]
    assert "observation-end" in json.dumps(llm.calls[0]["payload"], ensure_ascii=False)
    assert executor.calls[0]["rendered_patch"] == render_patch_for_executor(
        parse_source_replay_patch(_patch_response())
    )
    accepted = outcome["attempts"][0]
    trajectory = json.loads(Path(accepted["replay_trajectory_path"]).read_text())
    assert (
        trajectory["extra"]["source_replay"]["source_replay_protocol_sha256"]
        == outcome["source_replay_protocol_sha256"]
    )


def test_controller_stops_on_executor_context_overflow_with_explicit_marker(
    tmp_path: Path,
) -> None:
    executor = ContextOverflowExecutor()
    controller = SourceReplayController(
        patch_llm=ScriptedPatchLLM(_patch_response()),
        attempt_executor=executor,
        run_root=tmp_path / "replay",
    )

    outcome = controller.recover(_failed_record())

    assert outcome["status"] == "REPLAY_RUNTIME_FAILURE"
    assert outcome["error"] == "CONTEXT_LENGTH_EXCEEDED"
    assert len(executor.calls) == 1


def test_subprocess_adapter_uses_the_bundled_runtime_and_ordered_train_slice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "runtime"
    for relative in _UPSTREAM_RUNTIME_PATHS:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"synthetic upstream file: {relative}\n", encoding="utf-8")
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir()
    (dataset_path / "dataset.json").write_text(
        json.dumps(
            [
                {"id": index if index == 0 else f"synthetic-{index:03d}"}
                for index in range(200)
            ]
        ),
        encoding="utf-8",
    )

    executor = SubprocessReplayExecutor(
        ReplayRuntime(
            upstream_root=root,
            dataset_path=dataset_path,
            base_url="http://127.0.0.1:18081/v1",
            api_key_env="DEGS_API_KEY",
            python_executable="python",
        )
    )

    assert executor.train_task_ids == (
        "0",
        *(f"synthetic-{index:03d}" for index in range(1, 200)),
    )
    assert executor.protocol_identity["format"] == "degs_fresh_replay_executor_v1"
    assert executor.protocol_identity["context_overflow_reporting"] == "machine_readable_marker_v1"
    assert executor.protocol_identity["observation_policy"] == "full_no_truncation"
    assert executor.protocol_identity["outer_task_workers"] == 8
    assert executor.protocol_identity["max_completion_tokens"] == 16_384

    attempt_dir = tmp_path / "attempt"
    attempt_dir.mkdir()

    def context_overflow(command, _audit_path, *, context_overflow_marker=None):
        assert context_overflow_marker is not None
        assert command[command.index("--max-tokens") + 1] == "16384"
        context_overflow_marker.write_text(
            "CONTEXT_LENGTH_EXCEEDED\n", encoding="utf-8"
        )
        return ""

    monkeypatch.setattr(executor, "_run_command", context_overflow)
    execution = executor.execute(
        failed_record={
            "task_id": "0",
            "instruction": "Complete the task.",
        },
        rendered_patch="Use the complete repair.",
        patch_id="degs_patch_synthetic",
        attempt_index=1,
        attempt_dir=attempt_dir,
        source_replay_protocol_sha256="0" * 64,
    )
    assert execution.error == "CONTEXT_LENGTH_EXCEEDED"
    assert execution.trajectory["final_response"] == "CONTEXT_LENGTH_EXCEEDED"


def test_failed_tasks_run_with_eight_workers_and_return_in_train_order() -> None:
    barrier = threading.Barrier(SOURCE_REPLAY_TASK_WORKERS)
    lock = threading.Lock()
    active = 0
    maximum_active = 0
    records = [{"train_index": index} for index in range(12)]

    def recover(record):
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        if record["train_index"] < SOURCE_REPLAY_TASK_WORKERS:
            barrier.wait(timeout=2)
        time.sleep((12 - record["train_index"]) / 10_000)
        with lock:
            active -= 1
        return {"train_index": record["train_index"]}

    outcomes = _recover_in_train_order(records, recover)

    assert SOURCE_REPLAY_TASK_WORKERS == 8
    assert maximum_active == 8
    assert [row["train_index"] for row in outcomes] == list(range(12))


def test_replay_probe_observes_an_overflow_swallowed_by_the_pinned_style_runtime(
    tmp_path: Path,
) -> None:
    module = tmp_path / "synthetic_swallow.py"
    module.write_text(
        "from react_agent.models import RequestContextLengthExceeded\n"
        "try:\n"
        "    raise RequestContextLengthExceeded('provider detail')\n"
        "except RequestContextLengthExceeded:\n"
        "    result = 'Max turns exceeded'\n",
        encoding="utf-8",
    )
    marker = tmp_path / "context.marker"
    environment = dict(os.environ)
    environment[replay_overflow_probe.MARKER_ENV] = str(marker)
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(tmp_path), str(Path(__file__).parents[2] / "src"))
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(Path(replay_overflow_probe.__file__)),
            "synthetic_swallow",
        ],
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout
    assert marker.read_text(encoding="utf-8") == "CONTEXT_LENGTH_EXCEEDED\n"


def test_replay_adapter_preserves_a_complete_tool_observation(tmp_path: Path) -> None:
    output = tmp_path / "observation.txt"
    module = tmp_path / "synthetic_observation.py"
    module.write_text(
        "import os\n"
        "from pathlib import Path\n"
        "from react_agent.agent import truncate_observation\n"
        "value = 'BEGIN-' + ('x' * 7000) + '-END'\n"
        "Path(os.environ['OBSERVATION_OUTPUT']).write_text(\n"
        "    truncate_observation(value), encoding='utf-8'\n"
        ")\n",
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment["OBSERVATION_OUTPUT"] = str(output)
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(tmp_path), str(Path(__file__).parents[2] / "src"))
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(Path(replay_overflow_probe.__file__)),
            "synthetic_observation",
        ],
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout
    assert output.read_text(encoding="utf-8") == "BEGIN-" + ("x" * 7000) + "-END"


def test_subprocess_adapter_rejects_model_or_runtime_drift(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="fixed protocol"):
        ReplayRuntime(
            upstream_root=tmp_path,
            dataset_path=tmp_path,
            base_url="http://127.0.0.1:18081/v1",
            api_key_env="DEGS_API_KEY",
            python_executable="python",
            model="different-model",
        )


def test_subprocess_adapter_requires_the_runtime_files(tmp_path: Path) -> None:
    root = tmp_path / "runtime"
    for relative in _UPSTREAM_RUNTIME_PATHS:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"synthetic upstream file: {relative}\n", encoding="utf-8")
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir()
    (dataset_path / "dataset.json").write_text(
        json.dumps([{"id": f"synthetic-{index:03d}"} for index in range(200)]),
        encoding="utf-8",
    )

    (root / _UPSTREAM_RUNTIME_PATHS[0]).unlink()
    with pytest.raises(ValueError, match="runtime file is missing"):
        SubprocessReplayExecutor(
            ReplayRuntime(
                upstream_root=root,
                dataset_path=dataset_path,
                base_url="http://127.0.0.1:18081/v1",
                api_key_env="DEGS_API_KEY",
                python_executable="python",
            )
        )
