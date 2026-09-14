from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import pytest

from degs.benchmark import _write_json_atomic
from degs.soft_hard_benchmark import (
    RESULT_FORMAT,
    _GenerationTransportTracker,
    _TrackedAgentClient,
    _case_result_path,
    _load_completed_case_results,
)
from degs.validated_repair import (
    ProducerTransportGuard,
    SystemicProducerTransportFailure,
)
from react_agent.models import RequestRuntimeTimeout


class _TransportFailure(Exception):
    def __init__(self, status_code=None):
        super().__init__("transport")
        self.status_code = status_code


class _TimeoutProducer:
    protocol_identity = {}

    async def complete_json_async(self, *args, **kwargs):
        raise RequestRuntimeTimeout("timeout")


class _UnknownFailureClient:
    async def chat_async(self, *args, **kwargs):
        raise RuntimeError("unexpected client failure")


class _CountingClient:
    def __init__(self) -> None:
        self.calls = 0

    async def chat_async(self, *args, **kwargs):
        self.calls += 1
        return "ok"


def _plan():
    task = {
        "task_id": "task",
        "query_index": 0,
        "spreadsheet_path": "spreadsheet/task",
    }
    first = {
        "case_id": "task__1",
        "input_file": "1_task_input.xlsx",
        "output_file": "1_task_output.xlsx",
    }
    second = {
        "case_id": "task__2",
        "input_file": "2_task_input.xlsx",
        "output_file": "2_task_output.xlsx",
    }
    return task, first, second


def _row(task, case, protocol):
    return {
        "format": RESULT_FORMAT,
        "protocol_sha256": protocol,
        "case_id": case["case_id"],
        "task_id": task["task_id"],
        "query_index": task["query_index"],
        "input_file": case["input_file"],
        "output_file": case["output_file"],
        "output_path": "/tmp/output.xlsx",
        "output_sha256": "",
        "output_size": 0,
        "agent_success": False,
        "agent_completed": False,
        "output_preserved": False,
        "turns": 0,
        "answer": "",
        "error": "terminal item-local failure",
        "failure_kind": "agent_terminal_failure",
    }


def test_resume_skips_durable_success_or_failure_and_runs_only_absent_case(
    tmp_path: Path,
) -> None:
    task, first, second = _plan()
    protocol = "a" * 64
    case_results = tmp_path / "case_results"
    _write_json_atomic(
        _case_result_path(case_results, first["case_id"]),
        _row(task, first, protocol),
    )

    completed, pending = _load_completed_case_results(
        case_plan=[(task, first), (task, second)],
        case_results_dir=case_results,
        protocol_sha256=protocol,
    )

    assert list(completed) == [first["case_id"]]
    assert pending == [(task, second)]


def test_resume_rejects_case_from_another_protocol(tmp_path: Path) -> None:
    task, first, _second = _plan()
    case_results = tmp_path / "case_results"
    _write_json_atomic(
        _case_result_path(case_results, first["case_id"]),
        _row(task, first, "b" * 64),
    )

    with pytest.raises(ValueError, match="identity differs"):
        _load_completed_case_results(
            case_plan=[(task, first)],
            case_results_dir=case_results,
            protocol_sha256="a" * 64,
        )


def test_resume_rejects_changed_preserved_output(tmp_path: Path) -> None:
    task, first, _second = _plan()
    protocol = "a" * 64
    output_dir = tmp_path / "outputs"
    output_path = output_dir / task["spreadsheet_path"] / first["output_file"]
    output_path.parent.mkdir(parents=True)
    output_path.write_bytes(b"original")
    row = _row(task, first, protocol)
    row.update(
        {
            "output_path": str(output_path),
            "output_preserved": True,
            "output_sha256": hashlib.sha256(b"original").hexdigest(),
            "output_size": len(b"original"),
        }
    )
    case_results = tmp_path / "case_results"
    _write_json_atomic(
        _case_result_path(case_results, first["case_id"]),
        row,
    )
    output_path.write_bytes(b"changed")

    with pytest.raises(ValueError, match="output identity differs"):
        _load_completed_case_results(
            case_plan=[(task, first)],
            case_results_dir=case_results,
            protocol_sha256=protocol,
            output_dir=output_dir,
        )


def test_generation_transport_tracker_requires_three_distinct_failures() -> None:
    tracker = _GenerationTransportTracker()
    wave = tracker.begin_wave()
    for case_id in ("a", "b", "c"):
        tracker.record_failure(case_id, _TransportFailure())

    with pytest.raises(RuntimeError, match="systemic generation transport"):
        tracker.raise_if_systemic(wave)


def test_generation_transport_success_prevents_false_systemic_wave() -> None:
    tracker = _GenerationTransportTracker()
    wave = tracker.begin_wave()
    for case_id in ("a", "b", "c"):
        tracker.record_failure(case_id, _TransportFailure(500))
    tracker.record_success("d")

    tracker.raise_if_systemic(wave)


def test_generation_transport_tracker_rejects_auth_configuration() -> None:
    tracker = _GenerationTransportTracker()
    wave = tracker.begin_wave()
    tracker.record_failure("a", _TransportFailure(401))

    with pytest.raises(RuntimeError, match="configuration failed"):
        tracker.raise_if_systemic(wave)


def test_tracked_agent_client_records_unknown_client_failure() -> None:
    async def exercise() -> None:
        tracker = _GenerationTransportTracker()
        wave = tracker.begin_wave()
        client = _TrackedAgentClient(_UnknownFailureClient(), tracker)
        client._case_id = "a"
        with pytest.raises(RuntimeError, match="unexpected client failure"):
            await client.chat_async()
        with pytest.raises(RuntimeError, match="unknown generation client"):
            tracker.raise_if_fatal(wave)

    asyncio.run(exercise())


def test_tracked_agent_client_cooperatively_stops_after_fatal_signal() -> None:
    async def exercise() -> None:
        tracker = _GenerationTransportTracker()
        tracker.record_failure("a:1", _TransportFailure(401), case_id="a")
        raw_client = _CountingClient()
        client = _TrackedAgentClient(raw_client, tracker)
        client._case_id = "b"
        with pytest.raises(RuntimeError, match="configuration failed"):
            await client.chat_async()
        assert raw_client.calls == 0

    asyncio.run(exercise())


def test_runner_failure_signal_cooperatively_stops_sibling_client() -> None:
    async def exercise() -> None:
        tracker = _GenerationTransportTracker()
        tracker.signal_fatal("unknown Agent/runner failure")
        raw_client = _CountingClient()
        sibling = _TrackedAgentClient(raw_client, tracker)
        sibling._case_id = "sibling"
        with pytest.raises(RuntimeError, match="Agent/runner"):
            await sibling.chat_async()
        assert raw_client.calls == 0

    asyncio.run(exercise())


def test_bundle_guard_uses_distinct_native_request_failures() -> None:
    async def exercise() -> None:
        guard = ProducerTransportGuard(stage="test")
        wave = await guard.begin_wave()
        for request_id in ("a", "b", "c"):
            await guard.record_failure(request_id=request_id, error=RequestRuntimeTimeout("synthetic"))
        with pytest.raises(SystemicProducerTransportFailure):
            await guard.raise_if_systemic(wave)

    asyncio.run(exercise())
