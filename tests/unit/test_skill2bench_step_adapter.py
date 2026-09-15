from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys

import httpx
from openai import InternalServerError, RateLimitError
import pytest
from react_agent.models import RequestRuntimeTimeout

from degs.core import canonical_json_bytes
from degs.source_review import ExperienceSourceReviewer
from degs.validated_repair import ValidatedRepairMemory
from degs.validated_repair import SystemicProducerTransportFailure
from degs_skill2bench.contract import skill2bench_protocol
from degs_skill2bench import campaign
from degs_skill2bench.dataset import public_task_view
from degs_skill2bench.source_extraction import (
    StepExperienceExtractor,
    build_source_batch,
    validate_batch_source_audit,
)
from degs_skill2bench.repair import failed_step_payload, render_repair_skill, step_outcome
from degs_skill2bench.retrieval import _step_items
from degs_skill2bench.runtime import Skill2BenchWorkerError, render_agent_skill
from degs_skill2bench.step_evidence import (
    original_success_record,
    validated_repair_record,
)


class _LLM:
    def __init__(self) -> None:
        self.payloads = []

    @property
    def protocol_identity(self):
        return {"service_url": "http://127.0.0.1:18081/v1", "model": "test"}

    async def complete_json_async(self, **kwargs):
        self.payloads.append(kwargs["payload"])
        return {
            "experience_nodes": [
                {
                    "operation": "Apply the accepted target-Step rule to the selected input.",
                    "applicability": ["The target input satisfies the patch condition."],
                    "inputs": [{"type": "value", "description": "selected target input"}],
                    "outputs": [{"type": "decision", "description": "target-Step result"}],
                }
            ],
            "edges": [],
        }


class _ReviewLLM(_LLM):
    async def complete_json_async(self, **kwargs):
        self.payloads.append(kwargs["payload"])
        draft = kwargs["payload"]["draft_graph"]
        return {
            **draft,
            "review_decisions": [
                {
                    "draft_node": index,
                    "decision": "KEEP",
                    "final_nodes": [index],
                    "basis": "The target-Step evidence supports this operation.",
                }
                for index in range(len(draft["experience_nodes"]))
            ],
        }


class _BadReviewLLM(_LLM):
    async def complete_json_async(self, **kwargs):
        raise ValueError("invalid review")


class _TimeoutLLM(_LLM):
    async def complete_json_async(self, **kwargs):
        raise RequestRuntimeTimeout("transport timeout")


def _task():
    return {
        "instance_id": "task-a",
        "scenario": "Shared background",
        "steps": [
            {"question": "Find the target value."},
            {"question": "Classify the target value."},
        ],
    }


def test_repair_record_keeps_target_patch_and_full_successful_replay():
    replay = {
        "react_steps": [
            {"thought": "Step 1: find it", "action": "a", "observation": "x"},
            {"thought": "Step 2: classify it", "action": "b", "observation": "y"},
        ]
    }
    record = validated_repair_record(
        evidence_id="train-000::step-02::repair-01",
        step_number=2,
        memory=ValidatedRepairMemory(("Use the category boundary.",), ("Check the boundary." ,)),
        successful_replay=replay,
    )
    assert record["trace_scope"] == "FULL_SUCCESSFUL_REPLAY"
    assert record["validated_repair_memory"]["instructions"] == [
        "Use the category boundary."
    ]
    assert record["successful_trajectory"] == replay["react_steps"]


def test_multistep_original_trace_has_no_implicit_step_one_owner():
    record = original_success_record(
        evidence_id="train-000::step-01::original",
        step_number=1,
        rollout={"react_steps": [{"thought": "unmarked setup"}]},
        expected_steps=2,
    )
    assert record is None


def test_source_extraction_is_bound_to_one_step_and_edges_stay_local():
    llm = _LLM()
    review_llm = _ReviewLLM()
    task = public_task_view(_task())
    evidence = validated_repair_record(
        evidence_id="train-000::step-02::repair-01",
        step_number=2,
        memory=ValidatedRepairMemory(("Use the category boundary.",), ("Check the boundary.",)),
        successful_replay={
            "react_steps": [
                {"thought": "Step 1: unrelated lookup"},
                {"thought": "Step 2: applied the category boundary"},
            ]
        },
    )
    source, audit = asyncio.run(
        build_source_batch(
            batch_task_indices=(0,),
            public_tasks={0: task},
            evidence_by_index={0: [evidence]},
            extractor=StepExperienceExtractor(llm),
            reviewer=ExperienceSourceReviewer(review_llm),
        )
    )
    assert [row.train_index for row in source.workflows] == [1]
    assert source.workflows[0].query_text == "Classify the target value."
    assert llm.payloads[0]["target_step"]["number"] == 2
    assert len(llm.payloads[0]["accepted_evidence"][0]["successful_trajectory"]) == 2
    assert review_llm.payloads[0]["evidence"]["target_step"]["number"] == 2
    indices = tuple(range(10))
    _sha, statuses = validate_batch_source_audit(
        batch_source=source,
        batch_train_indices=indices,
        audit=audit,
        expected_generation_endpoint="http://127.0.0.1:18081/v1",
    )
    assert statuses[1] == "INGESTED"
    assert statuses[0] == "SOURCE_EXCLUDED_NO_VALIDATED_SUCCESS"


def test_review_exhaustion_excludes_only_that_step():
    task = public_task_view(_task())
    evidence = validated_repair_record(
        evidence_id="train-000::step-02::repair-01",
        step_number=2,
        memory=ValidatedRepairMemory(("Use the boundary.",), ("Check it.",)),
        successful_replay={"react_steps": [{"thought": "Step 2: use boundary"}]},
    )
    source, audit = asyncio.run(
        build_source_batch(
            batch_task_indices=(0,),
            public_tasks={0: task},
            evidence_by_index={0: [evidence]},
            extractor=StepExperienceExtractor(_LLM()),
            reviewer=ExperienceSourceReviewer(_BadReviewLLM()),
        )
    )

    assert source.workflows == ()
    by_index = {
        row["train_index"]: row for row in (*audit["rows"], *audit["exclusions"])
    }
    assert by_index[1]["status"] == "SOURCE_EXCLUDED_REVIEW_FAILURE"


def test_skill2bench_contract_maps_eight_tasks_to_eighty_step_slots():
    contract = skill2bench_protocol("9b").graph_contract
    assert len(contract.batch_indices(0)) == 80
    assert contract.batch_indices(12) == tuple(range(960, 1000))
    assert skill2bench_protocol("27b").model == "Qwen3.5-27B-AWQ"


def test_partial_open_ended_credit_is_still_repair_eligible():
    assert step_outcome(
        {"status": "scored", "score": 0.75, "is_open_ended": True}
    ) == "FAILURE"
    assert step_outcome(
        {"status": "scored", "score": 1.0, "is_open_ended": True}
    ) == "SUCCESS"


def test_agent_profile_rejects_step_order_as_implicit_causality():
    prompt = render_agent_skill("Step 2:\nUse the selected operation.")

    assert "Step numbering or presentation order alone never creates a dependency" in prompt
    assert "Step 2:\nUse the selected operation." in prompt


def test_repair_replay_keeps_the_same_dataset_step_semantics():
    prompt = render_repair_skill(
        2,
        ValidatedRepairMemory(("Use the accepted condition.",), ("Check it.",)),
    )

    assert "Step numbering or presentation order alone never creates a dependency" in prompt
    assert "Apply this guidance only to Step 2" in prompt


def test_failed_patch_payload_exposes_only_the_target_step():
    payload = failed_step_payload(
        task=_task(),
        rollout={"react_steps": [{"thought": "unmarked work"}]},
        evaluated_step={
            "step": 2,
            "prediction": "wrong",
            "score": 0.0,
            "status": "scored",
        },
        attempt_index=1,
    )

    assert payload["target_step"]["question"] == "Classify the target value."
    assert "questions" not in payload
    assert "public_task" not in payload
    assert payload["failed_trajectory"] == []
    assert payload["trace_scope"] == "NO_ATTRIBUTABLE_TARGET_TRACE"


def test_test_steps_become_independent_eir_queries():
    items, coordinates = _step_items((_task(),))

    assert coordinates == ((0, 1), (0, 2))
    assert [row.query_text for row in items] == [
        "Find the target value.",
        "Classify the target value.",
    ]
    assert all(
        row.observable_context[1].kind == "independent_target_step"
        for row in items
    )


def test_cli_selects_27b_before_model_dependent_imports():
    code = """
import sys, types
campaign = types.ModuleType('degs_skill2bench.campaign')
def fake_main(argv):
    from degs.validated_repair import REPAIR_SOURCE_MODEL
    print(REPAIR_SOURCE_MODEL)
    return 0
campaign.main = fake_main
sys.modules['degs_skill2bench.campaign'] = campaign
from degs_skill2bench.cli import main
raise SystemExit(main(['--profile', '27b']))
"""
    environment = dict(os.environ)
    environment.pop("DEGS_MODEL", None)
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.stdout.strip() == "Qwen3.5-27B-AWQ"


def test_cli_help_does_not_require_a_model_profile():
    result = subprocess.run(
        [sys.executable, "-m", "degs_skill2bench.cli", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--profile {9b,27b}" in result.stdout


def test_systemic_population_failures_are_retryable(tmp_path: Path, monkeypatch):
    calls = []

    async def direct_to_thread(function, **kwargs):
        return function(**kwargs)

    def transport(**kwargs):
        calls.append(kwargs["task"]["instance_id"])
        if len(calls) <= 3:
            raise Skill2BenchWorkerError(
                "RequestRuntimeTimeout: transport timeout",
                worker_error_type="RequestRuntimeTimeout",
                status_code=None,
                transport_failure=True,
            )
        task = kwargs["task"]
        return (
            {
                "instance_id": task["instance_id"],
                "answer": "ok",
                "agent_success": True,
                "react_steps": [{"thought": "Step 1: answer"}],
            },
            {
                "instance_id": task["instance_id"],
                "success": True,
                "score": 1.0,
                "steps": [{"step": 1, "status": "scored", "score": 1.0}],
            },
        )

    monkeypatch.setattr(campaign, "run_and_evaluate_task", transport)
    monkeypatch.setattr(campaign.asyncio, "to_thread", direct_to_thread)
    tasks = [
        {
            "instance_id": f"task-{index}",
            "scenario": "scenario",
            "steps": [{"question": "q", "solution": "a"}],
        }
        for index in range(3)
    ]
    kwargs = {
        "tasks": tasks,
        "output_root": tmp_path / "population",
        "skill_by_index": {index: None for index in range(3)},
        "baseline_root": tmp_path,
        "official_evaluator_root": tmp_path,
        "base_url": "http://127.0.0.1:18081/v1",
        "api_key": "test",
        "protocol": skill2bench_protocol("9b"),
        "campaign_sha256": "a" * 64,
    }
    with pytest.raises(SystemicProducerTransportFailure, match="transport failed"):
        asyncio.run(campaign._run_population(**kwargs))
    assert not list((tmp_path / "population/items").glob("*.json"))

    rollouts, evaluations = asyncio.run(campaign._run_population(**kwargs))
    assert len(calls) == 6
    assert len(rollouts) == len(evaluations) == 3


def test_population_receipt_detects_modified_cached_output(tmp_path: Path):
    artifact = tmp_path / "items/000.json"
    receipt = tmp_path / "receipts/000.json"
    rollout = {"instance_id": "task-0", "answer": "ok"}
    evaluation = {"instance_id": "task-0", "score": 1.0}
    campaign._write_population_item(
        artifact,
        receipt,
        request_sha256="a" * 64,
        rollout=rollout,
        evaluation=evaluation,
    )
    assert campaign._load_population_item(
        artifact, receipt, request_sha256="a" * 64, index=0
    ) == (rollout, evaluation)

    changed = json.loads(artifact.read_text())
    changed["evaluation"]["score"] = 0.0
    artifact.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="identity differs"):
        campaign._load_population_item(
            artifact, receipt, request_sha256="a" * 64, index=0
        )


def test_population_transport_wave_with_a_success_is_item_local(
    tmp_path: Path, monkeypatch
):
    async def direct_to_thread(function, **kwargs):
        return function(**kwargs)

    def transport(**kwargs):
        if kwargs["task"]["instance_id"] != "task-3":
            raise ConnectionError("connection refused")
        return (
            {"instance_id": "task-3", "answer": "ok", "react_steps": []},
            {"instance_id": "task-3", "score": 1.0, "steps": []},
        )

    monkeypatch.setattr(campaign, "run_and_evaluate_task", transport)
    monkeypatch.setattr(campaign.asyncio, "to_thread", direct_to_thread)
    tasks = [
        {"instance_id": f"task-{index}", "steps": []}
        for index in range(4)
    ]
    rollouts, evaluations = asyncio.run(
        campaign._run_population(
            tasks=tasks,
            output_root=tmp_path / "population",
            skill_by_index={index: None for index in range(4)},
            baseline_root=tmp_path,
            official_evaluator_root=tmp_path,
            base_url="http://127.0.0.1:18081/v1",
            api_key="test",
            protocol=skill2bench_protocol("9b"),
            campaign_sha256="a" * 64,
        )
    )
    assert len(rollouts) == len(evaluations) == 4
    assert sum(bool(row.get("item_local_failure")) for row in rollouts) == 3


def test_worker_failure_envelope_preserves_terminal_http_status():
    failure = Skill2BenchWorkerError(
        "AuthenticationError: Error code: 401",
        worker_error_type="AuthenticationError",
        status_code=401,
        transport_failure=True,
    )
    assert campaign._transport_failure(failure) is failure

    service_failure = Skill2BenchWorkerError(
        "InternalServerError: Error code: 500",
        worker_error_type="InternalServerError",
        status_code=500,
        transport_failure=True,
    )
    assert campaign._transport_failure(service_failure) is service_failure


def test_in_process_patch_api_failures_enter_transport_guard():
    request = httpx.Request("POST", "http://127.0.0.1:18081/v1/chat/completions")
    for status_code, error_type in (
        (429, RateLimitError),
        (500, InternalServerError),
    ):
        error = error_type(
            f"Error code: {status_code}",
            response=httpx.Response(status_code, request=request),
            body=None,
        )
        assert campaign._transport_failure(error) is error


def test_single_repair_transport_failure_does_not_abort_population(tmp_path: Path):
    evidence = asyncio.run(
        campaign._collect_train_evidence(
            tasks=[_task()],
            rollouts=[
                {
                    "instance_id": "task-a",
                    "react_steps": [{"thought": "Step 1: attempted target"}],
                }
            ],
            evaluations=[
                {
                    "steps": [
                        {"step": 1, "status": "scored", "score": 0.0}
                    ]
                }
            ],
            run_root=tmp_path,
            patch_llm=_TimeoutLLM(),
            baseline_root=tmp_path,
            official_evaluator_root=tmp_path,
            base_url="http://127.0.0.1:18081/v1",
            api_key="test",
            protocol=skill2bench_protocol("9b"),
            campaign_sha256="a" * 64,
        )
    )
    assert evidence == {0: []}


def test_step_source_uses_shared_zero_success_transport_wave_policy():
    tasks = {
        index: public_task_view(
            {**_task(), "instance_id": f"task-{index}"}
        )
        for index in range(3)
    }
    evidence = {
        index: [
            validated_repair_record(
                evidence_id=f"train-{index:03d}::step-01::repair-01",
                step_number=1,
                memory=ValidatedRepairMemory(("Use the target rule.",), ("Check it.",)),
                successful_replay={
                    "react_steps": [{"thought": "Step 1: apply target rule"}]
                },
            )
        ]
        for index in range(3)
    }
    with pytest.raises(SystemicProducerTransportFailure, match="transport failed"):
        asyncio.run(
            build_source_batch(
                batch_task_indices=(0, 1, 2),
                public_tasks=tasks,
                evidence_by_index=evidence,
                extractor=StepExperienceExtractor(_TimeoutLLM()),
                reviewer=ExperienceSourceReviewer(_ReviewLLM()),
            )
        )


def test_official_evaluator_identity_includes_examples_dependency(tmp_path: Path):
    root = tmp_path / "official"
    (root / "calculate_skill_entropy").mkdir(parents=True)
    math_root = root / "evaluation/math_evaluation"
    math_root.mkdir(parents=True)
    (math_root / "latex2sympy").mkdir()
    (root / "calculate_skill_entropy/calculate_entropy_w_outputs.py").write_text("main")
    for name in ("parser.py", "grader.py", "utils.py", "examples.py"):
        (math_root / name).write_text(name)
    before = campaign._official_evaluator_sha256(root)
    (math_root / "examples.py").write_text("changed")
    assert campaign._official_evaluator_sha256(root) != before
    (math_root / "examples.py").unlink()
    with pytest.raises(ValueError, match="dependency closure differs"):
        campaign._official_evaluator_sha256(root)
