from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from openai import APIError

from react_agent.models import (
    RequestCompletionLengthExceeded,
    RequestContextLengthExceeded,
    RequestRuntimeTimeout,
)
import degs.source_rebuild as source_rebuild_module
import degs.successful_source as successful_source_module
import degs.source_replay as source_replay_module
from degs.core import canonical_json_bytes
from degs.dataset import DATASET_SHA256
from degs.incremental_graph import _validate_batch_source_audit
from degs.section_graph import SECTION_GRAPH_FORMAT, load_section_graphs
from degs.source_rebuild import (
    SOURCE_EXTRACTION_SEMANTIC_ATTEMPTS,
    SOURCE_REBUILD_WORKERS,
    SOURCE_REVIEW_RETRY_STATUS,
    rebuild_section_source_parallel as _rebuild_section_source_parallel,
)
from degs.source_review import (
    SOURCE_REVIEW_KIND,
    SOURCE_REVIEW_PROMPT_SHA256,
    SOURCE_REVIEW_PROTOCOL_FORMAT,
    ExperienceSourceReviewer,
)
from degs.source_replay import (
    SOURCE_REPLAY_OUTCOME_FORMAT,
    SOURCE_REPLAY_BASH_TIMEOUT_SECONDS,
    SOURCE_REPLAY_EXECUTOR_RETRY_WAITS,
    SOURCE_REPLAY_MAX_TURNS,
    SOURCE_REPLAY_MODEL,
    SOURCE_REPLAY_PATCH_KIND,
    SOURCE_REPLAY_PATCH_MAX_TOKENS,
    SOURCE_REPLAY_PATCH_RETRY_WAITS,
    SOURCE_REPLAY_PATCH_RUNTIME_TIMEOUT_RETRIES,
    SOURCE_REPLAY_PATCH_PROMPT_SHA256,
    SOURCE_REPLAY_PROTOCOL_FORMAT,
    SOURCE_REPLAY_TIMEOUT_SECONDS,
    _patch_sha256,
    _stable_patch_id,
    parse_source_replay_patch,
    source_replay_patch_response_schema,
    validate_source_replay_outcome_protocol,
)
from degs.successful_source import (
    SUCCESS_EXTRACTION_KIND,
    SUCCESS_PROMPT_SHA256,
    SUCCESS_SOURCE_PROTOCOL_FORMAT,
    SUCCESS_SYSTEM_PROMPT,
    SuccessfulTrajectoryExperienceExtractor,
    success_response_schema,
)
from degs.validated_repair import (
    PRODUCER_RUNTIME_TIMEOUT_RETRIES,
    PRODUCER_TRANSPORT_RETRY_WAITS,
    OpenAIJsonObjectLLM,
    REPAIR_SYSTEM_PROMPT,
    REPAIR_SOURCE_MAX_TOKENS,
    REPAIR_SOURCE_MODEL,
    REPAIR_SOURCE_TEMPERATURE,
    REPAIR_SOURCE_THINKING,
    REPAIR_SOURCE_TIMEOUT_SECONDS,
    SOURCE_RAW_RESPONSE_FORMAT,
    SystemicProducerTransportFailure,
    ValidatedRepairExperienceExtractor,
    _source_generation_config,
    _parse_llm_experience_graph,
    experience_node_schema,
    parse_experience_nodes,
    producer_transport_failure_policy,
    repair_response_schema,
    render_successful_replay,
    select_validated_repair_example,
)


class ScriptedLLM:
    def __init__(self, response):
        self.response = response
        self.calls = []

    @property
    def protocol_identity(self):
        return {"format": "synthetic_source_protocol_v2", "model": "synthetic"}

    def complete_json(self, **request):
        self.calls.append(request)
        return json.loads(json.dumps(self.response))

    async def complete_json_async(self, **request):
        await asyncio.sleep(0)
        return self.complete_json(**request)


class _PassThroughReviewLLM:
    @property
    def protocol_identity(self):
        return {
            "format": SOURCE_REVIEW_PROTOCOL_FORMAT,
            "request_kind": SOURCE_REVIEW_KIND,
            "model": REPAIR_SOURCE_MODEL,
            "temperature": REPAIR_SOURCE_TEMPERATURE,
            "thinking": REPAIR_SOURCE_THINKING,
            "max_tokens": REPAIR_SOURCE_MAX_TOKENS,
            "timeout_seconds": REPAIR_SOURCE_TIMEOUT_SECONDS,
            "generation_config": _source_generation_config(),
            "retry_waits_seconds": list(PRODUCER_TRANSPORT_RETRY_WAITS),
            "runtime_timeout_retries": PRODUCER_RUNTIME_TIMEOUT_RETRIES,
            "prompt_sha256": SOURCE_REVIEW_PROMPT_SHA256,
            "service_url": "http://127.0.0.1:9999/v1",
        }

    async def complete_json_async(self, **request):
        draft = request["payload"]["draft_graph"]
        return {
            **json.loads(json.dumps(draft)),
            "review_decisions": [
                {
                    "draft_node": index,
                    "decision": "KEEP",
                    "final_nodes": [index],
                    "basis": "The draft node is causally supported and reusable.",
                }
                for index, _row in enumerate(draft["experience_nodes"])
            ],
        }


def rebuild_section_source_parallel(**kwargs):
    kwargs.setdefault(
        "reviewer_factory",
        lambda _raw_path: ExperienceSourceReviewer(_PassThroughReviewLLM()),
    )
    return _rebuild_section_source_parallel(**kwargs)


def _successful_trajectory(task_id: str = "synthetic-task") -> dict:
    trajectory_id = f"{task_id}::source_replay::accepted"
    return {
        "task_id": task_id,
        "trajectory_id": trajectory_id,
        "instruction": "Repair only the inconsistent region and preserve valid state.",
        "success": True,
        "steps": [
            {
                "step_id": 1,
                "action": "Inspect the current artifact and identify the inconsistent region.",
                "observation": "One region is inconsistent; unrelated state is valid.",
                "action_valid": True,
                "tool_name": "Inspect",
            },
            {
                "step_id": 2,
                "action": "Update only the inconsistent region.",
                "observation": "The scoped update completed.",
                "action_valid": True,
                "tool_name": "Edit",
            },
        ],
        "final_response": "The scoped repair was applied.",
        "extra": {
            "source_replay": {
                "attempt_index": 1,
                "patch_id": _accepted_patch_id(task_id),
                "source_task_id": task_id,
                "parent_trajectory_id": f"{task_id}::parent-failure",
                "source_replay_protocol_sha256": _source_replay_protocol_sha256(),
            }
        },
    }


def _source_replay_protocol() -> dict:
    return {
        "format": SOURCE_REPLAY_PROTOCOL_FORMAT,
        "patch_request_kind": SOURCE_REPLAY_PATCH_KIND,
        "patch_prompt_sha256": SOURCE_REPLAY_PATCH_PROMPT_SHA256,
        "patch_response_schema_sha256": hashlib.sha256(
            canonical_json_bytes(source_replay_patch_response_schema())
        ).hexdigest(),
        "no_input_truncation": True,
        "fresh_input_each_attempt": True,
        "max_attempts": 3,
        "patch_llm": {
            "format": SOURCE_REPLAY_PROTOCOL_FORMAT,
            "request_kind": SOURCE_REPLAY_PATCH_KIND,
            "model": SOURCE_REPLAY_MODEL,
            "temperature": 0,
            "thinking": False,
            "max_tokens": SOURCE_REPLAY_PATCH_MAX_TOKENS,
            "timeout_seconds": SOURCE_REPLAY_TIMEOUT_SECONDS,
            "retry_waits_seconds": list(SOURCE_REPLAY_PATCH_RETRY_WAITS),
            "runtime_timeout_retries": SOURCE_REPLAY_PATCH_RUNTIME_TIMEOUT_RETRIES,
            "generation_config": {
                "temperature": 0,
                "max_tokens": SOURCE_REPLAY_PATCH_MAX_TOKENS,
                "extra_body": {
                    "chat_template_kwargs": {"enable_thinking": False}
                },
            },
            "prompt_sha256": SOURCE_REPLAY_PATCH_PROMPT_SHA256,
            "service_url": "http://127.0.0.1:18081/v1",
        },
        "attempt_executor": {
            "format": "degs_fresh_replay_executor_v1",
            "dataset_json_sha256": DATASET_SHA256,
            "model": SOURCE_REPLAY_MODEL,
            "base_url": "http://127.0.0.1:18081/v1",
            "max_turns": SOURCE_REPLAY_MAX_TURNS,
            "max_completion_tokens": SOURCE_REPLAY_PATCH_MAX_TOKENS,
            "bash_timeout": SOURCE_REPLAY_BASH_TIMEOUT_SECONDS,
            "llm_timeout": SOURCE_REPLAY_TIMEOUT_SECONDS,
            "retry_waits": list(SOURCE_REPLAY_EXECUTOR_RETRY_WAITS),
            "workers": 1,
            "outer_task_workers": 8,
            "temperature": 0,
            "thinking": False,
            "fresh_input_each_attempt": True,
            "context_overflow_reporting": "machine_readable_marker_v1",
            "observation_policy": "full_no_truncation",
        },
    }


def _source_replay_protocol_sha256() -> str:
    return hashlib.sha256(
        canonical_json_bytes(_source_replay_protocol())
    ).hexdigest()


def _with_source_replay_protocol(outcome: dict) -> dict:
    outcome["format"] = SOURCE_REPLAY_OUTCOME_FORMAT
    outcome["source_replay_protocol"] = _source_replay_protocol()
    outcome["source_replay_protocol_sha256"] = _source_replay_protocol_sha256()
    return outcome


@pytest.mark.parametrize(
    "mutate",
    (
        lambda protocol: protocol["attempt_executor"].__setitem__(
            "dataset_json_sha256", "0" * 64
        ),
        lambda protocol: protocol["attempt_executor"].__setitem__(
            "max_turns", 29
        ),
    ),
)
def test_current_replay_protocol_rejects_protocol_drift(
    mutate,
) -> None:
    outcome = _with_source_replay_protocol(
        {
            "task_id": "synthetic",
            "parent_trajectory_id": "synthetic::parent",
            "status": "REPLAY_EXHAUSTED",
            "attempts": [],
        }
    )
    protocol = copy.deepcopy(outcome["source_replay_protocol"])
    mutate(protocol)
    outcome["source_replay_protocol"] = protocol
    outcome["source_replay_protocol_sha256"] = hashlib.sha256(
        canonical_json_bytes(protocol)
    ).hexdigest()
    with pytest.raises(ValueError, match="current no-truncation identity"):
        validate_source_replay_outcome_protocol(outcome)


def test_replay_protocol_accepts_runtime_endpoint_and_worker_changes() -> None:
    outcome = _with_source_replay_protocol(
        {
            "task_id": "synthetic",
            "parent_trajectory_id": "synthetic::parent",
            "status": "REPLAY_EXHAUSTED",
            "attempts": [],
        }
    )
    protocol = copy.deepcopy(outcome["source_replay_protocol"])
    protocol["patch_llm"]["service_url"] = "http://127.0.0.1:28081/v1"
    protocol["attempt_executor"]["base_url"] = "http://127.0.0.1:38081/v1"
    protocol["attempt_executor"]["workers"] = 4
    protocol["attempt_executor"]["outer_task_workers"] = 64
    outcome["source_replay_protocol"] = protocol
    outcome["source_replay_protocol_sha256"] = hashlib.sha256(
        canonical_json_bytes(protocol)
    ).hexdigest()

    validate_source_replay_outcome_protocol(outcome)


def _patch_payload() -> dict:
    return {
        "diagnosis": "The failed attempt mutated state before isolating the inconsistency.",
        "instructions": [
            "Identify the inconsistent region before editing.",
            "Modify only that region and preserve valid state.",
        ],
        "checks": [
            "Confirm the repaired state satisfies the requested constraint."
        ],
    }


def _parsed_patch(payload: dict):
    return parse_source_replay_patch(payload)


def _accepted_patch_id(task_id: str) -> str:
    return _stable_patch_id(
        f"{task_id}::parent-failure",
        1,
        _parsed_patch(_patch_payload()),
    )


def _reseal_content_identity(outcome: dict, trajectory: dict) -> None:
    attempt = outcome["attempts"][0]
    parsed = _parsed_patch(attempt["patch"])
    patch_id = _stable_patch_id(outcome["parent_trajectory_id"], 1, parsed)
    outcome["accepted_patch_id"] = patch_id
    attempt["patch_id"] = patch_id
    attempt["patch_sha256"] = _patch_sha256(parsed)
    trajectory["extra"]["source_replay"]["patch_id"] = patch_id
    attempt["replay_trajectory_sha256"] = hashlib.sha256(
        canonical_json_bytes(trajectory)
    ).hexdigest()


def _outcome(task_id: str = "synthetic-task") -> dict:
    trajectory_id = f"{task_id}::source_replay::accepted"
    outcome = _with_source_replay_protocol({
        "task_id": task_id,
        "parent_trajectory_id": f"{task_id}::parent-failure",
        "status": "REPLAY_VALIDATED_SUCCESS",
        "accepted_attempt_index": 1,
        "accepted_patch_id": _accepted_patch_id(task_id),
        "accepted_trajectory_id": trajectory_id,
        "attempts": [
            {
                "attempt_index": 1,
                "task_id": task_id,
                "success": True,
                "patch_id": _accepted_patch_id(task_id),
                "replay_trajectory_id": trajectory_id,
                "replay_trajectory_path": "accepted.json",
                "patch": _patch_payload(),
            }
        ],
    })
    _reseal_content_identity(outcome, _successful_trajectory(task_id))
    return outcome


def _experience_response() -> dict:
    return {
        "experience_nodes": [
            {
                "operation": (
                    "Inspect the current artifact to isolate inconsistent state while "
                    "recording the valid state that must remain unchanged."
                ),
                "applicability": [
                    "The artifact contains a localized inconsistency while surrounding state must be preserved."
                ],
                "inputs": [
                    {
                        "type": "current task artifact",
                        "description": "The external artifact before repair.",
                    }
                ],
                "outputs": [
                    {
                        "type": "scoped inconsistency evidence",
                        "description": (
                            "The inconsistent region together with surrounding valid state."
                        ),
                    }
                ],
            },
            {
                "operation": (
                    "Use the scoped evidence and accepted repair constraints to select "
                    "the smallest state change that fixes the inconsistency."
                ),
                "applicability": [
                    "Scoped inconsistency evidence and explicit preservation constraints are available."
                ],
                "inputs": [
                    {
                        "type": "scoped inconsistency evidence",
                        "description": "The region to repair and state to preserve.",
                    },
                    {
                        "type": "accepted repair constraints",
                        "description": "The operation boundary and completion conditions.",
                    },
                ],
                "outputs": [
                    {
                        "type": "bounded repair decision",
                        "description": "The exact repair scope and preservation constraints.",
                    }
                ],
            },
            {
                "operation": (
                    "Apply only the bounded repair decision and preserve all unrelated "
                    "valid state."
                ),
                "applicability": [
                    "A bounded repair decision identifies both the mutation and the state that must remain unchanged."
                ],
                "inputs": [
                    {
                        "type": "bounded repair decision",
                        "description": "The repair operation and state-preservation boundary.",
                    },
                    {
                        "type": "current task artifact",
                        "description": "The external artifact immediately before mutation.",
                    },
                ],
                "outputs": [
                    {
                        "type": "repaired task artifact",
                        "description": (
                            "The artifact with the inconsistency corrected and valid state preserved."
                        ),
                    }
                ],
            },
        ],
        "edges": [
            {"source": 0, "target": 1},
            {"source": 1, "target": 2},
        ],
    }


def _all_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _all_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _all_keys(item)


def _without_active_exclusion_identity(row: dict) -> dict:
    value = dict(row)
    protocol = value.pop("source_protocol")
    assert value.pop("prompt_sha256") == protocol["prompt_sha256"]
    assert value.pop("source_protocol_sha256") == hashlib.sha256(
        canonical_json_bytes(protocol)
    ).hexdigest()
    assert len(value.pop("request_payload_sha256")) == 64
    assert value.pop("response_schema_sha256") == hashlib.sha256(
        canonical_json_bytes(repair_response_schema())
    ).hexdigest()
    return value


def test_shared_schema_is_open_vocabulary_and_has_no_old_ontology() -> None:
    node = experience_node_schema()
    response = repair_response_schema()

    assert success_response_schema() == response
    assert set(node["properties"]) == {
        "operation", "applicability", "inputs", "outputs"
    }
    assert set(response["properties"]) == {"experience_nodes", "edges"}
    assert set(response["properties"]["edges"]["items"]["properties"]) == {
        "source",
        "target",
    }
    assert response["properties"]["experience_nodes"]["minItems"] == 0
    assert "enum" not in node["properties"]["operation"]
    assert node["properties"]["applicability"]["minItems"] == 1
    assert "maxLength" not in set(_all_keys(response))
    assert "uniqueItems" not in set(_all_keys(response))
    assert {
        "effect",
        "input_types",
        "output_types",
        "repair_alignment",
        "section_indices",
        "successful_turn_ids",
    }.isdisjoint(_all_keys(response))


def test_open_contract_order_and_long_operation_are_preserved() -> None:
    response = _experience_response()
    response["experience_nodes"][1]["operation"] = "x" * 20_000
    response["experience_nodes"][1]["inputs"] = [
        {"type": "zeta evidence", "description": "First semantic contract."},
        {"type": "alpha constraint", "description": "Second semantic contract."},
    ]

    nodes = parse_experience_nodes(response["experience_nodes"])

    assert len(nodes[1].operation) == 20_000
    assert [row.type for row in nodes[1].inputs] == [
        "zeta evidence",
        "alpha constraint",
    ]


def test_empty_experience_graph_is_valid_but_cannot_have_edges() -> None:
    assert _parse_llm_experience_graph(
        {"experience_nodes": [], "edges": []}
    ) == ((), (), ())
    with pytest.raises(ValueError, match="empty experience graph"):
        _parse_llm_experience_graph(
            {"experience_nodes": [], "edges": [{"source": 0, "target": 1}]}
        )


def test_validated_repair_memory_has_no_old_instruction_or_check_count_caps() -> None:
    outcome = _outcome()
    outcome["attempts"][0]["patch"]["instructions"] = [
        f"Instruction {index} with complete scope." for index in range(12)
    ]
    outcome["attempts"][0]["patch"]["checks"] = [
        f"Check {index} against observable state." for index in range(9)
    ]
    trajectory = _successful_trajectory()
    _reseal_content_identity(outcome, trajectory)

    example = select_validated_repair_example(outcome, trajectory)

    assert len(example.memory.instructions) == 12
    assert len(example.memory.checks) == 9


def test_old_replay_outcome_without_new_no_truncation_identity_is_rejected() -> None:
    outcome = _outcome()
    del outcome["format"]
    del outcome["source_replay_protocol"]
    del outcome["source_replay_protocol_sha256"]

    with pytest.raises(ValueError, match="source replay protocol"):
        select_validated_repair_example(outcome, _successful_trajectory())


def test_replay_outcome_with_model_drift_is_rejected_even_if_rehashed() -> None:
    outcome = _outcome()
    outcome["source_replay_protocol"]["patch_llm"]["model"] = "different-model"
    outcome["source_replay_protocol_sha256"] = hashlib.sha256(
        canonical_json_bytes(outcome["source_replay_protocol"])
    ).hexdigest()

    with pytest.raises(ValueError, match="source replay protocol"):
        select_validated_repair_example(outcome, _successful_trajectory())


@pytest.mark.parametrize(
    ("edge", "reason"),
    [
        (
            {"source": 1, "target": 1},
            "experience edge must reference a forward pair of distinct nodes",
        ),
        (
            {"source": 2, "target": 1},
            "experience edge must reference a forward pair of distinct nodes",
        ),
        (
            {"source": -1, "target": 1},
            "experience edge must reference a forward pair of distinct nodes",
        ),
        (
            {"source": 0, "target": 3},
            "experience edge must reference a forward pair of distinct nodes",
        ),
        (
            {"source": "0", "target": 1},
            "experience edge must reference a forward pair of distinct nodes",
        ),
        ({"source": 0}, "experience edge fields differ"),
        ("0 -> 1", "experience edge fields differ"),
    ],
)
def test_invalid_edges_are_discarded_without_losing_nodes(edge, reason) -> None:
    response = _experience_response()
    response["edges"] = [{"source": 0, "target": 1}, edge]

    nodes, edges, discarded = _parse_llm_experience_graph(response)

    assert len(nodes) == 3
    assert [(row.source, row.target) for row in edges] == [(0, 1)]
    assert discarded == (f"edge[1]: {reason}",)


def test_edge_order_has_no_semantics_and_is_canonicalized() -> None:
    response = _experience_response()
    response["edges"] = [
        {"source": 1, "target": 2},
        {"source": 0, "target": 1},
    ]

    _nodes, edges, discarded = _parse_llm_experience_graph(response)

    assert [(edge.source, edge.target) for edge in edges] == [(0, 1), (1, 2)]
    assert discarded == ()


def test_duplicate_edges_are_deduplicated_and_recorded() -> None:
    response = _experience_response()
    response["edges"] = [
        {"source": 0, "target": 1},
        {"source": 0, "target": 1},
    ]

    _nodes, edges, discarded = _parse_llm_experience_graph(response)

    assert [(edge.source, edge.target) for edge in edges] == [(0, 1)]
    assert discarded == ("edge[1]: duplicate experience edge",)


def test_extraction_prompts_use_domain_general_validation_and_experience_rules() -> None:
    for prompt in (SUCCESS_SYSTEM_PROMPT, REPAIR_SYSTEM_PROMPT):
        normalized = " ".join(prompt.split())
        assert "sorted unique zero-based" not in normalized
        assert "sorted and duplicate-free" not in normalized
        assert "coordinate convention" not in normalized
        assert "coordinate bases" not in normalized
        assert '"edges"' in normalized
        assert "0 <= source < target < len(experience_nodes)" in normalized
        assert "Node indices are zero-based" in normalized
        assert "validation_feedback" not in normalized
        assert "depends_on" not in normalized
        assert "task-specific postconditions" not in normalized
        assert "merely saving a requested artifact" not in normalized
        assert "output file exists" not in normalized
        assert "task-specific validation logic" in normalized
        assert "Applicability is an open natural-language usage description" in normalized
        assert "predefined ontology" in normalized
    assert "original failure" in REPAIR_SYSTEM_PROMPT


def test_only_accepted_memory_and_successful_replay_enter_repair_request() -> None:
    llm = ScriptedLLM(_experience_response())
    example = select_validated_repair_example(_outcome(), _successful_trajectory())

    result = ValidatedRepairExperienceExtractor(llm).extract(example)

    assert result.experience_node_dicts() == _experience_response()["experience_nodes"]
    call = llm.calls[0]
    assert set(call["payload"]) == {
        "validated_repair_memory",
        "successful_replay_trajectory",
    }
    serialized = json.dumps(call["payload"], sort_keys=True)
    for forbidden in (
        "parent-failure",
        "repair_alignment",
        "section_indices",
        "successful_turn_ids",
        "verifier_feedback",
        "verifier_score",
    ):
        assert forbidden not in serialized


def test_accepted_patch_content_is_bound_to_its_replay_identity() -> None:
    outcome = _outcome()
    outcome["attempts"][0]["patch"]["instructions"][0] = "Changed after replay."

    with pytest.raises(ValueError, match="patch content identity"):
        select_validated_repair_example(outcome, _successful_trajectory())


def test_successful_replay_content_is_bound_to_the_accepted_attempt() -> None:
    trajectory = _successful_trajectory()
    trajectory["steps"][0]["observation"] = "Changed after verification."

    with pytest.raises(ValueError, match="trajectory content identity"):
        select_validated_repair_example(_outcome(), trajectory)


def test_ordinary_and_repair_extractors_accept_the_same_response_shape() -> None:
    response = _experience_response()
    ordinary_llm = ScriptedLLM(response)
    repair_llm = ScriptedLLM(response)

    ordinary = SuccessfulTrajectoryExperienceExtractor(ordinary_llm).extract(
        _successful_trajectory("ordinary-task")
    )
    repair = ValidatedRepairExperienceExtractor(repair_llm).extract(
        select_validated_repair_example(_outcome(), _successful_trajectory())
    )

    assert ordinary.experience_node_dicts() == repair.experience_node_dicts()
    assert ordinary.edge_dicts() == repair.edge_dicts()
    assert set(ordinary_llm.calls[0]["response_schema"]["properties"]) == {
        "experience_nodes",
        "edges",
    }


def test_accepted_replay_must_be_the_unique_final_success() -> None:
    outcome = _outcome()
    outcome["attempts"].append(
        {
            **outcome["attempts"][0],
            "attempt_index": 2,
            "success": False,
        }
    )
    with pytest.raises(ValueError, match="final attempt|unique final success"):
        select_validated_repair_example(outcome, _successful_trajectory())


def test_rendered_trajectory_preserves_every_input_character_without_truncation_metadata() -> None:
    trajectory = _successful_trajectory()
    long_instruction = "instruction-start\n" + "i" * 20_000 + "\ninstruction-end"
    long_action = "action-start\n" + "a" * 17_000 + "\naction-end"
    long_raw = "reasoning-start\n" + "r" * 19_000 + "\nreasoning-end"
    long_observation = (
        "observation-start\n" + "o" * 150_000 + "\nobservation-end"
    )
    long_final = "final-start\n" + "f" * 18_000 + "\nfinal-end"
    trajectory["instruction"] = long_instruction
    trajectory["steps"][0]["action"] = long_action
    trajectory["steps"][0]["raw_model_output"] = long_raw
    trajectory["steps"][0]["observation"] = long_observation
    trajectory["final_response"] = long_final

    rendered = render_successful_replay(trajectory)

    assert set(rendered.__dict__) == {"payload"}
    assert rendered.payload["turns"][0]["content"] == long_instruction
    assert rendered.payload["turns"][1]["action"] == long_action
    assert rendered.payload["turns"][1]["raw_model_output"] == long_raw
    assert rendered.payload["turns"][1]["observation"] == long_observation
    assert rendered.payload["turns"][-1]["content"] == long_final
    assert "input_truncated" not in set(_all_keys(rendered.payload))


def _source_client(**overrides):
    fields = {
        "model": REPAIR_SOURCE_MODEL,
        "timeout": REPAIR_SOURCE_TIMEOUT_SECONDS,
        "retry_times": PRODUCER_TRANSPORT_RETRY_WAITS,
        "runtime_timeout_retries": PRODUCER_RUNTIME_TIMEOUT_RETRIES,
        "generation_config": {
            "temperature": REPAIR_SOURCE_TEMPERATURE,
            "max_tokens": REPAIR_SOURCE_MAX_TOKENS,
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": REPAIR_SOURCE_THINKING}
            },
        },
        "base_url": "http://synthetic.invalid/v1",
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_source_llm_adapter_rejects_protocol_drift() -> None:
    with pytest.raises(ValueError, match="retry policy"):
        OpenAIJsonObjectLLM(_source_client(retry_times=(1,)))


def test_source_llm_request_has_one_system_and_one_user_message() -> None:
    llm = OpenAIJsonObjectLLM(_source_client())

    messages, _settings = llm._request_parts(
        kind=llm.request_kind,
        system_prompt=REPAIR_SYSTEM_PROMPT,
        payload={"source": {}},
        response_schema=repair_response_schema(),
    )

    assert [message.role for message in messages] == ["system", "user"]


def test_source_llm_protocol_follows_selected_model_profile(monkeypatch) -> None:
    monkeypatch.setenv("DEGS_MODEL", "Qwen3.5-27B-AWQ")
    llm = OpenAIJsonObjectLLM(_source_client(model="Qwen3.5-27B-AWQ"))

    assert llm.protocol_identity["model"] == "Qwen3.5-27B-AWQ"


def test_source_rebuild_uses_experience_nodes_and_no_alignment_metadata(
    tmp_path: Path,
) -> None:
    original_success = _successful_trajectory("synthetic-original")
    original_records = [original_success]
    replay_parent = {
        "task_id": "synthetic-task",
        "trajectory_id": "synthetic-task::parent-failure",
        "instruction": "Repair the synthetic artifact while preserving valid state.",
        "success": False,
    }
    original_records.append(replay_parent)
    for index in range(2, 200):
        original_records.append(
            {
                "task_id": f"synthetic-failed-{index:03d}",
                "trajectory_id": f"synthetic-failed-{index:03d}::parent",
                "instruction": f"Complete synthetic task {index:03d}.",
                "success": False,
            }
        )
    replay_outcomes = [_outcome()]
    replay_outcomes[0]["attempts"][0]["replay_trajectory_path"] = "accepted.json"
    replay_outcomes.extend(
        _with_source_replay_protocol({
            "task_id": f"synthetic-failed-{index:03d}",
            "parent_trajectory_id": f"synthetic-failed-{index:03d}::parent",
            "status": "REPLAY_EXHAUSTED",
            "attempts": [],
        })
        for index in range(2, 200)
    )

    build = rebuild_section_source_parallel(
        original_records=original_records,
        replay_outcomes=replay_outcomes,
        replay_trajectory_loader=lambda _path: _successful_trajectory(),
        extractor_factory=lambda origin, _raw_path: (
            SuccessfulTrajectoryExperienceExtractor(
                ScriptedLLM(_experience_response())
            )
            if origin == "ORIGINAL_SUCCESS"
            else ValidatedRepairExperienceExtractor(
                ScriptedLLM(_experience_response())
            )
        ),
        checkpoint_dir=tmp_path / "reviewed-source-checkpoints",
        selected_train_indices=range(8),
    )

    assert build.section_graphs["format"] == SECTION_GRAPH_FORMAT
    assert [row["train_index"] for row in build.section_graphs["workflows"]] == [0, 1]
    assert all(
        set(row)
        == {"train_index", "task_id", "query_text", "experience_nodes", "edges"}
        for row in build.section_graphs["workflows"]
    )
    assert {
        "input_truncated",
        "repair_alignment",
        "alignment_normalized",
        "section_indices",
        "successful_turn_ids",
    }.isdisjoint(_all_keys(build.source_audit))
    path = tmp_path / "experience_graphs.json"
    path.write_bytes(canonical_json_bytes(build.section_graphs))
    assert len(load_section_graphs(path).workflows) == 2


def test_parallel_source_rebuild_selects_exactly_one_eight_index_batch(
    tmp_path: Path,
) -> None:
    records = [
        _successful_trajectory(f"synthetic-success-{index:03d}")
        for index in range(200)
    ]

    class BatchLLM(ScriptedLLM):
        @property
        def protocol_identity(self):
            return {
                **super().protocol_identity,
                "prompt_sha256": SUCCESS_PROMPT_SHA256,
            }

        async def complete_json_async(self, **request):
            await asyncio.sleep(0)
            return self.complete_json(**request)

    build = rebuild_section_source_parallel(
        original_records=records,
        replay_outcomes=[],
        replay_trajectory_loader=lambda _path: _successful_trajectory(),
        extractor_factory=lambda _origin, _raw_path: (
            SuccessfulTrajectoryExperienceExtractor(BatchLLM(_experience_response()))
        ),
        checkpoint_dir=tmp_path / "checkpoints",
        selected_train_indices=range(8),
    )
    assert [
        row["train_index"] for row in build.section_graphs["workflows"]
    ] == list(range(8))
    assert len(list((tmp_path / "checkpoints").rglob("extraction.json"))) == 8

    changed_records = [dict(row) for row in records]
    changed_records[0]["instruction"] += " Changed after the checkpoint."
    with pytest.raises(ValueError, match="checkpoint identity differs"):
        rebuild_section_source_parallel(
            original_records=changed_records,
            replay_outcomes=[],
            replay_trajectory_loader=lambda _path: _successful_trajectory(),
            extractor_factory=lambda _origin, _raw_path: (
                SuccessfulTrajectoryExperienceExtractor(
                    BatchLLM(_experience_response())
                )
            ),
            checkpoint_dir=tmp_path / "checkpoints",
            selected_train_indices=range(8),
        )

    with pytest.raises(ValueError, match="exactly 8 train indices"):
        rebuild_section_source_parallel(
            original_records=records,
            replay_outcomes=[],
            replay_trajectory_loader=lambda _path: _successful_trajectory(),
            extractor_factory=lambda _origin, _raw_path: (
                SuccessfulTrajectoryExperienceExtractor(
                    BatchLLM(_experience_response())
                )
            ),
            checkpoint_dir=tmp_path / "invalid-checkpoints",
            selected_train_indices=range(7),
        )
    with pytest.raises(ValueError, match="aligned contiguous"):
        rebuild_section_source_parallel(
            original_records=records,
            replay_outcomes=[],
            replay_trajectory_loader=lambda _path: _successful_trajectory(),
            extractor_factory=lambda _origin, _raw_path: (
                SuccessfulTrajectoryExperienceExtractor(
                    BatchLLM(_experience_response())
                )
            ),
            checkpoint_dir=tmp_path / "misaligned-checkpoints",
            selected_train_indices=range(1, 9),
        )


def test_incremental_source_batch_audits_every_missing_replay_source(
    tmp_path: Path,
) -> None:
    records = [_successful_trajectory("synthetic-kept")]
    outcomes = []
    for index in range(1, 200):
        task_id = f"synthetic-failed-{index:03d}"
        records.append(
            {
                "task_id": task_id,
                "trajectory_id": f"{task_id}::parent",
                "success": False,
            }
        )
        outcomes.append(
            _with_source_replay_protocol(
                {
                    "task_id": task_id,
                    "parent_trajectory_id": f"{task_id}::parent",
                    "status": "REPLAY_EXHAUSTED",
                    "attempts": [],
                }
            )
        )

    class BatchLLM(ScriptedLLM):
        @property
        def protocol_identity(self):
            return {
                **super().protocol_identity,
                "prompt_sha256": SUCCESS_PROMPT_SHA256,
            }

        async def complete_json_async(self, **request):
            return self.complete_json(**request)

    build = rebuild_section_source_parallel(
        original_records=records,
        replay_outcomes=outcomes,
        replay_trajectory_loader=lambda _path: _successful_trajectory(),
        extractor_factory=lambda _origin, _raw_path: (
            SuccessfulTrajectoryExperienceExtractor(BatchLLM(_experience_response()))
        ),
        checkpoint_dir=tmp_path / "audit-checkpoints",
        selected_train_indices=range(8),
    )
    assert [row["train_index"] for row in build.source_audit] == [0]
    assert [row["train_index"] for row in build.source_exclusions] == list(range(1, 8))
    assert {
        row["status"] for row in build.source_exclusions
    } == {"SOURCE_EXCLUDED_NO_VALIDATED_SUCCESS"}
    assert {
        row["replay_terminal_status"] for row in build.source_exclusions
    } == {"REPLAY_EXHAUSTED"}


def test_parallel_source_rebuild_uses_sixteen_workers_and_resumes_checkpoints(
    tmp_path: Path,
) -> None:
    all_started = asyncio.Event()
    lock = asyncio.Lock()
    active = 0
    maximum_active = 0
    started = 0
    original_records = [
        _successful_trajectory(f"synthetic-success-{index:03d}")
        for index in range(20)
    ]
    replay_outcomes = []
    for index in range(20, 200):
        task_id = f"synthetic-failed-{index:03d}"
        original_records.append(
            {
                "task_id": task_id,
                "trajectory_id": f"{task_id}::parent",
                "success": False,
            }
        )
        replay_outcomes.append(
            _with_source_replay_protocol(
                {
                    "task_id": task_id,
                    "parent_trajectory_id": f"{task_id}::parent",
                    "status": "REPLAY_EXHAUSTED",
                    "attempts": [],
                }
            )
        )

    class ConcurrentLLM(ScriptedLLM):
        @property
        def protocol_identity(self):
            return {
                **super().protocol_identity,
                "prompt_sha256": SUCCESS_PROMPT_SHA256,
            }

        async def complete_json_async(self, **request):
            nonlocal active, maximum_active, started
            async with lock:
                active += 1
                started += 1
                maximum_active = max(maximum_active, active)
                if started == SOURCE_REBUILD_WORKERS:
                    all_started.set()
            await asyncio.wait_for(all_started.wait(), timeout=2)
            await asyncio.sleep(0.005)
            try:
                return super().complete_json(**request)
            finally:
                async with lock:
                    active -= 1

    def extractor_factory(origin: str, _raw_response_path: Path):
        assert origin == "ORIGINAL_SUCCESS"
        return SuccessfulTrajectoryExperienceExtractor(
            ConcurrentLLM(_experience_response())
        )

    checkpoints = tmp_path / "checkpoints"
    build = rebuild_section_source_parallel(
        original_records=original_records,
        replay_outcomes=replay_outcomes,
        replay_trajectory_loader=lambda _path: _successful_trajectory(),
        extractor_factory=extractor_factory,
        checkpoint_dir=checkpoints,
    )

    assert SOURCE_REBUILD_WORKERS == 16
    assert maximum_active == 16
    assert [row["train_index"] for row in build.section_graphs["workflows"]] == list(
        range(20)
    )
    assert {row["status"] for row in build.source_audit} == {"INGESTED"}
    assert len(list(checkpoints.rglob("extraction.json"))) == 20

    started_before_resume = started
    resumed = rebuild_section_source_parallel(
        original_records=original_records,
        replay_outcomes=replay_outcomes,
        replay_trajectory_loader=lambda _path: _successful_trajectory(),
        extractor_factory=extractor_factory,
        checkpoint_dir=checkpoints,
    )
    assert resumed == build
    assert started == started_before_resume


def test_parallel_source_rebuild_discards_bad_edges_without_retry(
    tmp_path: Path,
) -> None:
    original_records = [_successful_trajectory("synthetic-success")]
    replay_outcomes = []
    for index in range(1, 200):
        task_id = f"synthetic-failed-{index:03d}"
        original_records.append(
            {
                "task_id": task_id,
                "trajectory_id": f"{task_id}::parent",
                "success": False,
            }
        )
        replay_outcomes.append(
            _with_source_replay_protocol(
                {
                    "task_id": task_id,
                    "parent_trajectory_id": f"{task_id}::parent",
                    "status": "REPLAY_EXHAUSTED",
                    "attempts": [],
                }
            )
        )

    calls: list[dict] = []

    class AttemptLLM(ScriptedLLM):
        def __init__(self, response, raw_path: Path):
            super().__init__(response)
            self.raw_path = raw_path

        @property
        def protocol_identity(self):
            return {
                **super().protocol_identity,
                "prompt_sha256": SUCCESS_PROMPT_SHA256,
            }

        async def complete_json_async(self, **request):
            calls.append(json.loads(json.dumps(request)))
            self.raw_path.parent.mkdir(parents=True, exist_ok=True)
            response = json.loads(json.dumps(self.response))
            self.raw_path.write_text(
                json.dumps(
                    {
                        "format": SOURCE_RAW_RESPONSE_FORMAT,
                        "outcome": "COMPLETE",
                        "response": json.dumps(response),
                    }
                )
            )
            return response

    def extractor_factory(origin: str, raw_path: Path):
        assert origin == "ORIGINAL_SUCCESS"
        response = _experience_response()
        if raw_path.name == "raw_response_attempt_01.json":
            response["edges"] = [
                {"source": 0, "target": 1},
                {"source": 2, "target": 2},
            ]
        return SuccessfulTrajectoryExperienceExtractor(
            AttemptLLM(response, raw_path)
        )

    build = rebuild_section_source_parallel(
        original_records=original_records,
        replay_outcomes=replay_outcomes,
        replay_trajectory_loader=lambda _path: _successful_trajectory(),
        extractor_factory=extractor_factory,
        checkpoint_dir=tmp_path / "checkpoints",
    )

    assert SOURCE_EXTRACTION_SEMANTIC_ATTEMPTS == 3
    assert build.source_audit[0]["source_extraction_attempt_index"] == 1
    assert build.source_audit[0]["invalid_response_attempts"] == []
    assert build.source_audit[0]["draft_discarded_edge_reasons"] == [
        "edge[1]: experience edge must reference a forward pair of distinct nodes"
    ]
    assert build.source_audit[0]["discarded_edge_reasons"] == []
    assert build.section_graphs["workflows"][0]["edges"] == [
        {"source": 0, "target": 1}
    ]
    assert len(calls) == 1
    assert "validation_feedback" not in calls[0]["payload"]


def test_parallel_source_rebuild_excludes_only_context_length_overflow(
    tmp_path: Path,
) -> None:
    original_records = [
        _successful_trajectory("synthetic-overflow"),
        _successful_trajectory("synthetic-kept"),
    ]
    replay_outcomes = []
    for index in range(2, 200):
        task_id = f"synthetic-failed-{index:03d}"
        original_records.append(
            {
                "task_id": task_id,
                "trajectory_id": f"{task_id}::parent",
                "success": False,
            }
        )
        replay_outcomes.append(
            _with_source_replay_protocol(
                {
                    "task_id": task_id,
                    "parent_trajectory_id": f"{task_id}::parent",
                    "status": "REPLAY_EXHAUSTED",
                    "attempts": [],
                }
            )
        )

    calls: list[str] = []

    class ContextAwareLLM(ScriptedLLM):
        @property
        def protocol_identity(self):
            return {
                **super().protocol_identity,
                "prompt_sha256": SUCCESS_PROMPT_SHA256,
            }

        async def complete_json_async(self, **request):
            calls.append(request["request_id"])
            if request["request_id"].startswith("synthetic-overflow::"):
                raise RequestContextLengthExceeded(
                    "maximum context length is 100000 tokens; requested total is 100001"
                )
            return self.complete_json(**request)

    progress: list[dict] = []
    build = rebuild_section_source_parallel(
        original_records=original_records,
        replay_outcomes=replay_outcomes,
        replay_trajectory_loader=lambda _path: _successful_trajectory(),
        extractor_factory=lambda _origin, _raw_path: (
            SuccessfulTrajectoryExperienceExtractor(
                ContextAwareLLM(_experience_response())
            )
        ),
        checkpoint_dir=tmp_path / "checkpoints",
        progress_callback=lambda row: progress.append(dict(row)),
    )

    assert [
        row["task_id"] for row in build.section_graphs["workflows"]
    ] == ["synthetic-kept"]
    assert [row["task_id"] for row in build.source_audit] == ["synthetic-kept"]
    assert _without_active_exclusion_identity(
        build.source_exclusions[0]
    ) == {
        "train_index": 0,
        "task_id": "synthetic-overflow",
        "trajectory_id": "synthetic-overflow::source_replay::accepted",
        "origin": "ORIGINAL_SUCCESS",
        "status": "SOURCE_EXCLUDED_CONTEXT_LENGTH",
        "error": (
            "maximum context length is 100000 tokens; requested total is 100001"
        ),
    }
    assert any(
        row["status"] == "SOURCE_EXCLUDED_CONTEXT_LENGTH"
        and row["task_id"] == "synthetic-overflow"
        for row in progress
    )
    calls_before_resume = len(calls)
    resumed = rebuild_section_source_parallel(
        original_records=original_records,
        replay_outcomes=replay_outcomes,
        replay_trajectory_loader=lambda _path: _successful_trajectory(),
        extractor_factory=lambda _origin, _raw_path: (
            SuccessfulTrajectoryExperienceExtractor(
                ContextAwareLLM(_experience_response())
            )
        ),
        checkpoint_dir=tmp_path / "checkpoints",
    )
    assert resumed == build
    assert len(calls) == calls_before_resume


def test_parallel_source_rebuild_excludes_exhausted_generation_failure(
    tmp_path: Path,
) -> None:
    failed_task_id = "synthetic-generation-failure"
    kept_task_id = "synthetic-kept"
    original_records = [
        _successful_trajectory(failed_task_id),
        _successful_trajectory(kept_task_id),
    ]
    replay_outcomes = []
    for index in range(2, 200):
        task_id = f"synthetic-failed-{index:03d}"
        original_records.append(
            {
                "task_id": task_id,
                "trajectory_id": f"{task_id}::parent",
                "success": False,
            }
        )
        replay_outcomes.append(
            _with_source_replay_protocol(
                {
                    "task_id": task_id,
                    "parent_trajectory_id": f"{task_id}::parent",
                    "status": "REPLAY_EXHAUSTED",
                    "attempts": [],
                }
            )
        )

    calls: list[str] = []

    class InvalidThenKeptLLM(ScriptedLLM):
        def __init__(self, raw_path: Path):
            super().__init__(_experience_response())
            self.raw_path = raw_path

        @property
        def protocol_identity(self):
            return {
                **super().protocol_identity,
                "prompt_sha256": SUCCESS_PROMPT_SHA256,
                "request_kind": SUCCESS_EXTRACTION_KIND,
            }

        async def complete_json_async(self, **request):
            request_id = request["request_id"]
            calls.append(request_id)
            if request_id.startswith(f"{failed_task_id}::"):
                raise RequestCompletionLengthExceeded(
                    "completion reached max_tokens before a complete response",
                    partial_content='{"experience_nodes":[',
                )
            response = _experience_response()
            protocol = dict(self.protocol_identity)
            self.raw_path.parent.mkdir(parents=True, exist_ok=True)
            self.raw_path.write_text(
                json.dumps(
                    {
                        "format": SOURCE_RAW_RESPONSE_FORMAT,
                        "outcome": "COMPLETE",
                        "request_kind": request["kind"],
                        "request_id": request_id,
                        "system_prompt_sha256": hashlib.sha256(
                            request["system_prompt"].encode("utf-8")
                        ).hexdigest(),
                        "payload_sha256": hashlib.sha256(
                            canonical_json_bytes(request["payload"])
                        ).hexdigest(),
                        "source_protocol": protocol,
                        "source_protocol_sha256": hashlib.sha256(
                            canonical_json_bytes(protocol)
                        ).hexdigest(),
                        "response": json.dumps(response),
                    }
                )
            )
            return response

    def extractor_factory(origin: str, raw_path: Path):
        assert origin == "ORIGINAL_SUCCESS"
        return SuccessfulTrajectoryExperienceExtractor(
            InvalidThenKeptLLM(raw_path)
        )

    checkpoints = tmp_path / "checkpoints"
    progress: list[dict] = []
    build = rebuild_section_source_parallel(
        original_records=original_records,
        replay_outcomes=replay_outcomes,
        replay_trajectory_loader=lambda _path: _successful_trajectory(),
        extractor_factory=extractor_factory,
        checkpoint_dir=checkpoints,
        progress_callback=lambda row: progress.append(dict(row)),
    )

    assert [row["task_id"] for row in build.section_graphs["workflows"]] == [
        kept_task_id
    ]
    assert _without_active_exclusion_identity(
        build.source_exclusions[0]
    ) == {
        "train_index": 0,
        "task_id": failed_task_id,
        "trajectory_id": f"{failed_task_id}::source_replay::accepted",
        "origin": "ORIGINAL_SUCCESS",
        "status": "SOURCE_EXCLUDED_GENERATION_FAILURE",
        "semantic_attempt_count": SOURCE_EXTRACTION_SEMANTIC_ATTEMPTS,
        "invalid_response_attempts": [
            "RequestCompletionLengthExceeded: completion reached max_tokens before a complete response; "
            "partial_chars=21; "
            "partial_sha256=1a4d5dc870d2337ad8a2c041b5028ec5e92f6160656c18c4187e60f11582b173"
        ]
        * SOURCE_EXTRACTION_SEMANTIC_ATTEMPTS,
    }
    assert sum(request_id.startswith(f"{failed_task_id}::") for request_id in calls) == 3
    assert any(
        row["status"] == "SOURCE_EXCLUDED_GENERATION_FAILURE"
        and row["task_id"] == failed_task_id
        for row in progress
    )

    calls_before_resume = len(calls)
    resumed = rebuild_section_source_parallel(
        original_records=original_records,
        replay_outcomes=replay_outcomes,
        replay_trajectory_loader=lambda _path: _successful_trajectory(),
        extractor_factory=extractor_factory,
        checkpoint_dir=checkpoints,
    )
    assert resumed == build
    assert len(calls) == calls_before_resume


def test_parallel_source_rebuild_contains_transport_failure_to_one_item(
    tmp_path: Path,
) -> None:
    records = [
        _successful_trajectory(f"transport-{index:03d}")
        for index in range(200)
    ]
    calls: list[str] = []

    class TransportAwareLLM(ScriptedLLM):
        @property
        def protocol_identity(self):
            return {
                "format": SUCCESS_SOURCE_PROTOCOL_FORMAT,
                "request_kind": SUCCESS_EXTRACTION_KIND,
                "model": REPAIR_SOURCE_MODEL,
                "temperature": REPAIR_SOURCE_TEMPERATURE,
                "thinking": REPAIR_SOURCE_THINKING,
                "max_tokens": REPAIR_SOURCE_MAX_TOKENS,
                "timeout_seconds": REPAIR_SOURCE_TIMEOUT_SECONDS,
                "generation_config": _source_generation_config(),
                "retry_waits_seconds": list(PRODUCER_TRANSPORT_RETRY_WAITS),
                "runtime_timeout_retries": PRODUCER_RUNTIME_TIMEOUT_RETRIES,
                "prompt_sha256": SUCCESS_PROMPT_SHA256,
                "service_url": "http://127.0.0.1:9999/v1",
            }

        async def complete_json_async(self, **request):
            calls.append(request["request_id"])
            if request["request_id"].startswith("transport-000::"):
                raise RequestRuntimeTimeout("synthetic provider timeout")
            return await super().complete_json_async(**request)

    build = rebuild_section_source_parallel(
        original_records=records,
        replay_outcomes=[],
        replay_trajectory_loader=lambda _path: _successful_trajectory(),
        extractor_factory=lambda _origin, _raw_path: (
            SuccessfulTrajectoryExperienceExtractor(
                TransportAwareLLM(_experience_response())
            )
        ),
        checkpoint_dir=tmp_path / "checkpoints",
        selected_train_indices=range(8),
    )
    assert [
        row["train_index"] for row in build.section_graphs["workflows"]
    ] == list(range(1, 8))
    assert build.source_exclusions[0]["status"] == (
        "SOURCE_EXCLUDED_GENERATION_FAILURE"
    )
    assert build.source_exclusions[0]["invalid_response_attempts"] == [
        "RequestRuntimeTimeout: synthetic provider timeout"
    ] * SOURCE_EXTRACTION_SEMANTIC_ATTEMPTS
    assert sum(call.startswith("transport-000::") for call in calls) == 3
    calls_before_resume = len(calls)
    resumed = rebuild_section_source_parallel(
        original_records=records,
        replay_outcomes=[],
        replay_trajectory_loader=lambda _path: _successful_trajectory(),
        extractor_factory=lambda _origin, _raw_path: (
            SuccessfulTrajectoryExperienceExtractor(
                TransportAwareLLM(_experience_response())
            )
        ),
        checkpoint_dir=tmp_path / "checkpoints",
        selected_train_indices=range(8),
    )
    assert resumed == build
    assert len(calls) == calls_before_resume


def test_parallel_source_rebuild_escalates_a_wave_wide_transport_outage(
    tmp_path: Path,
) -> None:
    records = [
        _successful_trajectory(f"systemic-transport-{index:03d}")
        for index in range(200)
    ]
    calls: list[str] = []

    class UnavailableLLM(ScriptedLLM):
        @property
        def protocol_identity(self):
            return {
                "format": SUCCESS_SOURCE_PROTOCOL_FORMAT,
                "request_kind": SUCCESS_EXTRACTION_KIND,
                "model": REPAIR_SOURCE_MODEL,
                "temperature": REPAIR_SOURCE_TEMPERATURE,
                "thinking": REPAIR_SOURCE_THINKING,
                "max_tokens": REPAIR_SOURCE_MAX_TOKENS,
                "timeout_seconds": REPAIR_SOURCE_TIMEOUT_SECONDS,
                "generation_config": _source_generation_config(),
                "retry_waits_seconds": list(PRODUCER_TRANSPORT_RETRY_WAITS),
                "runtime_timeout_retries": PRODUCER_RUNTIME_TIMEOUT_RETRIES,
                "prompt_sha256": SUCCESS_PROMPT_SHA256,
                "service_url": "http://127.0.0.1:9999/v1",
            }

        async def complete_json_async(self, **request):
            calls.append(request["request_id"])
            raise RequestRuntimeTimeout("synthetic endpoint outage")

    checkpoints = tmp_path / "checkpoints"
    with pytest.raises(
        SystemicProducerTransportFailure,
        match="8 distinct requests in one stage wave",
    ):
        rebuild_section_source_parallel(
            original_records=records,
            replay_outcomes=[],
            replay_trajectory_loader=lambda _path: _successful_trajectory(),
            extractor_factory=lambda _origin, _raw_path: (
                SuccessfulTrajectoryExperienceExtractor(
                    UnavailableLLM(_experience_response())
                )
            ),
            checkpoint_dir=checkpoints,
            selected_train_indices=range(8),
        )
    assert len(calls) == 8 * SOURCE_EXTRACTION_SEMANTIC_ATTEMPTS
    assert not list(checkpoints.glob("*/terminal_exclusion.json"))
    assert len(
        list((checkpoints / "_systemic_transport_waves").glob("wave_*"))
    ) == 1

    with pytest.raises(SystemicProducerTransportFailure):
        rebuild_section_source_parallel(
            original_records=records,
            replay_outcomes=[],
            replay_trajectory_loader=lambda _path: _successful_trajectory(),
            extractor_factory=lambda _origin, _raw_path: (
                SuccessfulTrajectoryExperienceExtractor(
                    UnavailableLLM(_experience_response())
                )
            ),
            checkpoint_dir=checkpoints,
            selected_train_indices=range(8),
        )
    assert len(calls) == 2 * 8 * SOURCE_EXTRACTION_SEMANTIC_ATTEMPTS
    assert len(
        list((checkpoints / "_systemic_transport_waves").glob("wave_*"))
    ) == 2

    restored_calls: list[str] = []

    class RestoredLLM(ScriptedLLM):
        async def complete_json_async(self, **request):
            restored_calls.append(request["request_id"])
            return await super().complete_json_async(**request)

    restored = rebuild_section_source_parallel(
        original_records=records,
        replay_outcomes=[],
        replay_trajectory_loader=lambda _path: _successful_trajectory(),
        extractor_factory=lambda _origin, _raw_path: (
            SuccessfulTrajectoryExperienceExtractor(
                RestoredLLM(_experience_response())
            )
        ),
        checkpoint_dir=checkpoints,
        selected_train_indices=range(8),
    )
    assert len(restored.section_graphs["workflows"]) == 8
    assert len(restored_calls) == 8
    assert not (tmp_path / "section-graphs.json").exists()


def test_source_auth_failure_cancels_and_drains_sibling_jobs(
    tmp_path: Path,
) -> None:
    records = [
        _successful_trajectory(f"auth-drain-{index:03d}")
        for index in range(200)
    ]
    ready = asyncio.Event()
    started = 0
    cancelled: list[str] = []
    completed: list[str] = []

    class AuthDrainLLM(ScriptedLLM):
        @property
        def protocol_identity(self):
            return {
                "format": SUCCESS_SOURCE_PROTOCOL_FORMAT,
                "request_kind": SUCCESS_EXTRACTION_KIND,
                "model": REPAIR_SOURCE_MODEL,
                "temperature": REPAIR_SOURCE_TEMPERATURE,
                "thinking": REPAIR_SOURCE_THINKING,
                "max_tokens": REPAIR_SOURCE_MAX_TOKENS,
                "timeout_seconds": REPAIR_SOURCE_TIMEOUT_SECONDS,
                "generation_config": _source_generation_config(),
                "retry_waits_seconds": list(PRODUCER_TRANSPORT_RETRY_WAITS),
                "runtime_timeout_retries": PRODUCER_RUNTIME_TIMEOUT_RETRIES,
                "prompt_sha256": SUCCESS_PROMPT_SHA256,
                "service_url": "http://127.0.0.1:9999/v1",
            }

        async def complete_json_async(self, **request):
            nonlocal started
            request_id = request["request_id"]
            if request_id.startswith("auth-drain-000::"):
                await ready.wait()
                error = APIError(
                    "synthetic authentication failure",
                    httpx.Request("POST", "http://127.0.0.1:9999/v1"),
                    body=None,
                )
                error.status_code = 401
                raise error
            started += 1
            if started == 7:
                ready.set()
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                cancelled.append(request_id)
                raise
            completed.append(request_id)
            return await super().complete_json_async(**request)

    with pytest.raises(
        SystemicProducerTransportFailure,
        match="HTTP 401",
    ):
        rebuild_section_source_parallel(
            original_records=records,
            replay_outcomes=[],
            replay_trajectory_loader=lambda _path: _successful_trajectory(),
            extractor_factory=lambda _origin, _raw_path: (
                SuccessfulTrajectoryExperienceExtractor(
                    AuthDrainLLM(_experience_response())
                )
            ),
            checkpoint_dir=tmp_path / "checkpoints",
            selected_train_indices=range(8),
        )
    assert len(cancelled) == 7
    assert completed == []


def test_source_auth_failure_archives_every_failed_sibling_before_resume(
    tmp_path: Path,
) -> None:
    records = [
        _successful_trajectory(f"auth-wave-{index:03d}")
        for index in range(200)
    ]
    seven_distinct_timeouts = asyncio.Event()
    timed_out_requests: set[str] = set()

    class MixedAuthWaveLLM(ScriptedLLM):
        @property
        def protocol_identity(self):
            return {
                "format": SUCCESS_SOURCE_PROTOCOL_FORMAT,
                "request_kind": SUCCESS_EXTRACTION_KIND,
                "model": REPAIR_SOURCE_MODEL,
                "temperature": REPAIR_SOURCE_TEMPERATURE,
                "thinking": REPAIR_SOURCE_THINKING,
                "max_tokens": REPAIR_SOURCE_MAX_TOKENS,
                "timeout_seconds": REPAIR_SOURCE_TIMEOUT_SECONDS,
                "generation_config": _source_generation_config(),
                "retry_waits_seconds": list(PRODUCER_TRANSPORT_RETRY_WAITS),
                "runtime_timeout_retries": PRODUCER_RUNTIME_TIMEOUT_RETRIES,
                "prompt_sha256": SUCCESS_PROMPT_SHA256,
                "service_url": "http://127.0.0.1:9999/v1",
            }

        async def complete_json_async(self, **request):
            request_id = request["request_id"]
            if request_id.startswith("auth-wave-000::"):
                await seven_distinct_timeouts.wait()
                error = APIError(
                    "synthetic authentication failure",
                    httpx.Request("POST", "http://127.0.0.1:9999/v1"),
                    body=None,
                )
                error.status_code = 401
                raise error
            timed_out_requests.add(request_id)
            if len(timed_out_requests) == 7:
                seven_distinct_timeouts.set()
            raise RequestRuntimeTimeout("synthetic sibling timeout")

    checkpoints = tmp_path / "checkpoints"
    with pytest.raises(SystemicProducerTransportFailure, match="HTTP 401"):
        rebuild_section_source_parallel(
            original_records=records,
            replay_outcomes=[],
            replay_trajectory_loader=lambda _path: _successful_trajectory(),
            extractor_factory=lambda _origin, _raw_path: (
                SuccessfulTrajectoryExperienceExtractor(
                    MixedAuthWaveLLM(_experience_response())
                )
            ),
            checkpoint_dir=checkpoints,
            selected_train_indices=range(8),
        )

    assert not list(checkpoints.glob("*/raw_response_attempt_*.json"))
    manifests = list(
        (checkpoints / "_systemic_transport_waves").glob("wave_*/manifest.json")
    )
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert len(manifest["failed_request_ids"]) == 8

    restored = rebuild_section_source_parallel(
        original_records=records,
        replay_outcomes=[],
        replay_trajectory_loader=lambda _path: _successful_trajectory(),
        extractor_factory=lambda _origin, _raw_path: (
            SuccessfulTrajectoryExperienceExtractor(
                ScriptedLLM(_experience_response())
            )
        ),
        checkpoint_dir=checkpoints,
        selected_train_indices=range(8),
    )
    assert len(restored.section_graphs["workflows"]) == 8


def test_source_transport_archive_recovers_empty_prepublication_staging(
    tmp_path: Path,
) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    staging = (
        checkpoint_root
        / "_systemic_transport_waves"
        / ".wave_0001.building"
    )
    staging.mkdir(parents=True)

    source_rebuild_module._recover_source_systemic_transport_archives(
        checkpoint_root
    )

    assert not staging.exists()


def test_source_transport_archive_recovers_manifest_temporary_before_publish(
    tmp_path: Path,
) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    staging = (
        checkpoint_root
        / "_systemic_transport_waves"
        / ".wave_0001.building"
    )
    staging.mkdir(parents=True)
    temporary = staging / ".manifest.json.interrupted"
    temporary.write_text("partial", encoding="utf-8")

    source_rebuild_module._recover_source_systemic_transport_archives(
        checkpoint_root
    )

    assert not staging.exists()


def test_interrupted_transport_preserves_later_valid_complete_response(
    tmp_path: Path,
) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir(mode=0o700)
    checkpoint = checkpoint_root / "000_0123456789ab"
    checkpoint.mkdir(mode=0o700)
    protocol = {
        "format": SUCCESS_SOURCE_PROTOCOL_FORMAT,
        "request_kind": SUCCESS_EXTRACTION_KIND,
        "model": REPAIR_SOURCE_MODEL,
        "temperature": REPAIR_SOURCE_TEMPERATURE,
        "thinking": REPAIR_SOURCE_THINKING,
        "max_tokens": REPAIR_SOURCE_MAX_TOKENS,
        "timeout_seconds": REPAIR_SOURCE_TIMEOUT_SECONDS,
        "generation_config": _source_generation_config(),
        "retry_waits_seconds": list(PRODUCER_TRANSPORT_RETRY_WAITS),
        "runtime_timeout_retries": PRODUCER_RUNTIME_TIMEOUT_RETRIES,
        "prompt_sha256": SUCCESS_PROMPT_SHA256,
        "service_url": "http://127.0.0.1:9999/v1",
    }
    request_id = "saved-complete-after-transport"
    payload_sha256 = "1" * 64
    common = {
        "format": SOURCE_RAW_RESPONSE_FORMAT,
        "request_kind": SUCCESS_EXTRACTION_KIND,
        "request_id": request_id,
        "system_prompt_sha256": SUCCESS_PROMPT_SHA256,
        "payload_sha256": payload_sha256,
        "source_protocol": protocol,
        "source_protocol_sha256": hashlib.sha256(
            canonical_json_bytes(protocol)
        ).hexdigest(),
    }
    transport = checkpoint / "raw_response_attempt_01.json"
    complete = checkpoint / "raw_response_attempt_02.json"
    transport.write_bytes(
        canonical_json_bytes(
            {
                **common,
                "outcome": "TRANSPORT_EXHAUSTED",
                "error": "RequestRuntimeTimeout: interrupted",
                "response": "",
            }
        )
    )
    complete.write_bytes(
        canonical_json_bytes(
            {
                **common,
                "outcome": "COMPLETE",
                "response": json.dumps(_experience_response()),
            }
        )
    )

    source_rebuild_module._archive_interrupted_source_transport_waves(
        checkpoint_root
    )

    assert transport.is_file()
    assert complete.is_file()
    assert not (checkpoint_root / "_systemic_transport_waves").exists()
    invalid, saved = source_rebuild_module._load_saved_invalid_response_attempts(
        checkpoint,
        expected_protocol={
            **protocol,
            "service_url": "http://127.0.0.1:29999/v1",
        },
        expected_request_id=request_id,
        expected_payload_sha256=payload_sha256,
    )
    assert len(invalid) == 1
    assert saved == _experience_response()


def test_interrupted_review_preserves_complete_response_before_publication(
    tmp_path: Path,
) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir(mode=0o700)
    checkpoint = checkpoint_root / "000_0123456789ab"
    checkpoint.mkdir(mode=0o700)
    protocol = dict(_PassThroughReviewLLM().protocol_identity)
    request_id = "source-review-saved-complete-after-transport"
    payload_sha256 = "2" * 64
    common = {
        "format": SOURCE_RAW_RESPONSE_FORMAT,
        "request_kind": SOURCE_REVIEW_KIND,
        "request_id": request_id,
        "system_prompt_sha256": SOURCE_REVIEW_PROMPT_SHA256,
        "payload_sha256": payload_sha256,
        "source_protocol": protocol,
        "source_protocol_sha256": hashlib.sha256(
            canonical_json_bytes(protocol)
        ).hexdigest(),
    }
    transport = checkpoint / "raw_review_response_attempt_01.json"
    complete = checkpoint / "raw_review_response_attempt_02.json"
    response = {
        **_experience_response(),
        "review_decisions": [
            {
                "draft_node": index,
                "decision": "KEEP",
                "final_nodes": [index],
                "basis": "The draft node is supported by the evidence.",
            }
            for index in range(3)
        ],
    }
    transport.write_bytes(
        canonical_json_bytes(
            {
                **common,
                "outcome": "TRANSPORT_EXHAUSTED",
                "error": "RequestRuntimeTimeout: interrupted",
                "response": "",
            }
        )
    )
    complete.write_bytes(
        canonical_json_bytes(
            {
                **common,
                "outcome": "COMPLETE",
                "response": json.dumps(response),
            }
        )
    )

    source_rebuild_module._archive_interrupted_source_transport_waves(
        checkpoint_root
    )

    assert transport.is_file()
    assert complete.is_file()
    assert not (checkpoint_root / "_systemic_transport_waves").exists()
    invalid, saved = source_rebuild_module._load_saved_review_response_attempts(
        checkpoint,
        expected_protocol=protocol,
        expected_request_id=request_id,
        expected_payload_sha256=payload_sha256,
        draft_count=3,
    )
    assert len(invalid) == 1
    assert saved is not None
    assert len(saved.experience_nodes) == 3


def test_source_resume_archives_transport_raw_left_by_process_interruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = [
        _successful_trajectory(f"crash-wave-{index:03d}")
        for index in range(200)
    ]

    class CurrentSourceLLM(ScriptedLLM):
        @property
        def protocol_identity(self):
            return {
                "format": SUCCESS_SOURCE_PROTOCOL_FORMAT,
                "request_kind": SUCCESS_EXTRACTION_KIND,
                "model": REPAIR_SOURCE_MODEL,
                "temperature": REPAIR_SOURCE_TEMPERATURE,
                "thinking": REPAIR_SOURCE_THINKING,
                "max_tokens": REPAIR_SOURCE_MAX_TOKENS,
                "timeout_seconds": REPAIR_SOURCE_TIMEOUT_SECONDS,
                "generation_config": _source_generation_config(),
                "retry_waits_seconds": list(PRODUCER_TRANSPORT_RETRY_WAITS),
                "runtime_timeout_retries": PRODUCER_RUNTIME_TIMEOUT_RETRIES,
                "prompt_sha256": SUCCESS_PROMPT_SHA256,
                "service_url": "http://127.0.0.1:9999/v1",
            }

    class UnavailableSourceLLM(CurrentSourceLLM):
        async def complete_json_async(self, **_request):
            raise RequestRuntimeTimeout("synthetic provider timeout")

    checkpoints = tmp_path / "checkpoints"
    original_record_failure = (
        source_rebuild_module.ProducerTransportGuard.record_failure
    )

    async def crash_after_raw_transport(self, *, request_id, error):
        del self, request_id, error
        raise RuntimeError("synthetic process interruption")

    with monkeypatch.context() as context:
        context.setattr(
            source_rebuild_module.ProducerTransportGuard,
            "record_failure",
            crash_after_raw_transport,
        )
        with pytest.raises(RuntimeError, match="process interruption"):
            rebuild_section_source_parallel(
                original_records=records,
                replay_outcomes=[],
                replay_trajectory_loader=lambda _path: _successful_trajectory(),
                extractor_factory=lambda _origin, _raw_path: (
                    SuccessfulTrajectoryExperienceExtractor(
                        UnavailableSourceLLM(_experience_response())
                    )
                ),
                checkpoint_dir=checkpoints,
                selected_train_indices=range(8),
            )
    assert (
        source_rebuild_module.ProducerTransportGuard.record_failure
        is original_record_failure
    )
    assert list(checkpoints.glob("*/raw_response_attempt_*.json"))

    restored = rebuild_section_source_parallel(
        original_records=records,
        replay_outcomes=[],
        replay_trajectory_loader=lambda _path: _successful_trajectory(),
        extractor_factory=lambda _origin, _raw_path: (
            SuccessfulTrajectoryExperienceExtractor(
                CurrentSourceLLM(_experience_response())
            )
        ),
        checkpoint_dir=checkpoints,
        selected_train_indices=range(8),
    )
    assert len(restored.section_graphs["workflows"]) == 8
    assert not list(checkpoints.glob("*/raw_response_attempt_*.json"))
    manifests = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (
            checkpoints / "_systemic_transport_waves"
        ).glob("wave_*/manifest.json")
    ]
    assert {manifest["cause"] for manifest in manifests} == {
        "INTERRUPTED_WAVE"
    }


def test_parallel_source_review_contains_transport_failure_to_one_item(
    tmp_path: Path,
) -> None:
    records = [
        _successful_trajectory(f"review-transport-{index:03d}")
        for index in range(200)
    ]
    review_calls: list[str] = []

    class TransportReviewLLM(_PassThroughReviewLLM):
        async def complete_json_async(self, **request):
            review_calls.append(request["request_id"])
            if request["request_id"].startswith(
                "source-review-review-transport-000::"
            ):
                raise RequestRuntimeTimeout("synthetic review timeout")
            return await super().complete_json_async(**request)

    build = _rebuild_section_source_parallel(
        original_records=records,
        replay_outcomes=[],
        replay_trajectory_loader=lambda _path: _successful_trajectory(),
        extractor_factory=lambda _origin, _raw_path: (
            SuccessfulTrajectoryExperienceExtractor(
                ScriptedLLM(_experience_response())
            )
        ),
        reviewer_factory=lambda _raw_path: ExperienceSourceReviewer(
            TransportReviewLLM()
        ),
        checkpoint_dir=tmp_path / "checkpoints",
        selected_train_indices=range(8),
    )
    assert len(build.section_graphs["workflows"]) == 8
    assert build.source_audit[0]["source_review_status"] == (
        "REVIEW_DRAFT_FALLBACK"
    )
    assert build.source_audit[0][
        "source_review_invalid_response_attempts"
    ] == ["RequestRuntimeTimeout: synthetic review timeout"] * 3
    calls_before_resume = len(review_calls)
    resumed = _rebuild_section_source_parallel(
        original_records=records,
        replay_outcomes=[],
        replay_trajectory_loader=lambda _path: _successful_trajectory(),
        extractor_factory=lambda _origin, _raw_path: (
            SuccessfulTrajectoryExperienceExtractor(
                ScriptedLLM(_experience_response())
            )
        ),
        reviewer_factory=lambda _raw_path: ExperienceSourceReviewer(
            TransportReviewLLM()
        ),
        checkpoint_dir=tmp_path / "checkpoints",
        selected_train_indices=range(8),
    )
    assert resumed == build
    assert len(review_calls) == calls_before_resume


def test_parallel_source_review_retries_after_a_systemic_outage(
    tmp_path: Path,
) -> None:
    records = [
        _successful_trajectory(f"systemic-review-{index:03d}")
        for index in range(200)
    ]
    extraction_calls: list[str] = []
    review_calls: list[str] = []

    class CountingExtractionLLM(ScriptedLLM):
        async def complete_json_async(self, **request):
            extraction_calls.append(request["request_id"])
            return await super().complete_json_async(**request)

    class UnavailableReviewLLM(_PassThroughReviewLLM):
        async def complete_json_async(self, **request):
            review_calls.append(request["request_id"])
            raise RequestRuntimeTimeout("synthetic review endpoint outage")

    checkpoints = tmp_path / "checkpoints"
    with pytest.raises(SystemicProducerTransportFailure):
        _rebuild_section_source_parallel(
            original_records=records,
            replay_outcomes=[],
            replay_trajectory_loader=lambda _path: _successful_trajectory(),
            extractor_factory=lambda _origin, _raw_path: (
                SuccessfulTrajectoryExperienceExtractor(
                    CountingExtractionLLM(_experience_response())
                )
            ),
            reviewer_factory=lambda _raw_path: ExperienceSourceReviewer(
                UnavailableReviewLLM()
            ),
            checkpoint_dir=checkpoints,
            selected_train_indices=range(8),
        )
    assert len(extraction_calls) == 8
    assert len(review_calls) == 8 * SOURCE_EXTRACTION_SEMANTIC_ATTEMPTS
    assert len(list(checkpoints.glob("*/extraction.json"))) == 8
    assert not list(checkpoints.glob("*/review.json"))

    class MustNotExtract(ScriptedLLM):
        async def complete_json_async(self, **_request):
            raise AssertionError("accepted extraction must be resumed")

    restored_review_calls: list[str] = []

    class RestoredReviewLLM(_PassThroughReviewLLM):
        async def complete_json_async(self, **request):
            restored_review_calls.append(request["request_id"])
            return await super().complete_json_async(**request)

    restored = _rebuild_section_source_parallel(
        original_records=records,
        replay_outcomes=[],
        replay_trajectory_loader=lambda _path: _successful_trajectory(),
        extractor_factory=lambda _origin, _raw_path: (
            SuccessfulTrajectoryExperienceExtractor(
                MustNotExtract(_experience_response())
            )
        ),
        reviewer_factory=lambda _raw_path: ExperienceSourceReviewer(
            RestoredReviewLLM()
        ),
        checkpoint_dir=checkpoints,
        selected_train_indices=range(8),
    )
    assert len(restored.section_graphs["workflows"]) == 8
    assert all(
        row["source_review_status"] == "REVIEW_ACCEPTED"
        for row in restored.source_audit
    )
    assert len(restored_review_calls) == 8
    assert len(extraction_calls) == 8


def test_source_wave_archives_systemic_extraction_and_review_stages_together(
    tmp_path: Path,
) -> None:
    records = [
        _successful_trajectory(f"dual-stage-{index:03d}")
        for index in range(200)
    ]
    checkpoints = tmp_path / "checkpoints"

    class CurrentSourceLLM(ScriptedLLM):
        @property
        def protocol_identity(self):
            return {
                "format": SUCCESS_SOURCE_PROTOCOL_FORMAT,
                "request_kind": SUCCESS_EXTRACTION_KIND,
                "model": REPAIR_SOURCE_MODEL,
                "temperature": REPAIR_SOURCE_TEMPERATURE,
                "thinking": REPAIR_SOURCE_THINKING,
                "max_tokens": REPAIR_SOURCE_MAX_TOKENS,
                "timeout_seconds": REPAIR_SOURCE_TIMEOUT_SECONDS,
                "generation_config": _source_generation_config(),
                "retry_waits_seconds": list(PRODUCER_TRANSPORT_RETRY_WAITS),
                "runtime_timeout_retries": PRODUCER_RUNTIME_TIMEOUT_RETRIES,
                "prompt_sha256": SUCCESS_PROMPT_SHA256,
                "service_url": "http://127.0.0.1:9999/v1",
            }

    healthy_kwargs = {
        "original_records": records,
        "replay_outcomes": [],
        "replay_trajectory_loader": lambda _path: _successful_trajectory(),
        "extractor_factory": lambda _origin, _raw_path: (
            SuccessfulTrajectoryExperienceExtractor(
                CurrentSourceLLM(_experience_response())
            )
        ),
        "reviewer_factory": lambda _raw_path: ExperienceSourceReviewer(
            _PassThroughReviewLLM()
        ),
        "checkpoint_dir": checkpoints,
        "selected_train_indices": range(8),
    }
    initial = _rebuild_section_source_parallel(**healthy_kwargs)
    assert len(initial.section_graphs["workflows"]) == 8

    task_dirs = sorted(
        path
        for path in checkpoints.iterdir()
        if path.is_dir() and path.name[:3].isdigit()
    )
    assert len(task_dirs) == 8
    for index, task_dir in enumerate(task_dirs):
        (task_dir / "review.json").unlink()
        if index >= 4:
            (task_dir / "extraction.json").unlink()

    class UnavailableExtractionLLM(CurrentSourceLLM):
        async def complete_json_async(self, **_request):
            raise RequestRuntimeTimeout("synthetic extraction outage")

    class UnavailableReviewLLM(_PassThroughReviewLLM):
        async def complete_json_async(self, **_request):
            raise RequestRuntimeTimeout("synthetic review outage")

    with pytest.raises(SystemicProducerTransportFailure):
        _rebuild_section_source_parallel(
            **{
                **healthy_kwargs,
                "extractor_factory": lambda _origin, _raw_path: (
                    SuccessfulTrajectoryExperienceExtractor(
                        UnavailableExtractionLLM(_experience_response())
                    )
                ),
                "reviewer_factory": lambda _raw_path: ExperienceSourceReviewer(
                    UnavailableReviewLLM()
                ),
            }
        )

    manifests = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(
            (checkpoints / "_systemic_transport_waves").glob(
                "wave_*/manifest.json"
            )
        )
    ]
    assert {manifest["stage"] for manifest in manifests} == {
        "source extraction",
        "source review",
    }
    assert not list(checkpoints.glob("*/raw_response_attempt_*.json"))
    assert not list(checkpoints.glob("*/raw_review_response_attempt_*.json"))
    assert not list(checkpoints.glob("*/terminal_exclusion.json"))
    assert not list(checkpoints.glob("*/review.json"))

    restored = _rebuild_section_source_parallel(**healthy_kwargs)
    assert len(restored.section_graphs["workflows"]) == 8
    assert all(
        row["source_review_status"] == "REVIEW_ACCEPTED"
        for row in restored.source_audit
    )


def test_systemic_review_retry_preserves_the_previous_draft_fallback(
    tmp_path: Path,
) -> None:
    records = [
        _successful_trajectory(f"retry-outage-{index:03d}")
        for index in range(200)
    ]
    checkpoints = tmp_path / "checkpoints"
    extraction_calls: list[str] = []

    class CountingSourceLLM(ScriptedLLM):
        async def complete_json_async(self, **request):
            extraction_calls.append(request["request_id"])
            return await super().complete_json_async(**request)

    class PartialReviewFailureLLM(_PassThroughReviewLLM):
        async def complete_json_async(self, **request):
            request_id = request["request_id"]
            if any(
                request_id.startswith(f"source-review-retry-outage-{index:03d}::")
                for index in range(4)
            ):
                raise RequestRuntimeTimeout("synthetic item-local review timeout")
            return await super().complete_json_async(**request)

    common = {
        "original_records": records,
        "replay_outcomes": [],
        "replay_trajectory_loader": lambda _path: _successful_trajectory(),
        "extractor_factory": lambda _origin, _raw_path: (
            SuccessfulTrajectoryExperienceExtractor(
                CountingSourceLLM(_experience_response())
            )
        ),
        "checkpoint_dir": checkpoints,
        "selected_train_indices": range(8),
    }
    initial = _rebuild_section_source_parallel(
        **common,
        reviewer_factory=lambda _raw_path: ExperienceSourceReviewer(
            PartialReviewFailureLLM()
        ),
    )
    assert sum(
        row["source_review_status"] == "REVIEW_DRAFT_FALLBACK"
        for row in initial.source_audit
    ) == 4
    extraction_count = len(extraction_calls)

    class UnavailableRetryLLM(_PassThroughReviewLLM):
        async def complete_json_async(self, **_request):
            raise RequestRuntimeTimeout("synthetic retry endpoint outage")

    with pytest.raises(SystemicProducerTransportFailure):
        _rebuild_section_source_parallel(
            **common,
            reviewer_factory=lambda _raw_path: ExperienceSourceReviewer(
                UnavailableRetryLLM()
            ),
            retry_failed_reviews=True,
        )
    fallback_checkpoints = []
    for path in checkpoints.glob("*/review.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["review_audit"]["source_review_status"] == (
            "REVIEW_DRAFT_FALLBACK"
        ):
            fallback_checkpoints.append(path)
    assert len(fallback_checkpoints) == 4
    assert not list(
        checkpoints.glob("*/raw_review_retry_response_attempt_*.json")
    )

    restored = _rebuild_section_source_parallel(
        **common,
        reviewer_factory=lambda _raw_path: ExperienceSourceReviewer(
            _PassThroughReviewLLM()
        ),
        retry_failed_reviews=True,
    )
    assert all(
        row["source_review_status"] == "REVIEW_ACCEPTED"
        for row in restored.source_audit
    )
    assert len(extraction_calls) == extraction_count


def test_parallel_source_rebuild_excludes_an_empty_reusable_graph(
    tmp_path: Path,
) -> None:
    original_records = [
        _successful_trajectory(f"synthetic-success-{index:03d}")
        for index in range(8)
    ]
    replay_outcomes = []
    for index in range(8, 200):
        task_id = f"synthetic-failed-{index:03d}"
        original_records.append(
            {
                "task_id": task_id,
                "trajectory_id": f"{task_id}::parent",
                "success": False,
            }
        )
        replay_outcomes.append(
            _with_source_replay_protocol(
                {
                    "task_id": task_id,
                    "parent_trajectory_id": f"{task_id}::parent",
                    "status": "REPLAY_EXHAUSTED",
                    "attempts": [],
                }
            )
        )

    calls: list[str] = []

    class EmptyFirstLLM(ScriptedLLM):
        @property
        def protocol_identity(self):
            return {
                **super().protocol_identity,
                "prompt_sha256": SUCCESS_PROMPT_SHA256,
                "request_kind": SUCCESS_EXTRACTION_KIND,
            }

        async def complete_json_async(self, **request):
            calls.append(request["request_id"])
            if request["request_id"].startswith("synthetic-success-000::"):
                return {"experience_nodes": [], "edges": []}
            return _experience_response()

    def extractor_factory(origin: str, _raw_path: Path):
        assert origin == "ORIGINAL_SUCCESS"
        return SuccessfulTrajectoryExperienceExtractor(
            EmptyFirstLLM(_experience_response())
        )

    checkpoints = tmp_path / "checkpoints"
    build = rebuild_section_source_parallel(
        original_records=original_records,
        replay_outcomes=replay_outcomes,
        replay_trajectory_loader=lambda _path: _successful_trajectory(),
        extractor_factory=extractor_factory,
        checkpoint_dir=checkpoints,
        selected_train_indices=range(8),
    )

    assert len(build.section_graphs["workflows"]) == 7
    assert _without_active_exclusion_identity(
        build.source_exclusions[0]
    ) == {
        "train_index": 0,
        "task_id": "synthetic-success-000",
        "trajectory_id": "synthetic-success-000::source_replay::accepted",
        "origin": "ORIGINAL_SUCCESS",
        "status": "SOURCE_EXCLUDED_NO_REUSABLE_EXPERIENCE",
    }
    calls_before_resume = len(calls)
    resumed = rebuild_section_source_parallel(
        original_records=original_records,
        replay_outcomes=replay_outcomes,
        replay_trajectory_loader=lambda _path: _successful_trajectory(),
        extractor_factory=extractor_factory,
        checkpoint_dir=checkpoints,
        selected_train_indices=range(8),
    )
    assert resumed == build
    assert len(calls) == calls_before_resume


def test_parallel_source_rebuild_resumes_after_one_saved_invalid_attempt(
    tmp_path: Path,
) -> None:
    original_records = [
        _successful_trajectory(f"synthetic-success-{index:03d}")
        for index in range(8)
    ]
    replay_outcomes = []
    for index in range(8, 200):
        task_id = f"synthetic-failed-{index:03d}"
        original_records.append(
            {
                "task_id": task_id,
                "trajectory_id": f"{task_id}::parent",
                "success": False,
            }
        )
        replay_outcomes.append(
            _with_source_replay_protocol(
                {
                    "task_id": task_id,
                    "parent_trajectory_id": f"{task_id}::parent",
                    "status": "REPLAY_EXHAUSTED",
                    "attempts": [],
                }
            )
        )

    interrupt_once = True
    attempted_paths: list[str] = []

    class InterruptingLLM(ScriptedLLM):
        def __init__(self, raw_path: Path):
            super().__init__(_experience_response())
            self.raw_path = raw_path

        @property
        def protocol_identity(self):
            return {
                **super().protocol_identity,
                "prompt_sha256": SUCCESS_PROMPT_SHA256,
                "request_kind": SUCCESS_EXTRACTION_KIND,
            }

        async def complete_json_async(self, **request):
            nonlocal interrupt_once
            attempted_paths.append(self.raw_path.name)
            if request["request_id"].startswith("synthetic-success-000::"):
                if self.raw_path.name == "raw_response_attempt_01.json":
                    protocol = dict(self.protocol_identity)
                    self.raw_path.parent.mkdir(parents=True, exist_ok=True)
                    self.raw_path.write_text(
                        json.dumps(
                            {
                                "format": SOURCE_RAW_RESPONSE_FORMAT,
                                "outcome": "COMPLETE",
                                "request_kind": request["kind"],
                                "request_id": request["request_id"],
                                "system_prompt_sha256": hashlib.sha256(
                                    request["system_prompt"].encode("utf-8")
                                ).hexdigest(),
                                "payload_sha256": hashlib.sha256(
                                    canonical_json_bytes(request["payload"])
                                ).hexdigest(),
                                "source_protocol": protocol,
                                "source_protocol_sha256": hashlib.sha256(
                                    canonical_json_bytes(protocol)
                                ).hexdigest(),
                                "response": "{}",
                            }
                        )
                    )
                    return {}
                if interrupt_once:
                    interrupt_once = False
                    raise RuntimeError("synthetic process interruption")
            return _experience_response()

    def extractor_factory(origin: str, raw_path: Path):
        assert origin == "ORIGINAL_SUCCESS"
        return SuccessfulTrajectoryExperienceExtractor(InterruptingLLM(raw_path))

    kwargs = {
        "original_records": original_records,
        "replay_outcomes": replay_outcomes,
        "replay_trajectory_loader": lambda _path: _successful_trajectory(),
        "extractor_factory": extractor_factory,
        "checkpoint_dir": tmp_path / "checkpoints",
        "selected_train_indices": range(8),
    }
    with pytest.raises(RuntimeError, match="synthetic process interruption"):
        rebuild_section_source_parallel(**kwargs)

    resumed = rebuild_section_source_parallel(**kwargs)

    assert len(resumed.section_graphs["workflows"]) == 8
    assert attempted_paths.count("raw_response_attempt_01.json") == 8
    assert "raw_response_attempt_02.json" in attempted_paths


@pytest.mark.parametrize("empty_response", [False, True])
def test_parallel_source_rebuild_recovers_a_valid_raw_response_after_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    empty_response: bool,
) -> None:
    original_records = [
        _successful_trajectory(f"synthetic-success-{index:03d}")
        for index in range(8)
    ]
    replay_outcomes = []
    for index in range(8, 200):
        task_id = f"synthetic-failed-{index:03d}"
        original_records.append(
            {
                "task_id": task_id,
                "trajectory_id": f"{task_id}::parent",
                "success": False,
            }
        )
        replay_outcomes.append(
            _with_source_replay_protocol(
                {
                    "task_id": task_id,
                    "parent_trajectory_id": f"{task_id}::parent",
                    "status": "REPLAY_EXHAUSTED",
                    "attempts": [],
                }
            )
        )

    calls: list[str] = []
    response = (
        {"experience_nodes": [], "edges": []}
        if empty_response
        else _experience_response()
    )

    class RawSavingLLM(ScriptedLLM):
        def __init__(self, raw_path: Path):
            super().__init__(response)
            self.raw_path = raw_path

        @property
        def protocol_identity(self):
            return {
                **super().protocol_identity,
                "prompt_sha256": SUCCESS_PROMPT_SHA256,
                "request_kind": SUCCESS_EXTRACTION_KIND,
            }

        async def complete_json_async(self, **request):
            calls.append(request["request_id"])
            protocol = dict(self.protocol_identity)
            self.raw_path.parent.mkdir(parents=True, exist_ok=True)
            self.raw_path.write_text(
                json.dumps(
                    {
                        "format": SOURCE_RAW_RESPONSE_FORMAT,
                        "outcome": "COMPLETE",
                        "request_kind": request["kind"],
                        "request_id": request["request_id"],
                        "system_prompt_sha256": hashlib.sha256(
                            request["system_prompt"].encode("utf-8")
                        ).hexdigest(),
                        "payload_sha256": hashlib.sha256(
                            canonical_json_bytes(request["payload"])
                        ).hexdigest(),
                        "source_protocol": protocol,
                        "source_protocol_sha256": hashlib.sha256(
                            canonical_json_bytes(protocol)
                        ).hexdigest(),
                        "response": json.dumps(response),
                    }
                )
            )
            return json.loads(json.dumps(response))

    def extractor_factory(origin: str, raw_path: Path):
        assert origin == "ORIGINAL_SUCCESS"
        return SuccessfulTrajectoryExperienceExtractor(RawSavingLLM(raw_path))

    original_writer = source_rebuild_module._write_json_output
    crash_target = (
        "terminal_exclusion.json" if empty_response else "extraction.json"
    )
    crashed = False

    def crash_after_raw_response(path: Path, payload: dict) -> None:
        nonlocal crashed
        if (
            not crashed
            and path.name == crash_target
            and path.parent.name.startswith("000_")
        ):
            crashed = True
            raise RuntimeError("synthetic post-response crash")
        original_writer(path, payload)

    monkeypatch.setattr(
        source_rebuild_module,
        "_write_json_output",
        crash_after_raw_response,
    )
    kwargs = {
        "original_records": original_records,
        "replay_outcomes": replay_outcomes,
        "replay_trajectory_loader": lambda _path: _successful_trajectory(),
        "extractor_factory": extractor_factory,
        "checkpoint_dir": tmp_path / "checkpoints",
        "selected_train_indices": range(8),
    }
    with pytest.raises(RuntimeError, match="synthetic post-response crash"):
        rebuild_section_source_parallel(**kwargs)

    monkeypatch.setattr(
        source_rebuild_module,
        "_write_json_output",
        original_writer,
    )
    calls_before_resume = len(calls)
    resumed = rebuild_section_source_parallel(**kwargs)

    assert len(calls) == calls_before_resume
    assert len(resumed.section_graphs["workflows"]) == (0 if empty_response else 8)
    if empty_response:
        assert resumed.source_exclusions[0]["status"] == (
            "SOURCE_EXCLUDED_NO_REUSABLE_EXPERIENCE"
        )


def test_parallel_source_rebuild_does_not_retry_post_extraction_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_records = [_successful_trajectory("synthetic-success")]
    replay_outcomes = []
    for index in range(1, 200):
        task_id = f"synthetic-failed-{index:03d}"
        original_records.append(
            {
                "task_id": task_id,
                "trajectory_id": f"{task_id}::parent",
                "success": False,
            }
        )
        replay_outcomes.append(
            _with_source_replay_protocol(
                {
                    "task_id": task_id,
                    "parent_trajectory_id": f"{task_id}::parent",
                    "status": "REPLAY_EXHAUSTED",
                    "attempts": [],
                }
            )
        )

    calls = 0

    class SavedResponseLLM(ScriptedLLM):
        def __init__(self, response, raw_path: Path):
            super().__init__(response)
            self.raw_path = raw_path

        @property
        def protocol_identity(self):
            return {
                **super().protocol_identity,
                "prompt_sha256": SUCCESS_PROMPT_SHA256,
            }

        async def complete_json_async(self, **request):
            nonlocal calls
            calls += 1
            self.raw_path.parent.mkdir(parents=True, exist_ok=True)
            self.raw_path.write_text(
                json.dumps(
                    {
                        "format": SOURCE_RAW_RESPONSE_FORMAT,
                        "outcome": "COMPLETE",
                        "response": json.dumps(self.response),
                    }
                )
            )
            return json.loads(json.dumps(self.response))

    def reject_post_extraction(**_kwargs):
        raise ValueError("post-response provenance failure")

    monkeypatch.setattr(
        source_rebuild_module, "_extract_source_row", reject_post_extraction
    )

    with pytest.raises(ValueError, match="post-response provenance failure"):
        rebuild_section_source_parallel(
            original_records=original_records,
            replay_outcomes=replay_outcomes,
            replay_trajectory_loader=lambda _path: _successful_trajectory(),
            extractor_factory=lambda _origin, raw_path: (
                SuccessfulTrajectoryExperienceExtractor(
                    SavedResponseLLM(_experience_response(), raw_path)
                )
            ),
            checkpoint_dir=tmp_path / "checkpoints",
        )

    assert calls == 1


def test_post_run_retry_reuses_draft_and_retries_only_failed_review(
    tmp_path: Path,
) -> None:
    records = [
        _successful_trajectory(f"review-retry-{index:03d}")
        for index in range(8)
    ]
    records.extend(
        {
            "task_id": f"unselected-failure-{index:03d}",
            "trajectory_id": f"unselected-failure-{index:03d}::parent",
            "instruction": "Unselected synthetic task.",
            "success": False,
        }
        for index in range(8, 200)
    )
    outcomes = [
        _with_source_replay_protocol(
            {
                "task_id": f"unselected-failure-{index:03d}",
                "parent_trajectory_id": (
                    f"unselected-failure-{index:03d}::parent"
                ),
                "status": "REPLAY_EXHAUSTED",
                "attempts": [],
            }
        )
        for index in range(8, 200)
    ]

    extraction_calls: list[str] = []

    class CurrentDraftLLM(ScriptedLLM):
        @property
        def protocol_identity(self):
            return {
                "format": SUCCESS_SOURCE_PROTOCOL_FORMAT,
                "request_kind": SUCCESS_EXTRACTION_KIND,
                "model": REPAIR_SOURCE_MODEL,
                "temperature": REPAIR_SOURCE_TEMPERATURE,
                "thinking": REPAIR_SOURCE_THINKING,
                "max_tokens": REPAIR_SOURCE_MAX_TOKENS,
                "timeout_seconds": REPAIR_SOURCE_TIMEOUT_SECONDS,
                "generation_config": _source_generation_config(),
                "retry_waits_seconds": list(PRODUCER_TRANSPORT_RETRY_WAITS),
                "runtime_timeout_retries": PRODUCER_RUNTIME_TIMEOUT_RETRIES,
                "prompt_sha256": SUCCESS_PROMPT_SHA256,
                "service_url": "http://127.0.0.1:9999/v1",
            }

        async def complete_json_async(self, **request):
            extraction_calls.append(request["request_id"])
            return await super().complete_json_async(**request)

    review_calls: list[str] = []

    class RetryReviewLLM(_PassThroughReviewLLM):
        def __init__(self, raw_path: Path) -> None:
            self.raw_path = raw_path

        async def complete_json_async(self, **request):
            review_calls.append(self.raw_path.name)
            if (
                self.raw_path.parent.name.startswith("000_")
                and self.raw_path.name.startswith(
                    "raw_review_response_attempt_"
                )
            ):
                raise RequestCompletionLengthExceeded(
                    "synthetic initial review exhaustion",
                    partial_content='{"experience_nodes": [',
                )
            return await super().complete_json_async(**request)

    checkpoints = tmp_path / "checkpoints"
    kwargs = {
        "original_records": records,
        "replay_outcomes": outcomes,
        "replay_trajectory_loader": lambda _path: _successful_trajectory(),
        "extractor_factory": lambda _origin, _raw_path: (
            SuccessfulTrajectoryExperienceExtractor(
                CurrentDraftLLM(_experience_response())
            )
        ),
        "reviewer_factory": lambda raw_path: ExperienceSourceReviewer(
            RetryReviewLLM(raw_path)
        ),
        "checkpoint_dir": checkpoints,
        "selected_train_indices": range(8),
    }
    first = _rebuild_section_source_parallel(**kwargs)
    assert first.source_audit[0]["source_review_status"] == (
        "REVIEW_DRAFT_FALLBACK"
    )
    assert len(extraction_calls) == 8
    extraction_count = len(extraction_calls)
    accepted_review_count = len(review_calls) - 3

    retried = _rebuild_section_source_parallel(
        **kwargs, retry_failed_reviews=True
    )

    assert len(extraction_calls) == extraction_count
    assert all(
        row["source_review_status"] == "REVIEW_ACCEPTED"
        for row in retried.source_audit
    )
    assert len(review_calls) == accepted_review_count + 4
    assert review_calls[-1] == "raw_review_retry_response_attempt_01.json"
    assert not any(
        path.name.startswith("raw_review_retry_response_attempt_")
        for path in checkpoints.glob("00[1-7]_*/*")
    )
    assert SOURCE_REVIEW_RETRY_STATUS != "REVIEW_DRAFT_FALLBACK"


def test_review_retry_cli_replaces_the_initial_split_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_path = tmp_path / "original.json"
    outcome_path = tmp_path / "outcomes.json"
    original_path.write_text("[{}]", encoding="utf-8")
    outcome_path.write_text("[]", encoding="utf-8")
    section_output = tmp_path / "batch.json"
    audit_output = tmp_path / "batch.audit.json"
    calls = 0

    def fake_build(**_kwargs):
        nonlocal calls
        calls += 1
        operation = f"Reviewed reusable operation version {calls}."
        graph = {
            "format": SECTION_GRAPH_FORMAT,
            "source_split": "train[0,200)",
            "workflows": [
                {
                    "train_index": 0,
                    "task_id": "retry-cli",
                    "query_text": "Review retry publication test.",
                    "experience_nodes": [
                        {
                            "operation": operation,
                            "applicability": ["Use for this synthetic test."],
                            "inputs": [
                                {"artifact": "input", "condition": "available"}
                            ],
                            "outputs": [
                                {"artifact": "output", "condition": "verified"}
                            ],
                        }
                    ],
                    "edges": [],
                }
            ],
        }
        return source_rebuild_module.ExperienceSourceBuild(
            graph,
            (
                {
                    "train_index": 0,
                    "task_id": "retry-cli",
                    "trajectory_id": "retry-cli::trajectory",
                    "origin": "ORIGINAL_SUCCESS",
                    "source_review_status": "REVIEW_ACCEPTED",
                    "discarded_edge_reasons": [],
                },
            ),
        )

    monkeypatch.setattr(
        source_rebuild_module,
        "rebuild_section_source_parallel",
        fake_build,
    )
    monkeypatch.setenv("DEGS_API_KEY", "synthetic-test-key")
    argv = [
        "--original-records",
        str(original_path),
        "--replay-outcomes",
        str(outcome_path),
        "--section-graphs-output",
        str(section_output),
        "--audit-output",
        str(audit_output),
        "--checkpoint-dir",
        str(tmp_path / "checkpoints"),
        "--base-url",
        "http://127.0.0.1:9999/v1",
    ]
    for index in range(8):
        argv.extend(("--batch-train-index", str(index)))

    assert source_rebuild_module.main(argv) == 0
    initial_source = section_output.read_bytes()
    initial_audit = audit_output.read_bytes()
    real_replace = source_rebuild_module._replace_output

    def interrupt_after_source_replace(path, payload):
        real_replace(path, payload)
        if Path(path) == section_output:
            raise RuntimeError("synthetic split-publication interruption")

    monkeypatch.setattr(
        source_rebuild_module,
        "_replace_output",
        interrupt_after_source_replace,
    )
    with pytest.raises(
        RuntimeError, match="synthetic split-publication interruption"
    ):
        source_rebuild_module.main([*argv, "--retry-failed-reviews"])
    assert section_output.read_bytes() != initial_source
    assert audit_output.read_bytes() == initial_audit

    monkeypatch.setattr(
        source_rebuild_module, "_replace_output", real_replace
    )
    assert source_rebuild_module.main(
        [*argv, "--retry-failed-reviews"]
    ) == 0
    assert audit_output.read_bytes() != initial_audit
    assert json.loads(section_output.read_text())["workflows"][0][
        "experience_nodes"
    ][0]["operation"].endswith("version 3.")
    assert json.loads(audit_output.read_text())["review_retry_mode"] is True
    assert section_output.stat().st_mode & 0o777 == 0o600
    assert audit_output.stat().st_mode & 0o777 == 0o600
