from __future__ import annotations

import pytest

from degs.contextual_binding import (
    BindingCondition,
    BoundParameter,
    ExperienceExpectation,
)
from degs.episode_evidence import (
    EvidenceItem,
    EpisodeEvidence,
    EpisodeOutcome,
    episode_evidence_from_payload,
)
from degs.episode_learning import (
    UpdateAction,
    learning_delta_from_dict,
    parse_learning_delta,
)
from degs.section_graph import ExperienceNode, IOContract


def _experience(applicability: str = "The task explicitly requests a case.") -> dict:
    return {
        "operation": "Convert text using the case requested by the current task.",
        "applicability": [applicability],
        "inputs": [
            {
                "type": "required_case binding",
                "description": "Read required_case from the current task constraint.",
            }
        ],
        "outputs": [
            {
                "type": "text",
                "description": "Text conforming to required_case.",
            }
        ],
    }


def _expectation() -> ExperienceExpectation:
    return ExperienceExpectation(
        canonical_id="C1",
        canonical_version=2,
        condition=BindingCondition.SATISFIED,
        condition_evidence_refs=("query:0",),
        expected_role="Apply the task-visible case rule.",
        bound_parameters=(
            BoundParameter("required_case", "uppercase", "query:0"),
        ),
        guidance="Convert using uppercase.",
        expected_observation="Target text is uppercase.",
    )


def _conflicting_expectation() -> ExperienceExpectation:
    return ExperienceExpectation(
        canonical_id="C1",
        canonical_version=2,
        condition=BindingCondition.CONFLICT,
        condition_evidence_refs=("query:0",),
        expected_role="The current task conflicts with this experience.",
        bound_parameters=(),
        guidance="",
        expected_observation="",
    )


def _episode(outcome: EpisodeOutcome) -> EpisodeEvidence:
    original_success = outcome is EpisodeOutcome.ORIGINAL_SUCCESS
    repair_success = outcome is EpisodeOutcome.REPAIR_SUCCESS
    return EpisodeEvidence(
        episode_id="episode-7",
        dataset_contract_id="spreadsheetbench-v1",
        train_index=7,
        task_id="task-7",
        read_snapshot_id="snapshot-1",
        query_text="Make the target uppercase.",
        observable_context=(EvidenceItem("context:0", "workbook", "Target cells contain text."),),
        retrieval_context={"anchors": [{"canonical_id": "C1", "canonical_version": 2}]},
        expectations=(_expectation(),),
        original_trace=(EvidenceItem("trace:original:turn:1:action", "action", "Converted target text."),),
        original_verifier=(
            EvidenceItem(
                "verifier:original:overall",
                "verifier_success" if original_success else "verifier_failure",
                "pass" if original_success else "case mismatch",
            ),
        ),
        outcome=outcome,
        final_patch=(EvidenceItem("patch:final", "patch", "Bind required_case from the task."),) if repair_success else (),
        replay_trace=(EvidenceItem("trace:replay:turn:1:action", "action", "Converted to uppercase."),) if repair_success else (),
        replay_verifier=(EvidenceItem("verifier:replay:overall", "verifier_success", "pass"),) if repair_success else (),
    )


def _update(action: str, *, revised: dict | None = None) -> dict:
    repair = action in {"QUALIFY", "CORRECT"}
    return {
        "canonical_id": "C1",
        "base_version": 2,
        "action": action,
        "usage_evidence_refs": [
            "trace:replay:turn:1:action" if repair else "trace:original:turn:1:action"
        ],
        "outcome_evidence_refs": [
            "verifier:replay:overall" if repair else "verifier:original:overall"
        ],
        "repair_evidence_refs": (
            [
                "verifier:original:overall",
                "patch:final",
                "trace:replay:turn:1:action",
                "verifier:replay:overall",
            ]
            if repair
            else []
        ),
        "reason": "Evidence supports this disposition.",
        "revised_experience": revised,
    }


def _delta(update: dict, *, new_nodes: list[dict] | None = None) -> dict:
    return {
        "retrieved_experience_updates": [update],
        "new_experience_graph": {
            "experience_nodes": new_nodes or [],
            "edges": [],
        },
        "episode_procedure": {"steps": [], "edges": []},
    }


def test_original_success_can_support_without_creating_a_duplicate() -> None:
    episode = _episode(EpisodeOutcome.ORIGINAL_SUCCESS)
    active = {"C1": ExperienceNode(
        _experience()["operation"],
        tuple(_experience()["applicability"]),
        (IOContract(**_experience()["inputs"][0]),),
        (IOContract(**_experience()["outputs"][0]),),
    )}
    delta = parse_learning_delta(
        _delta(_update("SUPPORT")),
        episode=episode,
        active_experiences=active,
    )
    assert delta.updates[0].action is UpdateAction.SUPPORT
    assert not delta.new_nodes
    assert learning_delta_from_dict(
        delta.to_dict(), episode=episode, active_experiences=active
    ) == delta


def test_conflicting_expectation_cannot_become_positive_evidence() -> None:
    episode = _episode(EpisodeOutcome.ORIGINAL_SUCCESS)
    episode = EpisodeEvidence(
        episode_id=episode.episode_id,
        dataset_contract_id=episode.dataset_contract_id,
        train_index=episode.train_index,
        task_id=episode.task_id,
        read_snapshot_id=episode.read_snapshot_id,
        query_text=episode.query_text,
        observable_context=episode.observable_context,
        retrieval_context=episode.retrieval_context,
        expectations=(_conflicting_expectation(),),
        original_trace=episode.original_trace,
        original_verifier=episode.original_verifier,
        outcome=episode.outcome,
        final_patch=episode.final_patch,
        replay_trace=episode.replay_trace,
        replay_verifier=episode.replay_verifier,
    )
    with pytest.raises(ValueError, match="CONFLICT"):
        parse_learning_delta(
            _delta(_update("SUPPORT")),
            episode=episode,
            active_experiences={},
        )


@pytest.mark.parametrize("action", ["QUALIFY", "CORRECT"])
def test_original_success_cannot_revise_old_experience(action: str) -> None:
    with pytest.raises(ValueError, match="repair-success"):
        parse_learning_delta(
            _delta(_update(action, revised=_experience("Only when explicitly requested."))),
            episode=_episode(EpisodeOutcome.ORIGINAL_SUCCESS),
            active_experiences={},
        )


def test_repair_success_can_qualify_only_applicability() -> None:
    delta = parse_learning_delta(
        _delta(_update("QUALIFY", revised=_experience("Only when the current task explicitly requests a case."))),
        episode=_episode(EpisodeOutcome.REPAIR_SUCCESS),
        active_experiences={"C1": ExperienceNode(
            _experience()["operation"],
            tuple(_experience()["applicability"]),
            (IOContract(**_experience()["inputs"][0]),),
            (IOContract(**_experience()["outputs"][0]),),
        )},
    )
    assert delta.updates[0].action is UpdateAction.QUALIFY


def test_qualify_cannot_change_operation_or_binding() -> None:
    revised = _experience("Only when explicitly requested.")
    revised["operation"] = "Always force lowercase."
    with pytest.raises(ValueError, match="QUALIFY"):
        parse_learning_delta(
            _delta(_update("QUALIFY", revised=revised)),
            episode=_episode(EpisodeOutcome.REPAIR_SUCCESS),
            active_experiences={"C1": ExperienceNode(
                _experience()["operation"],
                tuple(_experience()["applicability"]),
                (IOContract(**_experience()["inputs"][0]),),
                (IOContract(**_experience()["outputs"][0]),),
            )},
        )


def test_repair_revision_requires_complete_causal_evidence() -> None:
    update = _update("CORRECT", revised=_experience("Only when explicitly requested."))
    update["repair_evidence_refs"].remove("patch:final")
    with pytest.raises(ValueError, match="repair evidence"):
        parse_learning_delta(
            _delta(update),
            episode=_episode(EpisodeOutcome.REPAIR_SUCCESS),
            active_experiences={},
        )


def test_unresolved_failure_cannot_add_positive_graph_content() -> None:
    node = {**_experience(), "evidence_refs": ["trace:original:turn:1:action", "verifier:original:overall"]}
    with pytest.raises(ValueError, match="unresolved"):
        parse_learning_delta(
            _delta(_update("NO_EVIDENCE"), new_nodes=[node]),
            episode=_episode(EpisodeOutcome.UNRESOLVED_TASK_FAILURE),
            active_experiences={},
        )


def test_invalid_edge_is_dropped_without_discarding_valid_new_nodes() -> None:
    node = {
        **_experience(),
        "evidence_refs": ["trace:original:turn:1:action", "verifier:original:overall"],
    }
    raw = _delta(_update("NO_EVIDENCE"), new_nodes=[node])
    raw["new_experience_graph"]["edges"] = [{"source": 0, "target": 9}]
    delta = parse_learning_delta(
        raw,
        episode=_episode(EpisodeOutcome.ORIGINAL_SUCCESS),
        active_experiences={},
    )
    assert len(delta.new_nodes) == 1
    assert delta.new_edges == ()
    assert delta.discarded_edge_reasons


def test_duplicate_node_removal_does_not_shift_later_node_identity() -> None:
    duplicate = {
        **_experience(),
        "evidence_refs": ["trace:original:turn:1:action", "verifier:original:overall"],
    }
    residual = {
        **_experience("A task-specific validation is required."),
        "operation": "Validate the transformed text against the requested case.",
        "evidence_refs": ["trace:original:turn:1:action", "verifier:original:overall"],
    }
    raw = _delta(_update("SUPPORT"), new_nodes=[duplicate, residual])
    raw["new_experience_graph"]["edges"] = [{"source": 0, "target": 1}]
    raw["episode_procedure"] = {
        "steps": [
            {"kind": "NEW", "canonical_id": None, "node_index": 0},
            {"kind": "NEW", "canonical_id": None, "node_index": 1},
        ],
        "edges": [{"source": 0, "target": 1}],
    }
    active = ExperienceNode(
        _experience()["operation"],
        tuple(_experience()["applicability"]),
        (IOContract(**_experience()["inputs"][0]),),
        (IOContract(**_experience()["outputs"][0]),),
    )
    delta = parse_learning_delta(
        raw,
        episode=_episode(EpisodeOutcome.ORIGINAL_SUCCESS),
        active_experiences={"C1": active},
    )
    assert len(delta.new_nodes) == 1
    assert delta.new_nodes[0].experience.operation.startswith("Validate")
    assert delta.new_edges == ()
    assert delta.procedure_steps == (
        # The surviving original index 1 is safely remapped to compact index 0.
        delta.procedure_steps[0],
    )
    assert delta.procedure_steps[0].node_index == 0
    assert delta.procedure_edges == ()


def test_episode_evidence_storage_round_trip_is_exact() -> None:
    episode = _episode(EpisodeOutcome.REPAIR_SUCCESS)
    assert episode_evidence_from_payload(episode.to_learning_payload()) == episode
