from __future__ import annotations

import asyncio
from pathlib import Path
import hashlib

from degs.contextual_binding import BindingCondition, ExperienceExpectation
from degs.eir_graph import (
    CanonicalResolution,
    LearningDeltaApplier,
    compile_eir_experience_graph,
)
from degs.eir_graph_quality import build_eir_graph_quality_audit
from degs.episode_evidence import EvidenceItem, EpisodeEvidence, EpisodeOutcome
from degs.episode_learning import (
    ExperienceUpdate,
    LearnedExperienceNode,
    LearningDelta,
    ProcedureStep,
    ProcedureStepKind,
    UpdateAction,
)
from degs.graph_dataset_contract import SPREADSHEETBENCH_GRAPH_CONTRACT
from degs.core import canonical_json_bytes
from degs.section_graph import ExperienceEdge, ExperienceNode, IOContract
from degs.state_store import EIRStateStore


def _node(operation: str, applicability: str) -> ExperienceNode:
    return ExperienceNode(
        operation,
        (applicability,),
        (
            IOContract(
                "task-bound parameter",
                "Read the varying parameter from the current task or observable context.",
            ),
        ),
        (IOContract("state transition", "The intended state is produced."),),
    )


class _NewOnlyResolver:
    async def resolve(self, *, source_node_id, experience, active):
        return CanonicalResolution(None, experience, "No equivalent operation exists.")


def _episode(
    *,
    episode_id: str,
    train_index: int,
    read_snapshot_id: str,
    expectations: tuple[ExperienceExpectation, ...] = (),
    outcome: EpisodeOutcome = EpisodeOutcome.ORIGINAL_SUCCESS,
) -> EpisodeEvidence:
    repair = outcome is EpisodeOutcome.REPAIR_SUCCESS
    return EpisodeEvidence(
        episode_id=episode_id,
        dataset_contract_id="spreadsheetbench-v1",
        train_index=train_index,
        task_id=f"task-{train_index}",
        read_snapshot_id=read_snapshot_id,
        query_text="Apply the case requested by this task, then validate the result.",
        observable_context=(
            EvidenceItem("context:0", "workbook", "The target contains text."),
        ),
        retrieval_context={"anchors": []},
        expectations=expectations,
        original_trace=(
            EvidenceItem("trace:original:turn:1:action", "action", "Edited the target."),
        ),
        original_verifier=(
            EvidenceItem(
                "verifier:original:overall",
                "verifier_failure" if repair else "verifier_success",
                "mismatch" if repair else "pass",
            ),
        ),
        outcome=outcome,
        final_patch=(
            EvidenceItem("patch:final", "patch", "Bind case from the current task."),
        )
        if repair
        else (),
        replay_trace=(
            EvidenceItem(
                "trace:replay:turn:1:action", "action", "Applied the task-bound case."
            ),
        )
        if repair
        else (),
        replay_verifier=(
            EvidenceItem("verifier:replay:overall", "verifier_success", "pass"),
        )
        if repair
        else (),
    )


def test_two_batches_preserve_identity_and_add_cross_workflow_edge(
    tmp_path: Path,
) -> None:
    asyncio.run(_run_two_batch_case(tmp_path))


async def _run_two_batch_case(tmp_path: Path) -> None:
    first_node = _node(
        "Transform text using required_case read from the current task.",
        "The task requests a text case.",
    )
    second_node = _node(
        "Validate that transformed text follows required_case.",
        "Text was transformed under a task-visible case constraint.",
    )
    with EIRStateStore(
        tmp_path / "state.sqlite3",
        dataset_contract=SPREADSHEETBENCH_GRAPH_CONTRACT,
    ) as store:
        applier = LearningDeltaApplier(store, _NewOnlyResolver())
        snapshot_0 = store.begin_snapshot(
            batch_index=0,
            parent_snapshot_id=None,
            operation_input_sha256="1" * 64,
        )
        episode_0 = _episode(
            episode_id="episode-0", train_index=0, read_snapshot_id="G0"
        )
        delta_0 = LearningDelta(
            "episode-0",
            "G0",
            (),
            (
                LearnedExperienceNode(
                    first_node,
                    ("trace:original:turn:1:action", "verifier:original:overall"),
                ),
            ),
            (),
            (ProcedureStep(ProcedureStepKind.NEW, node_index=0),),
            (),
            (),
        )
        audit_0 = await applier.apply_batch(
            snapshot_id=snapshot_0, episodes=((episode_0, delta_0),)
        )
        store.commit_snapshot(snapshot_0)
        graph_0, versions_0 = compile_eir_experience_graph(
            store, snapshot_id=snapshot_0
        )
        assert audit_0[0].status == "COMMITTED"
        assert len(graph_0.nodes) == 1
        canonical_id = graph_0.nodes[0].canonical_id
        assert versions_0 == {canonical_id: 1}

        expectation = ExperienceExpectation(
            canonical_id,
            1,
            BindingCondition.SATISFIED,
            ("query:0",),
            "Perform the task-bound case transformation.",
            (),
            "Use the case requested by this task.",
            "The transformed value has the requested case.",
        )
        snapshot_1 = store.begin_snapshot(
            batch_index=1,
            parent_snapshot_id=snapshot_0,
            operation_input_sha256="2" * 64,
        )
        repair_episode = _episode(
            episode_id="episode-8",
            train_index=8,
            read_snapshot_id=snapshot_0,
            expectations=(expectation,),
            outcome=EpisodeOutcome.REPAIR_SUCCESS,
        )
        qualified = _node(
            first_node.operation,
            "The current task explicitly exposes a required text case.",
        )
        repair_delta = LearningDelta(
            "episode-8",
            snapshot_0,
            (
                ExperienceUpdate(
                    canonical_id,
                    1,
                    UpdateAction.QUALIFY,
                    ("trace:replay:turn:1:action",),
                    ("verifier:replay:overall",),
                    (
                        "verifier:original:overall",
                        "patch:final",
                        "trace:replay:turn:1:action",
                        "verifier:replay:overall",
                    ),
                    "Validated repair shows the binding must be task-visible.",
                    qualified,
                ),
            ),
            (),
            (),
            (ProcedureStep(ProcedureStepKind.CANONICAL, canonical_id=canonical_id),),
            (),
            (),
        )
        success_episode = _episode(
            episode_id="episode-9",
            train_index=9,
            read_snapshot_id=snapshot_0,
            expectations=(expectation,),
        )
        success_delta = LearningDelta(
            "episode-9",
            snapshot_0,
            (
                ExperienceUpdate(
                    canonical_id,
                    1,
                    UpdateAction.SUPPORT,
                    ("trace:original:turn:1:action",),
                    ("verifier:original:overall",),
                    (),
                    "The operation was used successfully.",
                    None,
                ),
            ),
            (
                LearnedExperienceNode(
                    second_node,
                    ("trace:original:turn:1:action", "verifier:original:overall"),
                ),
            ),
            (),
            (
                ProcedureStep(ProcedureStepKind.CANONICAL, canonical_id=canonical_id),
                ProcedureStep(ProcedureStepKind.NEW, node_index=0),
            ),
            (ExperienceEdge(0, 1),),
            (),
        )
        audit_1 = await applier.apply_batch(
            snapshot_id=snapshot_1,
            episodes=((success_episode, success_delta), (repair_episode, repair_delta)),
        )
        store.commit_snapshot(snapshot_1)
        graph_1, versions_1 = compile_eir_experience_graph(
            store, snapshot_id=snapshot_1
        )

        assert [row.train_index for row in audit_1] == [8, 9]
        assert store.active_canonical(canonical_id, snapshot_id=snapshot_0).version == 1
        assert store.active_canonical(canonical_id, snapshot_id=snapshot_1).version == 2
        assert versions_1[canonical_id] == 2
        assert len(graph_1.nodes) == 2
        assert len(graph_1.edges) == 1
        assert graph_1.edges[0].source == canonical_id
        assert graph_1.edges[0].supporting_workflow_ids == (9,)
        payload = graph_1.to_dict()
        assert payload["format"] == "degs_eir_experience_graph_v1"
        assert payload["snapshot_id"] == snapshot_1
        assert payload["canonical_versions"] == dict(versions_1)
        unsigned = {
            key: value
            for key, value in payload.items()
            if key != "experience_graph_sha256"
        }
        assert hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest() == payload[
            "experience_graph_sha256"
        ]
        quality = build_eir_graph_quality_audit(store, snapshot_id=snapshot_1)
        assert quality["experience_action_counts"] == {"QUALIFY": 1, "SUPPORT": 1}

    with EIRStateStore(
        tmp_path / "state.sqlite3",
        dataset_contract=SPREADSHEETBENCH_GRAPH_CONTRACT,
    ) as reopened:
        assert reopened.head_snapshot_id == snapshot_1
        persisted, persisted_versions = compile_eir_experience_graph(
            reopened, snapshot_id=snapshot_1
        )
        assert persisted.identity() == graph_1.identity()
        assert persisted_versions == versions_1
