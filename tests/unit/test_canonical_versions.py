from __future__ import annotations

import hashlib
from pathlib import Path
import sqlite3
import struct

import pytest

from degs.graph_dataset_contract import SPREADSHEETBENCH_GRAPH_CONTRACT
from degs.core import EmbeddedText, embedding_text_sha256
from degs.section_graph import ExperienceNode, IOContract
from degs.state_store import EIRStateStore


def _node(operation: str, applicability: str) -> ExperienceNode:
    return ExperienceNode(
        operation,
        (applicability,),
        (IOContract("parameter binding", "Read the value from the current task."),),
        (IOContract("state", "The requested state transition."),),
    )


def _snapshot(store: EIRStateStore, batch: int, parent: str | None) -> str:
    snapshot = store.begin_snapshot(
        batch_index=batch,
        parent_snapshot_id=parent,
        operation_input_sha256=f"{batch + 1:064x}",
    )
    return snapshot


def _source(
    store: EIRStateStore,
    *,
    snapshot: str,
    source_node_id: str,
    node_index: int,
    experience: ExperienceNode,
) -> None:
    episode_id = f"episode-{snapshot}"
    store.register_episode(
        snapshot_id=snapshot,
        episode_id=episode_id,
        train_index=0,
        task_id="task-0",
        read_snapshot_id="G0",
        outcome="ORIGINAL_SUCCESS",
        evidence={"episode_id": episode_id},
        expectations=[],
    )
    store.add_source_node(
        source_node_id=source_node_id,
        episode_id=episode_id,
        node_index=node_index,
        experience=experience,
        evidence_refs=("trace:original:turn:1:action",),
        snapshot_id=snapshot,
    )


def test_eir_source_state_supports_a_strict_read_only_open(tmp_path: Path) -> None:
    database = tmp_path / "state.sqlite3"
    with EIRStateStore(
        database, dataset_contract=SPREADSHEETBENCH_GRAPH_CONTRACT
    ) as store:
        assert store.head_snapshot_id is None

    with EIRStateStore(
        database,
        dataset_contract=SPREADSHEETBENCH_GRAPH_CONTRACT,
        readonly=True,
    ) as store:
        assert store.connection.execute("PRAGMA query_only").fetchone()[0] == 1
        with pytest.raises(sqlite3.OperationalError):
            store.connection.execute(
                "INSERT INTO eir_metadata(key, value) VALUES ('target', 'leak')"
            )


def test_eir_embedding_cache_persists_before_any_graph_transaction(
    tmp_path: Path,
) -> None:
    database = tmp_path / "state.sqlite3"
    text = "target retrieval document"
    vector_blob = b"".join(struct.pack("<d", value) for value in (1.0, 2.0))
    row = EmbeddedText(
        text,
        (1.0, 2.0),
        hashlib.sha256(vector_blob).hexdigest(),
        "2" * 64,
    )
    with EIRStateStore(
        database, dataset_contract=SPREADSHEETBENCH_GRAPH_CONTRACT
    ) as store:
        store.embedding_cache()[embedding_text_sha256(text)] = row

    with EIRStateStore(
        database, dataset_contract=SPREADSHEETBENCH_GRAPH_CONTRACT
    ) as store:
        assert store.embedding_cache()[embedding_text_sha256(text)] == row


def test_revision_preserves_canonical_identity_and_historical_version(
    tmp_path: Path,
) -> None:
    with EIRStateStore(tmp_path / "state.sqlite3", dataset_contract=SPREADSHEETBENCH_GRAPH_CONTRACT) as store:
        first = _snapshot(store, 0, None)
        experience = _node("Transform text using required_case.", "A case is requested.")
        _source(store, snapshot=first, source_node_id="leaf-0", node_index=0, experience=experience)
        canonical_id = store.create_canonical(
            source_node_id="leaf-0",
            experience=experience,
            snapshot_id=first,
            change_kind="CREATE",
            evidence_event_id=None,
        )
        store.commit_snapshot(first)

        second = _snapshot(store, 1, first)
        version = store.revise_canonical(
            canonical_id=canonical_id,
            base_version=1,
            experience=_node(
                "Transform text using required_case.",
                "The current task explicitly requests a case.",
            ),
            snapshot_id=second,
            change_kind="QUALIFY",
            evidence_event_id="event-1",
        )
        store.commit_snapshot(second)

        assert version == 2
        assert store.active_canonical(canonical_id, snapshot_id=first).version == 1
        current = store.active_canonical(canonical_id, snapshot_id=second)
        assert current.version == 2
        assert current.experience.applicability == (
            "The current task explicitly requests a case.",
        )


def test_merge_keeps_older_identity_and_resolves_alias(tmp_path: Path) -> None:
    with EIRStateStore(tmp_path / "state.sqlite3", dataset_contract=SPREADSHEETBENCH_GRAPH_CONTRACT) as store:
        first = _snapshot(store, 0, None)
        older_experience = _node("Apply a task-bound case.", "Case is specified.")
        newer_experience = _node("Use requested text case.", "Case is specified.")
        _source(store, snapshot=first, source_node_id="leaf-a", node_index=0, experience=older_experience)
        _source(store, snapshot=first, source_node_id="leaf-b", node_index=1, experience=newer_experience)
        older = store.create_canonical(
            source_node_id="leaf-a",
            experience=older_experience,
            snapshot_id=first,
            change_kind="CREATE",
            evidence_event_id=None,
        )
        newer = store.create_canonical(
            source_node_id="leaf-b",
            experience=newer_experience,
            snapshot_id=first,
            change_kind="CREATE",
            evidence_event_id=None,
        )
        store.commit_snapshot(first)

        second = _snapshot(store, 1, first)
        survivor = store.merge_canonical(
            left_canonical_id=older,
            right_canonical_id=newer,
            experience=_node("Use the text case bound from the task.", "Case is specified."),
            snapshot_id=second,
            evidence_event_id="merge-1",
        )
        store.commit_snapshot(second)

        assert survivor == older
        assert store.resolve_canonical_id(newer, snapshot_id=first) == newer
        assert store.resolve_canonical_id(newer, snapshot_id=second) == older
        assert store.active_canonical(older, snapshot_id=second).version == 2
        assert set(store.canonical_member_ids(older, snapshot_id=second)) == {
            "leaf-a",
            "leaf-b",
        }


def test_incompatible_stale_revision_is_recorded_not_applied(tmp_path: Path) -> None:
    with EIRStateStore(tmp_path / "state.sqlite3", dataset_contract=SPREADSHEETBENCH_GRAPH_CONTRACT) as store:
        first = _snapshot(store, 0, None)
        experience = _node("Apply an operation.", "Broad condition.")
        _source(store, snapshot=first, source_node_id="leaf-0", node_index=0, experience=experience)
        canonical_id = store.create_canonical(
            source_node_id="leaf-0",
            experience=experience,
            snapshot_id=first,
            change_kind="CREATE",
            evidence_event_id=None,
        )
        store.commit_snapshot(first)
        second = _snapshot(store, 1, first)
        assert store.revise_canonical(
            canonical_id=canonical_id,
            base_version=1,
            experience=_node("Apply an operation.", "First narrow condition."),
            snapshot_id=second,
            change_kind="QUALIFY",
            evidence_event_id="event-1",
        ) == 2
        assert store.revise_canonical(
            canonical_id=canonical_id,
            base_version=1,
            experience=_node("Apply an operation.", "Second narrow condition."),
            snapshot_id=second,
            change_kind="QUALIFY",
            evidence_event_id="event-2",
        ) is None
        store.commit_snapshot(second)
        assert store.active_canonical(canonical_id, snapshot_id=second).version == 2
        assert store.deferred_revision_count() == 1


def test_identical_same_batch_revision_collapses_to_one_version(tmp_path: Path) -> None:
    with EIRStateStore(
        tmp_path / "state.sqlite3",
        dataset_contract=SPREADSHEETBENCH_GRAPH_CONTRACT,
    ) as store:
        first = _snapshot(store, 0, None)
        experience = _node("Apply an operation.", "Broad condition.")
        _source(
            store,
            snapshot=first,
            source_node_id="leaf-0",
            node_index=0,
            experience=experience,
        )
        canonical_id = store.create_canonical(
            source_node_id="leaf-0",
            experience=experience,
            snapshot_id=first,
            change_kind="CREATE",
            evidence_event_id=None,
        )
        store.commit_snapshot(first)
        second = _snapshot(store, 1, first)
        revised = _node("Apply an operation.", "Narrow condition.")
        assert store.revise_canonical(
            canonical_id=canonical_id,
            base_version=1,
            experience=revised,
            snapshot_id=second,
            change_kind="QUALIFY",
            evidence_event_id="event-a",
        ) == 2
        assert store.revise_canonical(
            canonical_id=canonical_id,
            base_version=1,
            experience=revised,
            snapshot_id=second,
            change_kind="QUALIFY",
            evidence_event_id="event-b",
        ) == 2
        store.commit_snapshot(second)
        assert store.active_canonical(canonical_id, snapshot_id=second).version == 2
        assert store.deferred_revision_count() == 0
