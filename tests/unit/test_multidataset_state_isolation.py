from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from degs.graph_dataset_contract import GraphDatasetContract
from degs.retrieval_store import RetrievalStore
from degs.state_store import IncrementalStateStore


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_source_graph_database_is_byte_stable_when_opened_read_only(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.sqlite3"
    with IncrementalStateStore(source_path):
        pass
    before = _sha(source_path)

    with IncrementalStateStore(source_path, readonly=True) as source:
        assert source.head_snapshot_id is None

    assert _sha(source_path) == before


def test_target_retrieval_cache_cannot_be_reused_as_source_graph_state(
    tmp_path: Path,
) -> None:
    retrieval_path = tmp_path / "target/retrieval.sqlite3"
    with RetrievalStore(retrieval_path):
        pass

    with pytest.raises(ValueError, match="schema_version"):
        IncrementalStateStore(retrieval_path, readonly=True)


def test_graph_state_identity_includes_dataset_contract(tmp_path: Path) -> None:
    contract = GraphDatasetContract(
        identity="skill2bench-test",
        source_split="skill2bench/train-test",
        train_count=20,
        batch_size=10,
    )
    path = tmp_path / "skill2bench.sqlite3"
    with IncrementalStateStore(path, dataset_contract=contract):
        pass

    other = GraphDatasetContract(
        identity="other-dataset",
        source_split="other/train",
        train_count=20,
        batch_size=10,
    )
    with pytest.raises(ValueError, match="graph_dataset_contract"):
        IncrementalStateStore(path, dataset_contract=other, readonly=True)

    with pytest.raises(ValueError, match="graph_dataset_contract"):
        IncrementalStateStore(path)
