from __future__ import annotations

import fcntl
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from degs.ood_benchmark import _manifest
from degs.ood_benchmark import run as run_ood
from degs.ood_evaluate import parse_prediction


class _Provider:
    def identity(self) -> dict[str, object]:
        return {"provider": "test", "row_count": 1}


def test_ood_agent_manifest_binds_frozen_protocol_and_prompt(tmp_path: Path) -> None:
    population = {
        "dataset": "wikitq",
        "task_count": 1,
        "self_sha256": "population",
        "query_projection_sha256": "queries",
        "input_tree_sha256": "inputs",
    }
    bundle = SimpleNamespace(
        manifest={
            "self_sha256": "bundle",
            "experience_sha256": "experience",
            "row_count": 1,
        }
    )
    manifest = _manifest(
        population=population,
        prepared_data_path=tmp_path / "prepared",
        bundle=bundle,
        provider=_Provider(),
        source_dataset_path=tmp_path / "train.json",
        snapshot_manifest_path=tmp_path / "snapshot.json",
        state_db_path=tmp_path / "state.sqlite3",
        base_url="http://127.0.0.1:18081/v1",
    )
    assert manifest["gold_available_to_agent"] is False
    assert manifest["task_count"] == 1
    assert manifest["workers"] == 8
    assert manifest["max_turns"] == 30
    assert manifest["max_tokens"] == 32_000
    assert len(manifest["agent_prompt_sha256"]) == 64


def test_prediction_parser_preserves_official_multi_value_protocol() -> None:
    assert parse_prediction('["A", "B"]') == ["A", "B"]
    assert parse_prediction("A|B") == ["A", "B"]
    assert parse_prediction(3) == [3]
    assert parse_prediction(None) == []


def test_ood_run_rejects_concurrent_owner(tmp_path: Path) -> None:
    target = tmp_path / "run"
    lock_path = tmp_path / ".run.run.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="another OOD runner"):
            run_ood(
                source_dataset_path=tmp_path / "source.json",
                prepared_data_path=tmp_path / "prepared",
                snapshot_manifest_path=tmp_path / "snapshot.json",
                state_db_path=tmp_path / "state.sqlite3",
                bundle_dir=tmp_path / "bundle",
                run_dir=target,
                base_url="http://127.0.0.1:18081/v1",
            )
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
