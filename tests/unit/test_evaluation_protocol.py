from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from sb_adapter import evaluate as adapter
from degs import __version__, benchmark, evaluate


@pytest.fixture
def manifest_case(tmp_path, monkeypatch):
    data = tmp_path / "dataset"
    data.mkdir()
    (data / "dataset.json").write_text("[]")
    train = [
        {"task_id": f"train-{i}", "instruction": f"Transform item {i}."}
        for i in range(200)
    ]
    dev = [{"task_id": f"dev-{i}"} for i in range(200)]
    monkeypatch.setattr(adapter, "load_train_queries", lambda _: train)
    monkeypatch.setattr(adapter, "load_development_queries", lambda _: dev)
    monkeypatch.setattr(adapter, "_bundle_link_matches", lambda *a, **k: True)
    payload = benchmark._manifest(
        data_path=data,
        bundle_dir=tmp_path / "bundle",
        bundle_manifest={"self_sha256": "a" * 64},
        provider=SimpleNamespace(identity=lambda: {}),
        snapshot_manifest_path=tmp_path / "snapshot.json",
        state_db_path=tmp_path / "state.sqlite3",
        instance_ids=[row["task_id"] for row in dev],
        base_url="http://127.0.0.1:8000/v1",
    )
    manifest = tmp_path / "run_manifest.json"
    manifest.write_text(json.dumps(payload))
    return manifest, payload, {
        "path": manifest,
        "data_path": data,
        "start_idx": 200,
        "end_idx": 400,
    }


def test_current_manifest_writer_and_evaluator_share_method_version(manifest_case):
    _, payload, kwargs = manifest_case
    assert payload["method_version"] == __version__
    assert all(adapter._validate_run_manifest(**kwargs)["checks"].values())


def test_wrong_method_version_is_rejected(manifest_case):
    manifest, payload, kwargs = manifest_case
    payload["method_version"] = "not-the-current-method"
    protocol = {
        key: value
        for key, value in payload.items()
        if key not in {"protocol_sha256", "created_at"}
    }
    payload["protocol_sha256"] = hashlib.sha256(
        adapter._canonical_json_bytes(protocol)
    ).hexdigest()
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="method_version"):
        adapter._validate_run_manifest(**kwargs)


def test_evaluation_cli_forwards_only_live_run_inputs(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        evaluate, "evaluate_run", lambda **kwargs: calls.append(kwargs) or {}
    )
    assert evaluate.main(
        [
            "--data-path",
            str(tmp_path / "data"),
            "--run-dir",
            str(tmp_path / "run"),
            "--base-url",
            "http://127.0.0.1:8000/v1",
        ]
    ) == 0
    assert calls == [
        {
            "data_path": tmp_path / "data",
            "run_dir": tmp_path / "run",
            "base_url": "http://127.0.0.1:8000/v1",
            "model": "Qwen3.5-9B-AWQ",
        }
    ]
