from __future__ import annotations

import json

import pytest

from degs import benchmark


def test_resume_requires_an_existing_run_directory(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="must already exist"):
        benchmark.run(
            data_path=tmp_path,
            snapshot_manifest_path=tmp_path / "snapshot.json",
            state_db_path=tmp_path / "state.sqlite3",
            bundle_dir=tmp_path / "bundle",
            run_dir=tmp_path / "run",
            base_url="http://127.0.0.1:9999/v1",
            resume=True,
        )


def test_resume_cache_identity_tracks_actual_experience_content() -> None:
    first = {
        "bundle_self_sha256": "declared",
        "experience_provider": {"sha256": "a" * 64},
    }
    second = {
        **first,
        "experience_provider": {"sha256": "b" * 64},
    }
    assert benchmark._resume_cache_identity(first) != benchmark._resume_cache_identity(
        second
    )


def _prepare_resume_test(tmp_path, monkeypatch):
    records = [
        {
            "task_id": str(index),
            "instruction": f"q{index}",
            "spreadsheet_path": f"spreadsheet/{index}",
            "instruction_type": "Cell-Level Manipulation",
            "answer_position": "A1",
        }
        for index in range(200)
    ]
    monkeypatch.setattr(
        benchmark,
        "verify_contextual_bundle",
        lambda **_kwargs: type(
            "Bundle",
            (),
            {
                "manifest": {
                    "fixed_denominator": 200,
                    "row_count": 200,
                    "model": benchmark.MODEL,
                    "generation_base_url": "http://127.0.0.1:9999/v1",
                    "snapshot_id": "snapshot-final",
                }
            },
        )(),
    )

    class Provider:
        @classmethod
        def from_bundle(cls, _root):
            return cls()

        def for_instance(self, _task_id):
            return object()

        def identity(self):
            return {"sha256": "bundle"}

    calls: list[str] = []

    class Runner:
        def __init__(self, **_kwargs):
            pass

        def run_instance(self, instance):
            calls.append(str(instance.id))
            return type(
                "Result",
                (),
                {
                    "id": str(instance.id),
                    "instruction": instance.instruction,
                    "success": True,
                    "test_cases": [],
                },
            )()

    manifest = {
        "protocol_sha256": "same-run",
        "created_at": "fresh",
        "fixed_denominator": 200,
        "instance_ids": [str(index) for index in range(200)],
    }
    class Store:
        def __init__(self, *_args, **_kwargs):
            self.head_snapshot_id = "snapshot-final"

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    monkeypatch.setattr(benchmark, "EIRGuidanceProvider", Provider)
    monkeypatch.setattr(benchmark, "EIRStateStore", Store)
    monkeypatch.setattr(
        benchmark, "_load_development_harness_records", lambda _path: records
    )
    monkeypatch.setattr(benchmark, "_manifest", lambda **_kwargs: manifest)
    monkeypatch.setattr(benchmark, "DEGSExperienceAgent", lambda **_kwargs: object())
    monkeypatch.setattr(benchmark, "SpreadsheetBenchRunner", Runner)
    monkeypatch.setattr(
        benchmark,
        "_serialize_result",
        lambda result, _output, *, spreadsheet_path: {
            "id": result.id,
            "instruction": result.instruction,
            "success": result.success,
            "test_cases": [],
        },
    )
    monkeypatch.setenv(benchmark.API_KEY_ENV, "test-key")

    run_dir = tmp_path / "run"
    (run_dir / "outputs").mkdir(parents=True)
    (run_dir / "logs").mkdir()
    (run_dir / "outputs/run_manifest.json").write_text(
        json.dumps({**manifest, "created_at": "stored"})
    )
    (tmp_path / "snapshot.json").write_text(
        json.dumps(
            {
                "format": benchmark.DYNAMIC_TRAIN_FORMAT,
                "batch_index": 24,
                "snapshot_id": "snapshot-final",
            }
        )
    )
    return run_dir, calls


def test_resume_runs_only_missing_rows(tmp_path, monkeypatch) -> None:
    run_dir, calls = _prepare_resume_test(tmp_path, monkeypatch)
    existing = [
        {
            "id": str(index),
            "instruction": f"q{index}",
            "success": index != 1,
            "test_cases": [],
        }
        for index in range(199)
    ]
    (run_dir / "results.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in existing)
    )

    result = benchmark.run(
        data_path=tmp_path,
        snapshot_manifest_path=tmp_path / "snapshot.json",
        state_db_path=tmp_path / "state.sqlite3",
        bundle_dir=tmp_path / "bundle",
        run_dir=run_dir,
        base_url="http://127.0.0.1:9999/v1",
        resume=True,
    )

    assert calls == ["199"]
    assert len(result["results"]) == 200
    assert result["results"][1]["success"] is False

    with (run_dir / "results.jsonl").open("a") as ledger:
        ledger.write('{"id":')
    calls.clear()
    resumed = benchmark.run(
        data_path=tmp_path,
        snapshot_manifest_path=tmp_path / "snapshot.json",
        state_db_path=tmp_path / "state.sqlite3",
        bundle_dir=tmp_path / "bundle",
        run_dir=run_dir,
        base_url="http://127.0.0.1:9999/v1",
        resume=True,
    )
    assert calls == []
    assert len(resumed["results"]) == 200
    assert (run_dir / "results.jsonl").read_bytes().endswith(b"\n")


def test_resume_rejects_changed_instruction(tmp_path, monkeypatch) -> None:
    run_dir, _calls = _prepare_resume_test(tmp_path, monkeypatch)
    (run_dir / "results.jsonl").write_text(
        json.dumps({"id": "0", "instruction": "different"}) + "\n"
    )
    with pytest.raises(ValueError, match="result ledger differs"):
        benchmark.run(
            data_path=tmp_path,
            snapshot_manifest_path=tmp_path / "snapshot.json",
            state_db_path=tmp_path / "state.sqlite3",
            bundle_dir=tmp_path / "bundle",
            run_dir=run_dir,
            base_url="http://127.0.0.1:9999/v1",
            resume=True,
        )
