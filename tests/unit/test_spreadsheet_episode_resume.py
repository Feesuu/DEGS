from __future__ import annotations

import json
import hashlib
from pathlib import Path
import subprocess

from degs.contextual_retrieval import ContextualRetrieval
from degs.dynamic_train import PreparedEpisode
from degs.episode_evidence import EvidenceItem, EpisodeEvidence, EpisodeOutcome
from degs.spreadsheet_episode import SpreadsheetEpisodeAdapter, _batch_semantic_identity


def test_batch_resume_identity_ignores_runtime_location_and_concurrency() -> None:
    first = {
        "dataset_sha256": "a" * 64,
        "model": "Qwen3.5-9B-AWQ",
        "base_url": "http://first/v1",
        "workers": 8,
        "python_version": "3.12.1",
    }
    second = {
        **first,
        "base_url": "http://second/v1",
        "workers": 96,
        "python_version": "3.12.9",
    }
    assert _batch_semantic_identity(first) == _batch_semantic_identity(second)


def test_completed_spreadsheet_episode_batch_resumes_without_agent(
    tmp_path: Path,
) -> None:
    adapter = SpreadsheetEpisodeAdapter.__new__(SpreadsheetEpisodeAdapter)
    adapter.run_dir = tmp_path
    adapter.dataset_contract_id = "test-contract"
    prepared = PreparedEpisode(
        0,
        "task-0",
        "Apply the requested operation.",
        (EvidenceItem("context:0", "state", "Visible input."),),
        {},
    )
    retrieval = ContextualRetrieval("G0", (), (), ())
    episode = EpisodeEvidence(
        "episode-0",
        "test-contract",
        0,
        "task-0",
        "G0",
        prepared.query_text,
        prepared.observable_context,
        retrieval.to_dict(),
        (),
        (EvidenceItem("trace:original:turn:0:action", "action", "Acted."),),
        (EvidenceItem("verifier:original:overall", "verifier_success", "pass"),),
        EpisodeOutcome.ORIGINAL_SUCCESS,
    )
    path = tmp_path / "batches/batch_00/runtime/episodes.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps([episode.to_learning_payload()]), encoding="utf-8")

    resumed = adapter._execute_batch_sync(
        (prepared,),
        "G0",
        (retrieval,),
        ((),),
        ("",),
    )
    assert resumed == (episode,)


def test_train_verifier_runs_through_pinned_vendor_runtime(
    tmp_path: Path, monkeypatch
) -> None:
    adapter = SpreadsheetEpisodeAdapter.__new__(SpreadsheetEpisodeAdapter)
    adapter.dataset_path = tmp_path / "dataset"
    adapter.runtime_root = tmp_path / "runtime"
    adapter.generation_base_url = "http://generation.test/v1"
    adapter.model = "Qwen3.5-27B-AWQ"
    adapter.dataset_path.mkdir()
    (adapter.dataset_path / "dataset.json").write_text("[]", encoding="utf-8")
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")

    def fake_run(command, **kwargs):
        result_path = Path(command[command.index("--results_file") + 1])
        result_path.write_text('{"results": []}', encoding="utf-8")
        assert command[command.index("--expected-model") + 1] == adapter.model
        assert command[command.index("--start_idx") + 1] == "0"
        assert kwargs["env"]["PYTHONPATH"].split(":")[0] == str(
            adapter.runtime_root / "src"
        )
        return subprocess.CompletedProcess(command, 0, "verified\n", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = adapter._evaluate(
        output_dir=output_dir,
        manifest_path=manifest_path,
        start_idx=0,
        end_idx=8,
        recalc_dir=tmp_path / "recalculated",
    )
    assert result == {"results": []}
    assert (tmp_path / "verifier.log").read_text(encoding="utf-8") == "verified\n"


def test_spreadsheet_trajectory_resume_requires_exact_export_identity(
    tmp_path: Path,
) -> None:
    adapter = SpreadsheetEpisodeAdapter.__new__(SpreadsheetEpisodeAdapter)
    adapter.dataset_path = tmp_path / "dataset"
    adapter.dataset_path.mkdir()
    dataset_path = adapter.dataset_path / "dataset.json"
    dataset_path.write_text('[{"id": "task-0"}]', encoding="utf-8")
    evaluation_path = tmp_path / "evaluation.json"
    evaluation_path.write_text('{"results": []}', encoding="utf-8")
    run_manifest_path = tmp_path / "run_manifest.json"
    run_manifest_path.write_text(
        '{"protocol_sha256": "protocol-0"}', encoding="utf-8"
    )
    records_path = tmp_path / "records.json"
    records_path.write_text('[{"task_id": "task-0"}]', encoding="utf-8")

    def sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    export_manifest = {
        "format": "ordered_spreadsheetbench_trajectories_v1",
        "dataset_sha256": sha256(dataset_path),
        "evaluation_sha256": sha256(evaluation_path),
        "source_run_manifest_sha256": sha256(run_manifest_path),
        "source_protocol_sha256": "protocol-0",
        "start_idx": 0,
        "end_idx": 1,
        "record_count": 1,
        "task_ids": ["task-0"],
        "records_sha256": sha256(records_path),
    }
    records_path.with_suffix(".manifest.json").write_text(
        json.dumps(export_manifest), encoding="utf-8"
    )
    adapter._validate_records_artifact(
        records_path=records_path,
        evaluation_path=evaluation_path,
        run_manifest_path=run_manifest_path,
        start_idx=0,
        end_idx=1,
        expected_task_ids=["task-0"],
    )

    records_path.write_text('[{"task_id": "wrong"}]', encoding="utf-8")
    try:
        adapter._validate_records_artifact(
            records_path=records_path,
            evaluation_path=evaluation_path,
            run_manifest_path=run_manifest_path,
            start_idx=0,
            end_idx=1,
            expected_task_ids=["task-0"],
        )
    except ValueError as exc:
        assert "records_sha256" in str(exc)
    else:
        raise AssertionError("corrupt trajectory records were accepted")
