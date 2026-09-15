from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import fetch_spreadsheetbench, preflight_services, summarize_campaign
from scripts.run_full_campaign import (
    _parser,
    _declared_output_paths,
    _output_identities,
    _validate_output_identities,
)
from scripts.run_tableqa_ood import Campaign as OODCampaign
from scripts.run_train_source_replay import _adapt_evaluator_command


ROOT = Path(__file__).resolve().parents[2]


def test_ood_campaign_paths_are_portable_but_graph_identity_is_not(
    tmp_path: Path,
) -> None:
    def args(snapshot: Path, suffix: str = "first") -> SimpleNamespace:
        return SimpleNamespace(
            run_root=tmp_path / "run",
            profile="9b",
            generation_base_url=f"http://generation-{suffix}/v1",
            embedding_base_url=f"http://embedding-{suffix}/v1",
            source_dataset_path=tmp_path / suffix / "dataset.json",
            snapshot_manifest_path=snapshot,
            state_db=tmp_path / suffix / "state.sqlite3",
            wikitq_source_repo=tmp_path / suffix / "wikitq",
            hitab_source_repo=tmp_path / suffix / "hitab",
            python2=f"/opt/{suffix}/python2",
        )

    first = tmp_path / "first-snapshot.json"
    second = tmp_path / "second-snapshot.json"
    first.write_text('{"snapshot_id":"G-final"}', encoding="utf-8")
    second.write_text('{"snapshot_id":"G-final"}', encoding="utf-8")
    OODCampaign(args(first))
    OODCampaign(args(second, "second"))

    second.write_text('{"snapshot_id":"different-graph"}', encoding="utf-8")
    with pytest.raises(ValueError, match="graph/data/model boundary"):
        OODCampaign(args(second, "third"))


def test_selected_model_reaches_all_online_generation_stages() -> None:
    environment = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    environment["DEGS_MODEL"] = "Qwen3.5-27B-AWQ"
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json; "
                "from degs.benchmark import MODEL; "
                "from degs.source_replay import SOURCE_REPLAY_MODEL; "
                "from degs.validated_repair import REPAIR_SOURCE_MODEL; "
                "print(json.dumps([MODEL, SOURCE_REPLAY_MODEL, REPAIR_SOURCE_MODEL]))"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert json.loads(completed.stdout) == ["Qwen3.5-27B-AWQ"] * 3


def test_replay_wire_identity_matches_bundled_runtime() -> None:
    executor = (ROOT / "src/degs/source_replay_executor.py").read_text()
    runtime = (
        ROOT
        / "vendor/spreadsheetbench_runtime/src/spreadsheet_agent/agents/source_replay_patch_agent.py"
    ).read_text()
    assert '"format": "degs_source_replay_patch_v1"' in executor
    assert '!= "degs_source_replay_patch_v1"' in runtime


def test_replay_uses_the_selected_model_adapter() -> None:
    adapter = Path("/method/src/degs/fresh_train_evaluate.py")
    original = ["/python", "-m", "sb_adapter.evaluate", "--data_path", "/data"]

    adapted = _adapt_evaluator_command(
        original,
        model="Qwen3.5-27B-AWQ",
        python_executable="/python",
        runtime_root=Path("/runtime"),
        method_root=Path("/method"),
        evaluator_adapter=adapter,
    )

    assert adapted[:6] == [
        "/usr/bin/env",
        "PYTHONPATH=/runtime/src:/method/src",
        "/python",
        str(adapter),
        "--expected-model",
        "Qwen3.5-27B-AWQ",
    ]
    assert "--expected-max-tokens" not in adapted
    adapted_9b = _adapt_evaluator_command(
        original,
        model="Qwen3.5-9B-AWQ",
        python_executable="/python",
        runtime_root=Path("/runtime"),
        method_root=Path("/method"),
        evaluator_adapter=adapter,
    )
    assert adapted_9b[:6] == [
        "/usr/bin/env",
        "PYTHONPATH=/runtime/src:/method/src",
        "/python",
        str(adapter),
        "--expected-model",
        "Qwen3.5-9B-AWQ",
    ]
    assert "--expected-max-tokens" not in adapted_9b


def test_public_dataset_contract_pins_both_populations_and_splits() -> None:
    assert fetch_spreadsheetbench.SOURCE_COMMIT == (
        "3d0b52a140f002a512930252b613c49048f7d5ac"
    )
    assert fetch_spreadsheetbench.VERIFIED_RELATIVE.as_posix().endswith(
        "spreadsheetbench_verified_400"
    )
    assert fetch_spreadsheetbench.FULL_RELATIVE.as_posix().endswith(
        "all_data_912_v0.1"
    )
    assert len(fetch_spreadsheetbench.VERIFIED_TRAIN_IDS_SHA256) == 64
    assert len(fetch_spreadsheetbench.VERIFIED_DEVELOPMENT_IDS_SHA256) == 64
    assert len(fetch_spreadsheetbench.FULL_IDS_SHA256) == 64


def test_campaign_receipt_binds_declared_file_outputs(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    logs = tmp_path / "logs"
    batches = tmp_path / "batches"
    output.write_text("{}", encoding="utf-8")
    logs.mkdir()
    batches.mkdir()
    (logs / "task.md").write_text("trace", encoding="utf-8")
    (batches / "batch.json").write_text("{}", encoding="utf-8")
    paths = _declared_output_paths(
        [
            "python",
            "tool.py",
            "--output",
            str(output),
            "--log-dir",
            str(logs),
            "--fixed-batch-output-dir",
            str(batches),
            "--state-db",
            "ignored.db",
        ]
    )
    assert paths == (output, logs, batches)
    rows = _output_identities(paths)
    _validate_output_identities(rows)

    output.write_text('{"changed":true}', encoding="utf-8")
    with pytest.raises(ValueError, match="output changed"):
        _validate_output_identities(rows)

    output.write_text("{}", encoding="utf-8")
    rows = _output_identities(paths)
    (logs / "task.md").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="output changed"):
        _validate_output_identities(rows)

    directory = tmp_path / "bundle"
    directory.mkdir()
    (directory / "manifest.json").write_text("{}", encoding="utf-8")
    directory_rows = _output_identities((directory,))
    (directory / "manifest.json").write_text('{"changed":true}', encoding="utf-8")
    with pytest.raises(ValueError, match="output changed"):
        _validate_output_identities(directory_rows)


def test_full_campaign_uses_one_runtime() -> None:
    options = {
        option
        for action in _parser()._actions
        for option in action.option_strings
    }
    assert not any(option.startswith("--runtime-") for option in options)


def test_full_campaign_worker_defaults_are_configurable() -> None:
    args = _parser().parse_args(
        [
            "--profile", "9b",
            "--run-root", "/run",
            "--trace2skill-checkout", "/data",
            "--generation-base-url", "http://generation/v1",
            "--embedding-base-url", "http://embedding/v1",
        ]
    )
    assert (args.agent_workers, args.producer_workers, args.libreoffice_workers) == (8, 32, 16)
    overridden = _parser().parse_args(
        [
            "--profile", "9b",
            "--run-root", "/run",
            "--trace2skill-checkout", "/data",
            "--generation-base-url", "http://generation/v1",
            "--embedding-base-url", "http://embedding/v1",
            "--agent-workers", "64",
            "--producer-workers", "96",
            "--libreoffice-workers", "24",
        ]
    )
    assert (overridden.agent_workers, overridden.producer_workers, overridden.libreoffice_workers) == (64, 96, 24)


def test_producer_concurrency_reaches_canonical_workers() -> None:
    environment = dict(
        os.environ,
        PYTHONPATH=str(ROOT / "src"),
        DEGS_PRODUCER_WORKERS="47",
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from degs.canonicalize import CANONICAL_LLM_WORKERS; "
                "from degs.eir_canonical import EIR_CANONICAL_VIEW_WORKERS; "
                "print(CANONICAL_LLM_WORKERS, EIR_CANONICAL_VIEW_WORKERS)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.stdout.strip() == "47 47"


def test_service_preflight_requires_each_selected_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = []

    def fake_models(**kwargs):
        observed.append(kwargs["base_url"])
        return (
            ("Qwen3.5-9B-AWQ",)
            if len(observed) == 1
            else ("Qwen3-Embedding-8B",)
        )

    monkeypatch.setattr(preflight_services, "_models", fake_models)
    result = preflight_services.verify_services(
        generation_base_url="http://generation.invalid/v1",
        generation_model="Qwen3.5-9B-AWQ",
        generation_api_key_env="GEN_KEY",
        embedding_base_url="http://embedding.invalid/v1",
        embedding_model="Qwen3-Embedding-8B",
        embedding_api_key_env="EMBED_KEY",
    )

    assert result["status"] == "PASS"
    assert observed == [
        "http://generation.invalid/v1",
        "http://embedding.invalid/v1",
    ]


def test_campaign_summary_aggregates_stage_time_usage_and_metrics(
    tmp_path: Path,
) -> None:
    (tmp_path / "stages").mkdir()
    (tmp_path / "development").mkdir()
    (tmp_path / "campaign_manifest.json").write_text(
        json.dumps(
            {
                "format": "degs_full_reproduction_campaign_v1",
                "profile": "9b",
                "model": "Qwen3.5-9B-AWQ",
            }
        )
    )
    (tmp_path / "stages/01.json").write_text(
        json.dumps({"stage": "one", "status": "COMPLETED", "wall_seconds": 2.5})
    )
    (tmp_path / "producer_usage.jsonl").write_text(
        json.dumps(
            {
                "cache_hit": False,
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 4,
                    "total_tokens": 14,
                },
            }
        )
        + "\n"
        + json.dumps(
            {"cache_hit": True, "usage": {"input_tokens": 3, "output_tokens": 2}}
        )
        + "\n"
    )
    (tmp_path / "development/eval_summary.json").write_text(
        json.dumps({"fixed_denominator": 200, "fully_correct_instances": 88})
    )

    result = summarize_campaign.summarize(tmp_path)

    assert result["stage_count"] == 1
    assert result["usage_totals"] == {
        "cache_hits": 1,
        "input_tokens": 13,
        "output_tokens": 6,
        "records": 2,
        "total_tokens": 19,
    }
    assert result["metrics"]["development"]["fixed_denominator"] == 200
