#!/usr/bin/env python3
"""Run one complete, resumable DEGS SpreadsheetBench reproduction campaign."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
FETCH = ROOT / "scripts/fetch_spreadsheetbench.py"
TRAIN_RUNTIME_ROOT = ROOT / "vendor/spreadsheetbench_runtime"
SERVICE_PREFLIGHT = ROOT / "scripts/preflight_services.py"
CAMPAIGN_SUMMARY = ROOT / "scripts/summarize_campaign.py"
MODEL_BY_PROFILE = {
    "9b": "Qwen3.5-9B-AWQ",
    "27b": "Qwen3.5-27B-AWQ",
}


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


_OUTPUT_OPTIONS = frozenset(
    {
        "--output",
        "--output-dir",
        "--log-dir",
        "--results-file",
        "--results_file",
        "--manifest-path",
        "--retrieval-manifest-path",
        "--outcomes-output",
        "--section-graphs-output",
        "--fixed-batch-output-dir",
        "--audit-output",
        "--recalc_dir",
    }
)
def _file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _tree_sha256(root: Path) -> str:
    rows = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": _file_sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]
    return hashlib.sha256(_canonical(rows)).hexdigest()


def _declared_output_paths(command: Sequence[str]) -> tuple[Path, ...]:
    paths: list[Path] = []
    for index, value in enumerate(command[:-1]):
        if value in _OUTPUT_OPTIONS:
            paths.append(Path(command[index + 1]).expanduser().absolute())
    return tuple(dict.fromkeys(paths))


def _output_identities(paths: Sequence[Path]) -> list[dict[str, Any]]:
    identities: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"declared stage output is absent: {path}")
        identities.append(
            {
                "path": str(path),
                "kind": "directory" if path.is_dir() else "file",
                "sha256": _tree_sha256(path) if path.is_dir() else _file_sha256(path),
            }
        )
    return identities


def _validate_output_identities(rows: Any) -> None:
    if type(rows) is not list:
        raise ValueError("completed stage output receipt differs")
    for row in rows:
        if type(row) is not dict or set(row) != {"path", "kind", "sha256"}:
            raise ValueError("completed stage output receipt differs")
        path = Path(str(row["path"]))
        if not path.exists():
            raise FileNotFoundError(f"completed stage output is absent: {path}")
        if row["kind"] == "directory":
            if not path.is_dir() or row["sha256"] != _tree_sha256(path):
                raise ValueError(f"completed stage output type differs: {path}")
        elif row["kind"] == "file":
            if not path.is_file() or row["sha256"] != _file_sha256(path):
                raise ValueError(f"completed stage output hash differs: {path}")
        else:
            raise ValueError(f"completed stage output type differs: {path}")


def _runtime_identity(
    root: Path, *, expected_version: str, expected_model: str
) -> Mapping[str, Any]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root.resolve() / "src")
    environment["DEGS_MODEL"] = expected_model
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json; import degs; "
                "from degs.validated_repair import REPAIR_SOURCE_MODEL; "
                "print(json.dumps({'version': degs.__version__, "
                "'producer_model': REPAIR_SOURCE_MODEL}, sort_keys=True))"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    declared = json.loads(probe.stdout)
    if declared != {"producer_model": expected_model, "version": expected_version}:
        raise ValueError(
            f"method-runtime profile differs for {expected_version}: {declared}"
        )
    return {
        "root": str(root.resolve()),
        **declared,
    }


class Campaign:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.root = args.run_root.expanduser().resolve()
        if self.root.exists() and not (self.root / "campaign_manifest.json").is_file():
            if not self.root.is_dir() or any(self.root.iterdir()):
                raise FileExistsError(
                    "run root must be fresh or contain its campaign manifest"
                )
        self.root.mkdir(parents=True, exist_ok=True)
        self.model = MODEL_BY_PROFILE[args.profile]
        self.verified = (
            args.trace2skill_checkout.expanduser().resolve()
            / "data/spreadsheetbench_verified/spreadsheetbench_verified_400"
        )
        self.full = (
            args.trace2skill_checkout.expanduser().resolve()
            / "data/all_data_912_v0.1"
        )
        subprocess.run(
            [sys.executable, str(FETCH), "--checkout", str(args.trace2skill_checkout), "--verify-only"],
            check=True,
        )
        identity = {
            "format": "degs_full_reproduction_campaign_v1",
            "claim_scope": "fresh deployment reproduction; not historical artifact replay",
            "profile": args.profile,
            "model": self.model,
            "generation_base_url": args.generation_base_url,
            "embedding_base_url": args.embedding_base_url,
            "trace2skill_commit": "3d0b52a140f002a512930252b613c49048f7d5ac",
            "verified_train": [0, 200],
            "verified_development": [200, 400],
            "development_denominator": 200,
            "soft_hard_tasks": 912,
            "soft_hard_cases": 2529,
            "train_batch_size": 8,
            "train_batch_count": 25,
            "agent_workers": 8,
            "producer_workers": 16,
            "libreoffice_workers": 16,
            "agent_max_turns": 30,
            "agent_completion_tokens": 32000,
            "server_context_tokens": 100000,
            "temperature": 0,
            "thinking": False,
            "explicit_seed": None,
            "method_runtime": _runtime_identity(
                ROOT, expected_version="0.78.0", expected_model=self.model
            ),
        }
        manifest_path = self.root / "campaign_manifest.json"
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text())
            if existing != identity:
                raise ValueError("campaign identity changed; use a fresh run root")
        else:
            _write_json(manifest_path, identity)
        self.identity = hashlib.sha256(_canonical(identity)).hexdigest()
        self.base_env = dict(os.environ)
        self.base_env["DEGS_MODEL"] = self.model

    def _method_command(
        self, module: str, arguments: Sequence[str | Path]
    ) -> list[str]:
        return [
            sys.executable,
            "-m",
            module,
            *map(str, arguments),
        ]

    def stage(
        self,
        name: str,
        command: Sequence[str | Path],
        *,
        env: Mapping[str, str] | None = None,
        always_run: bool = False,
    ) -> None:
        command_text = list(map(str, command))
        fingerprint = hashlib.sha256(
            _canonical({"campaign": self.identity, "name": name, "command": command_text})
        ).hexdigest()
        receipt = self.root / "stages" / f"{name}.json"
        if receipt.exists():
            completed = json.loads(receipt.read_text())
            if (
                completed.get("fingerprint") != fingerprint
                or completed.get("status") != "COMPLETED"
                or completed.get("returncode") != 0
            ):
                raise ValueError(f"completed stage identity differs: {name}")
            if not always_run:
                _validate_output_identities(completed.get("outputs"))
                log_path = self.root / "logs" / f"{name}.log"
                if (
                    not log_path.is_file()
                    or completed.get("log_sha256") != _file_sha256(log_path)
                ):
                    raise ValueError(f"completed stage log differs: {name}")
                print(f"SKIP {name}", flush=True)
                return
        if self.args.dry_run:
            self.plan.append({"stage": name, "command": command_text})
            return
        logs = self.root / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        started = datetime.now(timezone.utc).isoformat()
        start = time.monotonic()
        _write_json(
            self.root / "status.json",
            {"stage": name, "status": "RUNNING", "started_at": started},
        )
        print(f"START {name}", flush=True)
        with (logs / f"{name}.log").open("a", encoding="utf-8") as stream:
            result = subprocess.run(
                command_text,
                cwd=ROOT,
                env=dict(env or self.base_env),
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        ended = datetime.now(timezone.utc).isoformat()
        payload = {
            "stage": name,
            "status": "COMPLETED" if result.returncode == 0 else "FAILED",
            "returncode": result.returncode,
            "started_at": started,
            "ended_at": ended,
            "wall_seconds": time.monotonic() - start,
            "command": command_text,
            "fingerprint": fingerprint,
        }
        _write_json(self.root / "status.json", payload)
        if result.returncode:
            raise RuntimeError(f"stage {name} failed; inspect {logs / (name + '.log')}")
        declared_outputs = list(_declared_output_paths(command_text))
        if name == "dynamic_train":
            run_dir = Path(command_text[command_text.index("--run-dir") + 1])
            declared_outputs.extend(
                [
                    run_dir / "state/eir_state.sqlite3",
                    run_dir / "batches/batch_24/manifest.json",
                    run_dir / "batches/batch_24/experience_graph.json",
                ]
            )
        elif name in {"development_agent", "soft_hard_agent"}:
            run_dir = Path(command_text[command_text.index("--run-dir") + 1])
            declared_outputs.extend(
                [run_dir / "results.jsonl", run_dir / "outputs"]
            )
        elif name in {"development_evaluate", "soft_hard_evaluate"}:
            run_dir = Path(command_text[command_text.index("--run-dir") + 1])
            declared_outputs.extend(
                [run_dir / "eval_summary.json", run_dir / "eval_details.json"]
            )
        payload["outputs"] = _output_identities(tuple(dict.fromkeys(declared_outputs)))
        payload["log_sha256"] = _file_sha256(logs / f"{name}.log")
        _write_json(receipt, payload)
        print(f"DONE {name}", flush=True)

    def run(self) -> None:
        self.plan: list[dict[str, Any]] = []
        a, root = self.args, self.root
        self.stage(
            "service_preflight",
            [
                sys.executable,
                SERVICE_PREFLIGHT,
                "--generation-base-url",
                a.generation_base_url,
                "--generation-model",
                self.model,
                "--embedding-base-url",
                a.embedding_base_url,
                "--embedding-model",
                "Qwen3-Embedding-8B",
            ],
            always_run=True,
        )
        final_env = dict(
            self.base_env,
            PYTHONPATH=str(ROOT / "src"),
            REACT_AGENT_USAGE_LOG=str(root / "producer_usage.jsonl"),
            REACT_AGENT_RUNTIME_EVENT_LOG=str(root / "producer_runtime_events.jsonl"),
        )
        dynamic = root / "dynamic_train"
        self.stage(
            "dynamic_train",
            [
                sys.executable, "-m", "degs.dynamic_train",
                "--dataset-path", self.verified,
                "--run-dir", dynamic,
                "--runtime-root", TRAIN_RUNTIME_ROOT,
                "--generation-base-url", a.generation_base_url,
                "--embedding-base-url", a.embedding_base_url,
                "--model", self.model,
            ],
            env=final_env,
        )
        state = dynamic / "state/eir_state.sqlite3"
        snapshot = dynamic / "batches/batch_24/manifest.json"
        self.stage(
            "graph_audit",
            [sys.executable, "-m", "degs.eir_graph_quality", "--state-db", state, "--output", root / "graph_quality.json"],
            env=final_env,
        )
        vrf_bundle = root / "development_bundle"
        self.stage(
            "development_retrieval",
            self._method_command(
                "degs.eir_bundle",
                [
                    "--dataset-path", self.verified,
                    "--state-db", state,
                    "--output-dir", vrf_bundle,
                    "--generation-base-url", a.generation_base_url,
                    "--embedding-base-url", a.embedding_base_url,
                    "--model", self.model,
                ],
            ),
            env=final_env,
        )
        development_agent_args: list[str | Path] = [
            "--data-path", self.verified,
            "--snapshot-manifest-path", snapshot,
            "--state-db", state,
            "--bundle-dir", vrf_bundle,
            "--run-dir", root / "development",
            "--base-url", a.generation_base_url,
        ]
        if (root / "development/outputs/run_manifest.json").is_file():
            development_agent_args.append("--resume")
        self.stage(
            "development_agent",
            self._method_command(
                "degs.benchmark",
                development_agent_args,
            ),
            env=final_env,
        )
        self.stage(
            "development_evaluate",
            self._method_command(
                "degs.evaluate",
                [
                    "--data-path", self.verified,
                    "--run-dir", root / "development",
                    "--base-url", a.generation_base_url,
                    "--model", self.model,
                ],
            ),
            env=final_env,
        )
        soft = root / "soft_hard"
        prepared = soft / "prepared_dataset"
        population = soft / "population_manifest.json"
        input_population = soft / "input_manifest.json"
        self.stage("soft_hard_population", self._method_command("degs.soft_hard_dataset", ["--full-data-path", self.full, "--verified-data-path", self.verified, "--output-dir", prepared, "--manifest-path", population, "--retrieval-manifest-path", input_population]), env=final_env)
        soft_common = ["--source-dataset-path", self.verified / "dataset.json", "--prepared-data-path", prepared, "--retrieval-manifest-path", input_population, "--snapshot-manifest-path", snapshot, "--state-db", state]
        soft_bundle = soft / "bundle"
        self.stage(
            "soft_hard_retrieval",
            self._method_command(
                "degs.soft_hard_bundle",
                [
                    "build",
                    *soft_common,
                    "--output-dir",
                    soft_bundle,
                    "--llm-base-url",
                    a.generation_base_url,
                    "--embedding-base-url",
                    a.embedding_base_url,
                    "--model",
                    self.model,
                ],
            ),
            env=final_env,
        )
        self.stage("soft_hard_verify", self._method_command("degs.soft_hard_bundle", ["verify", *soft_common, "--output-dir", soft_bundle]), env=final_env)
        soft_agent_args: list[str | Path] = [
            "--source-dataset-path", self.verified / "dataset.json",
            "--prepared-data-path", prepared,
            "--input-manifest-path", input_population,
            "--snapshot-manifest-path", snapshot,
            "--state-db", state,
            "--bundle-dir", soft_bundle,
            "--run-dir", soft / "agent_run",
            "--base-url", a.generation_base_url,
        ]
        if (soft / "agent_run/outputs/run_manifest.json").is_file():
            soft_agent_args.append("--resume")
        self.stage(
            "soft_hard_agent",
            self._method_command("degs.soft_hard_benchmark", soft_agent_args),
            env=final_env,
        )
        self.stage("soft_hard_evaluate", self._method_command("degs.soft_hard_evaluate", ["--prepared-data-path", prepared, "--population-manifest-path", population, "--input-manifest-path", input_population, "--run-dir", soft / "agent_run", "--workers", "16"]), env=final_env)
        self.stage(
            "campaign_summary",
            [sys.executable, CAMPAIGN_SUMMARY, "--run-root", root],
        )
        if self.args.dry_run:
            _write_json(self.root / "campaign_plan.json", {"campaign": self.identity, "stages": self.plan})
            print(json.dumps({"status": "DRY_RUN", "stage_count": len(self.plan), "plan": str(self.root / "campaign_plan.json")}, indent=2))
        else:
            # Refresh after the summary stage receipt exists so its own timing is
            # included in the final aggregate.
            subprocess.run(
                [sys.executable, str(CAMPAIGN_SUMMARY), "--run-root", str(root)],
                check=True,
                stdout=subprocess.DEVNULL,
            )
            _write_json(self.root / "status.json", {"status": "COMPLETED", "ended_at": datetime.now(timezone.utc).isoformat()})


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(MODEL_BY_PROFILE), required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--trace2skill-checkout", type=Path, required=True)
    parser.add_argument("--generation-base-url", required=True)
    parser.add_argument("--embedding-base-url", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    run_root = args.run_root.expanduser().absolute()
    run_root.parent.mkdir(parents=True, exist_ok=True)
    lock_path = run_root.parent / f".{run_root.name}.campaign.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"campaign is already running: {run_root}") from exc
        Campaign(args).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
