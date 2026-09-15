#!/usr/bin/env python3
"""Run one isolated DEGS 0.78.0 WikiTQ/HiTab transfer campaign."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
MODEL_BY_PROFILE = {
    "9b": "Qwen3.5-9B-AWQ",
    "27b": "Qwen3.5-27B-AWQ",
}


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


class Campaign:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.root = args.run_root.expanduser().absolute()
        manifest_path = self.root / "campaign_manifest.json"
        if self.root.exists() and not manifest_path.is_file():
            if not self.root.is_dir() or any(self.root.iterdir()):
                raise FileExistsError(
                    "OOD run root must be fresh or contain its campaign manifest"
                )
        self.root.mkdir(parents=True, exist_ok=True)
        self.env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
        self.model = MODEL_BY_PROFILE[args.profile]
        self.env["DEGS_MODEL"] = self.model
        body = {
            "format": "degs_tableqa_ood_campaign_v1",
            "profile": args.profile,
            "model": self.model,
            "generation_base_url": args.generation_base_url.rstrip("/"),
            "embedding_base_url": args.embedding_base_url.rstrip("/"),
            "source_dataset_path": str(args.source_dataset_path.expanduser().resolve()),
            "snapshot_manifest_path": str(
                args.snapshot_manifest_path.expanduser().resolve()
            ),
            "state_db": str(args.state_db.expanduser().resolve()),
            "wikitq_source_repo": str(args.wikitq_source_repo.expanduser().resolve()),
            "hitab_source_repo": str(args.hitab_source_repo.expanduser().resolve()),
        }
        self.campaign_sha256 = hashlib.sha256(
            json.dumps(
                body,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        manifest = {**body, "self_sha256": self.campaign_sha256}
        if manifest_path.is_file():
            if json.loads(manifest_path.read_text()) != manifest:
                raise ValueError("OOD campaign identity differs")
        else:
            _write_json(manifest_path, manifest)

    def stage(self, name: str, command: Sequence[str], *, done: Path) -> None:
        receipt = self.root / "stages" / f"{name}.json"
        command = list(command)
        if receipt.is_file() and done.exists():
            prior = json.loads(receipt.read_text())
            if (
                prior.get("returncode") != 0
                or prior.get("command") != command
                or prior.get("campaign_sha256") != self.campaign_sha256
            ):
                raise ValueError(f"completed stage identity differs: {name}")
            print(f"SKIP {name}", flush=True)
            return
        log = self.root / "logs" / f"{name}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        started = datetime.now(timezone.utc).isoformat()
        start = time.monotonic()
        print(f"START {name}", flush=True)
        with log.open("a", encoding="utf-8") as stream:
            completed = subprocess.run(
                command, cwd=ROOT, env=self.env, stdout=stream, stderr=subprocess.STDOUT
            )
        row = {
            "stage": name,
            "command": command,
            "returncode": completed.returncode,
            "campaign_sha256": self.campaign_sha256,
            "started_at": started,
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "wall_seconds": time.monotonic() - start,
        }
        _write_json(receipt, row)
        if completed.returncode:
            raise RuntimeError(f"{name} failed; inspect {log}")
        if not done.exists():
            raise RuntimeError(f"{name} did not create {done}")
        print(f"DONE {name}", flush=True)

    def run_dataset(self, dataset: str, source_repo: Path) -> None:
        root = self.root / dataset
        prepared = root / "prepared"
        bundle = root / "bundle"
        agent_run = root / "agent_run"
        evaluation = root / "evaluation"
        self.stage(
            f"{dataset}_prepare",
            [
                sys.executable,
                "-m",
                "degs.ood_dataset",
                "prepare",
                "--dataset",
                dataset,
                "--source-repo",
                str(source_repo),
                "--output-dir",
                str(prepared),
            ],
            done=prepared / "retrieval_manifest.json",
        )
        self.stage(
            f"{dataset}_retrieval",
            [
                sys.executable,
                "-m",
                "degs.ood_bundle",
                "build",
                "--source-dataset-path",
                str(self.args.source_dataset_path),
                "--prepared-data-path",
                str(prepared),
                "--snapshot-manifest-path",
                str(self.args.snapshot_manifest_path),
                "--state-db",
                str(self.args.state_db),
                "--output-dir",
                str(bundle),
                "--llm-base-url",
                self.args.generation_base_url,
                "--embedding-base-url",
                self.args.embedding_base_url,
                "--model",
                self.model,
            ],
            done=bundle / "bundle_manifest.json",
        )
        agent_command = [
            sys.executable,
            "-m",
            "degs.ood_benchmark",
            "--source-dataset-path",
            str(self.args.source_dataset_path),
            "--prepared-data-path",
            str(prepared),
            "--snapshot-manifest-path",
            str(self.args.snapshot_manifest_path),
            "--state-db",
            str(self.args.state_db),
            "--bundle-dir",
            str(bundle),
            "--run-dir",
            str(agent_run),
            "--base-url",
            self.args.generation_base_url,
        ]
        if agent_run.exists():
            agent_command.append("--resume")
        self.stage(
            f"{dataset}_agent",
            agent_command,
            done=agent_run / "results.json",
        )
        self.stage(
            f"{dataset}_evaluate",
            [
                sys.executable,
                "-m",
                "degs.ood_evaluate",
                "--prepared-data-path",
                str(prepared),
                "--run-dir",
                str(agent_run),
                "--source-repo",
                str(source_repo),
                "--output-dir",
                str(evaluation),
                "--python2",
                self.args.python2,
            ],
            done=evaluation / "eval_summary.json",
        )

    def run(self) -> None:
        self.run_dataset("wikitq", self.args.wikitq_source_repo)
        self.run_dataset("hitab", self.args.hitab_source_repo)
        _write_json(
            self.root / "completed.json",
            {
                "method": "DEGS 0.78.0 Evidence-Bounded EIR Dynamic",
                "profile": self.args.profile,
                "model": self.model,
                "datasets": ["wikitq", "hitab"],
                "completed_at": datetime.now(timezone.utc).isoformat(),
            },
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("9b", "27b"), required=True)
    parser.add_argument("--source-dataset-path", type=Path, required=True)
    parser.add_argument("--snapshot-manifest-path", type=Path, required=True)
    parser.add_argument("--state-db", type=Path, required=True)
    parser.add_argument("--wikitq-source-repo", type=Path, required=True)
    parser.add_argument("--hitab-source-repo", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--generation-base-url", required=True)
    parser.add_argument("--embedding-base-url", required=True)
    parser.add_argument("--python2", default="/usr/bin/python2")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    Campaign(_parser().parse_args(argv)).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
