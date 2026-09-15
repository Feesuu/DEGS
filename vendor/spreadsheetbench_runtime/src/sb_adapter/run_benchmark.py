from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from react_agent import OpenAIClient
from spreadsheet_agent import (
    CLIOnlyAgent,
    SourceReplayPatchAgent,
    SpreadsheetBenchRunner,
)

from .experience import EmptyExperienceProvider, FileExperienceProvider
from .transport import validate_service_url


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = ROOT / "data" / "spreadsheetbench_verified_400"
SYSTEM_PROMPT_DIR = ROOT / "src" / "spreadsheet_agent" / "system_prompt"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    files = []
    for item in root.rglob("*"):
        if item.is_dir():
            continue
        if not item.is_file():
            raise ValueError(f"hashed tree contains a non-regular entry: {item}")
        if "__pycache__" not in item.parts and item.suffix != ".pyc":
            files.append(item)
    for path in sorted(files):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _dependency_versions() -> dict[str, str]:
    versions = {}
    for distribution in ("openai", "openpyxl", "tqdm", "transformers"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "missing"
    return versions


def _parse_waits(value: str) -> tuple[int, ...]:
    waits = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if any(item < 0 for item in waits):
        raise argparse.ArgumentTypeError("retry waits must be non-negative")
    return waits


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the algorithm-independent SpreadsheetBench adapter."
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--working-dir", type=Path)
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--results-file", type=Path)
    parser.add_argument("--start-idx", type=int, default=200)
    parser.add_argument("--end-idx", type=int, default=400)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--bash-timeout", type=int, default=120)
    parser.add_argument("--model", default=os.getenv("MODEL", "Qwen3.5-9B-AWQ"))
    parser.add_argument(
        "--base-url",
        default=os.getenv(
            "BASE_URL",
            os.getenv("OPENAI_BASE_URL", "https://generation.example.invalid/v1"),
        ),
    )
    parser.add_argument("--api-key-env", default="DEGS_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--thinking", choices=("true", "false"), default="false")
    parser.add_argument("--llm-timeout", type=float, default=1200.0)
    parser.add_argument("--retry-waits", type=_parse_waits, default=(5, 10, 30))
    parser.add_argument("--usage-log", type=Path)
    parser.add_argument("--runtime-event-log", type=Path)
    parser.add_argument("--runtime-timeout-retries", type=_non_negative_int, default=0)
    parser.add_argument("--stagnation-repeat-limit", type=_non_negative_int, default=0)
    parser.add_argument(
        "--agent",
        choices=("cli_only", "source_replay_patch"),
        default="cli_only",
    )
    parser.add_argument("--experience-file", type=Path)
    parser.add_argument("--require-experience-for-all", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _serialize_result(
    result,
    output_dir: Path,
    *,
    spreadsheet_path: str,
) -> dict[str, Any]:
    payload = asdict(result)
    payload["id"] = str(result.id)
    for test_case in payload.get("test_cases", []):
        test_case["output_path"] = str(
            output_dir / spreadsheet_path / test_case["output_file"]
        )
    return payload


def _load_existing_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return rows
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        instance_id = str(row.get("id", ""))
        if not instance_id or instance_id in rows:
            raise ValueError(f"invalid result ledger row {line_number}: {instance_id!r}")
        rows[instance_id] = row
    return rows


def _resume_identity(manifest: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(
        manifest.get(key)
        for key in (
            "format",
            "dataset_sha256",
            "dataset_tree_sha256",
            "start_idx",
            "end_idx",
            "instance_ids",
            "model",
        )
    )


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def _build_provider(args: argparse.Namespace):
    if args.agent == "cli_only":
        if args.experience_file:
            raise ValueError("--experience-file requires --agent source_replay_patch")
        return EmptyExperienceProvider()
    if not args.experience_file:
        raise ValueError("source_replay_patch requires --experience-file")
    return FileExperienceProvider(args.experience_file)


def _build_client(args: argparse.Namespace, *, api_key: str):
    generation_config: dict[str, Any] = {
        "temperature": float(args.temperature),
        "extra_body": {
            "chat_template_kwargs": {
                "enable_thinking": args.thinking == "true",
            }
        },
    }
    if args.max_tokens is not None:
        generation_config["max_tokens"] = int(args.max_tokens)
    return OpenAIClient(
        model=args.model,
        api_key=api_key,
        base_url=args.base_url,
        generation_config=generation_config,
        retry_times=tuple(args.retry_waits),
        runtime_timeout_retries=int(args.runtime_timeout_retries),
        timeout=float(args.llm_timeout),
        trust_env=False,
    )


def _build_agent(args: argparse.Namespace, provider, *, api_key: str):
    kwargs = {
        "client": _build_client(args, api_key=api_key),
        "max_turns": int(args.max_turns),
        "verbose": bool(args.verbose),
        "timeout": int(args.bash_timeout),
        "sandbox_mode": "required",
        "log_dir": str(args.log_dir) if args.log_dir else None,
        "stagnation_repeat_limit": int(args.stagnation_repeat_limit),
        "libreoffice_output_feedback": False,
        "structured_process_contract": False,
    }
    if args.agent == "source_replay_patch":
        return SourceReplayPatchAgent(
            **kwargs,
            patch_provider=provider,
        )
    return CLIOnlyAgent(**kwargs)


def _system_prompt_templates(
    args: argparse.Namespace,
    provider,
    *,
    instance_ids: list[str],
) -> list[dict[str, str]]:
    name = (
        "cli_only_full_system_v1.txt"
        if args.agent == "cli_only"
        else "source_replay_patch_full_system_v1.txt"
    )
    return [{"file_name": name, "sha256": _sha256(SYSTEM_PROMPT_DIR / name)}]


def _run_manifest(args: argparse.Namespace, provider, *, instance_ids: list[str]):
    dataset_file = args.data_path / "dataset.json"
    protocol = {
        "format": "spreadsheetbench_adapter_run_v1",
        "dataset_name": args.data_path.name,
        "dataset_sha256": _sha256(dataset_file),
        "dataset_tree_sha256": _tree_sha256(args.data_path),
        "start_idx": int(args.start_idx),
        "end_idx": int(args.end_idx),
        "instance_ids": instance_ids,
        "agent": args.agent,
        "experience_provider": dict(provider.identity()),
        "system_prompt_templates": _system_prompt_templates(
            args,
            provider,
            instance_ids=instance_ids,
        ),
        "model": args.model,
        "base_url": args.base_url,
        "temperature": float(args.temperature),
        "max_tokens": args.max_tokens,
        "thinking": args.thinking,
        "max_turns": int(args.max_turns),
        "bash_timeout": int(args.bash_timeout),
        "bash_sandbox": "required",
        "workers": int(args.workers),
        "llm_timeout": float(args.llm_timeout),
        "retry_waits": list(args.retry_waits),
        "runtime_timeout_retries": int(args.runtime_timeout_retries),
        "stagnation_repeat_limit": int(args.stagnation_repeat_limit),
        "runtime_event_log_enabled": args.runtime_event_log is not None,
        "runtime_event_log": (
            str(args.runtime_event_log) if args.runtime_event_log is not None else None
        ),
        "response_cache_enabled": False,
        "python_executable": Path(sys.executable).name,
        "python_version": platform.python_version(),
        "dependency_versions": _dependency_versions(),
    }
    fingerprint_payload = json.dumps(
        protocol, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return {
        **protocol,
        "protocol_sha256": hashlib.sha256(
            fingerprint_payload.encode("utf-8")
        ).hexdigest(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.start_idx < 0 or args.end_idx <= args.start_idx:
        raise ValueError("require 0 <= start_idx < end_idx")
    if args.workers < 1:
        raise ValueError("workers must be positive")
    validate_service_url(args.base_url)
    args.data_path = args.data_path.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.results_file = (
        args.results_file.expanduser().resolve()
        if args.results_file
        else args.output_dir / "results.json"
    )
    args.log_dir = (
        args.log_dir.expanduser().resolve()
        if args.log_dir
        else args.output_dir / "logs"
    )
    if args.usage_log:
        args.usage_log = args.usage_log.expanduser().resolve()
        if not args.dry_run:
            args.usage_log.parent.mkdir(parents=True, exist_ok=True)
            os.environ["REACT_AGENT_USAGE_LOG"] = str(args.usage_log)
        else:
            os.environ.pop("REACT_AGENT_USAGE_LOG", None)
    if args.runtime_event_log:
        args.runtime_event_log = args.runtime_event_log.expanduser().resolve()
        if not args.dry_run:
            args.runtime_event_log.parent.mkdir(parents=True, exist_ok=True)
            if args.runtime_event_log.exists() and not args.runtime_event_log.is_file():
                raise ValueError("runtime event log must be a regular file")
            args.runtime_event_log.touch(exist_ok=True)
            os.environ["REACT_AGENT_RUNTIME_EVENT_LOG"] = str(args.runtime_event_log)
        else:
            os.environ.pop("REACT_AGENT_RUNTIME_EVENT_LOG", None)
    else:
        os.environ.pop("REACT_AGENT_RUNTIME_EVENT_LOG", None)

    provider = _build_provider(args)
    inspection_agent = CLIOnlyAgent(
        client=object(), max_turns=1, verbose=False
    )
    inspection_runner = SpreadsheetBenchRunner(
        agent=inspection_agent,
        data_path=str(args.data_path),
        output_dir=str(args.output_dir),
    )
    all_instances = inspection_runner.load_data()
    if args.end_idx > len(all_instances):
        raise ValueError(
            f"end_idx {args.end_idx} exceeds dataset size {len(all_instances)}"
        )
    selected = all_instances[args.start_idx : args.end_idx]
    instance_ids = [str(instance.id) for instance in selected]
    if len(instance_ids) != len(set(instance_ids)):
        raise ValueError("selected dataset slice contains duplicate instance IDs")
    if args.require_experience_for_all:
        missing = [
            instance_id
            for instance_id in instance_ids
            if not provider.for_instance(instance_id).experience
        ]
        if missing:
            raise ValueError(
                f"experience is missing for {len(missing)} selected tasks: {missing[:8]}"
            )

    manifest = _run_manifest(args, provider, instance_ids=instance_ids)
    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return {"manifest": manifest, "results": []}
    os.environ["SB_RUN_PROTOCOL_SHA256"] = manifest["protocol_sha256"]

    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"missing API key: set {args.api_key_env}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "run_manifest.json"
    ledger_path = args.results_file.with_suffix(".jsonl")
    existing_rows: dict[str, dict[str, Any]] = {}
    reuse_existing = False
    if args.resume:
        if not manifest_path.is_file():
            raise FileNotFoundError("resume requires an existing run_manifest.json")
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if _resume_identity(prior) != _resume_identity(manifest):
            raise ValueError("resume dataset/model boundary differs")
        reuse_existing = prior.get("protocol_sha256") == manifest["protocol_sha256"]
        existing_rows = _load_existing_rows(ledger_path) if reuse_existing else {}
        unexpected_ids = sorted(set(existing_rows).difference(instance_ids))
        if unexpected_ids:
            raise ValueError(
                "resume ledger contains tasks outside the selected dataset slice: "
                f"{unexpected_ids[:8]}"
            )
        _write_json_atomic(manifest_path, manifest)
    elif ledger_path.exists() or manifest_path.exists():
        raise FileExistsError(
            "run artifacts already exist; use a new output directory or --resume"
        )
    else:
        _write_json_atomic(manifest_path, manifest)

    pending = [item for item in selected if str(item.id) not in existing_rows]
    if args.working_dir:
        working_dir = args.working_dir.expanduser().resolve()
    else:
        working_dir = Path(tempfile.mkdtemp(prefix="spreadsheetbench_adapter_"))
    working_dir.mkdir(parents=True, exist_ok=True)
    thread_state = threading.local()

    def process(instance):
        if not hasattr(thread_state, "runner"):
            agent = _build_agent(args, provider, api_key=api_key)
            thread_state.runner = SpreadsheetBenchRunner(
                agent=agent,
                data_path=str(args.data_path),
                output_dir=str(args.output_dir),
                working_dir=str(working_dir),
                workbook_structure_preflight=False,
            )
        return thread_state.runner.run_instance(instance)

    new_rows: dict[str, dict[str, Any]] = {}
    started_at = datetime.now(timezone.utc)
    with ThreadPoolExecutor(max_workers=min(args.workers, max(1, len(pending)))) as pool:
        futures = {pool.submit(process, instance): instance for instance in pending}
        with ledger_path.open("a" if reuse_existing else "w", encoding="utf-8") as ledger:
            for future in as_completed(futures):
                instance = futures[future]
                try:
                    result = future.result()
                    row = _serialize_result(
                        result,
                        args.output_dir,
                        spreadsheet_path=str(instance.spreadsheet_path),
                    )
                except Exception as exc:
                    row = {
                        "id": str(instance.id),
                        "instruction": instance.instruction,
                        "success": False,
                        "test_cases": [],
                        "error": f"worker_internal_error:{type(exc).__name__}:{exc}",
                        "failure_type": "runtime_invalid_exception",
                    }
                new_rows[str(instance.id)] = row
                ledger.write(json.dumps(row, ensure_ascii=False) + "\n")
                ledger.flush()
                os.fsync(ledger.fileno())

    combined = {**existing_rows, **new_rows}
    ordered_rows = [combined[instance_id] for instance_id in instance_ids if instance_id in combined]
    payload = {
        "format": "spreadsheetbench_adapter_results_v1",
        "run_manifest": str(manifest_path),
        "protocol_sha256": manifest["protocol_sha256"],
        "started_at": started_at.isoformat(),
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "total_instances": len(instance_ids),
        "completed_instances": len(ordered_rows),
        "agent_completed_instances": sum(bool(row.get("success")) for row in ordered_rows),
        "results": ordered_rows,
    }
    _write_json_atomic(args.results_file, payload)
    print(json.dumps({key: payload[key] for key in (
        "total_instances", "completed_instances", "agent_completed_instances"
    )}, indent=2))
    return payload


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
