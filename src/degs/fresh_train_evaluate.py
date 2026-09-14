"""Evaluate a train or replay run with the selected DEGS model identity."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import sb_adapter.evaluate as upstream


MODELS = {"Qwen3.5-9B-AWQ", "Qwen3.5-27B-AWQ"}


def _validate_run_manifest(
    path: str | Path,
    *,
    data_path: str | Path,
    start_idx: int,
    end_idx: int,
    expected_base_url: str,
    expected_workers: int,
    expected_thinking: str,
    expected_max_tokens: int | None,
    expected_model: str,
) -> dict[str, Any]:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_file = Path(data_path) / "dataset.json"
    dataset = json.loads(dataset_file.read_text(encoding="utf-8"))
    expected_ids = [str(item["id"]) for item in dataset[start_idx:end_idx]]
    protocol = {
        key: value
        for key, value in payload.items()
        if key not in {"protocol_sha256", "created_at"}
    }
    protocol_sha = hashlib.sha256(
        json.dumps(
            protocol,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    checks = {
        "format": payload.get("format") == "spreadsheetbench_adapter_run_v1",
        "dataset_sha256": payload.get("dataset_sha256")
        == hashlib.sha256(dataset_file.read_bytes()).hexdigest(),
        "dataset_tree_sha256": payload.get("dataset_tree_sha256")
        == upstream._tree_sha256(Path(data_path)),
        "start_idx": payload.get("start_idx") == start_idx,
        "end_idx": payload.get("end_idx") == end_idx,
        "instance_ids": payload.get("instance_ids") == expected_ids,
        "model": payload.get("model") == expected_model,
        "base_url": payload.get("base_url") == expected_base_url,
        "temperature": payload.get("temperature") == 0.0,
        "max_tokens": payload.get("max_tokens") == expected_max_tokens,
        "thinking": payload.get("thinking") == expected_thinking,
        "max_turns": payload.get("max_turns") == 30,
        "bash_timeout": payload.get("bash_timeout") == 120,
        "bash_sandbox": payload.get("bash_sandbox") == "required",
        "workers": payload.get("workers") == expected_workers,
        "llm_timeout": payload.get("llm_timeout") == 600.0,
        "retry_waits": payload.get("retry_waits") == [5, 10, 30],
        "response_cache_disabled": payload.get("response_cache_enabled") is False,
        "protocol_sha256": payload.get("protocol_sha256") == protocol_sha,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"run manifest does not match train protocol: {failed}")
    return {
        "file_name": manifest_path.name,
        "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "protocol_sha256": protocol_sha,
        "checks": checks,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-model", choices=sorted(MODELS), required=True)
    args, remaining = parser.parse_known_args(argv)
    original = upstream._validate_run_manifest

    def validator(path: str | Path, **kwargs: Any) -> dict[str, Any]:
        return _validate_run_manifest(path, expected_model=args.expected_model, **kwargs)

    upstream._validate_run_manifest = validator
    try:
        sys.argv = ["sb_adapter.evaluate", *remaining]
        upstream.main()
    finally:
        upstream._validate_run_manifest = original
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
