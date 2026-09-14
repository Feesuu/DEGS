#!/usr/bin/env python3
"""Aggregate stage timing, LLM usage, and final metrics for one DEGS campaign."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError(f"expected JSON object: {path}")
    return value


def _token_value(usage: Mapping[str, Any], *names: str) -> int:
    for name in names:
        value = usage.get(name)
        if type(value) is int and value >= 0:
            return value
    return 0


def summarize(run_root: Path) -> dict[str, Any]:
    root = run_root.expanduser().resolve()
    manifest = _read_object(root / "campaign_manifest.json")
    stages = []
    for path in sorted((root / "stages").glob("*.json")):
        row = _read_object(path)
        stages.append(
            {
                "stage": row.get("stage"),
                "status": row.get("status"),
                "wall_seconds": row.get("wall_seconds"),
            }
        )

    usage_files = sorted(
        {
            path.resolve()
            for path in root.rglob("*.jsonl")
            if "usage" in path.name.casefold()
        }
    )
    totals = Counter()
    usage_by_file = []
    for path in usage_files:
        local = Counter()
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid usage JSONL: {path}:{line_number}") from exc
            usage = row.get("usage") if type(row) is dict else None
            if type(usage) is not dict:
                usage = {}
            input_tokens = _token_value(usage, "prompt_tokens", "input_tokens")
            output_tokens = _token_value(
                usage, "completion_tokens", "output_tokens"
            )
            total_tokens = _token_value(usage, "total_tokens")
            local["records"] += 1
            local["input_tokens"] += input_tokens
            local["output_tokens"] += output_tokens
            local["total_tokens"] += total_tokens or input_tokens + output_tokens
            if bool(row.get("cache_hit")):
                local["cache_hits"] += 1
        totals.update(local)
        usage_by_file.append(
            {"path": str(path.relative_to(root)), **dict(sorted(local.items()))}
        )

    metrics = {}
    for label, path in (
        ("development", root / "development/eval_summary.json"),
        ("soft_hard", root / "soft_hard/agent_run/eval_summary.json"),
    ):
        if path.is_file():
            metrics[label] = _read_object(path)
    return {
        "format": "degs_campaign_summary_v1",
        "campaign_format": manifest.get("format"),
        "profile": manifest.get("profile"),
        "model": manifest.get("model"),
        "stages": stages,
        "stage_count": len(stages),
        "usage_totals": dict(sorted(totals.items())),
        "usage_files": usage_by_file,
        "metrics": metrics,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args(argv)
    result = summarize(args.run_root)
    output = args.run_root.expanduser().resolve() / "campaign_summary.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
