#!/usr/bin/env python3
"""Exercise the real Agent Bash tool and record its Python/XLSX runtime."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import sys
import tempfile
from typing import Any, Callable, Sequence


PROBE_CODE = """
import json
import sys
import numpy
import openpyxl

path = "runtime_probe.xlsx"
workbook = openpyxl.Workbook()
workbook.active["A1"] = "runtime-ok"
workbook.save(path)
observed = openpyxl.load_workbook(path, data_only=False).active["A1"].value
print(json.dumps({
    "executable": sys.executable,
    "version": sys.version.split()[0],
    "prefix": sys.prefix,
    "base_prefix": sys.base_prefix,
    "openpyxl": openpyxl.__version__,
    "numpy": numpy.__version__,
    "xlsx_roundtrip": observed,
}, sort_keys=True))
""".strip()


def _tool_factory(mode: str, baseline_root: Path | None) -> Callable[..., Any]:
    if mode == "sandbox":
        from spreadsheet_agent.tools.bash import create_bash_tool

        return create_bash_tool
    if baseline_root is None:
        raise ValueError("--baseline-root is required for Skill2Bench")
    root = baseline_root.expanduser().resolve()
    if not (root / "spreadsheet_agent/tools/bash.py").is_file():
        raise ValueError("Skill2Bench baseline Bash tool is absent")
    sys.path.insert(0, str(root))
    from spreadsheet_agent.tools.bash import create_bash_tool

    return create_bash_tool


def _probe(mode: str, baseline_root: Path | None) -> dict[str, Any]:
    prefix = Path(sys.prefix).resolve()
    os.environ["PATH"] = os.pathsep.join((str(prefix / "bin"), "/usr/bin", "/bin"))
    os.environ["VIRTUAL_ENV"] = str(prefix)
    os.environ["PYTHONNOUSERSITE"] = "1"
    factory = _tool_factory(mode, baseline_root)
    with tempfile.TemporaryDirectory(prefix="degs-agent-runtime-") as temporary:
        options = {"sandbox_mode": "required"} if mode == "sandbox" else {}
        bash = factory(temporary, timeout=120, **options)
        interpreters = {}
        for command in ("python", "python3"):
            output = bash.execute(command=f"{command} -c {shlex.quote(PROBE_CODE)}")
            try:
                record = json.loads(output)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Agent Bash {command} probe failed: {output}") from exc
            if record.get("xlsx_roundtrip") != "runtime-ok":
                raise RuntimeError(f"Agent Bash {command} workbook roundtrip failed")
            if Path(str(record.get("prefix"))).resolve() != prefix:
                raise RuntimeError(f"Agent Bash {command} did not use the project environment")
            interpreters[command] = record
    return {
        "format": "degs_agent_runtime_preflight_v1",
        "tool_environment": (
            "bubblewrap" if mode == "sandbox" else "skill2bench_baseline_host_bash"
        ),
        "interpreters": interpreters,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("sandbox", "skill2bench"), required=True)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = _probe(args.mode, args.baseline_root)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        output = args.output.expanduser().absolute()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
        temporary.write_text(rendered, encoding="utf-8")
        os.replace(temporary, output)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
