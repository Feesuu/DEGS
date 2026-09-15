#!/usr/bin/env python3
"""Run source replay with the bundled SpreadsheetBench train runtime."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any, Sequence


def _adapt_evaluator_command(
    command: Sequence[str],
    *,
    model: str,
    python_executable: str,
    runtime_root: Path,
    method_root: Path,
    evaluator_adapter: Path,
    inherited_pythonpath: str = "",
) -> list[str]:
    child = list(command)
    if child[:3] != [python_executable, "-m", "sb_adapter.evaluate"]:
        return child
    return [
        "/usr/bin/env",
        "PYTHONPATH="
        + os.pathsep.join(
            filter(
                None,
                (
                    str(runtime_root / "src"),
                    str(method_root / "src"),
                    inherited_pythonpath,
                ),
            )
        ),
        python_executable,
        str(evaluator_adapter),
        "--expected-model",
        model,
        *child[3:],
    ]


def _load_method(method_root: Path, fallback_src: Path) -> Any:
    sys.path.insert(0, str(method_root.resolve() / "src"))
    sys.path.append(str(fallback_src))
    import degs

    fallback_package = fallback_src / "degs"
    if str(fallback_package) not in degs.__path__:
        degs.__path__.append(str(fallback_package))
    from degs import source_replay_executor

    return source_replay_executor


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--evaluator-adapter", type=Path, required=True)
    args, remaining = parser.parse_known_args(argv)
    repository = Path(__file__).resolve().parents[1]
    runtime_root = args.runtime_root.expanduser().resolve()
    module = _load_method(args.method_root, repository / "src")
    original_executor = module.SubprocessReplayExecutor
    evaluator_adapter = args.evaluator_adapter.expanduser().resolve()
    if not evaluator_adapter.is_file():
        raise FileNotFoundError("replay evaluator adapter is unavailable")

    class BundledExecutor(original_executor):
        def __init__(self, runtime: Any) -> None:
            if Path(runtime.upstream_root).resolve() != runtime_root:
                raise ValueError("source replay runtime root differs")
            super().__init__(runtime)

        def _run_command(
            self,
            command: list[str],
            audit_path: Path,
            *,
            context_overflow_marker: Path | None = None,
        ) -> str:
            child = _adapt_evaluator_command(
                command,
                model=self.runtime.model,
                python_executable=self.runtime.python_executable,
                runtime_root=runtime_root,
                method_root=args.method_root.expanduser().resolve(),
                evaluator_adapter=evaluator_adapter,
                inherited_pythonpath=os.environ.get("PYTHONPATH", ""),
            )
            return super()._run_command(
                child,
                audit_path,
                context_overflow_marker=context_overflow_marker,
            )
    module.SubprocessReplayExecutor = BundledExecutor
    sys.argv = ["degs.source_replay_executor", *remaining]
    try:
        return int(module.main() or 0)
    finally:
        module.SubprocessReplayExecutor = original_executor


if __name__ == "__main__":
    raise SystemExit(main())
