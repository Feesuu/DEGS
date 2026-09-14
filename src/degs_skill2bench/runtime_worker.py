from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback
from typing import Any, Sequence

from openai import APIError
from react_agent.models import RequestRuntimeTimeout

from .contract import skill2bench_protocol
from .runtime import _aggregate_metrics_local, _run_and_evaluate_task_local


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("run", "aggregate"), required=True)
    args = parser.parse_args(argv)
    request: Any = json.load(sys.stdin)
    if type(request) is not dict:
        raise ValueError("Skill2Bench worker request differs")
    baseline_root = Path(request["baseline_root"])
    try:
        if args.mode == "run":
            protocol = skill2bench_protocol(request["profile"])
            rollout, evaluation = _run_and_evaluate_task_local(
                task=request["task"],
                baseline_root=baseline_root,
                official_evaluator_root=Path(request["official_evaluator_root"]),
                base_url=request["base_url"],
                api_key=request["api_key"],
                working_dir=Path(request["working_dir"]),
                skill_path=(
                    None
                    if request["skill_path"] is None
                    else Path(request["skill_path"])
                ),
                protocol=protocol,
            )
            result = {"rollout": rollout, "evaluation": evaluation}
        else:
            result = _aggregate_metrics_local(
                request["evaluations"],
                request["rollouts"],
                baseline_root=baseline_root,
            )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "status_code": getattr(exc, "status_code", None),
                    "transport_failure": isinstance(
                        exc, (APIError, RequestRuntimeTimeout)
                    ),
                    "traceback": traceback.format_exc(),
                },
                ensure_ascii=False,
            )
        )
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
