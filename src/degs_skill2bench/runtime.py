from __future__ import annotations

import importlib
import importlib.resources
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any, Mapping

from .contract import Skill2BenchProtocol


AGENT_DATASET_PROFILE = (
    importlib.resources.files("degs_skill2bench")
    .joinpath("resources", "SKILL2BENCH_AGENT_PROFILE_V1.txt")
    .read_text(encoding="utf-8")
    .strip()
)
AGENT_DATASET_PROFILE_SHA256 = hashlib.sha256(
    AGENT_DATASET_PROFILE.encode()
).hexdigest()
_USAGE_LOCK = threading.Lock()


class Skill2BenchWorkerError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        worker_error_type: str,
        status_code: int | None,
        transport_failure: bool,
    ) -> None:
        super().__init__(message)
        self.worker_error_type = worker_error_type
        self.status_code = status_code
        self.transport_failure = transport_failure


def _usage_payload(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return dict(usage.model_dump())
    return {
        key: value
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        if (value := getattr(usage, key, None)) is not None
    }


class _RecordedOpenAIJudge:
    def __init__(self, evaluator_module: Any, *, model: str, base_url: str, api_key: str):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.system_prompt = evaluator_module.JUDGE_SYSTEM_PROMPT

    def score(self, question: str, rubric: str, response: str) -> float | None:
        result = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"Question:\n{question}\n\nGrading Rubric:\n{rubric}"
                        f"\n\nStudent Response:\n{response}"
                    ),
                },
            ],
            temperature=0.0,
            max_tokens=16,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        usage_path = os.getenv("REACT_AGENT_USAGE_LOG")
        if usage_path:
            record = {
                "component": "skill2bench_open_judge",
                "model": self.model,
                "endpoint": str(self.client.base_url),
                "usage": _usage_payload(result),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            path = Path(usage_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with _USAGE_LOCK, path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
        reply = result.choices[0].message.content
        try:
            return max(0.0, min(1.0, float((reply or "").strip())))
        except ValueError:
            return None


def _baseline_modules(baseline_root: Path):
    root = baseline_root.expanduser().resolve()
    if not (root / "skill2bench/agent.py").is_file():
        raise ValueError("Skill2Bench baseline root differs")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    agent = importlib.import_module("skill2bench.agent")
    evaluator = importlib.import_module("skill2bench.evaluator")
    react_agent = importlib.import_module("react_agent")
    if (
        not Path(agent.__file__).resolve().is_relative_to(root)
        or not Path(evaluator.__file__).resolve().is_relative_to(root)
        or not Path(react_agent.__file__).resolve().is_relative_to(root)
    ):
        raise ValueError("Skill2Bench baseline module was loaded from another runtime")
    return agent, evaluator, react_agent


def render_agent_skill(experience: str) -> str:
    guidance = experience.strip() or "No reusable graph experience was retrieved."
    return (
        "# Skill2Bench dataset profile\n\n"
        f"{AGENT_DATASET_PROFILE}\n\n"
        "# Step-scoped DEGS experience\n\n"
        f"{guidance}\n"
    )


def _run_and_evaluate_task_local(
    *,
    task: Mapping[str, Any],
    baseline_root: Path,
    official_evaluator_root: Path,
    base_url: str,
    api_key: str,
    working_dir: Path,
    skill_path: Path | None,
    protocol: Skill2BenchProtocol,
) -> tuple[dict[str, Any], dict[str, Any]]:
    agent_module, evaluator_module, react_agent = _baseline_modules(baseline_root)
    client = react_agent.OpenAIClient(
        model=protocol.model,
        api_key=api_key,
        base_url=base_url,
        generation_config={
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 40,
            "min_p": 0.0,
            "presence_penalty": 2.0,
            "repetition_penalty": 1.0,
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": protocol.thinking}
            },
        },
        retry_times=(5, 10, 30),
        timeout=600,
        trust_env=False,
        runtime_timeout_retries=1,
    )
    runner = agent_module.Skill2BenchAgent(
        client,
        working_dir=working_dir,
        max_turns=protocol.max_turns,
        max_tokens=None,
        temperature=1.0,
        skill_path=skill_path,
        bash_timeout=120,
        verbose=False,
    )
    rollout = dict(runner.run(dict(task)))
    judge = _RecordedOpenAIJudge(
        evaluator_module,
        model=protocol.model,
        base_url=base_url,
        api_key=api_key,
    )
    evaluation = dict(
        evaluator_module.evaluate_task(
            dict(task),
            str(rollout.get("answer") or ""),
            judge=judge,
            official_evaluator_root=official_evaluator_root,
            strict_official=True,
            require_open_ended_judge=True,
        )
    )
    return rollout, evaluation


def _aggregate_metrics_local(
    evaluations: list[dict[str, Any]],
    rollouts: list[dict[str, Any]],
    *,
    baseline_root: Path,
) -> dict[str, Any]:
    _agent, _evaluator, _react_agent = _baseline_modules(baseline_root)
    metrics = importlib.import_module("skill2bench.metrics")
    return dict(metrics.aggregate_metrics(evaluations, rollouts))


def _worker_request(
    mode: str, payload: Mapping[str, Any], *, baseline_root: Path
) -> dict[str, Any]:
    root = baseline_root.expanduser().resolve()
    environment = dict(os.environ)
    repository_src = Path(__file__).resolve().parents[1]
    prefix = Path(sys.prefix).resolve()
    environment["PATH"] = os.pathsep.join(
        (str(prefix / "bin"), "/usr/bin", "/bin")
    )
    environment["VIRTUAL_ENV"] = str(prefix)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(root / "src"), str(root), str(repository_src))
    )
    completed = subprocess.run(
        [sys.executable, "-m", "degs_skill2bench.runtime_worker", "--mode", mode],
        input=json.dumps(dict(payload), ensure_ascii=False),
        text=True,
        capture_output=True,
        env=environment,
    )
    if completed.returncode:
        try:
            failure = json.loads(completed.stdout)
        except json.JSONDecodeError:
            failure = None
        if (
            type(failure) is dict
            and set(failure) == {
                "error_type",
                "message",
                "status_code",
                "transport_failure",
                "traceback",
            }
            and type(failure["error_type"]) is str
            and type(failure["message"]) is str
            and (
                failure["status_code"] is None
                or type(failure["status_code"]) is int
            )
            and type(failure["transport_failure"]) is bool
        ):
            raise Skill2BenchWorkerError(
                (
                    f"Skill2Bench {mode} worker failed with "
                    f"{failure['error_type']}: {failure['message']}\n"
                    f"{failure['traceback']}"
                ),
                worker_error_type=failure["error_type"],
                status_code=failure["status_code"],
                transport_failure=failure["transport_failure"],
            )
        raise RuntimeError(
            f"Skill2Bench {mode} worker failed: {completed.stderr.strip()}"
        )
    value = json.loads(completed.stdout)
    if type(value) is not dict:
        raise ValueError("Skill2Bench worker response differs")
    return value


def run_and_evaluate_task(
    *,
    task: Mapping[str, Any],
    baseline_root: Path,
    official_evaluator_root: Path,
    base_url: str,
    api_key: str,
    working_dir: Path,
    skill_path: Path | None,
    protocol: Skill2BenchProtocol,
) -> tuple[dict[str, Any], dict[str, Any]]:
    result = _worker_request(
        "run",
        {
            "task": dict(task),
            "baseline_root": str(baseline_root),
            "official_evaluator_root": str(official_evaluator_root),
            "base_url": base_url,
            "api_key": api_key,
            "working_dir": str(working_dir),
            "skill_path": None if skill_path is None else str(skill_path),
            "profile": protocol.profile,
        },
        baseline_root=baseline_root,
    )
    return dict(result["rollout"]), dict(result["evaluation"])


def aggregate_metrics(
    evaluations: list[dict[str, Any]],
    rollouts: list[dict[str, Any]],
    *,
    baseline_root: Path,
) -> dict[str, Any]:
    return _worker_request(
        "aggregate",
        {
            "baseline_root": str(baseline_root),
            "evaluations": evaluations,
            "rollouts": rollouts,
        },
        baseline_root=baseline_root,
    )


__all__ = [
    "AGENT_DATASET_PROFILE_SHA256",
    "Skill2BenchWorkerError",
    "_aggregate_metrics_local",
    "_run_and_evaluate_task_local",
    "aggregate_metrics",
    "render_agent_skill",
    "run_and_evaluate_task",
]
