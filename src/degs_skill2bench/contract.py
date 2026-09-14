from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from degs.graph_dataset_contract import GraphDatasetContract


TRAIN_SHA256 = "7f0e2350db71b87b7f36e965fdf6e534afbf2b541f615818751ace61d10b93c2"
TEST_SHA256 = "d3d75c21a57b1173a28849d811424db7c116e47e3f8a6d77f97c53fedacff9c9"
BASELINE_PYTHON_SHA256 = "8c96fc11ad019096ebf06b5d3f77280f7d19c9610051c39c67979bd41b15a5f6"
OFFICIAL_EVALUATOR_SHA256 = "ae7e27f8278e1bebff88952a50de508ff7cfc64f1a186d115a3b2674f4fc6af4"
SKILL2BENCH_MAX_STEPS = 10
SKILL2BENCH_SOURCE_SPLIT = "skill2bench/train[0,100)/step-slots[0,1000)"
MODEL_BY_PROFILE = {
    "9b": "Qwen3.5-9B-AWQ",
    "27b": "Qwen3.5-27B-AWQ",
}


@dataclass(frozen=True)
class Skill2BenchProtocol:
    profile: Literal["9b", "27b"]
    model: str
    train_count: int = 100
    test_count: int = 200
    task_batch_size: int = 8
    agent_workers: int = 8
    producer_workers: int = 16
    max_turns: int = 30
    thinking: bool = False
    agent_max_tokens: None = None
    open_judge_max_tokens: int = 16
    train_sha256: str = TRAIN_SHA256
    test_sha256: str = TEST_SHA256

    @property
    def graph_contract(self) -> GraphDatasetContract:
        return GraphDatasetContract(
            identity=f"degs-07741-skill2bench-step-scoped-{self.profile}",
            source_split=SKILL2BENCH_SOURCE_SPLIT,
            train_count=self.train_count * SKILL2BENCH_MAX_STEPS,
            batch_size=self.task_batch_size * SKILL2BENCH_MAX_STEPS,
            allow_final_partial_batch=True,
        )

    def train_batches(self) -> tuple[tuple[int, ...], ...]:
        return tuple(
            tuple(range(start, min(start + self.task_batch_size, self.train_count)))
            for start in range(0, self.train_count, self.task_batch_size)
        )


def skill2bench_protocol(profile: Literal["9b", "27b"]) -> Skill2BenchProtocol:
    try:
        model = MODEL_BY_PROFILE[profile]
    except KeyError as exc:
        raise ValueError("Skill2Bench model profile differs") from exc
    return Skill2BenchProtocol(profile=profile, model=model)


SKILL2BENCH_GRAPH_CONTRACT = skill2bench_protocol("9b").graph_contract


__all__ = [
    "MODEL_BY_PROFILE",
    "BASELINE_PYTHON_SHA256",
    "OFFICIAL_EVALUATOR_SHA256",
    "SKILL2BENCH_GRAPH_CONTRACT",
    "SKILL2BENCH_MAX_STEPS",
    "SKILL2BENCH_SOURCE_SPLIT",
    "Skill2BenchProtocol",
    "skill2bench_protocol",
]
