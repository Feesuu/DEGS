from __future__ import annotations

from typing import Any, Mapping, Sequence

from .contract import SKILL2BENCH_MAX_STEPS
from .dataset import public_task_view


def step_workflow_id(train_index: int, step_number: int) -> int:
    if train_index < 0 or not 1 <= step_number <= SKILL2BENCH_MAX_STEPS:
        raise ValueError("Skill2Bench Step workflow identity differs")
    return train_index * SKILL2BENCH_MAX_STEPS + step_number - 1


def step_workflow_coordinates(workflow_id: int) -> tuple[int, int]:
    if workflow_id < 0:
        raise ValueError("Skill2Bench Step workflow identity differs")
    task_index, step_offset = divmod(workflow_id, SKILL2BENCH_MAX_STEPS)
    return task_index, step_offset + 1


def step_workflow_ids_for_task_batch(task_indices: Sequence[int]) -> tuple[int, ...]:
    indices = tuple(task_indices)
    if not indices or len(indices) != len(set(indices)) or min(indices) < 0:
        raise ValueError("Skill2Bench task batch identity differs")
    return tuple(
        step_workflow_id(task_index, step_number)
        for task_index in indices
        for step_number in range(1, SKILL2BENCH_MAX_STEPS + 1)
    )


def step_task_id(train_index: int, step_number: int) -> str:
    step_workflow_id(train_index, step_number)
    return f"train-{train_index:03d}-step-{step_number:02d}"


def public_step_view(
    task: Mapping[str, Any], *, step_number: int, instance_id: str | None = None
) -> dict[str, Any]:
    public = public_task_view(task)
    if not 1 <= step_number <= len(public["questions"]):
        raise ValueError("Skill2Bench public Step is outside the task")
    return {
        "instance_id": instance_id or public["instance_id"],
        "scenario_background": public["scenario"],
        "target_step": {
            "number": step_number,
            "total_steps": len(public["questions"]),
            "question": public["questions"][step_number - 1],
        },
    }


__all__ = [
    "public_step_view",
    "step_task_id",
    "step_workflow_coordinates",
    "step_workflow_id",
    "step_workflow_ids_for_task_batch",
]
