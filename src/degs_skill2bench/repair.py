from __future__ import annotations

import hashlib
import importlib.resources
from typing import Any, Mapping

from degs.core import canonical_json_bytes
from degs.validated_repair import JsonObjectLLM, ValidatedRepairMemory

from .runtime import AGENT_DATASET_PROFILE
from .step_evidence import source_step_fragments
from .step_units import public_step_view


PATCH_KIND = "generate_skill2bench_targeted_repair_v1"
PATCH_PROTOCOL_FORMAT = "degs_skill2bench_repair_patch_protocol_v1"
PATCH_SYSTEM_PROMPT = (
    importlib.resources.files("degs_skill2bench")
    .joinpath("resources", "SKILL2BENCH_REPAIR_PATCH_PROMPT_V1.txt")
    .read_text(encoding="utf-8")
    .strip()
)
PATCH_PROMPT_SHA256 = hashlib.sha256(PATCH_SYSTEM_PROMPT.encode()).hexdigest()


def patch_response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "instructions": {
                "type": "array",
                "minItems": 1,
                "items": {"type": "string", "minLength": 1},
            },
            "checks": {
                "type": "array",
                "minItems": 1,
                "items": {"type": "string", "minLength": 1},
            },
        },
        "required": ["instructions", "checks"],
        "additionalProperties": False,
    }


def step_outcome(step: Mapping[str, Any]) -> str:
    if step.get("status") in {"skipped", "unscorable"} or step.get("score") is None:
        return "UNSCORABLE"
    # The official adapter defines a correct Step as a score of one.  Partial
    # open-ended credit is still a failed Step and is eligible for repair.
    return "SUCCESS" if float(step["score"]) >= 0.999999 else "FAILURE"


def failed_step_payload(
    *,
    task: Mapping[str, Any],
    rollout: Mapping[str, Any],
    evaluated_step: Mapping[str, Any],
    attempt_index: int,
) -> dict[str, Any]:
    step_number = int(evaluated_step["step"])
    public = public_step_view(task, step_number=step_number)
    fragment = source_step_fragments(
        rollout,
        expected_steps=int(public["target_step"]["total_steps"]),
    ).get(step_number)
    trace_scope = "TARGET_STEP" if fragment else "NO_ATTRIBUTABLE_TARGET_TRACE"
    return {
        "scenario_background": public["scenario_background"],
        "target_step": dict(public["target_step"]),
        "failed_trajectory": list(fragment or ()),
        "trace_scope": trace_scope,
        "prediction": str(evaluated_step.get("prediction") or ""),
        "score": evaluated_step.get("score"),
        "status": str(evaluated_step.get("status") or ""),
        "attempt_index": attempt_index,
    }


async def produce_patch(
    *,
    llm: JsonObjectLLM,
    task_id: str,
    payload: Mapping[str, Any],
) -> tuple[ValidatedRepairMemory, str]:
    raw = await llm.complete_json_async(
        kind=PATCH_KIND,
        request_id=task_id,
        system_prompt=PATCH_SYSTEM_PROMPT,
        payload=payload,
        response_schema=patch_response_schema(),
    )
    if type(raw) is not dict or set(raw) != {"instructions", "checks"}:
        raise ValueError("Skill2Bench repair patch fields differ")
    memory = ValidatedRepairMemory(
        tuple(str(row).strip() for row in raw["instructions"]),
        tuple(str(row).strip() for row in raw["checks"]),
    )
    return memory, hashlib.sha256(canonical_json_bytes(memory.to_dict())).hexdigest()


def render_repair_skill(step_number: int, memory: ValidatedRepairMemory) -> str:
    return "\n".join(
        [
            "# Skill2Bench dataset profile",
            "",
            AGENT_DATASET_PROFILE,
            "",
            "# Targeted Repair Guidance",
            "",
            f"Apply this guidance only to Step {step_number} when its stated conditions hold.",
            "Do not alter unrelated Steps and do not treat this as a final answer.",
            "",
            "## Instructions",
            *(f"- {row}" for row in memory.instructions),
            "",
            "## Checks",
            *(f"- {row}" for row in memory.checks),
            "",
        ]
    )


__all__ = [
    "PATCH_PROMPT_SHA256",
    "PATCH_SYSTEM_PROMPT",
    "failed_step_payload",
    "patch_response_schema",
    "produce_patch",
    "render_repair_skill",
    "step_outcome",
]
