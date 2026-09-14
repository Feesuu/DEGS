from __future__ import annotations

import re
from typing import Any, Mapping

from degs.validated_repair import ValidatedRepairMemory


_STEP_MARKER = re.compile(
    r"(?im)^[ \t]*(?:#{1,6}[ \t]*)?(?:[-*+][ \t]+)?(?:\*\*|__)?"
    r"(?:(?:final[ \t]+)?(?:answer|response|result|solution)(?:[ \t]+for)?[ \t]+)?"
    r"step[ \t]*(\d+)\b[^\n]*"
)
_STEP_TRANSITION = re.compile(
    r"(?i)\b(?:now[ \t]+for|moving[ \t]+(?:on[ \t]+)?to|"
    r"proceed(?:ing)?[ \t]+to|turn(?:ing)?[ \t]+to|next[ \t]*,?)"
    r"[ \t]+(?:solve[ \t]+|answer[ \t]+|address[ \t]+)?step[ \t]*(\d+)\b"
)


def _events(rollout: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    rows = rollout.get("react_steps")
    if not isinstance(rows, list) or any(type(row) is not dict for row in rows):
        raise ValueError("Skill2Bench replay trace differs")
    return tuple(dict(row) for row in rows)


def source_step_fragments(
    rollout: Mapping[str, Any], *, expected_steps: int
) -> dict[int, tuple[dict[str, Any], ...]]:
    """Assign original-success events only after an explicit Step marker.

    A one-Step task is intrinsically attributable. A multi-Step trace never
    receives an implicit Step-1 owner.
    """
    if expected_steps <= 0:
        raise ValueError("Skill2Bench Step count differs")
    active = 1 if expected_steps == 1 else None
    result: dict[int, list[dict[str, Any]]] = {}
    for event in _events(rollout):
        thought = str(event.get("thought") or "")
        markers = [
            (match.start(), int(match.group(1)))
            for pattern in (_STEP_MARKER, _STEP_TRANSITION)
            for match in pattern.finditer(thought)
            if 1 <= int(match.group(1)) <= expected_steps
        ]
        if markers:
            active = min(markers, key=lambda row: row[0])[1]
        if active is not None:
            result.setdefault(active, []).append(event)
    return {key: tuple(value) for key, value in result.items() if value}


def original_success_record(
    *,
    evidence_id: str,
    step_number: int,
    rollout: Mapping[str, Any],
    expected_steps: int,
) -> dict[str, Any] | None:
    fragment = source_step_fragments(
        rollout,
        expected_steps=expected_steps,
    ).get(step_number)
    if not fragment:
        return None
    return {
        "evidence_id": evidence_id,
        "origin": "ORIGINAL_SUCCESS",
        "step_index": step_number,
        "trace_scope": "TARGET_STEP",
        "successful_trajectory": list(fragment),
    }


def validated_repair_record(
    *,
    evidence_id: str,
    step_number: int,
    memory: ValidatedRepairMemory | Mapping[str, Any],
    successful_replay: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(memory, ValidatedRepairMemory):
        memory = ValidatedRepairMemory(
            tuple(memory["instructions"]),
            tuple(memory["checks"]),
        )
    events = _events(successful_replay)
    if not events:
        raise ValueError("accepted repair replay has no usable trajectory")
    return {
        "evidence_id": evidence_id,
        "origin": "VALIDATED_REPAIR",
        "step_index": step_number,
        "trace_scope": "FULL_SUCCESSFUL_REPLAY",
        "validated_repair_memory": memory.to_dict(),
        "successful_trajectory": list(events),
    }


def validate_step_evidence(
    value: Mapping[str, Any], *, expected_step: int
) -> dict[str, Any]:
    origin = value.get("origin")
    fields = {
        "evidence_id",
        "origin",
        "step_index",
        "trace_scope",
        "successful_trajectory",
    }
    if origin == "VALIDATED_REPAIR":
        fields.add("validated_repair_memory")
    if type(value) is not dict or set(value) != fields:
        raise ValueError("Skill2Bench Step evidence fields differ")
    if (
        not value["evidence_id"]
        or value["step_index"] != expected_step
        or not isinstance(value["successful_trajectory"], list)
        or not value["successful_trajectory"]
        or any(type(row) is not dict for row in value["successful_trajectory"])
    ):
        raise ValueError("Skill2Bench Step evidence content differs")
    if origin == "ORIGINAL_SUCCESS":
        if value["trace_scope"] != "TARGET_STEP":
            raise ValueError("original success must be attributable to one Step")
    elif origin == "VALIDATED_REPAIR":
        if value["trace_scope"] != "FULL_SUCCESSFUL_REPLAY":
            raise ValueError("repair evidence must retain the full replay")
        memory = value["validated_repair_memory"]
        ValidatedRepairMemory(tuple(memory["instructions"]), tuple(memory["checks"]))
    else:
        raise ValueError("Skill2Bench Step evidence origin differs")
    return dict(value)


__all__ = [
    "original_success_record",
    "source_step_fragments",
    "validate_step_evidence",
    "validated_repair_record",
]
