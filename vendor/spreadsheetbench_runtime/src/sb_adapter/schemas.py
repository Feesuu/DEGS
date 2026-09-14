from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class TrajectoryStep:
    step_id: int
    raw_model_output: str
    action: str
    observation: str
    tool_name: str | None = None
    action_valid: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RawTrajectoryRecord:
    trajectory_id: str
    task_id: str
    trial_index: int
    instruction: str
    instruction_type: str = ""
    answer_position: str = ""
    spreadsheet_path: str = ""
    output_path: str = ""
    final_response: str | None = None
    success: bool = False
    verifier_score: float | None = None
    verifier_feedback: str | None = None
    steps: list[TrajectoryStep] = field(default_factory=list)
    runtime_metadata: dict[str, Any] = field(default_factory=dict)
    service_metadata: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["steps"] = [s.to_dict() for s in self.steps]
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RawTrajectoryRecord":
        data = dict(payload)
        data["steps"] = [s if isinstance(s, TrajectoryStep) else TrajectoryStep(**s) for s in data.get("steps", [])]
        return cls(**data)
