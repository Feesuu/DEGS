from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from .contextual_binding import ExperienceExpectation, experience_expectation_from_dict


class EpisodeOutcome(str, Enum):
    ORIGINAL_SUCCESS = "ORIGINAL_SUCCESS"
    REPAIR_SUCCESS = "REPAIR_SUCCESS"
    UNRESOLVED_TASK_FAILURE = "UNRESOLVED_TASK_FAILURE"
    ITEM_LOCAL_RUNTIME_FAILURE = "ITEM_LOCAL_RUNTIME_FAILURE"


@dataclass(frozen=True)
class EvidenceItem:
    evidence_id: str
    kind: str
    content: Any

    def __post_init__(self) -> None:
        if any(type(value) is not str or not value.strip() for value in (self.evidence_id, self.kind)):
            raise ValueError("episode evidence identity differs")

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "kind": self.kind,
            "content": self.content,
        }


@dataclass(frozen=True)
class EpisodeEvidence:
    episode_id: str
    dataset_contract_id: str
    train_index: int
    task_id: str
    read_snapshot_id: str
    query_text: str
    observable_context: tuple[EvidenceItem, ...]
    retrieval_context: Mapping[str, Any]
    expectations: tuple[ExperienceExpectation, ...]
    original_trace: tuple[EvidenceItem, ...]
    original_verifier: tuple[EvidenceItem, ...]
    outcome: EpisodeOutcome
    final_patch: tuple[EvidenceItem, ...] = ()
    replay_trace: tuple[EvidenceItem, ...] = ()
    replay_verifier: tuple[EvidenceItem, ...] = ()

    def __post_init__(self) -> None:
        if (
            any(
                type(value) is not str or not value.strip()
                for value in (
                    self.episode_id,
                    self.dataset_contract_id,
                    self.task_id,
                    self.read_snapshot_id,
                    self.query_text,
                )
            )
            or type(self.train_index) is not int
            or self.train_index < 0
            or not isinstance(self.outcome, EpisodeOutcome)
        ):
            raise ValueError("episode identity differs")
        items = (
            EvidenceItem("query:0", "query", self.query_text),
            *self.observable_context,
            *self.original_trace,
            *self.original_verifier,
            *self.final_patch,
            *self.replay_trace,
            *self.replay_verifier,
        )
        identities = [row.evidence_id for row in items]
        if len(identities) != len(set(identities)):
            raise ValueError("episode evidence IDs must be unique")
        if self.outcome is EpisodeOutcome.ITEM_LOCAL_RUNTIME_FAILURE:
            if self.final_patch or self.replay_trace or self.replay_verifier:
                raise ValueError("runtime failure cannot contain repair success evidence")
            return
        if not self.original_trace or not self.original_verifier:
            raise ValueError("observable episode requires original trace and verifier")
        if self.outcome is EpisodeOutcome.REPAIR_SUCCESS:
            if not self.final_patch or not self.replay_trace or not self.replay_verifier:
                raise ValueError("repair-success episode evidence is incomplete")
        elif self.final_patch or self.replay_trace or self.replay_verifier:
            raise ValueError("non-repair-success episode contains positive repair evidence")

    @property
    def evidence_by_id(self) -> Mapping[str, EvidenceItem]:
        rows = (
            EvidenceItem("query:0", "query", self.query_text),
            *self.observable_context,
            *self.original_trace,
            *self.original_verifier,
            *self.final_patch,
            *self.replay_trace,
            *self.replay_verifier,
        )
        return {row.evidence_id: row for row in rows}

    @property
    def successful(self) -> bool:
        return self.outcome in {
            EpisodeOutcome.ORIGINAL_SUCCESS,
            EpisodeOutcome.REPAIR_SUCCESS,
        }

    def to_learning_payload(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "dataset_contract_id": self.dataset_contract_id,
            "train_index": self.train_index,
            "task_id": self.task_id,
            "read_snapshot_id": self.read_snapshot_id,
            "query": EvidenceItem("query:0", "query", self.query_text).to_dict(),
            "observable_context": [row.to_dict() for row in self.observable_context],
            "retrieval_context": dict(self.retrieval_context),
            "experience_expectations": [row.to_dict() for row in self.expectations],
            "original_trace": [row.to_dict() for row in self.original_trace],
            "original_verifier": [row.to_dict() for row in self.original_verifier],
            "outcome": self.outcome.value,
            "final_effective_patch": [row.to_dict() for row in self.final_patch],
            "successful_fresh_replay": [row.to_dict() for row in self.replay_trace],
            "replay_verifier": [row.to_dict() for row in self.replay_verifier],
        }


def _stored_items(value: Any, *, label: str) -> tuple[EvidenceItem, ...]:
    if type(value) is not list:
        raise ValueError(f"stored {label} differs")
    rows = []
    for item in value:
        if type(item) is not dict or set(item) != {"evidence_id", "kind", "content"}:
            raise ValueError(f"stored {label} differs")
        rows.append(EvidenceItem(str(item["evidence_id"]), str(item["kind"]), item["content"]))
    return tuple(rows)


def episode_evidence_from_payload(value: Mapping[str, Any]) -> EpisodeEvidence:
    if type(value) is not dict:
        raise ValueError("stored episode evidence differs")
    query = value.get("query")
    if type(query) is not dict or query.get("evidence_id") != "query:0":
        raise ValueError("stored episode query differs")
    expectations = value.get("experience_expectations")
    if type(expectations) is not list:
        raise ValueError("stored episode expectations differ")
    try:
        outcome = EpisodeOutcome(value.get("outcome"))
    except (TypeError, ValueError) as exc:
        raise ValueError("stored episode outcome differs") from exc
    train_index = value.get("train_index")
    if type(train_index) is not int:
        raise ValueError("stored episode train index differs")
    episode = EpisodeEvidence(
        episode_id=str(value.get("episode_id") or ""),
        dataset_contract_id=str(value.get("dataset_contract_id") or ""),
        train_index=train_index,
        task_id=str(value.get("task_id") or ""),
        read_snapshot_id=str(value.get("read_snapshot_id") or ""),
        query_text=str(query.get("content") or ""),
        observable_context=_stored_items(value.get("observable_context"), label="context"),
        retrieval_context=dict(value.get("retrieval_context") or {}),
        expectations=tuple(experience_expectation_from_dict(row) for row in expectations),
        original_trace=_stored_items(value.get("original_trace"), label="original trace"),
        original_verifier=_stored_items(value.get("original_verifier"), label="original verifier"),
        outcome=outcome,
        final_patch=_stored_items(value.get("final_effective_patch"), label="final patch"),
        replay_trace=_stored_items(value.get("successful_fresh_replay"), label="replay trace"),
        replay_verifier=_stored_items(value.get("replay_verifier"), label="replay verifier"),
    )
    if episode.to_learning_payload() != dict(value):
        raise ValueError("stored episode evidence fields differ")
    return episode


__all__ = [
    "EpisodeEvidence",
    "EpisodeOutcome",
    "EvidenceItem",
    "episode_evidence_from_payload",
]
