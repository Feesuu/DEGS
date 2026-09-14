from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
import hashlib
import importlib.resources
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Sequence, TypeVar

import numpy as np

from react_agent.models import OpenAIClient

from .core import canonical_json_bytes
from .section_graph import CanonicalExperience, _canonical_experience
from .validated_repair import OpenAIJsonObjectLLM, gather_cancel_on_error

CANONICAL_VIEW_KIND = "derive_experience_canonicalization_view_v4"
CANONICAL_VIEW_PROTOCOL_FORMAT = "degs_canonicalization_view_protocol_v4"
CANONICAL_VIEW_PROMPT_RESOURCE = "CANONICALIZATION_VIEW_PROMPT_V4.txt"
CANONICAL_MERGE_KIND = "merge_experience_operations_v2"
CANONICAL_MERGE_PROTOCOL_FORMAT = "degs_canonical_operation_merge_protocol_v2"
CANONICAL_MERGE_PROMPT_RESOURCE = "CANONICAL_OPERATION_MERGE_PROMPT_V2.txt"
CANONICAL_LLM_WORKERS = 16
CANONICAL_SEMANTIC_ATTEMPTS = 3
CANONICAL_CANDIDATE_K = 16
CANONICAL_RECALL_AUDIT_K = 64
CANONICAL_CANDIDATE_POLICY = "distinct_canonical_alias_top_k_plus_exact_v2"
CANONICAL_PARTITION_POLICY = "indivisible_prior_micro_operation_monotonic_merge_v2"

_Job = TypeVar("_Job")
_Result = TypeVar("_Result")


async def _run_canonical_jobs(
    jobs: Sequence[_Job], worker: Callable[[_Job], Awaitable[_Result]],
) -> tuple[_Result, ...]:
    semaphore = asyncio.Semaphore(CANONICAL_LLM_WORKERS)

    async def run_one(job: _Job) -> _Result:
        async with semaphore:
            return await worker(job)

    return await gather_cancel_on_error(tuple(run_one(job) for job in jobs))


def _prompt_text(resource: str) -> str:
    return importlib.resources.files("degs").joinpath("resources", resource).read_text(encoding="utf-8").strip()


CANONICAL_VIEW_SYSTEM_PROMPT = _prompt_text(CANONICAL_VIEW_PROMPT_RESOURCE)
CANONICAL_VIEW_PROMPT_SHA256 = hashlib.sha256(CANONICAL_VIEW_SYSTEM_PROMPT.encode()).hexdigest()
CANONICAL_MERGE_SYSTEM_PROMPT = _prompt_text(CANONICAL_MERGE_PROMPT_RESOURCE)
CANONICAL_MERGE_PROMPT_SHA256 = hashlib.sha256(CANONICAL_MERGE_SYSTEM_PROMPT.encode()).hexdigest()


@dataclass(frozen=True)
class CanonicalizationView:
    reusable_identity: str
    applicability_boundary: str

    def to_dict(self) -> dict[str, str]:
        return {"reusable_identity": self.reusable_identity, "applicability_boundary": self.applicability_boundary}


class TemplateRelation(str, Enum):
    SAME_TEMPLATE = "SAME_TEMPLATE"
    DIFFERENT_TEMPLATE = "DIFFERENT_TEMPLATE"
    UNCERTAIN = "UNCERTAIN"


@dataclass(frozen=True)
class CanonicalMergeDecision:
    relation: TemplateRelation
    basis: str
    canonical_experience: CanonicalExperience | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "relation": self.relation.value,
            "basis": self.basis,
            "canonical_experience": self.canonical_experience.to_dict() if self.canonical_experience is not None else None,
        }


@dataclass(frozen=True)
class CandidateNeighbor:
    source_canonical_id: str
    target_canonical_id: str
    similarity: float
    rank: int
    exact_text_match: bool = False

    def __post_init__(self) -> None:
        if (
            not self.source_canonical_id or not self.target_canonical_id
            or self.source_canonical_id == self.target_canonical_id
            or type(self.rank) is not int or self.rank <= 0
            or not np.isfinite(self.similarity)
        ):
            raise ValueError("Canonical candidate neighbor differs")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_canonical_id": self.source_canonical_id,
            "target_canonical_id": self.target_canonical_id,
            "similarity": self.similarity, "rank": self.rank,
            "exact_text_match": self.exact_text_match,
        }


def _normalize_structured_array(value: Any, *, label: str) -> tuple[tuple[Any, ...], int]:
    if type(value) is not list or not value:
        raise ValueError(f"{label} must be a non-empty array")
    unique: list[Any] = []
    seen: set[bytes] = set()
    for row in value:
        identity = canonical_json_bytes(row)
        if identity not in seen:
            seen.add(identity)
            unique.append(row)
    return tuple(unique), len(value) - len(unique)


def canonical_structured_array_policy() -> dict[str, Any]:
    return {
        "format": "degs_canonical_structured_array_policy_v2",
        "schema_max_items": None,
        "duplicate_normalization": "stable_first_occurrence_exact_canonical_json",
        "fields": ["canonical_experience.applicability", "canonical_experience.inputs", "canonical_experience.outputs"],
    }


def _canonical_experience_schema() -> dict[str, Any]:
    contract = {
        "type": "object",
        "properties": {
            "type": {"type": "string", "minLength": 1},
            "description": {"type": "string", "minLength": 1},
        },
        "required": ["type", "description"], "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "operation": {"type": "string", "minLength": 1},
            "applicability": {"type": "array", "minItems": 1, "items": {"type": "string", "minLength": 1}},
            "inputs": {"type": "array", "minItems": 1, "items": contract},
            "outputs": {"type": "array", "minItems": 1, "items": contract},
        },
        "required": ["operation", "applicability", "inputs", "outputs"], "additionalProperties": False,
    }


def canonicalization_view_response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "reusable_identity": {"type": "string", "minLength": 1},
            "applicability_boundary": {"type": "string", "minLength": 1},
        },
        "required": ["reusable_identity", "applicability_boundary"], "additionalProperties": False,
    }


def canonical_merge_response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "relation": {"type": "string", "enum": [row.value for row in TemplateRelation]},
            "basis": {"type": "string", "minLength": 1},
            "canonical_experience": {"anyOf": [_canonical_experience_schema(), {"type": "null"}]},
        },
        "required": ["relation", "basis", "canonical_experience"], "additionalProperties": False,
    }


def _strict_nonempty(value: Any, *, label: str) -> str:
    if type(value) is not str or not value.strip() or value != value.strip():
        raise ValueError(f"{label} differs")
    return value


def parse_canonicalization_view(response: Mapping[str, Any]) -> CanonicalizationView:
    if type(response) is not dict or set(response) != {"reusable_identity", "applicability_boundary"}:
        raise ValueError("Canonicalization View fields differ")
    return CanonicalizationView(
        _strict_nonempty(response["reusable_identity"], label="View identity"),
        _strict_nonempty(response["applicability_boundary"], label="View boundary"),
    )


def parse_canonical_merge(response: Mapping[str, Any]) -> CanonicalMergeDecision:
    if type(response) is not dict or set(response) != {"relation", "basis", "canonical_experience"}:
        raise ValueError("Canonical merge fields differ")
    relation = TemplateRelation(response["relation"])
    basis = _strict_nonempty(response["basis"], label="Canonical merge basis")
    raw = response["canonical_experience"]
    if relation is not TemplateRelation.SAME_TEMPLATE:
        if raw is not None:
            raise ValueError("Non-merge decision must not synthesize an experience")
        return CanonicalMergeDecision(relation, basis, None)
    if type(raw) is not dict or set(raw) != {"operation", "applicability", "inputs", "outputs"}:
        raise ValueError("SAME decision requires a complete Canonical experience")
    normalized = dict(raw)
    for field in ("applicability", "inputs", "outputs"):
        values, _duplicates = _normalize_structured_array(raw[field], label=field)
        normalized[field] = list(values)
    return CanonicalMergeDecision(relation, basis, _canonical_experience(normalized))


def canonicalization_view_embedding_text(view: CanonicalizationView) -> str:
    return canonical_json_bytes(view.to_dict()).decode("utf-8")


def _unit_vector(values: Sequence[float], *, label: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or not len(vector) or not np.isfinite(vector).all():
        raise ValueError(f"{label} embedding vector differs")
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError(f"{label} embedding vector is zero")
    return vector / norm


def canonical_unit_candidates(
    source_id: str,
    aliases: Mapping[str, Sequence[str]],
    vectors: Mapping[str, Sequence[float]],
    *,
    exact_text_by_leaf: Mapping[str, str] | None = None,
    k: int = CANONICAL_CANDIDATE_K,
) -> tuple[CandidateNeighbor, ...]:
    if type(k) is not int or k <= 0 or source_id not in aliases or not aliases[source_id]:
        raise ValueError("Canonical candidate input differs")
    source_matrix = np.stack([_unit_vector(vectors[key], label="View") for key in aliases[source_id]])
    source_texts = {exact_text_by_leaf[key] for key in aliases[source_id]} if exact_text_by_leaf is not None else set()
    scored: list[tuple[float, str, bool]] = []
    for target in sorted(aliases):
        if target == source_id:
            continue
        target_matrix = np.stack([_unit_vector(vectors[key], label="View") for key in aliases[target]])
        if source_matrix.shape[1] != target_matrix.shape[1]:
            raise ValueError("Canonical View embedding dimensions differ")
        similarity = float((source_matrix @ target_matrix.T).max())
        exact = exact_text_by_leaf is not None and any(exact_text_by_leaf[key] in source_texts for key in aliases[target])
        scored.append((-similarity, target, exact))
    scored.sort()
    selected = [row for rank, row in enumerate(scored) if rank < k or row[2]]
    return tuple(
        CandidateNeighbor(source_id, target, -score, rank, exact)
        for rank, (score, target, exact) in enumerate(selected, 1)
    )


def openai_canonical_view_llm(client: OpenAIClient) -> OpenAIJsonObjectLLM:
    return OpenAIJsonObjectLLM(
        client, request_kind=CANONICAL_VIEW_KIND,
        source_protocol_format=CANONICAL_VIEW_PROTOCOL_FORMAT,
        prompt_sha256=CANONICAL_VIEW_PROMPT_SHA256,
        response_schema_name="degs_canonicalization_view_v4",
    )


def openai_canonical_merge_llm(client: OpenAIClient) -> OpenAIJsonObjectLLM:
    return OpenAIJsonObjectLLM(
        client, request_kind=CANONICAL_MERGE_KIND,
        source_protocol_format=CANONICAL_MERGE_PROTOCOL_FORMAT,
        prompt_sha256=CANONICAL_MERGE_PROMPT_SHA256,
        response_schema_name="degs_canonical_operation_merge_v2",
    )
