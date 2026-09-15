from __future__ import annotations

import math
from typing import Sequence

from .canonicalize import (
    CANONICAL_CANDIDATE_K,
    CANONICAL_MERGE_KIND,
    CANONICAL_MERGE_SYSTEM_PROMPT,
    CANONICAL_VIEW_KIND,
    CANONICAL_VIEW_SYSTEM_PROMPT,
    CanonicalizationView,
    TemplateRelation,
    canonical_merge_response_schema,
    canonicalization_view_embedding_text,
    canonicalization_view_response_schema,
    parse_canonical_merge,
    parse_canonicalization_view,
)
from .contextual_binding import JsonProducer
from .core import StrictEmbeddingAdapter
from .eir_graph import CanonicalResolution
from .section_graph import ExperienceNode
from .state_store import ActiveCanonicalVersion
from .validated_repair import SystemicProducerTransportFailure


class EIRCanonicalResolver:
    """Resolve new evidence nodes against active guard-binding-operation-effect units."""

    def __init__(
        self,
        *,
        view_llm: JsonProducer,
        merge_llm: JsonProducer,
        embedding: StrictEmbeddingAdapter,
    ) -> None:
        self.view_llm = view_llm
        self.merge_llm = merge_llm
        self.embedding = embedding
        self._views: dict[str, CanonicalizationView] = {}

    async def _view(
        self,
        *,
        key: str,
        experience: ExperienceNode,
    ) -> CanonicalizationView:
        cached = self._views.get(key)
        if cached is not None:
            return cached
        response = await self.view_llm.complete_json_async(
            kind=CANONICAL_VIEW_KIND,
            request_id=key,
            system_prompt=CANONICAL_VIEW_SYSTEM_PROMPT,
            payload={"experience_node": experience.to_dict()},
            response_schema=canonicalization_view_response_schema(),
        )
        view = parse_canonicalization_view(response)
        self._views[key] = view
        return view

    async def resolve(
        self,
        *,
        source_node_id: str,
        experience: ExperienceNode,
        active: Sequence[ActiveCanonicalVersion],
    ) -> CanonicalResolution:
        if not active:
            return CanonicalResolution(None, experience, "The active graph is empty.")
        try:
            source_view = await self._view(key=source_node_id, experience=experience)
            active_views = [
                await self._view(
                    key=f"{row.canonical_id}:v{row.version}:{row.document_sha256}",
                    experience=row.experience,
                )
                for row in active
            ]
            embedded = await self.embedding.embed_async(
                [
                    canonicalization_view_embedding_text(source_view),
                    *(canonicalization_view_embedding_text(row) for row in active_views),
                ]
            )
            query = embedded[0].vector
            query_norm = math.sqrt(sum(value * value for value in query))
            scored = sorted(
                (
                    (
                        sum(left * right for left, right in zip(query, embedded[index + 1].vector, strict=True))
                        / (
                            query_norm
                            * math.sqrt(
                                sum(
                                    value * value
                                    for value in embedded[index + 1].vector
                                )
                            )
                        ),
                        row.canonical_id,
                        row,
                        active_views[index],
                    )
                    for index, row in enumerate(active)
                ),
                key=lambda value: (-value[0], value[1]),
            )[:CANONICAL_CANDIDATE_K]
            for _score, canonical_id, candidate, candidate_view in scored:
                response = await self.merge_llm.complete_json_async(
                    kind=CANONICAL_MERGE_KIND,
                    request_id=f"{source_node_id}::{canonical_id}:v{candidate.version}",
                    system_prompt=CANONICAL_MERGE_SYSTEM_PROMPT,
                    payload={
                        "left": experience.to_dict(),
                        "left_view": source_view.to_dict(),
                        "right": candidate.experience.to_dict(),
                        "right_view": candidate_view.to_dict(),
                    },
                    response_schema=canonical_merge_response_schema(),
                )
                decision = parse_canonical_merge(response)
                if decision.relation is not TemplateRelation.SAME_TEMPLATE:
                    continue
                assert decision.canonical_experience is not None
                merged = decision.canonical_experience
                return CanonicalResolution(
                    canonical_id,
                    ExperienceNode(
                        merged.operation,
                        merged.applicability,
                        merged.inputs,
                        merged.outputs,
                    ),
                    decision.basis,
                )
            return CanonicalResolution(
                None,
                experience,
                "No active candidate had the same guard-binding-operation-effect function.",
            )
        except SystemicProducerTransportFailure:
            raise
        except Exception as exc:
            # A valid learned node remains evidence even when its optional semantic
            # disambiguation fails. Keep it as a singleton and audit the reason.
            return CanonicalResolution(
                None,
                experience,
                f"Canonical resolution failed item-locally: {type(exc).__name__}: {exc}",
            )


__all__ = ["EIRCanonicalResolver"]
