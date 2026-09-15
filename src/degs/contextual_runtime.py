"""Shared online Top-5 retrieval and evidence-bounded binding runtime."""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Mapping, Protocol, Sequence

from .contextual_binding import (
    ContextualBindingProducer,
    ExperienceExpectation,
    parse_experience_expectations,
)
from .contextual_retrieval import ContextualRetrieval, retrieve_contextual_subgraph
from .core import StrictEmbeddingAdapter
from .episode_evidence import EvidenceItem
from .section_graph import ExperienceGraph
from .validated_repair import SystemicProducerTransportFailure


class ContextualQuery(Protocol):
    @property
    def train_index(self) -> int: ...

    @property
    def task_id(self) -> str: ...

    @property
    def query_text(self) -> str: ...

    @property
    def observable_context(self) -> tuple[EvidenceItem, ...]: ...

    @property
    def retrieval_document(self) -> str: ...


async def retrieve_and_bind(
    *,
    items: Sequence[ContextualQuery],
    graph: ExperienceGraph,
    snapshot_id: str,
    canonical_versions: Mapping[str, int],
    embedding: StrictEmbeddingAdapter,
    binding: ContextualBindingProducer,
    workers: int = 16,
    request_prefix: str,
    cached_expectations: Mapping[str, Sequence[ExperienceExpectation]] | None = None,
    decision_callback: Callable[
        [ContextualQuery, ContextualRetrieval, tuple[ExperienceExpectation, ...]],
        Awaitable[None],
    ]
    | None = None,
) -> tuple[
    tuple[ContextualRetrieval, ...],
    tuple[tuple[ExperienceExpectation, ...], ...],
    Mapping[str, str],
]:
    rows = tuple(items)
    if not rows or type(workers) is not int or workers <= 0:
        raise ValueError("contextual retrieval batch differs")
    cached = dict(cached_expectations or {})
    task_ids = {row.task_id for row in rows}
    if len(task_ids) != len(rows) or not set(cached) <= task_ids:
        raise ValueError("contextual binding cache population differs")
    embedded = await embedding.embed_async(
        [row.retrieval_document for row in rows]
        + [node.document for node in graph.nodes]
    )
    canonical_vectors = {
        node.canonical_id: embedded[len(rows) + index].vector
        for index, node in enumerate(graph.nodes)
    }
    retrievals = tuple(
        retrieve_contextual_subgraph(
            graph=graph,
            snapshot_id=snapshot_id,
            query_vector=embedded[index].vector,
            canonical_vectors=canonical_vectors,
            canonical_versions=canonical_versions,
        )
        for index in range(len(rows))
    )
    decisions: list[tuple[ExperienceExpectation, ...]] = [() for _ in rows]
    failures: dict[str, str] = {}
    semaphore = asyncio.Semaphore(workers)

    async def one(index: int) -> None:
        row = rows[index]
        retrieval = retrievals[index]
        try:
            if row.task_id in cached:
                decisions[index] = parse_experience_expectations(
                    {
                        "expectations": [
                            item.to_dict() for item in cached[row.task_id]
                        ]
                    },
                    anchor_versions=retrieval.anchor_versions,
                    observable_evidence_ids={
                        "query:0",
                        *(item.evidence_id for item in row.observable_context),
                    },
                )
            else:
                async with semaphore:
                    decisions[index] = await binding.produce(
                        request_id=(
                            f"{request_prefix}-{row.train_index}-{row.task_id}"
                        ),
                        payload=retrieval.to_binding_payload(
                            query={"evidence_id": "query:0", "content": row.query_text},
                            observable_context=[
                                item.to_dict() for item in row.observable_context
                            ],
                        ),
                        anchor_versions=retrieval.anchor_versions,
                        observable_evidence_ids={
                            "query:0", *(item.evidence_id for item in row.observable_context)
                        },
                    )
        except SystemicProducerTransportFailure:
            raise
        except Exception as exc:
            failures[row.task_id] = f"{type(exc).__name__}: {exc}"
            return
        if decision_callback is not None:
            await decision_callback(row, retrieval, decisions[index])

    await asyncio.gather(*(one(index) for index in range(len(rows))))
    return retrievals, tuple(decisions), failures


__all__ = ["ContextualQuery", "retrieve_and_bind"]
