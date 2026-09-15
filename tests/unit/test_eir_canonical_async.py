from __future__ import annotations

import asyncio
import hashlib
from types import SimpleNamespace

from degs.eir_canonical import EIRCanonicalResolver
from degs.section_graph import ExperienceNode, IOContract
from degs.state_store import ActiveCanonicalVersion


def _experience(operation: str) -> ExperienceNode:
    return ExperienceNode(
        operation,
        ("The current task requires this operation.",),
        (IOContract("value", "Read the value from the current task."),),
        (IOContract("state", "The requested state transition is complete."),),
    )


def _active(canonical_id: str) -> ActiveCanonicalVersion:
    experience = _experience(f"candidate-{canonical_id}")
    document = f"document-{canonical_id}"
    return ActiveCanonicalVersion(
        canonical_id,
        1,
        experience,
        document,
        hashlib.sha256(document.encode()).hexdigest(),
    )


class _ViewLLM:
    def __init__(self) -> None:
        self.active = 0
        self.maximum_active = 0

    async def complete_json_async(self, **kwargs):
        self.active += 1
        self.maximum_active = max(self.maximum_active, self.active)
        try:
            await asyncio.sleep(0.01)
            operation = kwargs["payload"]["experience_node"]["operation"]
            return {
                "reusable_identity": operation,
                "applicability_boundary": "task-bound",
            }
        finally:
            self.active -= 1


class _MergeLLM:
    def __init__(self) -> None:
        self.active = 0
        self.maximum_active = 0

    async def complete_json_async(self, **kwargs):
        self.active += 1
        self.maximum_active = max(self.maximum_active, self.active)
        operation = kwargs["payload"]["right"]["operation"]
        try:
            if operation == "candidate-a":
                await asyncio.sleep(0.005)
                raise ValueError("one malformed candidate response")
            await asyncio.sleep(0.02 if operation == "candidate-b" else 0.001)
            return {
                "relation": "SAME_TEMPLATE",
                "basis": f"same as {operation}",
                "canonical_experience": kwargs["payload"]["left"],
            }
        finally:
            self.active -= 1


class _Embedding:
    async def embed_async(self, texts):
        return tuple(SimpleNamespace(vector=(1.0, 0.0)) for _ in texts)


def test_eir_canonical_parallelizes_independent_llm_work_and_keeps_rank_order() -> None:
    async def exercise() -> None:
        views = _ViewLLM()
        merges = _MergeLLM()
        resolver = EIRCanonicalResolver(
            view_llm=views,
            merge_llm=merges,
            embedding=_Embedding(),
        )
        result = await resolver.resolve(
            source_node_id="source-0",
            experience=_experience("source"),
            active=tuple(_active(key) for key in ("a", "b", "c")),
        )

        assert result.target_canonical_id == "b"
        assert views.maximum_active >= 2
        assert merges.maximum_active >= 2

    asyncio.run(exercise())
