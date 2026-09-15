from __future__ import annotations

import asyncio
from pathlib import Path

from degs import eir_bundle
from degs.contextual_binding import ContextualBindingProducer
from degs.contextual_runtime import retrieve_and_bind
from degs.core import EMBEDDING_MODEL, StrictEmbeddingAdapter
from degs.dynamic_train import PreparedEpisode
from degs.eir_bundle import build_contextual_bundle
from degs.episode_evidence import EvidenceItem
from degs.graph_dataset_contract import SPREADSHEETBENCH_GRAPH_CONTRACT
from degs.section_graph import (
    CanonicalExperience,
    CanonicalNode,
    ExperienceGraph,
    IOContract,
)
from degs.state_store import EIRStateStore


class _CountedEmbeddingTransport:
    def __init__(self) -> None:
        self.calls = 0

    def embed(self, request):
        self.calls += 1
        return {
            "model": EMBEDDING_MODEL,
            "data": [
                {"index": index, "embedding": [1.0, float(index + 1)]}
                for index, _text in enumerate(request["input"])
            ],
        }


class _ConstructedEmbeddingTransport(_CountedEmbeddingTransport):
    def __init__(self, **_kwargs) -> None:
        super().__init__()


class _CountedBindingLLM:
    protocol_identity = {"format": "counted_binding_llm_v1"}

    def __init__(self) -> None:
        self.calls = 0

    async def complete_json_async(self, **kwargs):
        self.calls += 1
        anchor = kwargs["payload"]["anchors"][0]
        return {
            "expectations": [
                {
                    "canonical_id": anchor["canonical_id"],
                    "canonical_version": anchor["canonical_version"],
                    "condition": "SATISFIED",
                    "condition_evidence_refs": ["query:0"],
                    "expected_role": "Apply the task-visible operation.",
                    "bound_parameters": [],
                    "guidance": "Apply the operation requested by this task.",
                    "expected_observation": "The requested state transition is visible.",
                }
            ]
        }


def test_empty_cache_executes_embedding_and_binding_producers() -> None:
    asyncio.run(_run())


async def _run() -> None:
    experience = CanonicalExperience(
        "Apply a task-visible operation.",
        ("The current task requests the operation.",),
        (IOContract("task state", "The current observable task state."),),
        (IOContract("updated state", "The requested state transition."),),
    )
    node = CanonicalNode("C1", experience, "canonical-document", "a" * 64)
    graph = ExperienceGraph(
        (node,),
        (),
        "b" * 64,
        "c" * 64,
        "d" * 64,
        graph_format="degs_eir_experience_graph_v1",
        snapshot_id="S1",
        canonical_versions={"C1": 1},
    )
    item = PreparedEpisode(
        0,
        "task-0",
        "Apply the visible operation.",
        (EvidenceItem("context:0", "state", "The target is observable."),),
        {},
    )
    embedding_transport = _CountedEmbeddingTransport()
    binding_llm = _CountedBindingLLM()
    retrievals, expectations, failures = await retrieve_and_bind(
        items=(item,),
        graph=graph,
        snapshot_id="S1",
        canonical_versions={"C1": 1},
        embedding=StrictEmbeddingAdapter(embedding_transport, cache={}),
        binding=ContextualBindingProducer(binding_llm),
        request_prefix="empty-cache",
    )
    assert embedding_transport.calls == 1
    assert binding_llm.calls == 1
    assert retrievals[0].anchors[0].canonical_id == "C1"
    assert expectations[0][0].guidance
    assert failures == {}
    await retrieve_and_bind(
        items=(item,),
        graph=graph,
        snapshot_id="S1",
        canonical_versions={"C1": 1},
        embedding=StrictEmbeddingAdapter(embedding_transport, cache={}),
        binding=ContextualBindingProducer(binding_llm),
        request_prefix="resume",
        cached_expectations={item.task_id: expectations[0]},
    )
    assert binding_llm.calls == 1


def test_target_retrieval_cache_does_not_mutate_source_graph_state(
    tmp_path: Path, monkeypatch
) -> None:
    state_path = tmp_path / "source.sqlite3"
    with EIRStateStore(
        state_path, dataset_contract=SPREADSHEETBENCH_GRAPH_CONTRACT
    ) as store:
        snapshot = store.begin_snapshot(
            batch_index=0,
            parent_snapshot_id=None,
            operation_input_sha256="1" * 64,
        )
        store.commit_snapshot(snapshot)
        assert len(store.embedding_cache()) == 0

    monkeypatch.setattr(
        eir_bundle, "QwenEmbeddingHTTPTransport", _ConstructedEmbeddingTransport
    )
    item = PreparedEpisode(
        200,
        "target-0",
        "Apply the visible operation.",
        (EvidenceItem("context:0", "state", "The target is observable."),),
        {},
    )
    output_dir = tmp_path / "bundle"
    asyncio.run(
        build_contextual_bundle(
            prepared=(item,),
            dataset_label="isolated-target",
            population_identity={"population": "target-0"},
            dataset_contract=SPREADSHEETBENCH_GRAPH_CONTRACT,
            state_db=state_path,
            output_dir=output_dir,
            generation_base_url="http://generation.test/v1",
            embedding_base_url="http://embedding.test/v1",
            model="Qwen3.5-9B-AWQ",
            generation_key="test-generation-key",
            embedding_key="test-embedding-key",
            fixed_denominator=1,
        )
    )

    with EIRStateStore(
        state_path, dataset_contract=SPREADSHEETBENCH_GRAPH_CONTRACT
    ) as source:
        assert len(source.embedding_cache()) == 0
    cache_path = tmp_path / ".bundle.checkpoints/retrieval_cache.sqlite3"
    with EIRStateStore(
        cache_path, dataset_contract=SPREADSHEETBENCH_GRAPH_CONTRACT
    ) as target_cache:
        assert len(target_cache.embedding_cache()) == 1
