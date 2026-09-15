from __future__ import annotations

import asyncio
from pathlib import Path

from degs.contextual_binding import ContextualBindingProducer
from degs.core import StrictEmbeddingAdapter
from degs.eir_graph import CanonicalResolution, compile_eir_experience_graph
from degs.episode_learning import LearnedExperienceNode, LearningDelta
from degs.section_graph import ExperienceNode, IOContract
from degs.state_store import EIRStateStore
from degs_skill2bench.contract import Skill2BenchProtocol
from degs_skill2bench.eir_dynamic import run_dynamic_training


class _Embedding:
    endpoint = "synthetic"

    def embed(self, request):
        return {
            "model": request["model"],
            "data": [
                {"index": index, "embedding": [1.0, float(index + 1)]}
                for index, _text in enumerate(request["input"])
            ],
        }


class _BindingLLM:
    @property
    def protocol_identity(self):
        return {"kind": "synthetic"}

    async def complete_json_async(self, **_kwargs):
        raise AssertionError("an empty graph must not call binding")


class _Reflection:
    async def produce(self, *, episode, active_experiences):
        assert not active_experiences
        node = ExperienceNode(
            "Answer the independent target Step using its explicit condition.",
            ("The target Step states the required condition.",),
            (IOContract("target step", "The independent Step question."),),
            (IOContract("answer", "The verified Step answer."),),
        )
        return LearningDelta(
            episode.episode_id,
            episode.read_snapshot_id,
            (),
            (LearnedExperienceNode(node, ("trace:original:event:0", "verifier:original:step")),),
            (),
            (),
            (),
            (),
        )


class _Resolver:
    async def resolve(self, *, source_node_id, experience, active):
        return CanonicalResolution(None, experience, "synthetic singleton")


async def _run_population(*, tasks, **_kwargs):
    rollouts = [
        {
            "instance_id": task["instance_id"],
            "react_steps": [{"thought": "Step 1: answer", "action": "answer"}],
        }
        for task in tasks
    ]
    evaluations = [
        {
            "instance_id": task["instance_id"],
            "num_steps": 1,
            "steps": [{"step": 1, "status": "scored", "score": 1.0}],
        }
        for task in tasks
    ]
    return rollouts, evaluations


async def _collect_evidence(*, tasks, **_kwargs):
    return {index: [] for index in range(len(tasks))}


def test_skill2bench_eight_task_batch_uses_one_frozen_eir_graph(tmp_path: Path):
    asyncio.run(_case(tmp_path))


async def _case(tmp_path: Path) -> None:
    protocol = Skill2BenchProtocol(
        profile="9b",
        model="Qwen3.5-9B-AWQ",
        train_count=8,
        test_count=1,
    )
    tasks = tuple(
        {
            "instance_id": f"task-{index}",
            "scenario": "Independent scenario.",
            "steps": [{"question": f"Answer item {index}."}],
        }
        for index in range(8)
    )
    with EIRStateStore(
        tmp_path / "state.sqlite3", dataset_contract=protocol.graph_contract
    ) as state:
        head = await run_dynamic_training(
            train_tasks=tasks,
            root=tmp_path,
            state=state,
            embedding=StrictEmbeddingAdapter(_Embedding()),
            binding=ContextualBindingProducer(_BindingLLM()),
            reflection=_Reflection(),
            resolver=_Resolver(),
            run_population=_run_population,
            collect_evidence=_collect_evidence,
            population_kwargs={},
            evidence_kwargs={},
            protocol=protocol,
        )
        graph, versions = compile_eir_experience_graph(state, snapshot_id=head)
        assert len(graph.nodes) == 1
        assert state.connection.execute(
            "SELECT COUNT(*) FROM eir_source_nodes"
        ).fetchone()[0] == 8
        assert set(versions.values()) == {1}
