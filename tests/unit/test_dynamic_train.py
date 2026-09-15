from __future__ import annotations

import asyncio
import os
from pathlib import Path
import subprocess
import sys

import pytest

from degs.contextual_binding import BindingCondition, ExperienceExpectation
from degs.core import StrictEmbeddingAdapter
from degs.dynamic_train import DynamicTrainCampaign, PreparedEpisode, _bind_protocol
from degs.eir_graph import CanonicalResolution
from degs.episode_evidence import EvidenceItem, EpisodeEvidence, EpisodeOutcome
from degs.episode_learning import (
    ExperienceUpdate,
    LearnedExperienceNode,
    LearningDelta,
    UpdateAction,
)
from degs.graph_dataset_contract import GraphDatasetContract
from degs.section_graph import ExperienceNode, IOContract
from degs.state_store import EIRStateStore


class _EmbeddingTransport:
    def embed(self, request):
        return {
            "model": request["model"],
            "data": [
                {"index": index, "embedding": [1.0, float((len(text) % 17) + 1)]}
                for index, text in enumerate(request["input"])
            ],
        }


class _Resolver:
    def __init__(self, store=None) -> None:
        self.store = store

    async def resolve(self, *, source_node_id, experience, active):
        if self.store is not None:
            assert not self.store.connection.in_transaction
        return CanonicalResolution(None, experience, "Distinct synthetic operation.")


class _Binding:
    async def produce(
        self,
        *,
        request_id,
        payload,
        anchor_versions,
        observable_evidence_ids,
    ):
        return tuple(
            ExperienceExpectation(
                canonical_id,
                version,
                BindingCondition.CONFLICT,
                ("query:0",),
                "This operation is not needed by the synthetic task.",
                (),
                "",
                "No use is expected.",
            )
            for canonical_id, version in anchor_versions.items()
        )


class _Adapter:
    def __init__(self) -> None:
        self.read_snapshots: dict[int, str] = {}

    async def prepare(self, task):
        return PreparedEpisode(
            task,
            f"task-{task}",
            f"Perform synthetic operation {task}.",
            (EvidenceItem("context:0", "synthetic", f"value-{task}"),),
            task,
        )

    async def execute_batch(
        self,
        *,
        prepared,
        read_snapshot_id,
        retrievals,
        expectations,
        guidance,
    ):
        rows = []
        for index, item in enumerate(prepared):
            self.read_snapshots[item.train_index] = read_snapshot_id
            rows.append(EpisodeEvidence(
                episode_id=f"episode-{item.train_index}",
                dataset_contract_id="synthetic-eir",
                train_index=item.train_index,
                task_id=item.task_id,
                read_snapshot_id=read_snapshot_id,
                query_text=item.query_text,
                observable_context=item.observable_context,
                retrieval_context=retrievals[index].to_dict(),
                expectations=tuple(expectations[index]),
                original_trace=(
                    EvidenceItem("trace:original:turn:1:action", "action", "Applied operation."),
                ),
                original_verifier=(
                    EvidenceItem("verifier:original:overall", "verifier_success", "pass"),
                ),
                outcome=EpisodeOutcome.ORIGINAL_SUCCESS,
            ))
        return tuple(rows)


class _FailingBatchAdapter(_Adapter):
    async def execute_batch(self, **kwargs):
        raise RuntimeError("systemic evaluator unavailable")


class _Reflection:
    async def produce(self, *, episode, active_experiences):
        if episode.read_snapshot_id == "G0":
            node = ExperienceNode(
                f"Perform operation {episode.train_index}.",
                (f"The task requests operation {episode.train_index}.",),
                (IOContract("input", "Current task input."),),
                (IOContract("output", "Requested task state."),),
            )
            return LearningDelta(
                episode.episode_id,
                episode.read_snapshot_id,
                (),
                (
                    LearnedExperienceNode(
                        node,
                        ("trace:original:turn:1:action", "verifier:original:overall"),
                    ),
                ),
                (),
                (),
                (),
                (),
            )
        return LearningDelta(
            episode.episode_id,
            episode.read_snapshot_id,
            tuple(
                ExperienceUpdate(
                    row.canonical_id,
                    row.canonical_version,
                    UpdateAction.NO_EVIDENCE,
                    (),
                    (),
                    (),
                    "The retrieved hypothesis was not used.",
                    None,
                )
                for row in episode.expectations
            ),
            (),
            (),
            (),
            (),
            (),
        )


def test_dynamic_campaign_freezes_each_eight_task_batch(tmp_path: Path) -> None:
    asyncio.run(_run_campaign(tmp_path))


def test_dynamic_protocol_resume_requires_exact_identity(tmp_path: Path) -> None:
    path = tmp_path / "dynamic_protocol.json"
    _bind_protocol(path, {"format": "test", "version": 1})
    _bind_protocol(path, {"format": "test", "version": 1})
    with pytest.raises(ValueError, match="fresh run directory"):
        _bind_protocol(path, {"format": "test", "version": 2})


def test_dynamic_cli_selects_27b_before_replay_imports() -> None:
    code = """
import os
os.environ.pop('DEGS_MODEL', None)
from degs.dynamic_train import main
try:
    main([
        '--dataset-path', '/missing-dataset',
        '--run-dir', '/missing-run',
        '--runtime-root', '/missing-runtime',
        '--generation-base-url', 'http://generation.test/v1',
        '--embedding-base-url', 'http://embedding.test/v1',
        '--model', 'Qwen3.5-27B-AWQ',
    ])
except ValueError as exc:
    assert 'API keys are required' in str(exc)
from degs.source_replay import SOURCE_REPLAY_MODEL
print(SOURCE_REPLAY_MODEL)
"""
    environment = dict(os.environ)
    environment.pop("DEGS_MODEL", None)
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.stdout.strip() == "Qwen3.5-27B-AWQ"


def test_batch_adapter_failure_does_not_publish_an_empty_snapshot(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        contract = GraphDatasetContract(
            identity="synthetic-eir-systemic",
            source_split="train[0,8)",
            train_count=8,
            batch_size=8,
        )
        with EIRStateStore(
            tmp_path / "state.sqlite3", dataset_contract=contract
        ) as store:
            campaign = DynamicTrainCampaign(
                store=store,
                contract=contract,
                embedding=StrictEmbeddingAdapter(_EmbeddingTransport()),
                binding=_Binding(),
                reflection=_Reflection(),
                resolver=_Resolver(store),
                adapter=_FailingBatchAdapter(),
                output_dir=tmp_path / "run",
            )
            with pytest.raises(RuntimeError, match="systemic evaluator unavailable"):
                await campaign.run(tuple(range(8)))
            assert store.head_snapshot_id is None
            assert store.committed_snapshot_for_batch(0) is None

    asyncio.run(run())


async def _run_campaign(tmp_path: Path) -> None:
    contract = GraphDatasetContract(
        identity="synthetic-eir",
        source_split="train[0,16)",
        train_count=16,
        batch_size=8,
    )
    adapter = _Adapter()
    with EIRStateStore(
        tmp_path / "state.sqlite3", dataset_contract=contract
    ) as store:
        campaign = DynamicTrainCampaign(
            store=store,
            contract=contract,
            embedding=StrictEmbeddingAdapter(_EmbeddingTransport()),
            binding=_Binding(),
            reflection=_Reflection(),
            resolver=_Resolver(store),
            adapter=adapter,
            output_dir=tmp_path / "run",
        )
        results = await campaign.run(tuple(range(16)))
        assert len(results) == 2
        assert {adapter.read_snapshots[index] for index in range(8)} == {"G0"}
        assert {adapter.read_snapshots[index] for index in range(8, 16)} == {
            results[0].snapshot_id
        }
        assert [row.train_index for row in results[0].commit_audits] == list(range(8))
        assert [row.train_index for row in results[1].commit_audits] == list(range(8, 16))
        assert len(results[0].graph.nodes) == 8
        assert len(results[1].graph.nodes) == 8
        assert (tmp_path / "run/batches/batch_01/experience_graph.json").is_file()
        resumed = await campaign.run(tuple(range(16)))
        assert [row.snapshot_id for row in resumed] == [
            row.snapshot_id for row in results
        ]
        assert store.head_snapshot_id == results[-1].snapshot_id
