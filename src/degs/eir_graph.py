from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
from typing import Any, Callable, Mapping, Protocol, Sequence

from .core import canonical_json_bytes
from .episode_evidence import EpisodeEvidence, EpisodeOutcome
from .episode_learning import (
    LearningDelta,
    ProcedureStepKind,
    UpdateAction,
)
from .section_graph import (
    CanonicalExperience,
    CanonicalNode,
    ExperienceGraph,
    ExperienceNode,
    ProjectedEdge,
    _canonical_document,
)
from .state_store import ActiveCanonicalVersion, EIRStateStore, _eir_stable_canonical_id


EIR_GRAPH_FORMAT = "degs_eir_experience_graph_v1"


@dataclass(frozen=True)
class CanonicalResolution:
    target_canonical_id: str | None
    experience: ExperienceNode
    basis: str

    def __post_init__(self) -> None:
        if (
            self.target_canonical_id is not None
            and (type(self.target_canonical_id) is not str or not self.target_canonical_id)
        ) or type(self.experience) is not ExperienceNode or type(self.basis) is not str or not self.basis:
            raise ValueError("Canonical resolution differs")


class CanonicalResolver(Protocol):
    async def resolve(
        self,
        *,
        source_node_id: str,
        experience: ExperienceNode,
        active: Sequence[ActiveCanonicalVersion],
    ) -> CanonicalResolution: ...


@dataclass(frozen=True)
class EpisodeCommitAudit:
    episode_id: str
    train_index: int
    status: str
    update_counts: Mapping[str, int]
    new_source_node_count: int
    new_canonical_count: int
    absorbed_node_count: int
    procedure_edge_count: int
    discarded: tuple[str, ...]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "train_index": self.train_index,
            "status": self.status,
            "update_counts": dict(self.update_counts),
            "new_source_node_count": self.new_source_node_count,
            "new_canonical_count": self.new_canonical_count,
            "absorbed_node_count": self.absorbed_node_count,
            "procedure_edge_count": self.procedure_edge_count,
            "discarded": list(self.discarded),
            "error": self.error,
        }


@dataclass(frozen=True)
class _PlannedNodeResolution:
    source_node_id: str
    exact: bool
    resolution: CanonicalResolution


def source_node_id(episode_id: str, node_index: int, node: ExperienceNode) -> str:
    body = {
        "episode_id": episode_id,
        "node_index": node_index,
        "experience": node.to_dict(),
    }
    return "source_" + hashlib.sha256(canonical_json_bytes(body)).hexdigest()[:24]


def _event_id(episode_id: str, kind: str, ordinal: int, body: Mapping[str, Any]) -> str:
    identity = {
        "episode_id": episode_id,
        "kind": kind,
        "ordinal": ordinal,
        "body": dict(body),
    }
    return "event_" + hashlib.sha256(canonical_json_bytes(identity)).hexdigest()[:24]


class LearningDeltaApplier:
    """Apply already-validated episode deltas in deterministic train order."""

    def __init__(self, store: EIRStateStore, resolver: CanonicalResolver) -> None:
        self.store = store
        self.resolver = resolver

    async def apply_batch(
        self,
        *,
        snapshot_id: str,
        episodes: Sequence[tuple[EpisodeEvidence, LearningDelta | None]],
        learning_errors: Mapping[str, str] | None = None,
        commit_manifest_factory: Callable[
            [tuple[EpisodeCommitAudit, ...]], Mapping[str, Any]
        ]
        | None = None,
    ) -> tuple[EpisodeCommitAudit, ...]:
        failures = dict(learning_errors or {})
        ordered = tuple(sorted(episodes, key=lambda row: row[0].train_index))
        if len({row[0].train_index for row in ordered}) != len(ordered):
            raise ValueError("EIR batch train indices differ")
        read_snapshots = {row[0].read_snapshot_id for row in ordered}
        if len(read_snapshots) > 1:
            raise ValueError("EIR batch did not read one frozen snapshot")
        resolutions = await self._plan_node_resolutions(ordered)
        audits: list[EpisodeCommitAudit] = []
        with self.store.transaction():
            for episode, delta in ordered:
                self.store.register_episode(
                    snapshot_id=snapshot_id,
                    episode_id=episode.episode_id,
                    train_index=episode.train_index,
                    task_id=episode.task_id,
                    read_snapshot_id=episode.read_snapshot_id,
                    outcome=episode.outcome.value,
                    evidence=episode.to_learning_payload(),
                    expectations=[row.to_dict() for row in episode.expectations],
                )
                if delta is None:
                    error = failures.get(episode.episode_id)
                    if error is not None:
                        self.store.record_learning_delta(
                            episode_id=episode.episode_id,
                            status="ITEM_LOCAL_FAILURE",
                            error=error,
                        )
                        audits.append(
                            EpisodeCommitAudit(
                                episode.episode_id,
                                episode.train_index,
                                "ITEM_LOCAL_FAILURE",
                                {},
                                0,
                                0,
                                0,
                                0,
                                (),
                                error,
                            )
                        )
                        continue
                    if episode.outcome is not EpisodeOutcome.ITEM_LOCAL_RUNTIME_FAILURE:
                        raise ValueError("observable EIR episode lacks a LearningDelta")
                    self.store.record_learning_delta(
                        episode_id=episode.episode_id,
                        status="EPISODE_UNOBSERVABLE",
                        error="episode has no interpretable action/outcome evidence",
                    )
                    audits.append(
                        EpisodeCommitAudit(
                            episode.episode_id,
                            episode.train_index,
                            "EPISODE_UNOBSERVABLE",
                            {},
                            0,
                            0,
                            0,
                            0,
                            (),
                        )
                    )
                    continue
                if delta.episode_id != episode.episode_id or delta.base_snapshot_id != episode.read_snapshot_id:
                    raise ValueError("LearningDelta episode binding differs")
                audit = self._apply_episode(
                    snapshot_id=snapshot_id,
                    episode=episode,
                    delta=delta,
                    resolutions=resolutions,
                )
                self.store.record_learning_delta(
                    episode_id=episode.episode_id,
                    status="VALID",
                    delta=delta.to_dict(),
                )
                audits.append(audit)
            frozen_audits = tuple(audits)
            if commit_manifest_factory is not None:
                self.store.commit_snapshot(
                    snapshot_id,
                    manifest=commit_manifest_factory(frozen_audits),
                )
        return tuple(audits)

    @staticmethod
    def _virtual_version(
        canonical_id: str,
        version: int,
        experience: ExperienceNode,
    ) -> ActiveCanonicalVersion:
        document = _canonical_document(
            CanonicalExperience(
                experience.operation,
                experience.applicability,
                experience.inputs,
                experience.outputs,
            )
        )
        return ActiveCanonicalVersion(
            canonical_id,
            version,
            experience,
            document,
            hashlib.sha256(document.encode("utf-8")).hexdigest(),
        )

    async def _plan_node_resolutions(
        self,
        ordered: Sequence[tuple[EpisodeEvidence, LearningDelta | None]],
    ) -> Mapping[tuple[str, int], _PlannedNodeResolution]:
        """Resolve semantic candidates without holding a database transaction."""
        active = {
            row.canonical_id: row for row in self.store.active_canonicals()
        }
        plans: dict[tuple[str, int], _PlannedNodeResolution] = {}
        for episode, delta in ordered:
            if delta is None:
                continue
            for update in delta.updates:
                if update.action not in {UpdateAction.QUALIFY, UpdateAction.CORRECT}:
                    continue
                assert update.revised_experience is not None
                canonical_id = self.store.resolve_canonical_id(update.canonical_id)
                current = active[canonical_id]
                if current.version == update.base_version:
                    active[canonical_id] = self._virtual_version(
                        canonical_id,
                        current.version + 1,
                        update.revised_experience,
                    )
                elif canonical_json_bytes(current.experience.to_dict()) == canonical_json_bytes(
                    update.revised_experience.to_dict()
                ):
                    continue
            for node_index, learned in enumerate(delta.new_nodes):
                leaf_id = source_node_id(
                    episode.episode_id, node_index, learned.experience
                )
                exact_ids = sorted(
                    canonical_id
                    for canonical_id, row in active.items()
                    if canonical_json_bytes(row.experience.to_dict())
                    == canonical_json_bytes(learned.experience.to_dict())
                )
                if exact_ids:
                    resolution = CanonicalResolution(
                        exact_ids[0], learned.experience, "Exact active experience."
                    )
                    plans[(episode.episode_id, node_index)] = _PlannedNodeResolution(
                        leaf_id, True, resolution
                    )
                    continue
                resolution = await self.resolver.resolve(
                    source_node_id=leaf_id,
                    experience=learned.experience,
                    active=tuple(active[key] for key in sorted(active)),
                )
                plans[(episode.episode_id, node_index)] = _PlannedNodeResolution(
                    leaf_id, False, resolution
                )
                if resolution.target_canonical_id is None:
                    canonical_id = _eir_stable_canonical_id(leaf_id)
                    active[canonical_id] = self._virtual_version(
                        canonical_id, 1, resolution.experience
                    )
                else:
                    canonical_id = self.store.resolve_canonical_id(
                        resolution.target_canonical_id
                    )
                    current = active[canonical_id]
                    active[canonical_id] = self._virtual_version(
                        canonical_id,
                        current.version + 1,
                        resolution.experience,
                    )
        return plans

    def _apply_episode(
        self,
        *,
        snapshot_id: str,
        episode: EpisodeEvidence,
        delta: LearningDelta,
        resolutions: Mapping[tuple[str, int], _PlannedNodeResolution],
    ) -> EpisodeCommitAudit:
        counts: defaultdict[str, int] = defaultdict(int)
        discarded = list(delta.discarded_edge_reasons)
        for ordinal, update in enumerate(delta.updates):
            counts[update.action.value] += 1
            evidence = update.to_dict()
            event_id = _event_id(episode.episode_id, "update", ordinal, evidence)
            if update.action in {UpdateAction.NO_EVIDENCE, UpdateAction.SUPPORT}:
                self.store.record_experience_event(
                    event_id=event_id,
                    episode_id=episode.episode_id,
                    canonical_id=update.canonical_id,
                    base_version=update.base_version,
                    action=update.action.value,
                    evidence=evidence,
                    snapshot_id=snapshot_id,
                )
                continue
            assert update.revised_experience is not None
            applied = self.store.revise_canonical(
                canonical_id=update.canonical_id,
                base_version=update.base_version,
                experience=update.revised_experience,
                snapshot_id=snapshot_id,
                change_kind=update.action.value,
                evidence_event_id=event_id,
                episode_id=episode.episode_id,
            )
            if applied is None:
                discarded.append(
                    f"{update.canonical_id} revision deferred after a same-batch version conflict"
                )

        node_to_canonical: dict[int, str] = {}
        created = 0
        absorbed = 0
        for node_index, learned in enumerate(delta.new_nodes):
            plan = resolutions[(episode.episode_id, node_index)]
            leaf_id = plan.source_node_id
            self.store.add_source_node(
                source_node_id=leaf_id,
                episode_id=episode.episode_id,
                node_index=node_index,
                experience=learned.experience,
                evidence_refs=learned.evidence_refs,
                snapshot_id=snapshot_id,
            )
            resolution = plan.resolution
            if plan.exact:
                assert resolution.target_canonical_id is not None
                event_id = _event_id(
                    episode.episode_id,
                    "absorb_exact",
                    node_index,
                    {
                        "source_node_id": leaf_id,
                        "canonical_id": resolution.target_canonical_id,
                    },
                )
                self.store.absorb_source_node(
                    canonical_id=resolution.target_canonical_id,
                    source_node_id=leaf_id,
                    experience=learned.experience,
                    snapshot_id=snapshot_id,
                    evidence_event_id=event_id,
                    episode_id=episode.episode_id,
                    exact=True,
                )
                node_to_canonical[node_index] = self.store.resolve_canonical_id(
                    resolution.target_canonical_id
                )
                absorbed += 1
                continue
            if resolution.target_canonical_id is None:
                if resolution.basis.startswith("Canonical resolution failed item-locally:"):
                    discarded.append(resolution.basis)
                canonical_id = self.store.create_canonical(
                    source_node_id=leaf_id,
                    experience=resolution.experience,
                    snapshot_id=snapshot_id,
                    change_kind="CREATE",
                    evidence_event_id=None,
                )
                created += 1
            else:
                event_id = _event_id(
                    episode.episode_id,
                    "merge",
                    node_index,
                    {
                        "source_node_id": leaf_id,
                        "target": resolution.target_canonical_id,
                        "basis": resolution.basis,
                    },
                )
                self.store.absorb_source_node(
                    canonical_id=resolution.target_canonical_id,
                    source_node_id=leaf_id,
                    experience=resolution.experience,
                    snapshot_id=snapshot_id,
                    evidence_event_id=event_id,
                    episode_id=episode.episode_id,
                    exact=False,
                )
                canonical_id = self.store.resolve_canonical_id(
                    resolution.target_canonical_id
                )
                absorbed += 1
            node_to_canonical[node_index] = canonical_id

        procedure_ids: list[str] = []
        for step in delta.procedure_steps:
            if step.kind is ProcedureStepKind.CANONICAL:
                assert step.canonical_id is not None
                procedure_ids.append(self.store.resolve_canonical_id(step.canonical_id))
            else:
                assert step.node_index is not None
                procedure_ids.append(node_to_canonical[step.node_index])
        edge_pairs = {
            (node_to_canonical[edge.source], node_to_canonical[edge.target])
            for edge in delta.new_edges
        }
        edge_pairs.update(
            (procedure_ids[edge.source], procedure_ids[edge.target])
            for edge in delta.procedure_edges
        )
        accepted_edges = 0
        for source, target in sorted(edge_pairs):
            if self.store.add_procedure_edge(
                episode_id=episode.episode_id,
                source_canonical_id=source,
                target_canonical_id=target,
                snapshot_id=snapshot_id,
            ):
                accepted_edges += 1
            else:
                discarded.append(f"self-edge {source}->{target} removed after Canonical fusion")
        return EpisodeCommitAudit(
            episode.episode_id,
            episode.train_index,
            "COMMITTED",
            dict(sorted(counts.items())),
            len(delta.new_nodes),
            created,
            absorbed,
            accepted_edges,
            tuple(discarded),
        )


def compile_eir_experience_graph(
    store: EIRStateStore,
    *,
    snapshot_id: str,
) -> tuple[ExperienceGraph, Mapping[str, int]]:
    versions = store.active_canonicals(snapshot_id=snapshot_id)
    nodes: list[CanonicalNode] = []
    version_by_id: dict[str, int] = {}
    for row in versions:
        experience = CanonicalExperience(
            row.experience.operation,
            row.experience.applicability,
            row.experience.inputs,
            row.experience.outputs,
        )
        nodes.append(
            CanonicalNode(
                row.canonical_id,
                experience,
                row.document,
                row.document_sha256,
            )
        )
        version_by_id[row.canonical_id] = row.version
    edge_support: defaultdict[tuple[str, str], set[int]] = defaultdict(set)
    for source, target, train_index, _episode_id in store.procedure_edge_rows(
        snapshot_id=snapshot_id
    ):
        edge_support[(source, target)].add(train_index)
    edges = tuple(
        ProjectedEdge(source, target, tuple(sorted(train_indices)))
        for (source, target), train_indices in sorted(edge_support.items())
    )
    nodes_tuple = tuple(sorted(nodes, key=lambda row: row.canonical_id))
    source_identity = hashlib.sha256(
        canonical_json_bytes(
            {
                "format": EIR_GRAPH_FORMAT,
                "snapshot_id": snapshot_id,
                "episodes": [
                    row[0]
                    for row in store.connection.execute(
                        """
                        SELECT episode.episode_id
                        FROM eir_episodes AS episode
                        JOIN eir_snapshots AS snapshot
                          ON snapshot.snapshot_id = episode.snapshot_id
                        WHERE snapshot.sequence <= (
                            SELECT sequence FROM eir_snapshots WHERE snapshot_id = ?
                        )
                        ORDER BY episode.train_index, episode.episode_id
                        """,
                        (snapshot_id,),
                    )
                ],
            }
        )
    ).hexdigest()
    partition_identity = hashlib.sha256(
        canonical_json_bytes(
            {
                "format": "degs_eir_active_partition_v1",
                "snapshot_id": snapshot_id,
                "canonical_versions": version_by_id,
            }
        )
    ).hexdigest()
    body = {
        "format": EIR_GRAPH_FORMAT,
        "snapshot_id": snapshot_id,
        "section_graphs_sha256": source_identity,
        "canonical_partition_sha256": partition_identity,
        "canonical_versions": version_by_id,
        "nodes": [row.to_dict() for row in nodes_tuple],
        "edges": [row.to_dict() for row in edges],
    }
    graph_sha = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    return (
        ExperienceGraph(
            nodes_tuple,
            edges,
            source_identity,
            partition_identity,
            graph_sha,
            graph_format=EIR_GRAPH_FORMAT,
            snapshot_id=snapshot_id,
            canonical_versions=version_by_id,
        ),
        version_by_id,
    )


__all__ = [
    "CanonicalResolution",
    "CanonicalResolver",
    "EIR_GRAPH_FORMAT",
    "EpisodeCommitAudit",
    "LearningDeltaApplier",
    "compile_eir_experience_graph",
    "source_node_id",
]
