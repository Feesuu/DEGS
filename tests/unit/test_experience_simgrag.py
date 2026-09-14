from __future__ import annotations

from dataclasses import replace

import pytest

from degs.core import normalize_embedding_text
from degs.experience_simgrag import (
    _recover_occurrence_path,
    build_retrieval_index,
    canonical_retrieval_document,
    need_retrieval_document,
    retrieve_experience_subgraphs,
)
from degs.section_graph import (
    CanonicalExperience,
    CanonicalGroup,
    CanonicalNode,
    CanonicalPartition,
    ExperienceEdge,
    ExperienceGraph,
    ExperienceNode,
    IOContract,
    ProjectedEdge,
    SectionGraphSource,
    WorkflowGraph,
    _canonical_id,
)
from degs.workflow_retrieval import (
    NeedEdge,
    NeedGraph,
    NeedNode,
    SubgraphSelection,
    WorkflowRecall,
    finalize_subgraph_selection,
    parse_selector_response,
    render_workflow_fallback,
    selector_payload,
    selector_response_schema,
    workflow_fallback_candidates,
    workflow_fallback_selector_payload,
)


def _contract(description: str = "table") -> IOContract:
    return IOContract("artifact", description)


def _experience(
    operation: str, applicability: str = "the task applies"
) -> CanonicalExperience:
    return CanonicalExperience(
        operation,
        (applicability,),
        (_contract("input table"),),
        (_contract("output table"),),
    )


def _fixture() -> tuple[SectionGraphSource, CanonicalPartition, ExperienceGraph]:
    members = (((0, 0),), ((0, 1),), ((0, 2),), ((1, 0),))
    experiences = (
        _experience("inspect source"),
        _experience("prepare intermediate"),
        _experience("write result"),
        _experience("unrelated operation"),
    )
    groups = tuple(
        CanonicalGroup(member, experience)
        for member, experience in zip(members, experiences, strict=True)
    )
    partition = CanonicalPartition(groups, "1" * 64, "2" * 64)
    ids = tuple(_canonical_id(group.members) for group in groups)
    graph = ExperienceGraph(
        tuple(
            CanonicalNode(
                cid,
                experience,
                canonical_retrieval_document(experience),
                f"{index:064x}",
            )
            for index, (cid, experience) in enumerate(
                zip(ids, experiences, strict=True), 1
            )
        ),
        (
            ProjectedEdge(ids[0], ids[1], (0,)),
            ProjectedEdge(ids[1], ids[2], (0,)),
        ),
        "1" * 64,
        "2" * 64,
        "3" * 64,
    )
    workflows = (
        WorkflowGraph(
            0,
            "task-0",
            "inspect then write",
            tuple(
                ExperienceNode(
                    row.operation, row.applicability, row.inputs, row.outputs
                )
                for row in experiences[:3]
            ),
            (ExperienceEdge(0, 1), ExperienceEdge(1, 2)),
        ),
        WorkflowGraph(
            1,
            "task-1",
            "unrelated",
            (
                ExperienceNode(
                    experiences[3].operation,
                    experiences[3].applicability,
                    experiences[3].inputs,
                    experiences[3].outputs,
                ),
            ),
            (),
        ),
    )
    return SectionGraphSource(workflows, "1" * 64), partition, graph


def _need_graph(*, reversed_edge: bool = False) -> NeedGraph:
    inspect = NeedNode(
        "inspect source",
        ("the task applies",),
        (_contract("query-specific input unlike source"),),
        (_contract("query-specific evidence unlike source"),),
    )
    write = NeedNode(
        "write result",
        ("a deliberately different applicability condition",),
        (_contract("query-specific value unlike source"),),
        (_contract("query-specific output unlike source"),),
    )
    nodes = (write, inspect) if reversed_edge else (inspect, write)
    return NeedGraph(nodes, (NeedEdge(0, 1),))


def _vectors(
    need_graph: NeedGraph, graph: ExperienceGraph
) -> dict[str, tuple[float, float]]:
    values: dict[str, tuple[float, float]] = {}
    for need in need_graph.nodes:
        values[normalize_embedding_text(need_retrieval_document(need))] = (
            (1.0, 0.0) if "inspect" in need.description else (0.0, 1.0)
        )
    for node in graph.nodes:
        operation = node.experience.operation
        if operation == "inspect source":
            vector = (1.0, 0.0)
        elif operation == "write result":
            vector = (0.0, 1.0)
        else:
            vector = (-1.0, -1.0)
        values[
            normalize_embedding_text(
                canonical_retrieval_document(node.experience)
            )
        ] = vector
    return values


def _search(need_graph: NeedGraph, *, need_top_k: int = 8):
    source, partition, graph = _fixture()
    vectors = _vectors(need_graph, graph)
    index = build_retrieval_index(graph, partition, source, vectors=vectors)
    result = retrieve_experience_subgraphs(
        need_graph,
        (WorkflowRecall(0, 0.9, 0), WorkflowRecall(1, 0.2, 1)),
        index=index,
        vectors=vectors,
        workflow_context_similarities={0: 0.9, 1: 0.2},
        need_top_k=need_top_k,
    )
    return source, partition, graph, index, result


def test_retrieval_index_requires_exact_source_partition_graph_coverage() -> None:
    source, partition, graph = _fixture()
    needs = _need_graph()
    with pytest.raises(ValueError, match="exactly cover"):
        build_retrieval_index(
            graph,
            replace(partition, groups=partition.groups[:-1]),
            source,
            vectors=_vectors(needs, graph),
        )


def test_workflow_context_late_fusion_has_no_threshold_or_contract_gate() -> None:
    _source, partition, _graph, _index, result = _search(_need_graph())
    assert all(
        len(rows) == len(partition.groups)
        for rows in result.need_candidates.values()
    )
    assert {row.canonical_id for row in result.need_candidates[0]} == {
        _canonical_id(group.members) for group in partition.groups
    }
    assert not hasattr(result.need_candidates[0][0], "input_matches")
    assert not hasattr(result.need_candidates[0][0], "applicability_similarity")
    assert result.need_candidates[0][0].recall_similarity == pytest.approx(
        (
            result.need_candidates[0][0].operation_similarity
            + result.need_candidates[0][0].workflow_context_similarity
        )
        / 2
    )


def test_workflow_context_reranks_complete_candidates_not_operation_recall() -> None:
    deposit = CanonicalExperience(
        "classify a row from a text condition",
        ("when the source text contains the word deposit",),
        (IOContract("source_text", "transaction description"),),
        (IOContract("label", "Revenue"),),
    )
    closed = CanonicalExperience(
        "classify a row from a text condition",
        ("when a status field equals closed",),
        (IOContract("status", "record status"),),
        (IOContract("label", "Completed"),),
    )
    groups = (
        CanonicalGroup(((0, 0),), deposit),
        CanonicalGroup(((1, 0),), closed),
    )
    partition = CanonicalPartition(groups, "1" * 64, "2" * 64)
    canonical_ids = tuple(_canonical_id(group.members) for group in groups)
    graph = ExperienceGraph(
        tuple(
            CanonicalNode(
                canonical_id,
                experience,
                canonical_retrieval_document(experience),
                f"{index:064x}",
            )
            for index, (canonical_id, experience) in enumerate(
                zip(canonical_ids, (deposit, closed), strict=True), 1
            )
        ),
        (),
        "1" * 64,
        "2" * 64,
        "3" * 64,
    )
    source = SectionGraphSource(
        tuple(
            WorkflowGraph(
                index,
                f"task-{index}",
                experience.operation,
                (
                    ExperienceNode(
                        experience.operation,
                        experience.applicability,
                        experience.inputs,
                        experience.outputs,
                    ),
                ),
                (),
            )
            for index, experience in enumerate((deposit, closed))
        ),
        "1" * 64,
    )
    need = NeedNode(
        "classify a row from a text condition",
        ("when the source text contains the word deposit",),
        (IOContract("source_text", "transaction description"),),
        (IOContract("label", "Revenue"),),
    )
    need_graph = NeedGraph((need,), ())
    vectors = {
        normalize_embedding_text(need_retrieval_document(need)): (1.0, 0.0),
        normalize_embedding_text(canonical_retrieval_document(deposit)): (
            1.0,
            0.0,
        ),
    }
    result = retrieve_experience_subgraphs(
        need_graph,
        (),
        index=build_retrieval_index(
            graph, partition, source, vectors=vectors
        ),
        vectors=vectors,
        workflow_context_similarities={0: 0.2, 1: 0.9},
        need_top_k=2,
        result_top_k=2,
    )
    assert tuple(row.canonical_id for row in result.need_candidates[0]) == tuple(
        sorted(canonical_ids)
    )
    selected = result.candidates[0]
    assert selected.need_mapping == ((0, canonical_ids[1]),)
    assert selected.operation_loss == pytest.approx(0.0)
    assert selected.workflow_context_loss == pytest.approx(0.1)


def test_workflow_context_is_soft_and_cannot_veto_a_strong_operation() -> None:
    source, partition, graph = _fixture()
    need_graph = NeedGraph((_need_graph().nodes[0],), ())
    vectors = _vectors(need_graph, graph)
    result = retrieve_experience_subgraphs(
        need_graph,
        (WorkflowRecall(1, 1.0, 0), WorkflowRecall(0, 0.2, 1)),
        index=build_retrieval_index(graph, partition, source, vectors=vectors),
        vectors=vectors,
        workflow_context_similarities={0: 0.2, 1: 1.0},
        need_top_k=1,
        result_top_k=1,
    )
    assert result.need_candidates[0][0].canonical_id == _canonical_id(
        partition.groups[0].members
    )


def test_each_need_keeps_exactly_min_eight_canonical_candidates() -> None:
    experience = _experience("inspect source")
    groups = tuple(CanonicalGroup(((index, 0),), experience) for index in range(10))
    partition = CanonicalPartition(groups, "1" * 64, "2" * 64)
    ids = tuple(_canonical_id(group.members) for group in groups)
    graph = ExperienceGraph(
        tuple(
            CanonicalNode(cid, experience, experience.operation, f"{i + 1:064x}")
            for i, cid in enumerate(ids)
        ),
        (),
        "1" * 64,
        "2" * 64,
        "3" * 64,
    )
    source = SectionGraphSource(
        tuple(
            WorkflowGraph(
                i,
                f"task-{i}",
                "inspect",
                (
                    ExperienceNode(
                        experience.operation,
                        experience.applicability,
                        experience.inputs,
                        experience.outputs,
                    ),
                ),
                (),
            )
            for i in range(10)
        ),
        "1" * 64,
    )
    needs = NeedGraph((_need_graph().nodes[0],), ())
    vectors = _vectors(needs, graph)
    result = retrieve_experience_subgraphs(
        needs,
        (),
        index=build_retrieval_index(graph, partition, source, vectors=vectors),
        vectors=vectors,
        workflow_context_similarities={i: 1.0 for i in range(10)},
    )
    assert len(result.need_candidates[0]) == 8
    assert len(result.candidates) == 8


def test_full_graph_connector_is_recovered_outside_anchor_region() -> None:
    _source, partition, _graph, _index, result = _search(
        _need_graph(), need_top_k=1
    )
    inspect_id, connector_id, write_id = (
        _canonical_id(group.members) for group in partition.groups[:3]
    )
    candidate = result.candidates[0]
    assert result.region_node_ids == tuple(sorted((inspect_id, write_id)))
    assert candidate.canonical_node_ids == tuple(
        sorted((inspect_id, connector_id, write_id))
    )
    assert candidate.dependency_witnesses[0].canonical_path == (
        inspect_id,
        connector_id,
        write_id,
    )
    assert candidate.dependency_witnesses[0].trace_realizable
    assert candidate.unsatisfied_need_edges == ()


def test_unreachable_need_edge_is_soft_and_all_needs_remain_mapped() -> None:
    _source, _partition, _graph, _index, result = _search(
        _need_graph(reversed_edge=True), need_top_k=1
    )
    candidate = result.candidates[0]
    assert len(candidate.need_mapping) == 2
    assert candidate.unmatched_need_indices == ()
    assert candidate.satisfied_need_edge_count == 0
    assert candidate.unsatisfied_need_edges == ((0, 1),)
    assert candidate.structural_loss == 1.0


def test_same_canonical_can_fill_multiple_need_nodes() -> None:
    need = _need_graph().nodes[0]
    _source, _partition, _graph, _index, result = _search(
        NeedGraph((need, need), ()), need_top_k=1
    )
    mapping = result.candidates[0].need_mapping
    assert len(mapping) == 2
    assert mapping[0][1] == mapping[1][1]


def test_beam_search_is_bounded_and_deterministic() -> None:
    needs = NeedGraph((_need_graph().nodes[0],) * 5, ())
    first = _search(needs)[-1]
    second = _search(needs)[-1]
    assert first.metrics["partial_beam_k"] == 32
    assert first.metrics["leaf_mappings"] <= 32
    assert first.metrics["online_pair_enumerations"] == 0
    assert [row.to_dict() for row in first.candidates] == [
        row.to_dict() for row in second.candidates
    ]


def test_complete_candidates_restore_additive_semantic_structural_rank() -> None:
    result = _search(_need_graph(reversed_edge=True))[-1]
    keys = [
        (
            -candidate.matched_need_count,
            candidate.total_cost,
        )
        for candidate in result.candidates
    ]
    assert keys == sorted(keys)


def test_single_workflow_edge_is_valid_provenance_not_a_quality_gate() -> None:
    _source, _partition, _graph, _index, result = _search(
        _need_graph(), need_top_k=1
    )
    witness = result.candidates[0].dependency_witnesses[0]
    assert witness.supporting_workflow_ids == (0,)
    assert witness.trace_realizable


def test_selector_is_mandatory_and_payload_contains_source_occurrence() -> None:
    source, _partition, graph, _index, result = _search(
        _need_graph(), need_top_k=1
    )
    candidates = result.candidates
    schema = selector_response_schema(candidates)
    assert schema["properties"]["selected_candidate_id"]["enum"] == ["C0"]
    assert parse_selector_response(
        {"selected_candidate_id": "C0"}, candidates=candidates
    ) == SubgraphSelection("C0")
    with pytest.raises(ValueError, match="unverified"):
        parse_selector_response(
            {"selected_candidate_id": None}, candidates=candidates
        )
    payload = selector_payload(
        query_text="inspect then write",
        need_graph=_need_graph(),
        workflow_recalls=(WorkflowRecall(0, 1.0, 0),),
        candidates=candidates,
        graph=graph,
        source=source,
    )
    assert set(payload) == {
        "query",
        "need_graph",
        "workflow_recalls",
        "candidates",
    }
    assert payload["need_graph"]["need_nodes"][0]["description"] == "inspect source"
    assert payload["need_graph"]["need_nodes"][0]["applicability_context"]
    evidence = payload["candidates"][0]["source_occurrence_evidence"]
    assert evidence
    assert evidence[0]["source_experience"]["operation"]


def test_renderer_keeps_need_semantics_alignment_and_source_evidence() -> None:
    needs = _need_graph(reversed_edge=True)
    source, _partition, graph, _index, search = _search(needs, need_top_k=1)
    result = finalize_subgraph_selection(
        SubgraphSelection("C0"),
        query_text="write then inspect",
        need_graph=needs,
        workflow_recalls=(WorkflowRecall(0, 1.0, 0),),
        search_result=search,
        graph=graph,
        source=source,
    )
    assert result.status == "OK"
    assert "NeedNode 0 uses" in result.experience
    assert "Requested micro-operation: write result" in result.experience
    assert (
        "Need applicability: a deliberately different applicability condition"
        in result.experience
    )
    assert "SOURCE OCCURRENCE EVIDENCE" in result.experience
    assert "SOFT STRUCTURAL MISMATCHES" in result.experience


def test_renderer_keeps_connector_as_full_executable_guidance() -> None:
    needs = _need_graph()
    source, _partition, graph, _index, search = _search(needs, need_top_k=1)
    candidate = search.candidates[0]
    assert candidate.connector_count == 1

    result = finalize_subgraph_selection(
        SubgraphSelection("C0"),
        query_text="inspect then write",
        need_graph=needs,
        workflow_recalls=(WorkflowRecall(0, 1.0, 0),),
        search_result=search,
        graph=graph,
        source=source,
    )
    assert "Operation: inspect source" in result.experience
    assert "Operation: write result" in result.experience
    assert "Operation: prepare intermediate" in result.experience
    assert "TEMPLATE X0" not in result.experience
    assert "STRUCTURAL CONNECTORS" not in result.experience
    assert "occurrence (0,1)" in result.experience


def test_need_graph_fallback_selects_and_renders_one_source_workflow() -> None:
    source, partition, graph = _fixture()
    canonical_by_workflow = {
        0: tuple(_canonical_id(group.members) for group in partition.groups[:3]),
        1: (_canonical_id(partition.groups[3].members),),
    }
    candidates = workflow_fallback_candidates(
        (WorkflowRecall(0, 0.8, 0), WorkflowRecall(1, 0.7, 1)),
        canonical_ids_by_workflow=canonical_by_workflow,
    )
    payload = workflow_fallback_selector_payload(
        query_text="inspect then write",
        candidates=candidates,
        graph=graph,
        source=source,
    )
    assert payload["fallback_mode"] == "NEED_GRAPH_UNAVAILABLE"
    assert len(payload["candidates"]) == 2
    rendered = render_workflow_fallback(
        selection=SubgraphSelection("C0"),
        query_text="inspect then write",
        candidates=candidates,
        source=source,
        graph=graph,
    )
    assert rendered.status == "NEED_GRAPH_FALLBACK_WORKFLOW"
    assert "inspect source" in rendered.experience
    assert "CAUSAL ORDER EVIDENCE" in rendered.experience


def test_workflow_fallback_can_audit_soft_search_failure() -> None:
    source, partition, graph = _fixture()
    candidates = workflow_fallback_candidates(
        (WorkflowRecall(0, 0.8, 0),),
        canonical_ids_by_workflow={
            0: tuple(_canonical_id(group.members) for group in partition.groups[:3])
        },
    )
    rendered = render_workflow_fallback(
        selection=SubgraphSelection("C0"),
        query_text="inspect then write",
        candidates=candidates,
        source=source,
        graph=graph,
        status="SEARCH_FALLBACK_WORKFLOW",
        reason="soft subgraph search returned no candidate",
    )
    assert rendered.status == "SEARCH_FALLBACK_WORKFLOW"
    assert rendered.audit["fallback_reason"] == "soft subgraph search returned no candidate"


def test_common_workflow_support_is_not_a_continuous_occurrence_path() -> None:
    class Index:
        occurrences_by_canonical = {
            "A": ((0, 0),),
            "B": ((0, 1), (0, 2)),
            "C": ((0, 3),),
        }
        occurrence_adjacency = {(0, 0): ((0, 1),), (0, 2): ((0, 3),)}

    assert _recover_occurrence_path(("A", "B", "C"), Index()) is None
