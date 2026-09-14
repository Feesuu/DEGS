"""Deterministic identity and contract of the DEGS 0.77.41 runtime."""

from __future__ import annotations

METHOD_ID = "DEGS 0.77.41 Stable R1"
METHOD_VERSION = "0.77.41"
METHOD_CONTRACT = {
    "method_id": METHOD_ID,
    "version": METHOD_VERSION,
    "graph": {
        "train_split": "[0,200)",
        "source_evidence": [
            "original_success",
            "final_patch_plus_replay_success",
        ],
        "source_extraction": "causal_micro_operation_v4_plus_review_v2",
        "publication": "25_contiguous_batches_of_8",
        "canonicalization": "monotonic_node_only_equivalence",
        "edges": "projected_real_occurrence_edges",
    },
    "retrieval": {
        "need_graph": "query_only_v2",
        "workflow_top_k": 8,
        "need_to_canonical_top_k": 8,
        "beam_size": 32,
        "clarification": "input_only_cited_existing_need_nodes",
        "late_fusion": "symmetric_value_masked_workbook_roles",
        "selection": "deterministic_top_ranked_c0_with_llm_fallback",
    },
    "development": {
        "split": "[200,400)",
        "denominator": 200,
        "agent_workers": 8,
        "agent_turns": 30,
        "completion_tokens": 32000,
        "thinking": False,
    },
    "forbidden": [
        "development_outcome",
        "development_gold",
        "development_verifier_result",
        "development_agent_trace",
    ],
}

__all__ = [
    "METHOD_CONTRACT",
    "METHOD_ID",
    "METHOD_VERSION",
]
