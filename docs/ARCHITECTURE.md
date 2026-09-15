# DEGS 0.78.0 architecture

## Shared algorithm core

- `contextual_retrieval.py`: active Canonical Top-5 and at most two direct
  context neighbors per anchor.
- `contextual_binding.py`: strict applicability/parameter binding schema,
  parser and Agent guidance rendering.
- `contextual_runtime.py`: shared async retrieval/binding runtime used by every
  dataset route.
- `episode_evidence.py`: immutable query/context/trace/verifier/repair evidence
  with stable evidence IDs.
- `episode_learning.py`: one Reflection call and mechanical LearningDelta
  authorization.
- `eir_canonical.py`: guard-binding-operation-effect VIEW/MERGE resolution.
- `eir_graph.py`: transaction-free semantic planning, ordered deterministic
  mutation and graph compilation.
- `state_store.py::EIRStateStore`: schema-v10 snapshots, stable Canonical
  versions, evidence events, edges and per-text embedding cache.
- `dynamic_train.py`: dataset-neutral frozen-batch orchestration.
- `eir_bundle.py`: shared final-snapshot retrieval/binding bundle.
- `eir_graph_quality.py`: diagnostics only; no READY gate.

## Dataset adapters

- `spreadsheet_episode.py` owns SpreadsheetBench input projection,
  Agent/verifier execution and final-patch replay. Development uses
  `eir_bundle.py` + `benchmark.py`; Soft/Hard uses the same core through
  `soft_hard_bundle.py`; OOD uses it read-only through `ood_bundle.py`.
- `degs_skill2bench/eir_dynamic.py` maps eight arriving tasks to independent
  Step episodes while running one full-task Agent. `retrieval.py` independently
  retrieves/binds every test Step and assembles Step-ordered guidance.

Adapters may change task loading, observable context, Agent I/O and official
evaluator invocation. They may not fork retrieval, reconciliation,
Canonicalization or graph update semantics.

## Formal entrypoints

- `degs-run-dynamic-train`
- `degs-audit-graph`
- `degs-bundle-development`
- `degs-run-development` / `degs-evaluate-development`
- `degs-prepare-soft-hard`, `degs-bundle-soft-hard`,
  `degs-run-soft-hard`, `degs-evaluate-soft-hard`
- `degs-prepare-ood`, `degs-bundle-ood`, `degs-run-ood`,
  `degs-evaluate-ood`
- `degs-run-skill2bench`

Historical NeedGraph, Experience-SimGRAG, post-hoc source extraction and v9
incremental modules remain only as Git-era source/test evidence. No 0.78.0
formal entrypoint imports them.
