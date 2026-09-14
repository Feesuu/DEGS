# DEGS 0.77.41 multi-dataset implementation plan

## Objective

Expose one DEGS 0.77.41 algorithm through three dataset-specific experiment
pipelines without sharing mutable run state:

1. SpreadsheetBench: train `[0,200)`, development `[200,400)`, and the fixed
   Soft/Hard population;
2. Skill2Bench: fixed seed-42 train-100/test-200 split, full-task Agent and
   evaluation, but Step-scoped source graphs and online retrieval;
3. WikiTQ and HiTab: read-only transfer from the matching model's frozen
   SpreadsheetBench graph.

Both Qwen3.5-9B-AWQ and Qwen3.5-27B-AWQ are supported. The first formal
cross-dataset run uses the 9B profile.

## Method invariants

The following are shared and are not reimplemented by an adapter:

- ExperienceNode schema: `operation`, `applicability`, `inputs`, `outputs`;
- original-success or final validated-patch plus replay-success source
  admission;
- monotonic node-only Canonical equivalence;
- real occurrence edges;
- query-only NeedGraph, workflow top-8, Need-to-Canonical top-8, beam-32
  Experience-SimGRAG, deterministic top-ranked C0, and the existing fallback;
- Qwen3-Embedding-8B embeddings;
- graph-quality audit is diagnostic and never an execution gate.

Dataset adapters may define only the data unit, split, task prompt profile,
source-evidence projection, Agent I/O contract, evaluator, and run sequence.

## Mutable-state boundary

Every run owns its mutable state:

```text
runs/<dataset>/<profile>/<run-id>/
  manifest.json
  train/
  graph/state.sqlite3
  graph/snapshots/
  retrieval/cache.sqlite3
  retrieval/bundle/
  agent/
  evaluation/
  logs/
  usage/
```

SpreadsheetBench and Skill2Bench never read each other's graph, state, cache,
or outputs. WikiTQ and HiTab receive a frozen SpreadsheetBench snapshot as a
read-only source reference. Each OOD dataset writes embeddings and LLM
responses only to its own retrieval cache and output directory.

Cache misses always call the configured online LLM or embedding endpoint. A
cache is an acceleration artifact, never a required method input.

## Shared-core changes

### Dataset graph contract

Add `GraphDatasetContract(identity, source_split, train_count, batch_size,
allow_final_partial_batch)`. Replace hard-coded `200` and `8` assumptions in
the state store, incremental graph builder, snapshot loader, and graph audit
with the supplied contract. SpreadsheetBench remains the default contract, so
its graph behavior is unchanged.

Skill2Bench uses 1,000 stable Step slots:

```text
workflow_id = train_task_index * 10 + (step_number - 1)
```

An arrival batch is eight train tasks, represented by 80 consecutive slots.
Unused slots are explicit no-source entries. The final train-task batch has
four tasks and therefore 40 slots.

### Retrieval cache

Separate snapshot validation from retrieval caching. The source state database
is opened only to validate/load its committed graph. Embeddings, NeedGraphs,
clarifications, and Selector responses are stored in a target-local
`RetrievalStore` keyed by normalized text or complete request identity.

This removes the current behavior where WikiTQ/HiTab query data is written to
the SpreadsheetBench state database.

### Prompt composition

Producer prompts are assembled from:

```text
shared DEGS producer instruction + dataset profile
```

The existing SpreadsheetBench prompt text is preserved byte-for-byte for the
SpreadsheetBench profile. Skill2Bench and table-QA receive narrow profiles;
they do not fork the graph or retrieval algorithm.

## SpreadsheetBench adapter

Preserve the existing 0.77.41 pipeline and exact 9B result identity. Add one
profile-aware campaign entry point for 9B/27B. Fix evaluator validation so the
expected model is supplied by the selected profile rather than hard-coded to
9B. Do not change the split, prompts, top-k, beam, Agent turns, completion
limit, evaluator, LibreOffice procedure, or denominator.

## Skill2Bench adapter

### Fixed protocol

- train: the checked seed-42 100-task split;
- test: the checked 200-task split;
- Agent/replay concurrency: 8;
- source/Canonical/retrieval producer concurrency: 16;
- Agent completion: omit `max_tokens`;
- thinking: false;
- maximum turns: 30;
- all failed closed and open Steps are eligible for patch/replay;
- open-ended judge uses the same profile model;
- one full-task Agent rollout and one official evaluation per task.

The fixed split manifest and upstream provenance are included. A preparation
script recreates the exact split from the pinned public Skill-Entropy-RL
snapshot and rejects any train/test file whose hash differs.

### Step-scoped source evidence

Each accepted Step becomes one independent source workflow. Source extraction
receives only the scenario background, target Step number/question, and
accepted evidence for that Step.

For original success, only a trace fragment attributable to the target Step is
accepted. A multi-Step trace has no implicit Step-1 owner.

For a validated repair whose successful replay lacks reliable Step markers,
the request contains:

```text
target Step question
+ that Step's accepted {instructions, checks} patch
+ complete successful replay
```

The extractor may emit only an operation/check that is jointly specified by
the patch and realized in the successful replay. It must ignore operations
from other Steps and return empty arrays if the intersection cannot be
demonstrated. Host code binds every emitted node to the accepted Step evidence;
the LLM does not output provenance. Edges are local to the Step workflow.

### Step-scoped query

Every non-empty public test Step independently runs the unchanged DEGS online
retrieval chain. Step order is not causality. An earlier Step result is a
symbolic input only when the target question explicitly references it.

The per-Step experiences are assembled in original Step order and injected
once into the full-task Agent prompt:

```text
Step 1:
<retrieved experience>

Step 2:
<retrieved experience>
```

No similarity threshold, relevance abstention, or dataset-only Selector is
added. Normal candidates use deterministic C0 exactly as SpreadsheetBench.

## WikiTQ and HiTab adapters

### Data/evaluation

- WikiTQ commit `7d455a5a707b96341ef72aff9428749d443d8aa9`,
  `random-split-1-dev.tsv`, 2,810 tasks;
- HiTab commit `d179602662b490249baf068a76fbe4137029126e`, official
  test split, 1,584 tasks;
- each table is exposed as an XLSX `Table` sheet;
- the Agent writes the final answer to `Answer!B1`;
- use the upstream WikiTQ evaluator and HiTab `hmt_score`.

### Transfer semantics

There is no target-domain graph update. A 9B OOD run reads only the frozen 9B
SpreadsheetBench graph; a 27B OOD run reads only the frozen 27B graph. WikiTQ
and HiTab have separate retrieval caches, bundles, Agent outputs, and evaluator
outputs even when they share the same source graph.

## Code changes

### Add

- `src/degs/graph_dataset_contract.py`
- `src/degs/retrieval_store.py`
- `src/degs_skill2bench/` dataset, protocol, runtime, source, retrieval,
  evaluation, prompts, and campaign modules
- the existing `degs.ood_*` data, retrieval, Agent, and evaluation modules are
  reused behind a profile-aware campaign launcher
- `scripts/run_skill2bench.py`
- `scripts/run_tableqa_ood.py`
- fixed Skill2Bench split manifest and external pinned local-baseline runtime contract
- unit and integration tests for every adapter and state boundary

### Modify

- shared state/snapshot/graph modules to accept `GraphDatasetContract`;
- retrieval construction to use a target-local `RetrievalStore`;
- SpreadsheetBench evaluator to accept the profile model;
- package metadata, README, architecture, experiment protocol, and runbook.

### Remove after replacement is verified

- `scripts/run_9b_ood_campaign.py`;
- duplicated or dead multi-dataset compatibility entry points.

Old experimental worktrees and run artifacts are evidence only and are never
runtime imports.

## Verification gates

1. Existing SpreadsheetBench unit suite remains green.
2. Dataset-contract tests cover 25x8 SpreadsheetBench batches and Skill2Bench
   12x8 plus one four-task batch.
3. A test opens a source SpreadsheetBench state database, runs OOD retrieval,
   and proves its bytes/tables/row counts did not change.
4. Skill2Bench tests cover Step splitting, transition phrases, no implicit
   Step 1, repair full-replay intersection, host provenance, local-only edges,
   per-Step retrieval, ordered injection, and official metrics.
5. 9B/27B dry-run manifests differ only in model profile fields.
6. Empty-cache smoke tests invoke live producers rather than requiring prior
   bundles.
7. Full test suite and `git diff --check` pass, followed by an independent
   local spec/protocol/regression/complexity review.
