# DEGS 0.78.0 Evidence-Bounded EIR Dynamic Experience Learning

## 1. Status and purpose

This document is the implementation plan for the next DEGS method version. It
does not modify the accepted `DEGS 0.77.41 Stable R1` result. The 0.77.41 code
and result remain a Git baseline; the new version replaces the current formal
runtime rather than adding a hidden compatibility branch.

The implementation objective is to turn the current post-hoc incremental graph
builder into one evidence-bounded online learning loop:

```text
frozen G(k-1)
  -> broad Top-5 recall
  -> one contextual binding call
  -> one Agent rollout
  -> verifier
  -> optional patch + fresh replay + verifier
  -> one reflection call
  -> LearningDelta
  -> deterministic batch commit
  -> G(k)
```

The graph is a single dynamic procedural graph. An ExperienceNode is a
conditional hypothesis

```text
E = (g, b, o, y)
```

where `applicability` represents the observable guard `g`, `inputs` include the
parameter binding rule `b`, `operation` is the transferable micro-operation
`o`, and `outputs` describe the expected state transition `y`.

## 2. Non-negotiable invariants

1. A logical train batch contains eight tasks. Every task in the batch reads
   the same frozen `G(k-1)`.
2. Rollout, verification, repair/replay, and reflection may run concurrently
   across tasks. Graph mutations are applied in ascending train-index order.
3. Top-5 recall returns hypotheses, not mandatory instructions. The binding
   stage may reject all five.
4. A normal success may `SUPPORT` an existing experience and may add genuinely
   residual new operations. It may not broaden an existing applicability.
5. `QUALIFY` and `CORRECT` require the complete causal chain: old experience
   was used, a relevant failure was observed, the final patch changed the
   relevant behavior, fresh replay passed, and the cited verifier evidence
   supports the revision.
6. A failed patch is not positive experience. Earlier failed replay history is
   excluded from positive learning input. Only the final effective patch and
   its one successful fresh replay are admitted.
7. An unresolved failure or infrastructure failure cannot add positive nodes
   or procedural edges to the active graph.
8. Reflection extracts only successful behavior not already explained by
   correctly used retrieved experience. `SUPPORT` must not create a paraphrase
   node.
9. Canonical equivalence compares parameter binding rules, not merely
   replaceable constants. One observation may narrow a rule; broadening needs
   an explicit binding function or cross-condition evidence.
10. Canonical groups remain monotonic. Semantic revisions preserve the stable
    Canonical identity and all historical versions; retrieval reads only the
    version active in its frozen snapshot.
11. Successful episode paths may connect old Canonical nodes and newly learned
    nodes across workflows. Failure paths never become positive graph edges.
12. Graph-quality audits are always saved and analyzed but never gate retrieval,
    Agent execution, or evaluation.
13. Cache is optional acceleration. An empty cache must execute every required
    online LLM and embedding call.
14. One item-local LLM, JSON, context-length, or runtime failure is recorded and
    the remaining tasks continue. Only systemic service failure stops a batch.
15. No trajectory, prompt payload, action, observation, or final response is
    silently truncated. A request that cannot fit the server context is recorded
    as an item-local context-length failure.

## 3. Current-to-target architecture change

### 3.1 Current behavior to remove from the formal path

`scripts/run_full_campaign.py::Campaign.run` currently performs all 200 train
rollouts before the graph exists, verifies and replays them globally, extracts
all source graphs, and only then publishes 25 graph batches. This is incremental
insertion, not dynamic experience use.

The following global train stages must disappear from the new formal path:

```text
train_rollout(all 200)
train_verifier(all 200)
train_export(all 200)
train_replay(all failures)
source_extraction(all accepted outcomes)
graph_batch_00 ... graph_batch_24
```

They are replaced by:

```text
dynamic_train_batch_00
...
dynamic_train_batch_24
```

Each dynamic batch owns retrieval, binding, rollout, verification,
repair/replay, reflection, and graph commit for exactly its eight tasks.

### 3.2 Target module boundaries

The core algorithm is dataset-neutral. Dataset adapters own task loading,
observable-context projection, Agent execution, verifier invocation, and replay
execution. They must not fork retrieval, reflection, reconciliation, or graph
update semantics.

```text
DatasetEpisodeAdapter
  -> DynamicTrainCampaign
       -> ContextualRetriever
       -> ContextualBindingProducer
       -> existing Agent/runtime through adapter
       -> existing verifier through adapter
       -> existing patch/replay through adapter
       -> EpisodeReflectionProducer
       -> LearningDeltaApplier
       -> IncrementalGraphBuilder
```

The first formal implementation and experiment target is SpreadsheetBench 9B.
Skill2Bench then supplies a Step-scoped adapter; WikiTQ and HiTab remain
read-only consumers of the matching SpreadsheetBench graph.

## 4. Retrieval and contextual binding

### 4.1 Retrieval policy

The new formal path uses a deliberately small graph-native retriever instead of
NeedGraph, beam search, workflow top-k, or a mandatory Selector.

1. Build one retrieval document from the original query and dataset-provided
   observable context.
2. Embed it and every active Canonical document in the frozen snapshot.
3. Select five unique Canonical anchor identities by cosine similarity.
4. For every anchor, include at most two direct graph neighbors as structural
   context. Prefer one incoming and one outgoing neighbor; if one direction is
   absent, fill from the other. Rank eligible neighbors by their similarity to
   the same query-context embedding.
5. Anchors are the only experiences that the binding LLM may select. Neighbor
   nodes and induced edges explain local procedure context; they do not silently
   enlarge the candidate set.
6. If the graph is empty, emit an empty retrieval record and empty expectation
   without calling the binding LLM.

The maximum binding payload therefore contains five anchor nodes, ten context
neighbors, and the induced one-hop edges among them. There is no two-hop graph
expansion, full-path enumeration, or combinatorial subgraph matching in this
version.

The same retriever and binding producer are used during train, development,
Soft/Hard, Skill2Bench, and read-only OOD evaluation. Dataset adapters vary only
the observable-context projection.

### 4.2 `ExperienceExpectation` schema

The binding producer receives the query, observable evidence with stable IDs,
five anchors, their current versions, and the one-hop context subgraph. It emits
strict JSON:

```json
{
  "expectations": [
    {
      "canonical_id": "canonical_...",
      "canonical_version": 3,
      "condition": "SATISFIED",
      "condition_evidence_refs": ["query:0"],
      "expected_role": "Read the required text case from the task constraint",
      "bound_parameters": [
        {
          "name": "required_case",
          "value": "uppercase",
          "source_evidence_ref": "query:0"
        }
      ],
      "guidance": "Use the case explicitly required by the current task.",
      "expected_observation": "Target text conforms to the requested case."
    }
  ]
}
```

Allowed `condition` values are:

- `SATISFIED`: emit bound operational guidance;
- `UNKNOWN`: emit a check that must be performed before applying the operation;
- `CONFLICT`: emit no guidance for that experience.

Mechanical validation enforces:

- every referenced Canonical ID is one of the five anchors;
- every version equals the version from the frozen snapshot;
- every evidence reference exists in the supplied observable context;
- every bound value cites the query or observable context, never the old
  source-task constant alone;
- a `CONFLICT` expectation has no guidance;
- zero selected expectations is valid.

The rendered Agent injection contains only the validated guidance. Raw
Canonical text, rejected experiences, scores, and hidden source trajectories
are not injected.

## 5. Episode evidence model

Every train task produces one immutable `EpisodeEvidence` record. Large traces
remain files; SQLite stores identities, paths, hashes, statuses, and compact
validated JSON.

Required identities are:

```text
episode_id
dataset_contract_id
train_index / task_id
read_snapshot_id
retrieval_context_sha256
expectation_sha256
original_rollout identity
original_verifier identity
final outcome kind
optional final patch identity
optional successful replay identity
optional replay verifier identity
```

All LLM-visible pieces are projected into stable evidence IDs, for example:

```text
query:0
context:sheet:0
expectation:C17
trace:original:turn:4:action
trace:original:turn:4:observation
verifier:original:overall
verifier:original:mismatch:0
patch:final
trace:replay:turn:5:action
verifier:replay:overall
```

The evidence projector wraps the existing verifier output; it does not alter
the evaluator or invent new correctness judgments.

Final outcome kinds are:

```text
ORIGINAL_SUCCESS
REPAIR_SUCCESS
UNRESOLVED_TASK_FAILURE
ITEM_LOCAL_RUNTIME_FAILURE
```

Semantic unresolved failures still receive one Reflection call so retrieved
experience usage can be documented, but their validator permits only
`NO_EVIDENCE`, an empty new graph, and an empty procedure. Infrastructure
failures have no interpretable action/outcome chain and skip Reflection with an
explicit `EPISODE_UNOBSERVABLE` record.

## 6. One Reflection call and `LearningDelta`

### 6.1 Reflection input

The Reflection producer receives exactly:

```text
query + observable context
retrieval context
ExperienceExpectation
original trajectory + verifier evidence
if and only if repair succeeded:
    final effective patch
    one successful fresh replay
    successful replay verifier evidence
```

It never receives earlier failed replay history. It compares predicted use,
actual use, external result, and the validated repair intervention.

### 6.2 `LearningDelta` schema

```json
{
  "retrieved_experience_updates": [
    {
      "canonical_id": "canonical_...",
      "base_version": 3,
      "action": "SUPPORT",
      "usage_evidence_refs": ["trace:original:turn:4:action"],
      "outcome_evidence_refs": ["verifier:original:overall"],
      "repair_evidence_refs": [],
      "reason": "The bound operation was used and its predicted result passed.",
      "revised_experience": null
    }
  ],
  "new_experience_graph": {
    "experience_nodes": [
      {
        "operation": "...",
        "applicability": ["..."],
        "inputs": [{"type": "...", "description": "..."}],
        "outputs": [{"type": "...", "description": "..."}],
        "evidence_refs": [
          "trace:original:turn:7:action",
          "verifier:original:overall"
        ]
      }
    ],
    "edges": [{"source": 0, "target": 1}]
  },
  "episode_procedure": {
    "steps": [
      {"kind": "CANONICAL", "canonical_id": "canonical_..."},
      {"kind": "NEW", "node_index": 0}
    ],
    "edges": [{"source": 0, "target": 1}]
  }
}
```

`retrieved_experience_updates` has exactly four actions:

- `NO_EVIDENCE`: no graph mutation;
- `SUPPORT`: append successful-use evidence; do not change node text;
- `QUALIFY`: keep operation, inputs, and outputs byte-identical and produce a
  new active version with narrower applicability;
- `CORRECT`: produce a complete revised `ExperienceNode` after validated repair
  evidence demonstrates that operation or binding logic was wrong.

### 6.3 Mechanical authorization rules

The validator, not only the prompt, enforces:

1. An update target must be an anchor present in `ExperienceExpectation`.
2. `base_version` must equal the frozen version used for guidance.
3. `NO_EVIDENCE` and `SUPPORT` require `revised_experience=null`.
4. `SUPPORT` requires a successful final outcome, at least one actual-use trace
   reference, and a success-verifier reference.
5. `QUALIFY` and `CORRECT` require `REPAIR_SUCCESS` plus references to the failed
   verifier, final patch, successful replay behavior, and replay verifier.
6. `QUALIFY` may modify only `applicability`; changed binding or operation is a
   `CORRECT` action.
7. New nodes require a successful final outcome. On `REPAIR_SUCCESS`, every new
   node must cite both the final patch and successful replay evidence.
8. An exact duplicate of an active Canonical version is not admitted as a new
   node. Near-duplicates are sent through normal Canonical resolution rather
   than a hard embedding threshold.
9. Episode procedure endpoints may reference only successfully used expected
   Canonical nodes or valid new-node indices.
10. Invalid individual edges are dropped with an audit entry. An invalid edge
    does not discard otherwise valid nodes or the whole episode.

## 7. Stable Canonical identity and versioned content

The current membership-hash `_canonical_id(members)` changes identity whenever
groups merge, while `canonical_nodes` also makes content immutable for an ID.
That representation cannot implement evidence-bounded revision.

The new representation separates identity, membership, and semantic version:

```text
canonical_entities
  canonical_id                  stable identity
  created_snapshot_id
  retired_snapshot_id           set only when aliased by a merge

canonical_versions
  canonical_id
  version                       1, 2, ...
  experience_json
  document
  change_kind                   CREATE | MERGE | QUALIFY | CORRECT
  evidence_event_id
  active_from_snapshot_id
  inactive_from_snapshot_id

canonical_members
  canonical_id
  source_node_id

canonical_aliases
  alias_canonical_id
  target_canonical_id
  merge_event_id
```

A singleton identity is deterministically derived from its first source leaf
and remains stable. When two groups are judged SAME, the older identity is the
survivor (tie-break by lexical ID), the other becomes an alias, members only
accumulate, and the synthesis becomes a new version of the survivor. Existing
IDs remain resolvable through `canonical_aliases`.

`QUALIFY` or `CORRECT` creates version `n+1` under the same stable identity.
Version `n` remains queryable as historical evidence but is not returned by
retrieval after the new snapshot becomes active.

All retrieval records contain both `canonical_id` and `canonical_version`.
Historical snapshot artifacts are materialized and remain reproducible.

### 7.1 Concurrent revision conflict within one batch

Every Reflection result is anchored to a frozen `base_version`. During ordered
commit:

1. identical revisions of the same base version collapse into one new version
   and both evidence events are retained;
2. the first non-identical revision in train-index order becomes active;
3. a later incompatible revision against the now-stale base version is stored
   as `DEFERRED_VERSION_CONFLICT` evidence and does not overwrite the active
   version or stop the batch.

This avoids silently composing two semantic edits that were independently
generated against old text and preserves the one-Reflection-call constraint.

## 8. Canonical equivalence under parameter binding

The Canonical VIEW prompt must normalize every source node into:

```text
guard: observable condition
binding: where each task-varying parameter comes from
operation: transferable micro-operation
effect: expected state transition
```

The MERGE prompt returns SAME only when both nodes share the same operation and
the same parameter-selection function. Different observed constants may merge
only when both trajectories support a common binding source.

Examples:

```text
SAME:
  required_case=lowercase read from the task constraint
  required_case=uppercase read from the task constraint

DIFFERENT:
  unconditional lowercase
  unconditional uppercase
```

Merge synthesis may preserve or narrow guards. It may not invent a wider guard
than the evidence of its members. Existing predecessor, successor, workflow,
or cluster context is not a prerequisite for SAME.

## 9. Cross-workflow procedure edges

`episode_procedure` is resolved only after new nodes are Canonicalized. Its
endpoints may therefore connect:

```text
existing Canonical -> new Canonical
new Canonical      -> existing Canonical
existing Canonical -> existing Canonical
new Canonical      -> new Canonical
```

The state store records `episode_id` provenance for each resolved edge. Graph
compilation unions these edges with projected source-workflow edges and
deduplicates by `(source_canonical_id, target_canonical_id)`. Supporting
workflow/episode counts are descriptive provenance, never an edge-quality gate.

Self-edges produced by Canonical fusion and invalid endpoint edges are dropped
and audited without discarding the episode.

## 10. Transactional batch execution and resume

### 10.1 Batch phases

For batch `k`:

1. Resolve and freeze `read_snapshot_id = HEAD`.
2. Produce retrieval contexts and expectations for all eight tasks.
3. Run eight Agent tasks with their validated guidance.
4. Invoke the unchanged evaluator on the batch outputs and project its evidence.
5. Run patch/replay for failed tasks, reusing the current repair controller and
   evaluator semantics.
6. Produce one `LearningDelta` per observable episode.
7. Validate every delta independently.
8. Apply valid deltas in ascending train-index order in one batch transaction.
9. Materialize graph, partition, audit, and snapshot artifacts.
10. Atomically publish `G(k)` as HEAD.

The batch transaction may contain item-level savepoints so one invalid delta is
recorded and skipped. No task can observe a partially committed batch graph.

### 10.2 Resume identity

Every producer request identity includes method version, dataset contract,
model/endpoint identity, prompt hash, response-schema hash, frozen snapshot ID,
and payload hash. Completed artifacts are reused only when the identity matches
exactly.

Per task:

```text
batches/batch_07/episodes/0056/
  retrieval.json
  expectation.json
  rollout/
  verifier.json
  final_patch.json              optional
  replay/                       optional
  replay_verifier.json          optional
  learning_delta.json
  status.json
```

Per batch:

```text
batches/batch_07/
  manifest.json
  commit_audit.json
  snapshot_manifest.json
```

A restart reuses completed matching phases, completes missing phases, and either
replays an already validated transaction idempotently or publishes the single
committed batch snapshot. It never reruns completed Agent work merely because a
later Reflection item failed.

## 11. Files and symbols to change

### 11.1 New modules

| File | Responsibility |
| --- | --- |
| `src/degs/contextual_retrieval.py` | Top-5 active-Canonical recall, bounded one-hop context, retrieval audit |
| `src/degs/contextual_binding.py` | binding prompt/schema/parser, `ExperienceExpectation`, Agent guidance rendering |
| `src/degs/episode_evidence.py` | immutable evidence projection and stable evidence IDs |
| `src/degs/episode_learning.py` | Reflection prompt/schema/parser and mechanical `LearningDelta` authorization |
| `src/degs/dynamic_train.py` | frozen-batch orchestration, concurrency, checkpoints, ordered commit |
| `src/degs/spreadsheet_episode.py` | SpreadsheetBench task/context/Agent/verifier/replay adapter using existing runtime |
| `src/degs/resources/CONTEXTUAL_BINDING_PROMPT_V1.txt` | contextual applicability and parameter rebinding prompt |
| `src/degs/resources/EPISODE_REFLECTION_PROMPT_V1.txt` | unified reconcile/residual-extract/procedure prompt |

These are concrete algorithm boundaries, not a plugin framework. A small
`Protocol` in `dynamic_train.py` defines only the adapter methods required by the
core loop.

### 11.2 Existing modules to modify

| File | Required change |
| --- | --- |
| `scripts/run_full_campaign.py` | Replace global train stages with 25 dynamic batches; leave downstream phases consuming final HEAD |
| `src/degs/incremental_graph.py` | Accept validated `LearningDelta` batches; apply support/revision/new-node/procedure operations; keep deterministic ordered commit |
| `src/degs/state_store.py` | Bump schema/method identity; add episodes, expectations, deltas, evidence events, stable Canonical entities/versions/aliases, procedure edges |
| `src/degs/section_graph.py` | Separate stable Canonical identity from active version; compile source and episode edges into one graph |
| `src/degs/canonicalize.py` | Make VIEW/MERGE operate on guard-binding-operation-effect semantics |
| `src/degs/provider.py` | Load validated contextual guidance without assuming a legacy NeedGraph bundle |
| `src/degs/agent.py` | Preserve the existing Agent behavior while accepting the new provider payload |
| `src/degs/benchmark.py` | Use the same contextual retriever/binder on a frozen final graph for development |
| `src/degs/soft_hard_bundle.py` | Use the same frozen retrieval/binding implementation for the fixed population |
| `src/degs/ood_bundle.py` | Use the same read-only retrieval/binding path over a matching SpreadsheetBench graph |
| `src/degs_skill2bench/retrieval.py` | Invoke the same retrieval/binding once per independent Step |
| `src/degs/graph_quality.py` | Add active-version, revision, residual-node, evidence-action, and cross-workflow-edge diagnostics without gating |

### 11.3 Current modules removed from the new formal call graph

The new Reflection call supersedes the two-call post-hoc combination of
successful/repair extraction plus source review. Therefore the formal campaign
must no longer call:

```text
degs.source_rebuild
SuccessfulTrajectoryExperienceExtractor
ValidatedRepairExperienceExtractor
ExperienceSourceReviewer
```

Before deletion, Serena reference checks must identify whether a parser or
artifact helper is still used by a dataset adapter. Reusable evidence parsing is
moved to `episode_evidence.py`; modules with no remaining live references are
deleted. Historical reproduction remains available through Git, not runtime
flags or compatibility branches.

Likewise, `workflow_retrieval.py`, `experience_simgrag.py`, and old NeedGraph
prompt resources are removed only after all formal dataset bundle builders use
`contextual_retrieval.py` and reference tests show no live import. There will be
one formal retrieval implementation, not an `--old-retrieval` switch.

### 11.4 Documentation and version identity

After implementation passes tests:

- update `AGENTS.md` to make 0.78.0 the sole formal method;
- update `VERSION.md` and package version;
- update `docs/METHOD_BOUNDARY.md`;
- replace the train section of `docs/INCREMENTAL_GRAPH_PROTOCOL.md`;
- update dataset protocols to state how observable context is projected;
- append `docs/VERSION_HISTORY.md` with commit, validation, run IDs, results,
  and known confounders.

The current 0.77.41 identity is not changed during planning and is not relabeled
as a 0.78.0 result.

## 12. Implementation stages and checkpoints

### Stage 1: schemas and pure validators

Implement `ExperienceExpectation`, `EpisodeEvidence`, `LearningDelta`, endpoint
references, JSON schemas, parsers, and authorization rules before any live LLM
or campaign changes.

Verification:

- valid examples for all four update actions;
- invalid evidence IDs rejected;
- success cannot authorize broadening;
- QUALIFY cannot alter operation/contracts;
- repair evidence is mandatory for QUALIFY/CORRECT;
- malformed edge drops do not discard valid nodes.

### Stage 2: stable Canonical identity and state schema

Implement schema version 10 in a fresh database, stable identities, semantic
versions, aliases, evidence events, and snapshot-specific active-version reads.
No automatic v9 state migration is added because the formal EIR run must begin
from empty `G0`.

Verification:

- revision preserves Canonical ID;
- old versions remain queryable;
- frozen snapshots return their historical active version;
- merges accumulate members and aliases resolve transitively;
- concurrent incompatible revisions follow deterministic conflict policy.

### Stage 3: simple retrieval and contextual binding

Implement Top-5 recall, at most two one-hop context neighbors per anchor, the
binding prompt, strict response schema, guidance rendering, cache identity, and
item-local retry behavior.

Verification:

- empty graph requires no LLM call;
- at most five anchors and ten context nodes;
- conflicting lowercase/uppercase cases do not leak source constants;
- zero selected guidance is valid;
- empty cache makes real producer calls in an integration test.

### Stage 4: Reflection and graph mutation

Implement evidence projection, one Reflection call, residual extraction,
reconciliation actions, active-version creation, new-node Canonicalization, and
episode-procedure edge resolution.

Verification:

- repeated covered behavior creates SUPPORT only;
- genuinely residual behavior creates one new node;
- original success cannot QUALIFY/CORRECT;
- validated repair can narrow/correct the cited node;
- failed repair produces no positive node or edge;
- old/new cross-workflow paths compile correctly.

### Stage 5: dynamic batch campaign

Replace the offline train chain with `DynamicTrainCampaign`, reuse the existing
Agent/verifier/repair semantics through `SpreadsheetEpisodeAdapter`, and add
phase-level resume.

Verification:

- all eight tasks read exactly one frozen snapshot;
- batch commits in train-index order irrespective of completion order;
- one item failure does not stop the batch;
- systemic endpoint failure stops the batch without partial HEAD publication;
- restart does not repeat completed Agent/LLM work.

### Stage 6: downstream adapters and dead-path removal

Move development, Soft/Hard, Skill2Bench Step retrieval, and OOD bundles to the
same contextual retrieval/binding path. Remove dead NeedGraph and post-hoc
source-extraction code only after symbol-reference and import checks pass.

Verification:

- each dataset/model owns isolated state and artifacts;
- Skill2Bench Steps retrieve independently;
- WikiTQ/HiTab cannot write the source graph;
- fixed splits, denominators, evaluator, model, turns, decoding, and context
  remain unchanged.

### Stage 7: documentation, full test, and independent review

Run the full test suite, protocol checks, `git diff --check`, and real independent
spec/regression/research-protocol/complexity reviews. Valid blockers are fixed
before the version identity is updated.

## 13. Test matrix

New targeted tests:

```text
tests/unit/test_contextual_retrieval.py
tests/unit/test_contextual_binding.py
tests/unit/test_episode_learning.py
tests/unit/test_canonical_versions.py
tests/unit/test_dynamic_train.py
tests/integration/test_eir_two_batch_learning.py
tests/integration/test_eir_empty_cache.py
```

Required regression coverage:

- existing Spreadsheet Agent/runtime behavior;
- existing LibreOffice verifier and fixed 200 denominator;
- repair uses only final patch plus one successful replay;
- text-level embedding cache remains keyed by normalized-text SHA-256;
- graph audit cannot block downstream execution;
- no dev/test outcome or trace reaches train learning payloads;
- no prompt input truncation;
- time, request count, prompt/completion tokens, retry count, and failure class
  are recorded per task and per batch.

Synthetic two-batch acceptance case:

1. Batch 1 learns that case conversion is parameterized by a task-visible
   `required_case`.
2. Batch 2 requests the opposite case, retrieves the old node, binds the new
   value, and does not inherit the old source constant.
3. A repair-success episode narrows an over-broad guard while preserving the
   Canonical ID and historical version.
4. A successful episode combines an old node and a new residual node, producing
   a cross-workflow procedural edge.

## 14. Formal experiment regeneration

This method changes train behavior, producer prompts, Canonical identity,
retrieval, and Agent guidance. Therefore no old source graph, NeedGraph bundle,
train trajectory, replay result, Canonical state, or heldout result is a valid
formal input.

Formal SpreadsheetBench 9B sequence:

```text
empty G0
-> train [0,200), 25x8 dynamic batches
-> graph audit and learning-action audit
-> frozen development retrieval/binding [200,400)
-> development Agent, 8 workers
-> unchanged LibreOffice evaluation, denominator 200
-> fixed Full Soft/Hard population
```

Only after the 9B chain is technically and scientifically auditable should the
same code run the independent 27B profile. Skill2Bench must rebuild its own
Step-scoped graph from its fixed train split. WikiTQ/HiTab may read the matching
new SpreadsheetBench graph but never update it.

Formal reports include:

- per-batch online train success and repair-success rates;
- Top-5 retrieval and binding decisions;
- `NO_EVIDENCE`/`SUPPORT`/`QUALIFY`/`CORRECT` counts;
- residual nodes per successful episode;
- active/history Canonical counts and merge/revision counts;
- graph components, giant-component size, isolated nodes, and cross-workflow
  procedure edges;
- timing, LLM calls, prompt/completion tokens, retries, and item/system failures;
- fixed development/Soft/Hard metrics and denominators.

These diagnostics inform analysis but never prevent the downstream experiment
from running.

## 15. Acceptance criteria

The code checkpoint is complete only when:

1. the formal train entrypoint is genuinely batch-online and starts from empty
   `G0`;
2. every batch task reads the same frozen parent graph;
3. Top-5 contextual binding can apply, defer, or reject every recalled
   experience without a mandatory choice;
4. all active experience text is evidence-bounded and parameter binding cites
   observable current-task evidence;
5. SUPPORT never creates duplicates and normal success never broadens a rule;
6. QUALIFY/CORRECT cannot commit without validated repair-success causality;
7. Canonical identity survives semantic revision and historical versions remain
   available;
8. only successful episode paths create cross-workflow graph edges;
9. individual failures are recorded and skipped without aborting healthy work;
10. all formal dataset routes use one retrieval and learning semantics with
    isolated state;
11. unit, integration, empty-cache, resume, protocol, and full regression tests
    pass;
12. independent review finds no unresolved blocker;
13. a complete fixed-denominator development run is executed regardless of the
    graph-quality audit status.
