# Evidence-bounded dynamic graph protocol

## Batch transition

For every batch of eight arriving logical tasks:

```text
freeze G(k-1)
  -> retrieve/bind independently for all tasks
  -> Agent + verifier independently
  -> optional repair/replay independently
  -> one Reflection per observable episode
  -> validate LearningDelta per episode
  -> resolve Canonical decisions outside the database transaction
  -> apply valid deltas in train-index order in one transaction
  -> publish G(k) as HEAD in that same transaction
```

Parallel completion order never changes graph order. All tasks in the batch
record the same `read_snapshot_id`; none can observe an intermediate mutation.

## State mutation

New evidence nodes first receive deterministic source-leaf identities. Exact
active duplicates are absorbed without changing Canonical text. Other nodes
are compared on guard-binding-operation-effect semantics. A SAME result absorbs
the source leaf and creates a new synthesized version under the stable target
Canonical ID. A distinct or item-locally unresolvable node becomes a singleton,
with the failure reason audited.

Canonical membership is monotonic. `QUALIFY` and `CORRECT` create new versions
without changing the entity ID. If two batch episodes propose incompatible
changes from the same frozen base version, train order decides the first and
the later proposal is retained as `DEFERRED_VERSION_CONFLICT`. Identical
revisions collapse to one active version while retaining both evidence events.

Only successful episode procedures contribute edges. Edges may connect old and
new Canonical nodes across source workflows. Invalid endpoints and self-edges
after fusion are discarded individually and audited; they never discard valid
nodes or the episode.

## Persistence and resume

SQLite stores snapshots, immutable EpisodeEvidence, expectations,
LearningDeltas, source leaves, stable Canonical entities/versions/members,
aliases, evidence events, procedure edges and normalized-text embedding cache.
LLM/embedding requests execute outside graph mutation transactions.

Spreadsheet batch runtime stores per-item Agent receipts, evaluator output,
repair outcomes and final EpisodeEvidence. A restart validates identities and
reuses completed work. It does not repeat completed Agent/Verifier/Replay when
Reflection or commit was interrupted. Valid binding decisions and
LearningDeltas are checkpointed per item before the batch barrier; standalone
development/Soft/Hard/OOD binding also uses identity-bound per-item
checkpoints. A committed batch must be a contiguous snapshot prefix and its
materialized manifest must match SQLite.

Item-local producer/runtime failures are recorded and skipped. A systemic
service outage, inconsistent committed prefix or corrupt artifact stops the
batch without publishing a partial HEAD.
