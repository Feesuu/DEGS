# Episode evidence and source-leaf contract

0.78.0 has no global post-hoc source-extraction artifact. Every train task
produces one immutable `EpisodeEvidence` containing dataset/task/snapshot
identity, query, observable context, retrieval/expectations, original trace and
verifier, outcome, and only for repair success the final effective patch, one
successful fresh replay and its successful verifier.

One Reflection produces a validated `LearningDelta`:

- exactly one disposition for every retrieved anchor;
- residual successful ExperienceNodes;
- node-local edges;
- the actual successful episode procedure over used Canonical and new nodes.

Each ExperienceNode contains:

- `operation`: one transferable causal micro-operation;
- `applicability`: observable guard conditions;
- `inputs`: current-task binding/source requirements;
- `outputs`: expected state transition;
- `evidence_refs`: stable IDs from the same episode.

Edges use zero-based `source -> target` indices. Invalid individual edges are
discarded and audited without discarding valid nodes. Exact active duplicates
are absorbed rather than recreated. Unresolved/runtime failures have empty
positive graph/procedure output. No input evidence is silently truncated.
