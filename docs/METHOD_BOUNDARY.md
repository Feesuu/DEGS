# DEGS 0.78.0 method boundary

The sole formal method is **DEGS 0.78.0 Evidence-Bounded EIR Dynamic**.
`0.77.41 Stable R1` is a historical Git baseline, not a selectable runtime
branch.

## Train boundary

At batch `k`, eight logical train tasks read frozen `G(k-1)`. Each task receives
Top-5 active Canonical hypotheses plus a bounded one-hop context, then one LLM
call binds or rejects those hypotheses using only the current query and
dataset-provided observable input. The Agent runs once. The unchanged verifier
judges its artifact. A failure may receive a patch and fresh replay; only the
final effective patch and its one successful replay enter positive evidence.

One Reflection call reconciles retrieved experience and extracts residual
successful micro-operations. Its strict validator authorizes:

- `NO_EVIDENCE`: no semantic change;
- `SUPPORT`: evidence event only;
- `QUALIFY`: narrower applicability after repair success;
- `CORRECT`: repaired operation or binding after repair success.

Unresolved/runtime failures cannot add nodes or edges. Valid deltas are
Canonicalized and committed by ascending train index. Semantic revisions keep
the stable Canonical ID and create a new version. HEAD publication is atomic.

## Retrieval boundary

The retrieval document is the original query plus observable target input.
Cosine similarity selects five unique active Canonical anchors. Each anchor may
carry at most two direct neighbors for procedure context. Neighbors are not
additional selectable candidates. There is no NeedGraph, workflow top-k,
beam/full-path enumeration, deterministic C0 or Selector in the formal path.

The binding LLM must decide every anchor as `SATISFIED`, `UNKNOWN` or
`CONFLICT`, cite current-task evidence for conditions and bound values, and may
reject all anchors. Only its validated guidance is injected; raw source traces,
scores and rejected Canonical text are not.

## Information and experiment boundary

- SpreadsheetBench graph: matching model's train `[0,200)` only; development
  `[200,400)` has fixed denominator 200.
- Skill2Bench graph: matching model's fixed seed-42 train-100 only; each Step
  is a separate evidence/retrieval unit, while one task gets one Agent run.
- WikiTQ/HiTab: read-only consumers of the matching SpreadsheetBench graph.
- No development/test outcome, gold, verifier result or trace may reach train
  learning or retrieval.
- State, cache and generated artifacts are isolated by dataset and model.
- Graph audits are diagnostic and never gate downstream execution.
