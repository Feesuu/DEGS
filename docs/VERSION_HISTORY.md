# Version history

## DEGS 0.78.0 Evidence-Bounded EIR Dynamic

- Development branch: `codex/degs-0780-eir-dynamic`.
- Replaced post-hoc train extraction with a true frozen-batch online loop:
  Top-5 contextual binding, Agent/verifier, optional final patch plus one
  successful replay, one Reflection, ordered atomic commit.
- Added immutable EpisodeEvidence, mechanically authorized LearningDelta,
  stable Canonical identities with historical versions, evidence events and
  cross-workflow successful procedure edges.
- Unified development, Soft/Hard, Skill2Bench Step and read-only WikiTQ/HiTab
  retrieval on one Top-5 + bounded one-hop + binding implementation.
- Added schema-v10 SQLite state, per-text embedding cache, item-level failure
  recording, phase resume, per-batch timing/token metrics and non-gating graph
  audit.
- Parallelized independent Canonical views and candidate comparisons while
  retaining deterministic score-order selection; Binding/Reflection waves use
  32 workers and Agent/replay remains bounded by each eight-task batch.
- Formal 0.78.0 experiments must begin at empty `G0`; no 0.77.41 generated
  artifact is a valid input.
- Portability checkpoint: runtime endpoints, filesystem locations and positive
  worker counts are recorded rather than enforced as exact signatures. Worker
  concurrency is configurable across Agent/replay, Binding/Reflection,
  Canonical and LibreOffice stages; stale item caches are regenerated. Dataset,
  split, denominator, model and graph isolation remain hard semantic boundaries.
- The end-to-end new-machine procedure is recorded in `docs/RUNBOOK.md`.
- Agent-runtime portability checkpoint: the shared SpreadsheetBench/WikiTQ/
  HiTab bubblewrap mounts both the project virtual environment and its real
  base interpreter; Skill2Bench keeps its baseline Bash tool and receives the
  same project environment through `PATH`. A real Bash-tool preflight records
  both interpreter names, dependencies and an XLSX roundtrip without enforcing
  machine-specific paths or versions.
- Runtime smoke on SpreadsheetBench development indices 200–203 completed all
  four Agent/evaluator paths without environment failures; LibreOffice 24.2.7.2
  passed 3/4 tasks. This selected four-task diagnostic is not a benchmark score.
- Code checkpoint result: 292 tests passed; focused Pyright reported zero
  errors (with one environment-only `tqdm` source-resolution warning); the
  formal route import audit loaded no legacy NeedGraph/C0/beam modules; the
  pinned Spreadsheet train verifier/exporter entrypoints passed import smoke
  checks. No 0.78.0 benchmark score is claimed until the full fixed protocol
  is run.

## DEGS 0.77.41 Stable R1

- Single formal runtime; historical retrieval stages are folded into one online implementation.
- SpreadsheetBench 9B development `[200,400)`: `88/200 = 44.0%` in the accepted audited run.
- Experience graph: 421 source nodes, 271 Canonical nodes, 290 projected occurrence edges.
- Retrieval: query-only V2 NeedGraph, operation top-8, workflow top-8, beam32, input-cited clarification, complete-candidate workbook-role late fusion, deterministic C0 with fallback Selector.
- The clean runtime preserves the accepted graph/retrieval semantics. The `88/200` score belongs to the audited source run; a fresh end-to-end run from this repository is required before reporting it as a clean-repository reproduction.
- Cache is optional; the empty retrieval-cache integration test calls all online producers.

Historical experimental version numbers are intentionally not public runtime choices. Their results remain research notes outside the formal execution path.

### Multi-dataset adapter checkpoint

- Added isolated 9B/27B runners for Skill2Bench and WikiTQ/HiTab without creating a second DEGS method variant.
- Skill2Bench uses target-Step source workflows and one unchanged 0.77.41 online retrieval invocation per test Step, followed by one full-task Agent run.
- Target-domain retrieval caches are separate from source graph state; OOD source state is read-only.
- No Skill2Bench or OOD benchmark score is claimed by this code checkpoint until a complete formal run finishes.
