# DEGS 0.78.0 experiment protocol

## Shared fixed settings

| Item | Value |
| --- | --- |
| Models | Qwen3.5-9B-AWQ or Qwen3.5-27B-AWQ; isolated runs |
| Embedding | Qwen3-Embedding-8B |
| Server context | 100,000 tokens |
| Spreadsheet completion | 32,000 tokens per Agent/producer/replay request |
| Temperature / thinking | 0 / false |
| Agent turns | 30 |
| Spreadsheet Agent/replay workers | 8 |
| Binding/Reflection/Canonical workers | 16 |
| Logical graph batch | 8 train tasks |
| Graph audit | always saved; never a downstream gate |

Cache is optional acceleration and is keyed by normalized-text SHA-256 for
embeddings. Empty state/cache is a supported formal start. Resume reuses only
identity-matching artifacts. Every run records stage/batch time, request/token
usage, retries/failures, commands, model, endpoint, graph snapshot and evaluator
output.

## SpreadsheetBench

- Trace2Skill data commit: `3d0b52a140f002a512930252b613c49048f7d5ac`.
- Train: verified dataset indices `[0,200)`, exactly 25 batches × 8, starting
  from empty `G0`. Each task retrieves/binds from its batch parent graph before
  the Agent runs.
- Development: verified indices `[200,400)`, fixed denominator 200.
- Full Soft/Hard: fixed 912-task / 2,529-case population after the existing
  train-case exclusion mapping.
- Evaluation: existing SpreadsheetBench comparator after LibreOffice
  recalculation; evaluator semantics are unchanged.
- No old train trajectory, replay, source graph, NeedGraph bundle, v9 state or
  heldout result is a formal 0.78.0 input.

## Skill2Bench

- Fixed entropy-stratified seed-42 split: train-100 / test-200, with hashes in
  `degs_skill2bench/contract.py`.
- Eight complete tasks arrive per graph batch; the final batch has four tasks.
- Each nonempty Step is a separate query/evidence episode. All Steps in the
  same task batch read the same graph. Their bound guidance is assembled in
  Step order, then the complete task gets one Agent execution.
- Closed and open Steps with score below one are repair candidates. The same
  profile model judges open Steps. Agent and replay requests omit explicit
  `max_tokens`, matching the local baseline protocol; producer calls use
  32,000.
- At test, each Step independently retrieves/binds and one full-task Agent run
  receives Step-ordered guidance.

## WikiTQ and HiTab

The official prepared population and evaluator are used. These are read-only
OOD consumers: query plus dataset-observable input retrieves from the matching
model's final SpreadsheetBench 0.78.0 graph. OOD outcomes, traces and gold never
update that graph.

## Information boundary and reporting

Train learning sees only train query/input, Agent trace, verifier evidence and,
when successful, the final patch plus one successful fresh replay. Earlier
failed replay history is excluded. Retrieval sees target query, permitted
observable target input and frozen train graph only. Gold/outcome/verifier data
is unavailable until evaluation.

Reports must name dataset version/split, population and denominator, model and
endpoint, fixed settings, evaluator/LibreOffice identity, exact run directory,
normal/abnormal completion counts, graph diagnostics, metrics and known
confounders. Smoke or partial results are not formal benchmark scores.
