# DEGS

> **0.78.0 Evidence-Bounded EIR Dynamic**

DEGS learns a versioned procedural experience graph while train tasks arrive
in batches. Each batch reads a frozen parent graph, retrieves and contextually
binds prior experience before execution, learns once from the verified episode,
and atomically publishes the next graph snapshot.

```text
frozen G(k-1) -> Top-5 + one-hop context -> binding -> Agent -> verifier
              -> optional final patch/replay -> Reflection -> G(k)
```

The formal method and invariants are in
[`docs/METHOD_BOUNDARY.md`](docs/METHOD_BOUNDARY.md). The complete design is in
[`docs/IMPLEMENTATION_PLAN_EIR_DYNAMIC.md`](docs/IMPLEMENTATION_PLAN_EIR_DYNAMIC.md).

## Environment

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e '.[test]'
```

Any Python 3.12 patch release is supported. Runtime concurrency is configurable;
the defaults are Agent/replay 8, producer 32 and LibreOffice 16, but these are
not acceptance gates.

Required online services:

- generation: `Qwen3.5-9B-AWQ` or `Qwen3.5-27B-AWQ`, 100,000-token context;
- embedding: `Qwen3-Embedding-8B`.

Set keys without placing them in commands or manifests:

```bash
export DEGS_API_KEY="..."
export DEGS_EMBEDDING_API_KEY="..."
```

## SpreadsheetBench full campaign

Fetch or verify the pinned Trace2Skill checkout/data:

```bash
python scripts/fetch_spreadsheetbench.py \
  --checkout /path/to/Trace2Skill
```

Inspect the exact stage plan:

```bash
python scripts/run_full_campaign.py \
  --profile 9b \
  --run-root /path/to/runs/degs_0780_9b \
  --trace2skill-checkout /path/to/Trace2Skill \
  --generation-base-url http://GENERATION/v1 \
  --embedding-base-url http://EMBEDDING/v1 \
  --dry-run
```

Remove `--dry-run` to execute:

1. train `[0,200)` from empty `G0` as 25×8 dynamic batches;
2. graph audit;
3. development `[200,400)` retrieval, Agent and fixed-200 evaluation;
4. fixed 912-task / 2,529-case Soft/Hard population, retrieval, Agent and
   LibreOffice evaluation;
5. timing/token/result summary.

The same command with `--profile 27b` creates an independent 27B run. Never
reuse generated graph, cache, bundle or output across profiles.

The complete new-machine procedure, resume semantics, output locations and
validation checklist are in [`docs/RUNBOOK.md`](docs/RUNBOOK.md).

## Individual entrypoints

```text
degs-run-dynamic-train
degs-audit-graph
degs-bundle-development
degs-run-development
degs-evaluate-development
degs-prepare-soft-hard / degs-bundle-soft-hard
degs-run-soft-hard / degs-evaluate-soft-hard
degs-prepare-ood / degs-bundle-ood / degs-run-ood / degs-evaluate-ood
degs-run-skill2bench
```

All bundle builders use the shared contextual retriever/binder. Empty caches
are valid. Compatible completed artifacts resume and stale cache entries are
regenerated; graph-quality findings never gate downstream execution.

## Dataset protocols

- [`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md)
- [`docs/RUNBOOK.md`](docs/RUNBOOK.md)
- [`docs/SKILL2BENCH_PROTOCOL.md`](docs/SKILL2BENCH_PROTOCOL.md)
- [`docs/OOD_PROTOCOL.md`](docs/OOD_PROTOCOL.md)
- [`docs/INCREMENTAL_GRAPH_PROTOCOL.md`](docs/INCREMENTAL_GRAPH_PROTOCOL.md)
- [`docs/EXPERIENCE_SIMGRAG_RETRIEVAL.md`](docs/EXPERIENCE_SIMGRAG_RETRIEVAL.md)

## Verification

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
git diff --check
```

No 0.78.0 score is attached to the code checkpoint. Historical 0.77.41 results
remain historical and cannot be relabeled because 0.78.0 changes train,
Reflection, Canonical versioning and retrieval.
