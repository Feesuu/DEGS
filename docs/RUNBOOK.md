# DEGS 0.78.0 end-to-end runbook

This is the operational source of truth for running the current method on a
new machine. The method is **DEGS 0.78.0 Evidence-Bounded EIR Dynamic**. A
SpreadsheetBench graph, a Skill2Bench graph, and each model profile use
separate run roots. WikiTQ and HiTab are read-only consumers of the matching
SpreadsheetBench graph.

For a ready-to-paste task for the Agent on that machine, use
[`docs/NEW_MACHINE_AGENT_HANDOFF.md`](NEW_MACHINE_AGENT_HANDOFF.md).

## 1. What is fixed and what is configurable

The following are experiment semantics and must not be changed inside a run:

- dataset identity, split, ordered task IDs and reported denominator;
- 9B versus 27B model identity;
- SpreadsheetBench train `[0,200)` and development `[200,400)`;
- 25 logical SpreadsheetBench train batches of 8 tasks;
- the fixed Skill2Bench seed-42 train-100/test-200 split;
- target-gold isolation and dataset/model graph isolation;
- the evaluator used for the reported metric.

The following are runtime resources, not acceptance gates:

- Agent/replay worker count (default 8);
- producer worker count (default 32);
- LibreOffice worker count (default 16);
- service host/port and local filesystem location;
- Python patch version and recorded dependency versions.

Every worker count only has to be a positive integer. Use the largest value
the deployed services and machine can sustain. A logical train batch remains
8 tasks even when execution concurrency is larger or smaller than 8.

Endpoints, versions, hashes and request settings are recorded in manifests.
They do not make a scientifically compatible run fail merely because the
machine path, endpoint or worker count changed. A stale or malformed cache
entry is a cache miss and is regenerated. Content and method changes that
would mix datasets, model profiles, graph semantics or evaluation populations
still require a new run root.

## 2. Install and services

```bash
git clone https://github.com/Feesuu/DEGS.git
cd DEGS
python3.12 -m venv .venv
.venv/bin/pip install -e '.[test]'
```

Any Python 3.12 patch release is supported. SpreadsheetBench execution also
requires `bwrap`, LibreOffice/`soffice`, and the normal shell tools used by the
Agent sandbox. WikiTQ's official evaluator additionally requires Python 2 at
the path passed to `--python2`.

Required OpenAI-compatible services:

- generation model: `Qwen3.5-9B-AWQ` or `Qwen3.5-27B-AWQ`;
- generation server context: 100,000 tokens;
- embedding model: `Qwen3-Embedding-8B`.

Set secrets in the environment or pass key-file paths; do not put key values
in run manifests:

```bash
export DEGS_API_KEY="$(< /path/to/generation_api_key)"
export DEGS_EMBEDDING_API_KEY="$(< /path/to/embedding_api_key)"
export GENERATION_URL=http://generation-host:port/v1
export EMBEDDING_URL=http://embedding-host:port/v1
```

Check both services before a long run:

```bash
.venv/bin/python scripts/preflight_services.py \
  --generation-base-url "$GENERATION_URL" \
  --generation-model Qwen3.5-9B-AWQ \
  --embedding-base-url "$EMBEDDING_URL" \
  --embedding-model Qwen3-Embedding-8B
```

This check verifies current connectivity and model names. It does not bind a
run permanently to that host or port.

### 2.1 Agent execution environment (plan A)

Keep each dataset's native execution protocol; align only the Python runtime
visible to its Bash tool:

| Dataset | Bash execution | Python made visible to the Agent |
|---|---|---|
| SpreadsheetBench | shared bubblewrap sandbox | project `.venv` plus its real base interpreter, both read-only |
| WikiTQ | same shared bubblewrap sandbox | same as SpreadsheetBench |
| HiTab | same shared bubblewrap sandbox | same as SpreadsheetBench |
| Skill2Bench | unchanged baseline host Bash tool | project `.venv/bin` first on the worker `PATH` |

This preserves `/workspace`, task isolation, input/output paths, network
behavior and experience injection. It does not install packages during a run
or replace a dataset's Bash tool. Both `python` and `python3` must resolve to
the project environment; user-level site-packages are not used as an implicit
fallback.

Before SpreadsheetBench, WikiTQ or HiTab, exercise the real sandboxed Bash
tool and save what the Agent actually sees:

```bash
.venv/bin/python scripts/preflight_agent_runtime.py \
  --mode sandbox \
  --output /runs/preflight/spreadsheet_agent_runtime.json
```

Before Skill2Bench, exercise its real baseline Bash tool:

```bash
.venv/bin/python scripts/preflight_agent_runtime.py \
  --mode skill2bench \
  --baseline-root /data/Trace2Skill_Skill2Bench \
  --output /runs/preflight/skill2bench_agent_runtime.json
```

The check imports the installed `numpy` and `openpyxl` and performs a real
XLSX save/load roundtrip through both command names. It records interpreter,
prefix, base interpreter and dependency versions. It checks functional access
to the selected project environment; it does not require a particular machine
path, Python patch release, package signature or worker count.

## 3. Shared dynamic method

For every logical train batch, all tasks read the same frozen parent graph:

```text
frozen G(k-1)
  -> query + dataset-observable context
  -> active Canonical Top-5 plus at most two direct neighbors per anchor
  -> one contextual applicability/binding call
  -> one Agent execution
  -> verifier
  -> if needed: patch, fresh replay, verifier
  -> one evidence-bounded Reflection call
  -> sequential Canonical resolution in stable task order
  -> atomic G(k) publication
```

Only a normal verified success or the final effective patch plus its successful
fresh replay can support positive experience. Earlier failed replay attempts
and unresolved/runtime failures add no positive nodes or edges. Graph-quality
audits are always written but never decide whether retrieval or evaluation is
allowed to run.

## 4. SpreadsheetBench: train, development, Soft and Hard

### 4.1 Data

Fetch a clean pinned Trace2Skill checkout, or verify an existing checkout:

```bash
.venv/bin/python scripts/fetch_spreadsheetbench.py \
  --checkout /data/Trace2Skill
```

The command checks out Trace2Skill commit
`3d0b52a140f002a512930252b613c49048f7d5ac` and verifies:

- `data/spreadsheetbench_verified/spreadsheetbench_verified_400`;
- `data/all_data_912_v0.1`;
- the ordered train, development and full-population task IDs.

### 4.2 Inspect, then run

The full campaign is the preferred entrypoint. First inspect its 13-stage
plan:

```bash
.venv/bin/python scripts/run_full_campaign.py \
  --profile 9b \
  --run-root /runs/degs_0780_spreadsheet_9b \
  --trace2skill-checkout /data/Trace2Skill \
  --generation-base-url "$GENERATION_URL" \
  --embedding-base-url "$EMBEDDING_URL" \
  --agent-workers 8 \
  --producer-workers 32 \
  --libreoffice-workers 16 \
  --dry-run
```

Remove `--dry-run` to execute. The worker arguments are defaults, not required
values; set them for the available cluster throughput. To run 27B, use
`--profile 27b`, point `GENERATION_URL` at the 27B service, and use a different
run root.

The campaign executes in this 13-stage order:

1. service preflight;
2. real Agent Bash/Python/XLSX runtime preflight;
3. 25×8 dynamic train from empty `G0`;
4. final graph audit;
5. 200 development retrieval/binding bundles;
6. development Agent `[200,400)`;
7. development evaluator with fixed denominator 200;
8. fixed 912-task/2,529-case Soft/Hard population preparation;
9. Soft/Hard retrieval;
10. Soft/Hard bundle structural verification;
11. Soft/Hard Agent cases;
12. LibreOffice recalculation and Soft/Hard evaluation;
13. timing, token and metric aggregation.

### 4.3 Resume and inspect

Re-run the same command and run root to resume. Completed graph batches are
read from SQLite and their public JSON artifacts are reconstructed if missing.
Completed task/case caches are reused only when compatible; invalid cache rows
are regenerated instead of terminating the campaign. Changing an endpoint or
worker count is allowed. Changing dataset, model, split, method prompts/schema,
or logical graph policy requires a new run root.

Primary artifacts:

```text
campaign_manifest.json
agent_runtime.json
status.json
campaign_summary.json
logs/<stage>.log
dynamic_train/state/eir_state.sqlite3
dynamic_train/batches/batch_24/manifest.json
dynamic_train/batches/batch_24/experience_graph.json
graph_quality.json
development_bundle/{bundle_manifest.json,experience.jsonl}
development/{results.jsonl,eval_summary.json,eval_details.json}
soft_hard/agent_run/{results.jsonl,eval_summary.json,soft_hard_report.md}
```

The development headline is read from
`development/eval_summary.json` and must report denominator 200. Soft and Hard
are read from `soft_hard/agent_run/eval_summary.json`; missing outputs,
LibreOffice errors and comparator errors remain in their fixed denominators.

## 5. Skill2Bench: independent Step retrieval, one task Agent

### 5.1 Data and runtime

Checkout `Gen-Verse/Skill-Entropy-RL` at commit
`813a07fb1e4ea629d86196b36022187f13e6c3dd`, and provide the local
Trace2Skill-Skill2Bench runtime and official evaluator roots used by the
baseline comparison. Recreate the fixed split:

```bash
.venv/bin/python scripts/prepare_skill2bench_data.py \
  --source-dir /data/Skill-Entropy-RL/skill2_bench \
  --baseline-root /data/Trace2Skill_Skill2Bench \
  --output-dir /data/degs_skill2bench_seed42
```

The command verifies `train_100.jsonl` and `test_200.jsonl` against the fixed
hashes in `degs_skill2bench.contract`.

### 5.2 Run

Skill2Bench receives generation and embedding key files directly:

```bash
.venv/bin/degs-run-skill2bench \
  --profile 9b \
  --train-path /data/degs_skill2bench_seed42/train_100.jsonl \
  --test-path /data/degs_skill2bench_seed42/test_200.jsonl \
  --run-root /runs/degs_0780_skill2bench_9b \
  --baseline-root /data/Trace2Skill_Skill2Bench \
  --official-evaluator-root /data/Skill-Entropy-RL \
  --generation-base-url "$GENERATION_URL" \
  --generation-api-key-file /path/to/generation_api_key \
  --embedding-base-url "$EMBEDDING_URL" \
  --embedding-api-key-file /path/to/embedding_api_key \
  --agent-workers 8 \
  --producer-workers 32
```

Run the Skill2Bench runtime preflight from section 2.1 first, using the same
`.venv` and `--baseline-root` as the campaign. Its saved JSON is the execution
environment record; the existing baseline Bash implementation remains
unchanged.

Eight complete train tasks arrive per graph batch; the last batch has four.
Every nonempty Step retrieves and learns as an independent episode, while all
Step guidance is assembled in display order for one full-task Agent run. Step
order alone never creates a temporal dependency. Closed and open Steps below
score one are repair eligible; open Steps use the same profile model as judge.

The final metrics are in `result.json`. Supporting artifacts include
`manifest.json`, `graph/eir_state.sqlite3`, `train/dynamic/`,
`retrieval/bundle/`, `test/`, and `usage/`. Re-run the same command to resume;
malformed per-item caches are regenerated. A different model, train/test split
or method definition uses another run root.

## 6. WikiTQ and HiTab read-only OOD transfer

Checkout the evaluator/data repositories at the commits in
`degs.ood_dataset.SOURCE_SPECS`:

- WikiTQ: `7d455a5a707b96341ef72aff9428749d443d8aa9`;
- HiTab: `d179602662b490249baf068a76fbe4137029126e`.

Use the final SpreadsheetBench graph from the same model profile:

```bash
.venv/bin/python scripts/run_tableqa_ood.py \
  --profile 9b \
  --source-dataset-path /data/Trace2Skill/data/spreadsheetbench_verified/spreadsheetbench_verified_400/dataset.json \
  --snapshot-manifest-path /runs/degs_0780_spreadsheet_9b/dynamic_train/batches/batch_24/manifest.json \
  --state-db /runs/degs_0780_spreadsheet_9b/dynamic_train/state/eir_state.sqlite3 \
  --wikitq-source-repo /data/WikiTableQuestions \
  --hitab-source-repo /data/HiTab \
  --run-root /runs/degs_0780_ood_9b \
  --generation-base-url "$GENERATION_URL" \
  --embedding-base-url "$EMBEDDING_URL" \
  --python2 /usr/bin/python2
```

Run the sandbox runtime preflight from section 2.1 on the OOD execution
machine first. WikiTQ and HiTab use that same Bash sandbox but keep separate
task, cache, output and evaluator directories.

Set `DEGS_AGENT_WORKERS` and `DEGS_PRODUCER_WORKERS` before this command to
change runtime concurrency. Each dataset executes population preparation,
Top-5 contextual retrieval, Agent generation and its official evaluator.
Neither dataset updates the SpreadsheetBench graph.

Results:

```text
wikitq/evaluation/eval_summary.json
hitab/evaluation/eval_summary.json
completed.json
```

The OOD run root is bound to the selected source graph, model and source
populations so a different graph cannot silently reuse these results. Endpoint
changes are operational and do not invalidate already completed results.

## 7. Validation checklist

Code and offline contracts:

```bash
git diff --check
PYTHONPATH=. .venv/bin/python -m pytest -q
```

For every formal run, verify all of the following from actual artifacts:

1. the run manifest names the intended dataset, model and population;
2. every expected stage/task/case is present;
3. the final graph snapshot is the stated graph HEAD;
4. graph audit was saved but did not gate retrieval;
5. target gold was unavailable to retrieval and Agent generation;
6. the official evaluator completed and the reported denominator is fixed;
7. failure counts distinguish Agent/runtime, missing output, recalculation,
   comparator and verifier/judge failures;
8. timing and token usage exist before comparing cost or throughput.

Service smoke tests and partial runs prove only that the pipeline can execute.
Only complete fixed-population evaluator output is a benchmark result.
