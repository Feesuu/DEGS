# New-machine Agent handoff

This file is the copy-paste task for an Agent running DEGS experiments on a
new cluster. Operational details and exact commands are in `docs/RUNBOOK.md`;
experiment invariants are in `AGENTS.md` and `docs/EXPERIMENT_PROTOCOL.md`.

## Prompt to give the new-machine Agent

```text
You are taking over the DEGS 0.78.0 Evidence-Bounded EIR Dynamic experiment.
Work from evidence in the live repository and complete the runs end to end;
do not redesign the method while reproducing it.

Repository and version
1. Clone https://github.com/Feesuu/DEGS.git into a clean directory.
2. Checkout the immutable `degs-0.78.0-runtime-a` tag and record the resolved
   commit from `git rev-parse HEAD` in every experiment manifest/report.
3. Read AGENTS.md, docs/RUNBOOK.md, docs/EXPERIMENT_PROTOCOL.md,
   docs/METHOD_BOUNDARY.md and docs/NEW_MACHINE_AGENT_HANDOFF.md completely.
4. Keep all data, model weights, API keys, caches and run outputs outside Git.

Services
1. Deploy OpenAI-compatible Qwen3.5-9B-AWQ and Qwen3.5-27B-AWQ services and a
   Qwen3-Embedding-8B service. Use the actual endpoints and key files supplied
   on this cluster; never copy endpoint paths or secrets from another machine.
2. Generation services must expose max_model_len=100000 and thinking=false.
   Select tensor parallelism, replicas, max_num_seqs and request concurrency
   for the maximum stable throughput of the available GPUs; do not hard-code
   the old cluster's concurrency and do not reduce the 100000-token context.
3. Use the repository's dataset-specific request settings. Do not silently
   change model identity, decoding, completion policy, turns, prompts,
   retrieval, logical train batch size, split, evaluator or denominator.

Environment and data
1. Create the Python 3.12 `.venv` and install `.[test]` as documented.
2. Install the host requirements used by the official paths: bubblewrap,
   LibreOffice/soffice, and WikiTQ's Python 2 evaluator runtime.
3. Fetch or verify every dataset and upstream runtime at the pinned commits in
   docs/RUNBOOK.md. Do not substitute a similar dataset dump.
4. Run `scripts/preflight_services.py` against every selected model endpoint.
5. Run `scripts/preflight_agent_runtime.py --mode sandbox` and save its JSON.
   It must exercise both `python` and `python3`, import numpy/openpyxl and pass
   the XLSX roundtrip through the real bubblewrap Bash tool.
6. Before Skill2Bench, run the same script with `--mode skill2bench` and the
   exact baseline root. Preserve Skill2Bench's baseline host Bash protocol.

Isolation and execution order
1. Use separate run roots for every dataset and model profile. Never share a
   state database, graph, retrieval cache, bundle, trajectory or evaluator
   output between 9B and 27B or between SpreadsheetBench and Skill2Bench.
2. Run the full SpreadsheetBench campaign for 9B. It must start from empty G0,
   learn from train [0,200) as 25 logical batches of 8, run development
   [200,400) with denominator 200, then run the fixed Soft/Hard population and
   LibreOffice evaluation.
3. Run the identical SpreadsheetBench workflow for 27B in a new run root.
4. Run Skill2Bench 9B and 27B independently. Each profile builds its own graph
   from the fixed seed-42 train-100 split and evaluates the fixed test-200
   split. Each nonempty Step retrieves independently; the full task runs one
   Agent. Skill2Bench must not read a SpreadsheetBench graph.
5. Run WikiTQ and HiTab for 9B and 27B. These are read-only OOD consumers of
   the matching model's final SpreadsheetBench graph: 9B reads only the 9B
   graph and 27B reads only the 27B graph. OOD evaluation must never update the
   graph or expose target answers, gold or evaluator output to retrieval.
6. Use the commands in docs/RUNBOOK.md. Choose positive worker counts that
   saturate the deployed services without OOM or systemic timeouts. Logical
   eight-task train batches remain unchanged regardless of worker count.

Failure and resume policy
1. Run long jobs under tmux or an equivalent scheduler and record the exact
   command, commit, endpoints without keys, worker counts and output root.
2. Item-local completion, schema, context, runtime or evaluator failures must
   be recorded and the remaining population must continue. Do not lower the
   denominator or silently drop tasks.
3. If a shared service, dependency or runtime is broken, diagnose health,
   server logs, proxy/bridge, disk, permissions, Python identity, Bash trace,
   LibreOffice and evaluator evidence before blaming model capability.
4. Fix only a general runtime defect that blocks the documented protocol, add
   a minimal regression test, then resume the same compatible run. Do not tune
   prompts, retrieval or graph logic from development/test outcomes.
5. A graph-quality audit is diagnostic and must always be saved, but it must
   never gate retrieval, Agent execution or evaluation.

Completion and report
1. Run `git diff --check` and `PYTHONPATH=. .venv/bin/python -m pytest -q`.
2. For each dataset/model, report the run root, completion status, population,
   denominator, metric, failed-item breakdown, model endpoint identity,
   evaluator/LibreOffice version, elapsed time, LLM call count and token use.
3. Distinguish environment/runtime, missing output, model/Agent execution,
   recalculation, comparator/verifier and judge failures using actual traces.
4. Do not call smoke tests or partial runs benchmark results. Completion means
   all requested fixed-population evaluators have written their final result
   artifacts, even when the measured score is poor.
5. Commit and push only general code/documentation fixes. Never commit API
   keys, datasets, model weights, generated caches or raw private run data.

Start by returning a short execution matrix containing the six run roots
(SpreadsheetBench 9B/27B, Skill2Bench 9B/27B, OOD 9B/27B), discovered service
endpoints/model identities, available GPU layout, selected stable worker
counts, and the preflight result. Then start the campaigns without waiting for
micro-approvals. Ask only if a missing user choice would change the dataset,
model, method, evaluator or other formal experiment semantics.
```

## Active method prompt ownership

Do not paste or reconstruct method prompts in the handoff message. The active
prompt resources are versioned with the code:

- Spreadsheet/general EIR: `src/degs/resources/CONTEXTUAL_BINDING_PROMPT_V1.txt`,
  `EPISODE_REFLECTION_PROMPT_V1.txt`, `SOURCE_REPLAY_PATCH_PROMPT_V2.txt`,
  `CANONICALIZATION_VIEW_PROMPT_V5.txt`, and
  `CANONICAL_OPERATION_MERGE_PROMPT_V3.txt`;
- Skill2Bench-specific additions: files under
  `src/degs_skill2bench/resources/`;
- Agent system prompts: files under `src/spreadsheet_agent/system_prompt/`.

The code records their hashes where they affect method identity. Files with
older version suffixes may remain for historical compatibility; the new
machine Agent must invoke the documented entrypoints instead of selecting
prompt files manually.
