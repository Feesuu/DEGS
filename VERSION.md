# DEGS current version

## Identity

| Item | Value |
| --- | --- |
| Method name | **DEGS — Dynamic Experience Graph for Skills** |
| Formal version | **0.77.41 Stable R1** |
| Repository | `https://github.com/Feesuu/DEGS` |
| Development branch | `codex/degs-multidataset-adapters` |
| Standalone baseline commit | `2338d597945f6162356eba3e244e69fe0da50bdc` |
| Standalone local directory | `/mnt/data/yaodong/skill-weaver/DEGS` |
| Python package version | `0.77.41` |

This directory is the only active development copy. Future code, prompt,
schema, workflow, protocol, and documentation changes start here. The parent
`skill-weaver` repository and its historical worktrees/runs are evidence only;
they are not runtime dependencies of this copy.

## Fixed method

DEGS extracts causal, reusable micro-operations from two kinds of training
evidence:

1. an original trajectory that passes its verifier; or
2. the final effective repair patch together with the single fresh replay
   trajectory that passes after applying that patch.

Training evidence arrives in batches. New ExperienceNodes are incrementally
merged into monotonic Canonical groups using node-local operation semantics,
and real source-workflow relations are projected as graph edges. At test time,
DEGS builds a NeedGraph online, performs Experience-SimGRAG retrieval over the
frozen source graph, and injects the retrieved experience into the task Agent.

The formal online retrieval configuration is:

- query-only V2 NeedGraph;
- Need-to-Canonical top-8;
- workflow top-8;
- beam size 32;
- input-cited clarification and workbook-role late fusion;
- deterministic top-ranked C0 on the normal path, with Selector only as the
  defined fallback.

## Dataset routes

| Dataset | Source graph | Target protocol |
| --- | --- | --- |
| SpreadsheetBench | model-specific train `[0,200)`, 25 batches of 8 | development `[200,400)`, fixed denominator 200; Full Soft/Hard uses the fixed 912-task population |
| Skill2Bench | model-specific seed-42 train-100, Step-scoped source workflows | seed-42 test-200; each Step retrieves independently, then experiences are injected in Step order into one full-task Agent run |
| WikiTQ | read-only graph built from the matching model's SpreadsheetBench train split | official WikiTQ data/evaluator; no OOD graph update |
| HiTab | read-only graph built from the matching model's SpreadsheetBench train split | official HiTab data/evaluator; no OOD graph update |

Every dataset and model profile owns an isolated run directory, graph/state,
retrieval cache, bundle, Agent output, and evaluation output. Only the explicit
read-only OOD transfer above crosses dataset boundaries.

## Fixed experimental identity

- Models: `Qwen3.5-9B-AWQ` and `Qwen3.5-27B-AWQ`, with completely separate
  generated artifacts.
- Embeddings: `Qwen3-Embedding-8B`.
- Server context: 100,000 tokens.
- Spreadsheet Agent/extraction/Canonical/retrieval completion limit: 32,000
  tokens per request.
- Spreadsheet repair-patch/replay completion limit: 16,384 tokens per request.
- Agent turns: 30; temperature: 0; thinking: false.
- Logical train graph arrival batch: 8 tasks. Runtime request concurrency is a
  deployment setting and may be raised to the maximum stable cluster
  throughput without changing the logical batch size.
- Spreadsheet results are evaluated after LibreOffice recalculation with the
  fixed evaluator and denominator.
- Cache is optional acceleration only. Empty caches must execute all online
  LLM and embedding producers.

## Audited result attached to this version

The accepted SpreadsheetBench 9B development result is **88/200 = 44.0%**.
It belongs to the audited source run recorded in the version history. It must
not be presented as a fresh reproduction from this standalone clone until the
full clean-repository campaign is run.

The associated audited 9B graph contains:

- 421 source ExperienceNodes;
- 271 Canonical nodes;
- 290 projected occurrence edges.

No completed Skill2Bench, WikiTQ, or HiTab score is claimed by this repository
checkpoint. Smoke, partial, cached, or selected-slice runs are not formal
benchmark results.

## Change rule

Any future version must update this file and `docs/VERSION_HISTORY.md` in the
same semantic checkpoint. If a producer prompt changes, every artifact created
by that prompt and all dependent downstream artifacts must be regenerated.
Splits, denominators, evaluator, model profile, graph/retrieval hyperparameters,
or information-access boundaries must never change silently.

Detailed execution instructions are in `docs/EXPERIMENT_PROTOCOL.md`; method
boundaries are in `docs/METHOD_BOUNDARY.md`; multi-dataset implementation and
state isolation are in `docs/IMPLEMENTATION_PLAN_MULTIDATASET.md`.
