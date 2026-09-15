# DEGS current version

## Identity

| Item | Value |
| --- | --- |
| Method | **DEGS 0.78.0 Evidence-Bounded EIR Dynamic** |
| Package | `degs==0.78.0` |
| Branch | `codex/degs-0780-eir-dynamic` |
| Repository | `https://github.com/Feesuu/DEGS` |
| Local root | `/mnt/data/yaodong/skill-weaver/DEGS` |
| Historical baseline | `0.77.41 Stable R1`, commit `2338d597945f6162356eba3e244e69fe0da50bdc` |

## Formal method

For each batch of eight arriving train tasks, all tasks read one frozen graph
`G(k-1)`. The online path is:

```text
query + observable context
  -> active Canonical Top-5 + at most two one-hop neighbors per anchor
  -> one contextual binding call
  -> one Agent rollout
  -> verifier
  -> optional final patch + one successful fresh replay
  -> one evidence-bounded Reflection
  -> ordered LearningDelta commit
  -> G(k)
```

An ExperienceNode is a conditional micro-operation with an observable guard,
a current-task parameter-binding rule, a locally executable operation and an
expected state transition. Retrieved nodes are hypotheses: binding may reject
all of them. Reflection records `NO_EVIDENCE`, `SUPPORT`, `QUALIFY` or
`CORRECT`, extracts only residual successful behavior, and connects the actual
successful procedure across workflows.

Canonical IDs remain stable. Content changes create new versions; historical
snapshots retain their old active version. Normal success cannot revise or
broaden existing experience. `QUALIFY/CORRECT` require verifier-grounded repair
success.

## Dataset routes

| Dataset | Learning | Evaluation |
| --- | --- | --- |
| SpreadsheetBench | model-specific `[0,200)`, empty `G0`, 25×8 dynamic batches | `[200,400)`, denominator 200; then fixed 912-task / 2,529-case Soft/Hard |
| Skill2Bench | model-specific seed-42 train-100, batches of 8 tasks, Step-scoped evidence | seed-42 test-200; each nonempty Step retrieves independently, one full-task Agent run |
| WikiTQ / HiTab | no target-domain learning | read-only transfer from the matching model's frozen SpreadsheetBench graph |

No model or dataset shares mutable state or generated artifacts. OOD routes are
the only intentional cross-dataset read and cannot update the source graph.

## Fixed experiment identity

- Models: `Qwen3.5-9B-AWQ` and `Qwen3.5-27B-AWQ`, isolated profiles.
- Embedding: `Qwen3-Embedding-8B`, cache keyed by normalized-text SHA-256.
- Spreadsheet Agent/replay workers: 8; Binding/Reflection workers: 32;
  Canonical view workers: 32; Canonical merge candidate wave: up to 16.
- Spreadsheet Agent and producer completion limit: 32,000; server context:
  100,000; Agent turns: 30; temperature: 0; thinking: false.
- Spreadsheet evaluation uses the pinned comparator after LibreOffice
  recalculation and preserves the fixed denominator.
- Graph audit is diagnostic and never blocks retrieval or evaluation.

## Result status

No 0.78.0 benchmark score is claimed at this code checkpoint. The accepted
`88/200 = 44.0%` SpreadsheetBench 9B result belongs to historical 0.77.41 and
must not be relabeled as 0.78.0. Because 0.78.0 changes train behavior,
prompts, Canonical identity and retrieval, its graph and all downstream bundles
and results must be regenerated from empty `G0`.
