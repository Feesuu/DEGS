# Skill2Bench protocol

## Population and evaluator

- Source: `Gen-Verse/Skill-Entropy-RL` commit
  `813a07fb1e4ea629d86196b36022187f13e6c3dd`.
- Split: entropy-stratified seed 42, train 100 and test 200, with ordered IDs
  recorded in `configs/skill2bench/split_manifest.json`.
- Required file identities:
  - train: `7f0e2350db71b87b7f36e965fdf6e534afbf2b541f615818751ace61d10b93c2`;
  - test: `d3d75c21a57b1173a28849d811424db7c116e47e3f8a6d77f97c53fedacff9c9`.
- Closed Steps use the pinned official evaluator. Open Steps use the same
  Qwen3.5 profile model as judge. Evaluation sees solutions only after Agent
  generation.

`--baseline-root` must point to the fixed local baseline runtime derived from
Trace2Skill commit `3d0b52a140f002a512930252b613c49048f7d5ac`. The campaign hashes its
Python runtime tree and the official evaluator dependency closure into the run identity, so a
resume cannot mix another runtime. DEGS-specific behavior lives only in
`src/degs_skill2bench`; upstream runtime source is not copied into this repo.
The accepted adapted archive SHA-256 is
`deda1711ce6cd2352c78b2845447c6829e29ae3b9e554ae9bfe9f5227b0c7c58`;
its expanded Python-tree SHA-256 is
`8c96fc11ad019096ebf06b5d3f77280f7d19c9610051c39c67979bd41b15a5f6`.

## Train and graph construction

Each of the 100 complete train tasks is run once without learned experience;
the dataset execution profile is still present so Step independence is
consistent in train, replay, and test. Agent concurrency is 8, maximum turns
is 30, thinking is false, and Agent requests omit `max_tokens`.

Every scored Step with official score below one is a failure, regardless of
whether it is closed or open. It receives up to three target-Step patch plus
fresh full-task replay attempts. A replay is accepted when that target Step
changes to official score one. Earlier failed patches/replays are not source
input.

The graph has 1,000 stable Step slots:

```text
workflow_id = train_task_index * 10 + (step_number - 1)
```

Eight arriving tasks therefore form 80 slots. There are 12 full batches and
one final four-task/40-slot batch. Missing Steps and Steps without accepted
evidence are recorded exclusions rather than fabricated workflows.

For original success, extraction uses only an explicitly attributable target
Step trace fragment. For repair success, extraction receives exactly:

```text
target Step question
+ target Step validated {instructions, checks} patch
+ complete successful full-task replay
```

The full replay is retained without truncation so the realized operation can
be checked, but it is not a license to learn other Steps. The extractor emits
only micro-operations jointly supported by the patch and replay and local
causal edges between those operations. If the intersection cannot be shown,
that Step contributes no source workflow.

## Retrieval and Agent execution

Every non-empty test Step independently runs the existing DEGS 0.77.41 chain:

```text
target Step query
  -> Step-profiled NeedGraph
  -> workflow top-8 + Need-to-Canonical top-8
  -> beam-32 Experience-SimGRAG
  -> deterministic C0 (existing fallback only)
```

There is no additional similarity threshold, rejection gate, abstention or
dataset-only Selector. Retrieved experiences are assembled in original Step
order and injected once into the full-task system prompt. The dataset profile
states that Steps are independent questions sharing a scenario unless a
target Step explicitly names an earlier result.

## Run identity and outputs

9B and 27B use separate run roots. A run contains train rollout/evaluation,
repairs, Step evidence, 13 source batches, transactional graph state and
snapshots, graph audit, target-local retrieval cache, per-Step bundle,
full-task test rollouts, official evaluation and aggregate metrics. Stage
timing and LLM usage are written under `usage/`.

The graph-quality audit is diagnostic. READY/NOT_READY never decides whether
retrieval or test execution runs.
