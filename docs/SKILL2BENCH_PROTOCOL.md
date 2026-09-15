# Skill2Bench protocol for DEGS 0.78.0

## Population and evaluator

- Source: `Gen-Verse/Skill-Entropy-RL` commit
  `813a07fb1e4ea629d86196b36022187f13e6c3dd`.
- Fixed entropy-stratified seed 42: train 100 / test 200.
- Train SHA-256:
  `7f0e2350db71b87b7f36e965fdf6e534afbf2b541f615818751ace61d10b93c2`.
- Test SHA-256:
  `d3d75c21a57b1173a28849d811424db7c116e47e3f8a6d77f97c53fedacff9c9`.
- Closed Steps use the pinned official evaluator. Open Steps use the same
  Qwen3.5 profile model as judge. Gold solutions are unavailable until after
  generation.

`--baseline-root` points to the fixed local runtime adapted from Trace2Skill
commit `3d0b52a140f002a512930252b613c49048f7d5ac`. Its Python-tree SHA-256 is
`8c96fc11ad019096ebf06b5d3f77280f7d19c9610051c39c67979bd41b15a5f6`.
DEGS behavior lives in `src/degs_skill2bench`; upstream code is not modified.

## Dynamic train

Eight complete tasks arrive per batch (12×8 plus one final batch of 4). Every
nonempty Step becomes an EIR episode with stable slot:

```text
step_slot = train_task_index * 10 + (step_number - 1)
```

All Steps in a batch read the same frozen parent graph. Each Step independently
runs Top-5 + one-hop retrieval and contextual binding. Step guidance is
assembled in original display order, with an explicit instruction that Steps
are independent questions sharing a scenario unless a question itself names a
dependency. The complete task then runs exactly once.

Every scored Step below one—closed or open—is eligible for up to three patch +
fresh full-task replay attempts. Only the final patch whose target Step reaches
score one and that successful replay may be positive evidence. Reflection
remains Step-scoped: it receives the target Step, its observable scenario, its
retrieval/expectation, attributable original execution, and optional accepted
repair. Other Steps in the full replay do not authorize new experience.

One Reflection reconciles retrieved anchors and extracts residual successful
micro-operations. Valid Step deltas are Canonicalized and committed in stable
slot order. Missing Steps and unobservable/runtime failures add no positive
nodes or edges.

Agent/replay concurrency defaults to 8 and producer concurrency defaults to 32;
both are positive runtime parameters, not acceptance gates. Maximum turns is
30 and thinking is false. Agent/replay calls omit explicit `max_tokens` to
match the local baseline; structured binding/reflection/Canonical producers
use 32,000.

## Test

Each nonempty test Step independently executes the shared 0.78.0 path:

```text
target Step + scenario context
  -> active Canonical Top-5 + bounded one-hop context
  -> contextual binding
  -> Step guidance
```

The task's Step guidance blocks are assembled in Step order and injected once
into one full-task Agent run. No Skill2Bench-only retrieval threshold,
abstention gate, Selector, NeedGraph or C0 exists.

## Isolation and outputs

9B and 27B own separate run roots, EIR state, embedding cache, bundle,
trajectories and results. A run saves per-batch retrieval/binding, rollout,
official evaluation, repair evidence, EpisodeEvidence, LearningDelta, snapshot,
graph, usage/timing and audit artifacts. Graph quality is diagnostic and never
blocks test execution.
