# WikiTQ and HiTab read-only transfer protocol

- WikiTQ source commit: `7d455a5a707b96341ef72aff9428749d443d8aa9`.
- HiTab source commit: `d179602662b490249baf068a76fbe4137029126e`.
- Population preparation and evaluators remain the pinned local baseline
  implementation: the upstream WikiTQ evaluator and HiTab `hmt_score`.

An OOD run reads the final committed SpreadsheetBench 0.78.0 EIR state for the
same model profile. Each query is projected with its dataset-observable table
context, then uses the shared active-Canonical Top-5, bounded one-hop context
and contextual binding implementation. The resulting guidance is injected into
the unchanged OOD Agent interface.

WikiTQ and HiTab never create episodes, LearningDeltas, Canonical versions or
edges. Their retrieval bundle, cache, Agent output and evaluator output are
isolated from each other and from SpreadsheetBench. Target gold, answer,
verifier result and trace are unavailable to retrieval and can only be consumed
by the official evaluator after generation.

Use `scripts/run_tableqa_ood.py` with the matching SpreadsheetBench state and
final batch manifest. The script executes prepare, contextual bundle, Agent and
official evaluation independently for both datasets and records the 0.78.0
method identity.
