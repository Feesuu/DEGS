# Version history

## DEGS 0.77.41 Stable R1

- Single formal runtime; historical retrieval stages are folded into one online implementation.
- SpreadsheetBench 9B development `[200,400)`: `88/200 = 44.0%` in the accepted audited run.
- Experience graph: 421 source nodes, 271 Canonical nodes, 290 projected occurrence edges.
- Retrieval: query-only V2 NeedGraph, operation top-8, workflow top-8, beam32, input-cited clarification, complete-candidate workbook-role late fusion, deterministic C0 with fallback Selector.
- The clean runtime preserves the accepted graph/retrieval semantics. The `88/200` score belongs to the audited source run; a fresh end-to-end run from this repository is required before reporting it as a clean-repository reproduction.
- Cache is optional; the empty retrieval-cache integration test calls all online producers.

Historical experimental version numbers are intentionally not public runtime choices. Their results remain research notes outside the formal execution path.
