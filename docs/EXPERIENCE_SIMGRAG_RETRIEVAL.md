# 0.78.0 contextual graph retrieval

This file records the replacement of the historical Experience-SimGRAG online
path. The formal 0.78.0 retriever is intentionally bounded and graph-native.

## Recall

For query-context embedding `q` and every active Canonical document embedding
`c_i`, rank by cosine similarity and keep five unique Canonical anchors. There
is no similarity rejection threshold. For each anchor, add at most two direct
neighbors as context, preferring one incoming and one outgoing node and ranking
eligible neighbors by similarity to the same `q`. The payload contains only the
induced edges among selected anchors and context nodes.

## Binding

One strict-JSON LLM call receives the original query, observable current-task
evidence, the five anchors with active versions, and the one-hop context. It
must produce one expectation per anchor:

- `SATISFIED`: applicability is observed; bind parameters from current evidence;
- `UNKNOWN`: state the check required before conditional use;
- `CONFLICT`: reject the anchor and emit no guidance.

All condition and parameter references must name supplied current-task evidence
IDs. Context neighbors cannot be selected as anchors. Empty graph yields empty
guidance without a binding call. Rejecting all five is valid.

## Injection and audit

Only validated non-conflicting guidance is rendered for the Agent. Raw source
constants, source trajectories, similarity scores, rejected nodes and hidden
verifier data are not injected. Every bundle retains retrieval nodes, active
versions, induced edges, binding decisions, failures, graph/snapshot identity
and a content hash.

The same implementation is used for train, development, Soft/Hard,
Skill2Bench Step queries and read-only WikiTQ/HiTab transfer.
