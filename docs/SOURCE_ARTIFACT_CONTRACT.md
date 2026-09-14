# Source artifact contract

Each workflow contains `train_index`, `task_id`, `query_text`, `experience_nodes` and `edges`.

Each ExperienceNode contains:

- `operation`: one independently transferable causal micro-operation;
- `applicability`: conditions under which the operation applies;
- `inputs`: open `type`/`description` contracts;
- `outputs`: open `type`/`description` contracts.

Edges use zero-based `source → target` node indices. A malformed individual edge is discarded without discarding valid nodes or the whole trajectory.

Source extraction reads the complete saved train evidence without character truncation. The reviewed graph is published globally, then deterministically split into 25 batches of 8 train indices. Prompt changes require regenerating every source artifact produced by that prompt.

Canonicalization is monotonic: a later batch can merge a new node into an existing Canonical or unite existing groups when the node operations are the same transferable template, but it cannot split a previously accepted group. Predecessors and successors are not node-identity requirements. ExperienceGraph edges are projections of real source occurrence edges.
