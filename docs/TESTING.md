# Testing

Run the offline suite:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
git diff --check
```

The suite covers source/replay admission, prompt schemas, incremental Canonical updates, graph topology audits, Experience-SimGRAG, empty-cache online producer execution, bundle verification, Agent resume, Soft/Hard population and evaluation, and WikiTQ/HiTab adapters.

Before a real campaign, run service preflight:

```bash
.venv/bin/python scripts/preflight_services.py \
  --generation-base-url http://host:port/v1 \
  --generation-model Qwen3.5-9B-AWQ \
  --embedding-base-url http://host:port/v1 \
  --embedding-model Qwen3-Embedding-8B
```

An API smoke proves only connectivity and response shape. A full benchmark claim requires the fixed population, complete Agent run and matching evaluator output.
