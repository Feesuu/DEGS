"""Shared online retrieval for evaluation populations outside development[200,400)."""

from __future__ import annotations

import asyncio
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

from . import bundle as retrieval
from .core import StrictEmbeddingAdapter, canonical_json_bytes
from .dataset import (
    EXPECTED_QUERY_PROJECTION_SHA256,
    _load_train_harness_records,
    load_train_queries,
    query_projection_sha256,
)
from .runtime_identity import METHOD_CONTRACT, METHOD_VERSION
from .retrieval_store import RetrievalStore
from .transport import seal_train_instruction_authority
from .validated_repair import (
    REPAIR_SOURCE_MAX_TOKENS,
    REPAIR_SOURCE_MODEL,
    REPAIR_SOURCE_TEMPERATURE,
    REPAIR_SOURCE_THINKING,
    REPAIR_SOURCE_TIMEOUT_SECONDS,
)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _protocols(need_llm: Any, clarification_llm: Any, selector_llm: Any) -> dict[str, Any]:
    specs = (
        (
            "need_producer",
            need_llm,
            retrieval.NEED_GRAPH_PROTOCOL_FORMAT,
            retrieval.NEED_GRAPH_KIND,
            retrieval.NEED_GRAPH_PROMPT_SHA256,
        ),
        (
            "clarification_producer",
            clarification_llm,
            retrieval.CLARIFICATION_PROTOCOL_FORMAT,
            retrieval.CLARIFICATION_KIND,
            retrieval.CLARIFICATION_PROMPT_SHA256,
        ),
        (
            "selector_producer",
            selector_llm,
            retrieval.SELECTOR_PROTOCOL_FORMAT,
            retrieval.SELECTOR_KIND,
            retrieval.SELECTOR_PROMPT_SHA256,
        ),
    )
    producers = {
        name: retrieval._validated_llm_protocol(
            producer.protocol_identity,
            expected_format=format_id,
            expected_kind=kind,
            expected_prompt_sha256=prompt_sha,
            expected_retry_waits=retrieval.BUNDLE_TRANSPORT_RETRY_WAITS,
            expected_runtime_timeout_retries=(
                retrieval.BUNDLE_RUNTIME_TIMEOUT_RETRIES
            ),
        )
        for name, producer, format_id, kind, prompt_sha in specs
    }
    if len({row["service_url"].rstrip("/") for row in producers.values()}) != 1:
        raise ValueError("retrieval producers must use the same LLM service")
    return {
        "model": REPAIR_SOURCE_MODEL,
        "temperature": REPAIR_SOURCE_TEMPERATURE,
        "thinking": REPAIR_SOURCE_THINKING,
        "max_tokens": REPAIR_SOURCE_MAX_TOKENS,
        "timeout_seconds": REPAIR_SOURCE_TIMEOUT_SECONDS,
        "semantic_attempts": retrieval.SEMANTIC_ATTEMPTS,
        "async_workers": retrieval.LLM_ASYNC_WORKERS,
        "need_prompt_sha256": retrieval.NEED_GRAPH_PROMPT_SHA256,
        "clarification_prompt_sha256": retrieval.CLARIFICATION_PROMPT_SHA256,
        "selector_prompt_sha256": retrieval.SELECTOR_PROMPT_SHA256,
        **producers,
    }


def _train_context(source_dataset_path: Path, snapshot_manifest_path: Path, state_db_path: Path):
    train_queries = load_train_queries(source_dataset_path)
    if query_projection_sha256(train_queries) != EXPECTED_QUERY_PROJECTION_SHA256:
        raise ValueError("train query projection differs")
    train_records = _load_train_harness_records(source_dataset_path)
    authority = seal_train_instruction_authority(
        [
            {"id": row["task_id"], "instruction": row["instruction"]}
            for row in train_queries
        ]
    ).authority
    context = retrieval._load_snapshot(
        snapshot_manifest_path,
        state_db_path=state_db_path,
        train_queries=train_queries,
    )
    train_cards = retrieval._train_evidence_cards(
        train_records, dataset_path=source_dataset_path
    )
    return train_queries, authority, context, train_cards


async def _build_async(
    *,
    format_id: str,
    claim_scope: str,
    population_identity: Mapping[str, Any],
    tasks: Sequence[Any],
    target_dataset_path: Path,
    source_dataset_path: Path,
    snapshot_manifest_path: Path,
    state_db_path: Path,
    output_dir: Path,
    retrieval_cache_path: Path | None = None,
    embedding_transport: Any,
    need_llm: Any,
    clarification_llm: Any,
    selector_llm: Any,
) -> Any:
    if not tasks:
        raise ValueError("retrieval population must not be empty")
    llm_protocol = _protocols(need_llm, clarification_llm, selector_llm)
    train_queries, authority, context, train_cards = _train_context(
        source_dataset_path, snapshot_manifest_path, state_db_path
    )
    target_cards = retrieval._target_evidence_cards(
        tasks, dataset_path=target_dataset_path
    )
    output = output_dir.expanduser().absolute()
    if output.exists():
        raise FileExistsError("retrieval bundle output directory must be fresh")
    output.parent.mkdir(parents=True, exist_ok=True)
    cache_path = (
        output.parent / "retrieval_cache.sqlite3"
        if retrieval_cache_path is None
        else retrieval_cache_path.expanduser().absolute()
    )
    with RetrievalStore(cache_path) as state:
        state.bind_embedding_endpoint(getattr(embedding_transport, "endpoint", ""))
        embedder = StrictEmbeddingAdapter(
            embedding_transport, cache=state.embedding_cache()
        )
        results, embedding_audit = await retrieval._build_tasks_clean_async(
            tasks=tasks,
            train_queries=train_queries,
            context=context,
            state=state,
            embedder=embedder,
            need_llm=need_llm,
            clarification_llm=clarification_llm,
            selector_llm=selector_llm,
            target_evidence_cards=target_cards,
            train_evidence_cards=train_cards,
        )
    rows, task_audits = retrieval._rows(
        tasks=tasks,
        results=results,
        authority_sha256=authority.authority_sha256,
    )
    experience_bytes = b"".join(
        canonical_json_bytes(row) + b"\n" for row in rows
    )
    body = {
        "format": format_id,
        "method_name": retrieval.METHOD_NAME,
        "method_version": METHOD_VERSION,
        "method_family": retrieval.METHOD_FAMILY,
        "method_contract": METHOD_CONTRACT,
        "claim_scope": claim_scope,
        "population": dict(population_identity),
        "fixed_denominator": len(tasks),
        "authority_sha256": authority.authority_sha256,
        "train_query_projection_sha256": EXPECTED_QUERY_PROJECTION_SHA256,
        "snapshot": dict(context.identity),
        "experience_graph": context.graph.identity(),
        "train_workflow_evidence_sha256": retrieval._evidence_cards_sha256(
            train_cards
        ),
        "embedding_protocol": embedding_audit,
        "llm_protocol": llm_protocol,
        "tasks": task_audits,
        "experience_file": "experience.jsonl",
        "experience_sha256": _sha(experience_bytes),
        "row_count": len(rows),
        "status_counts": dict(
            sorted(Counter(result.status for result in results).items())
        ),
    }
    manifest = {**body, "self_sha256": _sha(canonical_json_bytes(body))}
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    published = False
    try:
        retrieval._write_exclusive(staging / "experience.jsonl", experience_bytes)
        retrieval._write_exclusive(
            staging / "bundle_manifest.json", canonical_json_bytes(manifest)
        )
        os.replace(staging, output)
        published = True
    finally:
        if not published:
            shutil.rmtree(staging)
    return verify_bundle(
        format_id=format_id,
        claim_scope=claim_scope,
        population_identity=population_identity,
        tasks=tasks,
        target_dataset_path=target_dataset_path,
        source_dataset_path=source_dataset_path,
        snapshot_manifest_path=snapshot_manifest_path,
        state_db_path=state_db_path,
        output_dir=output,
    )


def build_bundle(**kwargs: Any) -> Any:
    return asyncio.run(_build_async(**kwargs))


def verify_bundle(
    *,
    format_id: str,
    claim_scope: str,
    population_identity: Mapping[str, Any],
    tasks: Sequence[Any],
    target_dataset_path: Path,
    source_dataset_path: Path,
    snapshot_manifest_path: Path,
    state_db_path: Path,
    output_dir: Path,
) -> Any:
    train_queries, authority, context, train_cards = _train_context(
        source_dataset_path, snapshot_manifest_path, state_db_path
    )
    target_cards = retrieval._target_evidence_cards(
        tasks, dataset_path=target_dataset_path
    )
    root = output_dir.expanduser().absolute()
    if not root.is_dir() or {path.name for path in root.iterdir()} != {
        "bundle_manifest.json",
        "experience.jsonl",
    }:
        raise ValueError("retrieval bundle files differ")
    manifest_bytes = (root / "bundle_manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    experience_bytes = (root / "experience.jsonl").read_bytes()
    lines = experience_bytes.splitlines()
    unsigned = {key: value for key, value in manifest.items() if key != "self_sha256"}
    expected_fields = {
        "format",
        "method_name",
        "method_version",
        "method_family",
        "method_contract",
        "claim_scope",
        "population",
        "fixed_denominator",
        "authority_sha256",
        "train_query_projection_sha256",
        "snapshot",
        "experience_graph",
        "train_workflow_evidence_sha256",
        "embedding_protocol",
        "llm_protocol",
        "tasks",
        "experience_file",
        "experience_sha256",
        "row_count",
        "status_counts",
        "self_sha256",
    }
    if (
        type(manifest) is not dict
        or set(manifest) != expected_fields
        or manifest_bytes != canonical_json_bytes(manifest)
        or manifest.get("format") != format_id
        or manifest.get("method_name") != retrieval.METHOD_NAME
        or manifest.get("method_version") != METHOD_VERSION
        or manifest.get("method_family") != retrieval.METHOD_FAMILY
        or manifest.get("method_contract") != METHOD_CONTRACT
        or manifest.get("claim_scope") != claim_scope
        or manifest.get("population") != dict(population_identity)
        or manifest.get("fixed_denominator") != len(tasks)
        or manifest.get("authority_sha256") != authority.authority_sha256
        or manifest.get("train_query_projection_sha256")
        != EXPECTED_QUERY_PROJECTION_SHA256
        or manifest.get("snapshot") != dict(context.identity)
        or manifest.get("experience_graph") != context.graph.identity()
        or manifest.get("train_workflow_evidence_sha256")
        != retrieval._evidence_cards_sha256(train_cards)
        or not retrieval._embedding_protocol_matches(
            manifest.get("embedding_protocol")
        )
        or manifest.get("experience_file") != "experience.jsonl"
        or manifest.get("experience_sha256") != _sha(experience_bytes)
        or manifest.get("row_count") != len(tasks)
        or len(lines) != len(tasks)
        or manifest.get("self_sha256") != _sha(canonical_json_bytes(unsigned))
    ):
        raise ValueError("retrieval bundle identity differs")
    llm_protocol = manifest.get("llm_protocol")
    if type(llm_protocol) is not dict:
        raise ValueError("retrieval LLM protocol differs")
    expected_protocol = _protocols(
        _ProtocolOnly(llm_protocol.get("need_producer")),
        _ProtocolOnly(llm_protocol.get("clarification_producer")),
        _ProtocolOnly(llm_protocol.get("selector_producer")),
    )
    if llm_protocol != expected_protocol:
        raise ValueError("retrieval LLM protocol differs")
    task_audits = manifest.get("tasks")
    train_card_sha = retrieval._evidence_cards_sha256(train_cards)
    if type(task_audits) is not list or len(task_audits) != len(tasks):
        raise ValueError("retrieval task audit differs")
    statuses: Counter[str] = Counter()
    for index, (line, task, task_audit) in enumerate(
        zip(lines, tasks, task_audits, strict=True)
    ):
        row = json.loads(line)
        metadata = row.get("metadata") if type(row) is dict else None
        audit = metadata.get("retrieval_audit") if type(metadata) is dict else None
        if (
            type(row) is not dict
            or set(row) != {"instance_id", "experience", "metadata"}
            or line != canonical_json_bytes(row)
            or row.get("instance_id") != task.task_id
            or type(row.get("experience")) is not str
            or not row["experience"]
            or type(metadata) is not dict
            or set(metadata) != retrieval._EXPERIENCE_METADATA_FIELDS
            or metadata.get("format") != retrieval.EXPERIENCE_FORMAT
            or metadata.get("method_family") != retrieval.METHOD_FAMILY
            or metadata.get("query_index") != task.query_index
            or metadata.get("dataset_index") != task.dataset_index
            or metadata.get("authority_sha256") != authority.authority_sha256
            or metadata.get("experience_sha256")
            != _sha(row["experience"].encode("utf-8"))
            or type(audit) is not dict
            or metadata.get("retrieval_audit_sha256")
            != _sha(canonical_json_bytes(audit))
            or metadata.get("status") not in retrieval._RESULT_STATUSES
            or type(task_audit) is not dict
            or task_audit.get("query_index") != task.query_index
            or task_audit.get("dataset_index") != task.dataset_index
            or task_audit.get("task_id") != task.task_id
            or task_audit.get("status") != metadata.get("status")
            or task_audit.get("row_sha256") != _sha(canonical_json_bytes(row))
            or task_audit.get("retrieval_audit_sha256")
            != metadata.get("retrieval_audit_sha256")
            or not retrieval._valid_clean_retrieval_audit(
                audit,
                card=target_cards[task.query_index],
                train_evidence_sha256=train_card_sha,
                instruction=task.instruction,
            )
        ):
            raise ValueError(f"retrieval row {index} differs")
        statuses[str(metadata["status"])] += 1
    if dict(sorted(statuses.items())) != manifest.get("status_counts"):
        raise ValueError("retrieval status summary differs")
    return retrieval._VerifiedExperienceBundle(retrieval._VERIFIED_TOKEN, root, manifest)


class _ProtocolOnly:
    def __init__(self, protocol: Any) -> None:
        self.protocol_identity = protocol
