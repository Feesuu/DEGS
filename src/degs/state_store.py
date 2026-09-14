from __future__ import annotations

from collections.abc import Iterator, MutableMapping
from contextlib import contextmanager
import hashlib
from pathlib import Path
import sqlite3
import struct
import json
from typing import Any, Callable, Mapping, Sequence

from .core import (
    EMBEDDING_MODEL,
    EmbeddedText,
    canonical_json_bytes,
    embedding_text_sha256,
)
from .graph_dataset_contract import (
    GraphDatasetContract,
    SPREADSHEETBENCH_GRAPH_CONTRACT,
)


STATE_SCHEMA_VERSION = 9
INCREMENTAL_METHOD_ID = (
    "DEGS 0.77.41 Stable R1 Incremental ExperienceGraph"
)
EMBEDDING_CACHE_NAMESPACE = hashlib.sha256(
    canonical_json_bytes(
        {
            "format": "degs_embedding_space_v2",
            "model": EMBEDDING_MODEL,
            "normalization": "unicode_nfc_and_lf",
            "vector_encoding": "little_endian_float64",
        }
    )
).hexdigest()
_SCHEMA = """
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
) STRICT;

CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_id TEXT PRIMARY KEY,
    parent_snapshot_id TEXT REFERENCES snapshots(snapshot_id),
    operation_kind TEXT NOT NULL DEFAULT 'TRAIN_BATCH' CHECK(operation_kind = 'TRAIN_BATCH'),
    operation_input_sha256 TEXT NOT NULL DEFAULT '',
    batch_source_sha256 TEXT NOT NULL,
    batch_source_audit_sha256 TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('BUILDING', 'COMMITTED')),
    manifest_sha256 TEXT,
    source_workflow_count INTEGER CHECK(source_workflow_count >= 0),
    canonical_count INTEGER CHECK(canonical_count >= 0),
    experience_node_count INTEGER CHECK(experience_node_count >= 0),
    experience_edge_count INTEGER CHECK(experience_edge_count >= 0),
    CHECK(
        status = 'BUILDING'
        OR (
            manifest_sha256 IS NOT NULL
            AND source_workflow_count IS NOT NULL
            AND canonical_count IS NOT NULL
            AND experience_node_count IS NOT NULL
            AND experience_edge_count IS NOT NULL
        )
    )
) STRICT;

CREATE TABLE IF NOT EXISTS snapshot_batch_items (
    snapshot_id TEXT NOT NULL REFERENCES snapshots(snapshot_id),
    train_index INTEGER NOT NULL CHECK(train_index >= 0 AND train_index < 200),
    status TEXT NOT NULL CHECK(status IN (
        'INGESTED',
        'SOURCE_EXCLUDED_NO_VALIDATED_SUCCESS',
        'SOURCE_EXCLUDED_CONTEXT_LENGTH',
        'SOURCE_EXCLUDED_GENERATION_FAILURE',
        'SOURCE_EXCLUDED_NO_REUSABLE_EXPERIENCE',
        'SOURCE_EXCLUDED_NO_STEP',
        'SOURCE_EXCLUDED_EMPTY_PUBLIC_QUESTION',
        'SOURCE_EXCLUDED_REVIEW_FAILURE'
    )),
    PRIMARY KEY(snapshot_id, train_index)
) STRICT;

CREATE TABLE IF NOT EXISTS embedding_cache (
    namespace_sha256 TEXT NOT NULL,
    normalized_text_sha256 TEXT NOT NULL,
    normalized_text TEXT NOT NULL,
    dimension INTEGER NOT NULL CHECK(dimension > 0),
    vector_blob BLOB NOT NULL,
    vector_sha256 TEXT NOT NULL,
    producer_request_sha256 TEXT NOT NULL,
    created_snapshot_id TEXT,
    PRIMARY KEY(namespace_sha256, normalized_text_sha256)
) STRICT;

CREATE TABLE IF NOT EXISTS workflows (
    train_index INTEGER PRIMARY KEY CHECK(train_index >= 0 AND train_index < 200),
    task_id TEXT NOT NULL UNIQUE,
    query_text TEXT NOT NULL,
    query_text_sha256 TEXT NOT NULL,
    workflow_sha256 TEXT NOT NULL,
    workflow_json BLOB NOT NULL,
    added_snapshot_id TEXT NOT NULL REFERENCES snapshots(snapshot_id)
) STRICT;

CREATE TABLE IF NOT EXISTS experience_nodes (
    node_id TEXT PRIMARY KEY,
    train_index INTEGER NOT NULL REFERENCES workflows(train_index),
    node_index INTEGER NOT NULL CHECK(node_index >= 0),
    node_sha256 TEXT NOT NULL,
    node_json BLOB NOT NULL,
    added_snapshot_id TEXT NOT NULL REFERENCES snapshots(snapshot_id),
    UNIQUE(train_index, node_index)
) STRICT;

CREATE TABLE IF NOT EXISTS source_edges (
    train_index INTEGER NOT NULL REFERENCES workflows(train_index),
    source_node_id TEXT NOT NULL REFERENCES experience_nodes(node_id),
    target_node_id TEXT NOT NULL REFERENCES experience_nodes(node_id),
    PRIMARY KEY(train_index, source_node_id, target_node_id)
) STRICT;

CREATE TABLE IF NOT EXISTS canonical_nodes (
    canonical_id TEXT PRIMARY KEY,
    canonical_sha256 TEXT NOT NULL,
    canonical_json BLOB NOT NULL,
    document_sha256 TEXT NOT NULL,
    document TEXT NOT NULL,
    member_count INTEGER NOT NULL CHECK(member_count > 0),
    created_snapshot_id TEXT NOT NULL REFERENCES snapshots(snapshot_id),
    retired_snapshot_id TEXT REFERENCES snapshots(snapshot_id)
) STRICT;

CREATE TABLE IF NOT EXISTS canonical_leaf_members (
    canonical_id TEXT NOT NULL REFERENCES canonical_nodes(canonical_id),
    node_id TEXT NOT NULL REFERENCES experience_nodes(node_id),
    PRIMARY KEY(canonical_id, node_id)
) STRICT;

CREATE TABLE IF NOT EXISTS canonical_heads (
    node_id TEXT PRIMARY KEY REFERENCES experience_nodes(node_id),
    canonical_id TEXT NOT NULL REFERENCES canonical_nodes(canonical_id)
) STRICT;

CREATE TABLE IF NOT EXISTS canonical_jobs (
    request_sha256 TEXT PRIMARY KEY,
    stage TEXT NOT NULL CHECK(stage IN ('VIEW', 'MERGE')),
    prompt_sha256 TEXT NOT NULL,
    producer_protocol_sha256 TEXT NOT NULL,
    validated_response_json BLOB NOT NULL,
    audit_json BLOB NOT NULL,
    created_snapshot_id TEXT NOT NULL REFERENCES snapshots(snapshot_id)
) STRICT;

CREATE TABLE IF NOT EXISTS canonical_attempts (
    attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_sha256 TEXT NOT NULL,
    stage TEXT NOT NULL CHECK(stage IN ('VIEW', 'MERGE')),
    semantic_attempt_index INTEGER NOT NULL CHECK(semantic_attempt_index > 0),
    response_sha256 TEXT,
    response_json BLOB,
    status TEXT NOT NULL CHECK(status IN ('ACCEPTED', 'REJECTED')),
    error_type TEXT,
    error_message TEXT,
    created_snapshot_id TEXT NOT NULL REFERENCES snapshots(snapshot_id),
    CHECK(
        (response_sha256 IS NULL AND response_json IS NULL)
        OR (response_sha256 IS NOT NULL AND response_json IS NOT NULL)
    ),
    CHECK(
        (status = 'ACCEPTED' AND error_type IS NULL AND error_message IS NULL)
        OR (status = 'REJECTED' AND error_type IS NOT NULL AND error_message IS NOT NULL)
    )
) STRICT;

CREATE TABLE IF NOT EXISTS canonical_transport_waves (
    wave_sha256 TEXT PRIMARY KEY,
    stage TEXT NOT NULL CHECK(stage IN ('VIEW', 'MERGE')),
    status TEXT NOT NULL CHECK(status IN ('ITEM_LOCAL', 'SYSTEMIC', 'ABORTED_BY_SYSTEMIC')),
    policy_sha256 TEXT NOT NULL,
    failed_request_ids_json BLOB NOT NULL,
    attempts_json BLOB NOT NULL,
    created_snapshot_id TEXT NOT NULL REFERENCES snapshots(snapshot_id)
) STRICT;

CREATE TABLE IF NOT EXISTS source_edge_projection (
    train_index INTEGER NOT NULL REFERENCES workflows(train_index),
    source_canonical_id TEXT NOT NULL REFERENCES canonical_nodes(canonical_id),
    target_canonical_id TEXT NOT NULL REFERENCES canonical_nodes(canonical_id),
    PRIMARY KEY(train_index, source_canonical_id, target_canonical_id)
) STRICT;

CREATE TABLE IF NOT EXISTS canonical_views (
    node_id TEXT PRIMARY KEY REFERENCES experience_nodes(node_id),
    source_node_sha256 TEXT NOT NULL,
    request_sha256 TEXT REFERENCES canonical_jobs(request_sha256),
    view_sha256 TEXT NOT NULL,
    view_json BLOB NOT NULL,
    normalized_text_sha256 TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('VIEW_ACCEPTED', 'VIEW_SOURCE_FALLBACK')),
    updated_snapshot_id TEXT NOT NULL REFERENCES snapshots(snapshot_id)
) STRICT;

CREATE TABLE IF NOT EXISTS canonical_neighbors (
    snapshot_id TEXT NOT NULL REFERENCES snapshots(snapshot_id),
    source_canonical_id TEXT NOT NULL REFERENCES canonical_nodes(canonical_id),
    target_canonical_id TEXT NOT NULL REFERENCES canonical_nodes(canonical_id),
    similarity REAL NOT NULL CHECK(similarity >= -1.0000001 AND similarity <= 1.0000001),
    rank INTEGER NOT NULL CHECK(rank > 0),
    exact_text_match INTEGER NOT NULL CHECK(exact_text_match IN (0, 1)),
    PRIMARY KEY(snapshot_id, source_canonical_id, rank),
    UNIQUE(snapshot_id, source_canonical_id, target_canonical_id),
    CHECK(source_canonical_id != target_canonical_id)
) STRICT;

CREATE TABLE IF NOT EXISTS canonical_merge_events (
    event_sha256 TEXT PRIMARY KEY,
    snapshot_id TEXT NOT NULL REFERENCES snapshots(snapshot_id),
    left_canonical_id TEXT NOT NULL REFERENCES canonical_nodes(canonical_id),
    right_canonical_id TEXT NOT NULL REFERENCES canonical_nodes(canonical_id),
    child_canonical_id TEXT NOT NULL REFERENCES canonical_nodes(canonical_id),
    request_sha256 TEXT NOT NULL REFERENCES canonical_jobs(request_sha256),
    apply_order INTEGER NOT NULL CHECK(apply_order >= 0),
    UNIQUE(snapshot_id, apply_order),
    UNIQUE(child_canonical_id),
    CHECK(left_canonical_id != right_canonical_id)
) STRICT;

CREATE TABLE IF NOT EXISTS canonical_resolution_events (
    event_sha256 TEXT PRIMARY KEY,
    stage TEXT NOT NULL CHECK(stage IN ('VIEW', 'MERGE')),
    status TEXT NOT NULL CHECK(status IN (
        'VIEW_SOURCE_FALLBACK',
        'MERGE_EXHAUSTED_NO_MERGE'
    )),
    subject_sha256 TEXT NOT NULL,
    subject_json BLOB NOT NULL,
    evidence_json BLOB NOT NULL,
    created_snapshot_id TEXT NOT NULL REFERENCES snapshots(snapshot_id)
) STRICT;

CREATE TABLE IF NOT EXISTS graph_quality_audits (
    snapshot_id TEXT PRIMARY KEY REFERENCES snapshots(snapshot_id),
    status TEXT NOT NULL CHECK(status IN ('READY', 'NOT_READY')),
    audit_protocol_sha256 TEXT NOT NULL,
    summary_sha256 TEXT NOT NULL,
    source_node_audit_sha256 TEXT NOT NULL,
    merge_ledger_sha256 TEXT NOT NULL,
    unresolved_candidates_sha256 TEXT NOT NULL,
    topology_sha256 TEXT NOT NULL,
    hard_violations_json BLOB NOT NULL
) STRICT;

"""


def _is_sha256(value: str) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


class SQLiteEmbeddingCache(MutableMapping[str, EmbeddedText]):
    def __init__(
        self,
        connection: sqlite3.Connection,
        *,
        namespace_sha256: str = EMBEDDING_CACHE_NAMESPACE,
        created_snapshot_id: str | None = None,
        harden_files: Callable[[], None] | None = None,
    ) -> None:
        if not _is_sha256(namespace_sha256):
            raise ValueError("embedding cache namespace differs")
        self._connection = connection
        self.namespace_sha256 = namespace_sha256
        self.created_snapshot_id = created_snapshot_id
        self._harden_files = harden_files or (lambda: None)

    def __getitem__(self, key: str) -> EmbeddedText:
        if not _is_sha256(key):
            raise KeyError(key)
        row = self._connection.execute(
            """
            SELECT normalized_text, dimension, vector_blob, vector_sha256,
                   producer_request_sha256
            FROM embedding_cache
            WHERE namespace_sha256 = ? AND normalized_text_sha256 = ?
            """,
            (self.namespace_sha256, key),
        ).fetchone()
        if row is None:
            raise KeyError(key)
        text, dimension, vector_blob, vector_sha256, request_sha256 = row
        if embedding_text_sha256(text) != key:
            raise ValueError("stored embedding text identity differs")
        if len(vector_blob) != dimension * 8:
            raise ValueError("stored embedding vector size differs")
        vector = struct.unpack(f"<{dimension}d", vector_blob)
        if hashlib.sha256(vector_blob).hexdigest() != vector_sha256:
            raise ValueError("stored embedding vector hash differs")
        return EmbeddedText(text, tuple(vector), vector_sha256, request_sha256)

    def __setitem__(self, key: str, value: EmbeddedText) -> None:
        if type(value) is not EmbeddedText or key != embedding_text_sha256(
            value.normalized_text
        ):
            raise ValueError("embedding cache key differs")
        if not _is_sha256(value.vector_sha256) or not _is_sha256(
            value.request_sha256
        ):
            raise ValueError("embedding cache hash differs")
        vector_blob = b"".join(struct.pack("<d", item) for item in value.vector)
        if hashlib.sha256(vector_blob).hexdigest() != value.vector_sha256:
            raise ValueError("embedding cache vector identity differs")
        try:
            self._connection.execute(
                """
                INSERT INTO embedding_cache(
                    namespace_sha256, normalized_text_sha256, normalized_text,
                    dimension, vector_blob, vector_sha256,
                    producer_request_sha256, created_snapshot_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.namespace_sha256,
                    key,
                    value.normalized_text,
                    len(value.vector),
                    vector_blob,
                    value.vector_sha256,
                    value.request_sha256,
                    self.created_snapshot_id,
                ),
            )
        except sqlite3.IntegrityError:
            existing = self[key]
            if existing != value:
                raise ValueError("embedding cache contains conflicting vectors")
        finally:
            self._harden_files()

    def __delitem__(self, key: str) -> None:
        cursor = self._connection.execute(
            "DELETE FROM embedding_cache WHERE namespace_sha256 = ? AND normalized_text_sha256 = ?",
            (self.namespace_sha256, key),
        )
        if cursor.rowcount != 1:
            raise KeyError(key)
        self._harden_files()

    def __iter__(self) -> Iterator[str]:
        rows = self._connection.execute(
            "SELECT normalized_text_sha256 FROM embedding_cache WHERE namespace_sha256 = ? ORDER BY normalized_text_sha256",
            (self.namespace_sha256,),
        )
        return (row[0] for row in rows.fetchall())

    def __len__(self) -> int:
        row = self._connection.execute(
            "SELECT COUNT(*) FROM embedding_cache WHERE namespace_sha256 = ?",
            (self.namespace_sha256,),
        ).fetchone()
        return int(row[0])


def _schema_for_contract(contract: GraphDatasetContract) -> str:
    return _SCHEMA.replace("train_index < 200", f"train_index < {contract.train_count}")


class IncrementalStateStore:
    def __init__(
        self,
        path: Path | str,
        *,
        dataset_contract: GraphDatasetContract = SPREADSHEETBENCH_GRAPH_CONTRACT,
        readonly: bool = False,
    ) -> None:
        self.path = Path(path).expanduser().absolute()
        self.dataset_contract = dataset_contract
        self.readonly = readonly
        if self.path.exists() and not self.path.is_file():
            raise ValueError("incremental state database path differs")
        if readonly and not self.path.is_file():
            raise ValueError("read-only incremental state database is missing")
        if not readonly:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        target = f"file:{self.path}?mode=ro" if readonly else str(self.path)
        self._connection = sqlite3.connect(
            target,
            isolation_level=None,
            uri=readonly,
        )
        try:
            existing_tables = {
                str(row[0])
                for row in self._connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            }
            if existing_tables:
                self._validate_existing_identity(existing_tables)
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.execute("PRAGMA busy_timeout = 30000")
            if readonly:
                self._connection.execute("PRAGMA query_only = ON")
            else:
                self._connection.execute("PRAGMA journal_mode = WAL")
                self._connection.executescript(_schema_for_contract(dataset_contract))
                self._initialize_identity()
        except BaseException:
            self._connection.close()
            raise

    def _validate_existing_identity(self, existing_tables: set[str]) -> None:
        if "metadata" not in existing_tables:
            raise ValueError("existing incremental state has no method identity")
        try:
            metadata = {
                str(key): str(value)
                for key, value in self._connection.execute(
                    "SELECT key, value FROM metadata"
                ).fetchall()
            }
        except sqlite3.DatabaseError as exc:
            raise ValueError("existing incremental state identity is unreadable") from exc
        expected = self._expected_identity()
        stored_contract = metadata.get("graph_dataset_contract")
        if stored_contract is not None and stored_contract != expected["graph_dataset_contract"]:
            raise ValueError("incremental state graph_dataset_contract differs")
        if (
            stored_contract is None
            and self.dataset_contract != SPREADSHEETBENCH_GRAPH_CONTRACT
        ):
            raise ValueError("incremental state graph_dataset_contract differs")
        for key, value in expected.items():
            if (
                key == "graph_dataset_contract"
                and stored_contract is None
                and self.dataset_contract == SPREADSHEETBENCH_GRAPH_CONTRACT
            ):
                # Existing 0.77.41 SpreadsheetBench databases predate this
                # metadata key. Only that exact default contract is legacy-compatible.
                continue
            if metadata.get(key) != value:
                raise ValueError(f"incremental state {key} differs")

    def _expected_identity(self) -> dict[str, str]:
        expected = {
            "schema_version": str(STATE_SCHEMA_VERSION),
            "method_id": INCREMENTAL_METHOD_ID,
            "embedding_cache_namespace": EMBEDDING_CACHE_NAMESPACE,
        }
        expected["graph_dataset_contract"] = json.dumps(
            self.dataset_contract.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        )
        return expected

    def _initialize_identity(self) -> None:
        expected = self._expected_identity()
        with self.transaction():
            for key, value in expected.items():
                row = self._connection.execute(
                    "SELECT value FROM metadata WHERE key = ?", (key,)
                ).fetchone()
                if row is None:
                    self._connection.execute(
                        "INSERT INTO metadata(key, value) VALUES (?, ?)", (key, value)
                    )
                elif row[0] != value:
                    raise ValueError(f"incremental state {key} differs")

    @property
    def connection(self) -> sqlite3.Connection:
        return self._connection

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            yield self._connection
        except BaseException:
            self._connection.execute("ROLLBACK")
            raise
        else:
            self._connection.execute("COMMIT")

    def embedding_cache(
        self, *, created_snapshot_id: str | None = None
    ) -> SQLiteEmbeddingCache:
        return SQLiteEmbeddingCache(
            self._connection,
            created_snapshot_id=created_snapshot_id,
        )

    @property
    def embedding_endpoint(self) -> str | None:
        row = self._connection.execute(
            "SELECT value FROM metadata WHERE key = 'embedding_endpoint'"
        ).fetchone()
        return None if row is None else str(row[0])

    def bind_embedding_endpoint(self, endpoint: str) -> None:
        if type(endpoint) is not str or not endpoint:
            raise ValueError("embedding endpoint identity differs")
        with self.transaction():
            row = self._connection.execute(
                "SELECT value FROM metadata WHERE key = 'embedding_endpoint'"
            ).fetchone()
            if row is None:
                self._connection.execute(
                    "INSERT INTO metadata(key, value) VALUES ('embedding_endpoint', ?)",
                    (endpoint,),
                )
            elif row[0] != endpoint:
                raise ValueError("embedding endpoint changes an existing vector space")

    @property
    def generation_endpoint(self) -> str | None:
        row = self._connection.execute(
            "SELECT value FROM metadata WHERE key = 'generation_endpoint'"
        ).fetchone()
        return None if row is None else str(row[0])

    def bind_generation_endpoint(self, endpoint: str) -> None:
        if type(endpoint) is not str or not endpoint:
            raise ValueError("generation endpoint identity differs")
        normalized = endpoint.rstrip("/")
        with self.transaction():
            row = self._connection.execute(
                "SELECT value FROM metadata WHERE key = 'generation_endpoint'"
            ).fetchone()
            if row is None:
                self._connection.execute(
                    "INSERT INTO metadata(key, value) VALUES ('generation_endpoint', ?)",
                    (normalized,),
                )
            elif row[0] != normalized:
                raise ValueError("generation endpoint changes an existing producer identity")

    def get_canonical_job(
        self,
        request_sha256: str,
        *,
        stage: str,
        prompt_sha256: str,
        producer_protocol_sha256: str,
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        if not _is_sha256(request_sha256):
            raise ValueError("Canonical request identity differs")
        row = self._connection.execute(
            """
            SELECT stage, prompt_sha256, producer_protocol_sha256,
                   validated_response_json, audit_json
            FROM canonical_jobs WHERE request_sha256 = ?
            """,
            (request_sha256,),
        ).fetchone()
        if row is None:
            return None
        if tuple(row[:3]) != (stage, prompt_sha256, producer_protocol_sha256):
            raise ValueError("Canonical cache protocol identity differs")
        response = json.loads(bytes(row[3]))
        audit = json.loads(bytes(row[4]))
        if type(response) is not dict or type(audit) is not dict:
            raise ValueError("Canonical cache payload differs")
        return response, audit

    def put_canonical_job(
        self,
        request_sha256: str,
        *,
        stage: str,
        semantic_attempt_index: int,
        prompt_sha256: str,
        producer_protocol_sha256: str,
        response: Mapping[str, Any],
        audit: Mapping[str, Any],
        created_snapshot_id: str,
    ) -> None:
        if not _is_sha256(request_sha256):
            raise ValueError("Canonical request identity differs")
        if type(semantic_attempt_index) is not int or semantic_attempt_index <= 0:
            raise ValueError("Canonical semantic attempt index differs")
        response_bytes = canonical_json_bytes(dict(response))
        audit_bytes = canonical_json_bytes(dict(audit))
        response_sha256 = hashlib.sha256(response_bytes).hexdigest()
        with self.transaction():
            self._connection.execute(
                """
                INSERT INTO canonical_attempts(
                    request_sha256, stage, semantic_attempt_index,
                    response_sha256, response_json, status,
                    error_type, error_message, created_snapshot_id
                ) VALUES (?, ?, ?, ?, ?, 'ACCEPTED', NULL, NULL, ?)
                """,
                (
                    request_sha256,
                    stage,
                    semantic_attempt_index,
                    response_sha256,
                    response_bytes,
                    created_snapshot_id,
                ),
            )
            try:
                self._connection.execute(
                    """
                    INSERT INTO canonical_jobs(
                        request_sha256, stage, prompt_sha256,
                        producer_protocol_sha256, validated_response_json,
                        audit_json, created_snapshot_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        request_sha256,
                        stage,
                        prompt_sha256,
                        producer_protocol_sha256,
                        response_bytes,
                        audit_bytes,
                        created_snapshot_id,
                    ),
                )
            except sqlite3.IntegrityError:
                existing = self.get_canonical_job(
                    request_sha256,
                    stage=stage,
                    prompt_sha256=prompt_sha256,
                    producer_protocol_sha256=producer_protocol_sha256,
                )
                if existing != (dict(response), dict(audit)):
                    raise ValueError("Canonical cache contains conflicting output")

    def record_canonical_attempt(
        self,
        *,
        request_sha256: str,
        stage: str,
        semantic_attempt_index: int,
        response: Mapping[str, Any] | None,
        validation_error: Exception | None,
        created_snapshot_id: str,
    ) -> None:
        if not _is_sha256(request_sha256):
            raise ValueError("Canonical request identity differs")
        if stage not in {"VIEW", "MERGE"}:
            raise ValueError("Canonical attempt stage differs")
        if type(semantic_attempt_index) is not int or semantic_attempt_index <= 0:
            raise ValueError("Canonical semantic attempt index differs")
        if type(created_snapshot_id) is not str or not created_snapshot_id:
            raise ValueError("Canonical attempt snapshot identity differs")
        response_bytes = (
            None if response is None else canonical_json_bytes(dict(response))
        )
        response_sha256 = (
            None
            if response_bytes is None
            else hashlib.sha256(response_bytes).hexdigest()
        )
        status = "ACCEPTED" if validation_error is None else "REJECTED"
        error_type = (
            None if validation_error is None else type(validation_error).__name__
        )
        error_message = None if validation_error is None else str(validation_error)
        with self.transaction():
            self._connection.execute(
                """
                INSERT INTO canonical_attempts(
                    request_sha256, stage, semantic_attempt_index,
                    response_sha256, response_json, status,
                    error_type, error_message, created_snapshot_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request_sha256,
                    stage,
                    semantic_attempt_index,
                    response_sha256,
                    response_bytes,
                    status,
                    error_type,
                    error_message,
                    created_snapshot_id,
                ),
            )

    def canonical_attempt_failures(
        self,
        *,
        request_sha256: str,
        stage: str,
        created_snapshot_id: str,
    ) -> tuple[tuple[int, str, str], ...]:
        """Return already-recorded semantic failures for a resumable request."""
        if not _is_sha256(request_sha256):
            raise ValueError("Canonical request identity differs")
        if stage not in {"VIEW", "MERGE"}:
            raise ValueError("Canonical attempt stage differs")
        if type(created_snapshot_id) is not str or not created_snapshot_id:
            raise ValueError("Canonical attempt snapshot identity differs")
        rows = self._connection.execute(
            """
            SELECT semantic_attempt_index, status, error_type, error_message
            FROM canonical_attempts
            WHERE request_sha256 = ? AND stage = ? AND created_snapshot_id = ?
            ORDER BY attempt_id
            """,
            (request_sha256, stage, created_snapshot_id),
        ).fetchall()
        if not rows:
            return ()
        indices = [row[0] for row in rows]
        if indices != list(range(1, len(rows) + 1)):
            raise ValueError("Canonical attempt history is not contiguous")
        if any(
            row[1] != "REJECTED"
            or type(row[2]) is not str
            or type(row[3]) is not str
            for row in rows
        ):
            raise ValueError("Canonical failed-attempt history differs")
        return tuple((row[0], row[2], row[3]) for row in rows)

    def put_canonical_transport_wave(
        self,
        *,
        stage: str,
        status: str,
        policy: Mapping[str, Any],
        failed_request_ids: Sequence[str],
        attempts: Sequence[Mapping[str, Any]],
        created_snapshot_id: str,
    ) -> str:
        if stage not in {"VIEW", "MERGE"}:
            raise ValueError("Canonical transport stage differs")
        if status not in {
            "ITEM_LOCAL",
            "SYSTEMIC",
            "ABORTED_BY_SYSTEMIC",
        }:
            raise ValueError("Canonical transport wave status differs")
        request_ids = sorted(set(failed_request_ids))
        attempt_rows = [dict(row) for row in attempts]
        if (
            type(policy) is not dict
            or not request_ids
            or any(type(row) is not str or not row for row in request_ids)
            or not attempt_rows
            or any(
                set(row)
                != {
                    "request_sha256",
                    "request_id",
                    "attempt_index",
                    "error_type",
                    "error_message",
                }
                or not _is_sha256(row["request_sha256"])
                or row["request_id"] not in request_ids
                or type(row["attempt_index"]) is not int
                or row["attempt_index"] <= 0
                or type(row["error_type"]) is not str
                or not row["error_type"]
                or type(row["error_message"]) is not str
                for row in attempt_rows
            )
            or type(created_snapshot_id) is not str
            or not created_snapshot_id
        ):
            raise ValueError("Canonical transport wave differs")
        policy_bytes = canonical_json_bytes(dict(policy))
        request_bytes = canonical_json_bytes(request_ids)
        attempts_bytes = canonical_json_bytes(attempt_rows)
        identity = {
            "stage": stage,
            "status": status,
            "policy_sha256": hashlib.sha256(policy_bytes).hexdigest(),
            "failed_request_ids_sha256": hashlib.sha256(
                request_bytes
            ).hexdigest(),
            "attempts_sha256": hashlib.sha256(attempts_bytes).hexdigest(),
            "created_snapshot_id": created_snapshot_id,
        }
        wave_sha256 = hashlib.sha256(
            canonical_json_bytes(identity)
        ).hexdigest()
        values = (
            wave_sha256,
            stage,
            status,
            identity["policy_sha256"],
            request_bytes,
            attempts_bytes,
            created_snapshot_id,
        )
        try:
            self._connection.execute(
                """
                INSERT INTO canonical_transport_waves(
                    wave_sha256, stage, status, policy_sha256,
                    failed_request_ids_json, attempts_json,
                    created_snapshot_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                values,
            )
        except sqlite3.IntegrityError:
            existing = self._connection.execute(
                """
                SELECT wave_sha256, stage, status, policy_sha256,
                       failed_request_ids_json, attempts_json,
                       created_snapshot_id
                FROM canonical_transport_waves WHERE wave_sha256 = ?
                """,
                (wave_sha256,),
            ).fetchone()
            if existing != values:
                raise ValueError("Canonical transport wave conflicts")
        return wave_sha256

    def put_canonical_view(
        self,
        *,
        node_id: str,
        source_node_sha256: str,
        request_sha256: str | None,
        view: Mapping[str, Any],
        normalized_text_sha256: str,
        status: str,
        updated_snapshot_id: str,
    ) -> None:
        view_bytes = canonical_json_bytes(dict(view))
        values = (
            node_id,
            source_node_sha256,
            request_sha256,
            hashlib.sha256(view_bytes).hexdigest(),
            view_bytes,
            normalized_text_sha256,
            status,
            updated_snapshot_id,
        )
        if (
            not node_id
            or not updated_snapshot_id
            or not _is_sha256(source_node_sha256)
            or (
                request_sha256 is not None
                and not _is_sha256(request_sha256)
            )
            or not _is_sha256(values[3])
            or not _is_sha256(normalized_text_sha256)
            or status not in {"VIEW_ACCEPTED", "VIEW_SOURCE_FALLBACK"}
            or (status == "VIEW_ACCEPTED") != (request_sha256 is not None)
        ):
            raise ValueError("Canonical View state identity differs")
        self._connection.execute(
            """
            INSERT INTO canonical_views(
                node_id, source_node_sha256, request_sha256, view_sha256,
                view_json, normalized_text_sha256, status, updated_snapshot_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(node_id) DO UPDATE SET
                source_node_sha256 = excluded.source_node_sha256,
                request_sha256 = excluded.request_sha256,
                view_sha256 = excluded.view_sha256,
                view_json = excluded.view_json,
                normalized_text_sha256 = excluded.normalized_text_sha256,
                status = excluded.status,
                updated_snapshot_id = excluded.updated_snapshot_id
            """,
            values,
        )

    def replace_snapshot_neighbors(
        self,
        *,
        snapshot_id: str,
        rows: Sequence[Mapping[str, Any]],
    ) -> None:
        if type(snapshot_id) is not str or not snapshot_id:
            raise ValueError("neighbor snapshot identity differs")
        normalized: list[tuple[Any, ...]] = []
        seen_rank: set[tuple[str, int]] = set()
        seen_target: set[tuple[str, str]] = set()
        for row in rows:
            source = row.get("source_canonical_id")
            target = row.get("target_canonical_id")
            similarity = row.get("similarity")
            rank = row.get("rank")
            exact = row.get("exact_text_match", False)
            if (
                type(source) is not str
                or not source
                or type(target) is not str
                or not target
                or source == target
                or type(similarity) not in {int, float}
                or not -1.0000001 <= float(similarity) <= 1.0000001
                or type(rank) is not int
                or rank <= 0
                or type(exact) is not bool
                or (source, rank) in seen_rank
                or (source, target) in seen_target
            ):
                raise ValueError("Canonical neighbor row differs")
            seen_rank.add((source, rank))
            seen_target.add((source, target))
            normalized.append(
                (snapshot_id, source, target, float(similarity), rank, int(exact))
            )
        self._connection.execute(
            "DELETE FROM canonical_neighbors WHERE snapshot_id = ?",
            (snapshot_id,),
        )
        self._connection.executemany(
            """
            INSERT INTO canonical_neighbors(
                snapshot_id, source_canonical_id, target_canonical_id,
                similarity, rank, exact_text_match
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            normalized,
        )

    def put_canonical_merge_event(self, *, snapshot_id: str, event: Mapping[str, Any]) -> None:
        keys = ("left_canonical_id", "right_canonical_id", "child_canonical_id", "request_sha256", "apply_order")
        if set(event) != set(keys) or type(event["apply_order"]) is not int or event["apply_order"] < 0:
            raise ValueError("Canonical merge event fields differ")
        if event["left_canonical_id"] == event["right_canonical_id"] or not _is_sha256(event["request_sha256"]):
            raise ValueError("Canonical merge event identity differs")
        job = self.connection.execute(
            "SELECT stage, validated_response_json FROM canonical_jobs WHERE request_sha256 = ?",
            (event["request_sha256"],),
        ).fetchone()
        if job is None or job[0] != "MERGE":
            raise ValueError("Canonical merge event requires a SAME decision")
        from .canonicalize import parse_canonical_merge
        from .section_graph import _canonical_id

        decision = parse_canonical_merge(json.loads(job[1]))
        if decision.relation.value != "SAME_TEMPLATE":
            raise ValueError("Canonical merge event requires a SAME decision")

        parents = []
        for key in ("left_canonical_id", "right_canonical_id", "child_canonical_id"):
            members = self.connection.execute(
                """SELECT n.train_index, n.node_index FROM canonical_leaf_members m
                   JOIN experience_nodes n USING(node_id) WHERE m.canonical_id = ?
                   ORDER BY n.train_index, n.node_index""", (event[key],),
            ).fetchall()
            if not members or _canonical_id(members) != event[key]:
                raise ValueError("Canonical merge event membership identity differs")
            parents.append(set(members))
        if parents[0] & parents[1] or parents[0] | parents[1] != parents[2]:
            raise ValueError("Canonical merge must preserve both complete disjoint parents")
        child = self.connection.execute("SELECT canonical_json FROM canonical_nodes WHERE canonical_id = ?",
                                        (event["child_canonical_id"],)).fetchone()
        if child is None or json.loads(child[0]) != decision.canonical_experience.to_dict():
            raise ValueError("Canonical child differs from its single MERGE response")
        identity = {"snapshot_id": snapshot_id, **dict(event)}
        event_sha = hashlib.sha256(canonical_json_bytes(identity)).hexdigest()
        values = (event_sha, snapshot_id, *(event[key] for key in keys))
        self.connection.execute(
            """INSERT INTO canonical_merge_events(
                event_sha256, snapshot_id, left_canonical_id, right_canonical_id,
                child_canonical_id, request_sha256, apply_order
            ) VALUES (?, ?, ?, ?, ?, ?, ?)""", values,
        )

    def put_canonical_resolution_event(
        self,
        *,
        stage: str,
        status: str,
        subject: Mapping[str, Any],
        evidence: Mapping[str, Any],
        created_snapshot_id: str,
    ) -> str:
        allowed = {"VIEW": {"VIEW_SOURCE_FALLBACK"}, "MERGE": {"MERGE_EXHAUSTED_NO_MERGE"}}
        if (
            stage not in allowed
            or status not in allowed[stage]
            or type(subject) is not dict
            or type(evidence) is not dict
            or type(created_snapshot_id) is not str
            or not created_snapshot_id
        ):
            raise ValueError("Canonical resolution event differs")
        subject_bytes = canonical_json_bytes(dict(subject))
        evidence_bytes = canonical_json_bytes(dict(evidence))
        subject_sha256 = hashlib.sha256(subject_bytes).hexdigest()
        identity = {
            "snapshot_id": created_snapshot_id,
            "stage": stage,
            "status": status,
            "subject_sha256": subject_sha256,
            "evidence_sha256": hashlib.sha256(evidence_bytes).hexdigest(),
        }
        event_sha256 = hashlib.sha256(canonical_json_bytes(identity)).hexdigest()
        values = (
            event_sha256,
            stage,
            status,
            subject_sha256,
            subject_bytes,
            evidence_bytes,
            created_snapshot_id,
        )
        try:
            self._connection.execute(
                """
                INSERT INTO canonical_resolution_events(
                    event_sha256, stage, status, subject_sha256,
                    subject_json, evidence_json, created_snapshot_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                values,
            )
        except sqlite3.IntegrityError:
            existing = self._connection.execute(
                """
                SELECT event_sha256, stage, status, subject_sha256,
                       subject_json, evidence_json, created_snapshot_id
                FROM canonical_resolution_events WHERE event_sha256 = ?
                """,
                (event_sha256,),
            ).fetchone()
            if existing != values:
                raise ValueError("Canonical resolution event conflicts")
        return event_sha256

    def put_graph_quality_audit(
        self,
        *,
        snapshot_id: str,
        status: str,
        audit_protocol_sha256: str,
        summary_sha256: str,
        source_node_audit_sha256: str,
        merge_ledger_sha256: str,
        unresolved_candidates_sha256: str,
        topology_sha256: str,
        hard_violations: Sequence[Mapping[str, Any]],
    ) -> None:
        hashes = (
            audit_protocol_sha256,
            summary_sha256,
            source_node_audit_sha256,
            merge_ledger_sha256,
            unresolved_candidates_sha256,
            topology_sha256,
        )
        if (
            type(snapshot_id) is not str
            or not snapshot_id
            or status not in {"READY", "NOT_READY"}
            or any(not _is_sha256(row) for row in hashes)
        ):
            raise ValueError("graph quality audit identity differs")
        self._connection.execute(
            """
            INSERT INTO graph_quality_audits(
                snapshot_id, status, audit_protocol_sha256, summary_sha256,
                source_node_audit_sha256, merge_ledger_sha256,
                unresolved_candidates_sha256, topology_sha256,
                hard_violations_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                snapshot_id,
                status,
                *hashes,
                canonical_json_bytes([dict(row) for row in hard_violations]),
            ),
        )

    @property
    def head_snapshot_id(self) -> str | None:
        row = self._connection.execute(
            "SELECT value FROM metadata WHERE key = 'head_snapshot_id'"
        ).fetchone()
        return None if row is None else str(row[0])

    def set_head_snapshot_id(self, snapshot_id: str) -> None:
        if type(snapshot_id) is not str or not snapshot_id:
            raise ValueError("snapshot identity differs")
        self._connection.execute(
            """
            INSERT INTO metadata(key, value) VALUES ('head_snapshot_id', ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (snapshot_id,),
        )

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "IncrementalStateStore":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
