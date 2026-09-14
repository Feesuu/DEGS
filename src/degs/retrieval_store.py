from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterator, Mapping

from .core import canonical_json_bytes
from .state_store import EMBEDDING_CACHE_NAMESPACE, SQLiteEmbeddingCache, _is_sha256


RETRIEVAL_STORE_SCHEMA_VERSION = 1
_SCHEMA = """
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
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

CREATE TABLE IF NOT EXISTS retrieval_jobs (
    request_sha256 TEXT PRIMARY KEY,
    stage TEXT NOT NULL CHECK(stage IN ('NEED_GRAPH', 'CLARIFICATION', 'SELECTOR')),
    prompt_sha256 TEXT NOT NULL,
    producer_protocol_sha256 TEXT NOT NULL,
    validated_response_json BLOB NOT NULL,
    audit_json BLOB NOT NULL
) STRICT;
"""


class RetrievalStore:
    """Target-local cache for online retrieval; it never owns source graph state."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path).expanduser().absolute()
        if self.path.exists() and not self.path.is_file():
            raise ValueError("retrieval cache path differs")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path, isolation_level=None)
        try:
            self._connection.execute("PRAGMA journal_mode = WAL")
            self._connection.execute("PRAGMA busy_timeout = 30000")
            self._connection.executescript(_SCHEMA)
            self._initialize_identity()
        except BaseException:
            self._connection.close()
            raise

    def _initialize_identity(self) -> None:
        expected = {
            "schema_version": str(RETRIEVAL_STORE_SCHEMA_VERSION),
            "embedding_cache_namespace": EMBEDDING_CACHE_NAMESPACE,
        }
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
                    raise ValueError(f"retrieval cache {key} differs")

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

    def embedding_cache(self) -> SQLiteEmbeddingCache:
        return SQLiteEmbeddingCache(self._connection)

    @property
    def embedding_endpoint(self) -> str | None:
        row = self._connection.execute(
            "SELECT value FROM metadata WHERE key = 'embedding_endpoint'"
        ).fetchone()
        return None if row is None else str(row[0])

    def bind_embedding_endpoint(self, endpoint: str) -> None:
        if not endpoint:
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

    def get_retrieval_job(
        self,
        request_sha256: str,
        *,
        stage: str,
        prompt_sha256: str,
        producer_protocol_sha256: str,
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        if not _is_sha256(request_sha256):
            raise ValueError("retrieval request identity differs")
        row = self._connection.execute(
            """
            SELECT stage, prompt_sha256, producer_protocol_sha256,
                   validated_response_json, audit_json
            FROM retrieval_jobs WHERE request_sha256 = ?
            """,
            (request_sha256,),
        ).fetchone()
        if row is None:
            return None
        if tuple(row[:3]) != (stage, prompt_sha256, producer_protocol_sha256):
            raise ValueError("retrieval cache protocol identity differs")
        response = json.loads(bytes(row[3]))
        audit = json.loads(bytes(row[4]))
        if type(response) is not dict or type(audit) is not dict:
            raise ValueError("retrieval cache payload differs")
        return response, audit

    def put_retrieval_job(
        self,
        request_sha256: str,
        *,
        stage: str,
        prompt_sha256: str,
        producer_protocol_sha256: str,
        response: Mapping[str, Any],
        audit: Mapping[str, Any],
    ) -> None:
        if not _is_sha256(request_sha256):
            raise ValueError("retrieval request identity differs")
        values = (
            request_sha256,
            stage,
            prompt_sha256,
            producer_protocol_sha256,
            canonical_json_bytes(dict(response)),
            canonical_json_bytes(dict(audit)),
        )
        try:
            self._connection.execute(
                """
                INSERT INTO retrieval_jobs(
                    request_sha256, stage, prompt_sha256,
                    producer_protocol_sha256, validated_response_json, audit_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                values,
            )
        except sqlite3.IntegrityError:
            existing = self.get_retrieval_job(
                request_sha256,
                stage=stage,
                prompt_sha256=prompt_sha256,
                producer_protocol_sha256=producer_protocol_sha256,
            )
            if existing != (dict(response), dict(audit)):
                raise ValueError("retrieval cache contains conflicting output")

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "RetrievalStore":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


__all__ = ["RETRIEVAL_STORE_SCHEMA_VERSION", "RetrievalStore"]
