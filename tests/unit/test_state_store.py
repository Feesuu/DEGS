from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
import struct
import threading

import pytest

from degs.core import (
    EMBEDDING_MODEL,
    StrictEmbeddingAdapter,
    embedding_text_sha256,
)
from degs.state_store import IncrementalStateStore
from degs.state_store import EMBEDDING_CACHE_NAMESPACE


class RecordingTransport:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []
        self._lock = threading.Lock()

    def embed(self, request):
        texts = tuple(request["input"])
        with self._lock:
            self.calls.append(texts)
        return {
            "model": EMBEDDING_MODEL,
            "data": [
                {
                    "index": index,
                    "embedding": [float(len(text)), float(index + 1)],
                }
                for index, text in enumerate(texts)
            ],
        }

    async def embed_async(self, request):
        await asyncio.sleep(0)
        return self.embed(request)


def test_persistent_embedding_cache_is_keyed_by_normalized_text(
    tmp_path: Path,
) -> None:
    database = tmp_path / "graph_state.sqlite3"
    first_transport = RecordingTransport()
    with IncrementalStateStore(database) as state:
        rows = StrictEmbeddingAdapter(
            first_transport, cache=state.embedding_cache()
        ).embed(["cafe\u0301\r\nrow", "café\nrow", "different"])

    assert len(first_transport.calls) == 1
    assert len(first_transport.calls[0]) == 2
    assert rows[0] == rows[1]
    second_transport = RecordingTransport()
    with IncrementalStateStore(database) as state:
        cached = StrictEmbeddingAdapter(
            second_transport, cache=state.embedding_cache()
        ).embed(["café\nrow", "different"])
        assert len(state.embedding_cache()) == 2

    assert second_transport.calls == []
    assert cached == (rows[0], rows[2])


def test_async_embedding_requests_only_missing_batches_and_preserves_order(
    tmp_path: Path,
) -> None:
    transport = RecordingTransport()
    texts = [f"text-{index:03d}" for index in range(70)]
    with IncrementalStateStore(tmp_path / "state.sqlite3") as state:
        adapter = StrictEmbeddingAdapter(transport, cache=state.embedding_cache())
        rows = asyncio.run(adapter.embed_async(texts + [texts[0]], workers=16))

    assert sorted(len(batch) for batch in transport.calls) == [6, 32, 32]
    assert [row.normalized_text for row in rows] == texts + [texts[0]]
    assert rows[-1] == rows[0]


def test_embedding_cache_detects_corrupted_vector_blob(tmp_path: Path) -> None:
    database = tmp_path / "state.sqlite3"
    transport = RecordingTransport()
    with IncrementalStateStore(database) as state:
        cache = state.embedding_cache()
        StrictEmbeddingAdapter(transport, cache=cache).embed(["one"])
        key = embedding_text_sha256("one")
        state.connection.execute(
            """
            UPDATE embedding_cache SET vector_blob = ?
            WHERE namespace_sha256 = ? AND normalized_text_sha256 = ?
            """,
            (struct.pack("<dd", 9.0, 9.0), cache.namespace_sha256, key),
        )
        with pytest.raises(ValueError, match="vector hash"):
            _ = cache[key]


def test_embedding_cache_rejects_conflicting_vector_for_same_text(
    tmp_path: Path,
) -> None:
    transport = RecordingTransport()
    with IncrementalStateStore(tmp_path / "state.sqlite3") as state:
        cache = state.embedding_cache()
        row = StrictEmbeddingAdapter(transport, cache=cache).embed(["same"])[0]
        vector = (99.0, 1.0)
        vector_blob = b"".join(struct.pack("<d", item) for item in vector)
        conflicting = type(row)(
            row.normalized_text,
            vector,
            hashlib.sha256(vector_blob).hexdigest(),
            row.request_sha256,
        )
        with pytest.raises(ValueError, match="conflicting vectors"):
            cache[embedding_text_sha256("same")] = conflicting


def test_schema9_has_only_the_two_producer_stages_and_no_old_veto_tables(tmp_path: Path) -> None:
    with IncrementalStateStore(tmp_path / "state.sqlite3") as state:
        tables = {row[0] for row in state.connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        assert {"canonical_merge_events", "canonical_neighbors", "canonical_views"} <= tables
        assert not {"canonical_relation_evidence", "canonical_fidelity_checks", "canonical_merge_proofs",
                    "canonical_transitions", "experience_neighbors"} & tables
