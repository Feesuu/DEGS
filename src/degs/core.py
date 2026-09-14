from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib
import json
import math
import struct
from typing import Any, Mapping, MutableMapping, Protocol, Sequence
import unicodedata


EMBEDDING_MODEL = "Qwen3-Embedding-8B"
EMBEDDING_TIMEOUT_S = 600
EMBEDDING_BATCH_SIZE = 32
EMBEDDING_ASYNC_WORKERS = 16
TRAIN_INSTRUCTION_COUNT = 200


def _validate_json_value(value: Any, *, path: str = "$") -> None:
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"canonical JSON contains a non-finite number at {path}")
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _validate_json_value(item, path=f"{path}[{index}]")
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError(f"canonical JSON has a non-string key at {path}")
            _validate_json_value(item, path=f"{path}.{key}")
        return
    raise ValueError(f"canonical JSON contains a non-JSON value at {path}")


def canonical_json_bytes(value: Any) -> bytes:
    _validate_json_value(value)
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha256(value: Any, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} SHA-256 differs")
    return value


def normalize_embedding_text(text: str) -> str:
    if type(text) is not str:
        raise ValueError("embedding text must be a string")
    return unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")


def embedding_request_payload(texts: Sequence[str]) -> dict[str, Any]:
    normalized = [normalize_embedding_text(text) for text in texts]
    if not normalized or len(normalized) > EMBEDDING_BATCH_SIZE:
        raise ValueError("embedding batch must contain 1..32 texts")
    return {
        "format": "degs_embedding_request_v1",
        "model": EMBEDDING_MODEL,
        "texts": normalized,
        "timeout_s": EMBEDDING_TIMEOUT_S,
        "semantic_attempts": 1,
        "transport_retries": 0,
    }


def embedding_request_sha256(texts: Sequence[str]) -> str:
    return _sha256_json(embedding_request_payload(texts))


def embedding_text_sha256(text: str) -> str:
    return hashlib.sha256(normalize_embedding_text(text).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EmbeddedText:
    normalized_text: str
    vector: tuple[float, ...]
    vector_sha256: str
    request_sha256: str


class EmbeddingTransport(Protocol):
    def embed(self, request: Mapping[str, Any]) -> Mapping[str, Any]: ...


class StrictEmbeddingAdapter:
    def __init__(
        self,
        transport: EmbeddingTransport,
        *,
        cache: MutableMapping[str, Any] | None = None,
        expected_dimension: int | None = None,
    ) -> None:
        if expected_dimension is not None and expected_dimension <= 0:
            raise ValueError("embedding expected dimension differs")
        self.transport = transport
        self.cache = cache if cache is not None else {}
        self.expected_dimension = expected_dimension

    def _validate_batch(
        self,
        value: Any,
        *,
        texts: tuple[str, ...],
        request_sha256: str,
    ) -> tuple[EmbeddedText, ...]:
        if type(value) is not dict or value.get("model") != EMBEDDING_MODEL:
            raise ValueError("embedding response model differs")
        data = value.get("data")
        if type(data) is not list or len(data) != len(texts):
            raise ValueError("embedding response shape differs")
        rows: list[EmbeddedText] = []
        for expected_index, raw in enumerate(data):
            if type(raw) is not dict or raw.get("index") != expected_index:
                raise ValueError("embedding response order differs")
            vector_raw = raw.get("embedding")
            if type(vector_raw) is not list or not vector_raw:
                raise ValueError("embedding vector shape differs")
            vector: list[float] = []
            for item in vector_raw:
                if type(item) not in (int, float) or not math.isfinite(float(item)):
                    raise ValueError("embedding vector contains a non-finite value")
                vector.append(float(item))
            dimension = len(vector)
            if self.expected_dimension is None:
                self.expected_dimension = dimension
            if dimension != self.expected_dimension:
                raise ValueError("embedding vector dimension differs")
            norm = math.sqrt(sum(item * item for item in vector))
            if not math.isfinite(norm) or norm == 0.0:
                raise ValueError("embedding vector must be finite and non-zero")
            vector_bytes = b"".join(struct.pack("<d", item) for item in vector)
            rows.append(
                EmbeddedText(
                    texts[expected_index],
                    tuple(vector),
                    hashlib.sha256(vector_bytes).hexdigest(),
                    request_sha256,
                )
            )
        return tuple(rows)

    def _validate_cached(self, value: Any, *, text: str) -> EmbeddedText:
        if type(value) is not EmbeddedText:
            raise ValueError("embedding cache row differs")
        if value.normalized_text != text:
            raise ValueError("embedding cache text identity differs")
        _sha256(value.request_sha256, label="embedding producer request")
        checked = self._validate_batch(
            {
                "model": EMBEDDING_MODEL,
                "data": [{"index": 0, "embedding": list(value.vector)}],
            },
            texts=(text,),
            request_sha256=value.request_sha256,
        )
        if checked[0].vector_sha256 != value.vector_sha256:
            raise ValueError("embedding cache vector identity differs")
        return value

    def _cached_and_missing(
        self, texts: Sequence[str]
    ) -> tuple[tuple[str, ...], dict[str, EmbeddedText], tuple[str, ...]]:
        normalized = tuple(normalize_embedding_text(text) for text in texts)
        if not normalized:
            raise ValueError("embedding input must be non-empty")
        cached: dict[str, EmbeddedText] = {}
        missing: list[str] = []
        for text in dict.fromkeys(normalized):
            key = embedding_text_sha256(text)
            if key in self.cache:
                cached[text] = self._validate_cached(self.cache[key], text=text)
            else:
                missing.append(text)
        return normalized, cached, tuple(missing)

    def _request_batch(self, batch: tuple[str, ...]) -> tuple[EmbeddedText, ...]:
        request_sha = _sha256_json(embedding_request_payload(batch))
        return self._validate_batch(
            self.transport.embed(
                {
                    "model": EMBEDDING_MODEL,
                    "input": list(batch),
                    "timeout_s": EMBEDDING_TIMEOUT_S,
                    "semantic_attempts": 1,
                    "transport_retries": 0,
                    "embedding_request_sha256": request_sha,
                }
            ),
            texts=batch,
            request_sha256=request_sha,
        )

    def _commit_rows(
        self,
        normalized: tuple[str, ...],
        cached: MutableMapping[str, EmbeddedText],
        batches: Sequence[Sequence[EmbeddedText]],
    ) -> tuple[EmbeddedText, ...]:
        for batch in batches:
            for row in batch:
                key = embedding_text_sha256(row.normalized_text)
                if key in self.cache:
                    existing = self._validate_cached(
                        self.cache[key], text=row.normalized_text
                    )
                    if existing.vector_sha256 != row.vector_sha256:
                        raise ValueError("embedding cache contains conflicting vectors")
                    cached[row.normalized_text] = existing
                    continue
                self.cache[key] = row
                cached[row.normalized_text] = row
        if set(cached) != set(normalized):
            raise ValueError("embedding cache did not cover every requested text")
        return tuple(cached[text] for text in normalized)

    def embed(self, texts: Sequence[str]) -> tuple[EmbeddedText, ...]:
        normalized, cached, missing = self._cached_and_missing(texts)
        batches: list[tuple[EmbeddedText, ...]] = []
        for start in range(0, len(missing), EMBEDDING_BATCH_SIZE):
            batch = missing[start : start + EMBEDDING_BATCH_SIZE]
            batches.append(self._request_batch(batch))
        return self._commit_rows(normalized, cached, batches)

    async def _request_batch_async(
        self, batch: tuple[str, ...]
    ) -> tuple[EmbeddedText, ...]:
        request_sha = _sha256_json(embedding_request_payload(batch))
        request = {
            "model": EMBEDDING_MODEL,
            "input": list(batch),
            "timeout_s": EMBEDDING_TIMEOUT_S,
            "semantic_attempts": 1,
            "transport_retries": 0,
            "embedding_request_sha256": request_sha,
        }
        async_embed = getattr(self.transport, "embed_async", None)
        response = (
            await async_embed(request)
            if callable(async_embed)
            else self.transport.embed(request)
        )
        return self._validate_batch(
            response,
            texts=batch,
            request_sha256=request_sha,
        )

    async def embed_async(
        self,
        texts: Sequence[str],
        *,
        workers: int = EMBEDDING_ASYNC_WORKERS,
    ) -> tuple[EmbeddedText, ...]:
        if type(workers) is not int or workers <= 0:
            raise ValueError("embedding async workers must be positive")
        normalized, cached, missing = self._cached_and_missing(texts)
        missing_batches = tuple(
            missing[start : start + EMBEDDING_BATCH_SIZE]
            for start in range(0, len(missing), EMBEDDING_BATCH_SIZE)
        )
        semaphore = asyncio.Semaphore(workers)

        async def request(batch: tuple[str, ...]) -> tuple[EmbeddedText, ...]:
            async with semaphore:
                return await self._request_batch_async(batch)

        batches = await asyncio.gather(*(request(batch) for batch in missing_batches))
        return self._commit_rows(normalized, cached, batches)


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("cosine vector dimension differs")
    left_values = tuple(float(value) for value in left)
    right_values = tuple(float(value) for value in right)
    if any(not math.isfinite(value) for value in (*left_values, *right_values)):
        raise ValueError("cosine vector contains a non-finite value")
    left_norm = math.sqrt(sum(value * value for value in left_values))
    right_norm = math.sqrt(sum(value * value for value in right_values))
    if left_norm == 0.0 or right_norm == 0.0:
        raise ValueError("cosine vector must be non-zero")
    return sum(left * right for left, right in zip(left_values, right_values)) / (
        left_norm * right_norm
    )


@dataclass(frozen=True)
class InstructionAuthorityRow:
    train_index: int
    task_id: str
    instruction: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_index": self.train_index,
            "task_id": self.task_id,
            "instruction": self.instruction,
        }


@dataclass(frozen=True)
class SealedInstructionAuthority:
    rows: tuple[InstructionAuthorityRow, ...]
    authority_sha256: str

    def __post_init__(self) -> None:
        _sha256(self.authority_sha256, label="instruction authority")
        if len(self.rows) != TRAIN_INSTRUCTION_COUNT:
            raise ValueError("instruction authority population differs")
        for index, row in enumerate(self.rows):
            if type(row) is not InstructionAuthorityRow or row.train_index != index:
                raise ValueError("instruction authority row order differs")

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "degs_instruction_authority_v1",
            "authority_sha256": self.authority_sha256,
            "rows": [row.to_dict() for row in self.rows],
        }


__all__ = [
    "EMBEDDING_ASYNC_WORKERS",
    "EMBEDDING_BATCH_SIZE",
    "EMBEDDING_MODEL",
    "EMBEDDING_TIMEOUT_S",
    "EmbeddedText",
    "InstructionAuthorityRow",
    "SealedInstructionAuthority",
    "StrictEmbeddingAdapter",
    "TRAIN_INSTRUCTION_COUNT",
    "canonical_json_bytes",
    "cosine_similarity",
    "embedding_request_sha256",
    "embedding_text_sha256",
    "normalize_embedding_text",
]
