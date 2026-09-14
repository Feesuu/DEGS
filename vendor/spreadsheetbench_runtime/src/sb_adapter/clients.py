from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .tokenization import TokenizerUnavailable, truncate_with_tokenizer
from .transport import validate_service_url


@dataclass
class EmbeddingConfig:
    base_url: str = "mock://deterministic"
    model: str = "Qwen3-Embedding-8B"
    api_key: str = "EMPTY"
    api_key_env_var: str | None = None
    # The default Qwen3-Embedding-8B service has a 32k input window. Callers
    # must chunk longer inputs before this final per-request guard is reached.
    max_model_len: int = 32000
    max_input_tokens: int | None = 32000
    truncate_long_texts: bool = True
    tokenizer_model: str | None = None
    tokenizer_required: bool = True
    tokenizer_backend: str = "huggingface"
    truncation_strategy: str = "head"
    batch_size: int = 8
    max_concurrency: int = 8
    cache_path: str | None = None
    cache_write_policy: str = "replace"
    deterministic_dim: int = 384
    trust_env: bool = True

    @property
    def effective_max_input_tokens(self) -> int:
        return int(self.max_input_tokens or self.max_model_len)

    @property
    def resolved_api_key(self) -> str:
        if self.api_key_env_var:
            return os.environ.get(self.api_key_env_var, self.api_key or "EMPTY")
        return self.api_key or "EMPTY"


def embedding_protocol_payload(
    config: EmbeddingConfig,
    *,
    model_name: str | None = None,
    include_execution_controls: bool = True,
) -> dict[str, Any]:
    payload = {
        "base_url": config.base_url,
        "model": model_name or config.model,
        "credential_source": (
            f"environment:{config.api_key_env_var}"
            if config.api_key_env_var
            else "empty"
            if config.resolved_api_key == "EMPTY"
            else "configured"
        ),
        "max_model_len": int(config.max_model_len),
        "max_input_tokens": int(config.effective_max_input_tokens),
        "truncate_long_texts": bool(config.truncate_long_texts),
        "tokenizer_model": config.tokenizer_model or "",
        "tokenizer_required": bool(config.tokenizer_required),
        "tokenizer_backend": config.tokenizer_backend,
        "truncation_strategy": config.truncation_strategy,
        "deterministic_dim": int(config.deterministic_dim),
        "trust_env": bool(config.trust_env),
    }
    if include_execution_controls:
        payload.update(
            {
                "batch_size": int(config.batch_size),
                "max_concurrency": int(config.max_concurrency),
                "cache_write_policy": config.cache_write_policy,
            }
        )
    return payload


def embedding_cache_namespace(
    config: EmbeddingConfig,
    logical_namespace: str,
    *,
    model_name: str | None = None,
) -> str:
    payload = embedding_protocol_payload(
        config,
        model_name=model_name or config.model,
        include_execution_controls=False,
    )
    digest = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return f"{logical_namespace}::protocol::{digest}"


def ordered_embedding_vectors(
    response_data: Any,
    *,
    expected_count: int,
) -> list[list[float]]:
    rows = list(response_data)
    if len(rows) != expected_count:
        raise RuntimeError(
            "embedding response count does not match request count: "
            f"expected={expected_count}, actual={len(rows)}"
        )
    vectors_by_index: dict[int, list[float]] = {}
    for row in rows:
        index = getattr(row, "index", None)
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or not 0 <= index < expected_count
            or index in vectors_by_index
        ):
            raise RuntimeError(
                "embedding response indices must be the unique range "
                f"0..{expected_count - 1}"
            )
        vector = list(row.embedding)
        if not vector or not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in vector
        ):
            raise RuntimeError("embedding response contains an invalid vector")
        vectors_by_index[index] = [float(value) for value in vector]
    if set(vectors_by_index) != set(range(expected_count)):
        raise RuntimeError(
            "embedding response indices do not cover every requested input"
        )
    ordered = [vectors_by_index[index] for index in range(expected_count)]
    dimensions = {len(vector) for vector in ordered}
    if len(dimensions) != 1:
        raise RuntimeError("embedding response vectors have inconsistent dimensions")
    return ordered


class EmbeddingClient:
    def __init__(self, config: EmbeddingConfig):
        validate_service_url(config.base_url)
        self.config = config
        if config.tokenizer_backend not in {"huggingface", "utf8_bytes", "regex"}:
            raise ValueError(
                "tokenizer_backend must be one of: huggingface, utf8_bytes, regex"
            )
        if config.tokenizer_backend == "regex" and not config.base_url.startswith("mock://"):
            raise ValueError("regex tokenizer backend is test-only and cannot be used with a real endpoint")
        self._sem = asyncio.Semaphore(max(1, int(config.max_concurrency)))
        self._cache: _SqliteEmbeddingCache | None = None
        self.truncation_events: list[dict[str, Any]] = []
        if config.cache_path:
            self._cache = _SqliteEmbeddingCache(
                config.cache_path,
                write_policy=config.cache_write_policy,
            )

    async def embed_texts(self, texts: list[str], *, model: str | None = None, batch_size: int | None = None, cache_namespace: str | None = None) -> list[list[float]]:
        if not texts:
            return []
        model_name = model or self.config.model
        namespace = self._cache_namespace(cache_namespace or model_name, model_name=model_name)
        prepared_pairs = [self._prepare_text(text, index=idx) for idx, text in enumerate(texts)]
        prepared_texts = [pair[0] for pair in prepared_pairs]
        results: list[list[float] | None] = [None] * len(prepared_texts)
        missing_by_text: dict[str, list[int]] = {}
        for idx, prepared_text in enumerate(prepared_texts):
            cached = self._cache.get(namespace, prepared_text) if self._cache else None
            if cached is not None:
                results[idx] = cached
            else:
                missing_by_text.setdefault(prepared_text, []).append(idx)
        missing = list(missing_by_text.items())
        bs = int(batch_size or self.config.batch_size or 1)
        batches = [missing[offset: offset + bs] for offset in range(0, len(missing), bs)]

        async def run_batch(batch: list[tuple[str, list[int]]]):
            batch_texts = [text for text, _indices in batch]
            vectors = await self._embed_uncached(batch_texts, model_name)
            return batch, vectors

        for batch, vectors in await asyncio.gather(*(run_batch(batch) for batch in batches)):
            if len(batch) != len(vectors):
                raise RuntimeError(
                    "embedding response count does not match request count"
                )
            for (prepared_text, indices), vector in zip(batch, vectors):
                if self._cache:
                    vector = self._cache.set(
                        namespace,
                        prepared_text,
                        vector,
                    )
                for idx in indices:
                    results[idx] = vector
        return [list(v or []) for v in results]

    def _cache_namespace(self, logical_namespace: str, *, model_name: str) -> str:
        return embedding_cache_namespace(
            self.config,
            logical_namespace,
            model_name=model_name,
        )

    def _prepare_text(self, text: str, *, index: int | None = None) -> tuple[str, dict[str, Any]]:
        max_tokens = max(1, int(self.config.effective_max_input_tokens))
        is_mock = self.config.base_url.startswith("mock://")
        backend = "regex" if is_mock else self.config.tokenizer_backend
        if backend == "huggingface":
            tokenizer_model = self.config.tokenizer_model or self.config.model
            fallback_mode = "regex" if not self.config.tokenizer_required else None
        else:
            tokenizer_model = None
            fallback_mode = backend
        try:
            result = truncate_with_tokenizer(
                text,
                tokenizer_model=tokenizer_model,
                max_tokens=max_tokens,
                strategy=self.config.truncation_strategy,
                allow_regex_fallback=fallback_mode == "regex",
                fallback_mode=fallback_mode,
            )
        except TokenizerUnavailable:
            raise
        if not result.truncated:
            return text, {
                "truncated": False,
                "token_count": result.token_count,
                "max_input_tokens": max_tokens,
                "tokenizer": result.tokenizer_name,
                "strategy": result.strategy,
            }
        if not self.config.truncate_long_texts:
            raise ValueError(
                f"embedding input exceeds max_input_tokens={max_tokens}: "
                f"token_count={result.token_count}; set truncate_long_texts=true to truncate"
            )
        event = {
            "index": index,
            "truncated": True,
            "original_chars": len(text),
            "truncated_chars": len(result.truncated_text),
            "token_count": result.token_count,
            "max_input_tokens": max_tokens,
            "tokenizer": result.tokenizer_name,
            "strategy": result.strategy,
        }
        self.truncation_events.append(event)
        return result.truncated_text, event

    def save_truncation_report(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "embedding_model": self.config.model,
            "max_model_len": self.config.max_model_len,
            "max_input_tokens": self.config.effective_max_input_tokens,
            "truncate_long_texts": self.config.truncate_long_texts,
            "tokenizer_model": self.config.tokenizer_model or self.config.model,
            "tokenizer_required": self.config.tokenizer_required,
            "tokenizer_backend": self.config.tokenizer_backend,
            "truncation_strategy": self.config.truncation_strategy,
            "event_count": len(self.truncation_events),
            "events": self.truncation_events,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    async def _embed_uncached(self, texts: list[str], model_name: str) -> list[list[float]]:
        async with self._sem:
            if self.config.base_url.startswith("mock://"):
                return [_deterministic_embedding(text, dim=self.config.deterministic_dim) for text in texts]
            # python-igraph can prevent Python 3.12's asyncio default executor
            # from shutting down. Use an explicit worker for the blocking SDK
            # call so graph builds terminate normally after Leiden is loaded.
            executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="embedding")
            try:
                return await asyncio.get_running_loop().run_in_executor(
                    executor,
                    self._embed_uncached_sync,
                    texts,
                    model_name,
                )
            finally:
                executor.shutdown(wait=True)

    def _embed_uncached_sync(self, texts: list[str], model_name: str) -> list[list[float]]:
        import httpx
        from openai import OpenAI

        client_kwargs: dict[str, Any] = {
            "api_key": self.config.resolved_api_key,
            "base_url": self.config.base_url,
        }
        if not self.config.trust_env:
            client_kwargs["http_client"] = httpx.Client(trust_env=False)
        with OpenAI(**client_kwargs) as client:
            response = client.embeddings.create(model=model_name, input=texts)
        _append_usage_record(
            "SB_ADAPTER_EMBEDDING_USAGE_LOG",
            {
                "component": "embedding",
                "client": "openai_embeddings",
                "model": model_name,
                "endpoint": self.config.base_url,
                "cache_hit": False,
                "usage": _response_usage_payload(response),
                "request": {"input_count": len(texts)},
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        )
        return ordered_embedding_vectors(
            response.data,
            expected_count=len(texts),
        )

    def close(self) -> None:
        if self._cache:
            self._cache.close()


def _safe_read_debug_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        _debug_io_warning(path, "read", exc)
        return None
    return payload if isinstance(payload, dict) else None


def _response_usage_payload(value: Any) -> dict[str, Any]:
    usage = getattr(value, "usage", None)
    if usage is None and isinstance(value, dict):
        usage = value.get("usage")
    if usage is None:
        return {}
    if isinstance(usage, dict):
        return dict(usage)
    if hasattr(usage, "model_dump"):
        return dict(usage.model_dump())
    if hasattr(usage, "dict"):
        return dict(usage.dict())
    payload: dict[str, Any] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens", "input_tokens", "output_tokens"):
        token_value = getattr(usage, key, None)
        if token_value is not None:
            payload[key] = token_value
    return payload


def _append_usage_record(env_var: str, payload: dict[str, Any]) -> None:
    path = os.getenv(env_var)
    if not path:
        return
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str) + "\n"
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, line.encode("utf-8"))
        finally:
            os.close(fd)
    except Exception as exc:  # pragma: no cover - telemetry must not break builds
        print(
            "[sb-adapter-usage-warning] "
            f"path={path} error={type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )


class _SqliteEmbeddingCache:
    def __init__(
        self,
        path: str | Path,
        *,
        write_policy: str = "replace",
    ):
        self.path = Path(path)
        self.write_policy = str(write_policy)
        if self.write_policy not in {"replace", "first_write_wins"}:
            raise ValueError(
                "embedding cache write_policy must be 'replace' or "
                "'first_write_wins'"
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, timeout=60.0)
        self.conn.execute("PRAGMA busy_timeout=60000")
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("CREATE TABLE IF NOT EXISTS embeddings (namespace TEXT, key TEXT, vector TEXT, PRIMARY KEY(namespace, key))")
        self.conn.commit()

    def _key(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get(self, namespace: str, text: str) -> list[float] | None:
        cur = self.conn.execute("SELECT vector FROM embeddings WHERE namespace=? AND key=?", (namespace, self._key(text)))
        row = cur.fetchone()
        return json.loads(row[0]) if row else None

    def set(
        self,
        namespace: str,
        text: str,
        vector: list[float],
    ) -> list[float]:
        key = self._key(text)
        insert_mode = (
            "REPLACE"
            if self.write_policy == "replace"
            else "IGNORE"
        )
        self.conn.execute(
            f"INSERT OR {insert_mode} INTO embeddings(namespace,key,vector) "
            "VALUES(?,?,?)",
            (namespace, key, json.dumps(vector)),
        )
        self.conn.commit()
        row = self.conn.execute(
            "SELECT vector FROM embeddings WHERE namespace=? AND key=?",
            (namespace, key),
        ).fetchone()
        if row is None:
            raise RuntimeError("embedding cache write did not persist a vector")
        return list(json.loads(row[0]))

    def close(self) -> None:
        self.conn.close()


def _deterministic_embedding(text: str, *, dim: int) -> list[float]:
    if dim < 1:
        raise ValueError("deterministic embedding dimension must be positive")
    vec = [0.0] * dim
    tokens = [tok for tok in ''.join(ch.lower() if ch.isalnum() else ' ' for ch in text).split() if tok]
    for tok in tokens:
        digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(value * value for value in vec))
    if norm <= 1.0e-12:
        vec[0] = 1.0
        norm = 1.0
    return [value / norm for value in vec]
