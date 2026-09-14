from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Awaitable, Callable, Mapping, Sequence
import urllib.request

import httpx

from sb_adapter.transport import validate_service_url

from .core import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL,
    EMBEDDING_TIMEOUT_S,
    TRAIN_INSTRUCTION_COUNT,
    InstructionAuthorityRow,
    SealedInstructionAuthority,
    canonical_json_bytes,
    embedding_request_sha256,
)


_SAFE_TASK_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_EMBEDDING_REQUEST_FIELDS = {
    "model",
    "input",
    "timeout_s",
    "semantic_attempts",
    "transport_retries",
    "embedding_request_sha256",
}

PostCallable = Callable[[str, bytes, Mapping[str, str], int], Any]
AsyncPostCallable = Callable[
    [str, bytes, Mapping[str, str], int], Awaitable[Any]
]


class TransportError(RuntimeError):
    """The fixed embedding endpoint did not satisfy its closed protocol."""


@dataclass(frozen=True)
class InstructionAuthoritySeal:
    authority: SealedInstructionAuthority
    canonical_bytes: bytes
    sha256: str

    def __post_init__(self) -> None:
        if type(self.authority) is not SealedInstructionAuthority:
            raise ValueError("sealed instruction authority capability is required")
        if type(self.canonical_bytes) is not bytes:
            raise ValueError("instruction authority bytes differ")
        if (
            type(self.sha256) is not str
            or len(self.sha256) != 64
            or hashlib.sha256(self.canonical_bytes).hexdigest() != self.sha256
            or self.authority.authority_sha256 != self.sha256
        ):
            raise ValueError("instruction authority seal differs")


def _task_id(raw: Any) -> str:
    if type(raw) is str:
        task_id = raw
    elif type(raw) is int and raw >= 0:
        task_id = str(raw)
    else:
        raise ValueError("train instruction task ID differs")
    if _SAFE_TASK_ID.fullmatch(task_id) is None:
        raise ValueError("train instruction task ID differs")
    return task_id


def seal_train_instruction_authority(
    dataset_train_0_200: Sequence[Mapping[str, Any]],
) -> InstructionAuthoritySeal:
    """Project raw train[0,200) rows to the only method-visible authority."""

    if (
        isinstance(dataset_train_0_200, (str, bytes, bytearray))
        or not isinstance(dataset_train_0_200, Sequence)
        or len(dataset_train_0_200) != TRAIN_INSTRUCTION_COUNT
    ):
        raise ValueError("authority preprocessing requires exactly train[0,200)")
    rows: list[InstructionAuthorityRow] = []
    seen_task_ids: set[str] = set()
    for train_index, raw in enumerate(dataset_train_0_200):
        if not isinstance(raw, Mapping):
            raise ValueError("train instruction row differs")
        task_id = _task_id(raw.get("id"))
        instruction = raw.get("instruction")
        if type(instruction) is not str or not instruction or len(instruction) > 8000:
            raise ValueError("train instruction text differs")
        if task_id in seen_task_ids:
            raise ValueError("train instruction task ID is repeated")
        seen_task_ids.add(task_id)
        rows.append(InstructionAuthorityRow(train_index, task_id, instruction))
    payload = {
        "format": "degs_instruction_authority_v1",
        "rows": [row.to_dict() for row in rows],
    }
    authority_bytes = canonical_json_bytes(payload)
    authority_sha256 = hashlib.sha256(authority_bytes).hexdigest()
    authority = SealedInstructionAuthority(tuple(rows), authority_sha256)
    return InstructionAuthoritySeal(authority, authority_bytes, authority_sha256)


def _default_post(
    url: str, body: bytes, headers: Mapping[str, str], timeout_s: int
) -> bytes:
    request = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=timeout_s) as response:
            if response.getcode() != 200:
                raise TransportError(f"HTTP status {response.getcode()} differs")
            return response.read()
    except TransportError:
        raise
    except Exception as exc:
        raise TransportError("HTTP request failed") from exc


async def _default_post_async(
    url: str, body: bytes, headers: Mapping[str, str], timeout_s: int
) -> bytes:
    try:
        async with httpx.AsyncClient(trust_env=False, timeout=timeout_s) as client:
            response = await client.post(url, content=body, headers=dict(headers))
    except Exception as exc:
        raise TransportError("HTTP request failed") from exc
    if response.status_code != 200:
        raise TransportError(f"HTTP status {response.status_code} differs")
    return response.content


def _response_object(value: Any) -> Mapping[str, Any]:
    if isinstance(value, (bytes, bytearray)):
        try:
            value = bytes(value).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise TransportError("embedding response is not UTF-8") from exc
    if type(value) is str:
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise TransportError("embedding response is not JSON") from exc
    if not isinstance(value, Mapping):
        raise TransportError("embedding response is not an object")
    return value


class QwenEmbeddingHTTPTransport:
    __slots__ = ("_api_key", "_endpoint", "_post", "_async_post")

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        post: PostCallable | None = None,
        async_post: AsyncPostCallable | None = None,
    ) -> None:
        if type(base_url) is not str:
            raise ValueError("embedding base URL differs")
        validate_service_url(base_url)
        normalized = base_url.rstrip("/")
        if not normalized.endswith("/v1"):
            raise ValueError("embedding base URL must end with /v1")
        if api_key is not None and (type(api_key) is not str or not api_key):
            raise ValueError("embedding API key differs")
        self._endpoint = f"{normalized}/embeddings"
        self._api_key = api_key
        self._post = post or _default_post
        self._async_post = async_post or _default_post_async

    @property
    def endpoint(self) -> str:
        return self._endpoint

    def __repr__(self) -> str:
        return f"{type(self).__name__}(endpoint={self._endpoint!r}, api_key=<redacted>)"

    def _request_parts(
        self, request: Mapping[str, Any]
    ) -> tuple[bytes, Mapping[str, str]]:
        if type(request) is not dict or set(request) != _EMBEDDING_REQUEST_FIELDS:
            raise TransportError("embedding request fields differ")
        inputs = request["input"]
        if (
            request["model"] != EMBEDDING_MODEL
            or type(inputs) is not list
            or not 1 <= len(inputs) <= EMBEDDING_BATCH_SIZE
            or any(type(text) is not str for text in inputs)
            or request["timeout_s"] != EMBEDDING_TIMEOUT_S
            or request["semantic_attempts"] != 1
            or request["transport_retries"] != 0
            or request["embedding_request_sha256"] != embedding_request_sha256(inputs)
        ):
            raise TransportError("embedding request protocol differs")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._api_key is not None:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return canonical_json_bytes({"model": EMBEDDING_MODEL, "input": inputs}), headers

    @staticmethod
    def _validated_response(raw_response: Any) -> Mapping[str, Any]:
        response = _response_object(raw_response)
        if response.get("model") != EMBEDDING_MODEL or type(response.get("data")) is not list:
            raise TransportError("embedding response model or data differs")
        return copy.deepcopy(dict(response))

    def embed(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        body, headers = self._request_parts(request)
        try:
            raw_response = self._post(
                self._endpoint,
                body,
                headers,
                EMBEDDING_TIMEOUT_S,
            )
        except TransportError:
            raise
        except Exception as exc:
            raise TransportError("embedding HTTP request failed") from exc
        return self._validated_response(raw_response)

    async def embed_async(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        body, headers = self._request_parts(request)
        try:
            raw_response = await self._async_post(
                self._endpoint,
                body,
                headers,
                EMBEDDING_TIMEOUT_S,
            )
        except TransportError:
            raise
        except Exception as exc:
            raise TransportError("embedding HTTP request failed") from exc
        return self._validated_response(raw_response)


__all__ = [
    "InstructionAuthoritySeal",
    "QwenEmbeddingHTTPTransport",
    "TransportError",
    "seal_train_instruction_authority",
]
