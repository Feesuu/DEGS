"""
LLM client abstractions for the ReAct agent.

This module provides a simple interface to interact with LLMs,
with a default implementation for OpenAI-compatible APIs.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

OPENAI_SDK_MAX_RETRIES = 0


class RequestContextLengthExceeded(RuntimeError):
    """Raised when a request exceeds the model context window."""


class RequestCompletionLengthExceeded(RuntimeError):
    """Raised when a model response reaches its configured completion limit."""

    def __init__(
        self,
        message: str,
        *,
        partial_content: str = "",
        reasoning_content: str = "",
    ) -> None:
        super().__init__(message)
        self.partial_content = partial_content
        self.reasoning_content = reasoning_content


class RequestRuntimeTimeout(TimeoutError):
    """Raised when request execution times out after configured replay attempts."""


def _extract_openai_error_message(exc: Exception) -> str:
    """Best-effort extraction of the provider error message."""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        details = error if isinstance(error, dict) else body
        message = details.get("message")
        if isinstance(message, str):
            return message
    return str(exc)


def _is_context_length_bad_request(exc: Exception) -> bool:
    """Return True when the provider rejected the request for context length."""
    body = getattr(exc, "body", None)
    param = None
    if isinstance(body, dict):
        error = body.get("error")
        details = error if isinstance(error, dict) else body
        param = details.get("param")

    message = _extract_openai_error_message(exc).lower()
    return (
        "maximum context length" in message
        or (
            "maximum input length" in message
            and (param == "input_tokens" or "input token" in message)
        )
        or "context window" in message and "exceed" in message
    )


def _is_runtime_timeout(exc: Exception) -> bool:
    """Recognize SDK and transport timeout errors through their exception chain."""
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, TimeoutError):
            return True
        if "timeout" in type(current).__name__.lower():
            return True
        current = current.__cause__ or current.__context__
    return "timed out" in _extract_openai_error_message(exc).lower()


def _raise_terminal_request_error(exc: Exception) -> None:
    if _is_context_length_bad_request(exc):
        raise RequestContextLengthExceeded(
            _extract_openai_error_message(exc)
        ) from exc
    if _is_runtime_timeout(exc):
        raise RequestRuntimeTimeout(
            "runtime_invalid_timeout: "
            + _extract_openai_error_message(exc)
        ) from exc


@dataclass
class Message:
    """A single message in a conversation."""
    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class ModelSettings:
    """Settings for LLM generation."""
    temperature: float | None = None
    max_tokens: int | None = None
    stop: list[str] = field(default_factory=list)
    extra_body: dict = field(default_factory=dict)
    response_format: dict[str, Any] | None = None
    
    def to_dict(self) -> dict:
        result = {}
        if self.temperature is not None:
            result["temperature"] = self.temperature
        if self.max_tokens is not None:
            result["max_tokens"] = self.max_tokens
        if self.stop:
            result["stop"] = self.stop
        if self.extra_body:
            result["extra_body"] = self.extra_body
        if self.response_format is not None:
            result["response_format"] = self.response_format
        return result


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
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        line = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str) + "\n"
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, line.encode("utf-8"))
        finally:
            os.close(fd)
    except Exception as exc:  # pragma: no cover - usage telemetry must not break rollouts
        log.warning("Failed to write usage record to %s: %s", path, exc)


def _record_react_usage(
    *,
    client: str,
    model: str,
    endpoint: str,
    cache_hit: bool,
    response: Any | None = None,
    request: dict[str, Any] | None = None,
) -> None:
    _append_usage_record(
        "REACT_AGENT_USAGE_LOG",
        {
            "component": "react_agent",
            "client": client,
            "model": model,
            "endpoint": endpoint,
            "cache_hit": cache_hit,
            "usage": _response_usage_payload(response),
            "request": dict(request or {}),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )


def record_runtime_event(event: str, **details: Any) -> None:
    """Append one runtime-control event when telemetry is enabled."""
    _append_usage_record(
        "REACT_AGENT_RUNTIME_EVENT_LOG",
        {
            "component": "react_agent",
            "event": event,
            **details,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )


TRANSIENT_REPLY_PATTERNS = (
    r"\b429\b",
    r"mpe-429",
    r"resource exhausted",
    r"resource_exhausted",
    r"rate limit",
    r"too many requests",
    r"请求服务异常",
    r"模型提供方限流",
)


def _is_transient_error_reply(reply: str) -> bool:
    """Return True when the reply looks like a retriable provider error."""
    import re

    if not reply:
        return False
    text = reply.strip().lower()
    return any(re.search(pattern, text) for pattern in TRANSIENT_REPLY_PATTERNS)


class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    def chat(self, messages: list[Message], settings: ModelSettings | None = None) -> str:
        """Send messages to the LLM and get a response."""
        pass
    
    @abstractmethod
    async def chat_async(self, messages: list[Message], settings: ModelSettings | None = None) -> str:
        """Async version of chat."""
        pass


# Generation config presets for various models (useful for vLLM/local serving)
GENERATION_CONFIG_PRESETS: dict[str, dict[str, Any]] = {
    "openai/gpt-oss-120b": {
        "extra_body": {"reasoning_effort": "medium"},
    },
    "Qwen/Qwen3-8B": {
        "temperature": 0.6,
        "top_p": 0.95,
        "extra_body": {"enable_thinking": True, "top_k": 20},
    },
    "microsoft/Phi-4-reasoning-plus": {
        "temperature": 0.8,
        "top_p": 0.95,
        "extra_body": {"enable_thinking": True, "top_k": 50},
    },
    "deepseek-reasoner": {
        "temperature": 0.6,
    },
}


class OpenAIClient(LLMClient):
    """
    OpenAI-compatible LLM client with explicit retry logic.
    
    Works with OpenAI API and compatible services (Azure, vLLM, LiteLLM, etc.)
    Supports reasoning models with thinking content parsing.
    """
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        generation_config: dict | None = None,
        retry_times: tuple[int, ...] = (5, 10, 30),
        timeout: float | None = 600.0,
        trust_env: bool = True,
        runtime_timeout_retries: int = 0,
    ):
        """
        Initialize OpenAI-compatible client.

        Args:
            model: Model name
            api_key: API key (defaults to OPENAI_API_KEY env var, use "EMPTY" for local vLLM)
            base_url: API endpoint (defaults to OPENAI_BASE_URL env var)
            generation_config: Custom generation config (uses preset from GENERATION_CONFIG_PRESETS if None and model matches)
            retry_times: Tuple of wait times (seconds) between retries
            timeout: Request timeout in seconds (default 600s). Pass None for no timeout.
            trust_env: Whether the HTTP transport reads proxy variables.
            runtime_timeout_retries: Number of explicit retries for request timeouts
        """
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.retry_times = retry_times
        if not isinstance(runtime_timeout_retries, int) or runtime_timeout_retries < 0:
            raise ValueError("runtime_timeout_retries must be a non-negative integer")
        self.runtime_timeout_retries = runtime_timeout_retries
        self.timeout = timeout
        self._trust_env = trust_env
        self._runtime_instance_id = ""
        self._runtime_protocol_sha256 = ""
        self._runtime_request_index = 0
        
        if not self.api_key:
            raise ValueError(
                "API key required. Set OPENAI_API_KEY environment variable, "
                "pass api_key parameter, or use api_key='EMPTY' for local vLLM."
            )
        
        # Get generation config from presets or use custom
        if generation_config is not None:
            self.generation_config = generation_config
        else:
            self.generation_config = GENERATION_CONFIG_PRESETS.get(model, {})
        
        from openai import OpenAI
        
        # Keep retries in our runner-level logs instead of hidden SDK retries.
        client_kwargs = {
            "api_key": self.api_key,
            "timeout": timeout,
            "max_retries": OPENAI_SDK_MAX_RETRIES,
        }
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        if not trust_env:
            import httpx

            client_kwargs["http_client"] = httpx.Client(trust_env=False)
        
        self._client = OpenAI(**client_kwargs)

    def set_runtime_context(self, *, instance_id: str, protocol_sha256: str) -> None:
        self._runtime_instance_id = str(instance_id)
        self._runtime_protocol_sha256 = str(protocol_sha256)
        self._runtime_request_index = 0
    
    def _parse_response(self, response) -> tuple[str, str]:
        """
        Parse response, extracting reasoning content if present.
        
        Returns:
            Tuple of (reply, reasoning_content)
        """
        choice = response.choices[0]
        message = choice.message
        reply = message.content or ""
        reasoning_content = getattr(message, "reasoning_content", "") or ""
        
        # Handle models that embed thinking in the response with </think> tags
        if "</think>" in reply:
            parts = reply.split("</think>")
            reasoning_content = parts[0].replace("<think>", "").strip()
            reply = parts[1].strip() if len(parts) > 1 else ""

        if getattr(choice, "finish_reason", None) == "length":
            raise RequestCompletionLengthExceeded(
                "completion reached max_tokens before a complete response",
                partial_content=reply,
                reasoning_content=reasoning_content,
            )
        
        return reply, reasoning_content
    
    def _send_request_with_retry(self, messages: list[dict], config: dict):
        """Send request with retry logic."""
        self._runtime_request_index += 1
        request_index = self._runtime_request_index
        request_sha256 = hashlib.sha256(
            json.dumps(
                {"messages": messages, "config": config},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        ordinary_retry_index = 0
        timeout_retry_index = 0
        while True:
            try:
                return self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    **config,
                )
            except Exception as e:
                if _is_context_length_bad_request(e):
                    _raise_terminal_request_error(e)
                if _is_runtime_timeout(e):
                    if timeout_retry_index >= self.runtime_timeout_retries:
                        record_runtime_event(
                            "llm_request_timeout_exhausted",
                            model=self.model,
                            endpoint=self.base_url or "",
                            instance_id=self._runtime_instance_id,
                            protocol_sha256=self._runtime_protocol_sha256,
                            request_index=request_index,
                            request_sha256=request_sha256,
                            timeout_seconds=self.timeout,
                            retries_used=timeout_retry_index,
                        )
                        _raise_terminal_request_error(e)
                    wait_time = self.retry_times[0] if self.retry_times else 0
                    timeout_retry_index += 1
                    record_runtime_event(
                        "llm_request_timeout_retry",
                        model=self.model,
                        endpoint=self.base_url or "",
                        instance_id=self._runtime_instance_id,
                        protocol_sha256=self._runtime_protocol_sha256,
                        request_index=request_index,
                        request_sha256=request_sha256,
                        timeout_seconds=self.timeout,
                        retry_number=timeout_retry_index,
                        retry_limit=self.runtime_timeout_retries,
                        wait_seconds=wait_time,
                    )
                    log.warning(
                        "Request timed out; explicit timeout retry #%s/%s after %s seconds.",
                        timeout_retry_index,
                        self.runtime_timeout_retries,
                        wait_time,
                    )
                    time.sleep(wait_time)
                    continue

                if ordinary_retry_index >= len(self.retry_times):
                    raise
                wait_time = self.retry_times[ordinary_retry_index]
                ordinary_retry_index += 1
                log.warning(
                    f"Request failed ({type(e).__name__}): {e}. "
                    f"Retry #{ordinary_retry_index}/{len(self.retry_times)} "
                    f"after {wait_time} seconds."
                )
                time.sleep(wait_time)

    async def _send_request_with_retry_async(
        self, messages: list[dict], config: dict
    ):
        """Send one request with native async I/O and the same retry policy."""

        from openai import AsyncOpenAI

        client_kwargs: dict[str, Any] = {
            "api_key": self.api_key,
            "timeout": self.timeout,
            "max_retries": OPENAI_SDK_MAX_RETRIES,
        }
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        if not self._trust_env:
            import httpx

            client_kwargs["http_client"] = httpx.AsyncClient(trust_env=False)

        self._runtime_request_index += 1
        request_index = self._runtime_request_index
        request_sha256 = hashlib.sha256(
            json.dumps(
                {"messages": messages, "config": config},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        ordinary_retry_index = 0
        timeout_retry_index = 0
        async with AsyncOpenAI(**client_kwargs) as client:
            while True:
                try:
                    return await client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        **config,
                    )
                except Exception as exc:
                    if _is_context_length_bad_request(exc):
                        _raise_terminal_request_error(exc)
                    if _is_runtime_timeout(exc):
                        if timeout_retry_index >= self.runtime_timeout_retries:
                            record_runtime_event(
                                "llm_request_timeout_exhausted",
                                model=self.model,
                                endpoint=self.base_url or "",
                                instance_id=self._runtime_instance_id,
                                protocol_sha256=self._runtime_protocol_sha256,
                                request_index=request_index,
                                request_sha256=request_sha256,
                                timeout_seconds=self.timeout,
                                retries_used=timeout_retry_index,
                            )
                            _raise_terminal_request_error(exc)
                        wait_time = self.retry_times[0] if self.retry_times else 0
                        timeout_retry_index += 1
                        record_runtime_event(
                            "llm_request_timeout_retry",
                            model=self.model,
                            endpoint=self.base_url or "",
                            instance_id=self._runtime_instance_id,
                            protocol_sha256=self._runtime_protocol_sha256,
                            request_index=request_index,
                            request_sha256=request_sha256,
                            timeout_seconds=self.timeout,
                            retry_number=timeout_retry_index,
                            retry_limit=self.runtime_timeout_retries,
                            wait_seconds=wait_time,
                        )
                        log.warning(
                            "Request timed out; explicit timeout retry #%s/%s "
                            "after %s seconds.",
                            timeout_retry_index,
                            self.runtime_timeout_retries,
                            wait_time,
                        )
                        await asyncio.sleep(wait_time)
                        continue

                    if ordinary_retry_index >= len(self.retry_times):
                        raise
                    wait_time = self.retry_times[ordinary_retry_index]
                    ordinary_retry_index += 1
                    log.warning(
                        "Request failed (%s): %s. Retry #%s/%s after %s seconds.",
                        type(exc).__name__,
                        exc,
                        ordinary_retry_index,
                        len(self.retry_times),
                        wait_time,
                    )
                    await asyncio.sleep(wait_time)
    
    def chat(
        self,
        messages: list[Message],
        settings: ModelSettings | None = None,
        return_reasoning: bool = False,
    ) -> str | tuple[str, str]:
        """
        Send messages and get a response.
        
        Args:
            messages: List of messages
            settings: Optional model settings (merged with generation_config)
            return_reasoning: If True, returns (reply, reasoning_content) tuple
            
        Returns:
            Response string, or tuple of (reply, reasoning_content) if return_reasoning=True
        """
        openai_messages = [{"role": m.role, "content": m.content} for m in messages]
        
        # Merge generation config with settings
        config = self.generation_config.copy()
        if settings:
            settings_dict = settings.to_dict()
            config.update(settings_dict)
        
        # Send request
        response = self._send_request_with_retry(openai_messages, config)
        _record_react_usage(
            client="openai",
            model=self.model,
            endpoint=self.base_url or "",
            cache_hit=False,
            response=response,
            request={
                "message_count": len(openai_messages),
                "temperature": config.get("temperature"),
                "max_tokens": config.get("max_tokens"),
            },
        )
        reply, reasoning_content = self._parse_response(response)
        
        return (reply, reasoning_content) if return_reasoning else reply
    
    async def chat_async(
        self,
        messages: list[Message],
        settings: ModelSettings | None = None,
        return_reasoning: bool = False,
    ) -> str | tuple[str, str]:
        """Send messages through native async HTTP."""

        openai_messages = [{"role": message.role, "content": message.content} for message in messages]
        config = self.generation_config.copy()
        if settings:
            config.update(settings.to_dict())
        response = await self._send_request_with_retry_async(openai_messages, config)
        _record_react_usage(
            client="openai",
            model=self.model,
            endpoint=self.base_url or "",
            cache_hit=False,
            response=response,
            request={
                "message_count": len(openai_messages),
                "temperature": config.get("temperature"),
                "max_tokens": config.get("max_tokens"),
            },
        )
        reply, reasoning_content = self._parse_response(response)
        return (reply, reasoning_content) if return_reasoning else reply
    
class MockLLMClient(LLMClient):
    """
    Mock LLM client for testing.
    
    Returns predefined responses or echoes input.
    """
    
    def __init__(self, responses: list[str] | None = None):
        self.responses = responses or []
        self._call_count = 0
        self.call_history: list[list[Message]] = []
    
    def chat(self, messages: list[Message], settings: ModelSettings | None = None) -> str:
        self.call_history.append(messages)
        
        if self._call_count < len(self.responses):
            response = self.responses[self._call_count]
            self._call_count += 1
            return response
        
        # Default: echo last user message
        for msg in reversed(messages):
            if msg.role == "user":
                return f"Echo: {msg.content}"
        return "No response configured"
    
    async def chat_async(self, messages: list[Message], settings: ModelSettings | None = None) -> str:
        return self.chat(messages, settings)
