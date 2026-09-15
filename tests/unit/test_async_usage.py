from __future__ import annotations

import asyncio
from types import SimpleNamespace

import openai

from react_agent import models
from react_agent.models import Message, OpenAIClient


class _AsyncOpenAI:
    def __init__(self, **_kwargs) -> None:
        self.chat = SimpleNamespace(completions=self)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def create(self, *, messages, **_kwargs):
        content = messages[0]["content"]
        await asyncio.sleep(0.02 if content == "slow" else 0)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=content, reasoning_content=""),
                    finish_reason="stop",
                )
            ],
            usage=None,
        )


def test_concurrent_async_requests_record_distinct_request_indices(monkeypatch) -> None:
    monkeypatch.setattr(openai, "OpenAI", lambda **_kwargs: object())
    monkeypatch.setattr(openai, "AsyncOpenAI", _AsyncOpenAI)
    records = []
    monkeypatch.setattr(
        models,
        "_record_react_usage",
        lambda **kwargs: records.append(kwargs["request"]),
    )
    client = OpenAIClient(model="test", api_key="test", generation_config={})

    async def exercise() -> None:
        replies = await asyncio.gather(
            client.chat_async([Message("user", "slow")]),
            client.chat_async([Message("user", "fast")]),
        )
        assert replies == ["slow", "fast"]

    asyncio.run(exercise())
    assert sorted(row["request_index"] for row in records) == [1, 2]
