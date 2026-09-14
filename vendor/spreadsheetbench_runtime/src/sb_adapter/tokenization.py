from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from functools import lru_cache
from typing import Any


@dataclass(frozen=True)
class TokenizedText:
    token_count: int
    truncated_text: str
    truncated: bool
    tokenizer_name: str
    strategy: str


class TokenizerUnavailable(RuntimeError):
    pass


class BaseTokenizer:
    name: str
    def count(self, text: str) -> int:
        raise NotImplementedError
    def truncate(self, text: str, max_tokens: int, *, strategy: str = "head") -> str:
        raise NotImplementedError


class RegexTokenizer(BaseTokenizer):
    """Test-only fallback tokenizer.

    It is intentionally named as a fallback so real Qwen runs can require a
    HuggingFace tokenizer and fail fast if it is unavailable.
    """
    name = "regex_fallback_test_only"
    _pattern = re.compile(r"\w+|[^\w\s]", re.UNICODE)
    def _tokens(self, text: str) -> list[str]:
        return self._pattern.findall(text)
    def count(self, text: str) -> int:
        return len(self._tokens(text))
    def truncate(self, text: str, max_tokens: int, *, strategy: str = "head") -> str:
        toks = self._tokens(text)
        if len(toks) <= max_tokens:
            return text
        if strategy != "head":
            raise ValueError(f"unsupported tokenizer fallback truncation strategy={strategy!r}")
        # Regex fallback cannot preserve exact original spacing. This is only
        # used in tests/mock mode.
        return " ".join(toks[:max_tokens])


class Utf8ByteUpperBoundTokenizer(BaseTokenizer):
    """Conservative, download-free input budget for byte-level tokenizers.

    Qwen's tokenizer cannot produce more ordinary text tokens than the UTF-8
    byte representation.  A small fixed margin covers service-added special
    tokens.  Counts are therefore upper bounds rather than exact token counts;
    truncation may be earlier than necessary but cannot split a Unicode code
    point or overflow the configured token window under that invariant.
    """

    def __init__(self, *, safety_margin: int = 16):
        if safety_margin < 0:
            raise ValueError("safety_margin must be non-negative")
        self.safety_margin = int(safety_margin)
        self.name = f"utf8_byte_upper_bound_margin_{self.safety_margin}"

    def count(self, text: str) -> int:
        return len(text.encode("utf-8")) + self.safety_margin

    @staticmethod
    def _head(text: str, byte_budget: int) -> str:
        used = 0
        chars: list[str] = []
        for char in text:
            width = len(char.encode("utf-8"))
            if used + width > byte_budget:
                break
            chars.append(char)
            used += width
        return "".join(chars)

    @staticmethod
    def _tail(text: str, byte_budget: int) -> str:
        used = 0
        chars: list[str] = []
        for char in reversed(text):
            width = len(char.encode("utf-8"))
            if used + width > byte_budget:
                break
            chars.append(char)
            used += width
        return "".join(reversed(chars))

    def truncate(self, text: str, max_tokens: int, *, strategy: str = "head") -> str:
        byte_budget = max(0, int(max_tokens) - self.safety_margin)
        if self.count(text) <= max_tokens:
            return text
        if strategy == "head":
            return self._head(text, byte_budget)
        if strategy == "tail":
            return self._tail(text, byte_budget)
        if strategy == "head_tail":
            head_budget = byte_budget // 2
            return self._head(text, head_budget) + self._tail(
                text,
                byte_budget - head_budget,
            )
        raise ValueError(f"unsupported truncation strategy={strategy!r}")


class HuggingFaceTokenizer(BaseTokenizer):
    def __init__(self, model_or_path: str):
        try:
            from transformers import AutoTokenizer  # type: ignore
        except Exception as exc:
            raise TokenizerUnavailable("transformers is required for real tokenizer-based truncation") from exc
        self.tokenizer = AutoTokenizer.from_pretrained(model_or_path, trust_remote_code=True)
        self.name = str(model_or_path)
    def count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))
    def truncate(self, text: str, max_tokens: int, *, strategy: str = "head") -> str:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) <= max_tokens:
            return text
        if strategy == "head":
            ids = ids[:max_tokens]
        elif strategy == "tail":
            ids = ids[-max_tokens:]
        elif strategy == "head_tail":
            head = max_tokens // 2
            tail = max_tokens - head
            ids = ids[:head] + ids[-tail:]
        else:
            raise ValueError(f"unsupported truncation strategy={strategy!r}")
        return self.tokenizer.decode(ids, skip_special_tokens=True)


_TOKENIZER_INIT_LOCK = threading.Lock()


@lru_cache(maxsize=8)
def _get_tokenizer_cached(
    model_or_path: str | None,
    *,
    allow_regex_fallback: bool,
    fallback_mode: str | None,
) -> BaseTokenizer:
    if fallback_mode not in {None, "regex", "utf8_bytes"}:
        raise ValueError(f"unsupported tokenizer fallback mode={fallback_mode!r}")
    if model_or_path:
        try:
            return HuggingFaceTokenizer(model_or_path)
        except TokenizerUnavailable:
            if not allow_regex_fallback:
                raise
        except Exception as exc:
            if not allow_regex_fallback:
                raise TokenizerUnavailable(f"failed to load tokenizer {model_or_path!r}: {exc}") from exc
    if fallback_mode == "utf8_bytes":
        return Utf8ByteUpperBoundTokenizer()
    if fallback_mode == "regex" or allow_regex_fallback:
        return RegexTokenizer()
    raise TokenizerUnavailable("No tokenizer configured and tokenizer fallback is disabled")


def get_tokenizer(
    model_or_path: str | None,
    *,
    allow_regex_fallback: bool,
    fallback_mode: str | None = None,
) -> BaseTokenizer:
    with _TOKENIZER_INIT_LOCK:
        return _get_tokenizer_cached(
            model_or_path,
            allow_regex_fallback=allow_regex_fallback,
            fallback_mode=fallback_mode,
        )


def truncate_with_tokenizer(
    text: str,
    *,
    tokenizer_model: str | None,
    max_tokens: int,
    strategy: str = "head",
    allow_regex_fallback: bool = False,
    fallback_mode: str | None = None,
) -> TokenizedText:
    tok = get_tokenizer(
        tokenizer_model,
        allow_regex_fallback=allow_regex_fallback,
        fallback_mode=fallback_mode,
    )
    count = tok.count(text)
    if count <= max_tokens:
        return TokenizedText(count, text, False, tok.name, strategy)
    truncated = tok.truncate(text, max_tokens, strategy=strategy)
    return TokenizedText(count, truncated, True, tok.name, strategy)
