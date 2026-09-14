"""Lazy package shim; concrete frozen agents are imported explicitly."""

from __future__ import annotations

from typing import Any

__all__ = [
    "BaseSpreadsheetAgent",
    "CLIOnlyAgent",
]


def __getattr__(name: str) -> Any:
    if name == "BaseSpreadsheetAgent":
        from .base import BaseSpreadsheetAgent

        return BaseSpreadsheetAgent
    if name == "CLIOnlyAgent":
        from .cli_only_agent import CLIOnlyAgent

        return CLIOnlyAgent
    raise AttributeError(name)
