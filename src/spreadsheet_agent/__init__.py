"""Lazy import shim for the frozen SpreadsheetBench runtime."""

from __future__ import annotations

from typing import Any

__all__ = [
    "CLIOnlyAgent",
    "SpreadsheetBenchRunner",
]


def __getattr__(name: str) -> Any:
    if name == "SpreadsheetBenchRunner":
        from .runner import SpreadsheetBenchRunner

        return SpreadsheetBenchRunner
    if name == "CLIOnlyAgent":
        from .agents.cli_only_agent import CLIOnlyAgent

        return CLIOnlyAgent
    raise AttributeError(name)
