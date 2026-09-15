"""Lazy package shim over the repository-pinned SpreadsheetBench runtime."""

from pathlib import Path


_VENDOR_PACKAGE = (
    Path(__file__).resolve().parents[2]
    / "vendor"
    / "spreadsheetbench_runtime"
    / "src"
    / "sb_adapter"
)
if _VENDOR_PACKAGE.is_dir() and str(_VENDOR_PACKAGE) not in __path__:
    __path__.append(str(_VENDOR_PACKAGE))

__all__: list[str] = []
