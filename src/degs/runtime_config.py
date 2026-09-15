from __future__ import annotations

import os


def worker_count(name: str, default: int) -> int:
    """Read a positive runtime concurrency setting without making it method identity."""

    value = int(os.getenv(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


__all__ = ["worker_count"]
