"""Report source-replay context overflow as a machine-readable marker."""

from __future__ import annotations

import os
from pathlib import Path
import runpy
import sys


MARKER_ENV = "DEGS_CONTEXT_OVERFLOW_MARKER"


def _mark_context_overflow() -> None:
    marker = os.environ.get(MARKER_ENV)
    if not marker:
        return
    path = Path(marker)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return
    try:
        os.write(descriptor, b"CONTEXT_LENGTH_EXCEEDED\n")
    finally:
        os.close(descriptor)


def _install_probe() -> None:
    import react_agent.agent as agent
    import react_agent.models as models

    original = models.RequestContextLengthExceeded

    class ProbedRequestContextLengthExceeded(original):
        def __init__(self, *args, **kwargs):
            _mark_context_overflow()
            super().__init__(*args, **kwargs)

    models.RequestContextLengthExceeded = ProbedRequestContextLengthExceeded

    def full_observation(text: str, **_kwargs) -> str:
        return text

    agent.truncate_observation = full_observation


def main() -> int:
    if len(sys.argv) < 2:
        raise ValueError("replay probe requires a target module")
    module = sys.argv[1]
    sys.argv = [module, *sys.argv[2:]]
    _install_probe()
    runpy.run_module(module, run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
