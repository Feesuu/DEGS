from __future__ import annotations

import os
import sys
from typing import Sequence

from .contract import MODEL_BY_PROFILE


def _profile(argv: Sequence[str]) -> str:
    for index, value in enumerate(argv):
        if value == "--profile" and index + 1 < len(argv):
            return argv[index + 1]
        if value.startswith("--profile="):
            return value.split("=", 1)[1]
    raise ValueError("Skill2Bench requires --profile 9b or --profile 27b")


def _option(argv: Sequence[str], name: str) -> str | None:
    for index, value in enumerate(argv):
        if value == name and index + 1 < len(argv):
            return argv[index + 1]
        if value.startswith(name + "="):
            return value.split("=", 1)[1]
    return None


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    profile = (
        "9b"
        if any(value in {"-h", "--help"} for value in arguments)
        and not any(
            value == "--profile" or value.startswith("--profile=")
            for value in arguments
        )
        else _profile(arguments)
    )
    try:
        model = MODEL_BY_PROFILE[profile]
    except KeyError as exc:
        raise ValueError("Skill2Bench model profile differs") from exc
    # Model-dependent shared modules read the selected model during import.
    # Select it before importing the campaign, not inside the running campaign.
    os.environ["DEGS_MODEL"] = model
    for option, environment in (
        ("--agent-workers", "DEGS_AGENT_WORKERS"),
        ("--producer-workers", "DEGS_PRODUCER_WORKERS"),
    ):
        value = _option(arguments, option)
        if value is not None:
            os.environ[environment] = value
    from .campaign import main as campaign_main

    return campaign_main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
