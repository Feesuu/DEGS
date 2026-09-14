#!/usr/bin/env python3
"""Verify that a campaign's generation and embedding models are reachable."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Sequence

import httpx

from sb_adapter.transport import validate_service_url


def _models(*, base_url: str, api_key_env: str) -> tuple[str, ...]:
    validate_service_url(base_url)
    api_key = os.environ.get(api_key_env, "")
    if not api_key:
        raise RuntimeError(f"missing API key: set {api_key_env}")
    response = httpx.get(
        f"{base_url.rstrip('/')}/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30.0,
        trust_env=False,
    )
    response.raise_for_status()
    payload: Any = response.json()
    rows = payload.get("data") if type(payload) is dict else None
    if type(rows) is not list:
        raise ValueError("model service response differs")
    identifiers = tuple(
        sorted(
            str(row.get("id"))
            for row in rows
            if type(row) is dict and type(row.get("id")) is str
        )
    )
    if not identifiers:
        raise ValueError("model service returned no model identities")
    return identifiers


def verify_services(
    *,
    generation_base_url: str,
    generation_model: str,
    generation_api_key_env: str,
    embedding_base_url: str,
    embedding_model: str,
    embedding_api_key_env: str,
) -> dict[str, Any]:
    generation_models = _models(
        base_url=generation_base_url,
        api_key_env=generation_api_key_env,
    )
    embedding_models = _models(
        base_url=embedding_base_url,
        api_key_env=embedding_api_key_env,
    )
    if generation_model not in generation_models:
        raise ValueError("required generation model is unavailable")
    if embedding_model not in embedding_models:
        raise ValueError("required embedding model is unavailable")
    return {
        "format": "degs_service_preflight_v1",
        "generation": {
            "base_url": generation_base_url.rstrip("/"),
            "model": generation_model,
        },
        "embedding": {
            "base_url": embedding_base_url.rstrip("/"),
            "model": embedding_model,
        },
        "status": "PASS",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-base-url", required=True)
    parser.add_argument("--generation-model", required=True)
    parser.add_argument(
        "--generation-api-key-env", default="DEGS_API_KEY"
    )
    parser.add_argument("--embedding-base-url", required=True)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument(
        "--embedding-api-key-env", default="DEGS_EMBEDDING_API_KEY"
    )
    args = parser.parse_args(argv)
    result = verify_services(
        generation_base_url=args.generation_base_url,
        generation_model=args.generation_model,
        generation_api_key_env=args.generation_api_key_env,
        embedding_base_url=args.embedding_base_url,
        embedding_model=args.embedding_model,
        embedding_api_key_env=args.embedding_api_key_env,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
