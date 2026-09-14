#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.resources
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from react_agent.models import OpenAIClient
from sb_adapter.transport import validate_service_url

from .core import canonical_json_bytes
from .section_graph import ExperienceEdge, ExperienceNode
from .validated_repair import (
    JsonObjectLLM,
    OpenAIJsonObjectLLM,
    REPAIR_SOURCE_MODEL,
    REPAIR_SOURCE_TIMEOUT_SECONDS,
    PRODUCER_RUNTIME_TIMEOUT_RETRIES,
    PRODUCER_TRANSPORT_RETRY_WAITS,
    _parse_llm_experience_graph,
    _source_generation_config,
    _strict_json_object,
    _write_json_output,
    render_successful_replay,
    repair_response_schema,
)


SUCCESS_EXTRACTION_KIND = "extract_successful_trajectory_experience_v6"
SUCCESS_EXTRACTION_FORMAT = "degs_successful_trajectory_extraction_v6"
SUCCESS_SOURCE_PROTOCOL_FORMAT = "degs_successful_source_protocol_v7"
SUCCESS_PROMPT_RESOURCE = "SUCCESSFUL_TRAJECTORY_EXPERIENCE_PROMPT_V4.txt"


def _prompt_text() -> str:
    return (
        importlib.resources.files("degs")
        .joinpath("resources", SUCCESS_PROMPT_RESOURCE)
        .read_text(encoding="utf-8")
        .strip()
    )


SUCCESS_SYSTEM_PROMPT = _prompt_text()
SUCCESS_PROMPT_SHA256 = hashlib.sha256(
    SUCCESS_SYSTEM_PROMPT.encode("utf-8")
).hexdigest()


def success_response_schema() -> dict[str, Any]:
    return repair_response_schema()


@dataclass(frozen=True)
class SuccessfulTrajectoryExtraction:
    experience_nodes: tuple[ExperienceNode, ...]
    edges: tuple[ExperienceEdge, ...]
    discarded_edge_reasons: tuple[str, ...]
    source_protocol: dict[str, Any]
    source_protocol_sha256: str
    request_payload_sha256: str
    response_schema_sha256: str

    def experience_node_dicts(self) -> list[dict[str, Any]]:
        return [row.to_dict() for row in self.experience_nodes]

    def edge_dicts(self) -> list[dict[str, int]]:
        return [row.to_dict() for row in self.edges]


class SuccessfulTrajectoryExperienceExtractor:
    def __init__(self, llm: JsonObjectLLM) -> None:
        self.llm = llm

    def extract(
        self,
        successful_trajectory: Mapping[str, Any],
    ) -> SuccessfulTrajectoryExtraction:
        rendered = render_successful_replay(successful_trajectory)
        payload = {"successful_trajectory": rendered.payload}
        response_schema = success_response_schema()
        source_protocol = dict(self.llm.protocol_identity)
        source_protocol_sha256 = hashlib.sha256(
            canonical_json_bytes(source_protocol)
        ).hexdigest()
        request_payload_sha256 = hashlib.sha256(
            canonical_json_bytes(payload)
        ).hexdigest()
        response_schema_sha256 = hashlib.sha256(
            canonical_json_bytes(response_schema)
        ).hexdigest()
        trajectory_id = successful_trajectory.get("trajectory_id")
        if not isinstance(trajectory_id, str) or not trajectory_id:
            raise ValueError("successful trajectory identity differs")
        raw = self.llm.complete_json(
            kind=SUCCESS_EXTRACTION_KIND,
            request_id=trajectory_id,
            system_prompt=SUCCESS_SYSTEM_PROMPT,
            payload=payload,
            response_schema=response_schema,
        )
        nodes, edges, discarded_edge_reasons = _parse_llm_experience_graph(raw)
        return SuccessfulTrajectoryExtraction(
            nodes,
            edges,
            discarded_edge_reasons,
            source_protocol,
            source_protocol_sha256,
            request_payload_sha256,
            response_schema_sha256,
        )

    async def extract_async(
        self,
        successful_trajectory: Mapping[str, Any],
    ) -> SuccessfulTrajectoryExtraction:
        rendered = render_successful_replay(successful_trajectory)
        payload = {"successful_trajectory": rendered.payload}
        response_schema = success_response_schema()
        source_protocol = dict(self.llm.protocol_identity)
        source_protocol_sha256 = hashlib.sha256(
            canonical_json_bytes(source_protocol)
        ).hexdigest()
        request_payload_sha256 = hashlib.sha256(
            canonical_json_bytes(payload)
        ).hexdigest()
        response_schema_sha256 = hashlib.sha256(
            canonical_json_bytes(response_schema)
        ).hexdigest()
        trajectory_id = successful_trajectory.get("trajectory_id")
        if not isinstance(trajectory_id, str) or not trajectory_id:
            raise ValueError("successful trajectory identity differs")
        raw = await self.llm.complete_json_async(
            kind=SUCCESS_EXTRACTION_KIND,
            request_id=trajectory_id,
            system_prompt=SUCCESS_SYSTEM_PROMPT,
            payload=payload,
            response_schema=response_schema,
        )
        nodes, edges, discarded_edge_reasons = _parse_llm_experience_graph(raw)
        return SuccessfulTrajectoryExtraction(
            nodes,
            edges,
            discarded_edge_reasons,
            source_protocol,
            source_protocol_sha256,
            request_payload_sha256,
            response_schema_sha256,
        )


def openai_success_llm(
    client: OpenAIClient,
    *,
    raw_response_output: Path | None = None,
) -> OpenAIJsonObjectLLM:
    return OpenAIJsonObjectLLM(
        client,
        raw_response_output=raw_response_output,
        request_kind=SUCCESS_EXTRACTION_KIND,
        source_protocol_format=SUCCESS_SOURCE_PROTOCOL_FORMAT,
        prompt_sha256=SUCCESS_PROMPT_SHA256,
        response_schema_name="degs_successful_trajectory_experience_v5",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract reusable experience nodes from one successful trajectory."
    )
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key-env", default="DEGS_API_KEY")
    parser.add_argument("--raw-response-output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = args.output.expanduser().absolute()
    raw_output = (
        args.raw_response_output.expanduser().absolute()
        if args.raw_response_output is not None
        else None
    )
    if output.exists():
        raise FileExistsError("successful trajectory extraction output must be fresh")
    if raw_output is not None and (
        raw_output == output or raw_output.exists()
    ):
        raise FileExistsError("raw and final source outputs must be distinct and fresh")
    validate_service_url(args.base_url)
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise ValueError(f"missing generation API key: set {args.api_key_env}")
    trajectory = _strict_json_object(args.trajectory.read_text(encoding="utf-8"))
    client = OpenAIClient(
        model=REPAIR_SOURCE_MODEL,
        api_key=api_key,
        base_url=args.base_url,
        generation_config=_source_generation_config(),
        retry_times=PRODUCER_TRANSPORT_RETRY_WAITS,
        runtime_timeout_retries=PRODUCER_RUNTIME_TIMEOUT_RETRIES,
        timeout=REPAIR_SOURCE_TIMEOUT_SECONDS,
        trust_env=False,
    )
    result = SuccessfulTrajectoryExperienceExtractor(
        openai_success_llm(client, raw_response_output=raw_output)
    ).extract(trajectory)
    body = {
        "format": SUCCESS_EXTRACTION_FORMAT,
        "task_id": trajectory.get("task_id"),
        "trajectory_id": trajectory.get("trajectory_id"),
        "prompt_sha256": SUCCESS_PROMPT_SHA256,
        "source_protocol": result.source_protocol,
        "source_protocol_sha256": result.source_protocol_sha256,
        "request_payload_sha256": result.request_payload_sha256,
        "response_schema_sha256": result.response_schema_sha256,
        "experience_nodes": result.experience_node_dicts(),
        "edges": result.edge_dicts(),
        "discarded_edge_reasons": list(result.discarded_edge_reasons),
    }
    _write_json_output(output, body)
    print(
        json.dumps(
            {
                "format": SUCCESS_EXTRACTION_FORMAT,
                "experience_node_count": len(result.experience_nodes),
                "edge_count": len(result.edges),
                "discarded_edge_count": len(result.discarded_edge_reasons),
                "prompt_sha256": SUCCESS_PROMPT_SHA256,
                "source_protocol_sha256": result.source_protocol_sha256,
                "output": str(output),
            },
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "SUCCESS_EXTRACTION_KIND",
    "SUCCESS_PROMPT_SHA256",
    "SuccessfulTrajectoryExtraction",
    "SuccessfulTrajectoryExperienceExtractor",
    "openai_success_llm",
    "success_response_schema",
]


if __name__ == "__main__":
    raise SystemExit(main())
