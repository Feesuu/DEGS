from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from spreadsheet_agent.agents.cli_only_agent import CLIOnlyAgent
from degs.core import canonical_json_bytes
from degs.eir_bundle import EIR_GUIDANCE_ROW_FORMAT, VerifiedEIRGuidanceBundle
from degs.soft_hard_benchmark import SoftHardExperienceAgent
from degs.soft_hard_bundle import FORMAT, SoftHardExperienceProvider, _parser
from degs.soft_hard_dataset import TESTCASE_COUNT


def _provider(tmp_path: Path) -> SoftHardExperienceProvider:
    rows = []
    for index in range(TESTCASE_COUNT):
        experience = f"Use operation {index}."
        audit = {"status": "OK"}
        rows.append(
            {
                "format": EIR_GUIDANCE_ROW_FORMAT,
                "instance_id": f"case-{index}",
                "experience": experience,
                "dataset_index": index,
                "snapshot_id": "snapshot-test",
                "retrieval": audit,
                "expectations": [],
                "status": "COMPLETE",
                "error": None,
            }
        )
    payload = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    body = {
        "format": FORMAT,
        "snapshot_id": "snapshot-test",
        "experience_file": "experience.jsonl",
        "experience_sha256": hashlib.sha256(payload).hexdigest(),
        "row_count": TESTCASE_COUNT,
    }
    manifest = {
        **body,
        "self_sha256": hashlib.sha256(canonical_json_bytes(body)).hexdigest(),
    }
    tmp_path.mkdir()
    (tmp_path / "experience.jsonl").write_bytes(payload)
    (tmp_path / "bundle_manifest.json").write_bytes(canonical_json_bytes(manifest))
    verified = VerifiedEIRGuidanceBundle(tmp_path, manifest)
    return SoftHardExperienceProvider(verified)


def test_soft_hard_provider_uses_case_identity(tmp_path: Path) -> None:
    provider = _provider(tmp_path / "bundle")
    assert provider.for_instance("case-0").experience == "Use operation 0."
    assert provider.identity()["row_count"] == TESTCASE_COUNT


def test_agent_preserves_runtime_id(tmp_path: Path, monkeypatch) -> None:
    provider = _provider(tmp_path / "bundle")
    agent = SoftHardExperienceAgent(client=object(), experience_provider=provider)
    monkeypatch.setattr(
        CLIOnlyAgent,
        "run",
        lambda self, context: {
            "experience": self._experience_content,
            "runtime_id": context.instance_id,
        },
    )
    result = agent.run(
        SimpleNamespace(instance_id="task-0__case-1", retrieval_id="case-0")
    )
    assert result == {
        "experience": "Use operation 0.",
        "runtime_id": "task-0__case-1",
    }


def test_soft_hard_cli_uses_one_online_bundle_interface() -> None:
    options = {
        option
        for action in _parser()._subparsers._group_actions[0].choices["build"]._actions
        for option in action.option_strings
    }
    assert "--accepted-bundle-dir" not in options
