from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from react_agent import Tool, ToolParameter


PROCESS_CONTRACT_POLICY = "SELECTED_FINE_MODULE_PRE_MUTATION_CONTRACT_V1"
PROCESS_CONTRACT_TOOL_NAME = "submit_process_contract"

_FAMILY_RE = re.compile(r"^[A-Z][A-Z0-9_]{1,39}$")
_STEP_FIELDS = {
    "operator_family",
    "source",
    "criterion",
    "target",
    "output_topology",
    "completion_check",
}
_TOPOLOGIES = {
    "NONE",
    "SCALAR",
    "ROW_VECTOR",
    "COLUMN_VECTOR",
    "TABLE",
    "WORKSHEET_STRUCTURE",
    "FORMAT_ONLY",
    "MIXED",
}


def _clean_text(value: Any, *, label: str, maximum: int = 300) -> str:
    if type(value) is not str or not value.strip() or len(value.strip()) > maximum:
        raise ValueError(f"invalid {label}")
    return value.strip()


def load_operator_vocabulary(path: str | Path) -> tuple[dict[str, str], ...]:
    unresolved = Path(path).expanduser()
    if unresolved.is_symlink() or not unresolved.is_file():
        raise ValueError("operator vocabulary must be a regular non-symlink file")
    value = json.loads(unresolved.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not 6 <= len(value) <= 16:
        raise ValueError("operator vocabulary family count differs")
    parsed: list[dict[str, str]] = []
    names: set[str] = set()
    for row in value:
        if not isinstance(row, dict) or set(row) != {"name", "definition"}:
            raise ValueError("operator vocabulary family fields differ")
        name = _clean_text(row.get("name"), label="operator family name", maximum=40)
        if not _FAMILY_RE.fullmatch(name) or name in names:
            raise ValueError("operator vocabulary family name differs")
        names.add(name)
        parsed.append(
            {
                "name": name,
                "definition": _clean_text(
                    row.get("definition"), label="operator family definition"
                ),
            }
        )
    return tuple(parsed)


def validate_process_contract(
    value: Mapping[str, Any],
    *,
    operator_names: Sequence[str],
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"steps", "global_postcondition"}:
        raise ValueError("process contract fields differ")
    steps = value.get("steps")
    if not isinstance(steps, list) or not 1 <= len(steps) <= 10:
        raise ValueError("process contract step count differs")
    allowed_operators = set(operator_names)
    parsed_steps: list[dict[str, str]] = []
    for row in steps:
        if not isinstance(row, dict) or set(row) != _STEP_FIELDS:
            raise ValueError("process contract step fields differ")
        family = _clean_text(
            row.get("operator_family"), label="operator family", maximum=40
        )
        if family not in allowed_operators:
            raise ValueError("process contract uses an unknown operator")
        topology = row.get("output_topology")
        if type(topology) is not str or topology not in _TOPOLOGIES:
            raise ValueError("process contract output topology differs")
        parsed_steps.append(
            {
                "operator_family": family,
                "source": _clean_text(row.get("source"), label="source"),
                "criterion": _clean_text(row.get("criterion"), label="criterion"),
                "target": _clean_text(row.get("target"), label="target"),
                "output_topology": topology,
                "completion_check": _clean_text(
                    row.get("completion_check"), label="completion check"
                ),
            }
        )
    return {
        "steps": parsed_steps,
        "global_postcondition": _clean_text(
            value.get("global_postcondition"), label="global postcondition"
        ),
    }


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class StructuredProcessContractGate:
    def __init__(
        self,
        *,
        working_dir: str | Path,
        operator_vocabulary: Sequence[Mapping[str, str]],
        instance_id: str,
        protocol_sha256: str,
        contract_log_path: str | Path,
    ) -> None:
        self.working_dir = Path(working_dir)
        self.input_path = self.working_dir / "input.xlsx"
        self.output_path = self.working_dir / "output.xlsx"
        self.operator_vocabulary = tuple(dict(row) for row in operator_vocabulary)
        self.operator_names = tuple(row["name"] for row in self.operator_vocabulary)
        self.instance_id = str(instance_id)
        self.protocol_sha256 = str(protocol_sha256)
        self.contract_log_path = Path(contract_log_path)
        self.inspection_observed = False
        self.accepted_contract: dict[str, Any] | None = None

    def _record(self, event: str, **details: Any) -> None:
        row = {
            "component": "structured_process_contract",
            "event": event,
            "instance_id": self.instance_id,
            "protocol_sha256": self.protocol_sha256,
            **details,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        payload = (
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
        with self.contract_log_path.open("ab") as handle:
            handle.write(payload)

    @staticmethod
    def _command_sha256(command: str) -> str:
        return hashlib.sha256(command.encode("utf-8")).hexdigest()

    def _remove_output(self) -> None:
        if self.output_path.is_symlink() or self.output_path.is_file():
            self.output_path.unlink()
        elif self.output_path.is_dir():
            shutil.rmtree(self.output_path)

    def create_contract_tool(self) -> Tool:
        vocabulary = "\n".join(
            f"- {row['name']}: {row['definition']}" for row in self.operator_vocabulary
        )
        description = f"""Submit and freeze the ordered Process execution contract before creating output.xlsx.
First use bash to inspect input.xlsx. When a retrieved fine module is present, adapt its ## Process order and interfaces to the current workbook; never copy incompatible literals. The contract must be one JSON object with exactly `steps` and `global_postcondition`. `steps` contains 1-10 objects with exactly `operator_family`, `source`, `criterion`, `target`, `output_topology`, and `completion_check`. Every value is a non-empty string. `output_topology` must be one of {sorted(_TOPOLOGIES)}. Use only these train-derived operator families:
{vocabulary}"""

        def submit(contract: dict) -> str:
            if self.accepted_contract is not None:
                self._record(
                    "process_contract_submission_rejected",
                    reason="contract_already_frozen",
                )
                return (
                    "PROCESS_CONTRACT_REJECTED: the accepted contract is already frozen. "
                    "Execute it and verify the saved output."
                )
            if not self.inspection_observed:
                self._record(
                    "process_contract_submission_rejected",
                    reason="inspection_required",
                )
                return (
                    "PROCESS_CONTRACT_REJECTED: inspect input.xlsx with bash first, then "
                    "submit a workbook-grounded contract."
                )
            try:
                parsed = validate_process_contract(
                    contract,
                    operator_names=self.operator_names,
                )
            except ValueError as exc:
                self._record(
                    "process_contract_submission_rejected",
                    reason="invalid_contract",
                )
                return f"PROCESS_CONTRACT_REJECTED: {exc}."
            self.accepted_contract = parsed
            self._record(
                "process_contract_accepted",
                contract_sha256=_sha256_json(parsed),
                step_count=len(parsed["steps"]),
                operator_families=[
                    row["operator_family"] for row in parsed["steps"]
                ],
                contract=parsed,
            )
            return (
                "PROCESS_CONTRACT_ACCEPTED: the contract is frozen. Execute the steps in "
                "order, create output.xlsx, and complete the declared checks."
            )

        return Tool(
            name=PROCESS_CONTRACT_TOOL_NAME,
            description=description,
            func=submit,
            parameters=[
                ToolParameter(
                    name="contract",
                    type="object",
                    description="The exact ordered Process contract object described above.",
                )
            ],
        )

    def wrap_bash(self, bash_tool: Tool) -> Tool:
        def execute(command: str) -> str:
            command_text = str(command)
            command_sha = self._command_sha256(command_text)
            if self.accepted_contract is None and "output.xlsx" in command_text:
                self._record(
                    "process_contract_output_blocked",
                    stage="pre_execution",
                    command_sha256=command_sha,
                )
                return (
                    "PROCESS_CONTRACT_REQUIRED: this command was not executed. Inspect "
                    "input.xlsx and call submit_process_contract before creating output.xlsx."
                )

            input_before = (
                hashlib.sha256(self.input_path.read_bytes()).hexdigest()
                if self.input_path.is_file() and not self.input_path.is_symlink()
                else None
            )
            observation = bash_tool.execute(command=command_text)
            input_after = (
                hashlib.sha256(self.input_path.read_bytes()).hexdigest()
                if self.input_path.is_file() and not self.input_path.is_symlink()
                else None
            )
            if self.accepted_contract is None and (
                self.output_path.exists() or self.output_path.is_symlink()
            ):
                self._remove_output()
                self._record(
                    "process_contract_output_blocked",
                    stage="post_execution",
                    command_sha256=command_sha,
                )
                return (
                    f"{observation}\n\nPROCESS_CONTRACT_REQUIRED: output.xlsx was removed. "
                    "Inspect input.xlsx and call submit_process_contract before creating it."
                )
            if (
                not self.inspection_observed
                and "input.xlsx" in command_text
                and input_before is not None
                and input_before == input_after
            ):
                self.inspection_observed = True
                self._record(
                    "process_contract_inspection_observed",
                    command_sha256=command_sha,
                    input_sha256=input_after,
                )
            return observation

        return Tool(
            name=bash_tool.name,
            description=bash_tool.description,
            func=execute,
            parameters=list(bash_tool.parameters),
        )


__all__ = [
    "PROCESS_CONTRACT_POLICY",
    "PROCESS_CONTRACT_TOOL_NAME",
    "StructuredProcessContractGate",
    "load_operator_vocabulary",
    "validate_process_contract",
]
