"""Bash-only spreadsheet agent used by the adapter."""

import os
from pathlib import Path

from react_agent import Tool

from .base import AgentContext, BaseSpreadsheetAgent
from ..system_prompts import load_full_system_prompt
from ..tools import create_bash_tool
from ..output_feedback import wrap_bash_with_output_feedback
from ..structured_process_contract import (
    StructuredProcessContractGate,
    load_operator_vocabulary,
)


class CLIOnlyAgent(BaseSpreadsheetAgent):
    def __init__(
        self,
        client,
        max_turns: int = 20,
        verbose: bool = True,
        timeout: int = 120,
        sandbox_mode: str = "required",
        log_dir: str | None = None,
        stagnation_repeat_limit: int = 0,
        libreoffice_output_feedback: bool = False,
        structured_process_contract: bool = False,
        process_contract_vocabulary_file: str | Path | None = None,
        process_contract_log: str | Path | None = None,
    ):
        super().__init__(
            client,
            max_turns,
            verbose,
            log_dir,
            stagnation_repeat_limit,
        )
        self.timeout = timeout
        self.sandbox_mode = sandbox_mode
        self.libreoffice_output_feedback = libreoffice_output_feedback
        self._output_feedback_context: AgentContext | None = None
        self.structured_process_contract = structured_process_contract
        self._structured_process_contract_active = structured_process_contract
        self.process_contract_log = (
            Path(process_contract_log) if process_contract_log is not None else None
        )
        if structured_process_contract:
            if process_contract_vocabulary_file is None or self.process_contract_log is None:
                raise ValueError(
                    "structured Process contract requires vocabulary and audit log paths"
                )
            self.process_contract_vocabulary = load_operator_vocabulary(
                process_contract_vocabulary_file
            )
        else:
            self.process_contract_vocabulary = ()

    @property
    def name(self) -> str:
        return "cli_only_agent"

    def _with_process_contract_tool(self, template: str) -> str:
        if not getattr(self, "_structured_process_contract_active", False):
            return template
        return (
            template.rstrip()
            + "\n\n## Runtime Tool Definitions\n\n"
            + "{tool_definitions}\n\n"
            + "When `submit_process_contract` is listed above, call it as a ReAct "
            + "Action with its `contract` object after inspecting `input.xlsx`. It is "
            + "an agent tool, not a Bash command or Python function.\n"
        )

    def get_system_template(self) -> str:
        return self._with_process_contract_tool(
            load_full_system_prompt("cli_only_full_system_v1.txt")
        )

    def create_tools(self, working_dir: str) -> list[Tool]:
        bash_tool = create_bash_tool(
            working_dir,
            timeout=self.timeout,
            sandbox_mode=self.sandbox_mode,
        )
        contract_tool = None
        if self._structured_process_contract_active:
            context = self._output_feedback_context
            if context is None or Path(context.working_dir) != Path(working_dir):
                raise RuntimeError("structured Process contract lacks the current task context")
            if self.process_contract_log is None:
                raise RuntimeError("structured Process contract lacks its audit log")
            gate = StructuredProcessContractGate(
                working_dir=working_dir,
                operator_vocabulary=self.process_contract_vocabulary,
                instance_id=context.instance_id,
                protocol_sha256=os.getenv("SB_RUN_PROTOCOL_SHA256", ""),
                contract_log_path=self.process_contract_log,
            )
            bash_tool = gate.wrap_bash(bash_tool)
            contract_tool = gate.create_contract_tool()
        if self.libreoffice_output_feedback:
            context = self._output_feedback_context
            if context is None or context.working_dir != working_dir:
                raise RuntimeError("LibreOffice output feedback lacks the current task context")
            bash_tool = wrap_bash_with_output_feedback(
                bash_tool,
                working_dir=working_dir,
                answer_position=context.answer_position,
                instance_id=context.instance_id,
            )
        return [contract_tool, bash_tool] if contract_tool is not None else [bash_tool]

    def run(self, context: AgentContext) -> dict:
        self._output_feedback_context = context
        return super().run(context)

    def build_task_prompt(self, context: AgentContext) -> str:
        return f"""Below is the spreadsheet manipulation question you need to solve:

### working_directory
.

### instruction
{context.instruction}

### spreadsheet_path
input.xlsx

### spreadsheet_content
{context.spreadsheet_content}

### instruction_type
{context.instruction_type}

### answer_position
{context.answer_position}

### output_path
output.xlsx

---
**REMINDER**: Your bash commands already run inside the current task directory. Read only `input.xlsx`, save the final workbook as `output.xlsx`, and do not search `/tmp` or copy files from other task directories. If answer_position is a range, update every required cell in that range and verify representative target cells before completion.
---

Solve the question and save the modified spreadsheet to the exact output_path shown above."""
