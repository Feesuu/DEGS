"""
Base class for spreadsheet manipulation agents.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import os
import re
from typing import Callable

from react_agent import ReActAgent, AgentConfig, AgentStep, LLMClient, Tool


@dataclass
class AgentContext:
    """Context information passed to the agent for each task."""
    working_dir: str
    input_file: str
    output_file: str
    instruction: str
    spreadsheet_content: str = ""
    instruction_type: str = ""
    answer_position: str = ""
    instance_id: str = ""  # Unique identifier for this instance (e.g., "13-1")


class ChatHistoryLogger:
    """Log the exact ReAct conversation as one Markdown file per task."""

    def __init__(
        self,
        log_dir: str = "logs",
        log_filename: str | None = None,
    ):
        self.log_dir = log_dir
        self.log_filename = log_filename
        self._current_file: str | None = None
        self._message_count = 0

        os.makedirs(log_dir, exist_ok=True)

    @staticmethod
    def _safe_instance_name(value: str) -> str:
        raw = str(value)
        if re.fullmatch(r"[A-Za-z0-9_.-]+", raw) and raw not in {".", ".."}:
            return raw
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._-")
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10]
        return f"{(safe or 'task')[:80]}_{digest}"

    def start_session(self, agent_name: str, task: str, context: AgentContext | None = None):
        """Start a new logging session."""
        self._message_count = 0

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.log_filename:
            filename = self.log_filename
        elif context and context.instance_id:
            # Use instance_id for stable per-task log filenames.
            filename = (
                f"{agent_name}_{self._safe_instance_name(context.instance_id)}.md"
            )
        else:
            filename = f"{agent_name}_{timestamp}.md"

        self._current_file = os.path.join(self.log_dir, filename)

        with open(self._current_file, "w") as f:
            f.write(f"# Chat History: {agent_name}\n\n")
            f.write(f"**Timestamp**: {datetime.now().isoformat()}\n\n")
            if context and context.instance_id:
                f.write(
                    "**Instance ID JSON**: "
                    + json.dumps(str(context.instance_id), ensure_ascii=False)
                    + "\n\n"
                )
            protocol_sha = os.getenv("SB_RUN_PROTOCOL_SHA256", "")
            if protocol_sha:
                f.write(f"**Run Protocol SHA256**: {protocol_sha}\n\n")
            f.write("---\n\n")

    def log_message(self, role: str, content: str):
        """Log a raw message."""
        if self._current_file is None:
            return

        self._message_count += 1

        with open(self._current_file, "a") as f:
            f.write(f"## [{self._message_count}] {role.upper()}\n\n")
            f.write(f"{content}\n\n")
            f.write("---\n\n")

    def log_step(self, step: AgentStep):
        """Log an agent step as raw messages."""
        if self._current_file is None:
            return

        # Log assistant response (thought or final answer)
        if step.thought:
            self.log_message("assistant", step.thought)

        # Log observation as user message (this is how ReAct formats it)
        if step.observation:
            fence = "```"
            while fence in step.observation:
                fence += "`"
            self.log_message("user", f"Observation:\n{fence}\n{step.observation}\n{fence}")

    def log_system_prompt(self, system_prompt: str):
        """Log the system prompt."""
        self.log_message("system", system_prompt)

    def log_user_task(self, task: str):
        """Log the initial user task."""
        self.log_message("user", task)

    def log_result(self, success: bool, answer: str, turns: int, error: str | None = None):
        """Log the final result summary."""
        if self._current_file is None:
            return

        with open(self._current_file, "a") as f:
            f.write("## RESULT\n\n")
            f.write(f"- Success: {success}\n")
            f.write(f"- Total Turns: {turns}\n")
            if error:
                f.write(f"- Error: {error}\n")

    def get_log_file(self) -> str | None:
        """Get the current log file path."""
        return self._current_file


class BaseSpreadsheetAgent(ABC):
    """
    Base class for spreadsheet manipulation agents.

    Subclasses provide one system template, one task prompt, and their tools.
    """

    def __init__(
        self,
        client: LLMClient,
        max_turns: int = 15,
        verbose: bool = True,
        log_dir: str | None = None,
        stagnation_repeat_limit: int = 0,
    ):
        """
        Initialize the agent.

        Args:
            client: LLM client for generation
            max_turns: Maximum reasoning turns
            verbose: Whether to print debug output
            log_dir: Directory for chat history logs (None to disable logging)
            stagnation_repeat_limit: Identical action/observation repeats allowed
        """
        self.client = client
        self.max_turns = max_turns
        self.verbose = verbose
        self.stagnation_repeat_limit = stagnation_repeat_limit
        self._working_dir: str | None = None
        self._agent: ReActAgent | None = None

        # Setup chat history logger
        if log_dir:
            self._logger = ChatHistoryLogger(log_dir=log_dir)
        else:
            self._logger = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the agent's name."""
        pass

    @abstractmethod
    def get_system_template(self) -> str:
        """Return a full system prompt template containing {tool_definitions}."""
        pass

    @abstractmethod
    def create_tools(self, working_dir: str) -> list[Tool]:
        """
        Create and return the tools for this agent.

        Args:
            working_dir: The working directory for tool execution
        """
        pass

    def get_no_truncate_patterns(self) -> list[str]:
        """
        Return paths that should not have their output truncated.
        
        Override this method to specify reference paths whose observations
        should not be truncated.
        
        Returns:
            List of path patterns that should not be truncated
        """
        return []

    @abstractmethod
    def build_task_prompt(self, context: AgentContext) -> str:
        """Build the task prompt using task-relative paths."""
        pass

    def _create_step_callback(self) -> Callable[[AgentStep], None] | None:
        """Create a step callback that logs to the history logger."""
        if self._logger is None:
            return None

        def on_step(step: AgentStep):
            self._logger.log_step(step)

        return on_step

    def _ensure_agent(self, working_dir: str) -> ReActAgent:
        """Ensure the agent is created with the correct working directory."""
        if self._agent is None or self._working_dir != working_dir:
            self._working_dir = working_dir
            tools = self.create_tools(working_dir)

            # Get reference patterns that should not be truncated.
            no_truncate_patterns = self.get_no_truncate_patterns()
            config = AgentConfig(
                max_turns=self.max_turns,
                verbose=self.verbose,
                system_template=self.get_system_template(),
                no_truncate_patterns=no_truncate_patterns,
                stagnation_repeat_limit=self.stagnation_repeat_limit,
            )

            self._agent = ReActAgent(
                client=self.client,
                tools=tools,
                config=config,
                on_step=self._create_step_callback(),
            )
        return self._agent

    def run(self, context: AgentContext) -> dict:
        """
        Run the agent on a task.

        Args:
            context: The task context with input/output paths and instructions

        Returns:
            Dictionary with 'success', 'answer', 'turns', and optionally 'error'
        """
        # Keep the agent-facing file contract relative to the task cwd. The
        # runner still uses absolute paths internally when checking outputs.
        os.environ["INPUT_FILE"] = "input.xlsx"
        os.environ["OUTPUT_FILE"] = "output.xlsx"

        # Build and run
        agent = self._ensure_agent(context.working_dir)
        agent.set_runtime_context(
            instance_id=context.instance_id,
            protocol_sha256=os.getenv("SB_RUN_PROTOCOL_SHA256", ""),
        )
        task_prompt = self.build_task_prompt(context)

        # Start logging session and log initial messages
        if self._logger:
            self._logger.start_session(self.name, task_prompt, context)
            full_system_prompt = agent.converter.build_system_prompt(
                tools=agent.tool_registry.list_tools(),
            )
            self._logger.log_system_prompt(full_system_prompt)
            # Log the user task
            self._logger.log_user_task(f"Task: {task_prompt}")

        try:
            result = agent.run(task_prompt)
            
            # Check if output file exists after agent signals completion
            output_exists = os.path.exists(context.output_file)
            
            # If agent succeeded but output doesn't exist, give it another chance
            if result.success and not output_exists:
                remaining_turns = self.max_turns - result.total_turns
                if remaining_turns > 0:
                    if self.verbose:
                        print(f"\n[WARNING] Output file not found at {context.output_file}")
                        print(f"[WARNING] Sending reminder to agent ({remaining_turns} turns remaining)...")
                    
                    if self._logger:
                        self._logger.log_user_task(
                            "[System Check] The output file was NOT created as `output.xlsx` in the current task directory.\n"
                            "Please return to work again until you create `output.xlsx`, then signal ACTION: TASK_COMPLETE again."
                        )

                    # Continue the conversation with a reminder
                    result = agent.continue_with_message(
                        "[System Check] The output file was NOT created as `output.xlsx` in the current task directory.\n"
                        "Please return to work again until you create `output.xlsx`, then signal ACTION: TASK_COMPLETE again."
                    )
                    
                    # Re-check output file
                    output_exists = os.path.exists(context.output_file)
            
            # Determine final success and failure reason
            success = result.success and output_exists
            
            # Build detailed error message for failures
            error = result.error
            if not success and not error:
                failure_reasons = []
                if not result.success:
                    failure_reasons.append("Agent did not complete successfully")
                if not output_exists:
                    failure_reasons.append(f"Output file was not created: {context.output_file}")
                error = "; ".join(failure_reasons)
            
            run_result = {
                "success": success,
                "answer": result.final_answer,
                "turns": result.total_turns,
                "error": error,
            }

            # Log failure reason to verbose output
            if not success and self.verbose:
                print(f"[FAILURE] {error}")

            # Log final result
            if self._logger:
                self._logger.log_result(
                    success=run_result["success"],
                    answer=run_result["answer"],
                    turns=run_result["turns"],
                    error=run_result["error"],
                )

            return run_result

        except Exception as e:
            error_msg = f"Exception during agent execution: {str(e)}"
            
            # Log failure to verbose output
            if self.verbose:
                print(f"[FAILURE] {error_msg}")
            
            error_result = {
                "success": False,
                "answer": "",
                "turns": 0,
                "error": error_msg,
            }

            if self._logger:
                self._logger.log_result(
                    success=False,
                    answer="",
                    turns=0,
                    error=error_msg,
                )

            return error_result

    def get_last_log_file(self) -> str | None:
        """Get the path to the last log file."""
        if self._logger:
            return self._logger.get_log_file()
        return None
