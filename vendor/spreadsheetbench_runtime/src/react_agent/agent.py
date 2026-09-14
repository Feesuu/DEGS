"""
Core ReAct Agent implementation.

The ReAct (Reasoning + Acting) pattern allows LLMs to solve complex tasks by:
1. THINK: Reasoning about the current state and what to do next
2. ACT: Calling a tool to perform an action
3. OBSERVE: Receiving the result of the action
4. REPEAT: Until the task is complete

This module provides the main ReActAgent class that orchestrates this loop.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from typing import Callable
from enum import Enum

from .converter import ReActConverter, ParsedAction, ParseResult, truncate_observation
from .models import (
    LLMClient,
    Message,
    ModelSettings,
    RequestContextLengthExceeded,
    record_runtime_event,
)
from .tools import Tool, ToolRegistry


def _strip_think(text: str) -> str:
    """Strip <think>...</think> prefix from an LLM response.

    Many reasoning models emit a thinking block before the actual content.
    When appending older assistant messages to conversation history, this
    thinking should be removed so it doesn't leak into subsequent turns.
    """
    if "</think>" in text:
        return text.rsplit("</think>", 1)[-1].lstrip("\n")
    return text

class AgentState(Enum):
    """Current state of the agent."""
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    OBSERVING = "observing"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class AgentConfig:
    """Configuration for the ReAct agent."""
    max_turns: int = 10
    """Maximum number of turns (tool calls) before stopping."""

    verbose: bool = True
    """Whether to print debug information."""

    system_instructions: str = ""
    """Additional instructions to include in the system prompt."""

    system_template: str | None = None
    """Optional full system prompt template (must include {tool_definitions})."""

    no_truncate_patterns: list[str] = field(default_factory=list)
    """List of path patterns - observations from commands containing these paths won't be truncated.
    Useful for reference files that should always be loaded in full."""

    stagnation_repeat_limit: int = 0
    """Block the next identical action after this many identical observations."""


@dataclass
class AgentStep:
    """Represents a single step in the agent's execution."""
    turn: int
    thought: str | None = None
    action: ParsedAction | None = None
    observation: str | None = None
    is_final: bool = False
    is_format_error: bool = False


@dataclass
class AgentResult:
    """Result of an agent run."""
    task: str
    final_answer: str
    steps: list[AgentStep] = field(default_factory=list)
    total_turns: int = 0
    success: bool = True
    error: str | None = None

    def __repr__(self) -> str:
        status = "✓" if self.success else "✗"
        return f"AgentResult({status}, turns={self.total_turns}, answer={self.final_answer[:50]}...)"


class ReActAgent:
    """
    A ReAct (Reasoning + Acting) agent that can use tools to solve tasks.

    The agent follows the think-act-observe loop:
    1. Receives a task from the user
    2. Thinks about what to do (via LLM)
    3. Optionally calls a tool (Action)
    4. Observes the result (Observation)
    5. Repeats until it has a final answer or max turns exceeded

    Example:
        ```python
        from react_agent import ReActAgent, OpenAIClient, tool

        @tool()
        def search(query: str) -> str:
            '''Search the web for information.'''
            return "Search results for: " + query

        client = OpenAIClient(model="gpt-4o-mini")
        agent = ReActAgent(client=client, tools=[search])

        result = agent.run("What is the capital of France?")
        print(result.final_answer)
        ```
    """

    def __init__(
        self,
        client: LLMClient,
        tools: list[Tool] | None = None,
        config: AgentConfig | None = None,
        on_step: Callable[[AgentStep], None] | None = None,
    ):
        """
        Initialize the ReAct agent.

        Args:
            client: LLM client for generating responses
            tools: List of tools the agent can use
            config: Agent configuration
            on_step: Optional callback called after each step
        """
        self.client = client
        self.config = config or AgentConfig()
        self.converter = ReActConverter(
            custom_system_template=self.config.system_template,
        )
        self.on_step = on_step

        # Set up tool registry
        self.tool_registry = ToolRegistry()
        for tool in (tools or []):
            self.tool_registry.register(tool)

        self.state = AgentState.IDLE
        if self.config.stagnation_repeat_limit < 0:
            raise ValueError("stagnation_repeat_limit must be non-negative")
        self._last_action_fingerprint: str | None = None
        self._last_action_raw_observation: str | None = None
        self._identical_action_observation_count = 0
        self._runtime_instance_id = ""
        self._runtime_protocol_sha256 = ""
        
        # Conversation state for continuation
        self._last_messages: list[Message] = []
        self._last_steps: list[AgentStep] = []
        self._last_turn: int = 0
        self._last_task: str = ""

    def add_tool(self, tool: Tool) -> None:
        """Add a tool to the agent."""
        self.tool_registry.register(tool)

    def _reset_stagnation_state(self) -> None:
        self._last_action_fingerprint = None
        self._last_action_raw_observation = None
        self._identical_action_observation_count = 0

    def set_runtime_context(self, *, instance_id: str, protocol_sha256: str) -> None:
        self._runtime_instance_id = str(instance_id)
        self._runtime_protocol_sha256 = str(protocol_sha256)
        setter = getattr(self.client, "set_runtime_context", None)
        if setter is not None:
            setter(
                instance_id=self._runtime_instance_id,
                protocol_sha256=self._runtime_protocol_sha256,
            )

    @staticmethod
    def _action_fingerprint(action: ParsedAction) -> str:
        return json.dumps(
            {"name": action.name, "arguments": action.arguments},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )

    def _execute_action(self, action: ParsedAction, *, turn: int) -> str:
        fingerprint = self._action_fingerprint(action)
        limit = self.config.stagnation_repeat_limit
        if (
            limit > 0
            and fingerprint == self._last_action_fingerprint
            and self._identical_action_observation_count >= limit
        ):
            record_runtime_event(
                "stagnation_action_blocked",
                turn=turn,
                instance_id=self._runtime_instance_id,
                protocol_sha256=self._runtime_protocol_sha256,
                action_name=action.name,
                action_fingerprint_sha256=hashlib.sha256(
                    fingerprint.encode("utf-8")
                ).hexdigest(),
                repeated_observation_sha256=hashlib.sha256(
                    (self._last_action_raw_observation or "").encode("utf-8")
                ).hexdigest(),
                repeat_limit=limit,
            )
            return (
                "STAGNATION_GUARD: This exact action already produced the same "
                f"observation {limit} consecutive times. It was not executed again. "
                "Use a different command or, if output.xlsx is already valid, signal "
                "ACTION: TASK_COMPLETE now."
            )

        raw_observation = self.tool_registry.execute(action.name, **action.arguments)
        if (
            fingerprint == self._last_action_fingerprint
            and raw_observation == self._last_action_raw_observation
        ):
            self._identical_action_observation_count += 1
        else:
            self._last_action_fingerprint = fingerprint
            self._last_action_raw_observation = raw_observation
            self._identical_action_observation_count = 1
        action_str = str(action.arguments.get("command", "")) if action.arguments else ""
        return truncate_observation(
            raw_observation,
            action_str=action_str,
            no_truncate_patterns=self.config.no_truncate_patterns,
        )

    def run(self, task: str) -> AgentResult:
        """
        Run the agent on a task (synchronous).

        Args:
            task: The task/question to solve

        Returns:
            AgentResult with the final answer and execution history
        """
        return asyncio.run(self.run_async(task))

    def _build_max_turns_exceeded_result(
        self,
        task: str,
        messages: list[Message],
        steps: list[AgentStep],
    ) -> AgentResult:
        """Return the standard max-turns termination result."""
        self.state = AgentState.ERROR
        error_msg = "Max turns exceeded"

        if self.config.verbose:
            print(f"\n[FAILURE] {error_msg}")

        self._last_messages = messages
        self._last_steps = steps
        self._last_turn = self.config.max_turns
        self._last_task = task

        return AgentResult(
            task=task,
            final_answer="Max turns exceeded. Could not complete the task.",
            steps=steps,
            total_turns=self.config.max_turns,
            success=False,
            error=error_msg,
        )

    async def run_async(self, task: str) -> AgentResult:
        """
        Run the agent on a task (asynchronous).

        Args:
            task: The task/question to solve

        Returns:
            AgentResult with the final answer and execution history
        """
        self.state = AgentState.THINKING
        self._reset_stagnation_state()
        steps: list[AgentStep] = []

        # Build initial messages
        tools = self.tool_registry.list_tools()
        system_prompt = self.converter.build_system_prompt(
            tools=tools,
            extra_instructions=self.config.system_instructions,
        )

        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=f"Task: {task}"),
        ]

        # Model settings with stop sequences (temperature handled by client)
        settings = ModelSettings(
            stop=self.converter.get_stop_sequences(),
        )

        if self.config.verbose:
            print(f"\n{'='*60}")
            print(f"[Task] {task}")
            print(f"[Tools] {[t.name for t in tools]}")
            print(f"{'='*60}\n")

        try:
            for turn in range(1, self.config.max_turns + 1):
                step = AgentStep(turn=turn)

                # THINK: Get LLM response
                self.state = AgentState.THINKING
                response = await self.client.chat_async(messages, settings)

                if self.config.verbose:
                    print(f"[Turn {turn}]")
                    print(f"   Response: {response[:500]}..." if len(response) > 500 else f"   Response: {response}")

                # Parse response for action, task completion, or format error
                parse_result = self.converter.parse_response(response)

                if parse_result.is_task_complete:
                    # Task complete signal detected
                    step.thought = response
                    step.is_final = True
                    steps.append(step)

                    if self.on_step:
                        self.on_step(step)

                    self.state = AgentState.COMPLETE

                    if self.config.verbose:
                        print(f"\n[DONE] Task completed (Turn {turn})")

                    # Save state for potential continuation
                    # Strip thinking from history so it doesn't leak into future turns
                    messages.append(Message(role="assistant", content=_strip_think(response)))
                    self._last_messages = messages
                    self._last_steps = steps
                    self._last_turn = turn
                    self._last_task = task

                    return AgentResult(
                        task=task,
                        final_answer=self.converter.extract_final_answer(response),
                        steps=steps,
                        total_turns=turn,
                        success=True,
                    )

                if parse_result.is_format_error:
                    # Format error - send error message and retry
                    step.thought = response
                    step.is_format_error = True
                    step.observation = parse_result.error_message
                    steps.append(step)

                    if self.config.verbose:
                        print(f"   [FORMAT ERROR] Sending reminder to model")

                    if self.on_step:
                        self.on_step(step)

                    # Add error message to conversation and retry
                    messages.append(Message(role="assistant", content=_strip_think(response)))
                    messages.append(Message(
                        role="user",
                        content=self.converter.format_observation(parse_result.error_message, truncate=False)
                    ))

                    # Don't count format errors as action turns
                    turn -= 1
                    continue

                # Valid action - execute it
                action = parse_result.action
                self.state = AgentState.ACTING
                step.action = action
                step.thought = response

                if self.config.verbose:
                    print(f"   [Action] {action.name}({action.arguments})")

                # OBSERVE: Add result to conversation (truncate long outputs)
                self.state = AgentState.OBSERVING
                observation = self._execute_action(action, turn=turn)
                step.observation = observation
                steps.append(step)

                if self.config.verbose:
                    obs_display = observation[:500] + "..." if len(observation) > 500 else observation
                    print(f"   [Observation] {obs_display}")

                if self.on_step:
                    self.on_step(step)

                # Add to message history
                messages.append(Message(role="assistant", content=_strip_think(response)))
                messages.append(Message(
                    role="user",
                    content=self.converter.format_observation(observation, truncate=False)
                ))

            return self._build_max_turns_exceeded_result(task, messages, steps)

        except RequestContextLengthExceeded:
            return self._build_max_turns_exceeded_result(task, messages, steps)

        except Exception as e:
            self.state = AgentState.ERROR
            error_msg = f"Exception during execution: {str(e)}"

            if self.config.verbose:
                print(f"\n[FAILURE] {error_msg}")

            self._last_messages = messages
            self._last_steps = steps
            self._last_turn = len(steps)
            self._last_task = task

            return AgentResult(
                task=task,
                final_answer=f"Error: {e}",
                steps=steps,
                total_turns=len(steps),
                success=False,
                error=error_msg,
            )
        finally:
            # Ensure async client resources are closed before loop teardown
            if hasattr(self.client, "aclose"):
                try:
                    await self.client.aclose()
                except Exception:
                    pass

    def continue_with_message(self, message: str) -> AgentResult:
        """
        Continue the conversation with an additional user message.
        
        This is useful when the agent signals TASK_COMPLETE but some
        post-completion check fails (e.g., output file doesn't exist).
        
        Args:
            message: The message to send to continue the conversation
            
        Returns:
            AgentResult from the continued execution
        """
        return asyncio.run(self.continue_with_message_async(message))

    async def _run_continuation_loop_async(
        self,
        messages: list[Message],
        steps: list[AgentStep],
        task: str,
        start_turn: int,
    ) -> AgentResult:
        """
        Shared turn loop used when the runner asks the agent to create a
        missing output file after an early completion signal.
        """
        self.state = AgentState.THINKING
        settings = ModelSettings(
            stop=self.converter.get_stop_sequences(),
        )

        for turn in range(start_turn, self.config.max_turns + 1):
            step = AgentStep(turn=turn)

            # THINK: Get LLM response
            self.state = AgentState.THINKING
            response = await self.client.chat_async(messages, settings)

            if self.config.verbose:
                print(f"[Turn {turn}]")
                print(f"   Response: {response[:500]}..." if len(response) > 500 else f"   Response: {response}")

            parse_result = self.converter.parse_response(response)

            if parse_result.is_task_complete:
                step.thought = response
                step.is_final = True
                steps.append(step)

                if self.on_step:
                    self.on_step(step)

                self.state = AgentState.COMPLETE

                if self.config.verbose:
                    print(f"\n[DONE] Task completed (Turn {turn})")

                messages.append(Message(role="assistant", content=_strip_think(response)))
                self._last_messages = messages
                self._last_steps = steps
                self._last_turn = turn
                self._last_task = task

                return AgentResult(
                    task=task,
                    final_answer=self.converter.extract_final_answer(response),
                    steps=steps,
                    total_turns=turn,
                    success=True,
                )

            if parse_result.is_format_error:
                step.thought = response
                step.is_format_error = True
                step.observation = parse_result.error_message
                steps.append(step)

                if self.config.verbose:
                    print(f"   [FORMAT ERROR] Sending reminder to model")

                if self.on_step:
                    self.on_step(step)

                messages.append(Message(role="assistant", content=_strip_think(response)))
                messages.append(Message(
                    role="user",
                    content=self.converter.format_observation(parse_result.error_message, truncate=False)
                ))

                turn -= 1
                continue

            # Valid action - execute it
            action = parse_result.action
            self.state = AgentState.ACTING
            step.action = action
            step.thought = response

            if self.config.verbose:
                print(f"   [Action] {action.name}({action.arguments})")

            self.state = AgentState.OBSERVING
            observation = self._execute_action(action, turn=turn)
            step.observation = observation
            steps.append(step)

            if self.config.verbose:
                obs_display = observation[:500] + "..." if len(observation) > 500 else observation
                print(f"   [Observation] {obs_display}")

            if self.on_step:
                self.on_step(step)

            messages.append(Message(role="assistant", content=_strip_think(response)))
            messages.append(Message(
                role="user",
                content=self.converter.format_observation(observation, truncate=False)
            ))

        return self._build_max_turns_exceeded_result(task, messages, steps)

    async def continue_with_message_async(self, message: str) -> AgentResult:
        """
        Continue the conversation with an additional user message (async).

        If the prior run stopped immediately after a tool observation, merge the
        continuation instruction into that observation message. This preserves
        valid alternating chat roles while allowing a runner to request a final
        output after the normal turn budget is exhausted.

        Args:
            message: The user message to append before resuming.

        Returns:
            AgentResult from the continued execution.
        """
        if not self._last_messages:
            raise RuntimeError("No previous conversation to continue. Call run() first.")

        messages = self._last_messages.copy()

        if messages[-1].role == "user":
            messages[-1] = Message(
                role="user",
                content=f"{messages[-1].content}\n\n{message}",
            )
        else:
            messages.append(Message(role="user", content=message))

        steps = self._last_steps.copy()
        turn = self._last_turn
        task = self._last_task

        if self.config.verbose:
            print(f"\n[CONTINUE] Sending message...")
            print(f"   Message: {message[:200]}..." if len(message) > 200 else f"   Message: {message}")

        try:
            return await self._run_continuation_loop_async(messages, steps, task, turn + 1)
        except RequestContextLengthExceeded:
            return self._build_max_turns_exceeded_result(task, messages, steps)
        except Exception as e:
            self.state = AgentState.ERROR
            error_msg = f"Exception during execution: {str(e)}"
            if self.config.verbose:
                print(f"\n[FAILURE] {error_msg}")
            return AgentResult(
                task=task,
                final_answer=f"Error: {e}",
                steps=steps,
                total_turns=len(steps),
                success=False,
                error=error_msg,
            )
        finally:
            if hasattr(self.client, "aclose"):
                try:
                    await self.client.aclose()
                except Exception:
                    pass

