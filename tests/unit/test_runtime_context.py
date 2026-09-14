from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from react_agent.agent import AgentConfig, ReActAgent
from react_agent.converter import ParsedAction
from react_agent.models import (
    OpenAIClient,
    RequestCompletionLengthExceeded,
    RequestContextLengthExceeded,
    _is_context_length_bad_request,
)
from react_agent.tools import Tool
from spreadsheet_agent.agents.cli_only_agent import CLIOnlyAgent


class ContextOverflowClient:
    async def chat_async(self, messages, settings=None):
        raise RequestContextLengthExceeded("maximum context length exceeded")


class CompletionOverflowClient:
    async def chat_async(self, messages, settings=None):
        raise RequestCompletionLengthExceeded("completion reached max_tokens")


class CompletionThenRecoveryClient:
    def __init__(self, partial: str, recovered: str) -> None:
        self.partial = partial
        self.recovered = recovered
        self.calls: list[list] = []

    async def chat_async(self, messages, settings=None):
        self.calls.append(list(messages))
        if len(self.calls) == 1:
            raise RequestCompletionLengthExceeded(
                "completion reached max_tokens",
                partial_content=self.partial,
            )
        return self.recovered


class ScriptedClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.settings = []

    async def chat_async(self, messages, settings=None):
        self.settings.append(settings)
        return self.responses.pop(0)


def test_context_overflow_is_reported_explicitly_instead_of_as_max_turns() -> None:
    agent = ReActAgent(
        client=ContextOverflowClient(),
        config=AgentConfig(max_turns=30),
    )

    result = agent.run("Use the complete retrieved experience context.")

    assert result.success is False
    assert result.error == "CONTEXT_LENGTH_EXCEEDED"
    assert result.final_answer == "CONTEXT_LENGTH_EXCEEDED"
    assert result.total_turns == 0


def test_vllm_context_error_is_detected_without_a_param_field() -> None:
    error = RuntimeError(
        "This model's maximum context length is 100000 tokens, but the request has 100001."
    )

    assert _is_context_length_bad_request(error) is True


def _unchanged_tool() -> Tool:
    return Tool(
        name="inspect",
        description="Inspect the current state.",
        func=lambda: "unchanged",
    )


def test_generic_stagnation_prompt_has_no_dataset_specific_instruction() -> None:
    action = ParsedAction(name="inspect", arguments={})
    agent = ReActAgent(
        client=ContextOverflowClient(),
        tools=[_unchanged_tool()],
        config=AgentConfig(max_turns=30, stagnation_repeat_limit=1),
    )

    assert agent._execute_action(action, turn=1) == "unchanged"
    feedback = agent._execute_action(action, turn=2)

    assert "General recovery guidance" in feedback
    assert "output.xlsx" not in feedback
    assert "ACTION: TASK_COMPLETE" not in feedback


def test_spreadsheet_stagnation_prompt_is_composed_with_generic_prompt() -> None:
    spreadsheet_agent = CLIOnlyAgent(client=ContextOverflowClient())
    task_prompt = spreadsheet_agent.get_stagnation_recovery_prompt()
    action = ParsedAction(name="inspect", arguments={})
    agent = ReActAgent(
        client=ContextOverflowClient(),
        tools=[_unchanged_tool()],
        config=AgentConfig(
            max_turns=30,
            stagnation_repeat_limit=1,
            task_specific_stagnation_prompt=task_prompt,
        ),
    )

    assert agent._execute_action(action, turn=1) == "unchanged"
    feedback = agent._execute_action(action, turn=2)

    assert "General recovery guidance" in feedback
    assert "Task-specific recovery guidance" in feedback
    assert "output.xlsx" in feedback
    assert "ACTION: TASK_COMPLETE" in feedback


def test_cli_agent_wires_task_specific_prompt_into_react_agent(tmp_path: Path) -> None:
    spreadsheet_agent = CLIOnlyAgent(
        client=ContextOverflowClient(),
        max_completion_tokens=16_384,
        completion_recovery_attempt_limit=1,
        max_consecutive_format_errors=2,
        stagnation_recovery_attempt_limit=1,
        truncate_observations=False,
    )

    react_agent = spreadsheet_agent._ensure_agent(str(tmp_path))

    assert react_agent.config.task_specific_stagnation_prompt == (
        spreadsheet_agent.get_stagnation_recovery_prompt()
    )
    assert react_agent.config.max_completion_tokens == 16_384
    assert react_agent.config.completion_recovery_attempt_limit == 1
    assert react_agent.config.max_consecutive_format_errors == 2
    assert react_agent.config.stagnation_recovery_attempt_limit == 1
    assert react_agent.config.truncate_observations is False


def test_no_truncation_runtime_preserves_a_long_tool_observation() -> None:
    long_observation = "BEGIN-" + ("x" * 7000) + "-END"
    tool = Tool(
        name="inspect_long",
        description="Return complete evidence.",
        func=lambda: long_observation,
    )
    agent = ReActAgent(
        client=ContextOverflowClient(),
        tools=[tool],
        config=AgentConfig(truncate_observations=False),
    )

    observed = agent._execute_action(
        ParsedAction(name="inspect_long", arguments={}), turn=1
    )

    assert observed == long_observation


def test_max_completion_tokens_are_applied_to_each_agent_request() -> None:
    client = ScriptedClient(["ACTION: TASK_COMPLETE"])
    agent = ReActAgent(
        client=client,
        config=AgentConfig(max_turns=30, max_completion_tokens=16_384),
    )

    result = agent.run("Finish normally.")

    assert result.success is True
    assert len(client.settings) == 1
    assert client.settings[0].max_tokens == 16_384


def test_completion_limit_is_a_distinct_terminal_failure() -> None:
    agent = ReActAgent(
        client=CompletionOverflowClient(),
        config=AgentConfig(max_turns=30, max_completion_tokens=16_384),
    )

    result = agent.run("Do not return a partial action.")

    assert result.success is False
    assert result.error == "COMPLETION_LENGTH_EXCEEDED"
    assert result.final_answer == "COMPLETION_LENGTH_EXCEEDED"
    assert result.total_turns == 0


def test_completion_limit_continues_once_from_preserved_partial_response() -> None:
    client = CompletionThenRecoveryClient(
        'Thought: prepare the command\nAction:\n{"name":"inspect",',
        "ACTION: TASK_COMPLETE",
    )
    agent = ReActAgent(
        client=client,
        tools=[_unchanged_tool()],
        config=AgentConfig(
            max_turns=1,
            max_completion_tokens=32_000,
            completion_recovery_attempt_limit=1,
        ),
    )

    result = agent.run("Recover the interrupted protocol response.")

    assert result.success is True
    assert len(client.calls) == 2
    recovery_messages = client.calls[1]
    assert recovery_messages[-2].role == "assistant"
    assert recovery_messages[-2].content.endswith('"inspect",')
    assert "without repeating the analysis" in recovery_messages[-1].content


def test_length_limited_prose_is_not_mistaken_for_task_completion() -> None:
    client = CompletionThenRecoveryClient(
        "I am still analyzing the workbook and have not produced an action",
        "ACTION: TASK_COMPLETE",
    )
    agent = ReActAgent(
        client=client,
        config=AgentConfig(
            max_turns=1,
            max_completion_tokens=32_000,
            completion_recovery_attempt_limit=1,
        ),
    )

    result = agent.run("Recover instead of accepting truncated prose.")

    assert result.success is True
    assert len(client.calls) == 2
    assert client.calls[1][-2].role == "assistant"
    assert client.calls[1][-1].role == "user"


def test_length_limited_direct_answer_is_reassembled_from_its_continuation() -> None:
    client = CompletionThenRecoveryClient(
        "The complete answer is forty-",
        "two.",
    )
    agent = ReActAgent(
        client=client,
        config=AgentConfig(
            max_turns=1,
            max_completion_tokens=32_000,
            completion_recovery_attempt_limit=1,
        ),
    )

    result = agent.run("Answer directly without a tool.")

    assert result.success is True
    assert result.final_answer == "The complete answer is forty-two."
    assert len(client.calls) == 2


def test_complete_action_in_length_limited_content_is_accepted_without_retry() -> None:
    complete_action = 'Action:\n{"name":"inspect","arguments":{}}'
    client = CompletionThenRecoveryClient(complete_action, "ACTION: TASK_COMPLETE")
    agent = ReActAgent(
        client=client,
        tools=[_unchanged_tool()],
        config=AgentConfig(
            max_turns=2,
            max_completion_tokens=32_000,
            completion_recovery_attempt_limit=1,
        ),
    )

    result = agent.run("Use an already complete partial action.")

    assert result.success is True
    assert len(client.calls) == 2


def test_openai_length_finish_reason_is_not_parsed_as_a_partial_reply() -> None:
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="length",
                message=SimpleNamespace(content="Action:\n{", reasoning_content=""),
            )
        ]
    )

    try:
        OpenAIClient._parse_response(object.__new__(OpenAIClient), response)
    except RequestCompletionLengthExceeded as exc:
        assert "max_tokens" in str(exc)
        assert exc.partial_content == "Action:\n{"
    else:
        raise AssertionError("length-limited reply was accepted as a normal response")


def test_two_consecutive_format_errors_terminate_without_using_all_turns() -> None:
    client = ScriptedClient(["Action:\nnot-json", "Action:\nstill-not-json"])
    agent = ReActAgent(
        client=client,
        config=AgentConfig(max_turns=30, max_consecutive_format_errors=2),
    )

    result = agent.run("Use valid ReAct actions.")

    assert result.success is False
    assert result.error == "FORMAT_ERROR_EXHAUSTED"
    assert result.total_turns == 2
    assert len(client.settings) == 2


def test_valid_action_resets_consecutive_format_error_count() -> None:
    valid_action = 'Action:\n{"name":"inspect","arguments":{}}'
    client = ScriptedClient(
        ["Action:\nnot-json", valid_action, "Action:\nnot-json", "ACTION: TASK_COMPLETE"]
    )
    agent = ReActAgent(
        client=client,
        tools=[_unchanged_tool()],
        config=AgentConfig(max_turns=4, max_consecutive_format_errors=2),
    )

    result = agent.run("Recover after isolated format errors.")

    assert result.success is True
    assert result.total_turns == 4


def test_stagnation_prompt_has_only_one_recovery_attempt() -> None:
    repeated_action = 'Action:\n{"name":"inspect","arguments":{}}'
    client = ScriptedClient([repeated_action, repeated_action, repeated_action])
    agent = ReActAgent(
        client=client,
        tools=[_unchanged_tool()],
        config=AgentConfig(
            max_turns=30,
            stagnation_repeat_limit=1,
            stagnation_recovery_attempt_limit=1,
        ),
    )

    result = agent.run("Do not repeat an action after the recovery prompt.")

    assert result.success is False
    assert result.error == "STAGNATION_EXHAUSTED"
    assert result.total_turns == 3
    assert len(client.settings) == 3


def test_continuation_uses_the_same_completion_and_format_limits() -> None:
    client = ScriptedClient(
        ["ACTION: TASK_COMPLETE", "Action:\nnot-json", "Action:\nstill-not-json"]
    )
    agent = ReActAgent(
        client=client,
        config=AgentConfig(
            max_turns=3,
            max_completion_tokens=16_384,
            max_consecutive_format_errors=2,
        ),
    )
    initial = agent.run("Signal completion before the adapter check.")
    assert initial.success is True

    continued = agent.continue_with_message("The task-specific check failed.")

    assert continued.success is False
    assert continued.error == "FORMAT_ERROR_EXHAUSTED"
    assert continued.total_turns == 3
    assert [settings.max_tokens for settings in client.settings] == [
        16_384,
        16_384,
        16_384,
    ]
