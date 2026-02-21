"""Tests for the Agent class."""

from __future__ import annotations

import sys
import types
from collections.abc import AsyncIterator, Iterator
from typing import Any
from unittest.mock import patch

from astro_context.agent.agent import Agent
from astro_context.agent.tools import AgentTool
from astro_context.memory.manager import MemoryManager
from astro_context.storage.json_memory_store import InMemoryEntryStore

# ---------------------------------------------------------------------------
# Fake Anthropic types for testing (no anthropic dependency needed)
# ---------------------------------------------------------------------------


class _Tok:
    """Minimal tokenizer for tests."""

    def count_tokens(self, text: str) -> int:
        return len(text.split()) if text.strip() else 0

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        return " ".join(text.split()[:max_tokens])


class FakeTextBlock:
    """Mock Anthropic TextBlock."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.type = "text"


class FakeToolUseBlock:
    """Mock Anthropic ToolUseBlock."""

    def __init__(self, block_id: str, name: str, block_input: dict[str, Any]) -> None:
        self.id = block_id
        self.name = name
        self.input = block_input
        self.type = "tool_use"


class FakeMessage:
    """Mock Anthropic Message."""

    def __init__(
        self, content: list[Any], stop_reason: str = "end_turn",
    ) -> None:
        self.content = content
        self.stop_reason = stop_reason


class FakeStream:
    """Mock Anthropic MessageStream context manager (sync)."""

    def __init__(self, text: str, message: FakeMessage) -> None:
        self._text = text
        self._message = message

    @property
    def text_stream(self) -> Iterator[str]:
        if self._text:
            yield self._text

    def get_final_message(self) -> FakeMessage:
        return self._message

    def __enter__(self) -> FakeStream:
        return self

    def __exit__(self, *args: Any) -> None:
        pass


class FakeAsyncStream:
    """Mock Anthropic MessageStream context manager (async)."""

    def __init__(self, text: str, message: FakeMessage) -> None:
        self._text = text
        self._message = message

    @property
    async def text_stream(self) -> AsyncIterator[str]:
        if self._text:
            yield self._text

    def get_final_message(self) -> FakeMessage:
        return self._message

    async def __aenter__(self) -> FakeAsyncStream:
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass


class FakeMessages:
    """Mock for client.messages with stream() support."""

    def __init__(
        self,
        responses: list[tuple[str, FakeMessage]],
        *,
        use_async: bool = False,
    ) -> None:
        self._responses = list(responses)
        self._call_index = 0
        self._use_async = use_async

    def stream(self, **kwargs: Any) -> FakeStream | FakeAsyncStream:
        if self._call_index < len(self._responses):
            text, msg = self._responses[self._call_index]
            self._call_index += 1
            if self._use_async:
                return FakeAsyncStream(text, msg)
            return FakeStream(text, msg)
        empty = FakeMessage(content=[FakeTextBlock(text="")])
        if self._use_async:
            return FakeAsyncStream("", empty)
        return FakeStream("", empty)


class FakeAnthropicClient:
    """Mock Anthropic client that returns canned responses."""

    def __init__(
        self,
        responses: list[tuple[str, FakeMessage]],
        *,
        use_async: bool = False,
    ) -> None:
        self.messages = FakeMessages(responses, use_async=use_async)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(
    responses: list[tuple[str, FakeMessage]],
    *,
    tools: list[AgentTool] | None = None,
    memory: MemoryManager | None = None,
    max_rounds: int = 10,
    max_retries: int = 3,
    system_prompt: str = "You are helpful.",
    use_async: bool = False,
) -> Agent:
    """Create an Agent with a fake Anthropic client."""
    client = FakeAnthropicClient(responses, use_async=use_async)
    agent = Agent(
        model="test-model", client=client,
        max_rounds=max_rounds, max_retries=max_retries,
    )
    agent.with_system_prompt(system_prompt)
    if memory is not None:
        agent.with_memory(memory)
    if tools:
        agent.with_tools(tools)
    return agent


# ---------------------------------------------------------------------------
# Tests — sync chat
# ---------------------------------------------------------------------------


def test_basic_chat():
    """Agent returns streamed text for a simple response."""
    responses = [
        ("Hello there!", FakeMessage(
            content=[FakeTextBlock(text="Hello there!")],
            stop_reason="end_turn",
        )),
    ]
    agent = _make_agent(responses)
    chunks = list(agent.chat("Hi"))
    assert "".join(chunks) == "Hello there!"


def test_memory_user_and_assistant_recorded():
    """User and assistant messages are recorded in memory."""
    memory = MemoryManager(conversation_tokens=2000, tokenizer=_Tok())
    responses = [
        ("Hi!", FakeMessage(
            content=[FakeTextBlock(text="Hi!")],
            stop_reason="end_turn",
        )),
    ]
    agent = _make_agent(responses, memory=memory)
    list(agent.chat("Hello"))

    turns = memory.conversation.turns
    assert len(turns) == 2
    assert turns[0].role == "user"
    assert turns[0].content == "Hello"
    assert turns[1].role == "assistant"
    assert turns[1].content == "Hi!"


def test_tool_loop():
    """Tools are executed and results fed back for another response."""
    tool_calls: list[str] = []

    def echo_tool(text: str) -> str:
        tool_calls.append(text)
        return f"Echo: {text}"

    tool = AgentTool(
        name="echo",
        description="Echoes text back",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        fn=echo_tool,
    )

    # First response: model calls tool
    tool_use_msg = FakeMessage(
        content=[FakeToolUseBlock(block_id="tu_1", name="echo", block_input={"text": "hello"})],
        stop_reason="tool_use",
    )
    # Second response: model returns text
    text_msg = FakeMessage(
        content=[FakeTextBlock(text="Done!")],
        stop_reason="end_turn",
    )

    responses = [("", tool_use_msg), ("Done!", text_msg)]
    agent = _make_agent(responses, tools=[tool])
    chunks = list(agent.chat("Test tool"))

    assert "".join(chunks) == "Done!"
    assert tool_calls == ["hello"]


def test_tool_loop_with_text_before_tool():
    """Text before a tool call is still yielded."""
    def noop(x: str) -> str:
        return "ok"

    tool = AgentTool(
        name="noop", description="noop",
        input_schema={"type": "object", "properties": {"x": {"type": "string"}}},
        fn=noop,
    )

    # First response: text + tool_use
    msg1 = FakeMessage(
        content=[
            FakeTextBlock(text="Thinking..."),
            FakeToolUseBlock(block_id="tu_1", name="noop", block_input={"x": "1"}),
        ],
        stop_reason="tool_use",
    )
    msg2 = FakeMessage(
        content=[FakeTextBlock(text=" All done.")],
        stop_reason="end_turn",
    )

    responses = [("Thinking...", msg1), (" All done.", msg2)]
    agent = _make_agent(responses, tools=[tool])
    chunks = list(agent.chat("Go"))
    assert "".join(chunks) == "Thinking... All done."


def test_max_rounds_stops_loop():
    """Agent stops after max_rounds even if model keeps calling tools."""
    call_count = [0]

    def counting_tool(x: str) -> str:
        call_count[0] += 1
        return "ok"

    tool = AgentTool(
        name="counter", description="counts",
        input_schema={
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"],
        },
        fn=counting_tool,
    )

    # Always return tool_use (more responses than max_rounds)
    responses = [
        (
            "",
            FakeMessage(
                content=[FakeToolUseBlock(
                    block_id=f"tu_{i}", name="counter", block_input={"x": "go"},
                )],
                stop_reason="tool_use",
            ),
        )
        for i in range(5)
    ]

    agent = _make_agent(responses, tools=[tool], max_rounds=3)
    list(agent.chat("Go"))

    assert call_count[0] == 3


def test_unknown_tool_returns_error():
    """Calling an unknown tool returns an error message instead of crashing."""
    msg = FakeMessage(
        content=[FakeToolUseBlock(
            block_id="tu_1", name="nonexistent", block_input={"q": "x"},
        )],
        stop_reason="tool_use",
    )
    text_msg = FakeMessage(
        content=[FakeTextBlock(text="OK")],
        stop_reason="end_turn",
    )

    responses = [("", msg), ("OK", text_msg)]
    # No tools registered
    agent = _make_agent(responses)
    chunks = list(agent.chat("Test"))
    assert "".join(chunks) == "OK"


def test_tool_exception_handled():
    """A tool that raises an exception returns an error string."""
    def failing_tool(x: str) -> str:
        msg = "boom"
        raise RuntimeError(msg)

    tool = AgentTool(
        name="fail", description="always fails",
        input_schema={"type": "object", "properties": {"x": {"type": "string"}}},
        fn=failing_tool,
    )

    msg1 = FakeMessage(
        content=[FakeToolUseBlock(block_id="tu_1", name="fail", block_input={"x": "go"})],
        stop_reason="tool_use",
    )
    msg2 = FakeMessage(
        content=[FakeTextBlock(text="Recovered.")],
        stop_reason="end_turn",
    )

    responses = [("", msg1), ("Recovered.", msg2)]
    agent = _make_agent(responses, tools=[tool])
    chunks = list(agent.chat("Go"))
    assert "".join(chunks) == "Recovered."


def test_no_memory_still_works():
    """Agent works without memory attached."""
    responses = [
        ("Hi!", FakeMessage(
            content=[FakeTextBlock(text="Hi!")],
            stop_reason="end_turn",
        )),
    ]
    agent = _make_agent(responses)
    assert agent.memory is None
    chunks = list(agent.chat("Hello"))
    assert "".join(chunks) == "Hi!"


def test_empty_response_no_memory_write():
    """Empty response text is not written to memory."""
    memory = MemoryManager(conversation_tokens=2000, tokenizer=_Tok())
    responses = [
        ("", FakeMessage(
            content=[FakeTextBlock(text="")],
            stop_reason="end_turn",
        )),
    ]
    agent = _make_agent(responses, memory=memory)
    list(agent.chat("Hello"))

    turns = memory.conversation.turns
    # User message recorded, but empty assistant response is NOT
    assert len(turns) == 1
    assert turns[0].role == "user"


def test_with_tools_is_additive():
    """Calling with_tools multiple times adds tools, doesn't replace."""

    def t1() -> str:
        return "1"

    def t2() -> str:
        return "2"

    tool_a = AgentTool(
        name="a", description="a",
        input_schema={"type": "object"}, fn=t1,
    )
    tool_b = AgentTool(
        name="b", description="b",
        input_schema={"type": "object"}, fn=t2,
    )

    responses = [
        ("ok", FakeMessage(content=[FakeTextBlock(text="ok")], stop_reason="end_turn")),
    ]
    agent = _make_agent(responses)
    agent.with_tools([tool_a])
    agent.with_tools([tool_b])

    # Access internal tools list
    assert len(agent._tools) == 2
    names = {t.name for t in agent._tools}
    assert names == {"a", "b"}


def test_pipeline_property():
    """Pipeline is accessible via property."""
    responses = [
        ("ok", FakeMessage(content=[FakeTextBlock(text="ok")], stop_reason="end_turn")),
    ]
    agent = _make_agent(responses, system_prompt="Be helpful.")
    pipeline = agent.pipeline
    assert pipeline is not None
    assert pipeline.max_tokens == 16384


def test_memory_with_persistent_store():
    """Agent with persistent memory store records facts via tools."""
    memory = MemoryManager(
        conversation_tokens=2000, tokenizer=_Tok(),
        persistent_store=InMemoryEntryStore(),
    )
    tools = [
        AgentTool(
            name="save_fact",
            description="Save a fact",
            input_schema={
                "type": "object",
                "properties": {"fact": {"type": "string"}},
                "required": ["fact"],
            },
            fn=lambda fact: (
                memory.add_fact(fact, tags=["auto"]) and ""  # type: ignore[func-returns-value]
            ) or "Saved",
        ),
    ]

    # Model calls save_fact then responds
    msg1 = FakeMessage(
        content=[FakeToolUseBlock(
            block_id="tu_1", name="save_fact",
            block_input={"fact": "User's name is Arthur"},
        )],
        stop_reason="tool_use",
    )
    msg2 = FakeMessage(
        content=[FakeTextBlock(text="Got it!")],
        stop_reason="end_turn",
    )

    responses = [("", msg1), ("Got it!", msg2)]
    agent = _make_agent(responses, tools=tools, memory=memory)
    list(agent.chat("My name is Arthur"))

    facts = memory.get_all_facts()
    assert len(facts) == 1
    assert "Arthur" in facts[0].content


# ---------------------------------------------------------------------------
# Tests — async chat (achat)
# ---------------------------------------------------------------------------


async def test_achat_basic():
    """Async chat returns text via async iteration."""
    responses = [
        ("Hello async!", FakeMessage(
            content=[FakeTextBlock(text="Hello async!")],
            stop_reason="end_turn",
        )),
    ]
    agent = _make_agent(responses, use_async=True)
    chunks: list[str] = []
    async for chunk in agent.achat("Hi"):
        chunks.append(chunk)
    assert "".join(chunks) == "Hello async!"


async def test_achat_with_memory():
    """Async chat records user and assistant messages in memory."""
    memory = MemoryManager(conversation_tokens=2000, tokenizer=_Tok())
    responses = [
        ("Hi!", FakeMessage(
            content=[FakeTextBlock(text="Hi!")],
            stop_reason="end_turn",
        )),
    ]
    agent = _make_agent(responses, memory=memory, use_async=True)
    chunks: list[str] = []
    async for chunk in agent.achat("Hello"):
        chunks.append(chunk)

    turns = memory.conversation.turns
    assert len(turns) == 2
    assert turns[0].role == "user"
    assert turns[0].content == "Hello"
    assert turns[1].role == "assistant"
    assert turns[1].content == "Hi!"


async def test_achat_tool_loop():
    """Async chat handles tool use loop."""
    tool_calls: list[str] = []

    def echo_tool(text: str) -> str:
        tool_calls.append(text)
        return f"Echo: {text}"

    tool = AgentTool(
        name="echo",
        description="Echoes text back",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        fn=echo_tool,
    )

    # First response: model calls tool (async stream)
    tool_use_msg = FakeMessage(
        content=[FakeToolUseBlock(block_id="tu_1", name="echo", block_input={"text": "hello"})],
        stop_reason="tool_use",
    )
    # Second response: model returns text (async stream)
    text_msg = FakeMessage(
        content=[FakeTextBlock(text="Done!")],
        stop_reason="end_turn",
    )

    responses = [("", tool_use_msg), ("Done!", text_msg)]
    agent = _make_agent(responses, tools=[tool], use_async=True)
    chunks: list[str] = []
    async for chunk in agent.achat("Test tool"):
        chunks.append(chunk)

    assert "".join(chunks) == "Done!"
    assert tool_calls == ["hello"]


# ---------------------------------------------------------------------------
# Tests — retry logic
# ---------------------------------------------------------------------------


def _ensure_mock_anthropic() -> types.ModuleType:
    """Ensure a mock anthropic module is available for retry tests.

    If the real ``anthropic`` package is installed, return it.
    Otherwise, create a minimal mock module with the exception
    classes needed by ``Agent._retryable_errors()``.
    """
    if "anthropic" in sys.modules:
        return sys.modules["anthropic"]

    mod = types.ModuleType("anthropic")

    class _APIError(Exception):
        """Base mock API error."""

    class RateLimitError(_APIError):
        pass

    class APIConnectionError(_APIError):
        pass

    class APITimeoutError(_APIError):
        pass

    mod.RateLimitError = RateLimitError  # type: ignore[attr-defined]
    mod.APIConnectionError = APIConnectionError  # type: ignore[attr-defined]
    mod.APITimeoutError = APITimeoutError  # type: ignore[attr-defined]
    sys.modules["anthropic"] = mod
    return mod


def test_retry_on_rate_limit():
    """Agent retries on RateLimitError with exponential backoff."""
    mock_anthropic = _ensure_mock_anthropic()

    call_count = [0]
    responses = [
        ("Success!", FakeMessage(
            content=[FakeTextBlock(text="Success!")],
            stop_reason="end_turn",
        )),
    ]
    client = FakeAnthropicClient(responses)

    # Wrap stream() to fail twice, then succeed
    original_stream = client.messages.stream

    def flaky_stream(**kwargs: Any) -> FakeStream:
        call_count[0] += 1
        if call_count[0] <= 2:
            raise mock_anthropic.RateLimitError("rate limited")  # type: ignore[attr-defined]
        return original_stream(**kwargs)

    client.messages.stream = flaky_stream  # type: ignore[assignment]

    agent = Agent(model="test-model", client=client, max_retries=3)
    agent.with_system_prompt("You are helpful.")

    with patch("astro_context.agent.agent.time.sleep") as mock_sleep:
        chunks = list(agent.chat("Hi"))

    assert "".join(chunks) == "Success!"
    assert call_count[0] == 3  # 2 failures + 1 success
    # Verify exponential backoff: sleep(1), sleep(2)
    assert mock_sleep.call_count == 2
    mock_sleep.assert_any_call(1)
    mock_sleep.assert_any_call(2)


# ---------------------------------------------------------------------------
# Tests — tool call memory recording
# ---------------------------------------------------------------------------


def test_tool_calls_recorded_in_memory():
    """Tool calls are recorded as tool messages in memory."""
    memory = MemoryManager(conversation_tokens=2000, tokenizer=_Tok())

    def greet(name: str) -> str:
        return f"Hello, {name}!"

    tool = AgentTool(
        name="greet",
        description="Greet someone",
        input_schema={
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
        fn=greet,
    )

    # First response: model calls tool
    tool_use_msg = FakeMessage(
        content=[FakeToolUseBlock(
            block_id="tu_1", name="greet", block_input={"name": "Alice"},
        )],
        stop_reason="tool_use",
    )
    # Second response: model returns text
    text_msg = FakeMessage(
        content=[FakeTextBlock(text="Done greeting!")],
        stop_reason="end_turn",
    )

    responses = [("", tool_use_msg), ("Done greeting!", text_msg)]
    agent = _make_agent(responses, tools=[tool], memory=memory)
    list(agent.chat("Greet Alice"))

    turns = memory.conversation.turns
    # Expect: user, tool, assistant
    assert len(turns) == 3
    assert turns[0].role == "user"
    assert turns[1].role == "tool"
    assert "[Tool: greet]" in turns[1].content
    assert "Alice" in turns[1].content
    assert "Hello, Alice!" in turns[1].content
    assert turns[2].role == "assistant"
    assert turns[2].content == "Done greeting!"
