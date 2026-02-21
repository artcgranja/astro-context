"""Integration tests for Agent + skills system."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest

from astro_context.agent.agent import Agent
from astro_context.agent.skills.models import Skill
from astro_context.agent.tools import AgentTool

# ---------------------------------------------------------------------------
# Fake Anthropic types (mirrors from test_agent.py)
# ---------------------------------------------------------------------------


class FakeTextBlock:
    def __init__(self, text: str) -> None:
        self.text = text
        self.type = "text"


class FakeToolUseBlock:
    def __init__(self, block_id: str, name: str, block_input: dict[str, Any]) -> None:
        self.id = block_id
        self.name = name
        self.input = block_input
        self.type = "tool_use"


class FakeMessage:
    def __init__(self, content: list[Any], stop_reason: str = "end_turn") -> None:
        self.content = content
        self.stop_reason = stop_reason


class FakeStream:
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
    def __init__(
        self, responses: list[tuple[str, FakeMessage]], *, use_async: bool = False,
    ) -> None:
        self._responses = list(responses)
        self._call_index = 0
        self._use_async = use_async
        self.last_kwargs: dict[str, Any] = {}

    def stream(self, **kwargs: Any) -> FakeStream | FakeAsyncStream:
        self.last_kwargs = kwargs
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


class FakeClient:
    def __init__(
        self, responses: list[tuple[str, FakeMessage]], *, use_async: bool = False,
    ) -> None:
        self.messages = FakeMessages(responses, use_async=use_async)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _noop() -> str:
    return "ok"


def _make_tool(name: str) -> AgentTool:
    return AgentTool(
        name=name, description=f"Tool {name}",
        input_schema={"type": "object", "properties": {}}, fn=_noop,
    )


def _always_skill(name: str = "mem", tool_name: str = "save_fact") -> Skill:
    return Skill(
        name=name, description=f"{name} skill",
        tools=(_make_tool(tool_name),), activation="always",
    )


def _on_demand_skill(
    name: str = "rag", tool_name: str = "search_docs",
) -> Skill:
    return Skill(
        name=name, description=f"{name} skill",
        instructions=f"Use {tool_name} to search.",
        tools=(_make_tool(tool_name),), activation="on_demand",
    )


def _make_agent(
    responses: list[tuple[str, FakeMessage]], **kwargs: Any,
) -> Agent:
    client = FakeClient(responses, use_async=kwargs.pop("use_async", False))
    agent = Agent(model="test-model", client=client, **kwargs)
    agent.with_system_prompt("You are helpful.")
    return agent


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAlwaysLoadedSkill:
    def test_tools_available_immediately(self) -> None:
        agent = _make_agent([("Hi", FakeMessage([FakeTextBlock("Hi")]))])
        agent.with_skill(_always_skill("mem", "save_fact"))

        # Tools should include skill's tools
        all_tools = agent._all_active_tools()
        names = [t.name for t in all_tools]
        assert "save_fact" in names

    def test_chat_includes_skill_tools(self) -> None:
        msg = FakeMessage([FakeTextBlock("Hello!")])
        agent = _make_agent([("Hello!", msg)])
        agent.with_skill(_always_skill("mem", "save_fact"))

        result = "".join(agent.chat("Hi"))
        assert result == "Hello!"

        # Verify tools were sent to the API
        kwargs = agent._client.messages.last_kwargs
        tool_names = [t["name"] for t in kwargs["tools"]]
        assert "save_fact" in tool_names


class TestOnDemandSkill:
    def test_tools_not_available_initially(self) -> None:
        agent = _make_agent([("Hi", FakeMessage([FakeTextBlock("Hi")]))])
        agent.with_skill(_on_demand_skill("rag", "search_docs"))

        all_tools = agent._all_active_tools()
        tool_names = [t.name for t in all_tools]
        assert "search_docs" not in tool_names
        assert "activate_skill" in tool_names

    def test_activate_skill_meta_tool_injected(self) -> None:
        agent = _make_agent([("Hi", FakeMessage([FakeTextBlock("Hi")]))])
        agent.with_skill(_on_demand_skill())

        all_tools = agent._all_active_tools()
        names = [t.name for t in all_tools]
        assert "activate_skill" in names

    def test_no_activate_tool_when_only_always_skills(self) -> None:
        agent = _make_agent([("Hi", FakeMessage([FakeTextBlock("Hi")]))])
        agent.with_skill(_always_skill())

        all_tools = agent._all_active_tools()
        names = [t.name for t in all_tools]
        assert "activate_skill" not in names

    def test_activation_mid_conversation(self) -> None:
        """Simulate: round 1 = agent calls activate_skill, round 2 = agent uses new tool."""
        # Round 1: model calls activate_skill
        activate_call = FakeToolUseBlock(
            "tool_1", "activate_skill", {"skill_name": "rag"},
        )
        round1_msg = FakeMessage(
            [FakeTextBlock("Let me enable that."), activate_call],
            stop_reason="tool_use",
        )
        # Round 2: model uses the now-available search_docs
        search_call = FakeToolUseBlock("tool_2", "search_docs", {})
        round2_msg = FakeMessage(
            [search_call], stop_reason="tool_use",
        )
        # Round 3: final text response
        round3_msg = FakeMessage([FakeTextBlock("Here are the results.")])

        agent = _make_agent([
            ("Let me enable that.", round1_msg),
            ("", round2_msg),
            ("Here are the results.", round3_msg),
        ])
        agent.with_skill(_on_demand_skill("rag", "search_docs"))

        result = "".join(agent.chat("Find docs about pipeline"))
        assert "results" in result

        # After activation, the skill should be active
        assert agent._skill_registry.is_active("rag")


class TestMixedToolsAndSkills:
    def test_direct_tools_and_skills_coexist(self) -> None:
        agent = _make_agent([("Hi", FakeMessage([FakeTextBlock("Hi")]))])
        direct_tool = _make_tool("my_direct_tool")
        agent.with_tools([direct_tool])
        agent.with_skill(_always_skill("mem", "save_fact"))

        all_tools = agent._all_active_tools()
        names = [t.name for t in all_tools]
        assert "my_direct_tool" in names
        assert "save_fact" in names

    def test_with_tools_backward_compat(self) -> None:
        """The old with_tools() API still works unchanged."""
        agent = _make_agent([("Hi", FakeMessage([FakeTextBlock("Hi")]))])
        tool = _make_tool("legacy_tool")
        agent.with_tools([tool])

        result = "".join(agent.chat("Hi"))
        assert result == "Hi"

        kwargs = agent._client.messages.last_kwargs
        tool_names = [t["name"] for t in kwargs["tools"]]
        assert "legacy_tool" in tool_names


class TestWithSkillsFluent:
    def test_with_skills_registers_multiple(self) -> None:
        agent = _make_agent([("Hi", FakeMessage([FakeTextBlock("Hi")]))])
        agent.with_skills([
            _always_skill("a", "tool_a"),
            _always_skill("b", "tool_b"),
        ])
        tools = agent._all_active_tools()
        names = [t.name for t in tools]
        assert "tool_a" in names
        assert "tool_b" in names

    def test_chaining_works(self) -> None:
        agent = _make_agent([("Hi", FakeMessage([FakeTextBlock("Hi")]))])
        result = agent.with_skill(_always_skill("a", "t1")).with_skill(
            _on_demand_skill("b", "t2"),
        )
        assert result is agent


class TestDiscoveryPromptInSystem:
    def test_discovery_appended_when_on_demand_exists(self) -> None:
        msg = FakeMessage([FakeTextBlock("Hi")])
        agent = _make_agent([("Hi", msg)])
        agent.with_skill(_on_demand_skill("rag", "search_docs"))

        "".join(agent.chat("Hello"))

        kwargs = agent._client.messages.last_kwargs
        system = kwargs["system"]
        # Last system block should contain the discovery prompt
        last_block = system[-1]
        assert "activate_skill" in last_block["text"]
        assert "rag" in last_block["text"]

    def test_no_discovery_when_only_always(self) -> None:
        msg = FakeMessage([FakeTextBlock("Hi")])
        agent = _make_agent([("Hi", msg)])
        agent.with_skill(_always_skill("mem", "save_fact"))

        "".join(agent.chat("Hello"))

        kwargs = agent._client.messages.last_kwargs
        system = kwargs["system"]
        # Should be the original system blocks, no discovery appended
        for block in system:
            if isinstance(block, dict) and "text" in block:
                assert "activate_skill" not in block["text"]


class TestAsyncSkills:
    async def test_achat_with_always_skill(self) -> None:
        msg = FakeMessage([FakeTextBlock("Async hi")])
        agent = _make_agent([("Async hi", msg)], use_async=True)
        agent.with_skill(_always_skill("mem", "save_fact"))

        chunks: list[str] = []
        async for chunk in agent.achat("Hello"):
            chunks.append(chunk)
        assert "".join(chunks) == "Async hi"

        kwargs = agent._client.messages.last_kwargs
        tool_names = [t["name"] for t in kwargs["tools"]]
        assert "save_fact" in tool_names


class TestDuplicateSkillRegistration:
    def test_duplicate_raises(self) -> None:
        agent = _make_agent([("Hi", FakeMessage([FakeTextBlock("Hi")]))])
        agent.with_skill(_always_skill("dup", "t1"))
        with pytest.raises(ValueError, match="already registered"):
            agent.with_skill(_always_skill("dup", "t2"))
