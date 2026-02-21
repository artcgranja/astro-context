"""Agent class that wraps ContextPipeline + Anthropic SDK + tool loop."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

from astro_context.formatters.anthropic import AnthropicFormatter
from astro_context.memory.manager import MemoryManager
from astro_context.models.context import ContextResult
from astro_context.pipeline.pipeline import ContextPipeline

from .tools import AgentTool

logger = logging.getLogger(__name__)


class _WhitespaceTokenizer:
    """Minimal tokenizer that counts whitespace-separated words.

    Used internally by Agent to avoid tiktoken's network dependency.
    """

    __slots__ = ()

    def count_tokens(self, text: str) -> int:
        if not text or not text.strip():
            return 0
        return len(text.split())

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        return " ".join(text.split()[:max_tokens])


class Agent:
    """High-level agent combining context pipeline with Anthropic's API.

    Provides streaming chat with automatic tool use, memory management,
    and agentic RAG -- all powered by the astro-context pipeline.

    Usage::

        agent = (
            Agent(model="claude-haiku-4-5-20251001")
            .with_system_prompt("You are a helpful assistant.")
            .with_memory(memory)
            .with_tools(memory_tools(memory))
        )

        for chunk in agent.chat("Hello!"):
            print(chunk, end="", flush=True)
    """

    __slots__ = (
        "_client",
        "_last_result",
        "_max_response_tokens",
        "_max_rounds",
        "_memory",
        "_model",
        "_pipeline",
        "_system_prompt",
        "_tools",
    )

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        client: Any = None,
        max_tokens: int = 16384,
        max_response_tokens: int = 1024,
        max_rounds: int = 10,
    ) -> None:
        if client is not None:
            self._client: Any = client
        else:
            try:
                import anthropic
            except ImportError:
                msg = (
                    "anthropic is required for Agent. "
                    "Install with: pip install astro-context[anthropic]"
                )
                raise ImportError(msg) from None
            self._client = anthropic.Anthropic(api_key=api_key)

        self._model = model
        self._max_response_tokens = max_response_tokens
        self._max_rounds = max_rounds
        self._system_prompt = ""
        self._tools: list[AgentTool] = []
        self._memory: MemoryManager | None = None
        self._last_result: ContextResult | None = None

        tokenizer = _WhitespaceTokenizer()
        self._pipeline = ContextPipeline(max_tokens=max_tokens, tokenizer=tokenizer)
        self._pipeline.with_formatter(AnthropicFormatter())

    # -- Fluent configuration (all return self) --

    def with_system_prompt(self, prompt: str) -> Agent:
        """Set the system prompt. Returns self for chaining."""
        self._system_prompt = prompt
        self._pipeline._system_items.clear()
        self._pipeline.add_system_prompt(prompt)
        return self

    def with_memory(self, memory: MemoryManager) -> Agent:
        """Attach memory for conversation history and facts. Returns self for chaining."""
        self._memory = memory
        self._pipeline.with_memory(memory)
        return self

    def with_tools(self, tools: list[AgentTool]) -> Agent:
        """Add tools (additive). Returns self for chaining."""
        self._tools.extend(tools)
        return self

    # -- Accessors --

    @property
    def memory(self) -> MemoryManager | None:
        """The attached memory manager, if any."""
        return self._memory

    @property
    def pipeline(self) -> ContextPipeline:
        """The underlying context pipeline."""
        return self._pipeline

    @property
    def last_result(self) -> ContextResult | None:
        """The ContextResult from the most recent ``chat()`` call."""
        return self._last_result

    # -- Internal helpers --

    def _execute_tool(self, name: str, tool_input: dict[str, Any]) -> str:
        """Look up and execute a tool by name."""
        for tool in self._tools:
            if tool.name == name:
                try:
                    return tool.fn(**tool_input)
                except Exception:
                    logger.exception("Tool '%s' failed", name)
                    return f"Error: tool '{name}' failed."
        return f"Unknown tool: {name}"

    @staticmethod
    def _serialize_response(content: list[Any]) -> list[dict[str, Any]]:
        """Serialize response content blocks to dicts for the next request."""
        blocks: list[dict[str, Any]] = []
        for block in content:
            if block.type == "text":
                blocks.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                blocks.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })
        return blocks

    def _run_tools(self, content: list[Any]) -> list[dict[str, Any]]:
        """Execute all tool_use blocks and return tool_result dicts."""
        results: list[dict[str, Any]] = []
        for block in content:
            if block.type == "tool_use":
                result_text = self._execute_tool(block.name, block.input)
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_text,
                })
        return results

    # -- Chat --

    def chat(self, message: str) -> Iterator[str]:
        """Send a message and stream the response.

        Handles the full tool-use loop: if the model calls tools,
        they are executed and results fed back until the model
        produces a final text response or ``max_rounds`` is reached.

        Yields text chunks as they arrive from the API.
        """
        if self._memory is not None:
            self._memory.add_user_message(message)

        ctx_result = self._pipeline.build(message)
        self._last_result = ctx_result
        formatted = ctx_result.formatted_output
        if not isinstance(formatted, dict):
            msg = "Agent requires AnthropicFormatter (dict output), got str"
            raise TypeError(msg)

        system: Any = formatted["system"]
        messages: list[Any] = list(formatted["messages"])
        tools_param = (
            [t.to_anthropic_schema() for t in self._tools] if self._tools else None
        )

        final_text = ""

        for _round in range(self._max_rounds):
            kwargs: dict[str, Any] = {
                "model": self._model,
                "max_tokens": self._max_response_tokens,
                "system": system,
                "messages": messages,
            }
            if tools_param:
                kwargs["tools"] = tools_param

            with self._client.messages.stream(**kwargs) as stream:
                for text in stream.text_stream:
                    final_text += text
                    yield text
                response = stream.get_final_message()

            if response.stop_reason != "tool_use":
                break

            messages.append({
                "role": "assistant",
                "content": self._serialize_response(response.content),
            })
            messages.append({"role": "user", "content": self._run_tools(response.content)})

        if self._memory is not None and final_text:
            self._memory.add_assistant_message(final_text)
