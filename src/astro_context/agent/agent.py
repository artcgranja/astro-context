"""Agent class that wraps ContextPipeline + Anthropic SDK + tool loop."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator, Iterator
from typing import Any

from astro_context.formatters.anthropic import AnthropicFormatter
from astro_context.memory.manager import MemoryManager
from astro_context.models.context import ContextResult
from astro_context.pipeline.pipeline import ContextPipeline

from .tools import AgentTool

logger = logging.getLogger(__name__)

# Maximum character length for tool input/result recorded in memory.
_TOOL_MEMORY_TRUNCATE = 200


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
        "_max_retries",
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
        max_retries: int = 3,
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
        self._max_retries = max_retries
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
        """Look up and execute a tool by name.

        Validates input against the tool's schema before calling.
        """
        for tool in self._tools:
            if tool.name == name:
                valid, err = tool.validate_input(tool_input)
                if not valid:
                    logger.warning(
                        "Tool '%s' input validation failed: %s", name, err,
                    )
                    return f"Error: invalid input for tool '{name}': {err}"
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
                self._record_tool_call(block.name, block.input, result_text)
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_text,
                })
        return results

    def _record_tool_call(
        self, name: str, tool_input: dict[str, Any], result: str,
    ) -> None:
        """Record a tool call in memory for conversation history."""
        if self._memory is not None:
            input_str = json.dumps(tool_input)[:_TOOL_MEMORY_TRUNCATE]
            result_str = result[:_TOOL_MEMORY_TRUNCATE]
            tool_summary = (
                f"[Tool: {name}] Input: {input_str} → Result: {result_str}"
            )
            self._memory.add_tool_message(tool_summary)

    @staticmethod
    def _retryable_errors() -> tuple[type[Exception], ...]:
        """Return the tuple of retryable Anthropic exception types.

        Returns an empty tuple when the ``anthropic`` package is not installed,
        which means the retry loop will never catch anything (correct behaviour
        for test environments that use a fake client).
        """
        try:
            import anthropic as _anthropic
        except ImportError:
            return ()
        return (
            _anthropic.RateLimitError,
            _anthropic.APIConnectionError,
            _anthropic.APITimeoutError,
        )

    def _call_api_with_retry(self, **kwargs: Any) -> Any:
        """Call the streaming API with exponential backoff on transient errors.

        Retries on ``RateLimitError``, ``APIConnectionError``, and
        ``APITimeoutError`` up to ``self._max_retries`` times with
        exponential backoff (1s, 2s, 4s, ...).
        """
        retryable = self._retryable_errors()
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                return self._client.messages.stream(**kwargs)
            except retryable as exc:
                last_exc = exc
                delay = 2**attempt  # 1, 2, 4, ...
                logger.warning(
                    "API call failed (attempt %d/%d): %s. Retrying in %ds...",
                    attempt + 1, self._max_retries, exc, delay,
                )
                time.sleep(delay)
        if last_exc is not None:
            raise last_exc
        msg = "max_retries must be >= 1"
        raise ValueError(msg)

    async def _call_api_with_retry_async(self, **kwargs: Any) -> Any:
        """Async variant of ``_call_api_with_retry``."""
        import asyncio

        retryable = self._retryable_errors()
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                return self._client.messages.stream(**kwargs)
            except retryable as exc:
                last_exc = exc
                delay = 2**attempt
                logger.warning(
                    "Async API call failed (attempt %d/%d): %s. Retrying in %ds...",
                    attempt + 1, self._max_retries, exc, delay,
                )
                await asyncio.sleep(delay)
        if last_exc is not None:
            raise last_exc
        msg = "max_retries must be >= 1"
        raise ValueError(msg)

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

            with self._call_api_with_retry(**kwargs) as stream:
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

    async def achat(self, message: str) -> AsyncIterator[str]:
        """Send a message and stream the response asynchronously.

        Async mirror of :meth:`chat`. Uses ``pipeline.abuild()`` and
        async iteration over the streaming API.

        Yields text chunks as they arrive from the API.
        """
        if self._memory is not None:
            self._memory.add_user_message(message)

        ctx_result = await self._pipeline.abuild(message)
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

            stream_ctx = await self._call_api_with_retry_async(**kwargs)
            async with stream_ctx as stream:
                async for text in stream.text_stream:
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
