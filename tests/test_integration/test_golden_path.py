"""Golden path integration test mirroring the README usage pattern.

Validates the end-to-end pipeline: system prompts, conversation memory,
Anthropic formatting, and diagnostics -- all with FakeTokenizer to avoid
tiktoken dependency.
"""

from __future__ import annotations

from astro_context.formatters.anthropic import AnthropicFormatter
from astro_context.memory.manager import MemoryManager
from astro_context.models.context import ContextResult, SourceType
from astro_context.pipeline.pipeline import ContextPipeline
from tests.conftest import FakeTokenizer


class TestGoldenPath:
    """Full pipeline integration test matching README usage."""

    def test_full_pipeline_with_memory_and_anthropic_formatter(self) -> None:
        tokenizer = FakeTokenizer()

        # 1. Create pipeline
        pipeline = ContextPipeline(max_tokens=8192, tokenizer=tokenizer)

        # 2. Add a system prompt
        pipeline.add_system_prompt("You are a helpful assistant.")

        # 3. Create memory manager and add conversation turns
        memory = MemoryManager(conversation_tokens=4096, tokenizer=tokenizer)
        memory.add_user_message("What is context engineering?")
        memory.add_assistant_message(
            "Context engineering is the art of assembling the right "
            "information for LLMs to produce high-quality outputs."
        )
        memory.add_user_message("How does it differ from prompt engineering?")

        # 4. Attach memory and formatter
        pipeline.with_memory(memory)
        pipeline.with_formatter(AnthropicFormatter())

        # 5. Build context
        result = pipeline.build("What is context engineering?")

        # 6. Assertions on result type
        assert isinstance(result, ContextResult)

        # 7. formatted_output should be a dict with "system" and "messages" keys
        assert isinstance(result.formatted_output, dict)
        assert "system" in result.formatted_output
        assert "messages" in result.formatted_output

        # 8. diagnostics should have timing info
        assert "steps" in result.diagnostics
        assert result.build_time_ms >= 0

        # 9. window.items should contain both system and memory items
        sources = {item.source for item in result.window.items}
        assert SourceType.SYSTEM in sources
        # Conversation items use CONVERSATION or MEMORY source
        assert SourceType.CONVERSATION in sources or SourceType.MEMORY in sources

        # 10. overflow_items should be empty (we have plenty of budget)
        assert result.overflow_items == []

    def test_system_prompt_appears_first_in_context(self) -> None:
        tokenizer = FakeTokenizer()
        pipeline = ContextPipeline(max_tokens=8192, tokenizer=tokenizer)
        pipeline.add_system_prompt("System instruction here.")

        memory = MemoryManager(conversation_tokens=4096, tokenizer=tokenizer)
        memory.add_user_message("Hello")

        pipeline.with_memory(memory)
        pipeline.with_formatter(AnthropicFormatter())

        result = pipeline.build("test query")

        # System items have priority=10, conversation has priority=7
        # System should appear before conversation in the window
        system_items = [
            i for i in result.window.items if i.source == SourceType.SYSTEM
        ]
        assert len(system_items) >= 1
        assert system_items[0].content == "System instruction here."

    def test_format_type_is_anthropic(self) -> None:
        tokenizer = FakeTokenizer()
        pipeline = (
            ContextPipeline(max_tokens=8192, tokenizer=tokenizer)
            .with_formatter(AnthropicFormatter())
            .add_system_prompt("Hi")
        )

        result = pipeline.build("test")
        assert result.format_type == "anthropic"

    def test_diagnostics_include_items_included_count(self) -> None:
        tokenizer = FakeTokenizer()

        memory = MemoryManager(conversation_tokens=4096, tokenizer=tokenizer)
        memory.add_user_message("First message")
        memory.add_assistant_message("Response")

        pipeline = (
            ContextPipeline(max_tokens=8192, tokenizer=tokenizer)
            .with_memory(memory)
            .add_system_prompt("System prompt")
        )

        result = pipeline.build("query")

        # Should report at least 3 items: 1 system + 2 conversation
        assert result.diagnostics["items_included"] >= 3

    def test_plain_string_query_works(self) -> None:
        """build() accepts a plain string -- no QueryBundle needed."""
        tokenizer = FakeTokenizer()
        pipeline = (
            ContextPipeline(max_tokens=8192, tokenizer=tokenizer)
            .add_system_prompt("System")
        )

        result = pipeline.build("plain string query")
        assert isinstance(result, ContextResult)
        assert len(result.window.items) >= 1
