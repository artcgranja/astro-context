"""Tests for astro_context.formatters.generic."""

from __future__ import annotations

from astro_context.formatters.generic import GenericTextFormatter
from astro_context.models.context import ContextItem, ContextWindow, SourceType


class TestGenericTextFormatter:
    """GenericTextFormatter produces structured text with section headers."""

    def test_format_type(self) -> None:
        formatter = GenericTextFormatter()
        assert formatter.format_type == "generic"

    def test_system_section(self) -> None:
        window = ContextWindow(max_tokens=10000)
        window.add_item(
            ContextItem(content="You are helpful.", source=SourceType.SYSTEM, token_count=5)
        )
        formatter = GenericTextFormatter()
        output = formatter.format(window)
        assert "=== SYSTEM ===" in output
        assert "You are helpful." in output

    def test_memory_section(self) -> None:
        window = ContextWindow(max_tokens=10000)
        window.add_item(
            ContextItem(content="user: Hello", source=SourceType.MEMORY, token_count=5)
        )
        formatter = GenericTextFormatter()
        output = formatter.format(window)
        assert "=== MEMORY ===" in output
        assert "user: Hello" in output

    def test_context_section_for_retrieval(self) -> None:
        window = ContextWindow(max_tokens=10000)
        window.add_item(
            ContextItem(content="doc about Python", source=SourceType.RETRIEVAL, token_count=5)
        )
        formatter = GenericTextFormatter()
        output = formatter.format(window)
        assert "=== CONTEXT ===" in output
        assert "doc about Python" in output

    def test_context_section_for_tool_source(self) -> None:
        window = ContextWindow(max_tokens=10000)
        window.add_item(
            ContextItem(content="tool output", source=SourceType.TOOL, token_count=5)
        )
        formatter = GenericTextFormatter()
        output = formatter.format(window)
        assert "=== CONTEXT ===" in output
        assert "tool output" in output

    def test_multiple_sections(self) -> None:
        window = ContextWindow(max_tokens=10000)
        window.add_item(
            ContextItem(content="system msg", source=SourceType.SYSTEM, token_count=5)
        )
        window.add_item(
            ContextItem(content="memory msg", source=SourceType.MEMORY, token_count=5)
        )
        window.add_item(
            ContextItem(content="retrieved doc", source=SourceType.RETRIEVAL, token_count=5)
        )
        formatter = GenericTextFormatter()
        output = formatter.format(window)
        assert "=== SYSTEM ===" in output
        assert "=== MEMORY ===" in output
        assert "=== CONTEXT ===" in output

    def test_empty_window(self) -> None:
        window = ContextWindow(max_tokens=10000)
        formatter = GenericTextFormatter()
        output = formatter.format(window)
        assert output == ""

    def test_multiple_items_in_same_section(self) -> None:
        window = ContextWindow(max_tokens=10000)
        window.add_item(
            ContextItem(content="doc one", source=SourceType.RETRIEVAL, token_count=5)
        )
        window.add_item(
            ContextItem(content="doc two", source=SourceType.RETRIEVAL, token_count=5)
        )
        formatter = GenericTextFormatter()
        output = formatter.format(window)
        assert "doc one" in output
        assert "doc two" in output
        # Only one CONTEXT header
        assert output.count("=== CONTEXT ===") == 1
