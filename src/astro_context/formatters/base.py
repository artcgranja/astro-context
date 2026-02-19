"""Formatter protocol definition.

Any object with ``format_type`` and ``format()`` matching this interface
can be used as a formatter -- no inheritance required (PEP 544 structural subtyping).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from astro_context.models.context import ContextWindow


@runtime_checkable
class BaseFormatter(Protocol):
    """Protocol for context formatters.

    Formatters convert a ContextWindow into the format expected
    by a specific LLM provider (Anthropic, OpenAI, etc.) or
    a generic text format.

    This is a Protocol (not an ABC): any class with matching
    ``format_type`` and ``format()`` satisfies it via structural subtyping.
    """

    @property
    def format_type(self) -> str:
        """Identifier for this format type (e.g., 'anthropic', 'openai')."""
        ...

    def format(self, window: ContextWindow) -> str | dict[str, Any]:
        """Format the context window into the target format."""
        ...
