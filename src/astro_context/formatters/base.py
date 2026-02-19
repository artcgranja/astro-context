"""Base formatter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from astro_context.models.context import ContextWindow


class BaseFormatter(ABC):
    """Abstract base class for context formatters.

    Formatters convert a ContextWindow into the format expected
    by a specific LLM provider (Anthropic, OpenAI, etc.) or
    a generic text format.
    """

    @property
    @abstractmethod
    def format_type(self) -> str:
        """Identifier for this format type (e.g., 'anthropic', 'openai')."""
        ...

    @abstractmethod
    def format(self, window: ContextWindow) -> str | dict[str, Any]:
        """Format the context window into the target format."""
        ...
