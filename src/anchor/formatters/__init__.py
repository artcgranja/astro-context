"""Output formatters for different LLM providers."""

from .anthropic import AnthropicFormatter
from .base import Formatter
from .generic import GenericTextFormatter
from .openai import OpenAIFormatter

# Backward-compatible alias (deprecated; prefer ``Formatter``).
BaseFormatter = Formatter

__all__ = [
    "AnthropicFormatter",
    "BaseFormatter",
    "Formatter",
    "GenericTextFormatter",
    "OpenAIFormatter",
]
