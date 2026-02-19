"""Anthropic Messages API formatter."""

from __future__ import annotations

from typing import Any

from astro_context.models.context import ContextWindow, SourceType


class AnthropicFormatter:
    """Formats context for the Anthropic Messages API.

    Produces a dict with:
    - "system": system prompt string
    - "messages": list of message dicts
    """

    @property
    def format_type(self) -> str:
        return "anthropic"

    def format(self, window: ContextWindow) -> dict[str, Any]:
        system_parts: list[str] = []
        messages: list[dict[str, str]] = []
        context_parts: list[str] = []

        for item in window.items:
            if item.source == SourceType.SYSTEM:
                system_parts.append(item.content)
            elif item.source == SourceType.MEMORY:
                role = item.metadata.get("role", "user")
                messages.append({"role": role, "content": item.content})
            else:
                context_parts.append(item.content)

        if context_parts:
            context_block = "Here is relevant context:\n\n" + "\n\n---\n\n".join(context_parts)
            messages.insert(0, {"role": "user", "content": context_block})

        return {
            "system": "\n\n".join(system_parts) if system_parts else "",
            "messages": messages,
        }
