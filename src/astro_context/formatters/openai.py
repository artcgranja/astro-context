"""OpenAI Chat Completions API formatter."""

from __future__ import annotations

from typing import Any

from astro_context.models.context import ContextWindow, SourceType


class OpenAIFormatter:
    """Formats context for the OpenAI Chat Completions API.

    Produces a dict with:
    - "messages": list of message dicts with role and content
    """

    @property
    def format_type(self) -> str:
        return "openai"

    def format(self, window: ContextWindow) -> dict[str, Any]:
        messages: list[dict[str, str]] = []
        system_parts: list[str] = []
        context_parts: list[str] = []
        conversation: list[dict[str, str]] = []

        for item in window.items:
            if item.source == SourceType.SYSTEM:
                system_parts.append(item.content)
            elif item.source == SourceType.MEMORY:
                role = item.metadata.get("role", "user")
                conversation.append({"role": role, "content": item.content})
            else:
                context_parts.append(item.content)

        if system_parts:
            messages.append({"role": "system", "content": "\n\n".join(system_parts)})

        if context_parts:
            context_block = "Relevant context:\n\n" + "\n\n---\n\n".join(context_parts)
            messages.append({"role": "system", "content": context_block})

        messages.extend(conversation)

        return {"messages": messages}
