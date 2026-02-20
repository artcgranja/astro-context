"""Anthropic Messages API formatter."""

from __future__ import annotations

from typing import Any

from astro_context.models.context import ContextWindow

from .utils import classify_window_items, get_message_role


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
        """Format the context window for the Anthropic Messages API.

        Returns a dict with a ``"system"`` key (joined system prompts) and
        a ``"messages"`` list.  Retrieval/tool context is prepended as a
        ``user`` message before the conversation history.
        """
        classified = classify_window_items(window)

        messages: list[dict[str, str]] = []

        for item in classified.memory_items:
            role = get_message_role(item)
            messages.append({"role": role, "content": item.content})

        if classified.context_parts:
            context_block = "Here is relevant context:\n\n" + "\n\n---\n\n".join(
                classified.context_parts
            )
            messages.insert(0, {"role": "user", "content": context_block})

        return {
            "system": "\n\n".join(classified.system_parts) if classified.system_parts else "",
            "messages": messages,
        }
