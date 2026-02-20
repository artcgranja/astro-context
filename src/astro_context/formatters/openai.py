"""OpenAI Chat Completions API formatter."""

from __future__ import annotations

from typing import Any

from astro_context.models.context import ContextWindow

from .utils import classify_window_items, get_message_role


class OpenAIFormatter:
    """Formats context for the OpenAI Chat Completions API.

    Produces a dict with:
    - "messages": list of message dicts with role and content
    """

    @property
    def format_type(self) -> str:
        return "openai"

    def format(self, window: ContextWindow) -> dict[str, Any]:
        """Format the context window for the OpenAI Chat Completions API.

        Returns a dict with a ``"messages"`` list.  System prompts and
        retrieval context are emitted as ``system`` messages, followed by the
        conversation history from memory items.
        """
        classified = classify_window_items(window)

        messages: list[dict[str, str]] = []

        if classified.system_parts:
            messages.append({"role": "system", "content": "\n\n".join(classified.system_parts)})

        if classified.context_parts:
            context_block = "Relevant context:\n\n" + "\n\n---\n\n".join(
                classified.context_parts
            )
            messages.append({"role": "system", "content": context_block})

        for item in classified.memory_items:
            role = get_message_role(item)
            messages.append({"role": role, "content": item.content})

        return {"messages": messages}
