"""Anthropic Messages API formatter."""

from __future__ import annotations

from typing import Any

from astro_context.models.context import ContextWindow

from .utils import classify_window_items, get_message_role


class AnthropicFormatter:
    """Formats context for the Anthropic Messages API.

    Produces a dict with:
    - "system": list of content blocks (Anthropic API accepts both string and list)
    - "messages": list of message dicts

    Parameters
    ----------
    enable_caching:
        When ``True``, adds ``"cache_control": {"type": "ephemeral"}`` to the
        last system content block and the last retrieval/context message.  This
        enables Anthropic prompt caching to reduce latency and cost for
        repeated prefixes.
    """

    def __init__(self, *, enable_caching: bool = False) -> None:
        self._enable_caching = enable_caching

    @property
    def format_type(self) -> str:
        return "anthropic"

    def format(self, window: ContextWindow) -> dict[str, Any]:
        """Format the context window for the Anthropic Messages API.

        Returns a dict with a ``"system"`` key (list of content blocks) and
        a ``"messages"`` list.  Retrieval/tool context is prepended as a
        ``user`` message before the conversation history.
        """
        classified = classify_window_items(window)

        # Build system content blocks
        system_blocks: list[dict[str, Any]] = []
        if classified.system_parts:
            system_text = "\n\n".join(classified.system_parts)
            block: dict[str, Any] = {"type": "text", "text": system_text}
            if self._enable_caching:
                block["cache_control"] = {"type": "ephemeral"}
            system_blocks.append(block)

        # Build messages
        messages: list[dict[str, Any]] = []

        for item in classified.memory_items:
            role = get_message_role(item)
            messages.append({"role": role, "content": item.content})

        if classified.context_parts:
            context_block = "Here is relevant context:\n\n" + "\n\n---\n\n".join(
                classified.context_parts
            )
            context_msg: dict[str, Any] = {"role": "user", "content": context_block}
            if self._enable_caching:
                context_msg["cache_control"] = {"type": "ephemeral"}
            messages.insert(0, context_msg)

        return {
            "system": system_blocks,
            "messages": messages,
        }
