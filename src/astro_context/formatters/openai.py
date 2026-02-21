"""OpenAI Chat Completions API formatter."""

from __future__ import annotations

from typing import Any

from astro_context.models.context import ContextWindow

from .utils import classify_window_items, get_message_role


class OpenAIFormatter:
    """Formats context for the OpenAI Chat Completions API.

    Produces a dict with:
    - "messages": list of message dicts with role and content

    Security Note:
        Content from memory and retrieval items is inserted verbatim into
        the formatted output without sanitization.  If these items originate
        from untrusted sources (e.g. user-supplied documents, web scrapes),
        they may contain prompt injection payloads.  Callers should implement
        content validation or filtering *before* items enter the pipeline.
    """

    @property
    def format_type(self) -> str:
        return "openai"

    def format(self, window: ContextWindow) -> dict[str, Any]:
        """Format the context window for the OpenAI Chat Completions API.

        Returns a dict with a ``"messages"`` list.  System prompts are emitted
        as ``system`` messages.  Retrieval context is emitted as a ``user``
        message to prevent privilege escalation from untrusted content,
        followed by the conversation history from memory items.
        """
        classified = classify_window_items(window)

        messages: list[dict[str, str]] = []

        if classified.system_parts:
            messages.append({"role": "system", "content": "\n\n".join(classified.system_parts)})

        if classified.context_parts:
            context_block = "Relevant context:\n\n" + "\n\n---\n\n".join(
                classified.context_parts
            )
            # Security: retrieval content is untrusted and must NOT use the
            # "system" role.  Placing it in a "user" message prevents an
            # attacker-controlled document from gaining system-level authority.
            messages.append({"role": "user", "content": context_block})

        for item in classified.memory_items:
            role = get_message_role(item)
            messages.append({"role": role, "content": item.content})

        return {"messages": messages}
