"""Internal utilities shared across formatter implementations.

These helpers eliminate duplication in the concrete formatters.
They are *not* part of the public API and should not be imported
outside this package.
"""

from __future__ import annotations

from typing import Any, NamedTuple

from astro_context.models.context import ContextItem, ContextWindow, SourceType

_ALLOWED_ROLES: frozenset[str] = frozenset({"user", "assistant", "system", "tool"})
"""Roles accepted by the Anthropic and OpenAI message APIs."""

_SAFE_CONVERSATION_ROLES: frozenset[str] = frozenset({"user", "assistant"})
"""Roles that untrusted sources (CONVERSATION, MEMORY, RETRIEVAL) may use.

Allowing ``"system"`` or ``"tool"`` for these source types would let an
attacker who controls a memory entry or retrieved document escalate to
system-level authority in the formatted prompt.
"""


def get_message_role(item: ContextItem) -> str:
    """Return the validated chat role for a context item.

    Falls back to ``"user"`` when the metadata key is missing or
    contains a value outside :data:`_ALLOWED_ROLES`.

    **Security:** Roles are further restricted based on the item's
    :class:`SourceType` to prevent privilege escalation:

    * ``SourceType.SYSTEM`` -- may use ``"system"`` (plus ``"user"``/``"assistant"``).
    * ``SourceType.TOOL`` -- may use ``"tool"`` (plus ``"user"``/``"assistant"``).
    * All other sources (``CONVERSATION``, ``MEMORY``, ``RETRIEVAL``, ``USER``)
      -- restricted to ``"user"`` and ``"assistant"`` only.

    Disallowed roles are silently downgraded to ``"user"``.
    """
    role: str = str(item.metadata.get("role", "user"))
    if role not in _ALLOWED_ROLES:
        return "user"

    # Restrict roles based on source type to prevent escalation.
    if item.source == SourceType.SYSTEM:
        # System items may legitimately carry the "system" role.
        return role
    if item.source == SourceType.TOOL:
        # Tool items may carry the "tool" role.
        if role in (_SAFE_CONVERSATION_ROLES | {"tool"}):
            return role
        return "user"

    # For CONVERSATION, MEMORY, RETRIEVAL, USER sources: only safe roles.
    if role not in _SAFE_CONVERSATION_ROLES:
        return "user"
    return role


class ClassifiedItems(NamedTuple):
    """Result of partitioning a :class:`ContextWindow` by source type."""

    system_parts: list[str]
    memory_items: list[ContextItem]
    context_parts: list[str]


def classify_window_items(window: ContextWindow) -> ClassifiedItems:
    """Partition window items into system, memory, and context buckets.

    * **system_parts** -- content strings from :attr:`SourceType.SYSTEM` items.
    * **memory_items** -- full :class:`ContextItem` objects for
      :attr:`SourceType.MEMORY` and :attr:`SourceType.CONVERSATION`
      (callers typically need the metadata).  Both persistent memory
      facts and session-scoped conversation turns appear here so that
      formatters place them in the messages section.
    * **context_parts** -- content strings for every other source type
      (retrieval, tool, user, etc.).
    """
    system_parts: list[str] = []
    memory_items: list[ContextItem] = []
    context_parts: list[str] = []

    for item in window.items:
        if item.source == SourceType.SYSTEM:
            system_parts.append(item.content)
        elif item.source in (SourceType.MEMORY, SourceType.CONVERSATION):
            memory_items.append(item)
        else:
            context_parts.append(item.content)

    memory_items.sort(key=lambda item: item.created_at)
    return ClassifiedItems(system_parts, memory_items, context_parts)


def ensure_alternating_roles(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge consecutive same-role messages to enforce role alternation.

    LLM APIs (Anthropic, OpenAI) require that ``user`` and ``assistant``
    messages strictly alternate.  When a context block (``role="user"``)
    is prepended before conversation history that also starts with a
    ``user`` message, the result is two consecutive ``user`` messages
    which the API rejects.

    This function merges consecutive messages that share the same role
    by joining their ``content`` fields with ``"\\n\\n"``.  Messages with
    the ``"system"`` role are never merged -- they are passed through
    unchanged because system messages do not participate in the
    user/assistant alternation requirement.

    Parameters
    ----------
    messages:
        List of message dicts, each with at least ``"role"`` and
        ``"content"`` keys.  May contain additional keys (e.g.
        ``"cache_control"``); extra keys from the *first* message in a
        merged group are preserved.

    Returns
    -------
    list[dict[str, Any]]
        A new list with consecutive same-role non-system messages merged.
    """
    if len(messages) <= 1:
        return list(messages)

    merged: list[dict[str, Any]] = []
    for msg in messages:
        if (
            merged
            and msg["role"] != "system"
            and merged[-1]["role"] == msg["role"]
        ):
            # Merge into the previous message, preserving extra keys
            # from the first message in the group.
            merged[-1] = {
                **merged[-1],
                "content": merged[-1]["content"] + "\n\n" + msg["content"],
            }
        else:
            merged.append(dict(msg))
    return merged
