"""Internal utilities shared across formatter implementations.

These helpers eliminate duplication in the concrete formatters.
They are *not* part of the public API and should not be imported
outside this package.
"""

from __future__ import annotations

from typing import NamedTuple

from astro_context.models.context import ContextItem, ContextWindow, SourceType

_ALLOWED_ROLES: frozenset[str] = frozenset({"user", "assistant", "system"})
"""Roles accepted by the Anthropic and OpenAI message APIs."""


def get_message_role(item: ContextItem) -> str:
    """Return the validated chat role for a context item.

    Falls back to ``"user"`` when the metadata key is missing or
    contains a value outside :data:`_ALLOWED_ROLES`.
    """
    role: str = str(item.metadata.get("role", "user"))
    if role not in _ALLOWED_ROLES:
        role = "user"
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
      :attr:`SourceType.MEMORY` (callers typically need the metadata).
    * **context_parts** -- content strings for every other source type
      (retrieval, tool, user, etc.).
    """
    system_parts: list[str] = []
    memory_items: list[ContextItem] = []
    context_parts: list[str] = []

    for item in window.items:
        if item.source == SourceType.SYSTEM:
            system_parts.append(item.content)
        elif item.source == SourceType.MEMORY:
            memory_items.append(item)
        else:
            context_parts.append(item.content)

    return ClassifiedItems(system_parts, memory_items, context_parts)
