"""Query enrichment for context-aware retrieval.

Provides a protocol and reference implementations for enriching queries
with memory context before retrieval steps execute.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from astro_context.models.context import ContextItem


@runtime_checkable
class ContextQueryEnricher(Protocol):
    """Enriches a query string using context items from memory.

    Implementations receive the raw query and the list of ``ContextItem``
    objects already collected from memory.  They return an enriched query
    string that downstream retrieval steps can use for better relevance.

    The library never calls an LLM -- enrichment is purely mechanical
    (template-based, keyword extraction, etc.).
    """

    def enrich(self, query: str, context_items: list[ContextItem]) -> str:
        """Return an enriched version of *query* using memory *context_items*."""
        ...


class MemoryContextEnricher:
    """Enriches queries by appending recent conversation context.

    Takes the most recent context items from memory and appends a summary
    to the query string.  This helps retrieval steps find documents that
    are relevant to the ongoing conversation, not just the literal query.

    Example::

        enricher = MemoryContextEnricher(max_items=3)
        pipeline = ContextPipeline(max_tokens=8192).with_query_enricher(enricher)

    If the user asks "what about the budget?" and recent memory includes
    discussion about Project X, the enriched query becomes::

        "what about the budget?\\n\\nConversation context: discussing Project X budget"

    Parameters:
        max_items: Maximum number of recent memory items to include.
        template: Format string with ``{query}`` and ``{context}`` placeholders.
    """

    __slots__ = ("_max_items", "_template")

    def __init__(
        self,
        max_items: int = 5,
        template: str = "{query}\n\nConversation context: {context}",
    ) -> None:
        if max_items <= 0:
            msg = "max_items must be a positive integer"
            raise ValueError(msg)
        self._max_items = max_items
        self._template = template

    def enrich(self, query: str, context_items: list[ContextItem]) -> str:
        """Append recent memory context to the query string.

        Items are sorted by ``created_at`` (newest last) and the last
        ``max_items`` are used.  If no items are available the original
        query is returned unchanged.
        """
        if not context_items:
            return query

        # Sort by created_at ascending (oldest first) and take the last N
        sorted_items = sorted(context_items, key=lambda i: i.created_at)
        recent = sorted_items[-self._max_items :]

        # Build a concise context string from item contents
        snippets = [item.content for item in recent]
        context_str = "; ".join(snippets)

        if not context_str.strip():
            return query

        return self._template.format(query=query, context=context_str)

    def __repr__(self) -> str:
        return f"MemoryContextEnricher(max_items={self._max_items})"
