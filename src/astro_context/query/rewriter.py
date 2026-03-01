"""Conversation-aware query rewriting transformers.

These transformers leverage chat history attached to a ``QueryBundle``
to produce context-enriched queries that improve retrieval quality in
multi-turn conversations.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from astro_context.models.memory import ConversationTurn
from astro_context.models.query import QueryBundle
from astro_context.protocols.query_transform import QueryTransformer

logger = logging.getLogger(__name__)


class ConversationRewriter:
    """Rewrites a query using conversation history via a user-supplied callback.

    When a ``QueryBundle`` carries non-empty ``chat_history``, the
    ``rewrite_fn`` is called to produce a self-contained query that
    incorporates conversational context.  If history is empty the
    original query is returned unchanged.

    Parameters:
        rewrite_fn: A callable ``(str, list[ConversationTurn]) -> str``
            that receives the current query string and conversation
            history and returns a rewritten query string.
    """

    __slots__ = ("_rewrite_fn",)

    def __init__(
        self,
        rewrite_fn: Callable[[str, list[ConversationTurn]], str],
    ) -> None:
        self._rewrite_fn = rewrite_fn

    def __repr__(self) -> str:
        return "ConversationRewriter()"

    def transform(self, query: QueryBundle) -> list[QueryBundle]:
        """Rewrite the query using conversation history if available.

        Parameters:
            query: The query bundle, optionally carrying ``chat_history``.

        Returns:
            A single-element list containing either the original query
            (when history is empty) or a rewritten ``QueryBundle`` with
            metadata recording the transformation.
        """
        if not query.chat_history:
            logger.debug("ConversationRewriter: no chat history, returning query unchanged")
            return [query]

        rewritten = self._rewrite_fn(query.query_str, query.chat_history)
        logger.debug(
            "ConversationRewriter rewrote query: %r -> %r",
            query.query_str,
            rewritten,
        )
        return [
            QueryBundle(
                query_str=rewritten,
                embedding=query.embedding,
                metadata={
                    **query.metadata,
                    "original_query": query.query_str,
                    "transform": "conversation_rewrite",
                },
                chat_history=query.chat_history,
            )
        ]


class ContextualQueryTransformer:
    """Wraps another transformer, prepending conversation context to the query.

    When a ``QueryBundle`` has non-empty ``chat_history``, a summary of
    recent turns is prepended to ``query_str`` before delegating to the
    inner transformer.  If history is empty the inner transformer
    receives the query unchanged.

    Parameters:
        inner: The wrapped ``QueryTransformer`` to delegate to.
        context_prefix: Text prepended before the conversation summary.
    """

    __slots__ = ("_context_prefix", "_inner")

    def __init__(
        self,
        inner: QueryTransformer,
        context_prefix: str = "Given the conversation context: ",
    ) -> None:
        self._inner = inner
        self._context_prefix = context_prefix

    def __repr__(self) -> str:
        return f"ContextualQueryTransformer(inner={self._inner!r})"

    @staticmethod
    def _summarize_turns(turns: list[ConversationTurn]) -> str:
        """Build a concise summary string from conversation turns."""
        parts: list[str] = []
        for turn in turns:
            parts.append(f"{turn.role}: {turn.content}")
        return " | ".join(parts)

    def transform(self, query: QueryBundle) -> list[QueryBundle]:
        """Optionally prepend context, then delegate to the inner transformer.

        Parameters:
            query: The query bundle, optionally carrying ``chat_history``.

        Returns:
            The result of the inner transformer, applied to a
            context-augmented query when history is present.
        """
        if not query.chat_history:
            logger.debug("ContextualQueryTransformer: no history, delegating directly")
            return self._inner.transform(query)

        summary = self._summarize_turns(query.chat_history)
        augmented_str = f"{self._context_prefix}{summary} | Query: {query.query_str}"
        augmented = QueryBundle(
            query_str=augmented_str,
            embedding=query.embedding,
            metadata=query.metadata,
            chat_history=query.chat_history,
        )
        logger.debug("ContextualQueryTransformer augmented query: %s", augmented_str)
        return self._inner.transform(augmented)
