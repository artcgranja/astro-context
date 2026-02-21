"""RAG search tool for the agent."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from astro_context.agent.models import AgentTool
from astro_context.agent.tool_decorator import tool
from astro_context.models.query import QueryBundle


def rag_tools(
    retriever: Any,
    embed_fn: Callable[[str], list[float]] | None = None,
) -> list[AgentTool]:
    """Create a ``search_docs`` tool for agentic RAG.

    The model decides when to search documentation, making this
    agentic RAG -- the model controls retrieval timing.

    Parameters
    ----------
    retriever:
        Any object with a ``retrieve(query, top_k)`` method.
    embed_fn:
        Optional embedding function.  If the retriever needs
        embeddings in the QueryBundle, provide this.
    """

    @tool(
        description=(
            "Search documentation for relevant information. Use when the user "
            "asks about features, APIs, concepts, or anything that might be in the docs."
        ),
    )
    def search_docs(query: str) -> str:
        """Search documentation for relevant information.

        Args:
            query: Search query for finding relevant documentation.
        """
        q = QueryBundle(query_str=query)
        if embed_fn is not None:
            q = q.model_copy(update={"embedding": embed_fn(query)})
        results = retriever.retrieve(q, top_k=5)
        if not results:
            return "No relevant documents found."
        parts: list[str] = []
        for item in results:
            section = item.metadata.get("section", "")
            prefix = f"[{section}] " if section else ""
            parts.append(f"{prefix}{item.content[:500]}")
        return "\n\n---\n\n".join(parts)

    return [search_docs]
