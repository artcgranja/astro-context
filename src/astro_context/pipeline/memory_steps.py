"""Pipeline step factories for memory-related operations.

Provides step factories that integrate graph-based entity lookup and
automatic memory promotion into the context assembly pipeline.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from astro_context.models.context import ContextItem, SourceType
from astro_context.pipeline.step import PipelineStep

if TYPE_CHECKING:
    from astro_context.memory.graph_memory import SimpleGraphMemory
    from astro_context.models.memory import ConversationTurn, MemoryEntry
    from astro_context.models.query import QueryBundle
    from astro_context.protocols.memory import MemoryConsolidator, MemoryExtractor
    from astro_context.protocols.storage import MemoryEntryStore

logger = logging.getLogger(__name__)


def graph_retrieval_step(
    graph: SimpleGraphMemory,
    store: MemoryEntryStore,
    entity_extractor: Callable[[str], list[str]],
    max_depth: int = 2,
    max_items: int = 5,
    name: str = "graph_retrieval",
    on_error: str = "skip",
) -> PipelineStep:
    """Create a pipeline step that retrieves memory entries linked to graph entities.

    Flow:
        1. Extract entity IDs from the query using *entity_extractor*.
        2. For each entity, traverse the graph via BFS up to *max_depth* hops.
        3. Collect memory IDs linked to those entities.
        4. Fetch the corresponding ``MemoryEntry`` objects from the *store*.
        5. Convert to ``ContextItem`` objects with ``source_type=MEMORY``,
           ``priority=6``.

    Parameters:
        graph: The ``SimpleGraphMemory`` instance to traverse.
        store: A ``MemoryEntryStore`` implementation that holds persistent
            ``MemoryEntry`` objects.
        entity_extractor: User-provided callable that maps a query string
            to a list of entity IDs.
        max_depth: Maximum BFS traversal depth (default 2).
        max_items: Maximum number of ``ContextItem`` objects to return.
        name: Step name for diagnostics.
        on_error: Error policy -- ``"skip"`` (default) or ``"raise"``.

    Returns:
        A ``PipelineStep`` suitable for ``pipeline.add_step()``.
    """

    def _retrieve(items: list[ContextItem], query: QueryBundle) -> list[ContextItem]:
        entity_ids = entity_extractor(query.query_str)
        if not entity_ids:
            return items

        # Collect memory IDs from all extracted entities and their neighbors
        seen_memory_ids: set[str] = set()
        ordered_memory_ids: list[str] = []
        for entity_id in entity_ids:
            related_ids = graph.get_related_memory_ids(entity_id, max_depth=max_depth)
            for mid in related_ids:
                if mid not in seen_memory_ids:
                    seen_memory_ids.add(mid)
                    ordered_memory_ids.append(mid)

        if not ordered_memory_ids:
            return items

        # Fetch entries from the store and convert to ContextItems
        all_entries = store.list_all()
        entry_map: dict[str, MemoryEntry] = {e.id: e for e in all_entries}

        new_items: list[ContextItem] = []
        for mid in ordered_memory_ids:
            if len(new_items) >= max_items:
                break
            entry = entry_map.get(mid)
            if entry is None:
                continue
            new_items.append(
                ContextItem(
                    content=entry.content,
                    source=SourceType.MEMORY,
                    score=entry.relevance_score,
                    priority=6,
                    metadata={
                        "memory_id": entry.id,
                        "memory_type": str(entry.memory_type),
                        "tags": list(entry.tags),
                        "source": "graph_retrieval",
                    },
                )
            )

        return items + new_items

    return PipelineStep(
        name=name,
        fn=_retrieve,
        on_error=on_error,  # type: ignore[arg-type]
    )


def auto_promotion_step(
    extractor: MemoryExtractor,
    store: MemoryEntryStore,
    consolidator: MemoryConsolidator | None = None,
    name: str = "auto_promotion",
    on_error: str = "skip",
) -> PipelineStep:
    """Create a pipeline step that extracts and stores memories from context.

    This is a post-processor-style step that runs **after** retrieval.  It
    inspects the memory-typed items currently in the pipeline, extracts
    structured ``MemoryEntry`` objects via the *extractor*, and persists
    them in *store*.

    If a *consolidator* is provided it is used to deduplicate against
    entries already present in the store.

    The step returns the original items unchanged -- it is side-effect only.

    Parameters:
        extractor: A ``MemoryExtractor`` implementation.
        store: A ``MemoryEntryStore`` for persistence.
        consolidator: Optional ``MemoryConsolidator`` for deduplication.
        name: Step name for diagnostics.
        on_error: Error policy -- ``"skip"`` (default) or ``"raise"``.

    Returns:
        A ``PipelineStep`` suitable for ``pipeline.add_step()``.
    """

    def _promote(items: list[ContextItem], query: QueryBundle) -> list[ContextItem]:
        # Gather memory-source items and convert to ConversationTurn-like objects
        from astro_context.models.memory import ConversationTurn

        memory_items = [
            item
            for item in items
            if item.source in (SourceType.MEMORY, SourceType.CONVERSATION)
        ]
        if not memory_items:
            return items

        # Build ConversationTurn objects from memory items
        turns: list[ConversationTurn] = []
        for item in memory_items:
            role = item.metadata.get("role", "user")
            turns.append(
                ConversationTurn(
                    role=role,
                    content=item.content,
                    token_count=item.token_count,
                    timestamp=item.created_at,
                )
            )

        new_entries = extractor.extract(turns)
        if not new_entries:
            return items

        if consolidator is not None:
            existing = store.list_all()
            actions = consolidator.consolidate(new_entries, existing)
            for action, entry in actions:
                if action in ("add", "update") and entry is not None:
                    store.add(entry)
                # "none" / "delete" -> skip
        else:
            for entry in new_entries:
                store.add(entry)

        return items

    return PipelineStep(
        name=name,
        fn=_promote,
        on_error=on_error,  # type: ignore[arg-type]
    )


def create_eviction_promoter(
    extractor: MemoryExtractor,
    store: MemoryEntryStore,
    consolidator: MemoryConsolidator | None = None,
) -> Callable[[list[ConversationTurn]], None]:
    """Create an ``on_evict`` callback that promotes evicted turns to long-term memory.

    The returned callback is designed for use with
    ``SlidingWindowMemory(on_evict=...)``.  When turns are evicted from
    the sliding window the callback:

    1. Calls ``extractor.extract(turns)`` to produce ``MemoryEntry`` objects.
    2. If *consolidator* is provided, consolidates against existing entries.
    3. Stores new/updated entries in *store*.

    Errors are logged but **never** propagated -- a failing promoter must
    not crash the memory pipeline.

    Usage::

        promoter = create_eviction_promoter(extractor, store, consolidator)
        memory = SlidingWindowMemory(max_tokens=4096, on_evict=promoter)

    Parameters:
        extractor: A ``MemoryExtractor`` implementation.
        store: A ``MemoryEntryStore`` for persistence.
        consolidator: Optional ``MemoryConsolidator`` for deduplication.

    Returns:
        A callable matching the ``on_evict`` signature.
    """

    def _on_evict(turns: list[ConversationTurn]) -> None:
        try:
            new_entries = extractor.extract(turns)
            if not new_entries:
                return

            if consolidator is not None:
                existing = store.list_all()
                actions = consolidator.consolidate(new_entries, existing)
                for action, entry in actions:
                    if action in ("add", "update") and entry is not None:
                        store.add(entry)
            else:
                for entry in new_entries:
                    store.add(entry)
        except Exception:
            logger.exception(
                "eviction promoter failed — ignoring to protect pipeline"
            )

    return _on_evict
