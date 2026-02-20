"""Tests for query enrichment in ContextPipeline."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from astro_context.models.context import ContextItem, SourceType
from astro_context.models.query import QueryBundle
from astro_context.pipeline.enrichment import ContextQueryEnricher, MemoryContextEnricher
from astro_context.pipeline.pipeline import ContextPipeline
from astro_context.pipeline.step import PipelineStep, retriever_step
from tests.conftest import FakeRetriever, FakeTokenizer, make_memory_manager
from tests.test_pipeline.conftest import make_pipeline

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TOK = FakeTokenizer()


def _make_memory_item(
    content: str,
    created_at: datetime | None = None,
    item_id: str | None = None,
) -> ContextItem:
    return ContextItem(
        id=item_id or f"mem-{content[:8]}",
        content=content,
        source=SourceType.MEMORY,
        score=0.8,
        priority=7,
        token_count=_TOK.count_tokens(content),
        created_at=created_at or datetime.now(UTC),
    )


class RecordingEnricher:
    """An enricher that records calls for test assertions."""

    def __init__(self, suffix: str = " [enriched]") -> None:
        self.calls: list[tuple[str, list[ContextItem]]] = []
        self._suffix = suffix

    def enrich(self, query: str, context_items: list[ContextItem]) -> str:
        self.calls.append((query, list(context_items)))
        return query + self._suffix


class RecordingStep:
    """A step that records the query it received."""

    def __init__(self) -> None:
        self.queries: list[str] = []

    def __call__(
        self, items: list[ContextItem], query: QueryBundle
    ) -> list[ContextItem]:
        self.queries.append(query.query_str)
        return items


# ===========================================================================
# TestPipelineWithoutEnricher -- no enricher, normal behaviour
# ===========================================================================


class TestPipelineWithoutEnricher:
    """Pipeline without enricher works normally."""

    def test_no_enricher_query_unchanged(self) -> None:
        recorder = RecordingStep()
        pipeline = make_pipeline()
        pipeline.add_step(PipelineStep(name="record", fn=recorder))

        pipeline.build(QueryBundle(query_str="original query"))
        assert recorder.queries == ["original query"]

    def test_no_enricher_diagnostics_no_enriched_flag(self) -> None:
        pipeline = make_pipeline()
        pipeline.add_system_prompt("Hello")
        result = pipeline.build(QueryBundle(query_str="test"))
        assert "query_enriched" not in result.diagnostics


# ===========================================================================
# TestWithQueryEnricher -- fluent API
# ===========================================================================


class TestWithQueryEnricher:
    """with_query_enricher() sets the enricher and returns self."""

    def test_returns_self_for_chaining(self) -> None:
        pipeline = make_pipeline()
        enricher = MemoryContextEnricher()
        result = pipeline.with_query_enricher(enricher)
        assert result is pipeline

    def test_full_fluent_chain(self) -> None:
        pipeline = (
            make_pipeline()
            .with_query_enricher(MemoryContextEnricher())
            .add_system_prompt("System")
            .with_memory(make_memory_manager())
        )
        assert isinstance(pipeline, ContextPipeline)


# ===========================================================================
# TestEnricherCalledWithCorrectArgs
# ===========================================================================


class TestEnricherCalledWithCorrectArgs:
    """Enricher receives the query string and memory context items."""

    def test_enricher_receives_query_and_memory_items(self) -> None:
        enricher = RecordingEnricher()
        memory = make_memory_manager()
        memory.add_user_message("Hello")
        memory.add_assistant_message("Hi there!")

        pipeline = make_pipeline()
        pipeline.with_query_enricher(enricher)
        pipeline.with_memory(memory)

        pipeline.build(QueryBundle(query_str="What is context engineering?"))

        assert len(enricher.calls) == 1
        call_query, call_items = enricher.calls[0]
        assert call_query == "What is context engineering?"
        assert len(call_items) == 2
        assert all(
            i.source in (SourceType.MEMORY, SourceType.CONVERSATION) for i in call_items
        )

    def test_enricher_not_called_without_memory(self) -> None:
        """If no memory is attached, there are no memory items -> enricher not called."""
        enricher = RecordingEnricher()
        pipeline = make_pipeline()
        pipeline.with_query_enricher(enricher)

        pipeline.build(QueryBundle(query_str="test"))
        assert len(enricher.calls) == 0

    def test_enricher_not_called_when_memory_empty(self) -> None:
        enricher = RecordingEnricher()
        memory = make_memory_manager()  # empty

        pipeline = make_pipeline()
        pipeline.with_query_enricher(enricher)
        pipeline.with_memory(memory)

        pipeline.build(QueryBundle(query_str="test"))
        assert len(enricher.calls) == 0


# ===========================================================================
# TestEnrichedQueryUsedDownstream
# ===========================================================================


class TestEnrichedQueryUsedDownstream:
    """Enriched query is used for subsequent pipeline steps."""

    def test_step_receives_enriched_query(self) -> None:
        enricher = RecordingEnricher(suffix=" [enriched]")
        recorder = RecordingStep()

        memory = make_memory_manager()
        memory.add_user_message("Prior message")

        pipeline = make_pipeline()
        pipeline.with_query_enricher(enricher)
        pipeline.with_memory(memory)
        pipeline.add_step(PipelineStep(name="record", fn=recorder))

        pipeline.build(QueryBundle(query_str="original"))

        assert len(recorder.queries) == 1
        assert recorder.queries[0] == "original [enriched]"

    def test_multiple_steps_all_see_enriched_query(self) -> None:
        enricher = RecordingEnricher(suffix=" +ctx")
        rec1 = RecordingStep()
        rec2 = RecordingStep()

        memory = make_memory_manager()
        memory.add_user_message("Hi")

        pipeline = make_pipeline()
        pipeline.with_query_enricher(enricher)
        pipeline.with_memory(memory)
        pipeline.add_step(PipelineStep(name="r1", fn=rec1))
        pipeline.add_step(PipelineStep(name="r2", fn=rec2))

        pipeline.build(QueryBundle(query_str="q"))

        assert rec1.queries == ["q +ctx"]
        assert rec2.queries == ["q +ctx"]

    async def test_abuild_uses_enriched_query(self) -> None:
        enricher = RecordingEnricher(suffix=" [async-enriched]")
        recorder = RecordingStep()

        memory = make_memory_manager()
        memory.add_user_message("Prior message")

        pipeline = make_pipeline()
        pipeline.with_query_enricher(enricher)
        pipeline.with_memory(memory)
        pipeline.add_step(PipelineStep(name="record", fn=recorder))

        await pipeline.abuild(QueryBundle(query_str="async query"))

        assert recorder.queries == ["async query [async-enriched]"]


# ===========================================================================
# TestMemoryContextEnricher -- reference implementation
# ===========================================================================


class TestMemoryContextEnricher:
    """Tests for the MemoryContextEnricher reference implementation."""

    def test_appends_context_to_query(self) -> None:
        enricher = MemoryContextEnricher(max_items=3)
        items = [_make_memory_item("discussing project X budget")]

        result = enricher.enrich("what about the budget?", items)
        assert "what about the budget?" in result
        assert "discussing project X budget" in result

    def test_default_template(self) -> None:
        enricher = MemoryContextEnricher()
        items = [_make_memory_item("topic A")]
        result = enricher.enrich("query", items)
        assert result == "query\n\nConversation context: topic A"

    def test_respects_max_items(self) -> None:
        enricher = MemoryContextEnricher(max_items=2)
        now = datetime.now(UTC)
        items = [
            _make_memory_item("old", created_at=now - timedelta(hours=3), item_id="old"),
            _make_memory_item("mid", created_at=now - timedelta(hours=2), item_id="mid"),
            _make_memory_item("new", created_at=now - timedelta(hours=1), item_id="new"),
        ]
        result = enricher.enrich("q", items)
        # max_items=2, so only the 2 most recent (mid, new) should be included
        assert "mid" in result
        assert "new" in result
        assert "old" not in result

    def test_items_sorted_by_created_at(self) -> None:
        enricher = MemoryContextEnricher(max_items=10)
        now = datetime.now(UTC)
        items = [
            _make_memory_item("third", created_at=now, item_id="c"),
            _make_memory_item("first", created_at=now - timedelta(hours=2), item_id="a"),
            _make_memory_item("second", created_at=now - timedelta(hours=1), item_id="b"),
        ]
        result = enricher.enrich("q", items)
        # Context should be ordered oldest-first: first; second; third
        assert result == "q\n\nConversation context: first; second; third"

    def test_custom_template(self) -> None:
        enricher = MemoryContextEnricher(
            template="Q: {query} | CTX: {context}",
        )
        items = [_make_memory_item("hello")]
        result = enricher.enrich("search", items)
        assert result == "Q: search | CTX: hello"

    def test_empty_items_returns_original(self) -> None:
        enricher = MemoryContextEnricher()
        result = enricher.enrich("original query", [])
        assert result == "original query"

    def test_whitespace_only_content_returns_original(self) -> None:
        enricher = MemoryContextEnricher()
        items = [_make_memory_item("   ")]
        result = enricher.enrich("original query", items)
        assert result == "original query"

    def test_multiple_items_joined_with_semicolon(self) -> None:
        enricher = MemoryContextEnricher(max_items=5)
        items = [
            _make_memory_item("point A", item_id="a"),
            _make_memory_item("point B", item_id="b"),
        ]
        result = enricher.enrich("q", items)
        assert "point A; point B" in result

    def test_invalid_max_items_raises(self) -> None:
        with pytest.raises(ValueError, match="max_items must be a positive"):
            MemoryContextEnricher(max_items=0)

    def test_invalid_max_items_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="max_items must be a positive"):
            MemoryContextEnricher(max_items=-1)

    def test_repr(self) -> None:
        enricher = MemoryContextEnricher(max_items=3)
        assert repr(enricher) == "MemoryContextEnricher(max_items=3)"


# ===========================================================================
# TestProtocolConformance
# ===========================================================================


class TestProtocolConformance:
    """MemoryContextEnricher satisfies the ContextQueryEnricher protocol."""

    def test_is_runtime_checkable(self) -> None:
        enricher = MemoryContextEnricher()
        assert isinstance(enricher, ContextQueryEnricher)

    def test_recording_enricher_satisfies_protocol(self) -> None:
        enricher = RecordingEnricher()
        assert isinstance(enricher, ContextQueryEnricher)

    def test_arbitrary_object_with_enrich_satisfies_protocol(self) -> None:
        class MyEnricher:
            def enrich(self, query: str, context_items: list[ContextItem]) -> str:
                return query.upper()

        assert isinstance(MyEnricher(), ContextQueryEnricher)


# ===========================================================================
# TestDiagnosticsQueryEnriched
# ===========================================================================


class TestDiagnosticsQueryEnriched:
    """Pipeline diagnostics records query_enriched flag."""

    def test_diagnostics_set_when_enriched(self) -> None:
        enricher = RecordingEnricher()
        memory = make_memory_manager()
        memory.add_user_message("Hello")

        pipeline = make_pipeline()
        pipeline.with_query_enricher(enricher)
        pipeline.with_memory(memory)

        result = pipeline.build(QueryBundle(query_str="test"))
        assert result.diagnostics.get("query_enriched") is True

    def test_diagnostics_not_set_without_enrichment(self) -> None:
        """When enricher is set but no memory items, query_enriched is not set."""
        enricher = RecordingEnricher()
        pipeline = make_pipeline()
        pipeline.with_query_enricher(enricher)

        result = pipeline.build(QueryBundle(query_str="test"))
        assert "query_enriched" not in result.diagnostics

    async def test_abuild_diagnostics_set_when_enriched(self) -> None:
        enricher = RecordingEnricher()
        memory = make_memory_manager()
        memory.add_user_message("Hello")

        pipeline = make_pipeline()
        pipeline.with_query_enricher(enricher)
        pipeline.with_memory(memory)

        result = await pipeline.abuild(QueryBundle(query_str="test"))
        assert result.diagnostics.get("query_enriched") is True


# ===========================================================================
# TestEnricherIntegration -- full pipeline integration
# ===========================================================================


class TestEnricherIntegration:
    """Full integration: enricher + retriever + memory + system prompt."""

    def test_full_pipeline_with_enricher(self) -> None:
        enricher = MemoryContextEnricher(max_items=3)
        recorder = RecordingStep()

        memory = make_memory_manager()
        memory.add_user_message("We were discussing Python testing")
        memory.add_assistant_message("Yes, pytest is great for testing")

        retrieval_items = [
            ContextItem(
                id="doc-1",
                content="pytest documentation guide",
                source=SourceType.RETRIEVAL,
                score=0.9,
                priority=5,
                token_count=_TOK.count_tokens("pytest documentation guide"),
            ),
        ]

        pipeline = (
            make_pipeline()
            .add_system_prompt("You are a Python expert.")
            .with_memory(memory)
            .with_query_enricher(enricher)
            .add_step(PipelineStep(name="record", fn=recorder))
            .add_step(retriever_step("search", FakeRetriever(retrieval_items)))
        )

        result = pipeline.build(QueryBundle(query_str="how do I test async code?"))

        # The recorder should have seen the enriched query
        assert len(recorder.queries) == 1
        assert "how do I test async code?" in recorder.queries[0]
        assert "Conversation context:" in recorder.queries[0]

        # Result should include system + conversation + retrieval items
        sources = {i.source for i in result.window.items}
        assert SourceType.SYSTEM in sources
        assert SourceType.CONVERSATION in sources
        assert SourceType.RETRIEVAL in sources
