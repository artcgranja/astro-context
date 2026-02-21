"""Tests for TokenBudget per-source allocation enforcement in ContextPipeline."""

from __future__ import annotations

from astro_context.models.budget import BudgetAllocation, TokenBudget
from astro_context.models.budget_defaults import (
    default_agent_budget,
    default_chat_budget,
    default_rag_budget,
)
from astro_context.models.context import ContextItem, SourceType
from astro_context.models.query import QueryBundle
from astro_context.pipeline.pipeline import ContextPipeline
from astro_context.pipeline.step import PipelineStep, retriever_step
from tests.conftest import FakeRetriever, FakeTokenizer, make_memory_manager
from tests.test_pipeline.conftest import make_pipeline

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TOK = FakeTokenizer()


def _make_item(
    content: str,
    source: SourceType = SourceType.RETRIEVAL,
    score: float = 0.8,
    priority: int = 5,
    item_id: str | None = None,
) -> ContextItem:
    """Create a ContextItem with a computed token count."""
    return ContextItem(
        id=item_id or f"item-{content[:8]}",
        content=content,
        source=source,
        score=score,
        priority=priority,
        token_count=_TOK.count_tokens(content),
    )


def _passthrough_step(items: list[ContextItem], query: QueryBundle) -> list[ContextItem]:
    """A no-op step that passes items through unchanged."""
    return items


def _make_pipeline_with_budget(
    max_tokens: int,
    budget: TokenBudget,
) -> ContextPipeline:
    """Create a pipeline with FakeTokenizer and a specific budget."""
    return ContextPipeline(
        max_tokens=max_tokens,
        tokenizer=FakeTokenizer(),
        budget=budget,
    )


# ===========================================================================
# TestNoBudget -- backwards compatibility
# ===========================================================================


class TestNoBudget:
    """Pipeline without a TokenBudget behaves exactly as before."""

    def test_no_budget_all_items_included(self) -> None:
        items = [_make_item(f"doc {i}") for i in range(3)]
        retriever = FakeRetriever(items)
        pipeline = make_pipeline(max_tokens=8192)
        pipeline.add_step(retriever_step("search", retriever))

        result = pipeline.build(QueryBundle(query_str="test"))
        assert len(result.window.items) == 3
        assert len(result.overflow_items) == 0

    def test_no_budget_diagnostics_have_no_source_usage(self) -> None:
        pipeline = make_pipeline()
        pipeline.add_system_prompt("Hello")
        result = pipeline.build(QueryBundle(query_str="test"))
        assert "token_usage_by_source" not in result.diagnostics

    def test_no_budget_property_is_none(self) -> None:
        pipeline = make_pipeline()
        assert pipeline.budget is None


# ===========================================================================
# TestPerSourceAllocation -- per-source caps
# ===========================================================================


class TestPerSourceAllocation:
    """Per-source allocations enforce max_tokens per source type."""

    def test_retrieval_items_capped_by_allocation(self) -> None:
        """Items exceeding the retrieval allocation go to overflow."""
        budget = TokenBudget(
            total_tokens=1000,
            allocations=[
                BudgetAllocation(source=SourceType.RETRIEVAL, max_tokens=10),
            ],
        )
        # Each "word word word" item is 3 tokens. Cap of 10 fits 3 items (9 tokens).
        items = [_make_item("word word word", item_id=f"r-{i}") for i in range(5)]
        retriever = FakeRetriever(items)

        pipeline = _make_pipeline_with_budget(max_tokens=1000, budget=budget)
        pipeline.add_step(retriever_step("search", retriever))

        result = pipeline.build(QueryBundle(query_str="test"))
        retrieval_in_window = [
            i for i in result.window.items if i.source == SourceType.RETRIEVAL
        ]
        # 10 tokens cap, 3 tokens each -> 3 items max (9 tokens)
        assert len(retrieval_in_window) == 3
        assert len(result.overflow_items) >= 2

    def test_system_items_capped_by_allocation(self) -> None:
        budget = TokenBudget(
            total_tokens=1000,
            allocations=[
                BudgetAllocation(source=SourceType.SYSTEM, max_tokens=5, priority=10),
            ],
        )
        pipeline = _make_pipeline_with_budget(max_tokens=1000, budget=budget)
        # "Hello world" = 2 tokens, "System prompt here now" = 4 tokens -> total 6 exceeds 5
        pipeline.add_system_prompt("Hello world")  # 2 tokens
        pipeline.add_system_prompt("System prompt here now")  # 4 tokens -> 6 total

        result = pipeline.build(QueryBundle(query_str="test"))
        system_items = [i for i in result.window.items if i.source == SourceType.SYSTEM]
        # Only the first system item (2 tokens) fits; second (4 tokens) makes total 6 > 5
        assert len(system_items) == 1
        assert len(result.overflow_items) >= 1

    def test_items_exceeding_source_budget_go_to_overflow(self) -> None:
        budget = TokenBudget(
            total_tokens=500,
            allocations=[
                BudgetAllocation(source=SourceType.RETRIEVAL, max_tokens=5),
            ],
        )
        # "big content here more words" = 5 tokens, exactly fills the cap
        # Second item won't fit
        items = [
            _make_item("big content here more words", item_id="fits"),
            _make_item("another item here", item_id="overflows"),
        ]
        retriever = FakeRetriever(items)
        pipeline = _make_pipeline_with_budget(max_tokens=500, budget=budget)
        pipeline.add_step(retriever_step("search", retriever))

        result = pipeline.build(QueryBundle(query_str="test"))
        assert any(i.id == "fits" for i in result.window.items)
        assert any(i.id == "overflows" for i in result.overflow_items)


# ===========================================================================
# TestMultipleSourceAllocations -- multiple source types each capped
# ===========================================================================


class TestMultipleSourceAllocations:
    """Multiple source types each respect their own cap."""

    def test_independent_caps_per_source(self) -> None:
        budget = TokenBudget(
            total_tokens=1000,
            allocations=[
                BudgetAllocation(source=SourceType.SYSTEM, max_tokens=10, priority=10),
                BudgetAllocation(source=SourceType.RETRIEVAL, max_tokens=10),
                BudgetAllocation(source=SourceType.CONVERSATION, max_tokens=10, priority=7),
            ],
        )

        # Use a step that injects both retrieval and tool items
        retrieval_items = [
            _make_item("ret one two", source=SourceType.RETRIEVAL, item_id="ret-1"),
            _make_item("ret three four five six", source=SourceType.RETRIEVAL, item_id="ret-2"),
        ]
        memory = make_memory_manager()
        memory.add_user_message("Hello how are you today")
        memory.add_assistant_message("I am fine thank you very much indeed")

        pipeline = _make_pipeline_with_budget(max_tokens=1000, budget=budget)
        pipeline.add_system_prompt("system instructions here")
        pipeline.with_memory(memory)
        pipeline.add_step(retriever_step("search", FakeRetriever(retrieval_items)))

        result = pipeline.build(QueryBundle(query_str="test"))

        # Each source type should be independently capped at 10 tokens
        retrieval_in = [i for i in result.window.items if i.source == SourceType.RETRIEVAL]
        system_in = [i for i in result.window.items if i.source == SourceType.SYSTEM]
        conversation_in = [
            i for i in result.window.items if i.source == SourceType.CONVERSATION
        ]

        retrieval_tokens = sum(i.token_count for i in retrieval_in)
        system_tokens = sum(i.token_count for i in system_in)
        conversation_tokens = sum(i.token_count for i in conversation_in)

        assert retrieval_tokens <= 10
        assert system_tokens <= 10
        assert conversation_tokens <= 10

    def test_diagnostics_include_token_usage_by_source(self) -> None:
        budget = TokenBudget(
            total_tokens=1000,
            allocations=[
                BudgetAllocation(source=SourceType.RETRIEVAL, max_tokens=100),
            ],
        )
        items = [_make_item("hello world")]
        pipeline = _make_pipeline_with_budget(max_tokens=1000, budget=budget)
        pipeline.add_step(retriever_step("search", FakeRetriever(items)))

        result = pipeline.build(QueryBundle(query_str="test"))
        assert "token_usage_by_source" in result.diagnostics
        assert "retrieval" in result.diagnostics["token_usage_by_source"]


# ===========================================================================
# TestSharedPool -- unallocated source items use the shared pool
# ===========================================================================


class TestSharedPool:
    """Items whose source has no explicit allocation pass through to the shared pool."""

    def test_unallocated_source_passes_through(self) -> None:
        """TOOL source has no allocation -- it competes in the shared pool."""
        budget = TokenBudget(
            total_tokens=1000,
            allocations=[
                BudgetAllocation(source=SourceType.RETRIEVAL, max_tokens=100),
            ],
        )
        # Inject a TOOL-source item via a step
        tool_item = _make_item("tool output data", source=SourceType.TOOL, item_id="tool-1")

        def add_tool(items: list[ContextItem], q: QueryBundle) -> list[ContextItem]:
            return [*items, tool_item]

        pipeline = _make_pipeline_with_budget(max_tokens=1000, budget=budget)
        pipeline.add_step(PipelineStep(name="add-tool", fn=add_tool))

        result = pipeline.build(QueryBundle(query_str="test"))
        tool_in = [i for i in result.window.items if i.source == SourceType.TOOL]
        assert len(tool_in) == 1

    def test_shared_pool_size_calculation(self) -> None:
        budget = TokenBudget(
            total_tokens=1000,
            reserve_tokens=100,
            allocations=[
                BudgetAllocation(source=SourceType.RETRIEVAL, max_tokens=200),
                BudgetAllocation(source=SourceType.MEMORY, max_tokens=300),
            ],
        )
        # shared_pool = 1000 - 200 - 300 - 100 = 400
        assert budget.shared_pool == 400


# ===========================================================================
# TestReserveTokens -- reserve is always subtracted
# ===========================================================================


class TestReserveTokens:
    """reserve_tokens is subtracted from the effective max for the window."""

    def test_reserve_reduces_effective_max(self) -> None:
        budget = TokenBudget(
            total_tokens=100,
            reserve_tokens=40,
        )
        # Effective max = 100 - 40 = 60 tokens
        # Each "word" item is 1 token. 65 items of 1 token each should overflow some.
        items = [_make_item("word", item_id=f"w-{i}") for i in range(65)]
        retriever = FakeRetriever(items)

        pipeline = _make_pipeline_with_budget(max_tokens=100, budget=budget)
        pipeline.add_step(retriever_step("search", retriever, top_k=65))

        result = pipeline.build(QueryBundle(query_str="test"))
        assert result.window.used_tokens <= 60
        assert len(result.overflow_items) > 0

    def test_reserve_with_allocations(self) -> None:
        budget = TokenBudget(
            total_tokens=200,
            reserve_tokens=50,
            allocations=[
                BudgetAllocation(source=SourceType.RETRIEVAL, max_tokens=50),
            ],
        )
        # Effective window max = 200 - 50 = 150
        # Retrieval cap = 50
        items = [
            _make_item("a b c d e f g h i j", item_id=f"r-{i}")  # 10 tokens each
            for i in range(10)
        ]
        retriever = FakeRetriever(items)
        pipeline = _make_pipeline_with_budget(max_tokens=200, budget=budget)
        pipeline.add_step(retriever_step("search", retriever, top_k=10))

        result = pipeline.build(QueryBundle(query_str="test"))
        retrieval_tokens = sum(
            i.token_count for i in result.window.items if i.source == SourceType.RETRIEVAL
        )
        # Retrieval capped at 50, so max 5 items of 10 tokens each
        assert retrieval_tokens <= 50
        assert result.window.used_tokens <= 150


# ===========================================================================
# TestBudgetFactoryDefaults -- default budget factories work with pipeline
# ===========================================================================


class TestBudgetFactoryDefaults:
    """Factory budget functions integrate correctly with the pipeline."""

    def test_default_chat_budget_with_pipeline(self) -> None:
        budget = default_chat_budget(max_tokens=1000)
        pipeline = _make_pipeline_with_budget(max_tokens=1000, budget=budget)
        pipeline.add_system_prompt("System message")

        items = [_make_item(f"doc {i}", item_id=f"r-{i}") for i in range(5)]
        pipeline.add_step(retriever_step("search", FakeRetriever(items)))

        result = pipeline.build(QueryBundle(query_str="test"))
        assert isinstance(result.window.used_tokens, int)
        assert result.window.used_tokens > 0
        # Reserve = 15% of 1000 = 150. Effective max = 850.
        assert result.window.max_tokens == 1000 - budget.reserve_tokens

    def test_default_rag_budget_with_pipeline(self) -> None:
        budget = default_rag_budget(max_tokens=2000)
        pipeline = _make_pipeline_with_budget(max_tokens=2000, budget=budget)
        pipeline.add_system_prompt("You are a helpful RAG assistant.")

        items = [_make_item(f"doc {i}", item_id=f"r-{i}") for i in range(3)]
        pipeline.add_step(retriever_step("search", FakeRetriever(items)))

        result = pipeline.build(QueryBundle(query_str="test"))
        assert "token_usage_by_source" in result.diagnostics

    def test_default_agent_budget_with_pipeline(self) -> None:
        budget = default_agent_budget(max_tokens=4000)
        pipeline = _make_pipeline_with_budget(max_tokens=4000, budget=budget)
        pipeline.add_system_prompt("You are an agent.")

        result = pipeline.build(QueryBundle(query_str="test"))
        assert result.window.max_tokens == 4000 - budget.reserve_tokens


# ===========================================================================
# TestBudgetWithFluentAPI -- with_budget() fluent setter
# ===========================================================================


class TestBudgetFluentAPI:
    """with_budget() sets the budget and returns self for chaining."""

    def test_with_budget_returns_self(self) -> None:
        pipeline = make_pipeline()
        budget = TokenBudget(total_tokens=1000)
        result = pipeline.with_budget(budget)
        assert result is pipeline

    def test_with_budget_sets_budget(self) -> None:
        pipeline = make_pipeline()
        budget = TokenBudget(total_tokens=1000, reserve_tokens=100)
        pipeline.with_budget(budget)
        assert pipeline.budget is budget
        assert pipeline.budget.reserve_tokens == 100

    def test_with_budget_overrides_constructor_budget(self) -> None:
        initial = TokenBudget(total_tokens=500)
        pipeline = ContextPipeline(
            max_tokens=500, tokenizer=FakeTokenizer(), budget=initial
        )
        replacement = TokenBudget(total_tokens=500, reserve_tokens=50)
        pipeline.with_budget(replacement)
        assert pipeline.budget is replacement


# ===========================================================================
# TestApplySourceBudgetsOrdering -- priority/score ordering within source cap
# ===========================================================================


class TestApplySourceBudgetsOrdering:
    """_apply_source_budgets respects priority and score when capping."""

    def test_higher_priority_items_kept_first(self) -> None:
        budget = TokenBudget(
            total_tokens=1000,
            allocations=[
                BudgetAllocation(source=SourceType.RETRIEVAL, max_tokens=5),
            ],
        )
        # Two items of 3 tokens each. Only one fits in 5-token cap.
        # Higher priority item should be kept.
        high_prio = _make_item("aaa bbb ccc", priority=8, item_id="high")
        low_prio = _make_item("ddd eee fff", priority=3, item_id="low")

        def inject(items: list[ContextItem], q: QueryBundle) -> list[ContextItem]:
            return [*items, low_prio, high_prio]

        pipeline = _make_pipeline_with_budget(max_tokens=1000, budget=budget)
        pipeline.add_step(PipelineStep(name="inject", fn=inject))

        result = pipeline.build(QueryBundle(query_str="test"))
        ids_in_window = {i.id for i in result.window.items}
        assert "high" in ids_in_window
        assert "low" not in ids_in_window

    def test_higher_score_items_kept_when_same_priority(self) -> None:
        budget = TokenBudget(
            total_tokens=1000,
            allocations=[
                BudgetAllocation(source=SourceType.RETRIEVAL, max_tokens=5),
            ],
        )
        high_score = _make_item("aaa bbb ccc", score=0.9, priority=5, item_id="hscore")
        low_score = _make_item("ddd eee fff", score=0.1, priority=5, item_id="lscore")

        def inject(items: list[ContextItem], q: QueryBundle) -> list[ContextItem]:
            return [*items, low_score, high_score]

        pipeline = _make_pipeline_with_budget(max_tokens=1000, budget=budget)
        pipeline.add_step(PipelineStep(name="inject", fn=inject))

        result = pipeline.build(QueryBundle(query_str="test"))
        ids_in_window = {i.id for i in result.window.items}
        assert "hscore" in ids_in_window
        assert "lscore" not in ids_in_window


# ===========================================================================
# TestBudgetAsync -- abuild() also respects budgets
# ===========================================================================


class TestBudgetAsync:
    """Budgets work with async abuild() too."""

    async def test_abuild_enforces_source_budget(self) -> None:
        budget = TokenBudget(
            total_tokens=1000,
            allocations=[
                BudgetAllocation(source=SourceType.RETRIEVAL, max_tokens=10),
            ],
        )
        items = [_make_item("word word word", item_id=f"r-{i}") for i in range(5)]
        retriever = FakeRetriever(items)

        pipeline = _make_pipeline_with_budget(max_tokens=1000, budget=budget)
        pipeline.add_step(retriever_step("search", retriever))

        result = await pipeline.abuild(QueryBundle(query_str="test"))
        retrieval_in = [i for i in result.window.items if i.source == SourceType.RETRIEVAL]
        retrieval_tokens = sum(i.token_count for i in retrieval_in)
        assert retrieval_tokens <= 10

    async def test_abuild_reserve_tokens_subtracted(self) -> None:
        budget = TokenBudget(total_tokens=100, reserve_tokens=40)
        items = [_make_item("w", item_id=f"w-{i}") for i in range(65)]
        retriever = FakeRetriever(items)

        pipeline = _make_pipeline_with_budget(max_tokens=100, budget=budget)
        pipeline.add_step(retriever_step("search", retriever, top_k=65))

        result = await pipeline.abuild(QueryBundle(query_str="test"))
        assert result.window.used_tokens <= 60


# ===========================================================================
# TestBudgetEdgeCases
# ===========================================================================


class TestBudgetEdgeCases:
    """Edge cases for budget enforcement."""

    def test_empty_allocations_no_source_filtering(self) -> None:
        """Budget with empty allocations means no per-source caps."""
        budget = TokenBudget(total_tokens=1000, reserve_tokens=100)
        items = [_make_item("hello world", item_id=f"r-{i}") for i in range(3)]
        retriever = FakeRetriever(items)

        pipeline = _make_pipeline_with_budget(max_tokens=1000, budget=budget)
        pipeline.add_step(retriever_step("search", retriever))

        result = pipeline.build(QueryBundle(query_str="test"))
        # All 3 items should fit (small tokens, large effective budget)
        assert len(result.window.items) == 3
        assert result.window.max_tokens == 900  # 1000 - 100 reserve

    def test_zero_allocation_for_source_blocks_all(self) -> None:
        """A source allocation of max_tokens=1 with items > 1 token blocks most items."""
        budget = TokenBudget(
            total_tokens=1000,
            allocations=[
                BudgetAllocation(source=SourceType.RETRIEVAL, max_tokens=1),
            ],
        )
        items = [_make_item("two words", item_id=f"r-{i}") for i in range(3)]
        retriever = FakeRetriever(items)

        pipeline = _make_pipeline_with_budget(max_tokens=1000, budget=budget)
        pipeline.add_step(retriever_step("search", retriever))

        result = pipeline.build(QueryBundle(query_str="test"))
        # Each item is 2 tokens, cap is 1 -> none fit
        retrieval_in = [i for i in result.window.items if i.source == SourceType.RETRIEVAL]
        assert len(retrieval_in) == 0

    def test_budget_overflow_combined_with_window_overflow(self) -> None:
        """Items can overflow from both budget caps and window size limits."""
        budget = TokenBudget(
            total_tokens=30,
            reserve_tokens=5,
            allocations=[
                BudgetAllocation(source=SourceType.RETRIEVAL, max_tokens=10),
            ],
        )
        # 5 items * 3 tokens each = 15 tokens. Budget cap = 10, so 3 fit.
        # Window max = 30 - 5 = 25. Even if all passed budget, window fits them.
        # But budget cuts first, so 3 pass budget, 2 budget-overflow.
        items = [_make_item("aaa bbb ccc", item_id=f"r-{i}") for i in range(5)]
        retriever = FakeRetriever(items)

        pipeline = _make_pipeline_with_budget(max_tokens=30, budget=budget)
        pipeline.add_step(retriever_step("search", retriever))

        result = pipeline.build(QueryBundle(query_str="test"))
        assert len(result.overflow_items) >= 2

    def test_string_query_build_with_budget(self) -> None:
        """build() accepts a plain string query with budget."""
        budget = TokenBudget(total_tokens=1000, reserve_tokens=100)
        pipeline = _make_pipeline_with_budget(max_tokens=1000, budget=budget)
        pipeline.add_system_prompt("Hello")

        result = pipeline.build("test query")
        assert result.window.max_tokens == 900


# ===========================================================================
# TestOverflowStrategyTruncate -- "truncate" keeps highest-priority items
# ===========================================================================


class TestOverflowStrategyTruncate:
    """The 'truncate' strategy keeps the highest-priority items within the cap."""

    def test_truncate_keeps_highest_priority_items(self) -> None:
        """With truncate, items are kept up to max_tokens in priority/score order."""
        budget = TokenBudget(
            total_tokens=1000,
            allocations=[
                BudgetAllocation(
                    source=SourceType.RETRIEVAL,
                    max_tokens=6,
                    overflow_strategy="truncate",
                ),
            ],
        )
        # Each item is 3 tokens. Cap of 6 fits exactly 2 items.
        high = _make_item("aaa bbb ccc", priority=9, score=0.9, item_id="high")
        mid = _make_item("ddd eee fff", priority=7, score=0.7, item_id="mid")
        low = _make_item("ggg hhh iii", priority=3, score=0.3, item_id="low")

        def inject(items: list[ContextItem], q: QueryBundle) -> list[ContextItem]:
            return [*items, low, mid, high]

        pipeline = _make_pipeline_with_budget(max_tokens=1000, budget=budget)
        pipeline.add_step(PipelineStep(name="inject", fn=inject))

        result = pipeline.build(QueryBundle(query_str="test"))
        ids_in_window = {i.id for i in result.window.items}
        assert "high" in ids_in_window
        assert "mid" in ids_in_window
        assert "low" not in ids_in_window
        assert any(i.id == "low" for i in result.overflow_items)

    def test_truncate_is_default_strategy(self) -> None:
        """When overflow_strategy is not specified, truncate is the default."""
        alloc = BudgetAllocation(
            source=SourceType.RETRIEVAL,
            max_tokens=100,
        )
        assert alloc.overflow_strategy == "truncate"


# ===========================================================================
# TestOverflowStrategyDrop -- "drop" drops ALL items when source exceeds cap
# ===========================================================================


class TestOverflowStrategyDrop:
    """The 'drop' strategy drops ALL items for a source when they exceed the cap."""

    def test_drop_discards_all_items_when_exceeding_cap(self) -> None:
        """With drop strategy, if total tokens exceed cap, ALL items are dropped."""
        budget = TokenBudget(
            total_tokens=1000,
            allocations=[
                BudgetAllocation(
                    source=SourceType.RETRIEVAL,
                    max_tokens=5,
                    overflow_strategy="drop",
                ),
            ],
        )
        # Each item is 3 tokens. Total = 9, exceeds cap of 5.
        # With "drop", ALL 3 items should be dropped.
        items = [_make_item("aaa bbb ccc", item_id=f"r-{i}") for i in range(3)]

        def inject(items_in: list[ContextItem], q: QueryBundle) -> list[ContextItem]:
            return [*items_in, *items]

        pipeline = _make_pipeline_with_budget(max_tokens=1000, budget=budget)
        pipeline.add_step(PipelineStep(name="inject", fn=inject))

        result = pipeline.build(QueryBundle(query_str="test"))
        retrieval_in = [i for i in result.window.items if i.source == SourceType.RETRIEVAL]
        assert len(retrieval_in) == 0
        assert len(result.overflow_items) == 3

    def test_drop_keeps_all_items_when_within_cap(self) -> None:
        """With drop strategy, if total tokens fit within cap, all items are kept."""
        budget = TokenBudget(
            total_tokens=1000,
            allocations=[
                BudgetAllocation(
                    source=SourceType.RETRIEVAL,
                    max_tokens=15,
                    overflow_strategy="drop",
                ),
            ],
        )
        # Each item is 3 tokens. Total = 9, within cap of 15.
        items = [_make_item("aaa bbb ccc", item_id=f"r-{i}") for i in range(3)]

        def inject(items_in: list[ContextItem], q: QueryBundle) -> list[ContextItem]:
            return [*items_in, *items]

        pipeline = _make_pipeline_with_budget(max_tokens=1000, budget=budget)
        pipeline.add_step(PipelineStep(name="inject", fn=inject))

        result = pipeline.build(QueryBundle(query_str="test"))
        retrieval_in = [i for i in result.window.items if i.source == SourceType.RETRIEVAL]
        assert len(retrieval_in) == 3
        assert len(result.overflow_items) == 0

    def test_drop_versus_truncate_behaviour(self) -> None:
        """Demonstrates the difference: truncate keeps some, drop keeps none."""
        # Same items, same cap -- only the strategy differs
        items = [_make_item("aaa bbb ccc", item_id=f"r-{i}") for i in range(3)]

        # Truncate: cap=6 fits 2 of 3 items (6 tokens out of 9)
        budget_truncate = TokenBudget(
            total_tokens=1000,
            allocations=[
                BudgetAllocation(
                    source=SourceType.RETRIEVAL,
                    max_tokens=6,
                    overflow_strategy="truncate",
                ),
            ],
        )

        def inject(items_in: list[ContextItem], q: QueryBundle) -> list[ContextItem]:
            return [*items_in, *items]

        pipeline_trunc = _make_pipeline_with_budget(max_tokens=1000, budget=budget_truncate)
        pipeline_trunc.add_step(PipelineStep(name="inject", fn=inject))
        result_trunc = pipeline_trunc.build(QueryBundle(query_str="test"))

        # Drop: cap=6 can't fit all 9 tokens -> all 3 are dropped
        budget_drop = TokenBudget(
            total_tokens=1000,
            allocations=[
                BudgetAllocation(
                    source=SourceType.RETRIEVAL,
                    max_tokens=6,
                    overflow_strategy="drop",
                ),
            ],
        )
        pipeline_drop = _make_pipeline_with_budget(max_tokens=1000, budget=budget_drop)
        pipeline_drop.add_step(PipelineStep(name="inject", fn=inject))
        result_drop = pipeline_drop.build(QueryBundle(query_str="test"))

        # Truncate kept 2 items, drop kept 0
        trunc_retrieval = [
            i for i in result_trunc.window.items if i.source == SourceType.RETRIEVAL
        ]
        drop_retrieval = [
            i for i in result_drop.window.items if i.source == SourceType.RETRIEVAL
        ]
        assert len(trunc_retrieval) == 2
        assert len(drop_retrieval) == 0

    def test_drop_with_exact_fit(self) -> None:
        """Drop strategy keeps items when total exactly equals the cap."""
        budget = TokenBudget(
            total_tokens=1000,
            allocations=[
                BudgetAllocation(
                    source=SourceType.RETRIEVAL,
                    max_tokens=9,
                    overflow_strategy="drop",
                ),
            ],
        )
        # 3 items * 3 tokens = 9 exactly matches cap
        items = [_make_item("aaa bbb ccc", item_id=f"r-{i}") for i in range(3)]

        def inject(items_in: list[ContextItem], q: QueryBundle) -> list[ContextItem]:
            return [*items_in, *items]

        pipeline = _make_pipeline_with_budget(max_tokens=1000, budget=budget)
        pipeline.add_step(PipelineStep(name="inject", fn=inject))

        result = pipeline.build(QueryBundle(query_str="test"))
        retrieval_in = [i for i in result.window.items if i.source == SourceType.RETRIEVAL]
        assert len(retrieval_in) == 3


# ===========================================================================
# TestBudgetDiagnostics -- budget_overflow_by_source and shared_pool_usage
# ===========================================================================


class TestBudgetDiagnostics:
    """Tests for the new budget diagnostic fields."""

    def test_budget_overflow_by_source_tracked(self) -> None:
        """budget_overflow_by_source reports the count of overflowed items per source."""
        budget = TokenBudget(
            total_tokens=1000,
            allocations=[
                BudgetAllocation(
                    source=SourceType.RETRIEVAL,
                    max_tokens=6,
                    overflow_strategy="truncate",
                ),
            ],
        )
        # 4 items * 3 tokens = 12 tokens. Cap = 6, so 2 fit, 2 overflow.
        items = [_make_item("aaa bbb ccc", item_id=f"r-{i}") for i in range(4)]

        def inject(items_in: list[ContextItem], q: QueryBundle) -> list[ContextItem]:
            return [*items_in, *items]

        pipeline = _make_pipeline_with_budget(max_tokens=1000, budget=budget)
        pipeline.add_step(PipelineStep(name="inject", fn=inject))

        result = pipeline.build(QueryBundle(query_str="test"))
        assert "budget_overflow_by_source" in result.diagnostics
        assert result.diagnostics["budget_overflow_by_source"]["retrieval"] == 2

    def test_budget_overflow_by_source_drop_strategy(self) -> None:
        """budget_overflow_by_source with drop strategy reports all items dropped."""
        budget = TokenBudget(
            total_tokens=1000,
            allocations=[
                BudgetAllocation(
                    source=SourceType.RETRIEVAL,
                    max_tokens=5,
                    overflow_strategy="drop",
                ),
            ],
        )
        items = [_make_item("aaa bbb ccc", item_id=f"r-{i}") for i in range(3)]

        def inject(items_in: list[ContextItem], q: QueryBundle) -> list[ContextItem]:
            return [*items_in, *items]

        pipeline = _make_pipeline_with_budget(max_tokens=1000, budget=budget)
        pipeline.add_step(PipelineStep(name="inject", fn=inject))

        result = pipeline.build(QueryBundle(query_str="test"))
        assert result.diagnostics["budget_overflow_by_source"]["retrieval"] == 3

    def test_no_overflow_no_overflow_diagnostics(self) -> None:
        """When no items overflow, budget_overflow_by_source is absent."""
        budget = TokenBudget(
            total_tokens=1000,
            allocations=[
                BudgetAllocation(source=SourceType.RETRIEVAL, max_tokens=100),
            ],
        )
        items = [_make_item("hello", item_id="r-0")]

        def inject(items_in: list[ContextItem], q: QueryBundle) -> list[ContextItem]:
            return [*items_in, *items]

        pipeline = _make_pipeline_with_budget(max_tokens=1000, budget=budget)
        pipeline.add_step(PipelineStep(name="inject", fn=inject))

        result = pipeline.build(QueryBundle(query_str="test"))
        assert "budget_overflow_by_source" not in result.diagnostics

    def test_shared_pool_usage_reported(self) -> None:
        """shared_pool_usage tracks tokens used by sources without allocations."""
        budget = TokenBudget(
            total_tokens=1000,
            allocations=[
                BudgetAllocation(source=SourceType.RETRIEVAL, max_tokens=100),
            ],
        )
        # TOOL source has no allocation -- it uses the shared pool
        tool_item = _make_item("tool output data here", source=SourceType.TOOL, item_id="tool-1")

        def inject(items_in: list[ContextItem], q: QueryBundle) -> list[ContextItem]:
            return [*items_in, tool_item]

        pipeline = _make_pipeline_with_budget(max_tokens=1000, budget=budget)
        pipeline.add_step(PipelineStep(name="inject", fn=inject))

        result = pipeline.build(QueryBundle(query_str="test"))
        assert "shared_pool_usage" in result.diagnostics
        assert result.diagnostics["shared_pool_usage"] == tool_item.token_count

    def test_shared_pool_usage_zero_when_all_allocated(self) -> None:
        """shared_pool_usage is 0 when all items belong to allocated sources."""
        budget = TokenBudget(
            total_tokens=1000,
            allocations=[
                BudgetAllocation(source=SourceType.RETRIEVAL, max_tokens=100),
                BudgetAllocation(source=SourceType.SYSTEM, max_tokens=100, priority=10),
            ],
        )
        items = [_make_item("hello world", item_id="r-0")]

        pipeline = _make_pipeline_with_budget(max_tokens=1000, budget=budget)
        pipeline.add_system_prompt("System message")
        pipeline.add_step(retriever_step("search", FakeRetriever(items)))

        result = pipeline.build(QueryBundle(query_str="test"))
        assert result.diagnostics["shared_pool_usage"] == 0

    def test_shared_pool_usage_with_budget_no_allocations(self) -> None:
        """With a budget but no allocations, all tokens go to shared pool."""
        budget = TokenBudget(total_tokens=1000)
        items = [_make_item("hello world", item_id="r-0")]

        pipeline = _make_pipeline_with_budget(max_tokens=1000, budget=budget)
        pipeline.add_step(retriever_step("search", FakeRetriever(items)))

        result = pipeline.build(QueryBundle(query_str="test"))
        # No allocations -> all items are unallocated -> shared_pool_usage = total tokens used
        assert result.diagnostics["shared_pool_usage"] == result.window.used_tokens

    def test_overflow_items_tracked_in_result(self) -> None:
        """Overflow items from budget enforcement appear in result.overflow_items."""
        budget = TokenBudget(
            total_tokens=1000,
            allocations=[
                BudgetAllocation(
                    source=SourceType.RETRIEVAL,
                    max_tokens=3,
                    overflow_strategy="truncate",
                ),
            ],
        )
        # Each item is 3 tokens. Cap = 3, so 1 fits, 2 overflow.
        items = [_make_item("aaa bbb ccc", item_id=f"r-{i}") for i in range(3)]

        def inject(items_in: list[ContextItem], q: QueryBundle) -> list[ContextItem]:
            return [*items_in, *items]

        pipeline = _make_pipeline_with_budget(max_tokens=1000, budget=budget)
        pipeline.add_step(PipelineStep(name="inject", fn=inject))

        result = pipeline.build(QueryBundle(query_str="test"))
        assert len(result.overflow_items) == 2
        overflow_ids = {i.id for i in result.overflow_items}
        window_ids = {i.id for i in result.window.items}
        # No overlap between window and overflow
        assert overflow_ids.isdisjoint(window_ids)


# ===========================================================================
# TestOverflowStrategyAsync -- abuild() respects overflow_strategy
# ===========================================================================


class TestOverflowStrategyAsync:
    """Async abuild() also respects the overflow_strategy."""

    async def test_abuild_drop_strategy(self) -> None:
        """abuild() with drop strategy drops all items exceeding cap."""
        budget = TokenBudget(
            total_tokens=1000,
            allocations=[
                BudgetAllocation(
                    source=SourceType.RETRIEVAL,
                    max_tokens=5,
                    overflow_strategy="drop",
                ),
            ],
        )
        items = [_make_item("aaa bbb ccc", item_id=f"r-{i}") for i in range(3)]
        retriever = FakeRetriever(items)

        pipeline = _make_pipeline_with_budget(max_tokens=1000, budget=budget)
        pipeline.add_step(retriever_step("search", retriever))

        result = await pipeline.abuild(QueryBundle(query_str="test"))
        retrieval_in = [i for i in result.window.items if i.source == SourceType.RETRIEVAL]
        assert len(retrieval_in) == 0
        assert len(result.overflow_items) == 3

    async def test_abuild_truncate_strategy(self) -> None:
        """abuild() with truncate strategy keeps items up to cap."""
        budget = TokenBudget(
            total_tokens=1000,
            allocations=[
                BudgetAllocation(
                    source=SourceType.RETRIEVAL,
                    max_tokens=6,
                    overflow_strategy="truncate",
                ),
            ],
        )
        # 3 items * 3 tokens = 9, cap = 6, 2 fit
        items = [_make_item("aaa bbb ccc", item_id=f"r-{i}") for i in range(3)]
        retriever = FakeRetriever(items)

        pipeline = _make_pipeline_with_budget(max_tokens=1000, budget=budget)
        pipeline.add_step(retriever_step("search", retriever))

        result = await pipeline.abuild(QueryBundle(query_str="test"))
        retrieval_in = [i for i in result.window.items if i.source == SourceType.RETRIEVAL]
        assert len(retrieval_in) == 2

    async def test_abuild_diagnostics_include_overflow_by_source(self) -> None:
        """abuild() diagnostics include budget_overflow_by_source."""
        budget = TokenBudget(
            total_tokens=1000,
            allocations=[
                BudgetAllocation(
                    source=SourceType.RETRIEVAL,
                    max_tokens=3,
                    overflow_strategy="truncate",
                ),
            ],
        )
        items = [_make_item("aaa bbb ccc", item_id=f"r-{i}") for i in range(3)]
        retriever = FakeRetriever(items)

        pipeline = _make_pipeline_with_budget(max_tokens=1000, budget=budget)
        pipeline.add_step(retriever_step("search", retriever))

        result = await pipeline.abuild(QueryBundle(query_str="test"))
        assert "budget_overflow_by_source" in result.diagnostics
        assert result.diagnostics["budget_overflow_by_source"]["retrieval"] == 2
