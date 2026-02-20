"""Tests for budget default factory functions."""

from __future__ import annotations

import pytest

from astro_context.models.budget import TokenBudget
from astro_context.models.budget_defaults import (
    default_agent_budget,
    default_chat_budget,
    default_rag_budget,
)
from astro_context.models.context import SourceType

# ---------------------------------------------------------------------------
# default_chat_budget
# ---------------------------------------------------------------------------


class TestDefaultChatBudget:
    """default_chat_budget returns a valid TokenBudget for chat use-cases."""

    def test_returns_token_budget(self) -> None:
        budget = default_chat_budget(8000)
        assert isinstance(budget, TokenBudget)

    def test_total_tokens_matches(self) -> None:
        budget = default_chat_budget(8000)
        assert budget.total_tokens == 8000

    def test_reserve_tokens_positive(self) -> None:
        budget = default_chat_budget(8000)
        assert budget.reserve_tokens > 0

    def test_reserve_is_fifteen_percent(self) -> None:
        budget = default_chat_budget(10000)
        assert budget.reserve_tokens == 1500

    def test_allocations_sum_plus_reserve_within_budget(self) -> None:
        budget = default_chat_budget(8000)
        alloc_sum = sum(a.max_tokens for a in budget.allocations)
        assert alloc_sum + budget.reserve_tokens <= budget.total_tokens

    def test_shared_pool_positive(self) -> None:
        budget = default_chat_budget(8000)
        assert budget.shared_pool > 0

    def test_has_system_allocation(self) -> None:
        budget = default_chat_budget(8000)
        sources = {a.source for a in budget.allocations}
        assert SourceType.SYSTEM in sources

    def test_has_memory_allocation(self) -> None:
        budget = default_chat_budget(8000)
        sources = {a.source for a in budget.allocations}
        assert SourceType.MEMORY in sources

    def test_has_conversation_allocation(self) -> None:
        budget = default_chat_budget(8000)
        sources = {a.source for a in budget.allocations}
        assert SourceType.CONVERSATION in sources

    def test_has_retrieval_allocation(self) -> None:
        budget = default_chat_budget(8000)
        sources = {a.source for a in budget.allocations}
        assert SourceType.RETRIEVAL in sources

    def test_memory_plus_conversation_larger_than_retrieval(self) -> None:
        """Chat budget prioritises memory + conversation over retrieval."""
        budget = default_chat_budget(10000)
        mem = next(a for a in budget.allocations if a.source == SourceType.MEMORY)
        conv = next(a for a in budget.allocations if a.source == SourceType.CONVERSATION)
        ret = next(a for a in budget.allocations if a.source == SourceType.RETRIEVAL)
        assert mem.max_tokens + conv.max_tokens > ret.max_tokens


# ---------------------------------------------------------------------------
# default_rag_budget
# ---------------------------------------------------------------------------


class TestDefaultRagBudget:
    """default_rag_budget returns a valid TokenBudget for RAG use-cases."""

    def test_returns_token_budget(self) -> None:
        budget = default_rag_budget(8000)
        assert isinstance(budget, TokenBudget)

    def test_total_tokens_matches(self) -> None:
        budget = default_rag_budget(8000)
        assert budget.total_tokens == 8000

    def test_reserve_tokens_positive(self) -> None:
        budget = default_rag_budget(8000)
        assert budget.reserve_tokens > 0

    def test_allocations_sum_plus_reserve_within_budget(self) -> None:
        budget = default_rag_budget(8000)
        alloc_sum = sum(a.max_tokens for a in budget.allocations)
        assert alloc_sum + budget.reserve_tokens <= budget.total_tokens

    def test_shared_pool_positive(self) -> None:
        budget = default_rag_budget(8000)
        assert budget.shared_pool > 0

    def test_retrieval_larger_than_memory(self) -> None:
        """RAG budget prioritises retrieval over persistent memory."""
        budget = default_rag_budget(10000)
        mem = next(a for a in budget.allocations if a.source == SourceType.MEMORY)
        ret = next(a for a in budget.allocations if a.source == SourceType.RETRIEVAL)
        assert ret.max_tokens > mem.max_tokens

    def test_retrieval_is_forty_percent(self) -> None:
        budget = default_rag_budget(10000)
        ret = next(a for a in budget.allocations if a.source == SourceType.RETRIEVAL)
        assert ret.max_tokens == 4000

    def test_has_conversation_allocation(self) -> None:
        budget = default_rag_budget(8000)
        sources = {a.source for a in budget.allocations}
        assert SourceType.CONVERSATION in sources


# ---------------------------------------------------------------------------
# default_agent_budget
# ---------------------------------------------------------------------------


class TestDefaultAgentBudget:
    """default_agent_budget returns a valid TokenBudget for agent use-cases."""

    def test_returns_token_budget(self) -> None:
        budget = default_agent_budget(8000)
        assert isinstance(budget, TokenBudget)

    def test_total_tokens_matches(self) -> None:
        budget = default_agent_budget(8000)
        assert budget.total_tokens == 8000

    def test_reserve_tokens_positive(self) -> None:
        budget = default_agent_budget(8000)
        assert budget.reserve_tokens > 0

    def test_allocations_sum_plus_reserve_within_budget(self) -> None:
        budget = default_agent_budget(8000)
        alloc_sum = sum(a.max_tokens for a in budget.allocations)
        assert alloc_sum + budget.reserve_tokens <= budget.total_tokens

    def test_has_tool_allocation(self) -> None:
        """Agent budget includes a TOOL allocation."""
        budget = default_agent_budget(8000)
        sources = {a.source for a in budget.allocations}
        assert SourceType.TOOL in sources

    def test_has_five_allocations(self) -> None:
        """Agent budget has system, memory, conversation, retrieval, and tool."""
        budget = default_agent_budget(8000)
        assert len(budget.allocations) == 5

    def test_system_is_fifteen_percent(self) -> None:
        budget = default_agent_budget(10000)
        sys_alloc = next(a for a in budget.allocations if a.source == SourceType.SYSTEM)
        assert sys_alloc.max_tokens == 1500

    def test_has_conversation_allocation(self) -> None:
        budget = default_agent_budget(8000)
        sources = {a.source for a in budget.allocations}
        assert SourceType.CONVERSATION in sources


# ---------------------------------------------------------------------------
# Various max_tokens values
# ---------------------------------------------------------------------------


class TestBudgetDefaultsVariousMaxTokens:
    """Budget defaults work correctly across a range of max_tokens values."""

    @pytest.mark.parametrize("max_tokens", [1000, 4096, 8000, 32000, 128000])
    def test_chat_budget_valid(self, max_tokens: int) -> None:
        budget = default_chat_budget(max_tokens)
        assert budget.total_tokens == max_tokens
        alloc_sum = sum(a.max_tokens for a in budget.allocations)
        assert alloc_sum + budget.reserve_tokens <= budget.total_tokens

    @pytest.mark.parametrize("max_tokens", [1000, 4096, 8000, 32000, 128000])
    def test_rag_budget_valid(self, max_tokens: int) -> None:
        budget = default_rag_budget(max_tokens)
        assert budget.total_tokens == max_tokens
        alloc_sum = sum(a.max_tokens for a in budget.allocations)
        assert alloc_sum + budget.reserve_tokens <= budget.total_tokens

    @pytest.mark.parametrize("max_tokens", [1000, 4096, 8000, 32000, 128000])
    def test_agent_budget_valid(self, max_tokens: int) -> None:
        budget = default_agent_budget(max_tokens)
        assert budget.total_tokens == max_tokens
        alloc_sum = sum(a.max_tokens for a in budget.allocations)
        assert alloc_sum + budget.reserve_tokens <= budget.total_tokens

    @pytest.mark.parametrize("max_tokens", [1000, 8000, 128000])
    def test_all_budgets_have_positive_reserve(self, max_tokens: int) -> None:
        for factory in (default_chat_budget, default_rag_budget, default_agent_budget):
            budget = factory(max_tokens)
            assert budget.reserve_tokens > 0, f"{factory.__name__}({max_tokens}) has no reserve"


# ---------------------------------------------------------------------------
# Invalid max_tokens
# ---------------------------------------------------------------------------


class TestBudgetDefaultsInvalidInput:
    """Budget defaults reject invalid max_tokens."""

    def test_chat_budget_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            default_chat_budget(0)

    def test_chat_budget_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            default_chat_budget(-100)

    def test_rag_budget_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            default_rag_budget(0)

    def test_rag_budget_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            default_rag_budget(-50)

    def test_agent_budget_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            default_agent_budget(0)

    def test_agent_budget_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            default_agent_budget(-1)


# ---------------------------------------------------------------------------
# Priority ordering in allocations
# ---------------------------------------------------------------------------


class TestBudgetAllocationPriorities:
    """Budget allocations have expected priority values."""

    def test_chat_system_priority_highest(self) -> None:
        budget = default_chat_budget(8000)
        sys_alloc = next(a for a in budget.allocations if a.source == SourceType.SYSTEM)
        assert sys_alloc.priority == 10

    def test_chat_memory_priority_high(self) -> None:
        budget = default_chat_budget(8000)
        mem_alloc = next(a for a in budget.allocations if a.source == SourceType.MEMORY)
        assert mem_alloc.priority == 8

    def test_chat_conversation_priority_medium(self) -> None:
        budget = default_chat_budget(8000)
        conv_alloc = next(a for a in budget.allocations if a.source == SourceType.CONVERSATION)
        assert conv_alloc.priority == 7

    def test_chat_retrieval_priority_default(self) -> None:
        budget = default_chat_budget(8000)
        ret_alloc = next(a for a in budget.allocations if a.source == SourceType.RETRIEVAL)
        assert ret_alloc.priority == 5

    def test_agent_tool_priority(self) -> None:
        budget = default_agent_budget(8000)
        tool_alloc = next(a for a in budget.allocations if a.source == SourceType.TOOL)
        assert tool_alloc.priority == 6
