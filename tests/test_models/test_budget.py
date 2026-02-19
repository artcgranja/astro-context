"""Tests for astro_context.models.budget."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from astro_context.models.budget import BudgetAllocation, TokenBudget
from astro_context.models.context import SourceType


class TestTokenBudgetCreation:
    """TokenBudget creation and basic properties."""

    def test_creation_no_allocations(self) -> None:
        budget = TokenBudget(total_tokens=8192)
        assert budget.total_tokens == 8192
        assert budget.allocations == []
        assert budget.reserve_tokens == 0

    def test_creation_with_allocations(self) -> None:
        alloc = BudgetAllocation(
            source=SourceType.RETRIEVAL, max_tokens=2000, priority=5
        )
        budget = TokenBudget(total_tokens=8192, allocations=[alloc])
        assert len(budget.allocations) == 1
        assert budget.allocations[0].max_tokens == 2000

    def test_creation_with_reserve(self) -> None:
        budget = TokenBudget(total_tokens=8192, reserve_tokens=500)
        assert budget.reserve_tokens == 500


class TestTokenBudgetValidation:
    """TokenBudget rejects over-allocation."""

    def test_over_allocation_raises(self) -> None:
        with pytest.raises(ValidationError, match="exceed total budget"):
            TokenBudget(
                total_tokens=100,
                allocations=[
                    BudgetAllocation(source=SourceType.RETRIEVAL, max_tokens=60),
                    BudgetAllocation(source=SourceType.MEMORY, max_tokens=60),
                ],
            )

    def test_reserve_plus_allocation_over_budget_raises(self) -> None:
        with pytest.raises(ValidationError, match="exceed total budget"):
            TokenBudget(
                total_tokens=100,
                allocations=[
                    BudgetAllocation(source=SourceType.RETRIEVAL, max_tokens=80),
                ],
                reserve_tokens=30,
            )

    def test_exact_allocation_is_valid(self) -> None:
        budget = TokenBudget(
            total_tokens=100,
            allocations=[
                BudgetAllocation(source=SourceType.RETRIEVAL, max_tokens=50),
                BudgetAllocation(source=SourceType.MEMORY, max_tokens=50),
            ],
        )
        assert budget.shared_pool == 0

    def test_total_tokens_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            TokenBudget(total_tokens=0)

    def test_total_tokens_negative_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TokenBudget(total_tokens=-100)


class TestTokenBudgetSharedPool:
    """shared_pool calculation."""

    def test_shared_pool_no_allocations(self) -> None:
        budget = TokenBudget(total_tokens=8192)
        assert budget.shared_pool == 8192

    def test_shared_pool_with_allocations(self) -> None:
        budget = TokenBudget(
            total_tokens=8192,
            allocations=[
                BudgetAllocation(source=SourceType.RETRIEVAL, max_tokens=2000),
                BudgetAllocation(source=SourceType.MEMORY, max_tokens=1000),
            ],
        )
        assert budget.shared_pool == 8192 - 2000 - 1000

    def test_shared_pool_with_reserve(self) -> None:
        budget = TokenBudget(
            total_tokens=8192,
            allocations=[
                BudgetAllocation(source=SourceType.RETRIEVAL, max_tokens=2000),
            ],
            reserve_tokens=500,
        )
        assert budget.shared_pool == 8192 - 2000 - 500


class TestTokenBudgetGetAllocation:
    """get_allocation for explicit allocations and fallback to shared pool."""

    def test_explicit_allocation(self) -> None:
        budget = TokenBudget(
            total_tokens=8192,
            allocations=[
                BudgetAllocation(source=SourceType.RETRIEVAL, max_tokens=2000),
            ],
        )
        assert budget.get_allocation(SourceType.RETRIEVAL) == 2000

    def test_fallback_to_shared_pool(self) -> None:
        budget = TokenBudget(
            total_tokens=8192,
            allocations=[
                BudgetAllocation(source=SourceType.RETRIEVAL, max_tokens=2000),
            ],
        )
        # MEMORY has no explicit allocation, so falls back to shared pool
        assert budget.get_allocation(SourceType.MEMORY) == 8192 - 2000

    def test_get_overflow_strategy_explicit(self) -> None:
        budget = TokenBudget(
            total_tokens=8192,
            allocations=[
                BudgetAllocation(
                    source=SourceType.RETRIEVAL, max_tokens=2000, overflow_strategy="drop"
                ),
            ],
        )
        assert budget.get_overflow_strategy(SourceType.RETRIEVAL) == "drop"

    def test_get_overflow_strategy_default(self) -> None:
        budget = TokenBudget(total_tokens=8192)
        assert budget.get_overflow_strategy(SourceType.SYSTEM) == "truncate"
