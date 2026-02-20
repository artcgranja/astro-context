"""Token budget allocation models."""

from __future__ import annotations

from typing import Literal, Self, TypeAlias

from pydantic import BaseModel, Field, model_validator

from .context import SourceType

OverflowStrategy: TypeAlias = Literal["truncate", "drop"]


class BudgetAllocation(BaseModel):
    """Token allocation for a specific source type."""

    source: SourceType
    max_tokens: int = Field(gt=0)
    priority: int = Field(default=5, ge=1, le=10)
    overflow_strategy: OverflowStrategy = Field(default="truncate")


class TokenBudget(BaseModel):
    """Manages token budget across all context sources.

    Allocates portions of the total token budget to different source types.
    Unallocated tokens form a shared pool.
    """

    total_tokens: int = Field(gt=0)
    allocations: list[BudgetAllocation] = Field(default_factory=list)
    reserve_tokens: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_allocations(self) -> Self:
        allocated = sum(a.max_tokens for a in self.allocations) + self.reserve_tokens
        if allocated > self.total_tokens:
            msg = f"Allocated tokens ({allocated}) exceed total budget ({self.total_tokens})"
            raise ValueError(msg)
        return self

    @property
    def shared_pool(self) -> int:
        """Tokens not explicitly allocated to any source."""
        allocated = sum(a.max_tokens for a in self.allocations) + self.reserve_tokens
        return self.total_tokens - allocated

    def get_allocation(self, source: SourceType) -> int:
        """Get the max tokens for a source type. Falls back to shared pool."""
        for alloc in self.allocations:
            if alloc.source == source:
                return alloc.max_tokens
        return self.shared_pool

    def get_overflow_strategy(self, source: SourceType) -> OverflowStrategy:
        """Get the overflow strategy for a source type."""
        for alloc in self.allocations:
            if alloc.source == source:
                return alloc.overflow_strategy
        return "truncate"
