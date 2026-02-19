"""Tests for astro_context.models.context."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from astro_context.models.context import (
    ContextItem,
    ContextResult,
    ContextWindow,
    SourceType,
)

# ---------------------------------------------------------------------------
# ContextItem creation
# ---------------------------------------------------------------------------


class TestContextItemCreation:
    """Test ContextItem construction with default and custom values."""

    def test_defaults(self) -> None:
        item = ContextItem(content="hello", source=SourceType.RETRIEVAL)
        assert item.content == "hello"
        assert item.source == SourceType.RETRIEVAL
        assert item.score == 0.0
        assert item.priority == 5
        assert item.token_count == 0
        assert item.metadata == {}
        assert item.id  # auto-generated UUID
        assert item.created_at is not None

    def test_custom_values(self) -> None:
        item = ContextItem(
            id="custom-id",
            content="world",
            source=SourceType.SYSTEM,
            score=0.9,
            priority=10,
            token_count=42,
            metadata={"key": "value"},
        )
        assert item.id == "custom-id"
        assert item.score == 0.9
        assert item.priority == 10
        assert item.token_count == 42
        assert item.metadata == {"key": "value"}

    def test_all_source_types(self) -> None:
        for src in SourceType:
            item = ContextItem(content="x", source=src)
            assert item.source == src


# ---------------------------------------------------------------------------
# ContextItem is frozen
# ---------------------------------------------------------------------------


class TestContextItemFrozen:
    """ContextItem is frozen (immutable) -- mutation raises ValidationError."""

    def test_mutate_content_raises(self) -> None:
        item = ContextItem(content="immutable", source=SourceType.USER)
        with pytest.raises(ValidationError):
            item.content = "changed"  # type: ignore[misc]

    def test_mutate_score_raises(self) -> None:
        item = ContextItem(content="x", source=SourceType.USER)
        with pytest.raises(ValidationError):
            item.score = 0.5  # type: ignore[misc]

    def test_mutate_priority_raises(self) -> None:
        item = ContextItem(content="x", source=SourceType.USER)
        with pytest.raises(ValidationError):
            item.priority = 8  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ContextItem validation
# ---------------------------------------------------------------------------


class TestContextItemValidation:
    """Score and priority field validators."""

    def test_score_below_zero_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ContextItem(content="x", source=SourceType.USER, score=-0.1)

    def test_score_above_one_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ContextItem(content="x", source=SourceType.USER, score=1.1)

    def test_score_boundary_zero(self) -> None:
        item = ContextItem(content="x", source=SourceType.USER, score=0.0)
        assert item.score == 0.0

    def test_score_boundary_one(self) -> None:
        item = ContextItem(content="x", source=SourceType.USER, score=1.0)
        assert item.score == 1.0

    def test_priority_below_one_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ContextItem(content="x", source=SourceType.USER, priority=0)

    def test_priority_above_ten_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ContextItem(content="x", source=SourceType.USER, priority=11)

    def test_priority_boundary_one(self) -> None:
        item = ContextItem(content="x", source=SourceType.USER, priority=1)
        assert item.priority == 1

    def test_priority_boundary_ten(self) -> None:
        item = ContextItem(content="x", source=SourceType.USER, priority=10)
        assert item.priority == 10


# ---------------------------------------------------------------------------
# ContextWindow
# ---------------------------------------------------------------------------


class TestContextWindow:
    """ContextWindow token-aware container."""

    def test_add_item_fits(self) -> None:
        window = ContextWindow(max_tokens=100)
        item = ContextItem(content="x", source=SourceType.USER, token_count=50)
        assert window.add_item(item) is True
        assert len(window.items) == 1
        assert window.used_tokens == 50

    def test_add_item_does_not_fit(self) -> None:
        window = ContextWindow(max_tokens=10)
        item = ContextItem(content="x", source=SourceType.USER, token_count=11)
        assert window.add_item(item) is False
        assert len(window.items) == 0
        assert window.used_tokens == 0

    def test_add_item_exact_fit(self) -> None:
        window = ContextWindow(max_tokens=50)
        item = ContextItem(content="x", source=SourceType.USER, token_count=50)
        assert window.add_item(item) is True
        assert window.remaining_tokens == 0

    def test_add_items_by_priority_orders_correctly(self) -> None:
        window = ContextWindow(max_tokens=10000)
        items = [
            ContextItem(
                content="low", source=SourceType.USER, priority=1, score=0.5, token_count=10
            ),
            ContextItem(
                content="high", source=SourceType.USER, priority=10, score=0.5, token_count=10
            ),
            ContextItem(
                content="mid", source=SourceType.USER, priority=5, score=0.9, token_count=10
            ),
            ContextItem(
                content="mid2", source=SourceType.USER, priority=5, score=0.3, token_count=10
            ),
        ]
        overflow = window.add_items_by_priority(items)
        assert overflow == []
        # Items should be ordered: priority 10 first, then priority 5 (higher score first), then 1
        assert window.items[0].content == "high"
        assert window.items[1].content == "mid"
        assert window.items[2].content == "mid2"
        assert window.items[3].content == "low"

    def test_add_items_by_priority_returns_overflow(self) -> None:
        window = ContextWindow(max_tokens=25)
        items = [
            ContextItem(content="a", source=SourceType.USER, priority=10, token_count=10),
            ContextItem(content="b", source=SourceType.USER, priority=5, token_count=10),
            ContextItem(content="c", source=SourceType.USER, priority=1, token_count=10),
        ]
        overflow = window.add_items_by_priority(items)
        # Two highest-priority items fit (20 tokens), third (10 tokens) overflows
        assert len(window.items) == 2
        assert len(overflow) == 1
        assert overflow[0].content == "c"

    def test_remaining_tokens(self) -> None:
        window = ContextWindow(max_tokens=100)
        window.add_item(ContextItem(content="x", source=SourceType.USER, token_count=30))
        assert window.remaining_tokens == 70

    def test_remaining_tokens_never_negative(self) -> None:
        window = ContextWindow(max_tokens=100, used_tokens=120)
        assert window.remaining_tokens == 0

    def test_utilization(self) -> None:
        window = ContextWindow(max_tokens=100)
        window.add_item(ContextItem(content="x", source=SourceType.USER, token_count=25))
        assert window.utilization == pytest.approx(0.25)

    def test_utilization_empty(self) -> None:
        window = ContextWindow(max_tokens=100)
        assert window.utilization == 0.0

    def test_utilization_full(self) -> None:
        window = ContextWindow(max_tokens=10)
        window.add_item(ContextItem(content="x", source=SourceType.USER, token_count=10))
        assert window.utilization == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# ContextResult
# ---------------------------------------------------------------------------


class TestContextResult:
    """ContextResult construction and defaults."""

    def test_construction_with_defaults(self) -> None:
        window = ContextWindow(max_tokens=100)
        result = ContextResult(window=window)
        assert result.window is window
        assert result.formatted_output == ""
        assert result.format_type == "generic"
        assert result.overflow_items == []
        assert result.diagnostics == {}
        assert result.build_time_ms == 0.0

    def test_construction_with_all_fields(self) -> None:
        window = ContextWindow(max_tokens=100)
        overflow_item = ContextItem(content="overflow", source=SourceType.USER)
        result = ContextResult(
            window=window,
            formatted_output="formatted",
            format_type="anthropic",
            overflow_items=[overflow_item],
            diagnostics={"steps": []},
            build_time_ms=42.5,
        )
        assert result.formatted_output == "formatted"
        assert result.format_type == "anthropic"
        assert len(result.overflow_items) == 1
        assert result.diagnostics == {"steps": []}
        assert result.build_time_ms == 42.5
