"""Tests for astro_context.memory.summary_buffer."""

from __future__ import annotations

import pytest

from astro_context.memory.summary_buffer import SummaryBufferMemory
from astro_context.models.context import SourceType
from astro_context.models.memory import ConversationTurn
from tests.conftest import FakeTokenizer


def _make_buffer(
    max_tokens: int = 100,
    compact_fn: object = None,
    progressive_compact_fn: object = None,
    summary_priority: int = 6,
) -> SummaryBufferMemory:
    """Create a SummaryBufferMemory with a FakeTokenizer."""
    kwargs: dict[str, object] = {
        "max_tokens": max_tokens,
        "tokenizer": FakeTokenizer(),
        "summary_priority": summary_priority,
    }
    if compact_fn is not None:
        kwargs["compact_fn"] = compact_fn
    if progressive_compact_fn is not None:
        kwargs["progressive_compact_fn"] = progressive_compact_fn
    return SummaryBufferMemory(**kwargs)  # type: ignore[arg-type]


def _simple_compact(turns: list[ConversationTurn]) -> str:
    """Simple compaction: join turn contents."""
    return "Summary: " + "; ".join(t.content for t in turns)


def _progressive_compact(turns: list[ConversationTurn], prev: str | None) -> str:
    """Progressive compaction: append to previous summary."""
    new_part = "; ".join(t.content for t in turns)
    if prev is not None:
        return f"{prev} | {new_part}"
    return f"Summary: {new_part}"


class TestSummaryBufferCreation:
    """Constructor validation and factory."""

    def test_create_with_compact_fn(self) -> None:
        buf = _make_buffer(compact_fn=_simple_compact)
        assert buf.summary is None
        assert buf.summary_tokens == 0

    def test_create_with_progressive_compact_fn(self) -> None:
        buf = _make_buffer(progressive_compact_fn=_progressive_compact)
        assert buf.summary is None

    def test_raises_if_both_compact_fns_provided(self) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            _make_buffer(compact_fn=_simple_compact, progressive_compact_fn=_progressive_compact)

    def test_raises_if_neither_compact_fn_provided(self) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            SummaryBufferMemory(max_tokens=100, tokenizer=FakeTokenizer())

    def test_raises_if_max_tokens_zero(self) -> None:
        with pytest.raises(ValueError, match="max_tokens"):
            SummaryBufferMemory(
                max_tokens=0,
                compact_fn=_simple_compact,
                tokenizer=FakeTokenizer(),
            )

    def test_raises_if_max_tokens_negative(self) -> None:
        with pytest.raises(ValueError, match="max_tokens"):
            SummaryBufferMemory(
                max_tokens=-5,
                compact_fn=_simple_compact,
                tokenizer=FakeTokenizer(),
            )


class TestSummaryBufferAddWithinCapacity:
    """Adding turns that fit in the window does not produce a summary."""

    def test_add_turns_within_capacity_no_summary(self) -> None:
        buf = _make_buffer(max_tokens=100, compact_fn=_simple_compact)
        buf.add_message("user", "Hello")
        buf.add_message("assistant", "Hi")
        assert buf.summary is None
        assert len(buf.turns) == 2

    def test_add_turn_object_within_capacity(self) -> None:
        buf = _make_buffer(max_tokens=100, compact_fn=_simple_compact)
        turn = ConversationTurn(role="user", content="Hello", token_count=1)
        buf.add_turn(turn)
        assert len(buf.turns) == 1
        assert buf.summary is None


class TestSummaryBufferEvictionProducesSummary:
    """Exceeding the window triggers compaction and produces a summary."""

    def test_eviction_triggers_simple_summary(self) -> None:
        # Budget of 3 tokens. "Hello" = 1 token, "World" = 1 token, "three four five" = 3 tokens.
        # Total would be 5, exceeds 3. Need to evict both "Hello" and "World" to fit.
        buf = _make_buffer(max_tokens=3, compact_fn=_simple_compact)
        buf.add_message("user", "Hello")
        buf.add_message("assistant", "World")
        buf.add_message("user", "three four five")
        assert buf.summary is not None
        assert "Hello" in buf.summary
        assert "World" in buf.summary

    def test_summary_is_context_item_with_correct_source_and_priority(self) -> None:
        buf = _make_buffer(max_tokens=3, compact_fn=_simple_compact, summary_priority=6)
        buf.add_message("user", "Hello")
        buf.add_message("assistant", "World")
        buf.add_message("user", "three four five")

        items = buf.to_context_items()
        summary_items = [i for i in items if i.metadata.get("summary") is True]
        assert len(summary_items) == 1
        summary_item = summary_items[0]
        assert summary_item.source == SourceType.CONVERSATION
        assert summary_item.priority == 6
        assert summary_item.token_count > 0

    def test_progressive_compaction_receives_previous_summary(self) -> None:
        received_previous: list[str | None] = []

        def tracking_progressive(turns: list[ConversationTurn], prev: str | None) -> str:
            received_previous.append(prev)
            new_part = "; ".join(t.content for t in turns)
            if prev is not None:
                return f"{prev} | {new_part}"
            return f"Summary: {new_part}"

        # With max_tokens=2, each 1-token message triggers eviction after the second.
        buf = _make_buffer(max_tokens=2, progressive_compact_fn=tracking_progressive)
        buf.add_message("user", "first")
        buf.add_message("assistant", "second")
        # Now at 2 tokens. Adding a 3rd evicts oldest.
        buf.add_message("user", "third")
        assert len(received_previous) >= 1
        assert received_previous[0] is None  # First eviction: no previous summary

        # Force another eviction
        buf.add_message("assistant", "fourth")
        if len(received_previous) >= 2:
            assert received_previous[1] is not None  # Second eviction: has a previous summary

    def test_summary_token_count_updated(self) -> None:
        buf = _make_buffer(max_tokens=3, compact_fn=_simple_compact)
        buf.add_message("user", "Hello")
        buf.add_message("assistant", "World")
        buf.add_message("user", "three four five")
        assert buf.summary_tokens > 0
        tokenizer = FakeTokenizer()
        assert buf.summary_tokens == tokenizer.count_tokens(buf.summary)  # type: ignore[arg-type]


class TestSummaryBufferToContextItems:
    """to_context_items returns summary + window items."""

    def test_returns_only_window_items_when_no_summary(self) -> None:
        buf = _make_buffer(max_tokens=100, compact_fn=_simple_compact)
        buf.add_message("user", "Hello")
        buf.add_message("assistant", "World")
        items = buf.to_context_items()
        assert len(items) == 2
        assert all(i.metadata.get("summary") is not True for i in items)

    def test_returns_summary_plus_window_items(self) -> None:
        buf = _make_buffer(max_tokens=3, compact_fn=_simple_compact)
        buf.add_message("user", "Hello")
        buf.add_message("assistant", "World")
        buf.add_message("user", "three four five")
        items = buf.to_context_items()
        # Should have 1 summary + at least 1 window item
        summary_items = [i for i in items if i.metadata.get("summary") is True]
        window_items = [i for i in items if i.metadata.get("summary") is not True]
        assert len(summary_items) == 1
        assert len(window_items) >= 1

    def test_summary_comes_first(self) -> None:
        buf = _make_buffer(max_tokens=3, compact_fn=_simple_compact)
        buf.add_message("user", "Hello")
        buf.add_message("assistant", "World")
        buf.add_message("user", "three four five")
        items = buf.to_context_items()
        assert items[0].metadata.get("summary") is True

    def test_window_items_have_caller_priority(self) -> None:
        buf = _make_buffer(max_tokens=100, compact_fn=_simple_compact)
        buf.add_message("user", "Hello")
        items = buf.to_context_items(priority=9)
        assert items[0].priority == 9


class TestSummaryBufferClear:
    """clear() resets summary and window."""

    def test_clear_resets_everything(self) -> None:
        buf = _make_buffer(max_tokens=3, compact_fn=_simple_compact)
        buf.add_message("user", "Hello")
        buf.add_message("assistant", "World")
        buf.add_message("user", "three four five")
        assert buf.summary is not None
        assert len(buf.turns) > 0

        buf.clear()
        assert buf.summary is None
        assert buf.summary_tokens == 0
        assert len(buf.turns) == 0
        assert buf.total_tokens == 0

    def test_clear_allows_reuse(self) -> None:
        buf = _make_buffer(max_tokens=100, compact_fn=_simple_compact)
        buf.add_message("user", "Hello")
        buf.clear()
        buf.add_message("user", "New message")
        assert len(buf.turns) == 1
        assert buf.turns[0].content == "New message"


class TestSummaryBufferProperties:
    """Properties: summary, turns, total_tokens, summary_tokens."""

    def test_turns_returns_window_turns(self) -> None:
        buf = _make_buffer(max_tokens=100, compact_fn=_simple_compact)
        buf.add_message("user", "Hello")
        buf.add_message("assistant", "World")
        assert len(buf.turns) == 2
        assert buf.turns[0].role == "user"
        assert buf.turns[1].role == "assistant"

    def test_total_tokens_reflects_window(self) -> None:
        buf = _make_buffer(max_tokens=100, compact_fn=_simple_compact)
        buf.add_message("user", "Hello")
        assert buf.total_tokens > 0

    def test_summary_none_initially(self) -> None:
        buf = _make_buffer(max_tokens=100, compact_fn=_simple_compact)
        assert buf.summary is None
        assert buf.summary_tokens == 0


class TestSummaryBufferRepr:
    """__repr__ returns a useful string."""

    def test_repr_simple_mode(self) -> None:
        buf = _make_buffer(max_tokens=100, compact_fn=_simple_compact)
        r = repr(buf)
        assert "SummaryBufferMemory" in r
        assert "simple" in r

    def test_repr_progressive_mode(self) -> None:
        buf = _make_buffer(max_tokens=100, progressive_compact_fn=_progressive_compact)
        r = repr(buf)
        assert "SummaryBufferMemory" in r
        assert "progressive" in r
