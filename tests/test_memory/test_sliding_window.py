"""Tests for astro_context.memory.sliding_window."""

from __future__ import annotations

import pytest

from astro_context.memory.sliding_window import SlidingWindowMemory
from astro_context.models.context import SourceType
from tests.conftest import FakeTokenizer


def _make_memory(max_tokens: int = 1000) -> SlidingWindowMemory:
    """Create a SlidingWindowMemory with a FakeTokenizer."""
    return SlidingWindowMemory(max_tokens=max_tokens, tokenizer=FakeTokenizer())


class TestSlidingWindowAddTurn:
    """add_turn within and over budget."""

    def test_add_turn_within_budget(self) -> None:
        mem = _make_memory(max_tokens=1000)
        turn = mem.add_turn("user", "Hello, how are you?")
        assert turn.role == "user"
        assert turn.content == "Hello, how are you?"
        assert turn.token_count > 0
        assert len(mem.turns) == 1
        assert mem.total_tokens == turn.token_count

    def test_add_multiple_turns_within_budget(self) -> None:
        mem = _make_memory(max_tokens=1000)
        mem.add_turn("user", "Hello")
        mem.add_turn("assistant", "Hi there!")
        mem.add_turn("user", "How are you?")
        assert len(mem.turns) == 3

    def test_add_turn_evicts_oldest_when_over_budget(self) -> None:
        tokenizer = FakeTokenizer()
        msg1 = "Hello"
        msg2 = "World"
        msg3 = "This is a third message that should cause eviction"
        tokens1 = tokenizer.count_tokens(msg1)
        tokens2 = tokenizer.count_tokens(msg2)
        tokens3 = tokenizer.count_tokens(msg3)
        budget = tokens1 + tokens2 + tokens3 - 1  # just barely not enough for all 3

        mem = SlidingWindowMemory(max_tokens=budget, tokenizer=tokenizer)
        mem.add_turn("user", msg1)
        mem.add_turn("assistant", msg2)
        mem.add_turn("user", msg3)

        # The oldest turn should have been evicted
        assert len(mem.turns) < 3
        # The most recent turn should always be present
        assert mem.turns[-1].content == msg3

    def test_eviction_removes_from_front(self) -> None:
        """Oldest turns (front of list) are evicted first."""
        # Budget of 5 words: "first"=1, "second"=1, "a longer third..."=8
        mem = _make_memory(max_tokens=10)
        mem.add_turn("user", "first")
        mem.add_turn("assistant", "second")
        # Add a turn that forces eviction
        mem.add_turn("user", "a longer third message that takes more tokens than budget")

        # The first turn ("first") should have been evicted
        remaining_contents = [t.content for t in mem.turns]
        assert "first" not in remaining_contents


class TestSlidingWindowTruncation:
    """Single turn exceeding budget is truncated."""

    def test_single_oversized_turn_is_truncated(self) -> None:
        mem = _make_memory(max_tokens=5)
        long_text = "one two three four five six seven eight nine ten eleven twelve"
        turn = mem.add_turn("user", long_text)

        assert turn.token_count <= 5
        assert turn.content != long_text  # Must have been truncated
        assert turn.metadata.get("truncated") is True
        assert len(mem.turns) == 1

    def test_truncated_turn_fits_in_budget(self) -> None:
        mem = _make_memory(max_tokens=3)
        long_text = "word " * 100
        mem.add_turn("user", long_text)
        assert mem.total_tokens <= 3


class TestSlidingWindowToContextItems:
    """to_context_items produces ContextItem list."""

    def test_produces_context_items(self) -> None:
        mem = _make_memory(max_tokens=1000)
        mem.add_turn("user", "Hello")
        mem.add_turn("assistant", "Hi there!")

        items = mem.to_context_items()
        assert len(items) == 2

    def test_context_items_have_conversation_source(self) -> None:
        mem = _make_memory(max_tokens=1000)
        mem.add_turn("user", "Hello")
        items = mem.to_context_items()
        assert items[0].source == SourceType.CONVERSATION

    def test_context_items_have_correct_priority(self) -> None:
        mem = _make_memory(max_tokens=1000)
        mem.add_turn("user", "Hello")
        items = mem.to_context_items(priority=8)
        assert items[0].priority == 8

    def test_context_items_content_has_no_role_prefix(self) -> None:
        """Content should be raw message text without role prefix.

        The role is conveyed via metadata so downstream formatters
        (Anthropic, OpenAI) can set it in their message structure
        without producing duplicated roles like
        ``{"role": "user", "content": "user: Hello"}``.
        """
        mem = _make_memory(max_tokens=1000)
        mem.add_turn("user", "Hello")
        items = mem.to_context_items()
        assert items[0].content == "Hello"
        assert not items[0].content.startswith("user: ")

    def test_context_items_have_role_metadata(self) -> None:
        mem = _make_memory(max_tokens=1000)
        mem.add_turn("assistant", "response text")
        items = mem.to_context_items()
        assert items[0].metadata["role"] == "assistant"

    def test_empty_memory_returns_empty_list(self) -> None:
        mem = _make_memory(max_tokens=1000)
        assert mem.to_context_items() == []

    def test_context_item_token_count_matches_content(self) -> None:
        """Token count should match the raw content, not a role-prefixed string."""
        mem = _make_memory(max_tokens=1000)
        mem.add_turn("user", "Hello world")
        items = mem.to_context_items()
        tokenizer = FakeTokenizer()
        assert items[0].token_count == tokenizer.count_tokens("Hello world")


class TestSlidingWindowClear:
    """clear resets state."""

    def test_clear_removes_all_turns(self) -> None:
        mem = _make_memory(max_tokens=1000)
        mem.add_turn("user", "Hello")
        mem.add_turn("assistant", "Hi")
        mem.clear()
        assert len(mem.turns) == 0
        assert mem.total_tokens == 0

    def test_clear_allows_adding_new_turns(self) -> None:
        mem = _make_memory(max_tokens=1000)
        mem.add_turn("user", "Hello")
        mem.clear()
        mem.add_turn("user", "New message")
        assert len(mem.turns) == 1
        assert mem.turns[0].content == "New message"


class TestSlidingWindowProperties:
    """Properties: turns, total_tokens, max_tokens."""

    def test_turns_returns_copy(self) -> None:
        mem = _make_memory(max_tokens=1000)
        mem.add_turn("user", "Hello")
        turns = mem.turns
        turns.clear()  # Modifying the copy
        assert len(mem.turns) == 1  # Original unchanged

    def test_max_tokens_property(self) -> None:
        mem = _make_memory(max_tokens=4096)
        assert mem.max_tokens == 4096


class TestSlidingWindowRepr:
    """__repr__ returns a useful string representation."""

    def test_repr_empty(self) -> None:
        mem = _make_memory(max_tokens=1000)
        r = repr(mem)
        assert "SlidingWindowMemory" in r
        assert "turns=0" in r
        assert "tokens=0/1000" in r

    def test_repr_with_turns(self) -> None:
        mem = _make_memory(max_tokens=1000)
        mem.add_turn("user", "Hello")
        r = repr(mem)
        assert "turns=1" in r


class TestSlidingWindowSlots:
    """__slots__ prevents arbitrary attribute assignment."""

    def test_cannot_set_arbitrary_attribute(self) -> None:
        mem = _make_memory(max_tokens=1000)
        with pytest.raises(AttributeError):
            mem.some_random_attr = "oops"  # type: ignore[attr-defined]
