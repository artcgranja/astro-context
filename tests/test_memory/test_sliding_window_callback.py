"""Tests for SlidingWindowMemory on_evict callback error handling."""

from __future__ import annotations

from astro_context.memory.sliding_window import SlidingWindowMemory
from astro_context.models.memory import ConversationTurn
from tests.conftest import FakeTokenizer


def _make_memory(
    max_tokens: int = 10,
    on_evict: object = None,
) -> SlidingWindowMemory:
    """Create a SlidingWindowMemory with a FakeTokenizer and optional callback."""
    return SlidingWindowMemory(
        max_tokens=max_tokens,
        tokenizer=FakeTokenizer(),
        on_evict=on_evict,  # type: ignore[arg-type]
    )


class TestOnEvictCallbackInvocation:
    """on_evict callback is called when turns are evicted."""

    def test_callback_called_on_eviction(self) -> None:
        called = []

        def on_evict(turns: list[ConversationTurn]) -> None:
            called.append(turns)

        mem = _make_memory(max_tokens=3, on_evict=on_evict)
        mem.add_turn("user", "first")       # 1 token
        mem.add_turn("assistant", "second")  # 1 token
        mem.add_turn("user", "three four five")  # 3 tokens -> must evict

        assert len(called) >= 1

    def test_callback_receives_evicted_turns(self) -> None:
        evicted_batches: list[list[ConversationTurn]] = []

        def on_evict(turns: list[ConversationTurn]) -> None:
            evicted_batches.append(turns)

        mem = _make_memory(max_tokens=3, on_evict=on_evict)
        mem.add_turn("user", "first")       # 1 token
        mem.add_turn("assistant", "second")  # 1 token
        # Now at 2 tokens. Adding 3-token message needs to free 2 tokens.
        mem.add_turn("user", "three four five")  # 3 tokens

        assert len(evicted_batches) == 1
        evicted = evicted_batches[0]
        # Should have evicted "first" and "second" (2 tokens total)
        contents = [t.content for t in evicted]
        assert "first" in contents
        assert "second" in contents

    def test_callback_receives_multiple_evicted_turns(self) -> None:
        evicted_batches: list[list[ConversationTurn]] = []

        def on_evict(turns: list[ConversationTurn]) -> None:
            evicted_batches.append(turns)

        # Budget of 5 tokens
        mem = _make_memory(max_tokens=5, on_evict=on_evict)
        mem.add_turn("user", "a")         # 1 token
        mem.add_turn("assistant", "b")     # 1 token
        mem.add_turn("user", "c")          # 1 token
        mem.add_turn("assistant", "d")     # 1 token
        # Now at 4 tokens. Add a 4-token message -> must free 3 tokens.
        mem.add_turn("user", "e f g h")    # 4 tokens

        assert len(evicted_batches) == 1
        batch = evicted_batches[0]
        # Should evict a, b, c (3 tokens) to fit the new 4-token message in a 5-token budget
        assert len(batch) >= 3

    def test_no_callback_when_no_eviction(self) -> None:
        called = []

        def on_evict(turns: list[ConversationTurn]) -> None:
            called.append(True)

        mem = _make_memory(max_tokens=100, on_evict=on_evict)
        mem.add_turn("user", "hello")
        mem.add_turn("assistant", "world")
        assert len(called) == 0

    def test_no_callback_when_not_provided(self) -> None:
        """SlidingWindowMemory works fine without on_evict."""
        mem = _make_memory(max_tokens=3, on_evict=None)
        mem.add_turn("user", "first")
        mem.add_turn("assistant", "second")
        mem.add_turn("user", "three four five")
        # Just verify it doesn't crash
        assert len(mem.turns) >= 1


class TestOnEvictCallbackErrorHandling:
    """on_evict callback errors are caught and do NOT crash the pipeline."""

    def test_callback_error_is_caught(self) -> None:
        def bad_callback(turns: list[ConversationTurn]) -> None:
            msg = "callback exploded"
            raise ValueError(msg)

        mem = _make_memory(max_tokens=3, on_evict=bad_callback)
        mem.add_turn("user", "first")
        mem.add_turn("assistant", "second")
        # This should NOT raise, despite the callback error
        mem.add_turn("user", "three four five")
        # The turn should still be added successfully
        assert mem.turns[-1].content == "three four five"

    def test_runtime_error_is_silently_handled(self) -> None:
        def runtime_callback(turns: list[ConversationTurn]) -> None:
            msg = "runtime failure"
            raise RuntimeError(msg)

        mem = _make_memory(max_tokens=3, on_evict=runtime_callback)
        mem.add_turn("user", "first")
        mem.add_turn("assistant", "second")
        # Should not raise
        mem.add_turn("user", "three four five")
        assert len(mem.turns) >= 1

    def test_type_error_is_silently_handled(self) -> None:
        def type_callback(turns: list[ConversationTurn]) -> None:
            raise TypeError("bad type")

        mem = _make_memory(max_tokens=3, on_evict=type_callback)
        mem.add_turn("user", "first")
        mem.add_turn("assistant", "second")
        mem.add_turn("user", "three four five")
        assert len(mem.turns) >= 1

    def test_eviction_still_happens_despite_callback_error(self) -> None:
        """The old turns should be evicted even if the callback fails."""

        def failing_callback(turns: list[ConversationTurn]) -> None:
            msg = "oops"
            raise Exception(msg)

        mem = _make_memory(max_tokens=3, on_evict=failing_callback)
        mem.add_turn("user", "first")        # 1 token
        mem.add_turn("assistant", "second")   # 1 token
        mem.add_turn("user", "three four five")  # 3 tokens

        # Despite callback failure, eviction should have happened
        # and the new turn should be present
        remaining_contents = [t.content for t in mem.turns]
        assert "three four five" in remaining_contents
        # Old turns should be gone
        assert "first" not in remaining_contents
