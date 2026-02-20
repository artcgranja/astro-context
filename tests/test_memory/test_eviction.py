"""Tests for astro_context.memory.eviction."""

from __future__ import annotations

from astro_context.memory.eviction import FIFOEviction, ImportanceEviction, PairedEviction
from astro_context.models.memory import ConversationTurn


def _make_turn(role: str, content: str, token_count: int) -> ConversationTurn:
    """Create a ConversationTurn with an explicit token count."""
    return ConversationTurn(
        role=role,  # type: ignore[arg-type]
        content=content,
        token_count=token_count,
    )


class TestFIFOEviction:
    """FIFOEviction evicts oldest turns first."""

    def test_evicts_oldest_first(self) -> None:
        turns = [
            _make_turn("user", "first", token_count=5),
            _make_turn("assistant", "second", token_count=5),
            _make_turn("user", "third", token_count=5),
        ]
        policy = FIFOEviction()
        indices = policy.select_for_eviction(turns, tokens_to_free=6)
        # Should evict index 0 (5 tokens) then index 1 (5 tokens) to free >= 6
        assert indices == [0, 1]

    def test_evicts_minimum_to_free_required(self) -> None:
        turns = [
            _make_turn("user", "first", token_count=10),
            _make_turn("assistant", "second", token_count=10),
            _make_turn("user", "third", token_count=10),
        ]
        policy = FIFOEviction()
        # Need only 5 tokens, first turn has 10 -> evict just index 0
        indices = policy.select_for_eviction(turns, tokens_to_free=5)
        assert indices == [0]

    def test_single_turn(self) -> None:
        turns = [_make_turn("user", "only turn", token_count=8)]
        policy = FIFOEviction()
        indices = policy.select_for_eviction(turns, tokens_to_free=5)
        assert indices == [0]

    def test_empty_turns(self) -> None:
        policy = FIFOEviction()
        indices = policy.select_for_eviction([], tokens_to_free=10)
        assert indices == []

    def test_no_tokens_needed(self) -> None:
        turns = [_make_turn("user", "hello", token_count=5)]
        policy = FIFOEviction()
        indices = policy.select_for_eviction(turns, tokens_to_free=0)
        assert indices == []

    def test_evict_all_if_needed(self) -> None:
        turns = [
            _make_turn("user", "a", token_count=3),
            _make_turn("assistant", "b", token_count=3),
            _make_turn("user", "c", token_count=3),
        ]
        policy = FIFOEviction()
        indices = policy.select_for_eviction(turns, tokens_to_free=100)
        assert indices == [0, 1, 2]

    def test_indices_within_bounds(self) -> None:
        turns = [
            _make_turn("user", f"turn {i}", token_count=2) for i in range(10)
        ]
        policy = FIFOEviction()
        indices = policy.select_for_eviction(turns, tokens_to_free=7)
        assert all(0 <= idx < len(turns) for idx in indices)


class TestImportanceEviction:
    """ImportanceEviction evicts lowest-importance turns first."""

    def test_evicts_lowest_importance(self) -> None:
        turns = [
            _make_turn("user", "important", token_count=5),
            _make_turn("assistant", "trivial", token_count=5),
            _make_turn("user", "medium", token_count=5),
        ]
        # Importance: "important"=10, "trivial"=1, "medium"=5
        importance = {"important": 10.0, "trivial": 1.0, "medium": 5.0}
        policy = ImportanceEviction(
            importance_fn=lambda t: importance.get(t.content, 0.0),
        )
        indices = policy.select_for_eviction(turns, tokens_to_free=5)
        # Should evict index 1 ("trivial", lowest importance) first
        assert 1 in indices

    def test_evicts_multiple_lowest_importance(self) -> None:
        turns = [
            _make_turn("user", "a", token_count=3),
            _make_turn("assistant", "b", token_count=3),
            _make_turn("user", "c", token_count=3),
            _make_turn("assistant", "d", token_count=3),
        ]
        # Importance: a=1, b=4, c=2, d=3
        scores = {"a": 1.0, "b": 4.0, "c": 2.0, "d": 3.0}
        policy = ImportanceEviction(
            importance_fn=lambda t: scores.get(t.content, 0.0),
        )
        indices = policy.select_for_eviction(turns, tokens_to_free=6)
        # Should evict "a" (importance 1) and "c" (importance 2) -> 6 tokens
        assert 0 in indices  # "a"
        assert 2 in indices  # "c"

    def test_custom_importance_function(self) -> None:
        turns = [
            _make_turn("user", "short", token_count=2),
            _make_turn("assistant", "a much longer message here", token_count=10),
        ]
        # Importance is based on token count (longer = more important)
        policy = ImportanceEviction(importance_fn=lambda t: float(t.token_count))
        indices = policy.select_for_eviction(turns, tokens_to_free=1)
        # Should evict the shorter one (index 0, token_count=2)
        assert indices == [0]

    def test_indices_within_bounds(self) -> None:
        turns = [_make_turn("user", f"t{i}", token_count=2) for i in range(8)]
        policy = ImportanceEviction(importance_fn=lambda t: float(len(t.content)))
        indices = policy.select_for_eviction(turns, tokens_to_free=5)
        assert all(0 <= idx < len(turns) for idx in indices)


class TestPairedEviction:
    """PairedEviction evicts user+assistant pairs together."""

    def test_evicts_user_assistant_pair(self) -> None:
        turns = [
            _make_turn("user", "question 1", token_count=3),
            _make_turn("assistant", "answer 1", token_count=3),
            _make_turn("user", "question 2", token_count=3),
            _make_turn("assistant", "answer 2", token_count=3),
        ]
        policy = PairedEviction()
        indices = policy.select_for_eviction(turns, tokens_to_free=5)
        # Should evict the first pair (indices 0 and 1) as a unit
        assert indices == [0, 1]

    def test_evicts_multiple_pairs(self) -> None:
        turns = [
            _make_turn("user", "q1", token_count=2),
            _make_turn("assistant", "a1", token_count=2),
            _make_turn("user", "q2", token_count=2),
            _make_turn("assistant", "a2", token_count=2),
            _make_turn("user", "q3", token_count=2),
            _make_turn("assistant", "a3", token_count=2),
        ]
        policy = PairedEviction()
        indices = policy.select_for_eviction(turns, tokens_to_free=7)
        # Need 7 tokens; first pair = 4 tokens, second pair = 4 tokens -> evict both pairs
        assert indices == [0, 1, 2, 3]

    def test_handles_lone_turn_at_boundary(self) -> None:
        turns = [
            _make_turn("assistant", "orphaned answer", token_count=5),
            _make_turn("user", "question", token_count=3),
            _make_turn("assistant", "answer", token_count=3),
        ]
        policy = PairedEviction()
        indices = policy.select_for_eviction(turns, tokens_to_free=4)
        # First turn is lone assistant (5 tokens), enough to free 4
        assert indices == [0]

    def test_handles_lone_user_at_end(self) -> None:
        turns = [
            _make_turn("user", "q1", token_count=3),
            _make_turn("assistant", "a1", token_count=3),
            _make_turn("user", "trailing question", token_count=3),
        ]
        policy = PairedEviction()
        # The pair [0,1] = 6 tokens. The lone user [2] = 3 tokens.
        # Need 7 tokens: evict pair + lone user
        indices = policy.select_for_eviction(turns, tokens_to_free=7)
        assert indices == [0, 1, 2]

    def test_system_turns_treated_as_lone(self) -> None:
        turns = [
            _make_turn("system", "system prompt", token_count=5),
            _make_turn("user", "hello", token_count=3),
            _make_turn("assistant", "hi", token_count=3),
        ]
        policy = PairedEviction()
        indices = policy.select_for_eviction(turns, tokens_to_free=4)
        # System turn is lone (5 tokens), enough to free 4
        assert indices == [0]

    def test_empty_turns(self) -> None:
        policy = PairedEviction()
        indices = policy.select_for_eviction([], tokens_to_free=10)
        assert indices == []

    def test_indices_within_bounds(self) -> None:
        turns = [
            _make_turn("user", f"q{i}", token_count=2)
            for i in range(6)
        ]
        policy = PairedEviction()
        indices = policy.select_for_eviction(turns, tokens_to_free=5)
        assert all(0 <= idx < len(turns) for idx in indices)
