"""Tests for QueryBundle backward compatibility with chat_history field."""

from __future__ import annotations

from astro_context.models.memory import ConversationTurn
from astro_context.models.query import QueryBundle


class TestQueryBundleChatHistoryCompat:
    """Verify QueryBundle works with and without chat_history."""

    def test_no_chat_history_still_works(self) -> None:
        """Existing code that omits chat_history must not break."""
        q = QueryBundle(query_str="hello world")
        assert q.query_str == "hello world"
        assert q.embedding is None
        assert q.metadata == {}
        assert q.chat_history == []

    def test_model_dump_includes_chat_history_empty(self) -> None:
        """model_dump() output includes chat_history even when empty."""
        q = QueryBundle(query_str="test")
        data = q.model_dump()
        assert "chat_history" in data
        assert data["chat_history"] == []

    def test_model_dump_includes_chat_history_with_turns(self) -> None:
        """model_dump() correctly serializes populated chat_history."""
        turns = [
            ConversationTurn(role="user", content="What is RAG?"),
            ConversationTurn(role="assistant", content="RAG is retrieval-augmented generation."),
        ]
        q = QueryBundle(query_str="Tell me more", chat_history=turns)
        data = q.model_dump()
        assert len(data["chat_history"]) == 2
        assert data["chat_history"][0]["role"] == "user"
        assert data["chat_history"][0]["content"] == "What is RAG?"
        assert data["chat_history"][1]["role"] == "assistant"

    def test_round_trip_serialization(self) -> None:
        """QueryBundle with chat_history survives model_dump -> model_validate."""
        turns = [
            ConversationTurn(role="user", content="Hi"),
            ConversationTurn(role="assistant", content="Hello!"),
            ConversationTurn(role="user", content="How are you?"),
        ]
        original = QueryBundle(
            query_str="follow up",
            embedding=[0.1, 0.2, 0.3],
            metadata={"source": "test"},
            chat_history=turns,
        )
        data = original.model_dump()
        restored = QueryBundle.model_validate(data)
        assert restored.query_str == original.query_str
        assert restored.embedding == original.embedding
        assert restored.metadata == original.metadata
        assert len(restored.chat_history) == 3
        assert restored.chat_history[0].role == "user"
        assert restored.chat_history[2].content == "How are you?"

    def test_without_chat_history_round_trip(self) -> None:
        """Old-style QueryBundle (no chat_history) round-trips cleanly."""
        original = QueryBundle(query_str="old style", embedding=[1.0, 2.0])
        data = original.model_dump()
        restored = QueryBundle.model_validate(data)
        assert restored.query_str == "old style"
        assert restored.chat_history == []
