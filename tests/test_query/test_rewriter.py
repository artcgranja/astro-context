"""Tests for conversation-aware query rewriters."""

from __future__ import annotations

from astro_context.models.memory import ConversationTurn
from astro_context.models.query import QueryBundle
from astro_context.protocols.query_transform import QueryTransformer
from astro_context.query.rewriter import ContextualQueryTransformer, ConversationRewriter
from astro_context.query.transformers import HyDETransformer


class TestConversationRewriter:
    """Tests for the ConversationRewriter class."""

    def test_protocol_compliance(self) -> None:
        rewriter = ConversationRewriter(rewrite_fn=lambda q, h: q)
        assert isinstance(rewriter, QueryTransformer)

    def test_empty_history_passthrough(self) -> None:
        rewriter = ConversationRewriter(rewrite_fn=lambda q, h: f"rewritten: {q}")
        query = QueryBundle(query_str="hello")
        result = rewriter.transform(query)
        assert len(result) == 1
        assert result[0] is query  # unchanged, returns original object

    def test_rewrite_with_history(self) -> None:
        def rewrite(q: str, history: list[ConversationTurn]) -> str:
            return f"[{len(history)} turns] {q}"

        rewriter = ConversationRewriter(rewrite_fn=rewrite)
        turns = [
            ConversationTurn(role="user", content="Hi"),
            ConversationTurn(role="assistant", content="Hello!"),
        ]
        query = QueryBundle(query_str="What about X?", chat_history=turns)
        result = rewriter.transform(query)
        assert len(result) == 1
        assert result[0].query_str == "[2 turns] What about X?"

    def test_metadata_propagation(self) -> None:
        rewriter = ConversationRewriter(rewrite_fn=lambda q, h: f"rewritten: {q}")
        turns = [ConversationTurn(role="user", content="context")]
        query = QueryBundle(
            query_str="test",
            metadata={"session": "s1"},
            chat_history=turns,
        )
        result = rewriter.transform(query)
        assert result[0].metadata["session"] == "s1"
        assert result[0].metadata["original_query"] == "test"
        assert result[0].metadata["transform"] == "conversation_rewrite"

    def test_embedding_preserved(self) -> None:
        rewriter = ConversationRewriter(rewrite_fn=lambda q, h: "rewritten")
        turns = [ConversationTurn(role="user", content="ctx")]
        query = QueryBundle(
            query_str="test",
            embedding=[1.0, 2.0, 3.0],
            chat_history=turns,
        )
        result = rewriter.transform(query)
        assert result[0].embedding == [1.0, 2.0, 3.0]

    def test_chat_history_preserved_in_result(self) -> None:
        rewriter = ConversationRewriter(rewrite_fn=lambda q, h: "rewritten")
        turns = [ConversationTurn(role="user", content="ctx")]
        query = QueryBundle(query_str="test", chat_history=turns)
        result = rewriter.transform(query)
        assert result[0].chat_history == turns

    def test_repr(self) -> None:
        rewriter = ConversationRewriter(rewrite_fn=lambda q, h: q)
        assert repr(rewriter) == "ConversationRewriter()"

    def test_rewrite_fn_receives_correct_args(self) -> None:
        captured: list[tuple[str, list[ConversationTurn]]] = []

        def rewrite(q: str, h: list[ConversationTurn]) -> str:
            captured.append((q, h))
            return q

        turns = [ConversationTurn(role="user", content="hello")]
        rewriter = ConversationRewriter(rewrite_fn=rewrite)
        rewriter.transform(QueryBundle(query_str="question", chat_history=turns))
        assert len(captured) == 1
        assert captured[0][0] == "question"
        assert captured[0][1] == turns

    def test_single_turn_history(self) -> None:
        """Rewriter works correctly with just 1 turn in chat_history."""

        def rewrite(q: str, history: list[ConversationTurn]) -> str:
            return f"[{len(history)} turn] {q}"

        rewriter = ConversationRewriter(rewrite_fn=rewrite)
        turns = [ConversationTurn(role="user", content="Setup context")]
        query = QueryBundle(query_str="follow up?", chat_history=turns)
        result = rewriter.transform(query)
        assert len(result) == 1
        assert result[0].query_str == "[1 turn] follow up?"

    def test_long_history_20_plus_turns(self) -> None:
        """Rewriter handles 20+ turns without issue."""
        turns = [
            ConversationTurn(
                role="user" if i % 2 == 0 else "assistant",
                content=f"Message number {i}",
            )
            for i in range(25)
        ]

        def rewrite(q: str, history: list[ConversationTurn]) -> str:
            return f"[{len(history)} turns] {q}"

        rewriter = ConversationRewriter(rewrite_fn=rewrite)
        query = QueryBundle(query_str="final question", chat_history=turns)
        result = rewriter.transform(query)
        assert len(result) == 1
        assert result[0].query_str == "[25 turns] final question"
        assert len(result[0].chat_history) == 25

    def test_metadata_preserved_after_rewrite(self) -> None:
        """All original metadata keys survive rewriting alongside injected keys."""
        rewriter = ConversationRewriter(rewrite_fn=lambda q, h: "rewritten")
        turns = [ConversationTurn(role="user", content="ctx")]
        query = QueryBundle(
            query_str="test",
            metadata={"user_id": "u42", "session": "s7", "priority": 3},
            chat_history=turns,
        )
        result = rewriter.transform(query)
        # Original metadata preserved
        assert result[0].metadata["user_id"] == "u42"
        assert result[0].metadata["session"] == "s7"
        assert result[0].metadata["priority"] == 3
        # Rewrite metadata injected
        assert result[0].metadata["original_query"] == "test"
        assert result[0].metadata["transform"] == "conversation_rewrite"

    def test_empty_query_string_with_history(self) -> None:
        """Empty query string with non-empty history still triggers rewrite."""

        def rewrite(q: str, history: list[ConversationTurn]) -> str:
            last_content = history[-1].content
            return f"Continue from: {last_content}"

        rewriter = ConversationRewriter(rewrite_fn=rewrite)
        turns = [ConversationTurn(role="user", content="Tell me about RAG")]
        query = QueryBundle(query_str="", chat_history=turns)
        result = rewriter.transform(query)
        assert len(result) == 1
        assert result[0].query_str == "Continue from: Tell me about RAG"
        assert result[0].metadata["original_query"] == ""


class TestContextualQueryTransformer:
    """Tests for the ContextualQueryTransformer class."""

    def test_protocol_compliance(self) -> None:
        inner = HyDETransformer(generate_fn=lambda q: q)
        t = ContextualQueryTransformer(inner=inner)
        assert isinstance(t, QueryTransformer)

    def test_empty_history_delegates_directly(self) -> None:
        inner = HyDETransformer(generate_fn=lambda q: f"hyp: {q}")
        t = ContextualQueryTransformer(inner=inner)
        query = QueryBundle(query_str="original question")
        result = t.transform(query)
        assert len(result) == 1
        assert result[0].query_str == "hyp: original question"

    def test_context_prepended_with_history(self) -> None:
        captured_queries: list[str] = []

        class CapturingTransformer:
            def transform(self, query: QueryBundle) -> list[QueryBundle]:
                captured_queries.append(query.query_str)
                return [query]

        turns = [
            ConversationTurn(role="user", content="Tell me about dogs"),
            ConversationTurn(role="assistant", content="Dogs are great pets"),
        ]
        t = ContextualQueryTransformer(inner=CapturingTransformer())
        query = QueryBundle(query_str="What breeds?", chat_history=turns)
        t.transform(query)

        assert len(captured_queries) == 1
        assert "user: Tell me about dogs" in captured_queries[0]
        assert "assistant: Dogs are great pets" in captured_queries[0]
        assert "What breeds?" in captured_queries[0]

    def test_custom_prefix(self) -> None:
        captured: list[str] = []

        class CapturingTransformer:
            def transform(self, query: QueryBundle) -> list[QueryBundle]:
                captured.append(query.query_str)
                return [query]

        turns = [ConversationTurn(role="user", content="hi")]
        t = ContextualQueryTransformer(
            inner=CapturingTransformer(),
            context_prefix="Context: ",
        )
        t.transform(QueryBundle(query_str="test", chat_history=turns))
        assert captured[0].startswith("Context: ")

    def test_repr(self) -> None:
        inner = HyDETransformer(generate_fn=lambda q: q)
        t = ContextualQueryTransformer(inner=inner)
        r = repr(t)
        assert "ContextualQueryTransformer" in r
        assert "HyDETransformer" in r

    def test_metadata_preserved(self) -> None:
        class IdentityTransformer:
            def transform(self, query: QueryBundle) -> list[QueryBundle]:
                return [query]

        turns = [ConversationTurn(role="user", content="hi")]
        t = ContextualQueryTransformer(inner=IdentityTransformer())
        query = QueryBundle(
            query_str="test",
            metadata={"key": "val"},
            chat_history=turns,
        )
        result = t.transform(query)
        assert result[0].metadata["key"] == "val"
