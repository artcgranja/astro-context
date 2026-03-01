"""Tests for the classified_retriever_step pipeline step."""

from __future__ import annotations

import pytest

from astro_context.exceptions import RetrieverError
from astro_context.models.context import ContextItem
from astro_context.models.memory import ConversationTurn
from astro_context.models.query import QueryBundle
from astro_context.pipeline.step import classified_retriever_step
from astro_context.query.classifiers import KeywordClassifier
from astro_context.query.rewriter import ConversationRewriter
from tests.conftest import FakeRetriever


class TestClassifiedRetrieverStep:
    """Tests for classified_retriever_step factory function."""

    def test_basic_classification_and_retrieval(self) -> None:
        tech_items = [ContextItem(id="t1", content="tech doc", source="retrieval")]
        science_items = [ContextItem(id="s1", content="science doc", source="retrieval")]

        classifier = KeywordClassifier(
            rules={"tech": ["python"], "science": ["biology"]},
            default="general",
        )
        retrievers = {
            "tech": FakeRetriever(tech_items),
            "science": FakeRetriever(science_items),
        }
        step = classified_retriever_step("classify", classifier, retrievers)
        result = step.execute([], QueryBundle(query_str="python decorators"))
        assert len(result) == 1
        assert result[0].id == "t1"

    def test_default_fallback(self) -> None:
        general_items = [ContextItem(id="g1", content="general doc", source="retrieval")]
        classifier = KeywordClassifier(
            rules={"tech": ["python"]},
            default="general",
        )
        retrievers = {
            "tech": FakeRetriever([]),
            "general_retriever": FakeRetriever(general_items),
        }
        step = classified_retriever_step(
            "classify",
            classifier,
            retrievers,
            default="general_retriever",
        )
        # "meaning of life" won't match "python", so classifier returns "general"
        # "general" is not in retrievers, but default="general_retriever" is
        result = step.execute([], QueryBundle(query_str="meaning of life"))
        assert len(result) == 1
        assert result[0].id == "g1"

    def test_unknown_class_no_default_raises(self) -> None:
        classifier = KeywordClassifier(
            rules={"tech": ["python"]},
            default="unknown",
        )
        retrievers = {"tech": FakeRetriever([])}
        step = classified_retriever_step("classify", classifier, retrievers)
        with pytest.raises(RetrieverError, match="No retriever found"):
            step.execute([], QueryBundle(query_str="meaning of life"))

    def test_step_name(self) -> None:
        classifier = KeywordClassifier(rules={}, default="x")
        step = classified_retriever_step("my-step", classifier, {"x": FakeRetriever([])})
        assert step.name == "my-step"

    def test_preserves_existing_items(self) -> None:
        existing = [ContextItem(id="existing", content="already here", source="system")]
        new_items = [ContextItem(id="new", content="new doc", source="retrieval")]
        classifier = KeywordClassifier(rules={"tech": ["python"]}, default="general")
        retrievers = {"tech": FakeRetriever(new_items)}
        step = classified_retriever_step("classify", classifier, retrievers)
        result = step.execute(existing, QueryBundle(query_str="python"))
        assert len(result) == 2
        assert result[0].id == "existing"
        assert result[1].id == "new"

    def test_top_k_respected(self) -> None:
        items = [
            ContextItem(id=f"item-{i}", content=f"doc {i}", source="retrieval") for i in range(10)
        ]
        classifier = KeywordClassifier(rules={"all": ["test"]}, default="all")
        retrievers = {"all": FakeRetriever(items)}
        step = classified_retriever_step("classify", classifier, retrievers, top_k=3)
        result = step.execute([], QueryBundle(query_str="test query"))
        assert len(result) == 3

    def test_multiple_queries_route_to_different_retrievers(self) -> None:
        """Different queries should be routed to different retrievers in sequence."""
        tech_items = [ContextItem(id="tech1", content="tech doc", source="retrieval")]
        science_items = [ContextItem(id="sci1", content="science doc", source="retrieval")]

        classifier = KeywordClassifier(
            rules={"tech": ["python"], "science": ["biology"]},
            default="general",
        )
        retrievers = {
            "tech": FakeRetriever(tech_items),
            "science": FakeRetriever(science_items),
        }
        step = classified_retriever_step("router", classifier, retrievers)

        # First query routes to tech
        result1 = step.execute([], QueryBundle(query_str="python programming"))
        assert len(result1) == 1
        assert result1[0].id == "tech1"

        # Second query routes to science
        result2 = step.execute([], QueryBundle(query_str="biology evolution"))
        assert len(result2) == 1
        assert result2[0].id == "sci1"

    def test_rewriter_then_classifier_chain(self) -> None:
        """Rewrite a query using conversation context, then classify it."""

        # Rewriter that expands the query with context from history
        def rewrite(q: str, history: list[ConversationTurn]) -> str:
            last_topic = history[-1].content
            return f"{last_topic} {q}"

        rewriter = ConversationRewriter(rewrite_fn=rewrite)

        turns = [ConversationTurn(role="user", content="python")]
        query = QueryBundle(query_str="how to use decorators?", chat_history=turns)

        # Rewrite: "python how to use decorators?"
        rewritten = rewriter.transform(query)[0]
        assert "python" in rewritten.query_str

        # Now classify the rewritten query
        tech_items = [ContextItem(id="t1", content="decorator guide", source="retrieval")]
        classifier = KeywordClassifier(
            rules={"tech": ["python"], "science": ["biology"]},
            default="general",
        )
        retrievers = {"tech": FakeRetriever(tech_items)}
        step = classified_retriever_step("classify", classifier, retrievers)
        result = step.execute([], rewritten)
        assert len(result) == 1
        assert result[0].id == "t1"

    def test_retriever_returns_empty_list(self) -> None:
        """Step works correctly when the matched retriever returns no items."""
        classifier = KeywordClassifier(
            rules={"tech": ["python"]},
            default="general",
        )
        retrievers = {
            "tech": FakeRetriever([]),  # empty retriever
        }
        step = classified_retriever_step("classify", classifier, retrievers)
        result = step.execute([], QueryBundle(query_str="python"))
        assert result == []
