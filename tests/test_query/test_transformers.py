"""Tests for built-in query transformers."""

from __future__ import annotations

import pytest

from astro_context.models.query import QueryBundle
from astro_context.protocols.query_transform import AsyncQueryTransformer, QueryTransformer
from astro_context.query.transformers import (
    DecompositionTransformer,
    HyDETransformer,
    MultiQueryTransformer,
    StepBackTransformer,
)


class TestHyDETransformer:
    """Tests for the HyDE (Hypothetical Document Embeddings) transformer."""

    def test_protocol_compliance(self) -> None:
        t = HyDETransformer(generate_fn=lambda q: f"Answer to {q}")
        assert isinstance(t, QueryTransformer)

    def test_transform_returns_single_query(self) -> None:
        t = HyDETransformer(generate_fn=lambda q: f"Hypothetical answer about {q}")
        query = QueryBundle(query_str="What is RAG?")
        result = t.transform(query)
        assert len(result) == 1

    def test_transform_uses_hypothetical_doc_as_query(self) -> None:
        t = HyDETransformer(generate_fn=lambda q: f"Hypothetical: {q}")
        query = QueryBundle(query_str="What is RAG?")
        result = t.transform(query)
        assert result[0].query_str == "Hypothetical: What is RAG?"

    def test_transform_preserves_original_in_metadata(self) -> None:
        t = HyDETransformer(generate_fn=lambda q: "answer")
        query = QueryBundle(query_str="original question")
        result = t.transform(query)
        assert result[0].metadata["original_query"] == "original question"
        assert result[0].metadata["transform"] == "hyde"

    def test_transform_preserves_existing_metadata(self) -> None:
        t = HyDETransformer(generate_fn=lambda q: "answer")
        query = QueryBundle(query_str="test", metadata={"user_id": "u1"})
        result = t.transform(query)
        assert result[0].metadata["user_id"] == "u1"
        assert result[0].metadata["original_query"] == "test"

    def test_repr(self) -> None:
        t = HyDETransformer(generate_fn=lambda q: q)
        assert repr(t) == "HyDETransformer()"

    def test_transform_empty_query(self) -> None:
        t = HyDETransformer(generate_fn=lambda q: f"answer for '{q}'")
        query = QueryBundle(query_str="")
        result = t.transform(query)
        assert len(result) == 1
        assert result[0].query_str == "answer for ''"


class TestMultiQueryTransformer:
    """Tests for the MultiQuery transformer."""

    def test_protocol_compliance(self) -> None:
        t = MultiQueryTransformer(generate_fn=lambda q, n: [f"v{i}" for i in range(n)])
        assert isinstance(t, QueryTransformer)

    def test_transform_returns_original_plus_variations(self) -> None:
        t = MultiQueryTransformer(
            generate_fn=lambda q, n: [f"variation {i} of {q}" for i in range(n)],
            num_queries=3,
        )
        query = QueryBundle(query_str="What is RAG?")
        result = t.transform(query)
        # 1 original + 3 variations
        assert len(result) == 4

    def test_first_result_is_original(self) -> None:
        t = MultiQueryTransformer(
            generate_fn=lambda q, n: [f"v{i}" for i in range(n)],
            num_queries=2,
        )
        query = QueryBundle(query_str="original")
        result = t.transform(query)
        assert result[0].query_str == "original"

    def test_variations_have_metadata(self) -> None:
        t = MultiQueryTransformer(
            generate_fn=lambda q, n: [f"v{i}" for i in range(n)],
            num_queries=2,
        )
        query = QueryBundle(query_str="test")
        result = t.transform(query)
        # Check variations (skip original at index 0)
        for i, r in enumerate(result[1:]):
            assert r.metadata["original_query"] == "test"
            assert r.metadata["transform"] == "multi_query"
            assert r.metadata["variation_index"] == i

    def test_num_queries_validation(self) -> None:
        with pytest.raises(ValueError, match="num_queries must be at least 1"):
            MultiQueryTransformer(generate_fn=lambda q, n: [], num_queries=0)

    def test_default_num_queries(self) -> None:
        calls: list[int] = []

        def gen(q: str, n: int) -> list[str]:
            calls.append(n)
            return [f"v{i}" for i in range(n)]

        t = MultiQueryTransformer(generate_fn=gen)
        t.transform(QueryBundle(query_str="test"))
        assert calls[0] == 3

    def test_repr(self) -> None:
        t = MultiQueryTransformer(generate_fn=lambda q, n: [], num_queries=5)
        assert repr(t) == "MultiQueryTransformer(num_queries=5)"

    def test_preserves_existing_metadata(self) -> None:
        t = MultiQueryTransformer(
            generate_fn=lambda q, n: ["v1"],
            num_queries=1,
        )
        query = QueryBundle(query_str="test", metadata={"session": "s1"})
        result = t.transform(query)
        assert result[1].metadata["session"] == "s1"


class TestDecompositionTransformer:
    """Tests for the Decomposition transformer."""

    def test_protocol_compliance(self) -> None:
        t = DecompositionTransformer(generate_fn=lambda q: [q])
        assert isinstance(t, QueryTransformer)

    def test_transform_produces_sub_questions(self) -> None:
        t = DecompositionTransformer(
            generate_fn=lambda q: ["sub-q1", "sub-q2", "sub-q3"],
        )
        query = QueryBundle(query_str="complex question")
        result = t.transform(query)
        assert len(result) == 3
        assert result[0].query_str == "sub-q1"
        assert result[1].query_str == "sub-q2"
        assert result[2].query_str == "sub-q3"

    def test_sub_questions_have_parent_metadata(self) -> None:
        t = DecompositionTransformer(generate_fn=lambda q: ["sub1", "sub2"])
        query = QueryBundle(query_str="parent question")
        result = t.transform(query)
        for i, r in enumerate(result):
            assert r.metadata["parent_query"] == "parent question"
            assert r.metadata["transform"] == "decomposition"
            assert r.metadata["sub_question_index"] == i

    def test_empty_decomposition(self) -> None:
        t = DecompositionTransformer(generate_fn=lambda q: [])
        query = QueryBundle(query_str="test")
        result = t.transform(query)
        assert len(result) == 0

    def test_single_decomposition(self) -> None:
        t = DecompositionTransformer(generate_fn=lambda q: ["only one"])
        query = QueryBundle(query_str="test")
        result = t.transform(query)
        assert len(result) == 1
        assert result[0].query_str == "only one"

    def test_repr(self) -> None:
        t = DecompositionTransformer(generate_fn=lambda q: [])
        assert repr(t) == "DecompositionTransformer()"

    def test_preserves_existing_metadata(self) -> None:
        t = DecompositionTransformer(generate_fn=lambda q: ["sub1"])
        query = QueryBundle(query_str="test", metadata={"key": "val"})
        result = t.transform(query)
        assert result[0].metadata["key"] == "val"


class TestStepBackTransformer:
    """Tests for the StepBack transformer."""

    def test_protocol_compliance(self) -> None:
        t = StepBackTransformer(generate_fn=lambda q: f"general: {q}")
        assert isinstance(t, QueryTransformer)

    def test_transform_returns_two_queries(self) -> None:
        t = StepBackTransformer(generate_fn=lambda q: f"abstract: {q}")
        query = QueryBundle(query_str="specific question")
        result = t.transform(query)
        assert len(result) == 2

    def test_first_is_original_second_is_step_back(self) -> None:
        t = StepBackTransformer(generate_fn=lambda q: f"broader: {q}")
        query = QueryBundle(query_str="specific question")
        result = t.transform(query)
        assert result[0].query_str == "specific question"
        assert result[1].query_str == "broader: specific question"

    def test_step_back_has_metadata(self) -> None:
        t = StepBackTransformer(generate_fn=lambda q: "abstract")
        query = QueryBundle(query_str="original")
        result = t.transform(query)
        assert result[1].metadata["original_query"] == "original"
        assert result[1].metadata["transform"] == "step_back"

    def test_original_unchanged(self) -> None:
        original = QueryBundle(query_str="test", metadata={"key": "val"})
        t = StepBackTransformer(generate_fn=lambda q: "abstract")
        result = t.transform(original)
        # First result should be the original query object
        assert result[0] is original

    def test_repr(self) -> None:
        t = StepBackTransformer(generate_fn=lambda q: q)
        assert repr(t) == "StepBackTransformer()"

    def test_preserves_existing_metadata_in_step_back(self) -> None:
        t = StepBackTransformer(generate_fn=lambda q: "abstract")
        query = QueryBundle(query_str="test", metadata={"session": "s1"})
        result = t.transform(query)
        assert result[1].metadata["session"] == "s1"


class TestProtocolNotAsyncByDefault:
    """Verify sync transformers do not satisfy AsyncQueryTransformer."""

    def test_hyde_is_not_async(self) -> None:
        t = HyDETransformer(generate_fn=lambda q: q)
        assert not isinstance(t, AsyncQueryTransformer)

    def test_multi_query_is_not_async(self) -> None:
        t = MultiQueryTransformer(generate_fn=lambda q, n: [])
        assert not isinstance(t, AsyncQueryTransformer)

    def test_decomposition_is_not_async(self) -> None:
        t = DecompositionTransformer(generate_fn=lambda q: [])
        assert not isinstance(t, AsyncQueryTransformer)

    def test_step_back_is_not_async(self) -> None:
        t = StepBackTransformer(generate_fn=lambda q: q)
        assert not isinstance(t, AsyncQueryTransformer)
