"""Tests for late interaction retrieval (ColBERT-style MaxSim)."""

from __future__ import annotations

import math

from astro_context.models.context import ContextItem, SourceType
from astro_context.models.query import QueryBundle
from astro_context.protocols.late_interaction import TokenLevelEncoder
from astro_context.protocols.retriever import Retriever
from astro_context.retrieval.late_interaction import (
    LateInteractionRetriever,
    LateInteractionScorer,
    MaxSimScorer,
)


class _FakeTokenLevelEncoder:
    """Produces one embedding per word using a simple hash."""

    def encode_tokens(self, text: str) -> list[list[float]]:
        words = text.split()
        return [[float(hash(w) % 100) / 100.0, float(hash(w) % 50) / 50.0] for w in words]


class _ControlledTokenEncoder:
    """Token encoder that returns predetermined embeddings per content string."""

    def __init__(self, mapping: dict[str, list[list[float]]]) -> None:
        self._mapping = mapping

    def encode_tokens(self, text: str) -> list[list[float]]:
        return self._mapping.get(text, [[0.0, 0.0]])


class _FakeFirstStageRetriever:
    """Returns pre-configured items as first-stage candidates."""

    def __init__(self, items: list[ContextItem]) -> None:
        self._items = items

    def retrieve(self, query: QueryBundle, top_k: int = 10) -> list[ContextItem]:
        return self._items[:top_k]


class _EmptyFirstStageRetriever:
    """Always returns an empty list of candidates."""

    def retrieve(self, query: QueryBundle, top_k: int = 10) -> list[ContextItem]:
        return []


def _make_item(content: str, item_id: str | None = None) -> ContextItem:
    kwargs: dict = {"content": content, "source": SourceType.RETRIEVAL}
    if item_id is not None:
        kwargs["id"] = item_id
    return ContextItem(**kwargs)


class TestTokenLevelEncoderProtocol:
    def test_token_level_encoder_protocol(self) -> None:
        encoder = _FakeTokenLevelEncoder()
        assert isinstance(encoder, TokenLevelEncoder)


class TestMaxSimScorer:
    def test_maxsim_basic(self) -> None:
        scorer = MaxSimScorer()
        query_tokens = [[1.0, 0.0], [0.0, 1.0]]
        doc_tokens = [[1.0, 0.0], [0.0, 1.0]]
        score = scorer.score(query_tokens, doc_tokens)
        # Each query token perfectly matches one doc token: 1.0 + 1.0 = 2.0
        assert score == 2.0

    def test_maxsim_empty_query(self) -> None:
        scorer = MaxSimScorer()
        assert scorer.score([], [[1.0, 0.0]]) == 0.0

    def test_maxsim_empty_doc(self) -> None:
        scorer = MaxSimScorer()
        assert scorer.score([[1.0, 0.0]], []) == 0.0

    def test_maxsim_identical(self) -> None:
        scorer = MaxSimScorer()
        tokens = [[0.5, 0.8], [0.3, 0.6], [0.9, 0.1]]
        score = scorer.score(tokens, tokens)
        # Each query token finds itself in doc tokens with cosine sim 1.0
        assert abs(score - 3.0) < 1e-9


class TestMaxSimScorerMathVerification:
    """Hand-calculated mathematical verification of MaxSim scoring."""

    def test_known_identity_vectors_hand_calculated(self) -> None:
        """query=[[1,0],[0,1]], doc=[[1,0],[0,1]] -> score = 2.0.

        q[0]=[1,0] vs d[0]=[1,0]: cos=1.0, vs d[1]=[0,1]: cos=0.0 -> max=1.0
        q[1]=[0,1] vs d[0]=[1,0]: cos=0.0, vs d[1]=[0,1]: cos=1.0 -> max=1.0
        Total = 1.0 + 1.0 = 2.0
        """
        scorer = MaxSimScorer()
        score = scorer.score([[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]])
        assert score == 2.0

    def test_orthogonal_vectors_score_zero(self) -> None:
        """Perpendicular query and doc tokens should produce score ~0.0.

        q=[[1,0]], d=[[0,1]]: cos(90 degrees) = 0.0
        """
        scorer = MaxSimScorer()
        score = scorer.score([[1.0, 0.0]], [[0.0, 1.0]])
        assert abs(score) < 1e-9

    def test_single_query_token_multiple_doc_tokens(self) -> None:
        """Single query token picks the max similarity across doc tokens.

        q=[[1,0]], d=[[0.5, 0.866], [1,0], [0,1]]
        cos(q, d0) = 0.5/1 = 0.5
        cos(q, d1) = 1.0/1 = 1.0
        cos(q, d2) = 0.0/1 = 0.0
        Max = 1.0
        """
        scorer = MaxSimScorer()
        score = scorer.score(
            [[1.0, 0.0]],
            [[0.5, 0.866], [1.0, 0.0], [0.0, 1.0]],
        )
        assert abs(score - 1.0) < 1e-9

    def test_negative_vector_components(self) -> None:
        """Negative components should produce negative cosine similarity.

        q=[[1,0]], d=[[-1,0]]: cos = -1.0 (antiparallel)
        """
        scorer = MaxSimScorer()
        score = scorer.score([[1.0, 0.0]], [[-1.0, 0.0]])
        assert abs(score - (-1.0)) < 1e-9

    def test_negative_components_mixed(self) -> None:
        """Mixed negative components, hand-calculated.

        q=[[3,4]], d=[[4,3]]:
        dot = 12 + 12 = 24
        mag_q = 5, mag_d = 5
        cos = 24/25 = 0.96
        """
        scorer = MaxSimScorer()
        score = scorer.score([[3.0, 4.0]], [[4.0, 3.0]])
        assert abs(score - 0.96) < 1e-9

    def test_zero_vector_in_query(self) -> None:
        """Zero query vector should contribute 0.0 to the sum."""
        scorer = MaxSimScorer()
        score = scorer.score([[0.0, 0.0]], [[1.0, 0.0]])
        assert score == 0.0

    def test_zero_vector_in_doc(self) -> None:
        """Zero doc vector: cosine is 0.0 for that pair.

        If the only doc token is zero, max sim for any query token is 0.0.
        """
        scorer = MaxSimScorer()
        score = scorer.score([[1.0, 0.0]], [[0.0, 0.0]])
        assert score == 0.0

    def test_zero_vectors_both(self) -> None:
        """Both zero vectors should produce 0.0."""
        scorer = MaxSimScorer()
        score = scorer.score([[0.0, 0.0]], [[0.0, 0.0]])
        assert score == 0.0

    def test_high_dimensional_hand_calculated(self) -> None:
        """3D vectors hand-calculated.

        q=[[1,0,0]], d=[[0,1,0], [1/sqrt(2), 1/sqrt(2), 0]]
        cos(q, d0) = 0
        cos(q, d1) = 1/sqrt(2) / (1 * 1) = 1/sqrt(2) ~= 0.7071
        Max = 0.7071
        """
        scorer = MaxSimScorer()
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        score = scorer.score(
            [[1.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0], [inv_sqrt2, inv_sqrt2, 0.0]],
        )
        assert abs(score - inv_sqrt2) < 1e-9


class TestLateInteractionScorer:
    def test_late_interaction_scorer_default(self) -> None:
        scorer = LateInteractionScorer()
        query_tokens = [[1.0, 0.0]]
        doc_tokens = [[1.0, 0.0]]
        score = scorer.score(query_tokens, doc_tokens)
        assert score == 1.0

    def test_late_interaction_scorer_custom(self) -> None:
        def custom_fn(q: list[list[float]], d: list[list[float]]) -> float:
            return 42.0

        scorer = LateInteractionScorer(score_fn=custom_fn)
        assert scorer.score([[1.0]], [[2.0]]) == 42.0


class TestLateInteractionRetriever:
    def test_late_interaction_retriever_protocol(self) -> None:
        items = [_make_item("hello world")]
        retriever = LateInteractionRetriever(
            first_stage=_FakeFirstStageRetriever(items),
            encoder=_FakeTokenLevelEncoder(),
        )
        assert isinstance(retriever, Retriever)

    def test_late_interaction_retriever_basic(self) -> None:
        items = [
            _make_item("the quick brown fox", item_id="a"),
            _make_item("lazy dog sleeps", item_id="b"),
            _make_item("quick fox jumps", item_id="c"),
        ]
        retriever = LateInteractionRetriever(
            first_stage=_FakeFirstStageRetriever(items),
            encoder=_FakeTokenLevelEncoder(),
        )
        query = QueryBundle(query_str="quick fox")
        results = retriever.retrieve(query, top_k=3)
        assert len(results) == 3
        # All candidates are returned, re-scored and sorted
        result_ids = [r.id for r in results]
        assert len(result_ids) == 3

    def test_late_interaction_retriever_top_k(self) -> None:
        items = [
            _make_item("alpha beta", item_id="1"),
            _make_item("gamma delta", item_id="2"),
            _make_item("epsilon zeta", item_id="3"),
        ]
        retriever = LateInteractionRetriever(
            first_stage=_FakeFirstStageRetriever(items),
            encoder=_FakeTokenLevelEncoder(),
        )
        query = QueryBundle(query_str="alpha")
        results = retriever.retrieve(query, top_k=2)
        assert len(results) == 2

    def test_late_interaction_first_stage_k(self) -> None:
        items = [_make_item(f"item {i}", item_id=str(i)) for i in range(10)]
        first_stage = _FakeFirstStageRetriever(items)
        retriever = LateInteractionRetriever(
            first_stage=first_stage,
            encoder=_FakeTokenLevelEncoder(),
            first_stage_k=5,
        )
        query = QueryBundle(query_str="test")
        results = retriever.retrieve(query, top_k=10)
        # First stage returns at most first_stage_k=5 items
        assert len(results) == 5


class TestLateInteractionRetrieverEdgeCases:
    """Edge cases for the two-stage retrieval pipeline."""

    def test_first_stage_returns_fewer_than_top_k(self) -> None:
        """First stage returns only 3 items but top_k=10 -- should return 3."""
        items = [
            _make_item("alpha", item_id="a"),
            _make_item("beta", item_id="b"),
            _make_item("gamma", item_id="c"),
        ]
        retriever = LateInteractionRetriever(
            first_stage=_FakeFirstStageRetriever(items),
            encoder=_FakeTokenLevelEncoder(),
            first_stage_k=100,
        )
        query = QueryBundle(query_str="test")
        results = retriever.retrieve(query, top_k=10)
        assert len(results) == 3

    def test_first_stage_returns_empty(self) -> None:
        """Empty first-stage result should produce empty final result."""
        retriever = LateInteractionRetriever(
            first_stage=_EmptyFirstStageRetriever(),
            encoder=_FakeTokenLevelEncoder(),
        )
        query = QueryBundle(query_str="anything")
        results = retriever.retrieve(query, top_k=5)
        assert results == []

    def test_all_items_score_equally_returns_top_k(self) -> None:
        """When all candidates score the same, still return exactly top_k items."""

        class _ConstantEncoder:
            def encode_tokens(self, text: str) -> list[list[float]]:
                # Every text produces the same single token embedding
                return [[1.0, 0.0]]

        items = [_make_item(f"item {i}", item_id=str(i)) for i in range(5)]
        retriever = LateInteractionRetriever(
            first_stage=_FakeFirstStageRetriever(items),
            encoder=_ConstantEncoder(),
            first_stage_k=100,
        )
        query = QueryBundle(query_str="query")
        results = retriever.retrieve(query, top_k=3)
        assert len(results) == 3

    def test_rescoring_changes_order(self) -> None:
        """Verify that re-scoring can change the first-stage ordering.

        We set up a controlled encoder where the first-stage order (a, b, c)
        differs from the MaxSim re-scored order.
        """
        items = [
            _make_item("doc_a", item_id="a"),
            _make_item("doc_b", item_id="b"),
            _make_item("doc_c", item_id="c"),
        ]
        # Controlled embeddings:
        # query -> [1,0]
        # doc_a -> [0,1]      cos(q, da) = 0.0
        # doc_b -> [1,0]      cos(q, db) = 1.0
        # doc_c -> [0.6,0.8]  cos(q, dc) = 0.6
        encoder = _ControlledTokenEncoder(
            {
                "query text": [[1.0, 0.0]],
                "doc_a": [[0.0, 1.0]],
                "doc_b": [[1.0, 0.0]],
                "doc_c": [[0.6, 0.8]],
            }
        )
        retriever = LateInteractionRetriever(
            first_stage=_FakeFirstStageRetriever(items),
            encoder=encoder,
            first_stage_k=100,
        )
        query = QueryBundle(query_str="query text")
        results = retriever.retrieve(query, top_k=3)
        result_ids = [r.id for r in results]
        # After re-scoring: b(1.0) > c(0.6) > a(0.0)
        assert result_ids == ["b", "c", "a"]


class TestRepr:
    def test_repr_all(self) -> None:
        scorer = MaxSimScorer()
        assert "MaxSimScorer" in repr(scorer)

        li_scorer = LateInteractionScorer()
        assert "LateInteractionScorer" in repr(li_scorer)

        items = [_make_item("test")]
        retriever = LateInteractionRetriever(
            first_stage=_FakeFirstStageRetriever(items),
            encoder=_FakeTokenLevelEncoder(),
        )
        assert "LateInteractionRetriever" in repr(retriever)
