"""Tests for built-in query classifiers."""

from __future__ import annotations

import math

import pytest

from astro_context.models.query import QueryBundle
from astro_context.protocols.classifier import QueryClassifier
from astro_context.query.classifiers import (
    CallbackClassifier,
    EmbeddingClassifier,
    KeywordClassifier,
)


class TestKeywordClassifier:
    """Tests for the KeywordClassifier class."""

    def test_protocol_compliance(self) -> None:
        c = KeywordClassifier(rules={"tech": ["python"]}, default="general")
        assert isinstance(c, QueryClassifier)

    def test_basic_match(self) -> None:
        c = KeywordClassifier(
            rules={"tech": ["python", "java"], "science": ["biology"]},
            default="general",
        )
        query = QueryBundle(query_str="How do I use python decorators?")
        assert c.classify(query) == "tech"

    def test_default_fallback(self) -> None:
        c = KeywordClassifier(
            rules={"tech": ["python"]},
            default="general",
        )
        query = QueryBundle(query_str="What is the meaning of life?")
        assert c.classify(query) == "general"

    def test_case_insensitive_by_default(self) -> None:
        c = KeywordClassifier(
            rules={"tech": ["Python"]},
            default="other",
        )
        query = QueryBundle(query_str="I love PYTHON programming")
        assert c.classify(query) == "tech"

    def test_case_sensitive(self) -> None:
        c = KeywordClassifier(
            rules={"tech": ["Python"]},
            default="other",
            case_sensitive=True,
        )
        assert c.classify(QueryBundle(query_str="Python is great")) == "tech"
        assert c.classify(QueryBundle(query_str="python is great")) == "other"

    def test_first_rule_wins(self) -> None:
        c = KeywordClassifier(
            rules={
                "first": ["test"],
                "second": ["test"],
            },
            default="none",
        )
        assert c.classify(QueryBundle(query_str="this is a test")) == "first"

    def test_repr(self) -> None:
        c = KeywordClassifier(
            rules={"a": ["x"], "b": ["y"]},
            default="z",
            case_sensitive=True,
        )
        r = repr(c)
        assert "KeywordClassifier" in r
        assert "'a'" in r
        assert "'b'" in r
        assert "default='z'" in r
        assert "case_sensitive=True" in r

    def test_substring_match(self) -> None:
        """Keywords match as substrings, not just whole words."""
        c = KeywordClassifier(rules={"code": ["program"]}, default="other")
        assert c.classify(QueryBundle(query_str="programming is fun")) == "code"

    def test_overlapping_keywords_first_rule_wins(self) -> None:
        """When a query matches keywords in multiple rules, first rule wins."""
        c = KeywordClassifier(
            rules={
                "api": ["python", "rest"],
                "tutorial": ["python", "learn"],
                "docs": ["documentation"],
            },
            default="general",
        )
        # "python" appears in both "api" and "tutorial" rules;
        # "api" comes first in insertion order, so it wins
        query = QueryBundle(query_str="learn python rest api")
        assert c.classify(query) == "api"

    def test_empty_query_string(self) -> None:
        """Empty query string matches nothing and falls back to default."""
        c = KeywordClassifier(
            rules={"tech": ["python", "java"], "science": ["biology"]},
            default="unknown",
        )
        query = QueryBundle(query_str="")
        assert c.classify(query) == "unknown"


class TestCallbackClassifier:
    """Tests for the CallbackClassifier class."""

    def test_protocol_compliance(self) -> None:
        c = CallbackClassifier(classify_fn=lambda q: "label")
        assert isinstance(c, QueryClassifier)

    def test_basic_callback(self) -> None:
        def classify_by_length(q: QueryBundle) -> str:
            return "short" if len(q.query_str) < 20 else "long"

        c = CallbackClassifier(classify_fn=classify_by_length)
        assert c.classify(QueryBundle(query_str="hi")) == "short"
        assert c.classify(QueryBundle(query_str="this is a much longer query string")) == "long"

    def test_callback_receives_full_query(self) -> None:
        def check_metadata(q: QueryBundle) -> str:
            return q.metadata.get("type", "unknown")

        c = CallbackClassifier(classify_fn=check_metadata)
        query = QueryBundle(query_str="test", metadata={"type": "technical"})
        assert c.classify(query) == "technical"

    def test_repr(self) -> None:
        c = CallbackClassifier(classify_fn=lambda q: "x")
        assert repr(c) == "CallbackClassifier()"

    def test_callback_returns_non_string(self) -> None:
        """CallbackClassifier passes through whatever the callback returns.

        If the callback returns a non-string value, classify() returns it
        without type enforcement (user's responsibility to conform).
        """

        def bad_callback(q: QueryBundle) -> str:
            return 42  # type: ignore[return-value]

        c = CallbackClassifier(classify_fn=bad_callback)
        # The classifier does not validate the return type
        result = c.classify(QueryBundle(query_str="test"))
        assert result == 42  # type: ignore[comparison-overlap]


class TestEmbeddingClassifier:
    """Tests for the EmbeddingClassifier class."""

    def test_protocol_compliance(self) -> None:
        c = EmbeddingClassifier(centroids={"a": [1.0, 0.0], "b": [0.0, 1.0]})
        assert isinstance(c, QueryClassifier)

    def test_closest_centroid(self) -> None:
        c = EmbeddingClassifier(
            centroids={
                "tech": [1.0, 0.0, 0.0],
                "science": [0.0, 1.0, 0.0],
                "art": [0.0, 0.0, 1.0],
            }
        )
        # Embedding closest to "tech"
        query = QueryBundle(query_str="test", embedding=[0.9, 0.1, 0.0])
        assert c.classify(query) == "tech"

        # Embedding closest to "art"
        query = QueryBundle(query_str="test", embedding=[0.0, 0.1, 0.9])
        assert c.classify(query) == "art"

    def test_missing_embedding_raises(self) -> None:
        c = EmbeddingClassifier(centroids={"a": [1.0, 0.0]})
        query = QueryBundle(query_str="test")  # embedding is None
        with pytest.raises(ValueError, match=r"requires query\.embedding"):
            c.classify(query)

    def test_custom_distance_fn(self) -> None:
        def negative_euclidean(a: list[float], b: list[float]) -> float:
            return -math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=True)))

        c = EmbeddingClassifier(
            centroids={
                "near": [1.0, 0.0],
                "far": [10.0, 10.0],
            },
            distance_fn=negative_euclidean,
        )
        query = QueryBundle(query_str="test", embedding=[0.9, 0.1])
        assert c.classify(query) == "near"

    def test_repr(self) -> None:
        c = EmbeddingClassifier(centroids={"x": [1.0], "y": [0.0]})
        r = repr(c)
        assert "EmbeddingClassifier" in r
        assert "'x'" in r
        assert "'y'" in r

    def test_equal_distance_picks_first(self) -> None:
        """When centroids are equidistant, the first in iteration order wins."""
        c = EmbeddingClassifier(
            centroids={
                "alpha": [1.0, 0.0],
                "beta": [1.0, 0.0],  # identical to alpha
            }
        )
        query = QueryBundle(query_str="test", embedding=[1.0, 0.0])
        # Both have cosine similarity 1.0; first one encountered wins
        label = c.classify(query)
        assert label in ("alpha", "beta")

    def test_high_dimensional_768_centroids(self) -> None:
        """EmbeddingClassifier works correctly with 768-dim vectors (BERT-sized)."""
        dim = 768
        # Create two orthogonal-ish centroids using deterministic patterns
        centroid_a = [math.sin(i) for i in range(dim)]
        centroid_b = [math.cos(i) for i in range(dim)]

        c = EmbeddingClassifier(centroids={"a": centroid_a, "b": centroid_b})

        # Query close to centroid_a
        query_a = QueryBundle(
            query_str="test",
            embedding=[math.sin(i) + 0.01 * math.cos(i) for i in range(dim)],
        )
        assert c.classify(query_a) == "a"

        # Query close to centroid_b
        query_b = QueryBundle(
            query_str="test",
            embedding=[math.cos(i) + 0.01 * math.sin(i) for i in range(dim)],
        )
        assert c.classify(query_b) == "b"

    def test_tie_breaking_equidistant_distinct_centroids(self) -> None:
        """When query is equidistant from two distinct centroids, first wins."""
        # Two centroids that are mirror images across the query vector
        c = EmbeddingClassifier(
            centroids={
                "left": [1.0, 0.0, 0.0],
                "right": [0.0, 1.0, 0.0],
            }
        )
        # Query equally similar to both (45 degrees from each)
        norm = math.sqrt(2) / 2
        query = QueryBundle(
            query_str="test",
            embedding=[norm, norm, 0.0],
        )
        label = c.classify(query)
        # Both have the same cosine similarity; first in dict order wins
        assert label == "left"
