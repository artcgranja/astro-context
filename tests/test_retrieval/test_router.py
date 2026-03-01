"""Tests for astro_context.retrieval.router."""

from __future__ import annotations

import pytest

from astro_context.exceptions import RetrieverError
from astro_context.models.context import ContextItem, SourceType
from astro_context.models.query import QueryBundle
from astro_context.protocols.retriever import Retriever
from astro_context.protocols.router import QueryRouter
from astro_context.retrieval.router import (
    CallbackRouter,
    KeywordRouter,
    MetadataRouter,
    RoutedRetriever,
)
from tests.conftest import FakeRetriever


def _make_item(item_id: str, content: str, score: float = 0.5) -> ContextItem:
    return ContextItem(
        id=item_id,
        content=content,
        source=SourceType.RETRIEVAL,
        score=score,
        priority=5,
        token_count=5,
    )


class TestKeywordRouter:
    """Tests for KeywordRouter."""

    def test_protocol_compliance(self) -> None:
        router = KeywordRouter(routes={"a": ["hello"]}, default="b")
        assert isinstance(router, QueryRouter)

    def test_routes_by_keyword(self) -> None:
        router = KeywordRouter(
            routes={"tech": ["python", "code"], "general": ["hello"]},
            default="fallback",
        )
        q = QueryBundle(query_str="python programming")
        assert router.route(q) == "tech"

    def test_falls_back_to_default(self) -> None:
        router = KeywordRouter(routes={"tech": ["python"]}, default="general")
        q = QueryBundle(query_str="hello world")
        assert router.route(q) == "general"

    def test_case_insensitive(self) -> None:
        router = KeywordRouter(
            routes={"tech": ["Python"]}, default="x", case_sensitive=False
        )
        q = QueryBundle(query_str="python is great")
        assert router.route(q) == "tech"

    def test_case_sensitive(self) -> None:
        router = KeywordRouter(
            routes={"tech": ["Python"]}, default="x", case_sensitive=True
        )
        q = QueryBundle(query_str="python is great")
        assert router.route(q) == "x"  # no match because lowercase

    def test_first_match_wins(self) -> None:
        router = KeywordRouter(
            routes={"first": ["hello"], "second": ["hello"]},
            default="x",
        )
        q = QueryBundle(query_str="hello world")
        assert router.route(q) == "first"

    def test_repr(self) -> None:
        router = KeywordRouter(routes={"a": ["x"]}, default="b")
        r = repr(router)
        assert "KeywordRouter" in r
        assert "routes=1" in r
        assert "'b'" in r


class TestCallbackRouter:
    """Tests for CallbackRouter."""

    def test_protocol_compliance(self) -> None:
        router = CallbackRouter(callback=lambda q: "a")
        assert isinstance(router, QueryRouter)

    def test_routes_via_callback(self) -> None:
        router = CallbackRouter(
            callback=lambda q: "tech" if "code" in q.query_str else None,
            default="general",
        )
        assert router.route(QueryBundle(query_str="code review")) == "tech"
        assert router.route(QueryBundle(query_str="hello")) == "general"

    def test_default_when_none(self) -> None:
        router = CallbackRouter(callback=lambda q: None, default="fallback")
        assert router.route(QueryBundle(query_str="anything")) == "fallback"

    def test_repr(self) -> None:
        router = CallbackRouter(callback=lambda q: "a", default="b")
        r = repr(router)
        assert "CallbackRouter" in r
        assert "'b'" in r


class TestMetadataRouter:
    """Tests for MetadataRouter."""

    def test_protocol_compliance(self) -> None:
        assert isinstance(MetadataRouter(), QueryRouter)

    def test_routes_by_metadata(self) -> None:
        router = MetadataRouter(metadata_key="route", default="fallback")
        q = QueryBundle(query_str="test", metadata={"route": "tech"})
        assert router.route(q) == "tech"

    def test_falls_back_when_key_missing(self) -> None:
        router = MetadataRouter(metadata_key="route", default="general")
        q = QueryBundle(query_str="test")
        assert router.route(q) == "general"

    def test_custom_metadata_key(self) -> None:
        router = MetadataRouter(metadata_key="target", default="x")
        q = QueryBundle(query_str="test", metadata={"target": "special"})
        assert router.route(q) == "special"

    def test_non_string_value_converted(self) -> None:
        router = MetadataRouter(metadata_key="route", default="x")
        q = QueryBundle(query_str="test", metadata={"route": 42})
        assert router.route(q) == "42"

    def test_repr(self) -> None:
        router = MetadataRouter(metadata_key="route", default="x")
        r = repr(router)
        assert "MetadataRouter" in r
        assert "'route'" in r


class TestRoutedRetriever:
    """Tests for RoutedRetriever."""

    def test_protocol_compliance(self) -> None:
        items_a = [_make_item("a1", "item A")]
        router = KeywordRouter(routes={"a": ["hello"]}, default="a")
        rr = RoutedRetriever(
            router=router,
            retrievers={"a": FakeRetriever(items_a)},
        )
        assert isinstance(rr, Retriever)

    def test_routes_to_correct_retriever(self) -> None:
        items_tech = [_make_item("t1", "tech item")]
        items_gen = [_make_item("g1", "general item")]
        router = KeywordRouter(
            routes={"tech": ["python"], "general": ["hello"]},
            default="general",
        )
        rr = RoutedRetriever(
            router=router,
            retrievers={
                "tech": FakeRetriever(items_tech),
                "general": FakeRetriever(items_gen),
            },
        )
        result = rr.retrieve(QueryBundle(query_str="python code"), top_k=10)
        assert len(result) == 1
        assert result[0].id == "t1"

        result = rr.retrieve(QueryBundle(query_str="hello world"), top_k=10)
        assert len(result) == 1
        assert result[0].id == "g1"

    def test_unknown_route_raises(self) -> None:
        router = CallbackRouter(callback=lambda q: "nonexistent")
        rr = RoutedRetriever(
            router=router,
            retrievers={"a": FakeRetriever([_make_item("a1", "item")])},
        )
        with pytest.raises(RetrieverError, match="nonexistent"):
            rr.retrieve(QueryBundle(query_str="test"))

    def test_default_retriever_fallback(self) -> None:
        items = [_make_item("d1", "default item")]
        router = CallbackRouter(callback=lambda q: "nonexistent")
        rr = RoutedRetriever(
            router=router,
            retrievers={
                "a": FakeRetriever([]),
                "fallback": FakeRetriever(items),
            },
            default_retriever="fallback",
        )
        result = rr.retrieve(QueryBundle(query_str="test"))
        assert len(result) == 1
        assert result[0].id == "d1"

    def test_default_retriever_also_missing_raises(self) -> None:
        router = CallbackRouter(callback=lambda q: "nonexistent")
        rr = RoutedRetriever(
            router=router,
            retrievers={"a": FakeRetriever([])},
            default_retriever="also_nonexistent",
        )
        with pytest.raises(RetrieverError):
            rr.retrieve(QueryBundle(query_str="test"))

    def test_respects_top_k(self) -> None:
        items = [_make_item(f"i{i}", f"item {i}") for i in range(5)]
        router = MetadataRouter(metadata_key="route", default="main")
        rr = RoutedRetriever(
            router=router,
            retrievers={"main": FakeRetriever(items)},
        )
        result = rr.retrieve(QueryBundle(query_str="test"), top_k=2)
        assert len(result) == 2

    def test_repr(self) -> None:
        router = KeywordRouter(routes={"a": ["x"]}, default="a")
        rr = RoutedRetriever(
            router=router,
            retrievers={"a": FakeRetriever([])},
            default_retriever="a",
        )
        r = repr(rr)
        assert "RoutedRetriever" in r
        assert "'a'" in r
