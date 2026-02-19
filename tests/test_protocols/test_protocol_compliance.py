"""Tests verifying Protocol compliance via structural subtyping.

These tests ensure that concrete implementations satisfy their respective
protocols without explicit inheritance (PEP 544).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from astro_context.formatters.anthropic import AnthropicFormatter
from astro_context.formatters.base import BaseFormatter
from astro_context.formatters.generic import GenericTextFormatter
from astro_context.formatters.openai import OpenAIFormatter
from astro_context.models.context import ContextItem, ContextWindow
from astro_context.models.query import QueryBundle
from astro_context.protocols.postprocessor import AsyncPostProcessor, PostProcessor
from astro_context.protocols.retriever import AsyncRetriever, Retriever
from astro_context.protocols.storage import ContextStore, DocumentStore, VectorStore
from astro_context.protocols.tokenizer import Tokenizer
from astro_context.storage.memory_store import (
    InMemoryContextStore,
    InMemoryDocumentStore,
    InMemoryVectorStore,
)
from tests.conftest import FakeTokenizer


class TestRetrieverProtocol:
    """Concrete retrievers satisfy the Retriever protocol."""

    def test_dense_retriever_is_retriever(self) -> None:
        """DenseRetriever satisfies the Retriever protocol."""
        with patch(
            "astro_context.retrieval.dense.get_default_counter",
            return_value=FakeTokenizer(),
        ):
            from astro_context.retrieval.dense import DenseRetriever

            store = InMemoryContextStore()
            vstore = InMemoryVectorStore()
            retriever = DenseRetriever(vector_store=vstore, context_store=store)
            assert isinstance(retriever, Retriever)

    def test_sparse_retriever_is_retriever(self) -> None:
        """SparseRetriever satisfies the Retriever protocol."""
        with patch(
            "astro_context.retrieval.sparse.get_default_counter",
            return_value=FakeTokenizer(),
        ):
            from astro_context.retrieval.sparse import SparseRetriever

            retriever = SparseRetriever()
            assert isinstance(retriever, Retriever)

    def test_hybrid_retriever_is_retriever(self) -> None:
        """HybridRetriever satisfies the Retriever protocol (checked via custom)."""

        class FakeSubRetriever:
            def retrieve(self, query: QueryBundle, top_k: int = 10) -> list[ContextItem]:
                return []

        from astro_context.retrieval.hybrid import HybridRetriever

        retriever = HybridRetriever(retrievers=[FakeSubRetriever()])
        assert isinstance(retriever, Retriever)

    def test_custom_retriever_structural_subtyping(self) -> None:
        """A plain class with matching interface satisfies Retriever."""

        class MyRetriever:
            def retrieve(self, query: QueryBundle, top_k: int = 10) -> list[ContextItem]:
                return []

        assert isinstance(MyRetriever(), Retriever)


class TestAsyncRetrieverProtocol:
    """AsyncRetriever protocol compliance."""

    def test_custom_async_retriever(self) -> None:

        class MyAsyncRetriever:
            async def aretrieve(self, query: QueryBundle, top_k: int = 10) -> list[ContextItem]:
                return []

        assert isinstance(MyAsyncRetriever(), AsyncRetriever)


class TestPostProcessorProtocol:
    """PostProcessor protocol compliance."""

    def test_custom_postprocessor(self) -> None:

        class MyProcessor:
            def process(
                self, items: list[ContextItem], query: QueryBundle | None = None
            ) -> list[ContextItem]:
                return items

        assert isinstance(MyProcessor(), PostProcessor)


class TestAsyncPostProcessorProtocol:
    """AsyncPostProcessor protocol compliance."""

    def test_custom_async_postprocessor(self) -> None:

        class MyAsyncProcessor:
            async def aprocess(
                self, items: list[ContextItem], query: QueryBundle | None = None
            ) -> list[ContextItem]:
                return items

        assert isinstance(MyAsyncProcessor(), AsyncPostProcessor)


class TestFormatterProtocol:
    """Formatters satisfy BaseFormatter protocol via structural subtyping."""

    def test_anthropic_formatter_is_base_formatter(self) -> None:
        assert isinstance(AnthropicFormatter(), BaseFormatter)

    def test_openai_formatter_is_base_formatter(self) -> None:
        assert isinstance(OpenAIFormatter(), BaseFormatter)

    def test_generic_formatter_is_base_formatter(self) -> None:
        assert isinstance(GenericTextFormatter(), BaseFormatter)

    def test_custom_formatter_structural_subtyping(self) -> None:
        """A plain class with matching interface satisfies BaseFormatter."""

        class MyFormatter:
            @property
            def format_type(self) -> str:
                return "custom"

            def format(self, window: ContextWindow) -> str | dict[str, Any]:
                return "formatted"

        fmt = MyFormatter()
        assert isinstance(fmt, BaseFormatter)

    def test_custom_formatter_works_in_pipeline(self) -> None:
        """A custom formatter (no inheritance) works in the pipeline."""
        from astro_context.pipeline.pipeline import ContextPipeline

        class MyFormatter:
            @property
            def format_type(self) -> str:
                return "custom"

            def format(self, window: ContextWindow) -> str:
                return f"custom output with {len(window.items)} items"

        pipeline = ContextPipeline(max_tokens=8192, tokenizer=FakeTokenizer())
        pipeline.with_formatter(MyFormatter())  # type: ignore[arg-type]
        pipeline.add_system_prompt("System")

        result = pipeline.build(QueryBundle(query_str="test"))
        assert result.format_type == "custom"
        assert "custom output" in str(result.formatted_output)


class TestStorageProtocols:
    """Storage implementations satisfy their protocols."""

    def test_in_memory_context_store(self) -> None:
        assert isinstance(InMemoryContextStore(), ContextStore)

    def test_in_memory_vector_store(self) -> None:
        assert isinstance(InMemoryVectorStore(), VectorStore)

    def test_in_memory_document_store(self) -> None:
        assert isinstance(InMemoryDocumentStore(), DocumentStore)


class TestTokenizerProtocol:
    """Tokenizer protocol compliance."""

    def test_fake_tokenizer_is_tokenizer(self) -> None:
        assert isinstance(FakeTokenizer(), Tokenizer)

    def test_custom_tokenizer_structural_subtyping(self) -> None:
        """A plain class with count_tokens and truncate_to_tokens satisfies Tokenizer."""

        class MyTokenizer:
            def count_tokens(self, text: str) -> int:
                return len(text)

            def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
                return text[:max_tokens]

        assert isinstance(MyTokenizer(), Tokenizer)
