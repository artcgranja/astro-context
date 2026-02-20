"""Tests for top-level package exports."""

from __future__ import annotations


class TestTopLevelExports:
    """Verify all expected symbols are importable from the top-level package."""

    def test_pipeline_exports(self) -> None:
        from astro_context import (
            ContextPipeline,
            PipelineStep,
            async_postprocessor_step,
            async_retriever_step,
            filter_step,
            postprocessor_step,
            retriever_step,
        )

        assert ContextPipeline is not None
        assert PipelineStep is not None
        assert async_postprocessor_step is not None
        assert async_retriever_step is not None
        assert filter_step is not None
        assert postprocessor_step is not None
        assert retriever_step is not None

    def test_model_exports(self) -> None:
        from astro_context import (
            BudgetAllocation,
            ContextItem,
            ContextResult,
            ContextWindow,
            ConversationTurn,
            MemoryEntry,
            OverflowStrategy,
            PipelineDiagnostics,
            QueryBundle,
            Role,
            SourceType,
            StepDiagnostic,
            TokenBudget,
        )

        assert ContextItem is not None
        assert ContextResult is not None
        assert ContextWindow is not None
        assert QueryBundle is not None
        assert SourceType is not None
        assert TokenBudget is not None
        assert BudgetAllocation is not None
        assert ConversationTurn is not None
        assert MemoryEntry is not None
        assert OverflowStrategy is not None
        assert PipelineDiagnostics is not None
        assert Role is not None
        assert StepDiagnostic is not None

    def test_protocol_exports(self) -> None:
        from astro_context import (
            AsyncPostProcessor,
            AsyncRetriever,
            BaseFormatter,
            ContextStore,
            Formatter,
            PostProcessor,
            Retriever,
            Tokenizer,
            VectorStore,
        )

        assert AsyncPostProcessor is not None
        assert AsyncRetriever is not None
        assert BaseFormatter is not None
        assert Formatter is not None
        assert BaseFormatter is Formatter  # backward-compat alias
        assert ContextStore is not None
        assert PostProcessor is not None
        assert Retriever is not None
        assert Tokenizer is not None
        assert VectorStore is not None

    def test_exception_exports(self) -> None:
        from astro_context import (
            AstroContextError,
            FormatterError,
            RetrieverError,
            StorageError,
            TokenBudgetExceededError,
        )

        assert AstroContextError is not None
        assert FormatterError is not None
        assert RetrieverError is not None
        assert StorageError is not None
        assert TokenBudgetExceededError is not None

    def test_retrieval_exports(self) -> None:
        from astro_context import DenseRetriever, HybridRetriever, SparseRetriever

        assert DenseRetriever is not None
        assert HybridRetriever is not None
        assert SparseRetriever is not None

    def test_formatter_exports(self) -> None:
        from astro_context import AnthropicFormatter, GenericTextFormatter, OpenAIFormatter

        assert AnthropicFormatter is not None
        assert GenericTextFormatter is not None
        assert OpenAIFormatter is not None

    def test_memory_exports(self) -> None:
        from astro_context import MemoryManager, SlidingWindowMemory

        assert MemoryManager is not None
        assert SlidingWindowMemory is not None

    def test_storage_exports(self) -> None:
        from astro_context import InMemoryContextStore, InMemoryVectorStore

        assert InMemoryContextStore is not None
        assert InMemoryVectorStore is not None

    def test_token_exports(self) -> None:
        from astro_context import TiktokenCounter

        assert TiktokenCounter is not None

    def test_version(self) -> None:
        from astro_context import __version__

        assert __version__ == "0.1.0"
