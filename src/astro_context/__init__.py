"""astro-context: Context engineering toolkit for AI applications."""

from astro_context.exceptions import (
    AstroContextError,
    FormatterError,
    RetrieverError,
    StorageError,
    TokenBudgetExceededError,
)
from astro_context.formatters import (
    AnthropicFormatter,
    BaseFormatter,
    GenericTextFormatter,
    OpenAIFormatter,
)
from astro_context.memory import MemoryManager, SlidingWindowMemory
from astro_context.models import (
    BudgetAllocation,
    ContextItem,
    ContextResult,
    ContextWindow,
    ConversationTurn,
    QueryBundle,
    SourceType,
    TokenBudget,
)
from astro_context.pipeline import (
    ContextPipeline,
    PipelineStep,
    async_postprocessor_step,
    async_retriever_step,
    filter_step,
    postprocessor_step,
    retriever_step,
)
from astro_context.protocols import (
    AsyncPostProcessor,
    AsyncRetriever,
    ContextStore,
    PostProcessor,
    Retriever,
    Tokenizer,
    VectorStore,
)
from astro_context.retrieval import DenseRetriever, HybridRetriever, SparseRetriever
from astro_context.storage import InMemoryContextStore, InMemoryVectorStore
from astro_context.tokens import TiktokenCounter

__version__ = "0.1.0"

__all__ = [
    "AnthropicFormatter",
    "AstroContextError",
    "AsyncPostProcessor",
    "AsyncRetriever",
    "BaseFormatter",
    "BudgetAllocation",
    "ContextItem",
    "ContextPipeline",
    "ContextResult",
    "ContextStore",
    "ContextWindow",
    "ConversationTurn",
    "DenseRetriever",
    "FormatterError",
    "GenericTextFormatter",
    "HybridRetriever",
    "InMemoryContextStore",
    "InMemoryVectorStore",
    "MemoryManager",
    "OpenAIFormatter",
    "PipelineStep",
    "PostProcessor",
    "QueryBundle",
    "Retriever",
    "RetrieverError",
    "SlidingWindowMemory",
    "SourceType",
    "SparseRetriever",
    "StorageError",
    "TiktokenCounter",
    "TokenBudget",
    "TokenBudgetExceededError",
    "Tokenizer",
    "VectorStore",
    "async_postprocessor_step",
    "async_retriever_step",
    "filter_step",
    "postprocessor_step",
    "retriever_step",
]
