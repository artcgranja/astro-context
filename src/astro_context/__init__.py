"""astro-context: Context engineering toolkit for AI applications."""

from importlib.metadata import PackageNotFoundError, version

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
    Formatter,
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
    MemoryEntry,
    OverflowStrategy,
    PipelineDiagnostics,
    QueryBundle,
    Role,
    SourceType,
    StepDiagnostic,
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
    DocumentStore,
    PostProcessor,
    Retriever,
    Tokenizer,
    VectorStore,
)
from astro_context.retrieval import DenseRetriever, HybridRetriever, SparseRetriever
from astro_context.storage import InMemoryContextStore, InMemoryDocumentStore, InMemoryVectorStore
from astro_context.tokens import TiktokenCounter

try:
    __version__ = version("astro-context")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

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
    "DocumentStore",
    "Formatter",
    "FormatterError",
    "GenericTextFormatter",
    "HybridRetriever",
    "InMemoryContextStore",
    "InMemoryDocumentStore",
    "InMemoryVectorStore",
    "MemoryEntry",
    "MemoryManager",
    "OpenAIFormatter",
    "OverflowStrategy",
    "PipelineDiagnostics",
    "PipelineStep",
    "PostProcessor",
    "QueryBundle",
    "Retriever",
    "RetrieverError",
    "Role",
    "SlidingWindowMemory",
    "SourceType",
    "SparseRetriever",
    "StepDiagnostic",
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
