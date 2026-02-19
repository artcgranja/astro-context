"""astro-context: Context engineering toolkit for AI applications."""

from astro_context.formatters import AnthropicFormatter, GenericTextFormatter, OpenAIFormatter
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
from astro_context.pipeline import ContextPipeline, PipelineStep, retriever_step
from astro_context.retrieval import DenseRetriever, HybridRetriever, SparseRetriever
from astro_context.storage import InMemoryContextStore, InMemoryVectorStore
from astro_context.tokens import TiktokenCounter

__version__ = "0.1.0"

__all__ = [
    # Pipeline
    "ContextPipeline",
    "PipelineStep",
    "retriever_step",
    # Models
    "ContextItem",
    "ContextResult",
    "ContextWindow",
    "QueryBundle",
    "SourceType",
    "TokenBudget",
    "BudgetAllocation",
    "ConversationTurn",
    # Retrieval
    "DenseRetriever",
    "SparseRetriever",
    "HybridRetriever",
    # Memory
    "MemoryManager",
    "SlidingWindowMemory",
    # Formatters
    "AnthropicFormatter",
    "GenericTextFormatter",
    "OpenAIFormatter",
    # Storage
    "InMemoryContextStore",
    "InMemoryVectorStore",
    # Tokens
    "TiktokenCounter",
]
