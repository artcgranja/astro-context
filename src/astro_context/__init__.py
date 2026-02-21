"""astro-context: Context engineering toolkit for AI applications.

Agent:
    Agent, AgentTool, memory_tools, rag_tools

Core Pipeline:
    ContextPipeline, ContextResult, PipelineStep, PipelineCallback,
    PipelineDiagnostics, StepDiagnostic, PipelineExecutionError,
    retriever_step, filter_step, postprocessor_step, async_retriever_step,
    async_postprocessor_step, auto_promotion_step, graph_retrieval_step,
    create_eviction_promoter

Memory Management:
    MemoryManager, SlidingWindowMemory, SummaryBufferMemory, SimpleGraphMemory,
    MemoryGarbageCollector, GCStats, MemoryCallback, CallbackExtractor,
    MemoryContextEnricher, ContextQueryEnricher,
    FIFOEviction, ImportanceEviction, PairedEviction,
    SimilarityConsolidator, ExponentialRecencyScorer, LinearRecencyScorer,
    EbbinghausDecay, LinearDecay

Retrieval:
    DenseRetriever, SparseRetriever, HybridRetriever, ScoreReranker,
    ScoredMemoryRetriever, MemoryRetrieverAdapter

Formatting:
    AnthropicFormatter, OpenAIFormatter, GenericTextFormatter, BaseFormatter

Protocols (extension points):
    Retriever, AsyncRetriever, PostProcessor, AsyncPostProcessor,
    Tokenizer, Formatter, QueryEnricher, MemoryQueryEnricher, RecencyScorer,
    EvictionPolicy, CompactionStrategy, AsyncCompactionStrategy,
    MemoryExtractor, AsyncMemoryExtractor, MemoryConsolidator, MemoryDecay,
    ContextStore, DocumentStore, VectorStore, MemoryEntryStore,
    GarbageCollectableStore, ConversationMemory, MemoryProvider

Storage:
    InMemoryContextStore, InMemoryDocumentStore, InMemoryVectorStore,
    InMemoryEntryStore, JsonFileMemoryStore

Models & Types:
    ContextItem, ContextWindow, QueryBundle, TokenBudget, BudgetAllocation,
    ConversationTurn, MemoryEntry, MemoryType, MemoryOperation,
    SourceType, OverflowStrategy, Role,
    StreamDelta, StreamResult, StreamUsage,
    default_chat_budget, default_rag_budget, default_agent_budget

Exceptions:
    AstroContextError, FormatterError, RetrieverError, StorageError,
    TokenBudgetExceededError

Tokens:
    TiktokenCounter
"""

from importlib.metadata import PackageNotFoundError, version

from astro_context.agent import (
    Agent,
    AgentTool,
    Skill,
    SkillRegistry,
    memory_skill,
    memory_tools,
    rag_skill,
    rag_tools,
)
from astro_context.exceptions import (
    AstroContextError,
    FormatterError,
    PipelineExecutionError,
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
from astro_context.memory import (
    CallbackExtractor,
    EbbinghausDecay,
    ExponentialRecencyScorer,
    FIFOEviction,
    GCStats,
    ImportanceEviction,
    LinearDecay,
    LinearRecencyScorer,
    MemoryCallback,
    MemoryGarbageCollector,
    MemoryManager,
    PairedEviction,
    SimilarityConsolidator,
    SimpleGraphMemory,
    SlidingWindowMemory,
    SummaryBufferMemory,
)
from astro_context.models import (
    BudgetAllocation,
    ContextItem,
    ContextResult,
    ContextWindow,
    ConversationTurn,
    MemoryEntry,
    MemoryType,
    OverflowStrategy,
    PipelineDiagnostics,
    QueryBundle,
    Role,
    SourceType,
    StepDiagnostic,
    StreamDelta,
    StreamResult,
    StreamUsage,
    TokenBudget,
    default_agent_budget,
    default_chat_budget,
    default_rag_budget,
)
from astro_context.pipeline import (
    ContextPipeline,
    ContextQueryEnricher,
    MemoryContextEnricher,
    PipelineCallback,
    PipelineStep,
    async_postprocessor_step,
    async_retriever_step,
    auto_promotion_step,
    create_eviction_promoter,
    filter_step,
    graph_retrieval_step,
    postprocessor_step,
    retriever_step,
)
from astro_context.protocols import (
    AsyncCompactionStrategy,
    AsyncMemoryExtractor,
    AsyncPostProcessor,
    AsyncRetriever,
    CompactionStrategy,
    ContextStore,
    ConversationMemory,
    DocumentStore,
    EvictionPolicy,
    GarbageCollectableStore,
    MemoryConsolidator,
    MemoryDecay,
    MemoryEntryStore,
    MemoryExtractor,
    MemoryOperation,
    MemoryProvider,
    MemoryQueryEnricher,
    PostProcessor,
    QueryEnricher,
    RecencyScorer,
    Retriever,
    Tokenizer,
    VectorStore,
)
from astro_context.retrieval import (
    DenseRetriever,
    HybridRetriever,
    MemoryRetrieverAdapter,
    ScoredMemoryRetriever,
    ScoreReranker,
    SparseRetriever,
)
from astro_context.storage import (
    InMemoryContextStore,
    InMemoryDocumentStore,
    InMemoryEntryStore,
    InMemoryVectorStore,
    JsonFileMemoryStore,
)
from astro_context.tokens import TiktokenCounter

try:
    __version__ = version("astro-context")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

__all__ = [
    "Agent",
    "AgentTool",
    "AnthropicFormatter",
    "AstroContextError",
    "AsyncCompactionStrategy",
    "AsyncMemoryExtractor",
    "AsyncPostProcessor",
    "AsyncRetriever",
    "BaseFormatter",
    "BudgetAllocation",
    "CallbackExtractor",
    "CompactionStrategy",
    "ContextItem",
    "ContextPipeline",
    "ContextQueryEnricher",
    "ContextResult",
    "ContextStore",
    "ContextWindow",
    "ConversationMemory",
    "ConversationTurn",
    "DenseRetriever",
    "DocumentStore",
    "EbbinghausDecay",
    "EvictionPolicy",
    "ExponentialRecencyScorer",
    "FIFOEviction",
    "Formatter",
    "FormatterError",
    "GCStats",
    "GarbageCollectableStore",
    "GenericTextFormatter",
    "HybridRetriever",
    "ImportanceEviction",
    "InMemoryContextStore",
    "InMemoryDocumentStore",
    "InMemoryEntryStore",
    "InMemoryVectorStore",
    "JsonFileMemoryStore",
    "LinearDecay",
    "LinearRecencyScorer",
    "MemoryCallback",
    "MemoryConsolidator",
    "MemoryContextEnricher",
    "MemoryDecay",
    "MemoryEntry",
    "MemoryEntryStore",
    "MemoryExtractor",
    "MemoryGarbageCollector",
    "MemoryManager",
    "MemoryOperation",
    "MemoryProvider",
    "MemoryQueryEnricher",
    "MemoryRetrieverAdapter",
    "MemoryType",
    "OpenAIFormatter",
    "OverflowStrategy",
    "PairedEviction",
    "PipelineCallback",
    "PipelineDiagnostics",
    "PipelineExecutionError",
    "PipelineStep",
    "PostProcessor",
    "QueryBundle",
    "QueryEnricher",
    "RecencyScorer",
    "Retriever",
    "RetrieverError",
    "Role",
    "ScoreReranker",
    "ScoredMemoryRetriever",
    "SimilarityConsolidator",
    "SimpleGraphMemory",
    "Skill",
    "SkillRegistry",
    "SlidingWindowMemory",
    "SourceType",
    "SparseRetriever",
    "StepDiagnostic",
    "StorageError",
    "StreamDelta",
    "StreamResult",
    "StreamUsage",
    "SummaryBufferMemory",
    "TiktokenCounter",
    "TokenBudget",
    "TokenBudgetExceededError",
    "Tokenizer",
    "VectorStore",
    "async_postprocessor_step",
    "async_retriever_step",
    "auto_promotion_step",
    "create_eviction_promoter",
    "default_agent_budget",
    "default_chat_budget",
    "default_rag_budget",
    "filter_step",
    "graph_retrieval_step",
    "memory_skill",
    "memory_tools",
    "postprocessor_step",
    "rag_skill",
    "rag_tools",
    "retriever_step",
]
