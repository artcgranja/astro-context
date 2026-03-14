---
title: Changelog
description: Release history for anchor.
---

# Changelog

All notable changes to anchor are documented here. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added

#### LLM Providers (`anchor.llm`)
- Multi-provider LLM interface with unified API across Anthropic, OpenAI, Gemini, Grok, Ollama, OpenRouter, and LiteLLM
- `FallbackProvider` for automatic failover across multiple LLM backends
- Unified error hierarchy: `ProviderError`, `AuthenticationError`, `RateLimitError`, `ContentFilterError`, `ModelNotFoundError`, `ServerError`, `TimeoutError`, `ProviderNotInstalledError`
- `create_provider` factory and `register_provider` for custom backends
- Cost calculation via `calculate_cost` with `MODEL_PRICING` registry
- Streaming support with `StreamChunk` and `ToolCallDelta`
- Typed message/response models: `Message`, `LLMResponse`, `ContentBlock`, `ToolCall`, `ToolResult`, `ToolSchema`, `Usage`, `StopReason`

#### MCP Bridge (`anchor.mcp`)
- Bidirectional Model Context Protocol integration (requires `astro-anchor[mcp]`)
- `FastMCPClientBridge` for consuming external MCP servers as anchor tools
- `FastMCPServerBridge` for exposing anchor pipelines and tools as MCP servers
- `MCPClientPool` for managing connections to multiple MCP servers
- `MCPClient` / `MCPServer` abstractions with typed config (`MCPServerConfig`)
- MCP resource and prompt models: `MCPResource`, `MCPPrompt`, `MCPPromptArgument`
- `mcp_tool_to_agent_tool` converter and `parse_server_string` utility
- Error hierarchy: `MCPError`, `MCPConfigError`, `MCPConnectionError`, `MCPTimeoutError`, `MCPToolError`

#### Agent Framework (`anchor.agent`)
- Tool-calling agent with `@tool` decorator for defining agent tools
- `Agent` class with async streaming via `achat`
- `Skill` and `SkillRegistry` for organizing related tools
- Built-in `memory_skill` / `memory_tools` and `rag_skill` / `rag_tools` for out-of-the-box capabilities

#### Advanced Retrieval
- `RoutedRetriever` with pluggable routing: `CallbackRouter`, `KeywordRouter`, `MetadataRouter`
- `LateInteractionRetriever` with `LateInteractionScorer` and `MaxSimScorer` for ColBERT-style retrieval
- `SharedSpaceRetriever` for cross-modal retrieval in a unified embedding space
- `CrossModalEncoder` and `TokenLevelEncoder` protocols for multi-modal embeddings
- `RoundRobinReranker` and `RerankerPipeline` for composable reranking strategies
- Async retriever/reranker variants: `AsyncDenseRetriever`, `AsyncHybridRetriever`, `AsyncCrossEncoderReranker`, `AsyncCohereReranker`

#### Evaluation Enhancements
- `ABTestRunner` and `ABTestResult` for A/B testing pipeline configurations
- `HumanEvaluationCollector` and `HumanJudgment` for human-in-the-loop evaluation
- `BatchEvaluator` with parallelization for large-scale evaluation runs
- `HumanEvaluator` protocol for pluggable human evaluation backends

#### Query Enhancements
- `ContextualQueryTransformer` for context-aware query rewriting
- `ConversationRewriter` for resolving coreferences in multi-turn conversations
- `KeywordClassifier`, `EmbeddingClassifier`, `CallbackClassifier` for query routing
- `QueryClassifier` protocol and `classified_retriever_step` pipeline step

#### Memory Enhancements
- `ProgressiveSummarizationMemory` with multi-tier compaction
- `TierCompactor` with `CompactionStrategy` and `AsyncCompactionStrategy` protocols

#### Ingestion Enhancements
- `CodeChunker` for language-aware code splitting
- `TableAwareChunker` for preserving table structures during chunking

#### Tests & Examples
- Unit tests for `_math.py` (cosine_similarity and clamp functions)
- `MemoryRetrieverAdapter` tests verifying Retriever protocol compliance
- `PipelineExecutionError` wrapping test with diagnostics verification
- Golden path integration test mirroring README usage pattern
- Example: `examples/hybrid_rag.py` -- hybrid RAG pipeline with dense retrieval
- Example: `examples/custom_retriever.py` -- custom Retriever protocol implementation
- Example: `examples/budget_management.py` -- token budget management and overflow handling
- README sections for Priority System (1--10 scale) and Token Budgets

### Fixed
- `test_consolidator.py`: eliminated shared mutable state (`_orthogonal_index` dict) by converting to factory function pattern (`make_orthogonal_embed()`)
- `test_graph_memory.py`: updated `link_memory` unknown entity test to expect `KeyError` instead of `ValueError`
- README: fixed retrieval example with runnable `embed_fn` and `ContextItem` creation
- README: updated test count from 961 to 1088

---

## [0.1.0] - 2026-02-20

### Added
- Core context pipeline with sync/async support (`ContextPipeline`)
- Token-aware sliding window memory (`SlidingWindowMemory`)
- Summary buffer memory with progressive compaction (`SummaryBufferMemory`)
- Memory manager facade unifying conversation and persistent memory (`MemoryManager`)
- Hybrid RAG retrieval: dense, sparse (BM25), and hybrid (RRF) retrievers
- Multi-signal memory retrieval with recency/relevance/importance scoring (`ScoredMemoryRetriever`)
- Provider-agnostic formatting: Anthropic, OpenAI, and generic text formatters
- Anthropic multi-block system formatting with prompt caching support
- Protocol-based extensibility (PEP 544) for all extension points
- Token budget management with per-source allocations and overflow tracking
- Pluggable eviction policies: FIFO, importance-based, and paired (user+assistant)
- Memory decay: Ebbinghaus forgetting curve and linear decay
- Recency scoring: exponential and linear strategies
- Memory consolidation with content-hash dedup and cosine-similarity merging
- Simple graph memory with BFS traversal for entity-relationship tracking
- Memory garbage collection with two-phase expired+decayed pruning
- Memory callback protocol for lifecycle observability
- Pipeline query enrichment with memory context
- Auto-promotion of evicted turns to long-term memory
- In-memory reference implementations for all storage protocols
- JSON file-backed persistent memory store
- CLI with index and query commands (via typer+rich)
- 961 tests with 94% coverage
