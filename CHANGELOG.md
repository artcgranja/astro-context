# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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
