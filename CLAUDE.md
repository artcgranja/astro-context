# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

astro-context is a **context engineering toolkit** for AI applications. It assembles, manages, and formats context for LLMs without ever calling an LLM itself. The core philosophy: *"Context is the product. The LLM is just the consumer."*

The library provides hybrid RAG pipelines, token-aware memory, provider-agnostic formatting, and protocol-based extensibility. Users provide their own embedding functions, LLM clients, and storage backends.

## Commands

```bash
uv sync                                    # Install all dependencies
uv run pytest                              # Run all tests (961 tests)
uv run pytest tests/test_pipeline/ -x -q   # Run one test directory
uv run pytest tests/test_pipeline/test_pipeline.py::test_name -x  # Run single test
uv run ruff check src/ tests/ examples/    # Lint
uv run ruff check --fix src/               # Auto-fix lint issues
uv run mypy src/                           # Type check (strict mode)
uv run pytest --cov=astro_context --cov-report=term-missing -q  # Coverage
```

Optional dependencies: `uv pip install -e ".[bm25]"` (BM25), `".[cli]"` (typer+rich), `".[anthropic]"` (Anthropic SDK), `".[all]"` (everything).

## Architecture

### Data Flow

```
ContextPipeline.build(query)
  1. Collect system items (priority=10)
  2. Collect memory items from MemoryManager (priority=7, recency-scored 0.5-1.0)
  3. Execute PipelineSteps sequentially (retrieval, post-processing, filtering)
  4. Count tokens via Tokenizer
  5. Assemble ContextWindow (sorted by -priority, -score; token-capped)
  6. Format via Formatter (Anthropic/OpenAI/Generic)
  7. Return ContextResult with diagnostics
```

### Key Design Decisions

**Protocol-based extensibility (PEP 544)**: All extension points (`Retriever`, `PostProcessor`, `Tokenizer`, `Formatter`, storage protocols) are `@runtime_checkable` Protocols. No inheritance required — any object with the right methods works.

**ContextItem is frozen**: The atomic unit (`models/context.py`) uses `ConfigDict(frozen=True)`. Mutate via `item.model_copy(update={...})`, never direct assignment.

**Memory ordering is centralized**: `formatters/utils.py:classify_window_items()` sorts memory items by `created_at`. Individual formatters must NOT re-sort. This was a critical bug fix — priority-based sorting in `add_items_by_priority()` reverses chronological order.

**Pipeline steps have error policies**: Each `PipelineStep` has `on_error: "raise" | "skip"`. On "skip", the step's failure is logged and recorded in diagnostics, and the pipeline continues with pre-step items.

**Callbacks swallow errors**: `PipelineCallback` hooks fire via `_fire()` which catches all exceptions. A buggy callback must never crash the pipeline.

**The library never calls an LLM**: Embedding functions, scoring functions, and API calls are always user-provided. The library only orchestrates context.

### Module Responsibilities

| Module | Role |
|--------|------|
| `pipeline/pipeline.py` | Main orchestrator — `build()` / `abuild()`, fluent chaining API |
| `pipeline/step.py` | `PipelineStep` dataclass + factory functions (`retriever_step`, `filter_step`, etc.) |
| `pipeline/callbacks.py` | `PipelineCallback` protocol for observability hooks |
| `formatters/` | Provider-specific output formatting. All share `utils.classify_window_items()` |
| `memory/` | `MemoryManager` facade + `SlidingWindowMemory` (FIFO, token-aware, eviction hooks) |
| `models/` | Pydantic models: `ContextItem`, `ContextWindow`, `ContextResult`, `QueryBundle`, `TokenBudget` |
| `protocols/` | PEP 544 Protocols for all extension points |
| `retrieval/` | `DenseRetriever`, `SparseRetriever`, `HybridRetriever` (RRF), `ScoreReranker` |
| `storage/` | In-memory reference implementations of storage protocols |
| `tokens/` | `TiktokenCounter` with LRU caching; `get_default_counter()` singleton |

## Code Conventions

- **Python 3.11+**: Uses `StrEnum`, `X | Y` union syntax, `from __future__ import annotations`
- **Strict mypy**: `strict = true` in pyproject.toml. All code must be fully typed.
- **Ruff**: line-length=100, rules: E, F, W, I, N, UP, B, A, SIM, RUF, C90, PT, S
- **Error messages**: Assign to `msg` variable before `raise` (ruff TRY003 compatibility): `msg = "..."; raise ValueError(msg)`
- **Slots**: Non-Pydantic classes use `__slots__` for memory efficiency
- **Lazy imports**: Heavy dependencies (tiktoken, rank_bm25) are imported inside `__init__` methods, not at module level
- **Fluent API**: Configuration methods (`with_memory()`, `with_formatter()`, `add_step()`, etc.) return `self`
- **Decorator API**: `@pipeline.step` and `@pipeline.async_step` use `@overload` for type-safe bare vs parameterized usage

## Testing Patterns

- **FakeTokenizer** (`tests/conftest.py`): Whitespace-splitting tokenizer used universally to avoid tiktoken's network dependency
- **FakeRetriever** (`tests/conftest.py`): Returns pre-configured items for pipeline tests
- **`make_embedding(seed, dim=128)`**: Deterministic sin-based fake embeddings for reproducible retrieval tests
- **pytest-asyncio**: `asyncio_mode = "auto"` — async test functions need no `@pytest.mark.asyncio`
- **Retrieval test helpers** (`tests/test_retrieval/conftest.py`): Patch `get_default_counter` to use `FakeTokenizer`
- Coverage threshold: 80% minimum (enforced in pyproject.toml)

## Context Engineering Guidelines

These guidelines are derived from research into modern best practices (documented in `docs/research/`). Apply them when extending the library:

### Token Budget Discipline
- Every context item must carry a `token_count`. The pipeline's `_count_tokens()` fills in zeros before assembly.
- `TokenBudget.reserve_tokens` is subtracted from `max_tokens` to guarantee room for the LLM response.
- Priority ranking: system (10) > memory (7) > retrieval (5, default). Higher priority items are placed first; overflow items are tracked in `ContextResult.overflow_items`.

### Cache-Friendly Context
- Static content (system prompts, instructions) should come first in the context window. Dynamic content (conversation, retrieval) comes last.
- `AnthropicFormatter(enable_caching=True)` adds `cache_control: {"type": "ephemeral"}` to system blocks and context messages for Anthropic prompt caching.

### Memory Architecture
- Current: sliding window with FIFO eviction and `on_evict` callback hook
- Memory items get recency-weighted scores (0.5 for oldest, 1.0 for newest) and `role` metadata
- The `on_evict` hook enables progressive summarization without breaking the existing API

### Parallel Agent Workflow
When tackling multiple independent improvements, use parallel agents. Each agent should:
1. Focus on a single, well-scoped module or concern
2. Add tests alongside implementation
3. Avoid editing files that other agents are likely modifying (especially `pipeline.py`, `__init__.py`)
4. Run `uv run pytest` and `uv run ruff check` before declaring done

After parallel agents finish, run a reconciliation pass: lint, test, fix any import conflicts, then commit.

## Known Dead Code (v0.1.x)

These are intentionally defined for future activation:
- `MemoryEntry` (`models/memory.py`): Mem0-style persistent memory — no cross-session store yet
- `TokenBudgetExceededError`: Defined but never raised — budget overflow uses overflow list instead
