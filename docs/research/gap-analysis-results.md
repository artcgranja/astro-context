# Gap Analysis Results — astro-context vs. Modern Best Practices

Generated: 2026-02-20

## Summary Table

| # | Gap | Score (0-100) | Priority | Key Finding |
|---|-----|---------------|----------|-------------|
| 1 | Long-Term Persistent Memory | 88 | Critical | MemoryEntry is dead code. Zero cross-session persistence, zero fact extraction. |
| 2 | Progressive Summarization | 85 | High | Destructive FIFO eviction. Evicted turns permanently lost, no summarization. |
| 3 | Structured Output & MCP | 85 | Critical | Zero tool_use, function calling, JSON schema, MCP support. SourceType.TOOL is cosmetic. |
| 4 | Prompt Caching | 82 | Critical | System prompt is plain string (not content blocks). Zero cache_control. ~90% cost savings lost. |
| 5 | Streaming Support | 82 | High | Zero streaming helpers. No response token tracking. Chat example is ad-hoc. |
| 6 | Observability & Tracing | 78 | High | Basic diagnostics exist. Zero OpenTelemetry, zero callbacks/hooks, zero structured logging. |
| 7 | Dynamic Budget Allocation | 78 | Critical | TokenBudget/BudgetAllocation are dead code. Pipeline uses flat max_tokens. No reserve_tokens. |
| 8 | Pipeline Error Handling | 75 | High | Fail-fast only. Zero retry, fallback, graceful degradation, circuit breaker. |
| 9 | Reranking & Query Transform | 72 | High | PostProcessor protocol supports reranking but zero implementations. Zero query transformation. |
| 10 | Message Ordering | 35 | High | Fix applied to Anthropic/OpenAI formatters. Missing: GenericFormatter, role alternation, centralized sort. |

Score: 0 = perfect as-is, 100 = needs major changes

---

## Dead Code Identified

| Model | File | Issue |
|-------|------|-------|
| `TokenBudget` / `BudgetAllocation` | models/budget.py | Fully tested but never consumed by ContextPipeline |
| `MemoryEntry` | models/memory.py | Defined with Mem0-style fields but never instantiated |
| `TokenBudgetExceededError` | exceptions.py | Defined and exported but never raised |

---

## Remaining Bugs

1. **GenericTextFormatter** does NOT sort memory items by `created_at` — conversation in wrong order
2. **`_ALLOWED_ROLES`** excludes `"tool"` — tool messages silently become `"user"`
3. **Context block insertion** in AnthropicFormatter can create consecutive user messages (API rejection)
4. **No integration tests** for full pipeline path (memory → priority sort → formatter ordering)

---

## Top 10 Recommendations (by Impact)

### P0 — Critical (activate dead code, fix structural issues)

1. **Integrate TokenBudget into ContextPipeline** — Accept optional `budget: TokenBudget`, enforce per-source-type allocations, subtract `reserve_tokens` from available window.

2. **Prompt caching in AnthropicFormatter** — Convert system prompt from plain string to content block array. Add `cache_control: {"type": "ephemeral"}` support. Estimated 90% cost reduction, 85% latency reduction.

3. **Activate MemoryEntry for persistent memory** — Wire MemoryEntry model into MemoryManager with a PersistentMemoryStore protocol and at least one backend (SQLite/JSON).

4. **Tool-use support in formatters** — Add `tools` key to output, support `tool_use`/`tool_result` content blocks (Anthropic) and `role: "tool"` messages (OpenAI). Add `"tool"` to `_ALLOWED_ROLES`.

### P1 — High (resilience, observability, extensibility)

5. **Graceful degradation in pipeline** — Add `on_error` field to PipelineStep (`"raise"` | `"skip"` | `"fallback"`). Allow partial context assembly when non-critical steps fail.

6. **Eviction hooks for summarization** — Add callback to SlidingWindowMemory.add_turn() when turns are evicted. Enable progressive summarization without breaking existing API.

7. **Callback/event hook system** — Add PipelineCallback protocol (on_step_start, on_step_end, on_pipeline_end, on_error). Prerequisite for Langfuse/LangSmith/OTel integration.

8. **Centralize memory ordering** — Move `sorted(memory_items, key=created_at)` into `classify_window_items()`. Fix GenericTextFormatter. Add Anthropic role alternation enforcement.

### P2 — Medium (advanced features)

9. **Built-in reranker** — Add CrossEncoderReranker PostProcessor and/or CohereReranker. Add two_stage_step() convenience factory.

10. **Streaming helpers** — StreamSession abstraction, structured event types (StreamDelta, StreamUsage, StreamComplete), token usage feedback loop.

---

## Architecture Strengths (What's Working Well)

- Protocol-based design allows extension without modification
- Pipeline step system is composable and supports sync/async
- Decorator API (@pipeline.step) is clean and Pythonic
- Diagnostics foundation exists (per-step timing, token utilization)
- HybridRetriever with RRF fusion is well-implemented
- ContextItem model is flexible with metadata dict
- Test coverage is thorough (312 tests) for existing behavior
- Clean exception hierarchy with cause chaining

---

## Implementation Roadmap Suggestion

### Phase 1: Fix & Activate (v0.2.0)
- Fix message ordering bugs (GenericFormatter, role alternation, centralize sort)
- Integrate TokenBudget into pipeline (activate dead code)
- Add graceful degradation to PipelineStep
- Add eviction hooks to SlidingWindowMemory

### Phase 2: Provider Features (v0.3.0)
- Prompt caching in AnthropicFormatter
- Tool-use/function-calling in formatters
- Callback/event hook system
- Streaming helpers

### Phase 3: Memory & Intelligence (v0.4.0)
- Persistent memory with MemoryEntry
- Progressive summarization
- Built-in reranker implementations
- Query transformation support

### Phase 4: Ecosystem (v0.5.0)
- OpenTelemetry integration
- MCP server/client adapters
- Structured output/JSON schema support
- Langfuse/LangSmith integrations
