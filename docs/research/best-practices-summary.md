# Best Practices Summary: Context Engineering & AI Pipelines (2025-2026)

Compiled from 6 parallel research agents. Use this as the reference for comparing against astro-context's current implementation.

---

## 1. RAG (Retrieval-Augmented Generation)

### Key Practices
- **Hybrid Search (Dense + Sparse)**: Combine vector embeddings with BM25/SPLADE. Use Reciprocal Rank Fusion (RRF) to merge results. 15-30% better accuracy than pure vector search.
- **Reranking**: Two-stage retrieval — broad recall (50-100 candidates) then cross-encoder reranking to top-k. 23-52% improvement.
- **Contextual Chunking (Anthropic)**: Prepend document-level context to each chunk before embedding. Reduces retrieval failure by 67%.
- **Semantic Chunking**: Split by meaning (embedding similarity) not arbitrary size. +9% recall.
- **Query Transformation**: Multi-query generation, HyDE, sub-question decomposition for complex queries.
- **Agentic RAG**: LLM acts as agent deciding retrieval strategy per query. Now "table stakes" for production.

### Frameworks
- LangGraph, LlamaIndex, Haystack 2.x, DSPy
- Vector DBs: Qdrant, Weaviate, Pinecone, Milvus, Chroma

### Sources
- [Anthropic: Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
- [8 RAG Architectures (Humanloop)](https://humanloop.com/blog/rag-architectures)
- [Hybrid Search & Reranking (Superlinked)](https://superlinked.com/vectorhub/articles/optimizing-rag-with-hybrid-search-reranking)

---

## 2. Context Engineering

### Key Practices
- **Priority-Based Assembly**: Assemble context in deterministic layers: pinned instructions > reference > memory > tools > history > current turn.
- **Token Budgeting**: Dynamically allocate budget across components. A focused 300-token context outperforms unfocused 113K tokens.
- **Context Compression**: Summarization (50-80% info retention), observation masking (52% cheaper than summarization), LLMLingua (20x compression).
- **Cache-Friendly Design**: Static content first, dynamic content last. Avoid timestamps/IDs in system prompts.
- **Just-in-Time Loading**: Maintain lightweight identifiers, load data via tools when needed.
- **Sub-Agent Isolation**: Split complex tasks across sub-agents with clean context windows. 90.2% improvement over single-agent.

### The Four Operations
1. **Write** — Save context outside the window (scratchpads, persistent memory)
2. **Select** — Pull the right context at the right time (RAG, memory retrieval)
3. **Compress** — Retain only required tokens (summarization, pruning)
4. **Isolate** — Split context across separate agents

### Sources
- [Anthropic: Effective Context Engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- [Chroma: Context Rot](https://research.trychroma.com/context-rot)
- [Weaviate: Context Engineering](https://weaviate.io/blog/context-engineering)
- [LangChain: Context Engineering for Agents](https://blog.langchain.com/context-engineering-for-agents/)

---

## 3. Conversation Memory

### Key Practices
- **Multi-Tier Memory**: Short-term (context window) + medium-term (session checkpoints) + long-term (persistent store).
- **Progressive Summarization**: Keep recent turns verbatim, summarize older ones. Hybrid buffer+summary is the standard pattern.
- **Memory Formation > Summarization**: Extract discrete facts/preferences (Mem0 approach). 26% quality improvement, 90% token reduction vs full-context.
- **Three Memory Types**: Semantic (facts), Episodic (experiences), Procedural (skills/behaviors).
- **Retrieval Scoring**: `score = alpha * recency + beta * relevance + gamma * importance` (Generative Agents model).
- **Selective Memory**: Both what to store AND what to remove. Up to 10% performance gains.

### Token Budget Allocation
| Component | Typical % |
|-----------|-----------|
| System prompt | 10-15% |
| Retrieved memories/RAG | 20-30% |
| Recent conversation | 30-40% |
| Tool descriptions+outputs | 10-20% |
| Reserved for response | 15-20% |

### Notable Implementations
- **Mem0**: Extraction + graph memory. 26% improvement over OpenAI memory, 91% lower latency.
- **Zep/Graphiti**: Temporal knowledge graph. 94.8% accuracy, sub-200ms retrieval.
- **Letta (MemGPT)**: OS analogy — context window = RAM, external storage = disk.
- **LangMem SDK**: Semantic + episodic + procedural memory extraction.

### Sources
- [Mem0 Research](https://mem0.ai/research)
- [LangMem Conceptual Guide](https://langchain-ai.github.io/langmem/concepts/conceptual_guide/)
- [Zep Knowledge Graph Architecture](https://arxiv.org/abs/2501.13956)
- [Generative Agents (Park et al.)](https://dl.acm.org/doi/fullHtml/10.1145/3586183.3606763)

---

## 4. Token Analysis & Budget Management

### Key Practices
- **Exact Tokenizer per Model**: Different models produce different counts for same text. 20-30% error with wrong tokenizer.
- **Dynamic Budget Allocation**: Adjust per-request based on query complexity. TALE framework: 67% cost reduction.
- **Smart Trimming**: Priority-based eviction (not naive truncation). Preserve structural integrity (sentence/paragraph boundaries).
- **Prompt Caching**: Anthropic: 90% cost reduction, 85% latency reduction. OpenAI: 50% cost reduction, automatic.
- **Model Routing**: Route simple queries to cheaper models. vLLM Semantic Router: 47% latency reduction, 48% token savings.
- **Output Token Control**: Output costs 3-5x more than input. Set `max_tokens` explicitly, use structured outputs.

### Observability
- **OpenTelemetry GenAI Conventions**: Standardized attributes (`gen_ai.usage.input_tokens`, etc.).
- Track: token utilization ratio, cost per conversation, cache hit rate, wasted tokens.
- Platforms: Langfuse (open source), Helicone, Arize Phoenix, LangSmith.

### Sources
- [Token Counting Guide (Winder AI)](https://winder.ai/calculating-token-counts-llm-context-windows-practical-guide/)
- [TALE: Token-Budget-Aware Reasoning (ACL 2025)](https://arxiv.org/abs/2412.18547)
- [Prompt Caching Economics](https://promptbuilder.cc/blog/prompt-caching-token-economics-2025)

---

## 5. LLM API Integration & Formatting

### Key Practices
- **Anthropic**: Strict role alternation (user/assistant). System prompt as top-level parameter. Content blocks model (text, image, tool_use, tool_result, thinking).
- **OpenAI**: Responses API recommended over Chat Completions for new projects. Structured Outputs with `strict: true` for 100% schema compliance.
- **Multi-Provider Abstraction**: LiteLLM (Python) or Vercel AI SDK (TypeScript). Avoid vendor lock-in.
- **Streaming**: SSE standard. Target TTFT <300-700ms. Use structured events, not raw text.
- **Error Handling**: Exponential backoff + jitter. Classify transient vs permanent errors. Circuit breakers for systemic failures.
- **Tool Use**: Detailed descriptions (3-4+ sentences). MCP as standard for tool integration.
- **Prompt Caching**: Static content first, dynamic last. Multi-layer caching: semantic cache > prefix cache > full inference.

### Sources
- [Anthropic Tool Use Guide](https://platform.claude.com/docs/en/agents-and-tools/tool-use/implement-tool-use)
- [OpenAI Responses vs Chat Completions](https://platform.openai.com/docs/guides/responses-vs-chat-completions)
- [MCP Specification (Nov 2025)](https://modelcontextprotocol.io/specification/2025-11-25)
- [Instructor Library](https://python.useinstructor.com/)

---

## 6. Pipeline Architecture & Orchestration

### Key Practices
- **DAG-Based Pipelines**: Directed acyclic graphs with typed schema contracts on every edge. Supersedes linear chains.
- **Composable Steps**: Self-contained units with typed I/O, single responsibility, stateless execution. Middleware for cross-cutting concerns.
- **Async Execution**: All I/O-bound steps should be async. 2x+ performance improvement. Rate-limit with `asyncio.Semaphore`.
- **Three-Layer Error Handling**: Retry (exponential backoff) → Fallback (secondary model) → Circuit Breaker (remove failing provider).
- **Testing**: Dataset + Evaluator + Threshold pattern (not assertions). RAGAS for RAG metrics. DeepEval as "pytest for LLMs".
- **Observability**: OpenTelemetry-based tracing with hierarchical spans. Token/cost/latency per step.

### Agentic Patterns
- **ReAct**: Thought → Action → Observation loop. Reduces hallucination through grounded reasoning.
- **Plan-and-Execute**: Separate planner from executor. More predictable, cost-efficient.
- **Reflection**: Memory of past failures for in-session learning.

### Framework Selection
| Use Case | Framework |
|----------|-----------|
| Data-heavy RAG | LlamaIndex |
| Complex multi-step workflows | LangGraph |
| Production enterprise RAG | Haystack |
| Systematic optimization | DSPy |
| Azure/.NET enterprise | Semantic Kernel |

### Sources
- [Modular LLM Pipelines (EmergentMind)](https://www.emergentmind.com/topics/modular-llm-pipelines)
- [Haystack Pipelines](https://docs.haystack.deepset.ai/docs/pipelines)
- [ReAct Agent (IBM)](https://www.ibm.com/think/topics/react-agent)

---

## Gaps to Evaluate Against astro-context

### Currently Implemented in astro-context
- [x] Priority-based context assembly (ContextWindow.add_items_by_priority)
- [x] Sliding window memory (SlidingWindowMemory)
- [x] Token counting (tiktoken via TokenCounter)
- [x] Multi-provider formatting (Anthropic, OpenAI, Generic)
- [x] Pipeline with composable steps (PipelineStep)
- [x] Async pipeline support (abuild)
- [x] Hybrid retrieval (dense + sparse + hybrid retrievers)
- [x] Diagnostics (build time, token utilization, items included/overflow)
- [x] Decorator-based step registration (@pipeline.step)

### Potential Gaps to Investigate
- [ ] **Message ordering bug**: Memory items reordered by score in window (confirmed & fixed in formatters)
- [ ] **Progressive summarization**: No summary memory — only hard eviction
- [ ] **Context compression**: No LLMLingua-style or summarization-based compression
- [ ] **Prompt caching support**: No cache_control breakpoints in Anthropic formatter
- [ ] **Reranking step**: No built-in reranker (cross-encoder or ColBERT)
- [ ] **Query transformation**: No multi-query, HyDE, or sub-question decomposition
- [ ] **Dynamic budget allocation**: Static max_tokens, no per-component budgets
- [ ] **Long-term persistent memory**: No cross-session memory store
- [ ] **Structured output support**: No JSON Schema / Pydantic output validation
- [ ] **Error handling in pipeline**: Basic exception handling, no retry/fallback/circuit breaker
- [ ] **Observability/tracing**: Diagnostics dict but no OpenTelemetry integration
- [ ] **Evaluation framework**: No built-in RAGAS/DeepEval integration
- [ ] **MCP support**: No Model Context Protocol integration
- [ ] **Streaming support**: No built-in streaming helpers
