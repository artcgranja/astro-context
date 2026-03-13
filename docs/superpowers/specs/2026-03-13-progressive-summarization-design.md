# Progressive Summarization Memory — Design Spec

**Date:** 2026-03-13
**Status:** Draft
**Module:** `anchor.memory`

## Overview

A multi-tier hierarchical progressive summarization system for anchor's memory module. Content cascades through 4 compression tiers as it ages, with key-fact extraction at each transition to prevent information loss. Uses anchor's `LLMProvider` for compaction.

This is listed in the v0.2.0 roadmap and extends the existing `SummaryBufferMemory` pattern.

## Problem

The current `SummaryBufferMemory` provides a basic 2-tier model: verbatim recent turns + a single running summary. This has limitations:

1. **Single compression level** — all evicted content gets the same treatment regardless of age
2. **No fact preservation** — aggressive summarization loses critical details (numbers, dates, decisions)
3. **Manual compaction** — users must provide their own `compact_fn`, no built-in LLM integration
4. **No async support** — compaction is synchronous only, blocking during LLM calls
5. **Flat priority** — summary gets a single priority level, no graduated importance

## Solution

A 4-tier cascading compression system with integrated LLM compaction and key-fact extraction.

### Tier Architecture

```
Tier 0 (Verbatim)     → Raw ConversationTurn objects in SlidingWindowMemory
Tier 1 (Detailed)     → Rich summary (~500 tokens), preserves reasoning and context
Tier 2 (Compact)      → Key points only (~100 tokens), stripped of noise
Tier 3 (Ultra-compact) → Headline-level (~20 tokens), thread continuity only
```

Content flows downward as tiers fill up. Key facts are extracted at every transition and stored in a persistent sidecar.

### Cascade Flow

```
New turn added → Tier 0 (SlidingWindowMemory)
                    │
                    ▼ (when Tier 0 exceeds max_tokens)
              Evicted turns (ConversationTurn[])
                    │
                    ├──▶ serialize_turns(turns) → str
                    │        Format: "{role}: {content}\n" per turn (no metadata)
                    │
                    ├──▶ TierCompactor.summarize(serialized, target=Tier1) → Tier 1 content
                    └──▶ TierCompactor.extract_facts(serialized) → KeyFact[]
                              │
                              ▼ (when Tier 1 exceeds max_tokens)
                    TierCompactor.summarize(tier1_content, target=Tier2) → Tier 2 content
                    TierCompactor.extract_facts(tier1_content) → KeyFact[]
                              │
                              ▼ (when Tier 2 exceeds max_tokens)
                    TierCompactor.summarize(tier2_content, target=Tier3) → Tier 3 content
                    (facts already extracted at prior levels)
```

When a higher tier receives new content, it merges with existing content via progressive compaction (the LLM receives both the existing summary and the new content to produce a unified result).

### Turn Serialization

`TierCompactor` accepts `str` content. `ProgressiveSummarizationMemory` is responsible for serializing `ConversationTurn` objects before passing to the compactor. The serialization format for Tier 0→1 transitions:

```python
def _serialize_turns(turns: list[ConversationTurn]) -> str:
    return "\n".join(f"{turn.role}: {turn.content}" for turn in turns)
```

Metadata is not included in serialization — it's conversational content only. For Tier 1→2 and 2→3 transitions, the input is already a summary string, so no serialization is needed.

## Data Models

### KeyFact

```python
class FactType(StrEnum):
    DECISION = "decision"
    ENTITY = "entity"
    NUMBER = "number"
    DATE = "date"
    PREFERENCE = "preference"
    CONSTRAINT = "constraint"

class KeyFact(BaseModel):
    """A structured fact extracted during tier transitions."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    fact_type: FactType
    content: str
    source_tier: int          # Which tier transition produced it
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    token_count: int = Field(default=0, ge=0)
```

### SummaryTier

```python
class SummaryTier(BaseModel):
    """A single compression tier holding a summary."""
    level: int                # 1, 2, or 3 (Tier 0 is the SlidingWindowMemory)
    content: str
    token_count: int = Field(default=0, ge=0)
    source_turn_count: int    # How many original turns this covers
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
```

### TierConfig

```python
@dataclass(frozen=True)
class TierConfig:
    """Configuration for a single compression tier."""
    level: int
    max_tokens: int           # Trigger cascade when exceeded
    target_tokens: int = 0    # Target size for summaries (0 = no target for Tier 0)
    priority: int = 7         # Priority when emitting ContextItems
```

Default configuration:

| Tier | max_tokens | target_tokens | priority | Description |
|------|-----------|---------------|----------|-------------|
| 0    | 4096      | 0 (verbatim)  | 7        | Live conversation window |
| 1    | 1024      | 500           | 6        | Detailed summary |
| 2    | 256       | 100           | 5        | Compact summary |
| 3    | 64        | 20            | 4        | Ultra-compact summary |

## Components

### File: `src/anchor/memory/progressive.py`

**`ProgressiveSummarizationMemory`** — Main class.

Satisfies the `ConversationMemory` protocol:
- `turns` property → returns Tier 0 (verbatim) turns
- `total_tokens` property → returns Tier 0 token count
- `to_context_items(priority)` → returns items from all tiers + facts
- `clear()` → resets all tiers and facts

Constructor:

```python
def __init__(
    self,
    max_tokens: int = 8192,
    llm: LLMProvider | str = "anthropic/claude-haiku-4-5-20251001",
    tier_config: list[TierConfig] | None = None,
    max_facts: int = 50,
    fact_token_budget: int = 500,
    tokenizer: Tokenizer | None = None,
) -> None:
```

- `max_tokens`: Total token budget across all tiers (used to derive defaults if `tier_config` not provided)
- `llm`: An `LLMProvider` instance or a model string (resolved via `create_provider()`)
- `tier_config`: Per-tier settings. Defaults derived from `max_tokens` if omitted.
- `max_facts`: Maximum number of key facts to retain (FIFO eviction of oldest when exceeded)
- `fact_token_budget`: Token budget for the facts sidecar
- `tokenizer`: Tokenizer for counting tokens (defaults to tiktoken)

Methods:

```python
def add_message(self, role: Role, content: str, **metadata) -> ConversationTurn:
    """Add a message. Triggers cascade if Tier 0 overflows."""

async def aadd_message(self, role: Role, content: str, **metadata) -> ConversationTurn:
    """Async add_message. Uses async LLM compaction."""

def add_turn(self, turn: ConversationTurn) -> None:
    """Add a pre-built turn."""

async def aadd_turn(self, turn: ConversationTurn) -> None:
    """Async add_turn."""

def to_context_items(self, priority: int = 7) -> list[ContextItem]:
    """Emit all tiers + facts as ContextItems."""

def clear(self) -> None:
    """Reset all state."""

# Introspection
@property
def tiers(self) -> dict[int, SummaryTier | None]: ...
@property
def facts(self) -> list[KeyFact]: ...
@property
def tier_tokens(self) -> dict[int, int]: ...
@property
def summary(self) -> str | None:
    """Compatibility with SummaryBufferMemory — returns Tier 1 summary."""
```

### File: `src/anchor/memory/compactor.py`

**`TierCompactor`** — Handles LLM calls for summarization and fact extraction.

```python
class TierCompactor:
    def __init__(self, llm: LLMProvider, tokenizer: Tokenizer | None = None) -> None: ...

    def summarize(
        self,
        content: str,
        target_tier: int,
        target_tokens: int,
        existing_summary: str | None = None,
    ) -> str:
        """Sync summarization. Returns summary text."""

    async def asummarize(
        self,
        content: str,
        target_tier: int,
        target_tokens: int,
        existing_summary: str | None = None,
    ) -> str:
        """Async summarization."""

    def extract_facts(self, content: str, source_tier: int) -> list[KeyFact]:
        """Sync key-fact extraction. Returns parsed facts.

        The LLM returns only {"type", "content"} pairs. The compactor
        injects source_tier, generates id, and computes token_count
        when constructing KeyFact objects from the parsed JSON."""

    async def aextract_facts(self, content: str, source_tier: int) -> list[KeyFact]:
        """Async key-fact extraction. Same source_tier injection as sync variant."""
```

**Prompt templates** (internal, not exposed):

Tier 0→1 (Detailed):
```
Summarize the following conversation preserving all reasoning, decisions made,
and important context. Be thorough but concise. Target approximately {target_tokens} tokens.

{content}
```

Tier 1→2 (Compact):
```
Compress the following summary to key points only. Remove conversational noise
and redundancy. Retain decisions, facts, and conclusions. Target approximately
{target_tokens} tokens.

{content}
```

Tier 2→3 (Ultra-compact):
```
Reduce the following to a single headline-level statement that captures the
essential thread of the conversation. Target approximately {target_tokens} tokens.

{content}
```

Progressive merge (when existing summary exists):
```
You have an existing summary and new content to incorporate. Produce a unified,
non-redundant summary that covers both. Target approximately {target_tokens} tokens.

EXISTING SUMMARY:
{existing_summary}

NEW CONTENT:
{new_content}
```

Fact extraction:
```
Extract key facts from the following content. Return a JSON array where each
element has "type" (one of: decision, entity, number, date, preference, constraint)
and "content" (the fact itself, concise).

Only extract facts that would be important to remember if the original text
were lost. Return [] if no key facts are found.

{content}
```

### Error Handling

1. **LLM summarization failure:** Fall back to raw content concatenation (truncated to target tokens). Log warning. Same pattern as `SummaryBufferMemory._handle_eviction`.
2. **Fact extraction failure:** Skip fact extraction for this cascade. Log warning. Summarization proceeds normally.
3. **JSON parsing failure:** Retry once with stricter prompt ("Return ONLY valid JSON, no markdown fences"). If still fails, skip facts.
4. **Token count mismatch:** If LLM produces a summary exceeding `max_tokens` for the tier, truncate at token boundary.

### Thread Safety

- Sync path: `threading.RLock` (reentrant lock). The outer class acquires its `RLock` around `add_turn`/`add_message`. Because `SlidingWindowMemory` calls `_handle_eviction` synchronously before returning, the eviction handler runs while the outer `RLock` is still held. Since the handler writes to `_tiers` and `_facts` (also gated by the outer `RLock`), the lock must be reentrant. Note: the `SlidingWindowMemory` has its own separate internal `Lock` — the `RLock` on the outer class is unrelated to that.
- Async path: No lock needed for tier writes in the eviction callback path — async compaction is performed after the window's `add_turn` returns, not inside the eviction callback
- The eviction callback (`_handle_eviction`) is called from within `SlidingWindowMemory`'s lock. It must NOT attempt to call back into the window. It only writes to `_tiers` and `_facts`.
- All mutable state (`_tiers`, `_facts`, `_window`) accessed only within the reentrant lock

### Observability

Defines a new `ProgressiveSummarizationCallback` protocol in `src/anchor/memory/callbacks.py`, extending the existing callback architecture. Uses the `_fire_memory_callback` helper from the same file for error-safe dispatch.

```python
@runtime_checkable
class ProgressiveSummarizationCallback(Protocol):
    """Callback protocol for progressive summarization events."""

    def on_tier_cascade(
        self, from_tier: int, to_tier: int, tokens_in: int, tokens_out: int
    ) -> None: ...

    def on_facts_extracted(self, facts: list[KeyFact], source_tier: int) -> None: ...

    def on_compaction_error(self, tier: int, error: Exception) -> None: ...
```

`ProgressiveSummarizationMemory` accepts an optional `callbacks: list[ProgressiveSummarizationCallback]` constructor parameter. Callback dispatch uses `_fire_memory_callback` from `anchor.memory.callbacks` (which internally delegates to `fire_callbacks` from `anchor._callbacks`).

## Context Output

`to_context_items(priority=7)` returns items in this order:

Priorities are always **relative** to the `priority` parameter (default 7):

| Source | Priority formula | Default (priority=7) | SourceType | Metadata |
|--------|-----------------|---------------------|------------|----------|
| Tier 3 (ultra-compact) | `priority - 3` | 4 | CONVERSATION | `{"tier": 3, "summary": True}` |
| Tier 2 (compact) | `priority - 2` | 5 | CONVERSATION | `{"tier": 2, "summary": True}` |
| Tier 1 (detailed) | `priority - 1` | 6 | CONVERSATION | `{"tier": 1, "summary": True}` |
| Tier 0 (verbatim turns) | `priority` | 7 | CONVERSATION | `{"role": "user"/"assistant"}` |
| Key facts | `priority + 1` | 8 | MEMORY | `{"fact_type": "decision", ...}` |

All tier priorities are derived from the `priority` parameter. This means calling `to_context_items(priority=5)` produces Tier 3 at priority 2, Tier 2 at 3, Tier 1 at 4, Tier 0 at 5, and facts at 6. The `TierConfig.priority` field is used only for the default configuration — at runtime, the `priority` parameter takes precedence.

## Integration Points

### MemoryManager

```python
memory = ProgressiveSummarizationMemory(llm="anthropic/claude-haiku-4-5-20251001")
manager = MemoryManager(conversation_memory=memory)
```

Works because `ProgressiveSummarizationMemory` satisfies `ConversationMemory` protocol.

**Required change to `MemoryManager._add_message`** (file: `src/anchor/memory/manager.py`):

Add a new `isinstance` branch for `ProgressiveSummarizationMemory` that calls `self._conversation.add_message(role, content)`. This follows the exact same pattern as the existing `SummaryBufferMemory` branch:

```python
elif isinstance(self._conversation, ProgressiveSummarizationMemory):
    self._conversation.add_message(role, content)
```

This change must be included in the implementation scope and tested in the integration test suite.

Additionally, update `MemoryManager.conversation_type` to recognize the new class:

```python
if isinstance(self._conversation, ProgressiveSummarizationMemory):
    return "progressive_summarization"
```

This branch should be checked before the `SummaryBufferMemory` branch (since `ProgressiveSummarizationMemory` is not a subclass of it, order doesn't technically matter, but placing it first is consistent).

### ContextPipeline

```python
pipeline = ContextPipeline(max_tokens=16384)
pipeline.with_memory(manager)
result = pipeline.run(query)
```

No changes needed to `ContextPipeline` — it consumes `ContextItem` objects from `MemoryProvider.get_context_items()`.

### Public API Export

Add to `src/anchor/__init__.py`:
```python
from anchor.memory.progressive import ProgressiveSummarizationMemory
from anchor.memory.compactor import TierCompactor
```

Add to `src/anchor/memory/__init__.py` (if exists, otherwise in the main init).

## Testing Plan

### Unit Tests: `tests/test_memory/test_progressive.py`

1. **Construction**
   - Default config from `max_tokens`
   - Custom `tier_config`
   - String LLM resolution (`"anthropic/claude-haiku-4-5-20251001"` → provider)
   - Invalid config (overlapping budgets, negative tokens) → ValueError

2. **Cascade Behavior**
   - Add turns until Tier 0 overflows → verify Tier 1 created
   - Continue until Tier 1 overflows → verify Tier 2 created
   - Full cascade through all 4 tiers
   - Progressive merge: new evictions merge with existing tier content

3. **Key Fact Extraction**
   - Facts extracted at each tier transition
   - FIFO eviction when `max_facts` exceeded
   - Fact token budget respected
   - Malformed JSON → graceful skip

4. **Context Output**
   - `to_context_items()` returns correct priorities
   - Tier ordering in output
   - Facts always included at highest priority
   - Empty tiers omitted from output

5. **Protocol Compliance**
   - `turns` returns Tier 0 turns
   - `total_tokens` returns Tier 0 count
   - `clear()` resets everything
   - Satisfies `ConversationMemory` at runtime (`isinstance` check)

6. **Async Symmetry**
   - `aadd_turn` delegates to async LLM compaction (not sync)
   - `aadd_message` uses async compaction path
   - Verify no sync LLM calls in async code path

7. **Thread Safety**
   - Concurrent `add_message` calls don't corrupt state
   - RLock prevents deadlock in eviction callback chain

8. **Error Handling**
   - LLM failure → fallback to raw concatenation
   - Fact extraction failure → skip, log warning
   - Over-budget summary → truncation

### Unit Tests: `tests/test_memory/test_compactor.py`

1. **Summarization**
   - Correct prompt sent for each tier level
   - Progressive merge includes existing summary in prompt
   - Token target passed to prompt

2. **Fact Extraction**
   - Valid JSON parsing → `KeyFact` objects
   - Empty array → empty list
   - Malformed JSON → retry → empty list
   - Invalid fact types → filtered out

3. **Async Variants**
   - `asummarize` calls `ainvoke`
   - `aextract_facts` calls `ainvoke`

### Integration Tests: `tests/test_memory/test_progressive_integration.py`

1. **MemoryManager integration** — `ProgressiveSummarizationMemory` as conversation backend, verify `add_user_message()` / `add_assistant_message()` dispatch correctly via the new `isinstance` branch
2. **Full 20-turn conversation** — verify all tiers populated after sufficient messages
3. **Context pipeline** — items flow through `ContextPipeline.run()` correctly
4. **MemoryManager._add_message dispatch** — verify the new `ProgressiveSummarizationMemory` branch in `_add_message` works correctly and doesn't raise `TypeError`

All tests use mocked `LLMProvider` returning canned responses. No real API calls.

## Non-Goals

- **Persistent tier storage** — tiers live in memory only (persistent storage is v0.2.0 scope, separate from this feature)
- **Custom tier counts** — fixed at 4 tiers (0-3). Users who want different counts should compose `SummaryBufferMemory` instances.
- **Streaming compaction** — compaction uses `invoke()`/`ainvoke()`, not streaming
- **Cross-session continuity** — tiers reset per session (persistent memory handles cross-session)

## References

- [LogRocket: The LLM context problem in 2026](https://blog.logrocket.com/llm-context-problem/)
- [JetBrains Research: Smarter Context Management for LLM-Powered Agents](https://blog.jetbrains.com/research/2025/12/efficient-context-management/)
- [Maxim: Context Window Management Strategies](https://www.getmaxim.ai/articles/context-window-management-strategies-for-long-context-ai-agents-and-chatbots/)
- [Mem0: LLM Chat History Summarization Guide](https://mem0.ai/blog/llm-chat-history-summarization-guide-2025/)
- [HuggingFace: Adaptive Summarization](https://huggingface.co/blog/chansung/adaptive-summarization)
