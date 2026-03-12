# Chatbot with Memory

Build a conversational chatbot with sliding-window memory, automatic eviction,
and persistent facts -- all without an external API key.

---

## Overview

This example demonstrates:

- Creating a `MemoryManager` with `SlidingWindowMemory`
- Building a `ContextPipeline` with memory, system prompt, and formatter
- Simulating a multi-turn conversation
- Observing automatic eviction when the token budget fills up
- Storing persistent facts with `add_fact()`

## Full Example

```python
from astro_context import (
    ContextPipeline,
    MemoryManager,
    SlidingWindowMemory,
    AnthropicFormatter,
    GenericTextFormatter,
    InMemoryEntryStore,
    QueryBundle,
)

# ---------------------------------------------------------------
# 1. Create a memory manager with a small token budget
#    (small budget so we can see eviction in action)
# ---------------------------------------------------------------
evicted_log: list[str] = []

def on_evict(turns):
    """Callback fired when turns are evicted from the window."""
    for t in turns:
        evicted_log.append(f"  Evicted [{t.role}]: {t.content[:60]}...")
    print(f">> {len(turns)} turn(s) evicted from memory")

memory = MemoryManager(
    conversation_tokens=150,  # intentionally small for demo
    persistent_store=InMemoryEntryStore(),
    conversation_memory=SlidingWindowMemory(
        max_tokens=150,
        on_evict=on_evict,
    ),
)

# ---------------------------------------------------------------
# 2. Build the pipeline
# ---------------------------------------------------------------
pipeline = (
    ContextPipeline(max_tokens=512)
    .with_memory(memory)
    .with_formatter(GenericTextFormatter())
    .add_system_prompt("You are a helpful travel assistant.")
)

# ---------------------------------------------------------------
# 3. Simulate a multi-turn conversation
# ---------------------------------------------------------------
conversations = [
    ("user", "I'm planning a trip to Japan next spring."),
    ("assistant", "Great choice! Spring is perfect for cherry blossoms. "
                  "Would you like suggestions for Tokyo, Kyoto, or Osaka?"),
    ("user", "I'd love to visit Kyoto. What are the must-see temples?"),
    ("assistant", "In Kyoto, don't miss Kinkaku-ji (Golden Pavilion), "
                  "Fushimi Inari Shrine, and Arashiyama Bamboo Grove."),
    ("user", "How many days do you recommend for Kyoto?"),
    ("assistant", "I'd recommend 3-4 days for Kyoto to cover the major "
                  "temples, shrines, and the Gion district."),
    ("user", "What about food recommendations?"),
]

print("=== Conversation Simulation ===\n")
for role, content in conversations:
    if role == "user":
        memory.add_user_message(content)
        print(f"User: {content}")
    else:
        memory.add_assistant_message(content)
        print(f"Assistant: {content}")

# ---------------------------------------------------------------
# 4. Build context and inspect what survived eviction
# ---------------------------------------------------------------
print("\n=== Building Context ===\n")
result = pipeline.build("What about food recommendations?")

print(f"Format type: {result.format_type}")
print(f"Items included: {result.diagnostics.get('items_included', 0)}")
print(f"Token utilization: {result.diagnostics.get('token_utilization', 0):.1%}")

# Show which turns are still in memory
print("\n=== Surviving Memory Turns ===\n")
turns = memory.conversation.turns
for turn in turns:
    print(f"  [{turn.role}] {turn.content[:80]}...")

# Show what was evicted
if evicted_log:
    print("\n=== Evicted Turns ===\n")
    for line in evicted_log:
        print(line)

# ---------------------------------------------------------------
# 5. Persistent facts survive eviction
# ---------------------------------------------------------------
print("\n=== Persistent Facts ===\n")

memory.add_fact("User is planning a trip to Japan in spring")
memory.add_fact("User wants to visit Kyoto for 3-4 days")
memory.add_fact("User is interested in temples and food")

# Facts persist even as conversation turns are evicted
facts = memory.get_all_facts()
for fact in facts:
    print(f"  [{fact.id[:8]}] {fact.content}")

# Facts appear in context at priority 8 (above conversation at 7)
result = pipeline.build("Summarize what you know about me.")
context_items = result.window.items
print(f"\n  Total context items: {len(context_items)}")
for item in context_items:
    src = item.source.value
    print(f"  [{src}] (priority={item.priority}) {item.content[:60]}...")

# ---------------------------------------------------------------
# 6. Duplicate facts are automatically deduplicated
# ---------------------------------------------------------------
print("\n=== Deduplication ===\n")
before = len(memory.get_all_facts())
memory.add_fact("User is planning a trip to Japan in spring")  # duplicate
after = len(memory.get_all_facts())
print(f"  Facts before duplicate add: {before}")
print(f"  Facts after duplicate add:  {after}")
print(f"  Deduplication working: {before == after}")
```

## Key Concepts

### Memory Priority

| Priority | Source | Description |
|----------|--------|-------------|
| 10 | System prompts | Always included first |
| 8 | Persistent facts | Long-term knowledge via `add_fact()` |
| 7 | Conversation turns | Recent chat history |

When context exceeds `max_tokens`, lower-priority items are dropped first.
Persistent facts at priority 8 survive even when conversation turns are evicted.

### Eviction Callback

The `on_evict` callback fires whenever turns are removed from the sliding window.
Use it to log, summarize, or archive evicted turns:

```python
def on_evict(turns):
    for turn in turns:
        archive_to_database(turn)
```

### Custom Eviction Policies

Replace FIFO eviction with importance-based eviction:

```python
from astro_context import ImportanceEviction, SlidingWindowMemory

window = SlidingWindowMemory(
    max_tokens=4096,
    eviction_policy=ImportanceEviction(),
)
```

!!! tip "Choosing a Token Budget"
    Set `conversation_tokens` to roughly 50-60% of your total `max_tokens`
    to leave room for system prompts and retrieval results.

!!! warning "Memory is Per-Instance"
    `SlidingWindowMemory` is in-memory only. For persistence across sessions,
    use `JsonFileMemoryStore` as the `persistent_store` or implement a custom
    `MemoryEntryStore`.

## Next Steps

- [RAG Pipeline](rag-pipeline.md) -- add retrieval to your chatbot
- [Document Ingestion](document-ingestion.md) -- ingest documents for RAG
- [Agent with Tools](agent-with-tools.md) -- give the chatbot tools
