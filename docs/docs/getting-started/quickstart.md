---
icon: material/timer-sand
---

# Quickstart

Build a working context pipeline in **30 seconds**. All you need is a system
prompt and a memory manager.

## Minimal pipeline

```python
from anchor import ContextPipeline, QueryBundle, MemoryManager

# Create pipeline with memory
memory = MemoryManager(conversation_tokens=4096)
pipeline = (
    ContextPipeline(max_tokens=8192)
    .with_memory(memory)
    .add_system_prompt("You are a helpful coding assistant.")
)

# Add conversation history
memory.add_user_message("Help me write a Python function")
memory.add_assistant_message("Sure! What should the function do?")

# Build context for the next query
result = pipeline.build(QueryBundle(query_str="It should sort a list"))
print(result.formatted_output)
```

## What just happened?

1. **MemoryManager** stores conversation turns in a sliding window capped at
   4 096 tokens.
2. **ContextPipeline** assembles a context window of up to 8 192 tokens,
   placing the system prompt first (highest priority), then conversation
   memory, then any retrieval results.
3. **`build()`** packs everything that fits into a `ContextResult` -- ready to
   send to any LLM.

!!! tip "String shorthand"
    `build()` also accepts a plain string. These two calls are equivalent:

    ```python
    result = pipeline.build("It should sort a list")
    result = pipeline.build(QueryBundle(query_str="It should sort a list"))
    ```

## What's next?

This pipeline has no retrieval -- it only uses memory and a system prompt. To
add semantic search, hybrid retrieval, token budgets, formatters, and more,
continue to **[Your First Pipeline](first-pipeline.md)**.
