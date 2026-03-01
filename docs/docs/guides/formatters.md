# Formatters Guide

Formatters convert the internal `ContextWindow` into the format expected by a
specific LLM provider. astro-context ships with three built-in formatters and
a `Formatter` protocol for building your own.

## Overview

After the pipeline assembles system prompts, memory, and retrieval context into
a `ContextWindow`, a formatter transforms that window into the final output.
Different providers expect different structures:

| Formatter | Provider | Output Type |
|---|---|---|
| `AnthropicFormatter` | Anthropic Messages API | `dict` with `system` + `messages` |
| `OpenAIFormatter` | OpenAI Chat Completions | `dict` with `messages` |
| `GenericTextFormatter` | Any (plain text) | `str` with section headers |

## AnthropicFormatter

Produces a dict ready for the Anthropic Messages API:

```python
from astro_context import ContextPipeline, AnthropicFormatter

pipeline = ContextPipeline(max_tokens=8192)
pipeline.with_formatter(AnthropicFormatter())
pipeline.add_system_prompt("You are a helpful assistant.")

result = pipeline.build("Hello!")
output = result.formatted_output
# output = {
#     "system": [{"type": "text", "text": "You are a helpful assistant."}],
#     "messages": [...]
# }
```

### Prompt Caching

Enable Anthropic prompt caching to reduce latency and cost:

```python
formatter = AnthropicFormatter(enable_caching=True)
```

When caching is enabled:

- The last system content block gets `"cache_control": {"type": "ephemeral"}`
- The last retrieval/context message also gets a cache control marker
- Repeated prefixes (system prompt + static context) are cached by Anthropic

!!! tip
    Prompt caching is most effective when the system prompt and retrieval
    context are stable across requests. Enable it for production workloads
    where you send many requests with the same prefix.

### Output Structure

```python
{
    "system": [
        {"type": "text", "text": "System prompt here"},
        # ... more system blocks, each from a separate system item
    ],
    "messages": [
        {"role": "user", "content": "Here is relevant context:\n\n..."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"},
        # ... alternating user/assistant messages
    ]
}
```

The formatter enforces strict user/assistant alternation required by the
Anthropic API. Consecutive messages with the same role are merged automatically.

## OpenAIFormatter

Produces a dict with a flat `messages` list for the OpenAI Chat Completions API:

```python
from astro_context import ContextPipeline, OpenAIFormatter

pipeline = ContextPipeline(max_tokens=8192)
pipeline.with_formatter(OpenAIFormatter())
pipeline.add_system_prompt("You are a helpful assistant.")

result = pipeline.build("Hello!")
output = result.formatted_output
# output = {
#     "messages": [
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "Relevant context:\n\n..."},
#         {"role": "user", "content": "Hello!"},
#     ]
# }
```

### Security

Retrieval content is placed in `user` role messages, not `system`. This prevents
prompt injection from untrusted retrieval sources from gaining system-level
authority.

!!! warning
    Content from memory and retrieval items is inserted verbatim without
    sanitization. If these items originate from untrusted sources (user-supplied
    documents, web scrapes), they may contain prompt injection payloads.
    Implement content validation or filtering before items enter the pipeline.

## GenericTextFormatter

Produces structured plain text with section headers -- useful for non-chat
models, logging, or debugging:

```python
from astro_context import ContextPipeline, GenericTextFormatter

pipeline = ContextPipeline(max_tokens=8192)
pipeline.with_formatter(GenericTextFormatter())
pipeline.add_system_prompt("You are a helpful assistant.")

result = pipeline.build("Hello!")
print(result.formatted_output)
# === SYSTEM ===
# You are a helpful assistant.
#
# === MEMORY ===
# [user] Hello!
#
# === CONTEXT ===
# Retrieved document chunk...
```

The formatter produces up to three sections (`SYSTEM`, `MEMORY`, `CONTEXT`),
each separated by blank lines. Empty sections are omitted.

## Custom Formatters

Implement the `Formatter` protocol to create your own formatter:

```python
from astro_context import Formatter, ContextPipeline
from astro_context.models import ContextWindow

class GeminiFormatter:
    """Formats context for the Google Gemini API."""

    @property
    def format_type(self) -> str:
        return "gemini"

    def format(self, window: ContextWindow) -> dict:
        contents = []
        for item in window.items:
            contents.append({
                "role": "user",
                "parts": [{"text": item.content}],
            })
        return {"contents": contents}

# Use it
pipeline = ContextPipeline(max_tokens=8192)
pipeline.with_formatter(GeminiFormatter())
```

The `Formatter` protocol requires:

| Member | Type | Description |
|---|---|---|
| `format_type` | `property -> str` | Identifier (e.g. `"anthropic"`, `"openai"`) |
| `format(window)` | `method -> str \| dict` | Converts `ContextWindow` to output |

!!! note
    `Formatter` is a PEP 544 `Protocol` -- no inheritance required. Any class
    with matching `format_type` and `format()` signatures satisfies it via
    structural subtyping.

## Choosing a Formatter

| Use Case | Formatter | Why |
|---|---|---|
| Anthropic Claude API | `AnthropicFormatter` | Native system block format |
| OpenAI GPT API | `OpenAIFormatter` | System/user/assistant roles |
| Local models, logging | `GenericTextFormatter` | Plain text, no structure |
| Agent class | `AnthropicFormatter` | Agent uses it internally |
| Custom provider | Custom `Formatter` | Match your API format |

## See Also

- [Pipeline Guide](../guides/pipeline.md) -- how formatters fit in the pipeline
- [Formatters API Reference](../api/formatters.md) -- complete signatures
- [Agent Guide](../guides/agent.md) -- Agent uses AnthropicFormatter internally
