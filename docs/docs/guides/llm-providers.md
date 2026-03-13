# LLM Providers Guide

The `anchor.llm` module provides a unified interface for multiple LLM providers
through the `LLMProvider` protocol. Provider-specific SDKs are optional
dependencies -- install only the ones you need.

## Overview

The LLM provider system is built around three layers:

1. **`LLMProvider` protocol** -- defines the standard interface (`invoke`, `stream`, `ainvoke`, `astream`)
2. **`BaseLLMProvider`** -- abstract base class with built-in retries, error mapping, and fallback support
3. **Concrete providers** -- thin adapters for each SDK (Anthropic, OpenAI, Gemini, etc.)

```
Model string "openai/gpt-4o"
    |
    v
create_provider()  -->  resolves prefix  -->  OpenAIProvider
    |
    v
Agent or direct usage
    |
    v
invoke / stream / ainvoke / astream
```

## Quick Start

```python
from anchor import Agent

# Anthropic (default when no prefix)
agent = Agent(model="claude-haiku-4-5-20251001")

# OpenAI
agent = Agent(model="openai/gpt-4o")

# Google Gemini
agent = Agent(model="gemini/gemini-2.0-flash")

# Local Ollama
agent = Agent(model="ollama/llama3")
```

!!! note
    Each provider requires its own SDK. See [Installation](#installation) for
    the optional extras.

## Installation

Each provider is packaged as an optional extra:

```bash
pip install astro-anchor[anthropic]   # Anthropic
pip install astro-anchor[openai]      # OpenAI, Grok, OpenRouter
pip install astro-anchor[gemini]      # Google Gemini
pip install astro-anchor[ollama]      # Ollama (local models)
pip install astro-anchor[litellm]     # LiteLLM (catch-all)
```

Install multiple extras at once:

```bash
pip install astro-anchor[anthropic,openai,gemini]
```

## Supported Providers

| Provider | Prefix | SDK | Install Extra | Env Variable |
|---|---|---|---|---|
| Anthropic | `anthropic/` | `anthropic` | `[anthropic]` | `ANTHROPIC_API_KEY` |
| OpenAI | `openai/` | `openai` | `[openai]` | `OPENAI_API_KEY` |
| Google Gemini | `gemini/` | `google-genai` | `[gemini]` | `GOOGLE_API_KEY` |
| Grok (xAI) | `grok/` | `openai` | `[openai]` | `XAI_API_KEY` |
| Ollama | `ollama/` | `ollama` | `[ollama]` | (none, local) |
| OpenRouter | `openrouter/` | `openai` | `[openai]` | `OPENROUTER_API_KEY` |
| LiteLLM | `litellm/` | `litellm` | `[litellm]` | (varies) |

!!! tip
    Grok, OpenRouter, and standard OpenAI all share the `openai` SDK. Installing
    `astro-anchor[openai]` covers all three.

## Model String Format

Model strings follow the pattern `"provider/model-name"`. The prefix is split
on the **first** `/` only, so model names containing slashes (e.g.
`openrouter/meta-llama/llama-3-70b`) work correctly.

```python
# Explicit provider prefix
agent = Agent(model="openai/gpt-4o")

# No prefix defaults to Anthropic (backward compatible)
agent = Agent(model="claude-haiku-4-5-20251001")
# Equivalent to:
agent = Agent(model="anthropic/claude-haiku-4-5-20251001")
```

## Fallback Chains

Configure fallback providers to handle transient failures gracefully:

```python
# Primary: Anthropic, fallback: OpenAI
agent = Agent(
    model="anthropic/claude-sonnet-4-20250514",
    fallbacks=["openai/gpt-4o"],
)
```

Fallback behavior depends on the call method:

| Method | Fallback Trigger |
|---|---|
| `invoke` / `ainvoke` | Any transient error triggers the next provider |
| `stream` / `astream` | Fallback only before the first chunk is yielded |

!!! warning
    Mid-stream errors propagate directly -- once streaming has started, the
    agent cannot transparently switch providers without losing already-yielded
    content.

## Injecting a Pre-Built Provider

For advanced configuration (custom base URLs, headers, or timeouts), create a
provider instance directly and pass it to the Agent:

```python
from anchor.llm import create_provider

provider = create_provider("openai/gpt-4o", api_key="sk-...")
agent = Agent(llm=provider)
```

## Direct Provider Usage

Providers implement the `LLMProvider` protocol and can be used independently
of the Agent:

```python
from anchor.llm import create_provider, Message, Role

llm = create_provider("openai/gpt-4o")
response = llm.invoke([Message(role=Role.USER, content="Hello!")])
print(response.content)
```

All four call methods are available:

```python
# Synchronous
response = llm.invoke(messages)

# Synchronous streaming
for chunk in llm.stream(messages):
    print(chunk.content, end="")

# Async
response = await llm.ainvoke(messages)

# Async streaming
async for chunk in llm.astream(messages):
    print(chunk.content, end="")
```

## Custom Providers

Implement `BaseLLMProvider` to add support for a new backend:

```python
from anchor.llm import BaseLLMProvider, register_provider

class MyProvider(BaseLLMProvider):
    provider_name = "my_provider"

    def _resolve_api_key(self): ...
    def _do_invoke(self, messages, tools, **kwargs): ...
    def _do_stream(self, messages, tools, **kwargs): ...
    async def _do_ainvoke(self, messages, tools, **kwargs): ...
    async def _do_astream(self, messages, tools, **kwargs): ...

register_provider("my_provider", MyProvider)
```

Once registered, the prefix routes automatically:

```python
agent = Agent(model="my_provider/my-model")
```

!!! note
    `BaseLLMProvider` handles retries, error mapping, and the public
    `invoke`/`stream`/`ainvoke`/`astream` API. Subclasses only implement the
    `_do_*` methods with provider-specific SDK calls.

## Error Handling

All provider errors are normalized into a common hierarchy:

```python
from anchor.llm import ProviderError, RateLimitError

try:
    response = llm.invoke(messages)
except RateLimitError as e:
    print(f"Rate limited by {e.provider}, retry after {e.retry_after}s")
except ProviderError as e:
    print(f"Error from {e.provider}: {e}")
```

`BaseLLMProvider` includes built-in retry with exponential backoff for
transient errors (rate limits, connection timeouts, server errors). Non-transient
errors (authentication failures, invalid requests) propagate immediately.

## See Also

- [LLM API Reference](../api/llm.md) -- complete API signatures and classes
- [Agent Guide](../guides/agent.md) -- using providers through the Agent
- [Protocols Reference](../api/protocols.md) -- `LLMProvider` protocol definition
