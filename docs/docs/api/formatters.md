# Formatters API Reference

Formatters convert a `ContextWindow` into the format expected by a specific
LLM provider. All formatters are importable from `astro_context`:

```python
from astro_context import (
    Formatter, AnthropicFormatter, OpenAIFormatter, GenericTextFormatter,
)
```

---

## Formatter (Protocol)

The protocol all formatters must satisfy. This is a PEP 544 structural
protocol -- no inheritance required.

### Definition

```python
@runtime_checkable
class Formatter(Protocol):
    @property
    def format_type(self) -> str: ...

    def format(self, window: ContextWindow) -> str | dict[str, Any]: ...
```

**Members**

| Member | Type | Description |
|---|---|---|
| `format_type` | `property -> str` | Identifier (e.g. `"anthropic"`, `"openai"`, `"generic"`) |
| `format(window)` | `method` | Converts a `ContextWindow` into the target format |

!!! note
    `BaseFormatter` is a deprecated alias for `Formatter`. Use `Formatter`
    in new code.

---

## AnthropicFormatter

Formats context for the Anthropic Messages API.

### Constructor

```python
class AnthropicFormatter:
    def __init__(self, *, enable_caching: bool = False) -> None
```

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `enable_caching` | `bool` | `False` | Enable Anthropic prompt caching markers |

### Properties

| Property | Type | Value |
|---|---|---|
| `format_type` | `str` | `"anthropic"` |

### Methods

#### format

```python
def format(self, window: ContextWindow) -> dict[str, Any]
```

Returns a dict with:

- `"system"` -- list of content block dicts (`{"type": "text", "text": "..."}`)
- `"messages"` -- list of message dicts with `role` and `content` keys

When `enable_caching=True`, adds `"cache_control": {"type": "ephemeral"}`
to the last system block and last context message.

The formatter enforces strict user/assistant alternation required by the
Anthropic API.

### Example

```python
from astro_context import ContextPipeline, AnthropicFormatter

pipeline = ContextPipeline(max_tokens=8192)
pipeline.with_formatter(AnthropicFormatter(enable_caching=True))
pipeline.add_system_prompt("You are helpful.")

result = pipeline.build("Hello")
output = result.formatted_output
# {"system": [...], "messages": [...]}
```

---

## OpenAIFormatter

Formats context for the OpenAI Chat Completions API.

### Constructor

```python
class OpenAIFormatter:
    def __init__(self) -> None
```

No constructor parameters.

### Properties

| Property | Type | Value |
|---|---|---|
| `format_type` | `str` | `"openai"` |

### Methods

#### format

```python
def format(self, window: ContextWindow) -> dict[str, Any]
```

Returns a dict with:

- `"messages"` -- list of message dicts with `role` (`"system"`, `"user"`,
  `"assistant"`) and `content` keys

System prompts use the `"system"` role. Retrieval content uses the `"user"`
role to prevent privilege escalation from untrusted content.

### Example

```python
from astro_context import ContextPipeline, OpenAIFormatter

pipeline = ContextPipeline(max_tokens=8192)
pipeline.with_formatter(OpenAIFormatter())
pipeline.add_system_prompt("You are helpful.")

result = pipeline.build("Hello")
output = result.formatted_output
# {"messages": [{"role": "system", "content": "..."}, ...]}
```

---

## GenericTextFormatter

Formats context as structured plain text with section headers.

### Constructor

```python
class GenericTextFormatter:
    def __init__(self) -> None
```

No constructor parameters.

### Properties

| Property | Type | Value |
|---|---|---|
| `format_type` | `str` | `"generic"` |

### Methods

#### format

```python
def format(self, window: ContextWindow) -> str
```

Returns a string with up to three sections separated by blank lines:

```
=== SYSTEM ===
System prompt text

=== MEMORY ===
Conversation history

=== CONTEXT ===
Retrieved content
```

Empty sections are omitted.

### Example

```python
from astro_context import ContextPipeline, GenericTextFormatter

pipeline = ContextPipeline(max_tokens=8192)
pipeline.with_formatter(GenericTextFormatter())
pipeline.add_system_prompt("You are helpful.")

result = pipeline.build("Hello")
print(result.formatted_output)
# === SYSTEM ===
# You are helpful.
# ...
```

---

## See Also

- [Formatters Guide](../guides/formatters.md) -- usage guide with examples
- [Pipeline API Reference](../api/pipeline.md) -- how formatters integrate
