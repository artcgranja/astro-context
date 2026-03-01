# Multi-Modal API Reference

The multi-modal module provides content models, encoders, table parsers,
and a converter for bridging multi-modal items with the text-based pipeline.

All classes are importable from `astro_context`:

```python
from astro_context import (
    ModalityType, MultiModalContent, MultiModalItem, MultiModalConverter,
    TextEncoder, TableEncoder, ImageDescriptionEncoder, CompositeEncoder,
    MarkdownTableParser, HTMLTableParser,
)
```

---

## ModalityType

Enum of supported content modalities.

```python
class ModalityType(StrEnum):
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    CODE = "code"
    AUDIO = "audio"
```

---

## MultiModalContent

Represents a single content modality with optional raw binary data.
Frozen (immutable) after creation.

### Constructor

```python
class MultiModalContent(BaseModel):
    modality: ModalityType
    content: str
    raw_data: bytes | None = None
    mime_type: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
```

**Fields**

| Field | Type | Default | Description |
|---|---|---|---|
| `modality` | `ModalityType` | required | Content type |
| `content` | `str` | required | Text representation or description |
| `raw_data` | `bytes \| None` | `None` | Raw binary data (images, audio) |
| `mime_type` | `str \| None` | `None` | MIME type for `raw_data` |
| `metadata` | `dict[str, Any]` | `{}` | Arbitrary metadata |

---

## MultiModalItem

Groups multiple `MultiModalContent` pieces into a single retrievable unit.
Frozen (immutable) after creation.

### Constructor

```python
class MultiModalItem(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    contents: list[MultiModalContent]
    source: SourceType
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    priority: int = Field(default=5, ge=1, le=10)
    token_count: int = Field(default=0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
```

**Fields**

| Field | Type | Default | Description |
|---|---|---|---|
| `id` | `str` | auto UUID | Unique identifier |
| `contents` | `list[MultiModalContent]` | required | Content pieces |
| `source` | `SourceType` | required | Origin type (`RETRIEVAL`, `MEMORY`, etc.) |
| `score` | `float` | `0.0` | Relevance score (0.0--1.0) |
| `priority` | `int` | `5` | Priority level (1--10, lower = higher) |
| `token_count` | `int` | `0` | Estimated token count |
| `metadata` | `dict[str, Any]` | `{}` | Arbitrary metadata |
| `created_at` | `datetime` | now (UTC) | Creation timestamp |

---

## MultiModalConverter

Static utility class for converting between `MultiModalItem` and `ContextItem`.

### Methods

#### to_context_item

```python
@staticmethod
def to_context_item(item: MultiModalItem, encoder: ModalityEncoder) -> ContextItem
```

Convert a single `MultiModalItem` to a `ContextItem`. All content pieces
are encoded to text and concatenated with double newlines. The resulting
`ContextItem` has `metadata["multimodal"] = True`.

#### to_context_items

```python
@staticmethod
def to_context_items(
    items: list[MultiModalItem], encoder: ModalityEncoder
) -> list[ContextItem]
```

Batch convert a list of `MultiModalItem` objects.

#### from_context_item

```python
@staticmethod
def from_context_item(
    item: ContextItem, modality: ModalityType = ModalityType.TEXT
) -> MultiModalItem
```

Convert a `ContextItem` back to a `MultiModalItem`. Wraps the text content
in a single `MultiModalContent` of the specified modality.

---

## TextEncoder

Pass-through encoder for text content. Returns the `content` field unchanged.

### Constructor

```python
class TextEncoder:
    def __init__(self) -> None
```

### Methods

#### encode

```python
def encode(self, content: MultiModalContent) -> str
```

Return the text content as-is.

### Properties

| Property | Type | Value |
|---|---|---|
| `supported_modalities` | `list[ModalityType]` | `[ModalityType.TEXT]` |

---

## TableEncoder

Converts table content to Markdown text. Pass-through if already Markdown.

### Constructor

```python
class TableEncoder:
    def __init__(self) -> None
```

### Methods

#### encode

```python
def encode(self, content: MultiModalContent) -> str
```

Return the table content (already in Markdown format).

### Properties

| Property | Type | Value |
|---|---|---|
| `supported_modalities` | `list[ModalityType]` | `[ModalityType.TABLE]` |

---

## ImageDescriptionEncoder

Encodes image content into text via an optional description callback.

### Constructor

```python
class ImageDescriptionEncoder:
    def __init__(self, describe_fn: Callable[[bytes], str] | None = None) -> None
```

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `describe_fn` | `Callable[[bytes], str] \| None` | `None` | Callback to generate text from image bytes |

### Methods

#### encode

```python
def encode(self, content: MultiModalContent) -> str
```

Encode image content into text. Fallback order:

1. `describe_fn(raw_data)` if both are available
2. `metadata["description"]` if present and non-empty
3. `content` field as last resort

### Properties

| Property | Type | Value |
|---|---|---|
| `supported_modalities` | `list[ModalityType]` | `[ModalityType.IMAGE]` |

---

## CompositeEncoder

Routes encoding to the appropriate sub-encoder based on modality type.

### Constructor

```python
class CompositeEncoder:
    def __init__(
        self,
        encoders: dict[ModalityType, TextEncoder | TableEncoder | ImageDescriptionEncoder] | None = None,
    ) -> None
```

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `encoders` | `dict \| None` | `None` | Custom encoder mapping. Defaults to TEXT, TABLE, IMAGE, CODE |

Default encoders:

| Modality | Encoder |
|---|---|
| `TEXT` | `TextEncoder()` |
| `TABLE` | `TableEncoder()` |
| `IMAGE` | `ImageDescriptionEncoder()` |
| `CODE` | `TextEncoder()` |

### Methods

#### encode

```python
def encode(self, content: MultiModalContent) -> str
```

Encode by delegating to the appropriate sub-encoder. Raises `ValueError`
if no encoder is registered for the content's modality.

### Properties

| Property | Type | Description |
|---|---|---|
| `supported_modalities` | `list[ModalityType]` | All registered modality types |

---

## MarkdownTableParser

Extracts tables from Markdown text using regex.

### Constructor

```python
class MarkdownTableParser:
    def __init__(self) -> None
```

### Methods

#### extract_tables

```python
def extract_tables(self, source: Path | bytes) -> list[MultiModalContent]
```

Extract Markdown tables from a file path or raw bytes. Returns a list of
`MultiModalContent` objects with `modality=TABLE` and
`metadata={"format": "markdown"}`.

---

## HTMLTableParser

Extracts `<table>` elements from HTML and converts them to Markdown.

### Constructor

```python
class HTMLTableParser:
    def __init__(self) -> None
```

### Methods

#### extract_tables

```python
def extract_tables(self, source: Path | bytes) -> list[MultiModalContent]
```

Extract HTML tables from a file path or raw bytes. Each table is converted
to Markdown format. Returns a list of `MultiModalContent` objects with
`modality=TABLE` and `metadata={"format": "html", "original_format": "html"}`.

---

## See Also

- [Multi-Modal Guide](../guides/multimodal.md) -- usage guide with examples
- [Protocols Reference](../api/protocols.md) -- `ModalityEncoder` and `TableExtractor` protocols
