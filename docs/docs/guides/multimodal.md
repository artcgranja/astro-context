# Multi-Modal Guide

astro-context supports multi-modal content -- text, images, tables, code, and
audio -- through a dedicated content model and encoder system. Multi-modal
items are converted to text representations for use with the text-based
context pipeline.

## Overview

The multi-modal system has three layers:

1. **Models** -- `ModalityType`, `MultiModalContent`, `MultiModalItem`
2. **Encoders** -- convert each modality to text (`TextEncoder`, `TableEncoder`,
   `ImageDescriptionEncoder`, `CompositeEncoder`)
3. **Table Parsers** -- extract tables from Markdown and HTML documents

```
Document (HTML/Markdown)
    |
    v
Table Parser --> MultiModalContent (modality=TABLE)
    |
    v
Encoder --> text representation
    |
    v
MultiModalConverter --> ContextItem
    |
    v
ContextPipeline
```

## Content Models

### ModalityType

An enum of supported content types:

```python
from astro_context import ModalityType

ModalityType.TEXT    # Plain text
ModalityType.IMAGE   # Image data
ModalityType.TABLE   # Tabular data
ModalityType.CODE    # Source code
ModalityType.AUDIO   # Audio data
```

### MultiModalContent

Represents a single modality with optional raw binary data:

```python
from astro_context import MultiModalContent, ModalityType

# Text content
text = MultiModalContent(
    modality=ModalityType.TEXT,
    content="This is a paragraph about climate change.",
)

# Image with description
image = MultiModalContent(
    modality=ModalityType.IMAGE,
    content="Photo of a sunset over the ocean",
    raw_data=image_bytes,
    mime_type="image/jpeg",
    metadata={"description": "Sunset at Malibu beach"},
)

# Table content
table = MultiModalContent(
    modality=ModalityType.TABLE,
    content="| Name | Age |\n| --- | --- |\n| Alice | 30 |",
    metadata={"format": "markdown"},
)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `modality` | `ModalityType` | required | Content type |
| `content` | `str` | required | Text representation or description |
| `raw_data` | `bytes \| None` | `None` | Raw binary data |
| `mime_type` | `str \| None` | `None` | MIME type for raw data |
| `metadata` | `dict[str, Any]` | `{}` | Additional metadata |

### MultiModalItem

Groups multiple content pieces into a single retrievable unit:

```python
from astro_context import MultiModalItem, MultiModalContent, ModalityType, SourceType

item = MultiModalItem(
    contents=[
        MultiModalContent(modality=ModalityType.TEXT, content="Chart analysis:"),
        MultiModalContent(modality=ModalityType.TABLE, content="| Q1 | Q2 |\n|---|---|\n| 100 | 150 |"),
        MultiModalContent(modality=ModalityType.IMAGE, content="Bar chart showing growth"),
    ],
    source=SourceType.RETRIEVAL,
    score=0.95,
    priority=3,
)
```

## Encoders

Encoders convert `MultiModalContent` into text for the pipeline.

### TextEncoder

Pass-through encoder for text and code content:

```python
from astro_context import TextEncoder, MultiModalContent, ModalityType

encoder = TextEncoder()
content = MultiModalContent(modality=ModalityType.TEXT, content="Hello world")
text = encoder.encode(content)  # "Hello world"
```

### TableEncoder

Converts table content to Markdown (pass-through if already Markdown):

```python
from astro_context import TableEncoder, MultiModalContent, ModalityType

encoder = TableEncoder()
content = MultiModalContent(
    modality=ModalityType.TABLE,
    content="| Name | Score |\n|---|---|\n| Alice | 95 |",
)
text = encoder.encode(content)
```

### ImageDescriptionEncoder

Converts images to text descriptions via an optional callback:

```python
from astro_context import ImageDescriptionEncoder, MultiModalContent, ModalityType

# Without callback -- uses metadata["description"] or content field
encoder = ImageDescriptionEncoder()

# With callback -- calls describe_fn(raw_bytes) for images with raw data
def describe_image(data: bytes) -> str:
    # Call a vision model, OCR service, etc.
    return "A bar chart showing quarterly revenue growth"

encoder = ImageDescriptionEncoder(describe_fn=describe_image)
```

The fallback order is:

1. `describe_fn(raw_data)` if both are available
2. `metadata["description"]` if present and non-empty
3. `content` field as last resort

### CompositeEncoder

Routes encoding to the appropriate sub-encoder based on modality:

```python
from astro_context import CompositeEncoder, MultiModalContent, ModalityType

# Default setup handles TEXT, TABLE, IMAGE, and CODE
encoder = CompositeEncoder()

text_content = MultiModalContent(modality=ModalityType.TEXT, content="Hello")
table_content = MultiModalContent(modality=ModalityType.TABLE, content="| A | B |")

print(encoder.encode(text_content))   # "Hello"
print(encoder.encode(table_content))  # "| A | B |"
print(encoder.supported_modalities)   # [TEXT, TABLE, IMAGE, CODE]
```

Provide custom encoders for specific modalities:

```python
encoder = CompositeEncoder(encoders={
    ModalityType.TEXT: TextEncoder(),
    ModalityType.IMAGE: ImageDescriptionEncoder(describe_fn=my_vision_fn),
})
```

!!! warning
    `CompositeEncoder` raises `ValueError` if it encounters a modality with no
    registered encoder.

## Table Parsers

### MarkdownTableParser

Extracts tables from Markdown text using regex:

```python
from astro_context import MarkdownTableParser

parser = MarkdownTableParser()
tables = parser.extract_tables(b"# Report\n\n| Q1 | Q2 |\n|---|---|\n| 100 | 200 |\n\nSome text.")

for table in tables:
    print(table.modality)  # ModalityType.TABLE
    print(table.content)   # "| Q1 | Q2 |\n|---|---|\n| 100 | 200 |"
```

### HTMLTableParser

Extracts `<table>` elements from HTML and converts them to Markdown:

```python
from astro_context import HTMLTableParser

parser = HTMLTableParser()
html = b"<table><tr><th>Name</th></tr><tr><td>Alice</td></tr></table>"
tables = parser.extract_tables(html)

print(tables[0].content)
# | Name |
# | --- |
# | Alice |
```

Both parsers accept either a `Path` object or raw `bytes`.

## Converting to ContextItems

Use `MultiModalConverter` to bridge multi-modal items with the text pipeline:

```python
from astro_context import (
    MultiModalConverter, MultiModalItem, MultiModalContent,
    CompositeEncoder, ModalityType, SourceType,
)

item = MultiModalItem(
    contents=[
        MultiModalContent(modality=ModalityType.TEXT, content="Summary of Q1 results"),
        MultiModalContent(modality=ModalityType.TABLE, content="| Revenue | 1.2M |"),
    ],
    source=SourceType.RETRIEVAL,
)

encoder = CompositeEncoder()
context_item = MultiModalConverter.to_context_item(item, encoder)
print(context_item.content)
# "Summary of Q1 results\n\nRevenue | 1.2M |"
print(context_item.metadata["multimodal"])  # True
```

### Batch Conversion

```python
items = [item1, item2, item3]
context_items = MultiModalConverter.to_context_items(items, encoder)
```

### Reverse Conversion

Convert a `ContextItem` back to a `MultiModalItem`:

```python
from astro_context import MultiModalConverter, ContextItem, SourceType

ctx = ContextItem(content="Hello world", source=SourceType.RETRIEVAL)
mm_item = MultiModalConverter.from_context_item(ctx, modality=ModalityType.TEXT)
```

## See Also

- [Ingestion Guide](../guides/ingestion.md) -- chunking and parsing documents
- [Multi-Modal API Reference](../api/multimodal.md) -- complete signatures
- [Protocols Reference](../api/protocols.md) -- `ModalityEncoder` and `TableExtractor` protocols
