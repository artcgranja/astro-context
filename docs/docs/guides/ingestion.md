# Ingestion Guide

The ingestion module converts raw documents into `ContextItem` objects ready for
indexing and retrieval. The pipeline follows four stages:

```
Parse --> Chunk --> Enrich --> Index
```

1. **Parse** -- extract text and metadata from files (Markdown, HTML, PDF, plain text).
2. **Chunk** -- split the text into retrieval-sized segments.
3. **Enrich** -- attach metadata (doc IDs, chunk positions, custom fields).
4. **Index** -- feed `ContextItem` objects to a retriever.

## Quick Start

```python
from astro_context.ingestion import DocumentIngester

ingester = DocumentIngester()

# Ingest raw text
items = ingester.ingest_text("Astro-context is a modular RAG framework.")
print(items[0].content)

# Ingest a single file
items = ingester.ingest_file("docs/intro.md")

# Ingest an entire directory
items = ingester.ingest_directory("docs/", extensions=[".md", ".txt"])
```

## DocumentIngester

`DocumentIngester` orchestrates the full pipeline. It auto-detects parsers from
file extensions and delegates chunking to any `Chunker` implementation.

```python
from astro_context.ingestion import (
    DocumentIngester,
    SentenceChunker,
    MetadataEnricher,
)
from astro_context.models.context import SourceType

def add_category(text, idx, total, meta):
    meta["category"] = "documentation"
    return meta

ingester = DocumentIngester(
    chunker=SentenceChunker(chunk_size=256, overlap=1),
    enricher=MetadataEnricher(enrichers=[add_category]),
    source_type=SourceType.RETRIEVAL,
    priority=5,
)

items = ingester.ingest_text(
    "First sentence. Second sentence. Third sentence.",
    doc_id="manual-id-001",
)
for item in items:
    print(item.id, item.metadata.get("category"))
```

### Constructor Parameters

| Parameter     | Type                              | Default                       | Description                                |
|---------------|-----------------------------------|-------------------------------|--------------------------------------------|
| `chunker`     | `Chunker \| None`                 | `RecursiveCharacterChunker()` | Chunking strategy                          |
| `tokenizer`   | `Tokenizer \| None`               | default counter               | Token counter for size-aware splitting     |
| `parsers`     | `dict[str, DocumentParser] \| None` | built-in parsers            | Extension-to-parser overrides              |
| `enricher`    | `MetadataEnricher \| None`         | `None`                        | Chain of metadata enrichment functions     |
| `source_type` | `SourceType`                      | `SourceType.RETRIEVAL`        | Source type tag for produced items         |
| `priority`    | `int`                             | `5`                           | Priority value for produced items          |

### Methods

- **`ingest_text(text, doc_id=None, doc_metadata=None)`** -- Chunk raw text into `ContextItem` objects.
- **`ingest_file(path, doc_id=None)`** -- Parse and chunk a single file.
- **`ingest_directory(directory, glob_pattern="**/*", extensions=None)`** -- Recursively ingest all matching files.

---

## Chunkers

All chunkers implement the `Chunker` protocol and expose a
`chunk(text, metadata=None) -> list[str]` method.

### FixedSizeChunker

Splits text into fixed-size chunks measured by token count, with configurable
overlap.

```python
from astro_context.ingestion import FixedSizeChunker

chunker = FixedSizeChunker(chunk_size=128, overlap=20)
chunks = chunker.chunk("A very long document text...")
```

| Parameter    | Type             | Default           | Description                    |
|------------- |------------------|-------------------|--------------------------------|
| `chunk_size` | `int`            | `512`             | Maximum tokens per chunk       |
| `overlap`    | `int`            | `50`              | Overlapping tokens at boundary |
| `tokenizer`  | `Tokenizer \| None` | default counter | Token counter                  |

### RecursiveCharacterChunker

Splits text using a hierarchy of separators, falling back to finer splits when a
section exceeds the token budget. Separator hierarchy:
`"\n\n"` --> `"\n"` --> `". "` --> `" "`.

```python
from astro_context.ingestion import RecursiveCharacterChunker

chunker = RecursiveCharacterChunker(chunk_size=256, overlap=30)
chunks = chunker.chunk("Paragraph one.\n\nParagraph two.\n\nParagraph three.")
```

| Parameter    | Type                      | Default              | Description                         |
|------------- |---------------------------|----------------------|-------------------------------------|
| `chunk_size` | `int`                     | `512`                | Maximum tokens per chunk            |
| `overlap`    | `int`                     | `50`                 | Overlapping tokens at boundary      |
| `separators` | `tuple[str, ...] \| None` | `("\n\n","\n",". "," ")` | Custom separator hierarchy      |
| `tokenizer`  | `Tokenizer \| None`       | default counter      | Token counter                       |

### SentenceChunker

Groups sentences to fill chunks up to the token budget. Overlap is measured in
sentences rather than tokens.

```python
from astro_context.ingestion import SentenceChunker

chunker = SentenceChunker(chunk_size=256, overlap=1)
chunks = chunker.chunk("First sentence. Second sentence. Third sentence.")
```

| Parameter    | Type             | Default       | Description                            |
|------------- |------------------|---------------|----------------------------------------|
| `chunk_size` | `int`            | `512`         | Maximum tokens per chunk               |
| `overlap`    | `int`            | `1`           | Overlapping **sentences** at boundary  |
| `tokenizer`  | `Tokenizer \| None` | default counter | Token counter                       |

### SemanticChunker

Splits text at semantic boundaries using embedding similarity. Sentences whose
adjacent cosine similarity drops below a threshold are split into separate chunks.

```python
import math
from astro_context.ingestion import SemanticChunker

# Deterministic embed function for demonstration
def embed_fn(texts: list[str]) -> list[list[float]]:
    return [
        [math.sin(i + c) for c in range(8)]
        for i, t in enumerate(texts)
    ]

chunker = SemanticChunker(
    embed_fn=embed_fn,
    threshold=0.5,
    chunk_size=256,
    min_chunk_size=30,
)
chunks = chunker.chunk("Sentence one. Sentence two. Totally different topic here.")
```

| Parameter        | Type                                        | Default       | Description                            |
|------------------|---------------------------------------------|---------------|----------------------------------------|
| `embed_fn`       | `Callable[[list[str]], list[list[float]]]`  | **required**  | Embedding function                     |
| `tokenizer`      | `Tokenizer \| None`                         | default counter | Token counter                        |
| `threshold`      | `float`                                     | `0.5`         | Cosine similarity split threshold      |
| `chunk_size`     | `int`                                       | `512`         | Maximum tokens per chunk               |
| `min_chunk_size` | `int`                                       | `50`          | Minimum tokens; smaller chunks merge   |

### CodeChunker

Splits source code at function, class, and top-level definition boundaries using
language-specific regex patterns. Falls back to `RecursiveCharacterChunker` when
no boundaries are detected.

Supported languages: Python, JavaScript, TypeScript, Go, Rust.

```python
from astro_context.ingestion import CodeChunker

chunker = CodeChunker(language="python", chunk_size=256)
code = '''
def hello():
    print("hello")

def world():
    print("world")

class Greeter:
    pass
'''
chunks = chunker.chunk(code)
```

| Parameter    | Type             | Default       | Description                                  |
|------------- |------------------|---------------|----------------------------------------------|
| `language`   | `str \| None`    | `None`        | Language name; auto-detected from metadata   |
| `chunk_size` | `int`            | `512`         | Maximum tokens per chunk                     |
| `overlap`    | `int`            | `50`          | Overlap tokens for fallback chunker          |
| `tokenizer`  | `Tokenizer \| None` | default counter | Token counter                            |

### TableAwareChunker

Detects markdown and HTML tables, preserves them as atomic units, and delegates
prose to an inner chunker. Oversized tables are split row-by-row with the header
preserved.

```python
from astro_context.ingestion import TableAwareChunker

chunker = TableAwareChunker(chunk_size=256)
text = """
Some introductory text.

| Name  | Value |
|-------|-------|
| alpha | 1     |
| beta  | 2     |

More prose after the table.
"""
chunks = chunker.chunk(text)
```

| Parameter       | Type             | Default                        | Description                         |
|-----------------|------------------|--------------------------------|-------------------------------------|
| `inner_chunker` | `Any \| None`    | `RecursiveCharacterChunker()`  | Chunker for non-table text          |
| `chunk_size`    | `int`            | `512`                          | Maximum tokens per chunk            |
| `tokenizer`     | `Tokenizer \| None` | default counter             | Token counter                       |

### ParentChildChunker

Two-level hierarchical chunker that produces large parent chunks for context and
small child chunks for retrieval. Use `chunk_with_metadata()` to get child chunks
with `parent_id` and `parent_text` in their metadata.

```python
from astro_context.ingestion import ParentChildChunker

chunker = ParentChildChunker(
    parent_chunk_size=512,
    child_chunk_size=128,
    parent_overlap=50,
    child_overlap=10,
)

# Plain string chunks (Chunker protocol)
children = chunker.chunk("A long document to split hierarchically...")

# Chunks with metadata (includes parent_id, parent_text)
children_with_meta = chunker.chunk_with_metadata("A long document...")
for text, meta in children_with_meta:
    print(meta["parent_id"], len(text))
```

| Parameter           | Type             | Default       | Description                           |
|---------------------|------------------|---------------|---------------------------------------|
| `parent_chunk_size` | `int`            | `1024`        | Token size for parent chunks          |
| `child_chunk_size`  | `int`            | `256`         | Token size for child chunks           |
| `parent_overlap`    | `int`            | `100`         | Token overlap between parent chunks   |
| `child_overlap`     | `int`            | `25`          | Token overlap between child chunks    |
| `tokenizer`         | `Tokenizer \| None` | default counter | Token counter                     |

---

## Parsers

Parsers implement the `DocumentParser` protocol and return `(text, metadata)` tuples.

| Parser             | Extensions               | Dependencies   |
|--------------------|--------------------------|----------------|
| `PlainTextParser`  | `.txt`                   | none           |
| `MarkdownParser`   | `.md`, `.markdown`       | none           |
| `HTMLParser`       | `.html`, `.htm`          | none           |
| `PDFParser`        | `.pdf`                   | `pypdf`        |

!!! note
    `PDFParser` requires the optional `pdf` extra:
    `pip install astro-context[pdf]`

`DocumentIngester` auto-selects the parser by file extension. Override via the
`parsers` constructor argument:

```python
from astro_context.ingestion import DocumentIngester, PlainTextParser

ingester = DocumentIngester(
    parsers={".log": PlainTextParser()},
)
```

---

## Metadata

### Helper Functions

- **`generate_doc_id(content, source_path=None)`** -- deterministic 16-char hex ID from SHA-256.
- **`generate_chunk_id(doc_id, chunk_index)`** -- returns `"{doc_id}-chunk-{chunk_index}"`.
- **`extract_chunk_metadata(chunk_text, chunk_index, total_chunks, doc_id, doc_metadata=None)`** -- standard metadata dict with `parent_doc_id`, `chunk_index`, `total_chunks`, `word_count`, `char_count`.

### MetadataEnricher

Chain multiple enrichment functions that run in order during ingestion.

```python
from astro_context.ingestion import MetadataEnricher

def tag_language(text, idx, total, meta):
    meta["language"] = "en"
    return meta

def add_summary_flag(text, idx, total, meta):
    meta["needs_summary"] = len(text.split()) > 100
    return meta

enricher = MetadataEnricher(enrichers=[tag_language, add_summary_flag])
enricher.add(lambda text, idx, total, meta: {**meta, "version": "1.0"})
```

---

## ParentExpander

`ParentExpander` is a post-processor that expands retrieved child chunks back to
their parent text, deduplicating by `parent_id`.

```python
from astro_context.ingestion import ParentExpander
from astro_context.pipeline import postprocessor_step

expander = ParentExpander(keep_child=True)
step = postprocessor_step("expand-parents", expander)
```

!!! tip
    Combine `ParentChildChunker` + `ParentExpander` for a complete
    hierarchical retrieval workflow: index small children, retrieve them,
    then expand to full parent context before the LLM sees them.

---

## Full Pipeline Example

```python
import math
from astro_context.ingestion import (
    DocumentIngester,
    SemanticChunker,
    MetadataEnricher,
)

def embed_fn(texts: list[str]) -> list[list[float]]:
    return [[math.sin(i + c) for c in range(8)] for i, _ in enumerate(texts)]

def add_source(text, idx, total, meta):
    meta["source"] = "user-docs"
    return meta

ingester = DocumentIngester(
    chunker=SemanticChunker(embed_fn=embed_fn, threshold=0.5, chunk_size=256),
    enricher=MetadataEnricher(enrichers=[add_source]),
)

items = ingester.ingest_text(
    "Machine learning models learn patterns from data. "
    "They generalize to unseen examples. "
    "Transformers use self-attention mechanisms. "
    "RAG combines retrieval with generation.",
    doc_id="ml-intro",
)
for item in items:
    print(f"{item.id}: {item.content[:60]}...")
    print(f"  metadata: {item.metadata}")
```

!!! warning
    `SemanticChunker` calls `embed_fn` once per `chunk()` invocation for all
    sentences in the document. Make sure your embedding function can handle
    batch sizes equal to the sentence count.
