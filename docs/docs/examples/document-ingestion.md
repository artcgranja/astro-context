# Document Ingestion

Ingest raw text and files, chunk them with different strategies, enrich
metadata, and index the results into a retriever.

---

## Overview

This example demonstrates:

- Using `DocumentIngester` with different chunkers
- Comparing `FixedSizeChunker`, `RecursiveCharacterChunker`, and `SentenceChunker`
- Enriching chunk metadata with `MetadataEnricher`
- Indexing chunks into a `DenseRetriever`
- Querying the indexed documents

## Full Example

```python
import math

from astro_context import (
    ContextPipeline,
    DenseRetriever,
    DocumentIngester,
    FixedSizeChunker,
    InMemoryContextStore,
    InMemoryVectorStore,
    MetadataEnricher,
    QueryBundle,
    RecursiveCharacterChunker,
    SentenceChunker,
    retriever_step,
)

# ---------------------------------------------------------------
# 1. Deterministic embedding function
# ---------------------------------------------------------------
def embed_fn(text: str) -> list[float]:
    seed = sum(ord(c) for c in text) % 10000
    raw = [math.sin(seed * 1000 + i) for i in range(64)]
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw] if norm else raw


# ---------------------------------------------------------------
# 2. Sample document (a technical article)
# ---------------------------------------------------------------
article = """
# Context Engineering for AI

Context engineering is the emerging discipline of designing systems that
provide language models with the right information at the right time.
Unlike prompt engineering, which focuses on crafting individual prompts,
context engineering builds entire pipelines that assemble context from
multiple sources.

## Why Context Matters

Language models are only as good as the context they receive. A model
with perfect reasoning but poor context will produce poor results.
Context engineering addresses this by systematically managing what
information flows into the model's context window.

## Key Components

The core components of a context engineering system include:

1. Retrieval: Finding relevant documents from a knowledge base using
   dense embeddings, sparse keyword matching, or hybrid approaches.

2. Memory: Managing conversation history with token-aware sliding
   windows that automatically evict old turns when the budget fills.

3. Token Budgets: Allocating finite context window space across
   competing sources like system prompts, memory, and retrieval.

4. Formatting: Converting assembled context into the specific format
   required by each LLM provider (Anthropic, OpenAI, etc.).

## Best Practices

Start simple and add complexity only when needed. A basic pipeline
with a single retriever and sliding-window memory handles most use
cases. Add reranking, hybrid retrieval, and query transformation
only after measuring baseline performance with evaluation metrics.
"""

# ---------------------------------------------------------------
# 3. Compare chunking strategies
# ---------------------------------------------------------------
print("=== Chunking Strategy Comparison ===\n")

chunkers = {
    "FixedSize(100 tokens)": FixedSizeChunker(chunk_size=100, overlap=20),
    "Recursive(100 tokens)": RecursiveCharacterChunker(chunk_size=100, overlap=20),
    "Sentence(100 tokens)":  SentenceChunker(chunk_size=100, overlap=1),
}

for name, chunker in chunkers.items():
    ingester = DocumentIngester(chunker=chunker)
    items = ingester.ingest_text(article, doc_id="context-engineering")
    print(f"{name}:")
    print(f"  Chunks produced: {len(items)}")
    for i, item in enumerate(items):
        tokens = item.token_count
        preview = item.content[:60].replace("\n", " ")
        print(f"    [{i}] ({tokens} tokens) {preview}...")
    print()

# ---------------------------------------------------------------
# 4. Use MetadataEnricher for custom metadata
# ---------------------------------------------------------------
print("=== Metadata Enrichment ===\n")

class TopicTagger(MetadataEnricher):
    """Tag chunks with topics based on keyword presence."""

    TOPICS = {
        "retrieval": ["retrieval", "embedding", "search", "keyword"],
        "memory": ["memory", "conversation", "sliding", "evict"],
        "formatting": ["format", "anthropic", "openai", "provider"],
        "budgets": ["token", "budget", "allocat", "window"],
    }

    def enrich(
        self,
        chunk_text: str,
        chunk_index: int,
        total_chunks: int,
        metadata: dict,
    ) -> dict:
        text_lower = chunk_text.lower()
        tags = []
        for topic, keywords in self.TOPICS.items():
            if any(kw in text_lower for kw in keywords):
                tags.append(topic)
        metadata["topics"] = tags
        metadata["position"] = f"{chunk_index + 1}/{total_chunks}"
        return metadata

ingester = DocumentIngester(
    chunker=RecursiveCharacterChunker(chunk_size=120, overlap=20),
    enricher=TopicTagger(),
)
items = ingester.ingest_text(article, doc_id="context-engineering")

for item in items:
    topics = item.metadata.get("topics", [])
    pos = item.metadata.get("position", "?")
    preview = item.content[:50].replace("\n", " ")
    print(f"  [{pos}] topics={topics} | {preview}...")

# ---------------------------------------------------------------
# 5. Index chunks into a DenseRetriever
# ---------------------------------------------------------------
print("\n=== Indexing into DenseRetriever ===\n")

dense = DenseRetriever(
    vector_store=InMemoryVectorStore(),
    context_store=InMemoryContextStore(),
    embed_fn=embed_fn,
)
count = dense.index(items)
print(f"  Indexed {count} chunks")

# ---------------------------------------------------------------
# 6. Query the indexed documents
# ---------------------------------------------------------------
print("\n=== Querying Indexed Documents ===\n")

pipeline = (
    ContextPipeline(max_tokens=2048)
    .add_step(retriever_step("search", dense, top_k=3))
    .add_system_prompt("Answer based on the provided context.")
)

queries = [
    "What is context engineering?",
    "How does memory management work?",
    "What are the best practices?",
]

for q in queries:
    query = QueryBundle(query_str=q, embedding=embed_fn(q))
    result = pipeline.build(query)
    retrieved = [
        item for item in result.window.items
        if item.source.value == "retrieval"
    ]
    print(f"  Q: {q}")
    print(f"    Results: {len(retrieved)} items")
    for r in retrieved:
        score = f"{r.score:.3f}"
        topics = r.metadata.get("topics", [])
        print(f"      [{score}] topics={topics} "
              f"| {r.content[:50].replace(chr(10), ' ')}...")
    print()

# ---------------------------------------------------------------
# 7. Inspect chunk metadata
# ---------------------------------------------------------------
print("=== Chunk Metadata Details ===\n")
sample = items[0]
print(f"  id:          {sample.id}")
print(f"  source:      {sample.source.value}")
print(f"  priority:    {sample.priority}")
print(f"  token_count: {sample.token_count}")
print(f"  metadata:")
for key, value in sample.metadata.items():
    print(f"    {key}: {value}")
```

## Key Concepts

### Built-in Chunkers

| Chunker | Strategy | Best For |
|---------|----------|----------|
| `FixedSizeChunker` | Fixed token count with overlap | Uniform chunk sizes |
| `RecursiveCharacterChunker` | Hierarchy: paragraph, line, sentence, word | General-purpose text |
| `SentenceChunker` | Sentence-boundary aware | Prose and articles |
| `SemanticChunker` | Embedding similarity boundaries | Topically diverse text |

### Built-in Parsers

`DocumentIngester` auto-detects parsers by file extension:

| Parser | Extensions | Dependencies |
|--------|------------|--------------|
| `PlainTextParser` | `.txt` | None |
| `MarkdownParser` | `.md`, `.markdown` | None |
| `HTMLParser` | `.html`, `.htm` | None |
| `PDFParser` | `.pdf` | `pypdf` (optional) |

### Ingesting Files

```python
from pathlib import Path

ingester = DocumentIngester(
    chunker=RecursiveCharacterChunker(chunk_size=512),
)

# Single file
items = ingester.ingest_file(Path("docs/guide.md"))

# Entire directory (recursive)
items = ingester.ingest_directory(
    Path("docs/"),
    extensions=[".md", ".txt"],
)
```

!!! tip "Chunk Size Guidelines"
    - Small chunks (100-200 tokens): better precision, more chunks to search
    - Large chunks (500-1000 tokens): more context per result, fewer chunks
    - Start with 256-512 tokens and adjust based on evaluation metrics

!!! warning "SemanticChunker Signature"
    `SemanticChunker` requires a batch embedding function with signature
    `(list[str]) -> list[list[float]]`, not the single-string `embed_fn`
    used by `DenseRetriever`.

## Next Steps

- [RAG Pipeline](rag-pipeline.md) -- build a retrieval pipeline around your documents
- [Custom Retriever](custom-retriever.md) -- implement a custom storage backend
- [Evaluation Workflow](evaluation-workflow.md) -- measure chunking quality
