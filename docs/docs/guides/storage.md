# Storage Guide

astro-context uses **protocol-based** storage. Any object that matches the
required method signatures can serve as a store -- no inheritance needed.
The library ships with in-memory implementations for development and a
JSON-file-backed store for lightweight persistence.

---

## Storage Protocols

Four protocols define the storage contracts. All are `@runtime_checkable`,
so you can use `isinstance()` checks at runtime.

| Protocol | Purpose | Key Methods |
|---|---|---|
| `VectorStore` | Embedding storage and similarity search | `add_embedding()`, `search()`, `delete()` |
| `ContextStore` | Full `ContextItem` storage and lookup | `add()`, `get()`, `get_all()`, `delete()`, `clear()` |
| `DocumentStore` | Raw document text storage | `add_document()`, `get_document()`, `list_documents()`, `delete_document()` |
| `MemoryEntryStore` | `MemoryEntry` CRUD and search | `add()`, `search()`, `list_all()`, `delete()`, `clear()` |

Import protocols from:

```python
from astro_context.protocols.storage import (
    VectorStore,
    ContextStore,
    DocumentStore,
    MemoryEntryStore,
)
```

---

## In-Memory Implementations

### InMemoryVectorStore

Brute-force cosine similarity search. Suitable for development and testing
with small datasets.

```python
from astro_context.storage import InMemoryVectorStore

store = InMemoryVectorStore()
store.add_embedding("item-1", [0.1, 0.2, 0.3], metadata={"source": "docs"})
results = store.search([0.1, 0.2, 0.3], top_k=5)
# Returns: [("item-1", 1.0)]
```

!!! warning
    `InMemoryVectorStore` performs linear scan with O(N) complexity per query.
    A warning is logged when the store exceeds 5,000 embeddings. For
    production, use FAISS, Chroma, Qdrant, or another dedicated vector
    database.

### InMemoryContextStore

Dict-backed store for `ContextItem` objects. Thread-safe.

```python
from astro_context.storage import InMemoryContextStore
from astro_context.models.context import ContextItem, SourceType

store = InMemoryContextStore()
item = ContextItem(content="Hello world", source=SourceType.RETRIEVAL)
store.add(item)

retrieved = store.get(item.id)   # ContextItem or None
all_items = store.get_all()      # list[ContextItem]
store.delete(item.id)            # True
store.clear()                    # removes everything
```

### InMemoryDocumentStore

Dict-backed store for raw document text. Useful for storing original documents
before chunking and indexing.

```python
from astro_context.storage import InMemoryDocumentStore

store = InMemoryDocumentStore()
store.add_document("doc-1", "Full document text...", metadata={"author": "team"})

text = store.get_document("doc-1")     # str or None
doc_ids = store.list_documents()       # ["doc-1"]
store.delete_document("doc-1")         # True
```

### InMemoryEntryStore

In-memory store for `MemoryEntry` objects with filtering and search.

```python
from astro_context.storage import InMemoryEntryStore
from astro_context.models.memory import MemoryEntry

store = InMemoryEntryStore()
entry = MemoryEntry(content="User prefers dark mode")
store.add(entry)

results = store.search("dark mode", top_k=5)
all_entries = store.list_all()            # non-expired only
all_raw = store.list_all_unfiltered()     # includes expired
```

The store also supports filtered search:

```python
results = store.search_filtered(
    "dark mode",
    top_k=5,
    user_id="user-123",
    memory_type="preference",
    tags=["ui"],
)
```

---

## JsonFileMemoryStore

`JsonFileMemoryStore` persists `MemoryEntry` objects to a JSON file on disk.
It uses atomic writes (temp file + rename) to prevent corruption.

```python
from astro_context.storage import JsonFileMemoryStore
from astro_context.models.memory import MemoryEntry

store = JsonFileMemoryStore("memories.json", auto_save=True)

# Mutations are persisted immediately when auto_save=True
store.add(MemoryEntry(content="User prefers dark mode"))
results = store.search("dark mode")

# For batch operations, disable auto_save
batch_store = JsonFileMemoryStore("batch.json", auto_save=False)
for entry in entries:
    batch_store.add(entry)
batch_store.save()  # explicit flush
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `file_path` | `str \| Path` | -- | Path to the JSON file. Created on first save. |
| `auto_save` | `bool` | `True` | Persist to disk after every mutation. |

Additional methods beyond the `MemoryEntryStore` protocol:

| Method | Description |
|---|---|
| `save()` | Explicitly flush to disk (atomic write). |
| `load()` | Reload entries from the JSON file. |
| `export_user_entries(user_id)` | Export all entries for a user (GDPR data portability). |
| `delete_by_user(user_id)` | Delete all entries for a user. Returns count deleted. |

!!! note
    `JsonFileMemoryStore` is not suitable for concurrent multi-process
    access. For production use with multiple workers, implement the
    `MemoryEntryStore` protocol backed by a database.

---

## Implementing a Custom Store

To create a custom storage backend, implement the methods defined by the
protocol. No base class inheritance is needed -- Python structural subtyping
handles the rest.

### Example: Custom VectorStore

```python
from typing import Any


class RedisVectorStore:
    """Example VectorStore backed by Redis with vector search."""

    def __init__(self, redis_client, index_name: str = "embeddings"):
        self._client = redis_client
        self._index = index_name

    def add_embedding(
        self, item_id: str, embedding: list[float], metadata: dict[str, Any] | None = None
    ) -> None:
        self._client.hset(f"vec:{item_id}", mapping={
            "embedding": embedding,
            **(metadata or {}),
        })

    def search(
        self, query_embedding: list[float], top_k: int = 10
    ) -> list[tuple[str, float]]:
        # Use Redis vector similarity search
        results = self._client.ft(self._index).search(query_embedding, top_k)
        return [(r.id, r.score) for r in results]

    def delete(self, item_id: str) -> bool:
        return bool(self._client.delete(f"vec:{item_id}"))
```

You can verify protocol conformance at runtime:

```python
from astro_context.protocols.storage import VectorStore

store = RedisVectorStore(redis_client)
assert isinstance(store, VectorStore)  # True -- structural subtyping
```

!!! tip
    All in-memory stores are thread-safe via `threading.Lock`. If your custom
    implementation will be accessed from multiple threads, add your own locking.

---

## What's Next

- [Retrieval Guide](retrieval.md) -- using stores with retrievers
- [Advanced Retrieval](advanced-retrieval.md) -- memory retrieval with stores
- [Storage API Reference](../api/storage.md) -- full method signatures
