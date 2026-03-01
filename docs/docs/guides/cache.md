# Cache Guide

astro-context provides a caching layer for pipeline step results. Caching
avoids redundant retrieval, reranking, or transformation work when the same
query is processed multiple times.

## Overview

The cache system has two parts:

1. **`CacheBackend` protocol** -- the interface any cache backend must satisfy
2. **`InMemoryCacheBackend`** -- the built-in, zero-dependency implementation

Cached pipeline steps check the cache before executing and store results
after execution, keyed by a combination of query text and step name.

## InMemoryCacheBackend

The built-in backend stores entries in a Python dict with optional TTL
expiration and LRU-style size limits:

```python
from astro_context import InMemoryCacheBackend

cache = InMemoryCacheBackend(
    default_ttl=300.0,  # 5 minutes
    max_size=1000,
)
```

### Constructor

```python
InMemoryCacheBackend(
    default_ttl: float | None = 300.0,
    max_size: int = 1000,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `default_ttl` | `float \| None` | `300.0` | Default time-to-live in seconds. `None` means no expiration |
| `max_size` | `int` | `1000` | Maximum number of entries before eviction |

### Basic Usage

```python
from astro_context import InMemoryCacheBackend

cache = InMemoryCacheBackend(default_ttl=60.0)

# Store a value
cache.set("query:hello", ["result1", "result2"])

# Retrieve it
value = cache.get("query:hello")  # ["result1", "result2"]

# After TTL expires
# value = cache.get("query:hello")  # None

# Override TTL for a specific entry
cache.set("important:key", "data", ttl=3600.0)  # 1 hour

# Manual invalidation
cache.invalidate("query:hello")

# Clear everything
cache.clear()
```

### TTL and Expiration

Expired entries are cleaned up lazily on access -- when you call `get()` on
an expired key, it is removed and `None` is returned. There is no background
cleanup thread.

```python
# No expiration
cache = InMemoryCacheBackend(default_ttl=None)
cache.set("permanent", "value")  # Never expires

# Per-entry TTL override
cache = InMemoryCacheBackend(default_ttl=300.0)
cache.set("short", "value", ttl=10.0)   # Expires in 10 seconds
cache.set("long", "value", ttl=3600.0)  # Expires in 1 hour
cache.set("default", "value")            # Uses default 300s TTL
```

### Size-Based Eviction

When the cache reaches `max_size`, the oldest entry (by creation time) is
evicted to make room for new entries:

```python
cache = InMemoryCacheBackend(max_size=3)

cache.set("a", 1)
cache.set("b", 2)
cache.set("c", 3)
cache.set("d", 4)  # Evicts "a" (oldest)

cache.get("a")  # None
cache.get("d")  # 4
```

!!! note
    Updating an existing key does not trigger eviction -- it updates the value
    and timestamp in place.

## CacheBackend Protocol

To implement a custom backend (e.g., Redis, disk-based), satisfy the
`CacheBackend` protocol:

```python
from astro_context import CacheBackend

class RedisCacheBackend:
    """Example Redis-based cache backend."""

    def __init__(self, redis_client, default_ttl=300):
        self._redis = redis_client
        self._ttl = default_ttl

    def get(self, key: str):
        import json
        raw = self._redis.get(key)
        return json.loads(raw) if raw else None

    def set(self, key: str, value, ttl: float | None = None):
        import json
        effective_ttl = ttl if ttl is not None else self._ttl
        if effective_ttl:
            self._redis.setex(key, int(effective_ttl), json.dumps(value))
        else:
            self._redis.set(key, json.dumps(value))

    def invalidate(self, key: str):
        self._redis.delete(key)

    def clear(self):
        self._redis.flushdb()
```

The protocol requires four methods:

| Method | Signature | Description |
|---|---|---|
| `get` | `(key: str) -> Any \| None` | Retrieve value or `None` |
| `set` | `(key: str, value: Any, ttl: float \| None) -> None` | Store a value |
| `invalidate` | `(key: str) -> None` | Remove a specific key |
| `clear` | `() -> None` | Remove all entries |

!!! tip
    `CacheBackend` is a PEP 544 `Protocol` -- no inheritance required.
    Any class with matching method signatures satisfies it via structural
    subtyping.

## See Also

- [Pipeline Guide](../guides/pipeline.md) -- using cached pipeline steps
- [Cache API Reference](../api/cache.md) -- complete signatures
- [Protocols Reference](../api/protocols.md) -- `CacheBackend` protocol
