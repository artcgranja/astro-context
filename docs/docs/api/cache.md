# Cache API Reference

The cache module provides an in-memory cache backend for pipeline step
results. All classes are importable from `astro_context`:

```python
from astro_context import CacheBackend, InMemoryCacheBackend
```

---

## CacheBackend (Protocol)

Protocol for cache backends used by pipeline steps. Any class with matching
method signatures satisfies it via structural subtyping (PEP 544).

### Definition

```python
@runtime_checkable
class CacheBackend(Protocol):
    def get(self, key: str) -> Any | None: ...
    def set(self, key: str, value: Any, ttl: float | None = None) -> None: ...
    def invalidate(self, key: str) -> None: ...
    def clear(self) -> None: ...
```

**Methods**

| Method | Parameters | Returns | Description |
|---|---|---|---|
| `get` | `key: str` | `Any \| None` | Retrieve value, or `None` if not found / expired |
| `set` | `key: str, value: Any, ttl: float \| None = None` | `None` | Store a value with optional TTL |
| `invalidate` | `key: str` | `None` | Remove a specific key |
| `clear` | -- | `None` | Remove all entries |

---

## InMemoryCacheBackend

In-memory cache with optional TTL expiration and size-based eviction.
Implements the `CacheBackend` protocol.

### Constructor

```python
class InMemoryCacheBackend:
    def __init__(
        self,
        default_ttl: float | None = 300.0,
        max_size: int = 1000,
    ) -> None
```

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `default_ttl` | `float \| None` | `300.0` | Default TTL in seconds. `None` = no expiration |
| `max_size` | `int` | `1000` | Maximum entries before oldest is evicted |

### Methods

#### get

```python
def get(self, key: str) -> Any | None
```

Retrieve a cached value. Returns `None` if not found or expired.
Expired entries are cleaned up lazily on access.

#### set

```python
def set(self, key: str, value: Any, ttl: float | None = None) -> None
```

Store a value. If `ttl` is `None`, uses the default TTL. If the key
already exists, updates in place. If at max capacity, evicts the oldest
entry first.

#### invalidate

```python
def invalidate(self, key: str) -> None
```

Remove a specific key from the cache.

#### clear

```python
def clear(self) -> None
```

Remove all entries from the cache.

### Example

```python
from astro_context import InMemoryCacheBackend

cache = InMemoryCacheBackend(default_ttl=60.0, max_size=500)

cache.set("query:hello", {"results": [1, 2, 3]})
value = cache.get("query:hello")  # {"results": [1, 2, 3]}

cache.invalidate("query:hello")
cache.get("query:hello")  # None
```

---

## See Also

- [Cache Guide](../guides/cache.md) -- usage guide with examples
- [Protocols Reference](../api/protocols.md) -- all protocol definitions
