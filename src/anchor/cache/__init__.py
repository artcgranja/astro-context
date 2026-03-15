"""Caching module for anchor pipeline steps."""

from .backend import InMemoryCacheBackend
from .sqlite_backend import SqliteCacheBackend

__all__ = [
    "InMemoryCacheBackend",
    "SqliteCacheBackend",
]

# Redis backend is optional (requires redis-py)
try:
    from .redis_backend import RedisCacheBackend
    __all__.append("RedisCacheBackend")
except ImportError:
    pass
