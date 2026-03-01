"""Protocol definition for cache backends."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class CacheBackend(Protocol):
    """Backend for caching pipeline step results.

    Implementations store and retrieve cached context items keyed
    by a string cache key (typically derived from query + step name).
    """

    def get(self, key: str) -> Any | None:
        """Retrieve a cached value, or None if not found / expired.

        Parameters:
            key: The cache key.

        Returns:
            The cached value, or None.
        """
        ...

    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        """Store a value in the cache.

        Parameters:
            key: The cache key.
            value: The value to cache.
            ttl: Time-to-live in seconds. None means no expiration.
        """
        ...

    def invalidate(self, key: str) -> None:
        """Remove a specific key from the cache.

        Parameters:
            key: The cache key to remove.
        """
        ...

    def clear(self) -> None:
        """Remove all entries from the cache."""
        ...
