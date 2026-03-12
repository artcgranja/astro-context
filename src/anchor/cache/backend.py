"""In-memory cache backend implementation."""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class InMemoryCacheBackend:
    """In-memory cache with optional TTL expiration.

    Thread-safe via a simple dict + timestamp tracking.
    Expired entries are cleaned up lazily on access.

    Implements the ``CacheBackend`` protocol.

    Parameters:
        default_ttl: Default TTL in seconds for entries without explicit TTL.
            None means no expiration. Default 300 (5 minutes).
        max_size: Maximum number of entries. When exceeded, oldest entries
            are evicted. Default 1000.
    """

    __slots__ = ("_data", "_default_ttl", "_max_size", "_timestamps")

    def __init__(
        self,
        default_ttl: float | None = 300.0,
        max_size: int = 1000,
    ) -> None:
        self._default_ttl = default_ttl
        self._max_size = max_size
        self._data: dict[str, Any] = {}
        self._timestamps: dict[str, tuple[float, float | None]] = {}

    def get(self, key: str) -> Any | None:
        """Retrieve a cached value, or None if not found / expired.

        Parameters:
            key: The cache key.

        Returns:
            The cached value, or None.
        """
        if key not in self._data:
            return None

        _created_at, expires_at = self._timestamps[key]
        if expires_at is not None and time.monotonic() >= expires_at:
            # Lazy cleanup of expired entry
            self._remove(key)
            return None

        return self._data[key]

    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        """Store a value in the cache.

        Parameters:
            key: The cache key.
            value: The value to cache.
            ttl: Time-to-live in seconds. None means use the default TTL.
        """
        now = time.monotonic()
        effective_ttl = ttl if ttl is not None else self._default_ttl
        expires_at = (now + effective_ttl) if effective_ttl is not None else None

        # If key already exists, update in place (no eviction needed)
        if key in self._data:
            self._data[key] = value
            self._timestamps[key] = (now, expires_at)
            return

        # Evict oldest entries if at max capacity
        while len(self._data) >= self._max_size:
            self._evict_oldest()

        self._data[key] = value
        self._timestamps[key] = (now, expires_at)

    def invalidate(self, key: str) -> None:
        """Remove a specific key from the cache.

        Parameters:
            key: The cache key to remove.
        """
        self._remove(key)

    def clear(self) -> None:
        """Remove all entries from the cache."""
        self._data.clear()
        self._timestamps.clear()

    def _remove(self, key: str) -> None:
        """Remove a key from both data and timestamps dicts."""
        self._data.pop(key, None)
        self._timestamps.pop(key, None)

    def _evict_oldest(self) -> None:
        """Evict the oldest entry by creation time."""
        if not self._timestamps:
            return
        oldest_key = min(self._timestamps, key=lambda k: self._timestamps[k][0])
        self._remove(oldest_key)

    def __repr__(self) -> str:
        return (
            f"InMemoryCacheBackend(default_ttl={self._default_ttl}, "
            f"max_size={self._max_size}, entries={len(self._data)})"
        )
