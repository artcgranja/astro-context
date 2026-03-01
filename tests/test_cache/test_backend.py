"""Tests for InMemoryCacheBackend."""

from __future__ import annotations

import time

from astro_context.cache.backend import InMemoryCacheBackend
from astro_context.protocols.cache import CacheBackend


class TestInMemoryCacheBackend:
    """Tests for the in-memory cache backend."""

    def test_protocol_compliance(self) -> None:
        assert isinstance(InMemoryCacheBackend(), CacheBackend)

    def test_get_miss(self) -> None:
        cache = InMemoryCacheBackend()
        assert cache.get("nonexistent") is None

    def test_set_and_get(self) -> None:
        cache = InMemoryCacheBackend()
        cache.set("key1", [1, 2, 3])
        assert cache.get("key1") == [1, 2, 3]

    def test_ttl_expiration(self) -> None:
        cache = InMemoryCacheBackend(default_ttl=0.01)  # 10ms
        cache.set("key1", "value")
        time.sleep(0.02)
        assert cache.get("key1") is None

    def test_explicit_ttl_overrides_default(self) -> None:
        cache = InMemoryCacheBackend(default_ttl=0.01)
        cache.set("key1", "value", ttl=10)  # 10 seconds
        time.sleep(0.02)
        assert cache.get("key1") == "value"  # still valid

    def test_no_ttl_never_expires(self) -> None:
        cache = InMemoryCacheBackend(default_ttl=None)
        cache.set("key1", "value")
        assert cache.get("key1") == "value"

    def test_invalidate(self) -> None:
        cache = InMemoryCacheBackend()
        cache.set("key1", "value")
        cache.invalidate("key1")
        assert cache.get("key1") is None

    def test_clear(self) -> None:
        cache = InMemoryCacheBackend()
        cache.set("k1", "v1")
        cache.set("k2", "v2")
        cache.clear()
        assert cache.get("k1") is None
        assert cache.get("k2") is None

    def test_max_size_eviction(self) -> None:
        cache = InMemoryCacheBackend(max_size=3, default_ttl=None)
        cache.set("k1", "v1")
        cache.set("k2", "v2")
        cache.set("k3", "v3")
        cache.set("k4", "v4")  # should evict k1
        assert cache.get("k1") is None
        assert cache.get("k4") == "v4"

    def test_invalidate_nonexistent_key(self) -> None:
        cache = InMemoryCacheBackend()
        cache.invalidate("nonexistent")  # should not raise

    def test_repr(self) -> None:
        cache = InMemoryCacheBackend(default_ttl=60, max_size=100)
        assert "InMemoryCacheBackend" in repr(cache)

    def test_overwrite_existing_key(self) -> None:
        cache = InMemoryCacheBackend(max_size=2, default_ttl=None)
        cache.set("k1", "v1")
        cache.set("k2", "v2")
        cache.set("k1", "v1_updated")  # overwrite, should not evict
        assert cache.get("k1") == "v1_updated"
        assert cache.get("k2") == "v2"

    def test_max_size_preserves_newest(self) -> None:
        cache = InMemoryCacheBackend(max_size=2, default_ttl=None)
        cache.set("k1", "v1")
        cache.set("k2", "v2")
        cache.set("k3", "v3")  # evicts k1
        assert cache.get("k2") == "v2"
        assert cache.get("k3") == "v3"
