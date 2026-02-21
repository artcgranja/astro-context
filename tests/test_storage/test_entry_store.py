"""Tests specific to astro_context.storage.json_memory_store.InMemoryEntryStore.

Shared behavioral tests (search, filtering, CRUD) have been moved to
``test_entry_store_shared.py`` which runs against both store implementations.
This file retains only InMemoryEntryStore-specific tests (repr, etc.).
"""

from __future__ import annotations

import pytest

from astro_context.storage.json_memory_store import InMemoryEntryStore
from tests.conftest import make_memory_entry as _make_entry


@pytest.fixture
def store() -> InMemoryEntryStore:
    """Return a fresh InMemoryEntryStore."""
    return InMemoryEntryStore()


# ---------------------------------------------------------------------------
# repr
# ---------------------------------------------------------------------------


class TestInMemoryEntryStoreRepr:
    """Test __repr__ for debugging."""

    def test_repr_includes_entry_count(self, store: InMemoryEntryStore) -> None:
        store.add(_make_entry(entry_id="r1"))
        assert "1" in repr(store)
        assert "InMemoryEntryStore" in repr(store)
