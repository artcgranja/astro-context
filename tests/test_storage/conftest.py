"""Shared fixtures for storage tests.

Provides a parametrized ``entry_store`` fixture that yields both
:class:`InMemoryEntryStore` and :class:`JsonFileMemoryStore` so that
behavioral tests covering search, filtering, and extra methods run
against both implementations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from astro_context.storage._base import BaseEntryStoreMixin
from astro_context.storage.json_file_store import JsonFileMemoryStore
from astro_context.storage.json_memory_store import InMemoryEntryStore


@pytest.fixture(params=["memory", "json_file"])
def entry_store(request: pytest.FixtureRequest, tmp_path: Path) -> BaseEntryStoreMixin:
    """Return an entry store instance -- parametrized across both implementations.

    Each test using this fixture runs once with :class:`InMemoryEntryStore` and
    once with :class:`JsonFileMemoryStore`.
    """
    kind: str = request.param
    if kind == "memory":
        return InMemoryEntryStore()
    store_path = tmp_path / "test_memories.json"
    return JsonFileMemoryStore(store_path)


@pytest.fixture
def store_factory(tmp_path: Path) -> Any:
    """Factory fixture that creates either store type on demand.

    Usage in tests that need two instances of the *same* concrete type
    (e.g. persistence round-trip tests).
    """

    def _make(kind: str = "memory") -> BaseEntryStoreMixin:
        if kind == "memory":
            return InMemoryEntryStore()
        store_path = tmp_path / "factory_memories.json"
        return JsonFileMemoryStore(store_path)

    return _make
