"""Integration tests for persistent conversation memory."""

import pytest

from anchor.memory.manager import MemoryManager
from anchor.storage.memory_store import InMemoryConversationStore
from tests.conftest import FakeTokenizer


class TestMemoryManagerConversationPersistence:
    def test_auto_persist_appends_turns(self):
        store = InMemoryConversationStore()
        mgr = MemoryManager(
            conversation_tokens=4096,
            tokenizer=FakeTokenizer(),
            conversation_store=store,
            session_id="sess-1",
            auto_persist=True,
        )
        mgr.add_user_message("hello")
        mgr.add_assistant_message("hi")
        turns = store.load_turns("sess-1")
        assert len(turns) == 2

    def test_requires_session_id_with_store(self):
        store = InMemoryConversationStore()
        with pytest.raises(ValueError, match="session_id"):
            MemoryManager(
                conversation_tokens=4096,
                tokenizer=FakeTokenizer(),
                conversation_store=store,
            )

    def test_save_and_load(self):
        store = InMemoryConversationStore()
        mgr1 = MemoryManager(
            conversation_tokens=4096,
            tokenizer=FakeTokenizer(),
            conversation_store=store,
            session_id="sess-1",
        )
        mgr1.add_user_message("hello")
        mgr1.add_assistant_message("hi there")
        mgr1.save()

        # Simulate restart: new MemoryManager loads from store
        mgr2 = MemoryManager(
            conversation_tokens=4096,
            tokenizer=FakeTokenizer(),
            conversation_store=store,
            session_id="sess-1",
        )
        mgr2.load()
        items = mgr2.get_context_items()
        contents = [item.content for item in items]
        assert any("hello" in c for c in contents)
