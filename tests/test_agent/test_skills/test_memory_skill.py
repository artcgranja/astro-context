"""Tests for the memory_skill factory."""

from __future__ import annotations

from astro_context.agent.skills.memory import memory_skill
from astro_context.memory.manager import MemoryManager
from astro_context.storage.json_memory_store import InMemoryEntryStore


class _Tok:
    def count_tokens(self, text: str) -> int:
        return len(text.split()) if text.strip() else 0

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        return " ".join(text.split()[:max_tokens])


def _make_memory(*, with_store: bool = False) -> MemoryManager:
    store = InMemoryEntryStore() if with_store else None
    return MemoryManager(
        conversation_tokens=500, tokenizer=_Tok(), persistent_store=store,
    )


class TestMemorySkill:
    def test_creates_valid_skill(self) -> None:
        skill = memory_skill(_make_memory())
        assert skill.name == "memory"
        assert skill.activation == "always"
        assert len(skill.tools) == 4

    def test_tool_names(self) -> None:
        skill = memory_skill(_make_memory())
        names = [t.name for t in skill.tools]
        assert "save_fact" in names
        assert "search_facts" in names
        assert "update_fact" in names
        assert "delete_fact" in names

    def test_has_instructions(self) -> None:
        skill = memory_skill(_make_memory())
        assert skill.instructions != ""
        assert "search_facts" in skill.instructions

    def test_has_tags(self) -> None:
        skill = memory_skill(_make_memory())
        assert "memory" in skill.tags
        assert "core" in skill.tags

    def test_tools_are_functional(self) -> None:
        mem = _make_memory(with_store=True)
        skill = memory_skill(mem)
        save_tool = next(t for t in skill.tools if t.name == "save_fact")
        result = save_tool.fn(fact="User likes Python")
        assert "Saved" in result

    def test_description_is_set(self) -> None:
        skill = memory_skill(_make_memory())
        assert skill.description != ""
