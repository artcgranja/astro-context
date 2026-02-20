"""Tests for memory protocol definitions and MemoryOperation enum.

Verifies that all memory protocols are runtime_checkable and that simple
concrete implementations pass isinstance checks (PEP 544 structural subtyping).
"""

from __future__ import annotations

from astro_context.models.memory import ConversationTurn, MemoryEntry
from astro_context.protocols.memory import (
    AsyncCompactionStrategy,
    AsyncMemoryExtractor,
    CompactionStrategy,
    EvictionPolicy,
    MemoryConsolidator,
    MemoryDecay,
    MemoryExtractor,
    MemoryOperation,
    QueryEnricher,
    RecencyScorer,
)

# ---------------------------------------------------------------------------
# MemoryOperation enum
# ---------------------------------------------------------------------------


class TestMemoryOperation:
    """MemoryOperation enum values."""

    def test_add_value(self) -> None:
        assert MemoryOperation.ADD == "add"

    def test_update_value(self) -> None:
        assert MemoryOperation.UPDATE == "update"

    def test_delete_value(self) -> None:
        assert MemoryOperation.DELETE == "delete"

    def test_none_value(self) -> None:
        assert MemoryOperation.NONE == "none"

    def test_all_values(self) -> None:
        expected = {"add", "update", "delete", "none"}
        assert {op.value for op in MemoryOperation} == expected

    def test_is_str_enum(self) -> None:
        assert isinstance(MemoryOperation.ADD, str)


# ---------------------------------------------------------------------------
# CompactionStrategy protocol
# ---------------------------------------------------------------------------


class TestCompactionStrategyProtocol:
    """CompactionStrategy is runtime_checkable and structurally satisfied."""

    def test_runtime_checkable(self) -> None:
        class MyCompactor:
            def compact(self, turns: list[ConversationTurn]) -> str:
                return "summary"

        assert isinstance(MyCompactor(), CompactionStrategy)

    def test_missing_method_fails(self) -> None:
        class NotCompactor:
            pass

        assert not isinstance(NotCompactor(), CompactionStrategy)

    def test_wrong_name_fails(self) -> None:
        class WrongName:
            def summarize(self, turns: list[ConversationTurn]) -> str:
                return "summary"

        assert not isinstance(WrongName(), CompactionStrategy)

    def test_implementation_works(self) -> None:
        class SimpleCompactor:
            def compact(self, turns: list[ConversationTurn]) -> str:
                return " | ".join(t.content for t in turns)

        compactor = SimpleCompactor()
        turns = [
            ConversationTurn(role="user", content="hello"),
            ConversationTurn(role="assistant", content="hi"),
        ]
        result = compactor.compact(turns)
        assert result == "hello | hi"


# ---------------------------------------------------------------------------
# AsyncCompactionStrategy protocol
# ---------------------------------------------------------------------------


class TestAsyncCompactionStrategyProtocol:
    """AsyncCompactionStrategy is runtime_checkable."""

    def test_runtime_checkable(self) -> None:
        class MyAsyncCompactor:
            async def compact(self, turns: list[ConversationTurn]) -> str:
                return "async summary"

        assert isinstance(MyAsyncCompactor(), AsyncCompactionStrategy)

    def test_missing_method_fails(self) -> None:
        class NotAsyncCompactor:
            pass

        assert not isinstance(NotAsyncCompactor(), AsyncCompactionStrategy)

    async def test_implementation_works(self) -> None:
        class MyAsyncCompactor:
            async def compact(self, turns: list[ConversationTurn]) -> str:
                return "async result"

        compactor = MyAsyncCompactor()
        result = await compactor.compact([])
        assert result == "async result"


# ---------------------------------------------------------------------------
# MemoryExtractor protocol
# ---------------------------------------------------------------------------


class TestMemoryExtractorProtocol:
    """MemoryExtractor is runtime_checkable."""

    def test_runtime_checkable(self) -> None:
        class MyExtractor:
            def extract(self, turns: list[ConversationTurn]) -> list[MemoryEntry]:
                return []

        assert isinstance(MyExtractor(), MemoryExtractor)

    def test_missing_method_fails(self) -> None:
        class NotExtractor:
            pass

        assert not isinstance(NotExtractor(), MemoryExtractor)

    def test_implementation_works(self) -> None:
        class SimpleExtractor:
            def extract(self, turns: list[ConversationTurn]) -> list[MemoryEntry]:
                return [MemoryEntry(content=t.content) for t in turns if t.role == "user"]

        extractor = SimpleExtractor()
        turns = [
            ConversationTurn(role="user", content="Remember this"),
            ConversationTurn(role="assistant", content="Ok"),
        ]
        entries = extractor.extract(turns)
        assert len(entries) == 1
        assert entries[0].content == "Remember this"


# ---------------------------------------------------------------------------
# AsyncMemoryExtractor protocol
# ---------------------------------------------------------------------------


class TestAsyncMemoryExtractorProtocol:
    """AsyncMemoryExtractor is runtime_checkable."""

    def test_runtime_checkable(self) -> None:
        class MyAsyncExtractor:
            async def extract(self, turns: list[ConversationTurn]) -> list[MemoryEntry]:
                return []

        assert isinstance(MyAsyncExtractor(), AsyncMemoryExtractor)

    def test_missing_method_fails(self) -> None:
        class NotAsyncExtractor:
            pass

        assert not isinstance(NotAsyncExtractor(), AsyncMemoryExtractor)

    async def test_implementation_works(self) -> None:
        class MyAsyncExtractor:
            async def extract(self, turns: list[ConversationTurn]) -> list[MemoryEntry]:
                return [MemoryEntry(content="async memory")]

        extractor = MyAsyncExtractor()
        result = await extractor.extract([])
        assert len(result) == 1
        assert result[0].content == "async memory"


# ---------------------------------------------------------------------------
# MemoryConsolidator protocol
# ---------------------------------------------------------------------------


class TestMemoryConsolidatorProtocol:
    """MemoryConsolidator is runtime_checkable."""

    def test_runtime_checkable(self) -> None:
        class MyConsolidator:
            def consolidate(
                self,
                new_entries: list[MemoryEntry],
                existing: list[MemoryEntry],
            ) -> list[tuple[MemoryOperation, MemoryEntry | None]]:
                return []

        assert isinstance(MyConsolidator(), MemoryConsolidator)

    def test_missing_method_fails(self) -> None:
        class NotConsolidator:
            pass

        assert not isinstance(NotConsolidator(), MemoryConsolidator)

    def test_implementation_returns_operations(self) -> None:
        class AddAllConsolidator:
            def consolidate(
                self,
                new_entries: list[MemoryEntry],
                existing: list[MemoryEntry],
            ) -> list[tuple[MemoryOperation, MemoryEntry | None]]:
                return [(MemoryOperation.ADD, entry) for entry in new_entries]

        consolidator = AddAllConsolidator()
        new = [MemoryEntry(content="new fact")]
        ops = consolidator.consolidate(new, [])
        assert len(ops) == 1
        assert ops[0][0] == MemoryOperation.ADD
        assert ops[0][1] is not None
        assert ops[0][1].content == "new fact"


# ---------------------------------------------------------------------------
# EvictionPolicy protocol
# ---------------------------------------------------------------------------


class TestEvictionPolicyProtocol:
    """EvictionPolicy is runtime_checkable."""

    def test_runtime_checkable(self) -> None:
        class MyPolicy:
            def select_for_eviction(
                self, turns: list[ConversationTurn], tokens_to_free: int
            ) -> list[int]:
                return []

        assert isinstance(MyPolicy(), EvictionPolicy)

    def test_missing_method_fails(self) -> None:
        class NotPolicy:
            pass

        assert not isinstance(NotPolicy(), EvictionPolicy)

    def test_implementation_selects_oldest(self) -> None:
        """A FIFO policy selects the oldest turns first."""

        class FifoPolicy:
            def select_for_eviction(
                self, turns: list[ConversationTurn], tokens_to_free: int
            ) -> list[int]:
                freed = 0
                indices: list[int] = []
                for i, turn in enumerate(turns):
                    if freed >= tokens_to_free:
                        break
                    indices.append(i)
                    freed += turn.token_count
                return indices

        policy = FifoPolicy()
        turns = [
            ConversationTurn(role="user", content="a", token_count=10),
            ConversationTurn(role="assistant", content="b", token_count=20),
            ConversationTurn(role="user", content="c", token_count=15),
        ]
        indices = policy.select_for_eviction(turns, tokens_to_free=25)
        assert indices == [0, 1]


# ---------------------------------------------------------------------------
# MemoryDecay protocol
# ---------------------------------------------------------------------------


class TestMemoryDecayProtocol:
    """MemoryDecay is runtime_checkable."""

    def test_runtime_checkable(self) -> None:
        class MyDecay:
            def compute_retention(self, entry: MemoryEntry) -> float:
                return 1.0

        assert isinstance(MyDecay(), MemoryDecay)

    def test_missing_method_fails(self) -> None:
        class NotDecay:
            pass

        assert not isinstance(NotDecay(), MemoryDecay)

    def test_implementation_decays_by_access(self) -> None:
        """Higher access_count -> higher retention."""

        class AccessDecay:
            def compute_retention(self, entry: MemoryEntry) -> float:
                return min(1.0, entry.access_count / 10.0)

        decay = AccessDecay()
        low = MemoryEntry(content="x", access_count=2)
        high = MemoryEntry(content="y", access_count=8)
        assert decay.compute_retention(low) < decay.compute_retention(high)


# ---------------------------------------------------------------------------
# QueryEnricher protocol
# ---------------------------------------------------------------------------


class TestQueryEnricherProtocol:
    """QueryEnricher is runtime_checkable."""

    def test_runtime_checkable(self) -> None:
        class MyEnricher:
            def enrich(self, query: str, memory_items: list[MemoryEntry]) -> str:
                return query

        assert isinstance(MyEnricher(), QueryEnricher)

    def test_missing_method_fails(self) -> None:
        class NotEnricher:
            pass

        assert not isinstance(NotEnricher(), QueryEnricher)

    def test_implementation_appends_context(self) -> None:
        class ContextEnricher:
            def enrich(self, query: str, memory_items: list[MemoryEntry]) -> str:
                if memory_items:
                    context = "; ".join(m.content for m in memory_items)
                    return f"{query} [context: {context}]"
                return query

        enricher = ContextEnricher()
        items = [MemoryEntry(content="user prefers Python")]
        result = enricher.enrich("best language?", items)
        assert "user prefers Python" in result
        assert "best language?" in result


# ---------------------------------------------------------------------------
# RecencyScorer protocol
# ---------------------------------------------------------------------------


class TestRecencyScorerProtocol:
    """RecencyScorer is runtime_checkable."""

    def test_runtime_checkable(self) -> None:
        class MyScorer:
            def score(self, index: int, total: int) -> float:
                return 0.5

        assert isinstance(MyScorer(), RecencyScorer)

    def test_missing_method_fails(self) -> None:
        class NotScorer:
            pass

        assert not isinstance(NotScorer(), RecencyScorer)

    def test_implementation_linear_scoring(self) -> None:
        """Linear scorer: 0.5 for oldest, 1.0 for newest."""

        class LinearScorer:
            def score(self, index: int, total: int) -> float:
                if total <= 1:
                    return 1.0
                return 0.5 + 0.5 * (index / (total - 1))

        scorer = LinearScorer()
        assert scorer.score(0, 5) == 0.5
        assert scorer.score(4, 5) == 1.0
        assert scorer.score(2, 5) == 0.75
        assert scorer.score(0, 1) == 1.0
