"""Tests for SimpleGraphMemory entity-relationship tracking."""

from __future__ import annotations

import pytest

from astro_context.memory.graph_memory import SimpleGraphMemory

# ===========================================================================
# TestAddEntity
# ===========================================================================


class TestAddEntity:
    """add_entity() adds nodes to the graph."""

    def test_add_entity_basic(self) -> None:
        graph = SimpleGraphMemory()
        graph.add_entity("alice")
        assert "alice" in graph.entities

    def test_add_entity_with_metadata(self) -> None:
        graph = SimpleGraphMemory()
        graph.add_entity("alice", {"type": "person", "role": "engineer"})
        meta = graph.get_entity_metadata("alice")
        assert meta == {"type": "person", "role": "engineer"}

    def test_add_entity_updates_metadata_on_duplicate(self) -> None:
        graph = SimpleGraphMemory()
        graph.add_entity("alice", {"type": "person"})
        graph.add_entity("alice", {"role": "engineer"})
        meta = graph.get_entity_metadata("alice")
        assert meta == {"type": "person", "role": "engineer"}

    def test_add_entity_without_metadata_creates_empty_dict(self) -> None:
        graph = SimpleGraphMemory()
        graph.add_entity("alice")
        meta = graph.get_entity_metadata("alice")
        assert meta == {}

    def test_add_multiple_entities(self) -> None:
        graph = SimpleGraphMemory()
        graph.add_entity("alice")
        graph.add_entity("bob")
        graph.add_entity("project-x")
        assert len(graph.entities) == 3

    def test_len_reflects_entity_count(self) -> None:
        graph = SimpleGraphMemory()
        assert len(graph) == 0
        graph.add_entity("a")
        graph.add_entity("b")
        assert len(graph) == 2


# ===========================================================================
# TestAddRelationship
# ===========================================================================


class TestAddRelationship:
    """add_relationship() creates directed edges."""

    def test_add_relationship_basic(self) -> None:
        graph = SimpleGraphMemory()
        graph.add_entity("alice")
        graph.add_entity("project-x")
        graph.add_relationship("alice", "works_on", "project-x")
        assert ("alice", "works_on", "project-x") in graph.relationships

    def test_auto_creates_source_node(self) -> None:
        graph = SimpleGraphMemory()
        graph.add_entity("project-x")
        graph.add_relationship("alice", "works_on", "project-x")
        assert "alice" in graph.entities

    def test_auto_creates_target_node(self) -> None:
        graph = SimpleGraphMemory()
        graph.add_entity("alice")
        graph.add_relationship("alice", "works_on", "project-x")
        assert "project-x" in graph.entities

    def test_auto_creates_both_nodes(self) -> None:
        graph = SimpleGraphMemory()
        graph.add_relationship("alice", "works_on", "project-x")
        assert "alice" in graph.entities
        assert "project-x" in graph.entities

    def test_multiple_relationships(self) -> None:
        graph = SimpleGraphMemory()
        graph.add_relationship("alice", "works_on", "project-x")
        graph.add_relationship("alice", "manages", "bob")
        assert len(graph.relationships) == 2


# ===========================================================================
# TestLinkMemory
# ===========================================================================


class TestLinkMemory:
    """link_memory() associates memory IDs with entities."""

    def test_link_memory_basic(self) -> None:
        graph = SimpleGraphMemory()
        graph.add_entity("alice")
        graph.link_memory("alice", "mem-001")
        assert "mem-001" in graph.get_memory_ids_for_entity("alice")

    def test_link_multiple_memories(self) -> None:
        graph = SimpleGraphMemory()
        graph.add_entity("alice")
        graph.link_memory("alice", "mem-001")
        graph.link_memory("alice", "mem-002")
        ids = graph.get_memory_ids_for_entity("alice")
        assert ids == ["mem-001", "mem-002"]

    def test_link_memory_raises_for_unknown_entity(self) -> None:
        graph = SimpleGraphMemory()
        with pytest.raises(ValueError, match="does not exist"):
            graph.link_memory("unknown", "mem-001")


# ===========================================================================
# TestGetRelatedEntities -- BFS traversal
# ===========================================================================


class TestGetRelatedEntities:
    """get_related_entities() performs BFS traversal."""

    def test_depth_1_direct_neighbors(self) -> None:
        graph = SimpleGraphMemory()
        graph.add_relationship("alice", "knows", "bob")
        graph.add_relationship("alice", "knows", "carol")
        graph.add_relationship("dave", "knows", "eve")

        related = graph.get_related_entities("alice", max_depth=1)
        assert set(related) == {"bob", "carol"}

    def test_depth_2_includes_hop_2(self) -> None:
        graph = SimpleGraphMemory()
        graph.add_relationship("alice", "knows", "bob")
        graph.add_relationship("bob", "knows", "carol")

        related = graph.get_related_entities("alice", max_depth=2)
        assert set(related) == {"bob", "carol"}

    def test_depth_1_excludes_hop_2(self) -> None:
        graph = SimpleGraphMemory()
        graph.add_relationship("alice", "knows", "bob")
        graph.add_relationship("bob", "knows", "carol")

        related = graph.get_related_entities("alice", max_depth=1)
        assert set(related) == {"bob"}
        assert "carol" not in related

    def test_source_entity_not_in_result(self) -> None:
        graph = SimpleGraphMemory()
        graph.add_relationship("alice", "knows", "bob")

        related = graph.get_related_entities("alice", max_depth=2)
        assert "alice" not in related

    def test_traverses_incoming_edges(self) -> None:
        """BFS goes in both directions (undirected traversal)."""
        graph = SimpleGraphMemory()
        graph.add_relationship("bob", "reports_to", "alice")

        related = graph.get_related_entities("alice", max_depth=1)
        assert "bob" in related

    def test_unknown_entity_returns_empty(self) -> None:
        graph = SimpleGraphMemory()
        graph.add_entity("alice")
        related = graph.get_related_entities("unknown")
        assert related == []

    def test_entity_with_no_edges_returns_empty(self) -> None:
        graph = SimpleGraphMemory()
        graph.add_entity("alice")
        related = graph.get_related_entities("alice")
        assert related == []

    def test_bfs_with_cycle_does_not_infinite_loop(self) -> None:
        graph = SimpleGraphMemory()
        graph.add_relationship("alice", "knows", "bob")
        graph.add_relationship("bob", "knows", "carol")
        graph.add_relationship("carol", "knows", "alice")

        related = graph.get_related_entities("alice", max_depth=10)
        assert set(related) == {"bob", "carol"}

    def test_bfs_diamond_pattern(self) -> None:
        """Diamond: A -> B, A -> C, B -> D, C -> D."""
        graph = SimpleGraphMemory()
        graph.add_relationship("a", "r", "b")
        graph.add_relationship("a", "r", "c")
        graph.add_relationship("b", "r", "d")
        graph.add_relationship("c", "r", "d")

        related = graph.get_related_entities("a", max_depth=2)
        assert set(related) == {"b", "c", "d"}

    def test_bfs_depth_0_returns_empty(self) -> None:
        """max_depth=0 means no traversal at all."""
        graph = SimpleGraphMemory()
        graph.add_relationship("alice", "knows", "bob")
        related = graph.get_related_entities("alice", max_depth=0)
        assert related == []


# ===========================================================================
# TestGetMemoryIdsForEntity
# ===========================================================================


class TestGetMemoryIdsForEntity:
    """get_memory_ids_for_entity() returns linked memory IDs."""

    def test_returns_linked_memories(self) -> None:
        graph = SimpleGraphMemory()
        graph.add_entity("alice")
        graph.link_memory("alice", "mem-1")
        graph.link_memory("alice", "mem-2")
        assert graph.get_memory_ids_for_entity("alice") == ["mem-1", "mem-2"]

    def test_unknown_entity_returns_empty(self) -> None:
        graph = SimpleGraphMemory()
        assert graph.get_memory_ids_for_entity("unknown") == []

    def test_entity_with_no_memories_returns_empty(self) -> None:
        graph = SimpleGraphMemory()
        graph.add_entity("alice")
        assert graph.get_memory_ids_for_entity("alice") == []

    def test_returns_copy_not_reference(self) -> None:
        graph = SimpleGraphMemory()
        graph.add_entity("alice")
        graph.link_memory("alice", "mem-1")
        ids = graph.get_memory_ids_for_entity("alice")
        ids.append("extra")
        # Original should be unchanged
        assert graph.get_memory_ids_for_entity("alice") == ["mem-1"]


# ===========================================================================
# TestGetRelatedMemoryIds
# ===========================================================================


class TestGetRelatedMemoryIds:
    """get_related_memory_ids() returns memories from BFS-reachable entities."""

    def test_includes_own_and_neighbor_memories(self) -> None:
        graph = SimpleGraphMemory()
        graph.add_entity("alice")
        graph.add_entity("bob")
        graph.add_relationship("alice", "knows", "bob")
        graph.link_memory("alice", "mem-a1")
        graph.link_memory("bob", "mem-b1")

        ids = graph.get_related_memory_ids("alice", max_depth=1)
        assert set(ids) == {"mem-a1", "mem-b1"}

    def test_depth_2_includes_transitive_memories(self) -> None:
        graph = SimpleGraphMemory()
        graph.add_relationship("alice", "knows", "bob")
        graph.add_relationship("bob", "knows", "carol")
        graph.add_entity("alice")
        graph.link_memory("alice", "mem-a")
        graph.link_memory("bob", "mem-b")
        graph.link_memory("carol", "mem-c")

        ids = graph.get_related_memory_ids("alice", max_depth=2)
        assert set(ids) == {"mem-a", "mem-b", "mem-c"}

    def test_depth_1_excludes_hop_2_memories(self) -> None:
        graph = SimpleGraphMemory()
        graph.add_relationship("alice", "knows", "bob")
        graph.add_relationship("bob", "knows", "carol")
        graph.add_entity("alice")
        graph.link_memory("carol", "mem-c")

        ids = graph.get_related_memory_ids("alice", max_depth=1)
        assert "mem-c" not in ids

    def test_deduplicated_results(self) -> None:
        """If multiple paths lead to the same entity, memories are not duplicated."""
        graph = SimpleGraphMemory()
        graph.add_relationship("a", "r", "b")
        graph.add_relationship("a", "r", "c")
        graph.add_relationship("b", "r", "d")
        graph.add_relationship("c", "r", "d")
        graph.link_memory("d", "mem-d")

        ids = graph.get_related_memory_ids("a", max_depth=2)
        assert ids.count("mem-d") == 1

    def test_entity_with_no_memories_in_neighborhood(self) -> None:
        graph = SimpleGraphMemory()
        graph.add_relationship("alice", "knows", "bob")
        ids = graph.get_related_memory_ids("alice", max_depth=2)
        assert ids == []


# ===========================================================================
# TestRemoveEntity
# ===========================================================================


class TestRemoveEntity:
    """remove_entity() cleans up edges and memory links."""

    def test_removes_node(self) -> None:
        graph = SimpleGraphMemory()
        graph.add_entity("alice")
        graph.remove_entity("alice")
        assert "alice" not in graph.entities

    def test_removes_outgoing_edges(self) -> None:
        graph = SimpleGraphMemory()
        graph.add_relationship("alice", "knows", "bob")
        graph.remove_entity("alice")
        assert len(graph.relationships) == 0

    def test_removes_incoming_edges(self) -> None:
        graph = SimpleGraphMemory()
        graph.add_relationship("bob", "reports_to", "alice")
        graph.remove_entity("alice")
        assert len(graph.relationships) == 0

    def test_removes_memory_links(self) -> None:
        graph = SimpleGraphMemory()
        graph.add_entity("alice")
        graph.link_memory("alice", "mem-1")
        graph.remove_entity("alice")
        assert graph.get_memory_ids_for_entity("alice") == []

    def test_preserves_unrelated_edges(self) -> None:
        graph = SimpleGraphMemory()
        graph.add_relationship("alice", "knows", "bob")
        graph.add_relationship("carol", "knows", "dave")
        graph.remove_entity("alice")
        assert ("carol", "knows", "dave") in graph.relationships

    def test_remove_nonexistent_entity_is_noop(self) -> None:
        graph = SimpleGraphMemory()
        graph.add_entity("alice")
        graph.remove_entity("nonexistent")
        assert "alice" in graph.entities

    def test_len_decreases_after_removal(self) -> None:
        graph = SimpleGraphMemory()
        graph.add_entity("alice")
        graph.add_entity("bob")
        assert len(graph) == 2
        graph.remove_entity("alice")
        assert len(graph) == 1


# ===========================================================================
# TestClear
# ===========================================================================


class TestClear:
    """clear() resets everything."""

    def test_clear_removes_all_entities(self) -> None:
        graph = SimpleGraphMemory()
        graph.add_entity("alice")
        graph.add_entity("bob")
        graph.clear()
        assert graph.entities == []

    def test_clear_removes_all_relationships(self) -> None:
        graph = SimpleGraphMemory()
        graph.add_relationship("alice", "knows", "bob")
        graph.clear()
        assert graph.relationships == []

    def test_clear_removes_all_memory_links(self) -> None:
        graph = SimpleGraphMemory()
        graph.add_entity("alice")
        graph.link_memory("alice", "mem-1")
        graph.clear()
        assert graph.get_memory_ids_for_entity("alice") == []

    def test_clear_makes_len_zero(self) -> None:
        graph = SimpleGraphMemory()
        graph.add_entity("a")
        graph.add_entity("b")
        graph.clear()
        assert len(graph) == 0


# ===========================================================================
# TestEntitiesProperty
# ===========================================================================


class TestEntitiesProperty:
    """entities property returns all entity IDs."""

    def test_empty_graph(self) -> None:
        graph = SimpleGraphMemory()
        assert graph.entities == []

    def test_returns_all_entity_ids(self) -> None:
        graph = SimpleGraphMemory()
        graph.add_entity("alice")
        graph.add_entity("bob")
        assert set(graph.entities) == {"alice", "bob"}

    def test_includes_auto_created_entities(self) -> None:
        graph = SimpleGraphMemory()
        graph.add_relationship("alice", "knows", "bob")
        assert set(graph.entities) == {"alice", "bob"}


# ===========================================================================
# TestRelationshipsProperty
# ===========================================================================


class TestRelationshipsProperty:
    """relationships property returns all triples."""

    def test_empty_graph(self) -> None:
        graph = SimpleGraphMemory()
        assert graph.relationships == []

    def test_returns_all_triples(self) -> None:
        graph = SimpleGraphMemory()
        graph.add_relationship("alice", "knows", "bob")
        graph.add_relationship("carol", "works_with", "dave")
        rels = graph.relationships
        assert len(rels) == 2
        assert ("alice", "knows", "bob") in rels
        assert ("carol", "works_with", "dave") in rels

    def test_returns_copy_not_reference(self) -> None:
        graph = SimpleGraphMemory()
        graph.add_relationship("alice", "knows", "bob")
        rels = graph.relationships
        rels.append(("x", "y", "z"))
        assert len(graph.relationships) == 1


# ===========================================================================
# TestGetEntityMetadata
# ===========================================================================


class TestGetEntityMetadata:
    """get_entity_metadata() returns metadata for an entity."""

    def test_returns_metadata(self) -> None:
        graph = SimpleGraphMemory()
        graph.add_entity("alice", {"role": "engineer"})
        assert graph.get_entity_metadata("alice") == {"role": "engineer"}

    def test_raises_key_error_for_unknown(self) -> None:
        graph = SimpleGraphMemory()
        with pytest.raises(KeyError, match="does not exist"):
            graph.get_entity_metadata("unknown")

    def test_returns_copy_not_reference(self) -> None:
        graph = SimpleGraphMemory()
        graph.add_entity("alice", {"role": "engineer"})
        meta = graph.get_entity_metadata("alice")
        meta["extra"] = "value"
        assert "extra" not in graph.get_entity_metadata("alice")


# ===========================================================================
# TestRepr
# ===========================================================================


class TestRepr:
    """__repr__ returns useful info."""

    def test_repr_empty(self) -> None:
        graph = SimpleGraphMemory()
        assert repr(graph) == "SimpleGraphMemory(entities=0, relationships=0)"

    def test_repr_with_data(self) -> None:
        graph = SimpleGraphMemory()
        graph.add_relationship("alice", "knows", "bob")
        assert repr(graph) == "SimpleGraphMemory(entities=2, relationships=1)"
