"""Simple graph memory for entity-relationship tracking.

Provides an in-memory directed graph that links entities to each other
and to memory entry IDs.  Useful for building entity-centric context
without requiring an external graph database.
"""

from __future__ import annotations

from collections import deque
from typing import Any


class SimpleGraphMemory:
    """In-memory graph for entity-relationship tracking.

    Stores entities as nodes and relationships as directed edges.
    Supports breadth-first traversal for finding related memories.

    This is a lightweight alternative to full graph databases like Neo4j.
    Users can provide their own entity extraction function to populate
    the graph from conversation turns or memory entries.

    Example::

        graph = SimpleGraphMemory()
        graph.add_entity("alice", {"type": "person"})
        graph.add_entity("project-x", {"type": "project"})
        graph.add_relationship("alice", "works_on", "project-x")
        graph.link_memory("alice", "mem-001")

        # Find all entities related to alice within 2 hops
        related = graph.get_related_entities("alice", max_depth=2)

        # Get memory IDs for alice and her related entities
        memory_ids = graph.get_related_memory_ids("alice", max_depth=2)
    """

    __slots__ = ("_edges", "_entity_to_memories", "_nodes")

    def __init__(self) -> None:
        self._nodes: dict[str, dict[str, Any]] = {}
        self._edges: list[tuple[str, str, str]] = []
        self._entity_to_memories: dict[str, list[str]] = {}

    def add_entity(self, entity_id: str, metadata: dict[str, Any] | None = None) -> None:
        """Add an entity node to the graph.

        If the entity already exists, its metadata is updated (merged).

        Parameters:
            entity_id: Unique identifier for the entity.
            metadata: Optional key-value metadata for the entity.
        """
        if entity_id in self._nodes:
            if metadata:
                self._nodes[entity_id].update(metadata)
        else:
            self._nodes[entity_id] = dict(metadata) if metadata else {}

    def add_relationship(self, source: str, relation: str, target: str) -> None:
        """Add a directed relationship (edge) between two entities.

        Automatically creates nodes for source and target if they do not
        already exist.

        Parameters:
            source: The source entity ID.
            relation: A label describing the relationship (e.g. "works_on").
            target: The target entity ID.
        """
        if source not in self._nodes:
            self._nodes[source] = {}
        if target not in self._nodes:
            self._nodes[target] = {}
        self._edges.append((source, relation, target))

    def link_memory(self, entity_id: str, memory_id: str) -> None:
        """Link a memory entry ID to an entity.

        Parameters:
            entity_id: The entity to link the memory to.
            memory_id: The ``MemoryEntry.id`` to associate.

        Raises:
            ValueError: If the entity does not exist in the graph.
        """
        if entity_id not in self._nodes:
            msg = f"Entity '{entity_id}' does not exist in the graph"
            raise ValueError(msg)
        self._entity_to_memories.setdefault(entity_id, []).append(memory_id)

    def get_related_entities(self, entity_id: str, max_depth: int = 2) -> list[str]:
        """Find entities related to *entity_id* via BFS traversal.

        Traverses edges in both directions (outgoing and incoming) up to
        *max_depth* hops.  The starting entity is **not** included in the
        result.

        Parameters:
            entity_id: The starting entity.
            max_depth: Maximum number of hops to traverse.

        Returns:
            A list of related entity IDs (deduplicated, order is BFS).
        """
        if entity_id not in self._nodes:
            return []

        visited: set[str] = {entity_id}
        queue: deque[tuple[str, int]] = deque([(entity_id, 0)])
        result: list[str] = []

        # Build adjacency index for fast lookup (both directions)
        adjacency: dict[str, set[str]] = {}
        for src, _rel, tgt in self._edges:
            adjacency.setdefault(src, set()).add(tgt)
            adjacency.setdefault(tgt, set()).add(src)

        while queue:
            current, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for neighbor in adjacency.get(current, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    result.append(neighbor)
                    queue.append((neighbor, depth + 1))

        return result

    def get_memory_ids_for_entity(self, entity_id: str) -> list[str]:
        """Get all memory IDs linked to a specific entity.

        Parameters:
            entity_id: The entity to look up.

        Returns:
            A list of memory IDs.  Returns an empty list if the entity
            does not exist or has no linked memories.
        """
        return list(self._entity_to_memories.get(entity_id, []))

    def get_related_memory_ids(self, entity_id: str, max_depth: int = 2) -> list[str]:
        """Get memory IDs for the entity and all related entities.

        Combines ``get_memory_ids_for_entity`` with ``get_related_entities``
        to collect memories from the entire neighborhood.

        Parameters:
            entity_id: The starting entity.
            max_depth: Maximum traversal depth.

        Returns:
            A deduplicated list of memory IDs from the entity and its
            neighbors.
        """
        all_entities = [entity_id, *self.get_related_entities(entity_id, max_depth)]
        seen: set[str] = set()
        result: list[str] = []
        for eid in all_entities:
            for mid in self._entity_to_memories.get(eid, []):
                if mid not in seen:
                    seen.add(mid)
                    result.append(mid)
        return result

    def remove_entity(self, entity_id: str) -> None:
        """Remove an entity and all its edges from the graph.

        Also removes any memory linkages for the entity.

        Parameters:
            entity_id: The entity to remove.
        """
        self._nodes.pop(entity_id, None)
        self._edges = [
            (s, r, t) for s, r, t in self._edges if s != entity_id and t != entity_id
        ]
        self._entity_to_memories.pop(entity_id, None)

    def clear(self) -> None:
        """Remove all entities, relationships, and memory linkages."""
        self._nodes.clear()
        self._edges.clear()
        self._entity_to_memories.clear()

    @property
    def entities(self) -> list[str]:
        """List all entity IDs in the graph."""
        return list(self._nodes)

    @property
    def relationships(self) -> list[tuple[str, str, str]]:
        """List all relationships as ``(source, relation, target)`` tuples."""
        return list(self._edges)

    def get_entity_metadata(self, entity_id: str) -> dict[str, Any]:
        """Get metadata for an entity.

        Parameters:
            entity_id: The entity to look up.

        Returns:
            A copy of the entity's metadata dict.

        Raises:
            KeyError: If the entity does not exist.
        """
        if entity_id not in self._nodes:
            msg = f"Entity '{entity_id}' does not exist in the graph"
            raise KeyError(msg)
        return dict(self._nodes[entity_id])

    def __repr__(self) -> str:
        return (
            f"SimpleGraphMemory(entities={len(self._nodes)}, "
            f"relationships={len(self._edges)})"
        )

    def __len__(self) -> int:
        return len(self._nodes)
