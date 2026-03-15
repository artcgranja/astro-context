"""Simple graph memory for entity-relationship tracking.

Provides a directed graph that links entities to each other
and to memory entry IDs. Supports an optional GraphStore backend
for persistence; defaults to in-memory dicts.
"""

from __future__ import annotations

from collections import deque
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from anchor.protocols.storage import GraphStore


class SimpleGraphMemory:
    """Graph for entity-relationship tracking with optional persistent backend.

    When *store* is provided, all operations delegate to it.
    When *store* is None, uses in-memory dicts (original behavior).
    """

    __slots__ = (
        "_adjacency",
        "_adjacency_dirty",
        "_edges",
        "_entity_to_memories",
        "_nodes",
        "_store",
    )

    def __init__(self, store: GraphStore | None = None) -> None:
        self._store = store
        # In-memory fallback (only used when store is None)
        self._nodes: dict[str, dict[str, Any]] = {}
        self._edges: list[tuple[str, str, str]] = []
        self._entity_to_memories: dict[str, list[str]] = {}
        self._adjacency: dict[str, set[str]] = {}
        self._adjacency_dirty: bool = True

    def add_entity(self, entity_id: str, metadata: dict[str, Any] | None = None) -> None:
        """Add an entity node. Alias for add_node (backwards compat)."""
        if self._store is not None:
            self._store.add_node(entity_id, metadata)
        else:
            if entity_id in self._nodes:
                if metadata:
                    self._nodes[entity_id].update(metadata)
            else:
                self._nodes[entity_id] = dict(metadata) if metadata else {}

    def add_relationship(self, source: str, relation: str, target: str) -> None:
        """Add a directed edge. Alias for add_edge (backwards compat)."""
        if self._store is not None:
            self._store.add_edge(source, relation, target)
        else:
            if source not in self._nodes:
                self._nodes[source] = {}
            if target not in self._nodes:
                self._nodes[target] = {}
            self._edges.append((source, relation, target))
            self._adjacency_dirty = True

    def get_related_entities(
        self, entity_id: str, max_depth: int = 2,
        relation_filter: str | list[str] | None = None,
    ) -> list[str]:
        """BFS traversal. Alias for get_neighbors (backwards compat)."""
        if self._store is not None:
            return self._store.get_neighbors(entity_id, max_depth=max_depth, relation_filter=relation_filter)
        # In-memory BFS
        if entity_id not in self._nodes:
            return []
        self._rebuild_adjacency(relation_filter)
        visited: set[str] = {entity_id}
        queue: deque[tuple[str, int]] = deque([(entity_id, 0)])
        result: list[str] = []
        while queue:
            current, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for neighbor in self._adjacency.get(current, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    result.append(neighbor)
                    queue.append((neighbor, depth + 1))
        return result

    def link_memory(self, entity_id: str, memory_id: str) -> None:
        if self._store is not None:
            self._store.link_memory(entity_id, memory_id)
        else:
            if entity_id not in self._nodes:
                msg = f"Entity '{entity_id}' does not exist in the graph"
                raise KeyError(msg)
            self._entity_to_memories.setdefault(entity_id, []).append(memory_id)

    def get_memory_ids_for_entity(self, entity_id: str) -> list[str]:
        if self._store is not None:
            return self._store.get_memory_ids(entity_id, max_depth=0)
        return list(self._entity_to_memories.get(entity_id, []))

    def get_related_memory_ids(self, entity_id: str, max_depth: int = 2) -> list[str]:
        if self._store is not None:
            return self._store.get_memory_ids(entity_id, max_depth=max_depth)
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
        if self._store is not None:
            self._store.remove_node(entity_id)
        else:
            self._nodes.pop(entity_id, None)
            self._edges = [
                (s, r, t) for s, r, t in self._edges if s != entity_id and t != entity_id
            ]
            self._entity_to_memories.pop(entity_id, None)
            self._adjacency_dirty = True

    def remove_edge(self, source: str, relation: str, target: str) -> bool:
        """Remove a specific edge. Returns True if it existed."""
        if self._store is not None:
            return self._store.remove_edge(source, relation, target)
        for i, (s, r, t) in enumerate(self._edges):
            if s == source and r == relation and t == target:
                self._edges.pop(i)
                self._adjacency_dirty = True
                return True
        return False

    def clear(self) -> None:
        if self._store is not None:
            self._store.clear()
        else:
            self._nodes.clear()
            self._edges.clear()
            self._entity_to_memories.clear()
            self._adjacency_dirty = True

    @property
    def entities(self) -> list[str]:
        if self._store is not None:
            return self._store.list_nodes()
        return list(self._nodes)

    @property
    def relationships(self) -> list[tuple[str, str, str]]:
        if self._store is not None:
            return self._store.list_edges()
        return list(self._edges)

    def get_entity_metadata(self, entity_id: str) -> dict[str, Any]:
        if self._store is not None:
            result = self._store.get_node_metadata(entity_id)
            if result is None:
                msg = f"Entity '{entity_id}' does not exist in the graph"
                raise KeyError(msg)
            return result
        if entity_id not in self._nodes:
            msg = f"Entity '{entity_id}' does not exist in the graph"
            raise KeyError(msg)
        return dict(self._nodes[entity_id])

    def _rebuild_adjacency(self, relation_filter: str | list[str] | None = None) -> None:
        if relation_filter is not None:
            allowed = {relation_filter} if isinstance(relation_filter, str) else set(relation_filter)
        else:
            allowed = None
        self._adjacency = {}
        for src, rel, tgt in self._edges:
            if allowed is not None and rel not in allowed:
                continue
            self._adjacency.setdefault(src, set()).add(tgt)
            self._adjacency.setdefault(tgt, set()).add(src)
        self._adjacency_dirty = False

    def __repr__(self) -> str:
        if self._store is not None:
            return f"SimpleGraphMemory(store={self._store!r})"
        return (
            f"SimpleGraphMemory(entities={len(self._nodes)}, "
            f"relationships={len(self._edges)})"
        )

    def __len__(self) -> int:
        if self._store is not None:
            return len(self._store.list_nodes())
        return len(self._nodes)
