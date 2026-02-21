#!/usr/bin/env python3
"""Graph Memory: entity-relationship tracking in astro-context.

Run with:  python examples/graph_memory.py

Demonstrates SimpleGraphMemory for building knowledge graphs, BFS traversal,
memory linking, and graph-based retrieval in a pipeline -- all without any
API keys or external services.
"""

from __future__ import annotations

from astro_context import (
    ContextPipeline,
    InMemoryEntryStore,
    MemoryEntry,
    MemoryManager,
    SimpleGraphMemory,
    graph_retrieval_step,
)

# ---------------------------------------------------------------------------
# Shared tokenizer (no external dependency)
# ---------------------------------------------------------------------------


class WhitespaceTokenizer:
    """Minimal tokenizer that counts whitespace-separated words."""

    def count_tokens(self, text: str) -> int:
        return len(text.split()) if text.strip() else 0

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        return " ".join(text.split()[:max_tokens])


tokenizer = WhitespaceTokenizer()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def header(number: int, title: str) -> None:
    """Print a boxed section header."""
    inner = f"  {number}. {title}"
    width = max(len(inner) + 2, 40)
    inner = inner.ljust(width - 2)
    print()
    print(f"\u2554{'═' * width}\u2557")
    print(f"\u2551{inner}\u2551")
    print(f"\u255a{'═' * width}\u255d")
    print()


def subheader(title: str) -> None:
    print(f"--- {title} ---")


# ===========================================================================
# 1. SimpleGraphMemory Basics
# ===========================================================================


def demo_graph_basics() -> SimpleGraphMemory:
    header(1, "SimpleGraphMemory Basics")

    print("Building a knowledge graph of Python web ecosystem.\n")

    graph = SimpleGraphMemory()

    # Add entities with metadata
    entities = [
        ("Python", {"type": "language", "version": "3.12"}),
        ("FastAPI", {"type": "framework", "category": "web"}),
        ("Pydantic", {"type": "library", "category": "validation"}),
        ("SQLAlchemy", {"type": "library", "category": "ORM"}),
        ("Uvicorn", {"type": "server", "category": "ASGI"}),
        ("Starlette", {"type": "framework", "category": "web"}),
        ("PostgreSQL", {"type": "database"}),
    ]

    for entity_id, metadata in entities:
        graph.add_entity(entity_id, metadata)
        print(f"  Added entity: {entity_id} {metadata}")

    print()

    # Add relationships
    relationships = [
        ("Python", "framework", "FastAPI"),
        ("Python", "framework", "Starlette"),
        ("Python", "library", "Pydantic"),
        ("Python", "library", "SQLAlchemy"),
        ("FastAPI", "uses", "Pydantic"),
        ("FastAPI", "built_on", "Starlette"),
        ("FastAPI", "served_by", "Uvicorn"),
        ("SQLAlchemy", "connects_to", "PostgreSQL"),
        ("Starlette", "served_by", "Uvicorn"),
    ]

    subheader("Adding relationships")
    for source, relation, target in relationships:
        graph.add_relationship(source, relation, target)
        print(f"  {source} --[{relation}]--> {target}")

    print(f"\nGraph: {graph!r}")
    print(f"Entities: {graph.entities}")
    print(f"Total relationships: {len(graph.relationships)}")

    return graph


# ===========================================================================
# 2. BFS Traversal
# ===========================================================================


def demo_bfs_traversal(graph: SimpleGraphMemory) -> None:
    header(2, "BFS Traversal")

    print("BFS (Breadth-First Search) finds entities connected to a starting")
    print("node. The max_depth parameter controls how far to traverse.\n")

    # Depth 1 vs Depth 2 from Python
    subheader("Starting from 'Python'")

    depth1 = graph.get_related_entities("Python", max_depth=1)
    depth2 = graph.get_related_entities("Python", max_depth=2)

    print(f"  max_depth=1: {depth1}")
    print(f"  max_depth=2: {depth2}")
    print()
    print(f"  At depth 1, we see Python's direct neighbors ({len(depth1)} entities).")
    print(f"  At depth 2, we also reach their neighbors ({len(depth2)} entities total).")

    print()

    # Depth 1 vs Depth 2 from FastAPI
    subheader("Starting from 'FastAPI'")

    depth1_fa = graph.get_related_entities("FastAPI", max_depth=1)
    depth2_fa = graph.get_related_entities("FastAPI", max_depth=2)

    print(f"  max_depth=1: {depth1_fa}")
    print(f"  max_depth=2: {depth2_fa}")
    print()
    print("  FastAPI is a hub node -- it connects to Python, Pydantic,")
    print("  Starlette, and Uvicorn directly.")

    print()

    # Starting from a leaf node
    subheader("Starting from 'PostgreSQL' (leaf node)")

    depth1_pg = graph.get_related_entities("PostgreSQL", max_depth=1)
    depth2_pg = graph.get_related_entities("PostgreSQL", max_depth=2)

    print(f"  max_depth=1: {depth1_pg}")
    print(f"  max_depth=2: {depth2_pg}")
    print()
    print("  PostgreSQL is only connected to SQLAlchemy, so depth 1 gives")
    print("  just SQLAlchemy, and depth 2 reaches Python as well.")


# ===========================================================================
# 3. Linking Memories to Entities
# ===========================================================================


def demo_memory_linking(graph: SimpleGraphMemory) -> dict[str, MemoryEntry]:
    header(3, "Linking Memories to Entities")

    print("Memory entries can be linked to entities in the graph.")
    print("This allows graph traversal to find relevant memories.\n")

    # Create some MemoryEntry objects
    memories = {
        "mem-python-intro": MemoryEntry(
            id="mem-python-intro",
            content=(
                "Python is a high-level, interpreted programming"
                " language known for its readability."
            ),
            tags=["python", "intro"],
        ),
        "mem-fastapi-perf": MemoryEntry(
            id="mem-fastapi-perf",
            content=(
                "FastAPI is one of the fastest Python web"
                " frameworks, comparable to Node.js and Go."
            ),
            tags=["fastapi", "performance"],
        ),
        "mem-pydantic-validation": MemoryEntry(
            id="mem-pydantic-validation",
            content="Pydantic provides data validation using Python type annotations.",
            tags=["pydantic", "validation"],
        ),
        "mem-sqlalchemy-orm": MemoryEntry(
            id="mem-sqlalchemy-orm",
            content="SQLAlchemy is the Python SQL toolkit and ORM that gives full SQL power.",
            tags=["sqlalchemy", "database"],
        ),
        "mem-postgres-scaling": MemoryEntry(
            id="mem-postgres-scaling",
            content="PostgreSQL supports horizontal scaling with read replicas and partitioning.",
            tags=["postgresql", "scaling"],
        ),
    }

    subheader("Created memory entries")
    for mid, entry in memories.items():
        print(f"  {mid}: {entry.content[:60]}...")

    print()

    # Link memories to entities
    links = [
        ("Python", "mem-python-intro"),
        ("FastAPI", "mem-fastapi-perf"),
        ("Pydantic", "mem-pydantic-validation"),
        ("SQLAlchemy", "mem-sqlalchemy-orm"),
        ("PostgreSQL", "mem-postgres-scaling"),
    ]

    subheader("Linking memories to entities")
    for entity_id, memory_id in links:
        graph.link_memory(entity_id, memory_id)
        print(f"  {entity_id} <-- linked -- {memory_id}")

    print()

    # Query linked memories
    subheader("get_memory_ids_for_entity()")
    for entity in ["Python", "FastAPI", "PostgreSQL"]:
        ids = graph.get_memory_ids_for_entity(entity)
        print(f"  {entity}: {ids}")

    print()

    # Get related memory IDs via graph traversal
    subheader("get_related_memory_ids() -- follows graph edges")

    print()
    print("  Starting from 'FastAPI' (max_depth=1):")
    related_1 = graph.get_related_memory_ids("FastAPI", max_depth=1)
    print(f"    Memory IDs: {related_1}")
    print("    This includes FastAPI's own memory plus memories from")
    print("    its direct neighbors (Python, Pydantic, Starlette, Uvicorn).")

    print()
    print("  Starting from 'PostgreSQL' (max_depth=2):")
    related_2 = graph.get_related_memory_ids("PostgreSQL", max_depth=2)
    print(f"    Memory IDs: {related_2}")
    print("    Starting from PostgreSQL, traversing through SQLAlchemy")
    print("    to reach Python, collecting memories along the way.")

    return memories


# ===========================================================================
# 4. Graph Retrieval in a Pipeline
# ===========================================================================


def demo_graph_pipeline(graph: SimpleGraphMemory, memories: dict[str, MemoryEntry]) -> None:
    header(4, "Graph Retrieval in a Pipeline")

    print("graph_retrieval_step() integrates graph memory with the")
    print("ContextPipeline, enriching context with entity-linked memories.\n")

    # Populate the InMemoryEntryStore with our memory entries
    store = InMemoryEntryStore()
    for entry in memories.values():
        store.add(entry)

    print(f"  Store contains {len(store.list_all())} memory entries.\n")

    # Entity extractor: a simple keyword-based extractor
    # In production, you would use NLP or an LLM for entity extraction
    entity_keywords = {
        "python": "Python",
        "fastapi": "FastAPI",
        "pydantic": "Pydantic",
        "sqlalchemy": "SQLAlchemy",
        "postgresql": "PostgreSQL",
        "postgres": "PostgreSQL",
        "database": "PostgreSQL",
        "web": "FastAPI",
        "validation": "Pydantic",
        "orm": "SQLAlchemy",
    }

    def extract_entities(query: str) -> list[str]:
        """Extract entity IDs from a query using keyword matching."""
        found: list[str] = []
        query_lower = query.lower()
        seen: set[str] = set()
        for keyword, entity in entity_keywords.items():
            if keyword in query_lower and entity not in seen:
                found.append(entity)
                seen.add(entity)
        return found

    # Set up conversation memory
    memory_manager = MemoryManager(conversation_tokens=200, tokenizer=tokenizer)
    memory_manager.add_user_message("I want to learn about FastAPI and databases.")
    memory_manager.add_assistant_message("FastAPI works great with SQLAlchemy for database access.")

    # Build the pipeline with graph retrieval
    step = graph_retrieval_step(
        graph=graph,
        store=store,
        entity_extractor=extract_entities,
        max_depth=2,
        max_items=5,
        name="tech_graph_lookup",
    )

    pipeline = (
        ContextPipeline(max_tokens=500, tokenizer=tokenizer)
        .with_memory(memory_manager)
        .add_system_prompt("You are a Python web development expert.")
        .add_step(step)
    )

    # Run the pipeline with different queries
    queries = [
        "Tell me about FastAPI performance",
        "How does PostgreSQL handle scaling?",
        "What is Python used for?",
    ]

    for query in queries:
        subheader(f"Query: '{query}'")

        # Show which entities the extractor finds
        extracted = extract_entities(query)
        print(f"  Extracted entities: {extracted}")

        result = pipeline.build(query)

        # Show the items in the context window
        print(f"  Context window: {result.window.used_tokens}/{result.window.max_tokens} tokens")
        print(f"  Total items: {len(result.window.items)}")
        print()

        for item in result.window.items:
            source = item.source.value
            source_detail = item.metadata.get("source", "")
            role = item.metadata.get("role", "")

            if source_detail == "graph_retrieval":
                label = f"[GRAPH] {item.metadata.get('memory_type', '')}"
            elif role:
                label = f"[{role}]"
            else:
                label = f"[{source}]"

            print(f"    prio={item.priority}  {label:<20} {item.content[:60]}...")

        print()


# ===========================================================================
# Main
# ===========================================================================


def main() -> None:
    print("=" * 60)
    print("  astro-context Graph Memory Showcase")
    print("  Entity-relationship tracking and graph retrieval")
    print("=" * 60)

    graph = demo_graph_basics()
    demo_bfs_traversal(graph)
    memories = demo_memory_linking(graph)
    demo_graph_pipeline(graph, memories)

    print("=" * 60)
    print("  All graph memory features demonstrated successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
