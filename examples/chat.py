#!/usr/bin/env python3
"""Interactive chat -- short-term memory, long-term facts, and agentic RAG.

Demonstrates the Agent class combining all three context engineering pillars:
  1. Short-term memory  -- SlidingWindowMemory keeps recent conversation turns
  2. Long-term memory   -- model saves facts via tools (agentic memory)
  3. Agentic RAG        -- model decides when to search documentation

Requirements:
    pip install astro-context[agents]
    pip install rich  # optional, for pretty output
    export ANTHROPIC_API_KEY=sk-ant-...
"""

from __future__ import annotations

import math
import os
import re
import sys
from collections import Counter
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.theme import Theme

    _theme = Theme({"info": "dim cyan", "warning": "yellow", "danger": "bold red"})
    console = Console(theme=_theme)
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

from astro_context import (
    Agent,
    ContextItem,
    DenseRetriever,
    MemoryManager,
    SourceType,
    memory_tools,
    rag_tools,
)
from astro_context.storage import InMemoryContextStore, InMemoryEntryStore, InMemoryVectorStore

MODEL = "claude-haiku-4-5-20251001"
REPO_ROOT = Path(__file__).resolve().parent.parent


# -- Display helpers --


def _print(text: str, *, style: str = "") -> None:
    if HAS_RICH:
        console.print(text, style=style)
    else:
        print(text)


def _print_markdown(text: str) -> None:
    if HAS_RICH:
        console.print(Markdown(text))
    else:
        print(text)


def _get_input(prompt: str) -> str:
    if HAS_RICH:
        return console.input(prompt)
    return input(prompt)


def _print_diagnostics(agent: Agent, memory: MemoryManager) -> None:
    """Display pipeline diagnostics after each turn."""
    result = agent.last_result
    if result is None:
        return
    d = result.diagnostics
    tokens_used = result.window.used_tokens
    tokens_max = result.window.max_tokens
    utilization = d.get("token_utilization", 0) * 100
    items_count = d.get("items_included", 0)
    build_ms = result.build_time_ms

    sources: dict[str, int] = {}
    for item in result.window.items:
        key = item.source.value
        sources[key] = sources.get(key, 0) + 1

    fact_count = len(memory.get_all_facts())
    src_str = " ".join(f"{k}={v}" for k, v in sorted(sources.items()))

    _print(
        f"  [{items_count} items | {tokens_used}/{tokens_max} tokens "
        f"({utilization:.0f}%) | {build_ms:.0f}ms | {src_str} | facts: {fact_count}]",
        style="info",
    )


# -- Whitespace tokenizer (avoids tiktoken network dependency) --


class _Tokenizer:
    def count_tokens(self, text: str) -> int:
        return len(text.split()) if text.strip() else 0

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        return " ".join(text.split()[:max_tokens])


# -- TF-IDF embedder (zero external dependencies) --


class TfidfEmbedder:
    """Lightweight TF-IDF embedder for demos. Replace with a real embedder in production."""

    __slots__ = ("_idf", "_vocab")

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}
        self._idf: list[float] = []

    def fit(self, docs: list[str]) -> None:
        n = len(docs)
        df: Counter[str] = Counter()
        for d in docs:
            df.update(set(re.findall(r"[a-z0-9_]+", d.lower())))
        self._vocab = {t: i for i, t in enumerate(sorted(df))}
        self._idf = [math.log(n / (1 + df[t])) for t in sorted(df)]

    def embed(self, text: str) -> list[float]:
        tf: Counter[str] = Counter(re.findall(r"[a-z0-9_]+", text.lower()))
        dim = len(self._vocab)
        if dim == 0:
            return [0.0]
        vec = [0.0] * dim
        for term, count in tf.items():
            if (idx := self._vocab.get(term)) is not None:
                vec[idx] = count * self._idf[idx]
        norm = math.sqrt(sum(x * x for x in vec))
        return [x / norm for x in vec] if norm > 0 else vec


# -- Load and chunk project documentation --


def load_docs(tok: _Tokenizer) -> list[ContextItem]:
    items: list[ContextItem] = []
    for fp in [REPO_ROOT / "README.md", REPO_ROOT / "CHANGELOG.md"]:
        if not fp.exists():
            continue
        for section in re.split(r"(?m)^## ", fp.read_text(encoding="utf-8")):
            section = section.strip()
            if not section:
                continue
            label = f"{fp.name} > {section.split(chr(10), 1)[0]}"
            items.append(ContextItem(
                content=section, source=SourceType.RETRIEVAL, score=0.0,
                priority=5, token_count=tok.count_tokens(section),
                metadata={"section": label},
            ))
    return items


# -- Setup --


def _build_agent() -> tuple[Agent, MemoryManager]:
    """Initialize agent with retriever, memory, and tools."""
    tok = _Tokenizer()
    doc_items = load_docs(tok)
    embedder = TfidfEmbedder()
    embedder.fit([i.content for i in doc_items])

    retriever = DenseRetriever(
        vector_store=InMemoryVectorStore(), context_store=InMemoryContextStore(),
        embed_fn=embedder.embed, tokenizer=tok,
    )
    retriever.index(doc_items)
    _print(f"  Indexed {len(doc_items)} doc chunks for RAG", style="info")

    memory = MemoryManager(
        conversation_tokens=8192, tokenizer=tok, persistent_store=InMemoryEntryStore(),
    )

    agent = (
        Agent(model=MODEL)
        .with_system_prompt(
            "You are a helpful assistant for the astro-context library. "
            "Save important user facts with save_fact. Search docs with search_docs when needed."
        )
        .with_memory(memory)
        .with_tools(memory_tools(memory))
        .with_tools(rag_tools(retriever, embed_fn=embedder.embed))
    )
    return agent, memory


def _handle_command(user_input: str, memory: MemoryManager) -> bool:
    """Handle slash commands. Returns True if input was a command."""
    if user_input == "/help":
        _print("  /facts           -- list saved facts", style="info")
        _print("  /remember <text> -- save a fact manually", style="info")
        _print("  /help            -- show this help", style="info")
        _print("  quit             -- exit", style="info")
        return True
    if user_input == "/facts":
        facts = memory.get_all_facts()
        for f in facts:
            _print(f"  {f.id[:8]}  {f.content}", style="info")
        if not facts:
            _print("  No facts saved yet.", style="info")
        return True
    if user_input.startswith("/remember "):
        fact = user_input[10:].strip()
        if fact:
            entry = memory.add_fact(fact, tags=["user"])
            _print(f'  Saved: "{fact}" (id: {entry.id[:8]})', style="info")
        return True
    return False


# -- Main --


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        _print(
            "Set ANTHROPIC_API_KEY environment variable to use this example.\n"
            "  export ANTHROPIC_API_KEY=sk-ant-...",
            style="danger",
        )
        sys.exit(1)

    agent, memory = _build_agent()
    _print("astro-context agent chat", style="bold")
    _print(f"  Model: {MODEL}", style="info")
    _print('  Type /help for commands, "quit" to exit.\n', style="info")

    while True:
        try:
            user_input = _get_input("[bold green]You:[/] " if HAS_RICH else "You: ")
        except (EOFError, KeyboardInterrupt):
            _print("\nGoodbye!", style="bold")
            break

        user_input = user_input.strip()
        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            _print("Goodbye!", style="bold")
            break
        if _handle_command(user_input, memory):
            print()
            continue

        _print("Assistant:", style="bold blue")
        response_text = ""
        for chunk in agent.chat(user_input):
            response_text += chunk
        _print_markdown(response_text)
        _print_diagnostics(agent, memory)
        print()


if __name__ == "__main__":
    main()
