#!/usr/bin/env python3
"""Interactive chat loop with Claude using astro-context.

Demonstrates the full pipeline: system prompts, sliding window memory,
Anthropic-formatted output, and token budget management.

Requirements:
    pip install astro-context[anthropic]
    pip install rich  # optional, for pretty output
    export ANTHROPIC_API_KEY=sk-ant-...
"""

from __future__ import annotations

import os
import sys

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

try:
    import anthropic
except ImportError:
    print(
        "anthropic is not installed. Install with: pip install astro-context[anthropic]",
        file=sys.stderr,
    )
    sys.exit(1)

try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.theme import Theme

    _theme = Theme({"info": "dim cyan", "warning": "yellow", "danger": "bold red"})
    console = Console(theme=_theme)
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

from astro_context.formatters.anthropic import AnthropicFormatter
from astro_context.memory.manager import MemoryManager
from astro_context.pipeline.pipeline import ContextPipeline

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL = "claude-haiku-4-5-20251001"
MAX_PIPELINE_TOKENS = 8192
CONVERSATION_TOKENS = 6144
SYSTEM_PROMPT = (
    "You are a helpful, concise assistant. "
    "Answer questions clearly and keep responses focused."
)


def _print(text: str, *, style: str = "") -> None:
    """Print helper that uses rich when available."""
    if HAS_RICH:
        console.print(text, style=style)
    else:
        print(text)


def _print_markdown(text: str) -> None:
    """Render markdown when rich is available, plain text otherwise."""
    if HAS_RICH:
        console.print(Markdown(text))
    else:
        print(text)


def _get_input(prompt: str) -> str:
    """Read user input via rich or builtins."""
    if HAS_RICH:
        return console.input(prompt)
    return input(prompt)


def _print_diagnostics(result) -> None:
    """Display pipeline diagnostics after each turn."""
    d = result.diagnostics
    tokens_used = result.window.used_tokens
    tokens_max = result.window.max_tokens
    utilization = d.get("token_utilization", 0) * 100
    items = d.get("items_included", 0)
    build_ms = result.build_time_ms

    _print(
        f"  [{items} items | {tokens_used}/{tokens_max} tokens "
        f"({utilization:.0f}%) | {build_ms:.0f}ms]",
        style="info",
    )


def main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        _print(
            "Set ANTHROPIC_API_KEY environment variable to use this example.\n"
            "  export ANTHROPIC_API_KEY=sk-ant-...",
            style="danger",
        )
        sys.exit(1)

    # -- Pipeline setup -------------------------------------------------------
    memory = MemoryManager(conversation_tokens=CONVERSATION_TOKENS)
    pipeline = (
        ContextPipeline(max_tokens=MAX_PIPELINE_TOKENS)
        .with_memory(memory)
        .with_formatter(AnthropicFormatter())
        .add_system_prompt(SYSTEM_PROMPT)
    )
    client = anthropic.Anthropic(api_key=api_key)

    _print("astro-context chat example", style="bold")
    _print(f"Model: {MODEL} | Budget: {MAX_PIPELINE_TOKENS} tokens", style="info")
    _print('Type "quit" or "exit" to end the session.\n', style="info")

    # -- Chat loop ------------------------------------------------------------
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

        # 1. Record user message
        memory.add_user_message(user_input)

        # 2. Build context through pipeline
        result = pipeline.build(user_input)
        formatted = result.formatted_output

        # 3. Stream response from Claude
        _print("Assistant:", style="bold blue")
        response_text = ""
        try:
            with client.messages.stream(
                model=MODEL,
                max_tokens=1024,
                system=formatted["system"],
                messages=formatted["messages"],
            ) as stream:
                for text in stream.text_stream:
                    response_text += text
        except anthropic.APIError as exc:
            _print(f"API error: {exc}", style="danger")
            memory.add_assistant_message("[error: API call failed]")
            continue
        except KeyboardInterrupt:
            _print("\n[interrupted]", style="warning")
            if response_text:
                memory.add_assistant_message(response_text + " [interrupted]")
            continue

        # 4. Display response and record it
        _print_markdown(response_text)

        # 5. Feed response back into memory
        memory.add_assistant_message(response_text)

        # 6. Show diagnostics
        _print_diagnostics(result)
        print()


if __name__ == "__main__":
    main()
