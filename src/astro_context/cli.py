"""CLI interface for astro-context.

Requires the 'cli' extra: pip install astro-context[cli]
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import typer
    from rich.console import Console
    from rich.table import Table
except ImportError:
    print(
        "CLI dependencies not installed. Install with: pip install astro-context[cli]",
        file=sys.stderr,
    )
    sys.exit(1)

from astro_context import __version__

app = typer.Typer(
    name="astro-context",
    help="Context engineering toolkit for AI applications.",
    add_completion=False,
)
console = Console()


@app.callback()
def main(
    version: bool = typer.Option(False, "--version", "-v", help="Show version"),
) -> None:
    if version:
        console.print(f"astro-context {__version__}")
        raise typer.Exit()


@app.command()
def info() -> None:
    """Show information about the astro-context installation."""
    table = Table(title="astro-context info")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Version", __version__)
    table.add_row("Python", sys.version.split()[0])

    for dep_name in ["rank_bm25", "tiktoken", "pydantic"]:
        try:
            mod = __import__(dep_name)
            ver = getattr(mod, "__version__", "installed")
            table.add_row(dep_name, str(ver))
        except ImportError:
            table.add_row(dep_name, "[red]not installed[/red]")

    console.print(table)


@app.command()
def index(
    path: Path = typer.Argument(..., help="Path to file or directory to index"),  # noqa: B008
    chunk_size: int = typer.Option(512, "--chunk-size", "-c", help="Chunk size in tokens"),
) -> None:
    """Index documents from a file or directory (placeholder for MVP)."""
    console.print(f"[yellow]Indexing from {path} (chunk_size={chunk_size})[/yellow]")
    console.print("[dim]Note: Full indexing requires an embedding function. See docs.[/dim]")

    if not path.exists():
        console.print(f"[red]Error: {path} does not exist[/red]")
        raise typer.Exit(code=1)

    if path.is_file():
        content = path.read_text()
        from astro_context.tokens import get_default_counter

        counter = get_default_counter()
        token_count = counter.count_tokens(content)
        console.print(f"  File: {path.name} ({token_count} tokens)")
    elif path.is_dir():
        files = list(path.glob("**/*.txt")) + list(path.glob("**/*.md"))
        console.print(f"  Found {len(files)} text files")


@app.command()
def query(
    query_text: str = typer.Argument(..., help="Query text"),
    max_tokens: int = typer.Option(4096, "--max-tokens", "-t", help="Max context tokens"),
    output_format: str = typer.Option(
        "generic", "--format", "-f", help="Output format: generic|anthropic|openai"
    ),
) -> None:
    """Query the context pipeline (placeholder for MVP)."""
    console.print(f"[yellow]Query: {query_text}[/yellow]")
    console.print(f"[dim]Max tokens: {max_tokens}, Format: {output_format}[/dim]")
    console.print("[dim]Note: Requires indexed documents. See 'astro-context index'.[/dim]")


if __name__ == "__main__":
    app()
