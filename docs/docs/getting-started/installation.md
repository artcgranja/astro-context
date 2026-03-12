---
icon: material/download
---

# Installation

## Requirements

- **Python 3.11+** is required.
- No system-level dependencies are needed for the core package.

## Install the package

=== "pip"

    ```bash
    pip install astro-context
    ```

=== "uv"

    ```bash
    uv add astro-context
    ```

## Optional extras

astro-context ships with several optional dependency groups. Install only
what you need, or grab everything at once.

=== "pip"

    ```bash
    pip install astro-context[bm25]       # BM25 sparse retrieval
    pip install astro-context[cli]        # CLI tools (typer + rich)
    pip install astro-context[flashrank]  # FlashRank reranker
    pip install astro-context[anthropic]  # Anthropic token counting
    pip install astro-context[otlp]       # OpenTelemetry export
    pip install astro-context[all]        # Everything above
    ```

=== "uv"

    ```bash
    uv add astro-context[bm25]
    uv add astro-context[cli]
    uv add astro-context[flashrank]
    uv add astro-context[anthropic]
    uv add astro-context[otlp]
    uv add astro-context[all]
    ```

| Extra | What it adds | When you need it |
|-------|-------------|-----------------|
| `bm25` | `rank-bm25` | Sparse / keyword retrieval with `SparseRetriever` |
| `cli` | `typer`, `rich` | Using the `astro-context` CLI |
| `flashrank` | `FlashRank` | Client-side reranking without an API call |
| `anthropic` | `anthropic` | Accurate token counting for Claude models |
| `otlp` | `opentelemetry-*` | Exporting traces and metrics via OTLP |
| `all` | All of the above | Kitchen-sink install for development |

## Verifying the installation

After installing, confirm the package is available:

```bash
python -c "import astro_context; print(astro_context.__version__)"
```

If you installed the `cli` extra you can also run:

```bash
astro-context --version
```

## Development setup

To work on astro-context itself, clone the repository and install in
editable mode with all extras:

=== "pip"

    ```bash
    git clone https://github.com/arthurgranja/astro-context.git
    cd astro-context
    pip install -e ".[all,dev]"
    ```

=== "uv"

    ```bash
    git clone https://github.com/arthurgranja/astro-context.git
    cd astro-context
    uv sync --all-extras
    ```

Run the test suite to make sure everything is working:

```bash
pytest
```

!!! tip "Next step"
    Head over to the [Quickstart](quickstart.md) to build your first pipeline
    in under 30 seconds.
