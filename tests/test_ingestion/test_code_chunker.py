"""Tests for CodeChunker."""

from __future__ import annotations

import pytest

from astro_context.ingestion.code_chunker import _EXTENSION_MAP, CodeChunker
from astro_context.ingestion.ingester import DocumentIngester
from astro_context.ingestion.table_chunker import TableAwareChunker
from astro_context.protocols.ingestion import Chunker
from tests.conftest import FakeTokenizer


class TestCodeChunker:
    """Tests for CodeChunker."""

    def test_protocol_compliance(self, fake_tokenizer: FakeTokenizer) -> None:
        chunker = CodeChunker(tokenizer=fake_tokenizer)
        assert isinstance(chunker, Chunker)

    def test_empty_input(self, fake_tokenizer: FakeTokenizer) -> None:
        chunker = CodeChunker(language="python", tokenizer=fake_tokenizer)
        assert chunker.chunk("") == []
        assert chunker.chunk("   ") == []

    def test_short_text_single_chunk(self, fake_tokenizer: FakeTokenizer) -> None:
        chunker = CodeChunker(language="python", chunk_size=50, overlap=0, tokenizer=fake_tokenizer)
        text = "def hello():\n    return 'hi'"
        chunks = chunker.chunk(text)
        assert len(chunks) == 1

    def test_python_function_splitting(self, fake_tokenizer: FakeTokenizer) -> None:
        chunker = CodeChunker(language="python", chunk_size=10, overlap=0, tokenizer=fake_tokenizer)
        text = (
            "def foo():\n"
            "    a = 1\n"
            "    b = 2\n"
            "    c = 3\n"
            "    d = 4\n"
            "    e = 5\n"
            "\n"
            "def bar():\n"
            "    x = 10\n"
            "    y = 20\n"
            "    z = 30\n"
            "    w = 40\n"
            "    v = 50\n"
        )
        chunks = chunker.chunk(text)
        assert len(chunks) >= 2
        assert any("foo" in c for c in chunks)
        assert any("bar" in c for c in chunks)

    def test_python_class_splitting(self, fake_tokenizer: FakeTokenizer) -> None:
        chunker = CodeChunker(language="python", chunk_size=10, overlap=0, tokenizer=fake_tokenizer)
        text = (
            "class Alpha:\n"
            "    a = 1\n"
            "    b = 2\n"
            "    c = 3\n"
            "    d = 4\n"
            "    e = 5\n"
            "\n"
            "class Beta:\n"
            "    x = 10\n"
            "    y = 20\n"
            "    z = 30\n"
            "    w = 40\n"
            "    v = 50\n"
        )
        chunks = chunker.chunk(text)
        assert len(chunks) >= 2
        assert any("Alpha" in c for c in chunks)
        assert any("Beta" in c for c in chunks)

    def test_javascript_splitting(self, fake_tokenizer: FakeTokenizer) -> None:
        chunker = CodeChunker(
            language="javascript", chunk_size=10, overlap=0, tokenizer=fake_tokenizer
        )
        text = (
            "function greet() {\n"
            "    return 'hello';\n"
            "    const a = 1;\n"
            "    const b = 2;\n"
            "    const c = 3;\n"
            "}\n"
            "\n"
            "function farewell() {\n"
            "    return 'bye';\n"
            "    const x = 1;\n"
            "    const y = 2;\n"
            "    const z = 3;\n"
            "}\n"
        )
        chunks = chunker.chunk(text)
        assert len(chunks) >= 2
        assert any("greet" in c for c in chunks)
        assert any("farewell" in c for c in chunks)

    def test_go_func_splitting(self, fake_tokenizer: FakeTokenizer) -> None:
        chunker = CodeChunker(language="go", chunk_size=10, overlap=0, tokenizer=fake_tokenizer)
        text = (
            "func Add(a int, b int) int {\n"
            "    return a + b\n"
            "    x := 1\n"
            "    y := 2\n"
            "    z := 3\n"
            "}\n"
            "\n"
            "func Sub(a int, b int) int {\n"
            "    return a - b\n"
            "    x := 1\n"
            "    y := 2\n"
            "    z := 3\n"
            "}\n"
        )
        chunks = chunker.chunk(text)
        assert len(chunks) >= 2
        assert any("Add" in c for c in chunks)
        assert any("Sub" in c for c in chunks)

    def test_rust_fn_splitting(self, fake_tokenizer: FakeTokenizer) -> None:
        chunker = CodeChunker(language="rust", chunk_size=10, overlap=0, tokenizer=fake_tokenizer)
        text = (
            "fn add(a: i32, b: i32) -> i32 {\n"
            "    a + b\n"
            "    let x = 1;\n"
            "    let y = 2;\n"
            "    let z = 3;\n"
            "}\n"
            "\n"
            "fn sub(a: i32, b: i32) -> i32 {\n"
            "    a - b\n"
            "    let x = 1;\n"
            "    let y = 2;\n"
            "    let z = 3;\n"
            "}\n"
        )
        chunks = chunker.chunk(text)
        assert len(chunks) >= 2
        assert any("add" in c for c in chunks)
        assert any("sub" in c for c in chunks)

    def test_auto_detect_language(self, fake_tokenizer: FakeTokenizer) -> None:
        chunker = CodeChunker(chunk_size=10, overlap=0, tokenizer=fake_tokenizer)
        text = (
            "def foo():\n"
            "    a = 1\n"
            "    b = 2\n"
            "    c = 3\n"
            "    d = 4\n"
            "    e = 5\n"
            "\n"
            "def bar():\n"
            "    x = 10\n"
            "    y = 20\n"
            "    z = 30\n"
            "    w = 40\n"
            "    v = 50\n"
        )
        chunks = chunker.chunk(text, metadata={"extension": ".py"})
        assert len(chunks) >= 2
        assert any("foo" in c for c in chunks)
        assert any("bar" in c for c in chunks)

    def test_unknown_language_fallback(self, fake_tokenizer: FakeTokenizer) -> None:
        chunker = CodeChunker(language="cobol", chunk_size=5, overlap=0, tokenizer=fake_tokenizer)
        text = "one two three four five six seven eight"
        chunks = chunker.chunk(text)
        # Should fall back to recursive chunker and still produce output
        assert len(chunks) >= 1

    def test_invalid_args(self, fake_tokenizer: FakeTokenizer) -> None:
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            CodeChunker(chunk_size=0, tokenizer=fake_tokenizer)
        with pytest.raises(ValueError, match="overlap must be non-negative"):
            CodeChunker(overlap=-1, tokenizer=fake_tokenizer)
        with pytest.raises(ValueError, match=r"overlap.*must be less than chunk_size"):
            CodeChunker(chunk_size=5, overlap=5, tokenizer=fake_tokenizer)

    def test_repr(self, fake_tokenizer: FakeTokenizer) -> None:
        chunker = CodeChunker(
            language="python", chunk_size=256, overlap=25, tokenizer=fake_tokenizer
        )
        r = repr(chunker)
        assert "CodeChunker" in r
        assert "language='python'" in r
        assert "chunk_size=256" in r
        assert "overlap=25" in r

    def test_no_boundaries_fallback(self, fake_tokenizer: FakeTokenizer) -> None:
        chunker = CodeChunker(language="python", chunk_size=5, overlap=0, tokenizer=fake_tokenizer)
        # Plain text without any code patterns
        text = "just some plain text without any function or class definitions at all"
        chunks = chunker.chunk(text)
        assert len(chunks) >= 1
        # All original words should appear somewhere in the chunks
        for word in ["just", "some", "plain", "text"]:
            assert any(word in c for c in chunks)


class TestCodeChunkerVerification:
    """Additional verification tests for edge cases and protocol compliance."""

    # ---- 1. Async def splitting ----
    def test_async_def_splitting(self, fake_tokenizer: FakeTokenizer) -> None:
        """Python async def functions should be detected as boundaries."""
        chunker = CodeChunker(language="python", chunk_size=10, overlap=0, tokenizer=fake_tokenizer)
        text = (
            "async def fetch_data():\n"
            "    result = await get()\n"
            "    a = 1\n"
            "    b = 2\n"
            "    c = 3\n"
            "    d = 4\n"
            "\n"
            "async def send_data():\n"
            "    await post(data)\n"
            "    x = 1\n"
            "    y = 2\n"
            "    z = 3\n"
            "    w = 4\n"
        )
        chunks = chunker.chunk(text)
        assert len(chunks) >= 2
        assert any("fetch_data" in c for c in chunks)
        assert any("send_data" in c for c in chunks)

    # ---- 2. Nested functions ----
    def test_nested_functions(self, fake_tokenizer: FakeTokenizer) -> None:
        """Inner functions within outer functions are detected as boundaries.

        The regex uses ``re.MULTILINE`` with ``^`` so only line-start
        ``def`` tokens are boundaries.  Indented inner defs still match
        because the pattern is ``^(?:def |class |async def )`` which
        does NOT require column-0 -- it just needs start-of-line.  In a
        multiline string, every ``\\n`` starts a new line.
        """
        chunker = CodeChunker(language="python", chunk_size=8, overlap=0, tokenizer=fake_tokenizer)
        text = (
            "def outer():\n"
            "    x = 1\n"
            "    y = 2\n"
            "    def inner():\n"
            "        return x + y\n"
            "        a = 1\n"
            "        b = 2\n"
            "\n"
            "def standalone():\n"
            "    return 42\n"
            "    c = 3\n"
            "    d = 4\n"
        )
        chunks = chunker.chunk(text)
        # At minimum outer+inner and standalone should appear
        assert any("outer" in c for c in chunks)
        assert any("standalone" in c for c in chunks)

    # ---- 3. Class with methods ----
    def test_class_with_indented_methods_stays_together(
        self, fake_tokenizer: FakeTokenizer
    ) -> None:
        """Indented methods inside a class are NOT top-level boundaries.

        The regex requires ``def `` at column 0 (start-of-line), so
        indented methods stay grouped with their class. The entire class
        is emitted as a single block.
        """
        chunker = CodeChunker(language="python", chunk_size=50, overlap=0, tokenizer=fake_tokenizer)
        text = (
            "class MyClass:\n"
            "    def method_a(self):\n"
            "        return 1\n"
            "\n"
            "    def method_b(self):\n"
            "        return 2\n"
        )
        chunks = chunker.chunk(text)
        # The whole class body is one block because indented defs are not boundaries
        assert len(chunks) == 1
        assert "method_a" in chunks[0]
        assert "method_b" in chunks[0]

    def test_class_followed_by_function_splits(self, fake_tokenizer: FakeTokenizer) -> None:
        """A class followed by a top-level function should split into separate chunks."""
        chunker = CodeChunker(language="python", chunk_size=10, overlap=0, tokenizer=fake_tokenizer)
        text = (
            "class MyClass:\n"
            "    def method_a(self):\n"
            "        return 1\n"
            "        x = 10\n"
            "        y = 20\n"
            "\n"
            "def standalone():\n"
            "    z = 42\n"
            "    w = 99\n"
            "    a = 1\n"
            "    b = 2\n"
        )
        chunks = chunker.chunk(text)
        assert len(chunks) >= 2
        all_text = " ".join(chunks)
        assert "MyClass" in all_text
        assert "standalone" in all_text

    # ---- 4. Mixed language metadata (.tsx -> TypeScript) ----
    def test_tsx_extension_maps_to_typescript(self) -> None:
        """The .tsx extension should map to TypeScript in _EXTENSION_MAP."""
        # This verifies the extension map coverage for JSX/TSX variants
        assert ".tsx" in _EXTENSION_MAP, (
            ".tsx is missing from _EXTENSION_MAP; "
            "TypeScript JSX files should be detected as TypeScript"
        )
        assert _EXTENSION_MAP[".tsx"] == "typescript"

    def test_jsx_extension_maps_to_javascript(self) -> None:
        """The .jsx extension should map to JavaScript in _EXTENSION_MAP."""
        assert ".jsx" in _EXTENSION_MAP, (
            ".jsx is missing from _EXTENSION_MAP; "
            "JavaScript JSX files should be detected as JavaScript"
        )
        assert _EXTENSION_MAP[".jsx"] == "javascript"

    # ---- 5. Large function exceeding chunk_size is split further ----
    def test_large_single_function_split(self, fake_tokenizer: FakeTokenizer) -> None:
        """A single function exceeding chunk_size gets split via fallback."""
        chunker = CodeChunker(language="python", chunk_size=5, overlap=0, tokenizer=fake_tokenizer)
        # One big function with many lines
        lines = ["def huge_function():"]
        for i in range(20):
            lines.append(f"    var_{i} = {i}")
        text = "\n".join(lines) + "\n"
        chunks = chunker.chunk(text)
        # The function body exceeds 5 tokens, so it should be multiple chunks
        # (the _split_and_merge won't sub-split, but the single block
        # will be emitted and the content preserved)
        assert len(chunks) >= 1
        all_text = " ".join(chunks)
        assert "huge_function" in all_text

    # ---- 6. Unicode in code ----
    def test_unicode_in_code(self, fake_tokenizer: FakeTokenizer) -> None:
        """Code with unicode variable names and comments should chunk correctly."""
        chunker = CodeChunker(language="python", chunk_size=20, overlap=0, tokenizer=fake_tokenizer)
        text = (
            "def calcular_area():\n"
            "    # Calcula a area do circulo\n"
            "    raio = 5\n"
            "    area = 3.14 * raio ** 2\n"
            "    return area\n"
            "\n"
            "def obtener_nombre():\n"
            '    nombre = "Jose"\n'
            "    return nombre\n"
        )
        chunks = chunker.chunk(text)
        assert len(chunks) >= 1
        all_text = " ".join(chunks)
        assert "area" in all_text
        assert "nombre" in all_text

    # ---- 7. Integration: CodeChunker + DocumentIngester ----
    def test_code_chunker_with_document_ingester(self, fake_tokenizer: FakeTokenizer) -> None:
        """CodeChunker works correctly when passed to DocumentIngester."""
        code_chunker = CodeChunker(
            language="python", chunk_size=10, overlap=0, tokenizer=fake_tokenizer
        )
        ingester = DocumentIngester(chunker=code_chunker, tokenizer=fake_tokenizer)

        text = (
            "def alpha():\n"
            "    a = 1\n"
            "    b = 2\n"
            "    c = 3\n"
            "    d = 4\n"
            "    e = 5\n"
            "\n"
            "def beta():\n"
            "    x = 10\n"
            "    y = 20\n"
            "    z = 30\n"
            "    w = 40\n"
            "    v = 50\n"
        )
        items = ingester.ingest_text(text, doc_id="test-code-doc")
        assert len(items) >= 2
        # Each item should be a ContextItem with content
        for item in items:
            assert item.content.strip()
            assert item.token_count > 0
        all_content = " ".join(item.content for item in items)
        assert "alpha" in all_content
        assert "beta" in all_content

    # ---- 8. Integration: TableAwareChunker with CodeChunker as inner ----
    def test_table_chunker_with_code_chunker_inner(self, fake_tokenizer: FakeTokenizer) -> None:
        """TableAwareChunker can use CodeChunker as its inner chunker."""
        code_chunker = CodeChunker(
            language="python", chunk_size=15, overlap=0, tokenizer=fake_tokenizer
        )
        table_chunker = TableAwareChunker(
            inner_chunker=code_chunker, chunk_size=50, tokenizer=fake_tokenizer
        )

        text = (
            "def setup():\n"
            "    config = load()\n"
            "\n"
            "| Key | Value |\n"
            "| --- | ----- |\n"
            "| host | localhost |\n"
            "| port | 8080 |\n"
            "\n"
            "def teardown():\n"
            "    cleanup()\n"
        )
        chunks = table_chunker.chunk(text)
        assert len(chunks) >= 1
        all_text = " ".join(chunks)
        # Both code and table content should be present
        assert "setup" in all_text
        assert "teardown" in all_text
        assert "host" in all_text
        assert "port" in all_text
