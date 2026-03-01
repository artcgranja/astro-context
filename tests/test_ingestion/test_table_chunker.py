"""Tests for TableAwareChunker."""

from __future__ import annotations

from astro_context.ingestion.chunkers import RecursiveCharacterChunker
from astro_context.ingestion.table_chunker import TableAwareChunker
from astro_context.protocols.ingestion import Chunker
from tests.conftest import FakeTokenizer


class TestTableAwareChunker:
    """Tests for TableAwareChunker."""

    def test_protocol_compliance(self, fake_tokenizer: FakeTokenizer) -> None:
        chunker = TableAwareChunker(tokenizer=fake_tokenizer)
        assert isinstance(chunker, Chunker)

    def test_empty_input(self, fake_tokenizer: FakeTokenizer) -> None:
        chunker = TableAwareChunker(tokenizer=fake_tokenizer)
        assert chunker.chunk("") == []
        assert chunker.chunk("   ") == []

    def test_text_without_tables(self, fake_tokenizer: FakeTokenizer) -> None:
        chunker = TableAwareChunker(chunk_size=50, tokenizer=fake_tokenizer)
        text = "This is just some plain text without any tables."
        chunks = chunker.chunk(text)
        assert len(chunks) >= 1
        assert any("plain text" in c for c in chunks)

    def test_markdown_table_preservation(self, fake_tokenizer: FakeTokenizer) -> None:
        chunker = TableAwareChunker(chunk_size=50, tokenizer=fake_tokenizer)
        table = "| Name | Age |\n| ---- | --- |\n| Alice | 30 |\n| Bob | 25 |\n"
        text = f"Some intro text.\n\n{table}\nSome outro text."
        chunks = chunker.chunk(text)
        # The table should be kept as an atomic chunk
        table_chunks = [c for c in chunks if "Alice" in c and "Bob" in c]
        assert len(table_chunks) >= 1

    def test_html_table_preservation(self, fake_tokenizer: FakeTokenizer) -> None:
        chunker = TableAwareChunker(chunk_size=50, tokenizer=fake_tokenizer)
        table = (
            "<table>\n"
            "<tr><th>Name</th><th>Age</th></tr>\n"
            "<tr><td>Alice</td><td>30</td></tr>\n"
            "<tr><td>Bob</td><td>25</td></tr>\n"
            "</table>"
        )
        text = f"Some intro text.\n\n{table}\n\nSome outro text."
        chunks = chunker.chunk(text)
        # Table should be preserved
        table_chunks = [c for c in chunks if "Alice" in c and "Bob" in c]
        assert len(table_chunks) >= 1

    def test_large_table_row_split(self, fake_tokenizer: FakeTokenizer) -> None:
        # Build a markdown table that exceeds chunk_size
        header = "| Col1 | Col2 | Col3 |\n| ---- | ---- | ---- |\n"
        rows = "".join(f"| val{i}a | val{i}b | val{i}c |\n" for i in range(20))
        table = header + rows
        chunker = TableAwareChunker(chunk_size=10, tokenizer=fake_tokenizer)
        chunks = chunker.chunk(table)
        assert len(chunks) >= 2
        # Each chunk should contain the header
        for c in chunks:
            assert "Col1" in c

    def test_markdown_header_preserved(self, fake_tokenizer: FakeTokenizer) -> None:
        header = "| Name | Score |\n| ---- | ----- |\n"
        rows = "".join(f"| Student{i} | {i * 10} |\n" for i in range(15))
        table = header + rows
        chunker = TableAwareChunker(chunk_size=10, tokenizer=fake_tokenizer)
        chunks = chunker.chunk(table)
        assert len(chunks) >= 2
        for c in chunks:
            assert "Name" in c
            assert "Score" in c

    def test_mixed_text_and_tables(self, fake_tokenizer: FakeTokenizer) -> None:
        chunker = TableAwareChunker(chunk_size=50, tokenizer=fake_tokenizer)
        text = (
            "Introduction paragraph here.\n\n"
            "| A | B |\n"
            "| - | - |\n"
            "| 1 | 2 |\n"
            "\n"
            "Middle paragraph.\n\n"
            "<table><tr><td>X</td></tr></table>\n\n"
            "Conclusion paragraph."
        )
        chunks = chunker.chunk(text)
        assert len(chunks) >= 1
        all_text = " ".join(chunks)
        assert "Introduction" in all_text
        assert "Conclusion" in all_text

    def test_custom_inner_chunker(self, fake_tokenizer: FakeTokenizer) -> None:
        inner = RecursiveCharacterChunker(chunk_size=5, overlap=0, tokenizer=fake_tokenizer)
        chunker = TableAwareChunker(inner_chunker=inner, chunk_size=50, tokenizer=fake_tokenizer)
        text = "one two three four five six seven eight nine ten"
        chunks = chunker.chunk(text)
        # The inner chunker should split at 5 words
        assert len(chunks) >= 2

    def test_repr(self, fake_tokenizer: FakeTokenizer) -> None:
        chunker = TableAwareChunker(chunk_size=256, tokenizer=fake_tokenizer)
        r = repr(chunker)
        assert "TableAwareChunker" in r
        assert "chunk_size=256" in r
        assert "inner=" in r


class TestTableAwareChunkerVerification:
    """Additional verification tests for edge cases."""

    # ---- 1. Multiple tables interspersed with prose ----
    def test_multiple_tables_with_prose(self, fake_tokenizer: FakeTokenizer) -> None:
        """Text with 2+ tables interspersed with prose preserves all content."""
        chunker = TableAwareChunker(chunk_size=50, tokenizer=fake_tokenizer)
        text = (
            "Introduction paragraph.\n\n"
            "| Name | Role |\n"
            "| ---- | ---- |\n"
            "| Alice | Engineer |\n"
            "\n"
            "Some middle prose connecting the tables.\n\n"
            "| Product | Price |\n"
            "| ------- | ----- |\n"
            "| Widget | 9.99 |\n"
            "| Gadget | 19.99 |\n"
            "\n"
            "Conclusion paragraph.\n"
        )
        chunks = chunker.chunk(text)
        all_text = " ".join(chunks)
        assert "Introduction" in all_text
        assert "Alice" in all_text
        assert "middle prose" in all_text
        assert "Widget" in all_text
        assert "Gadget" in all_text
        assert "Conclusion" in all_text

    # ---- 2. Empty table (header only) ----
    def test_empty_table_header_only(self, fake_tokenizer: FakeTokenizer) -> None:
        """A table with only a header row and separator should be handled."""
        chunker = TableAwareChunker(chunk_size=50, tokenizer=fake_tokenizer)
        text = "Before text.\n\n| Col1 | Col2 |\n| ---- | ---- |\n\nAfter text.\n"
        chunks = chunker.chunk(text)
        all_text = " ".join(chunks)
        assert "Col1" in all_text
        assert "Before" in all_text
        assert "After" in all_text

    # ---- 3. Table at start of text ----
    def test_table_at_start(self, fake_tokenizer: FakeTokenizer) -> None:
        """A table as the very first element in text should be preserved."""
        chunker = TableAwareChunker(chunk_size=50, tokenizer=fake_tokenizer)
        text = (
            "| Name | Age |\n| ---- | --- |\n| Bob | 30 |\n\nSome trailing text after the table.\n"
        )
        chunks = chunker.chunk(text)
        all_text = " ".join(chunks)
        assert "Bob" in all_text
        assert "trailing text" in all_text

    # ---- 4. Table at end of text ----
    def test_table_at_end(self, fake_tokenizer: FakeTokenizer) -> None:
        """A table as the last element in text should be preserved."""
        chunker = TableAwareChunker(chunk_size=50, tokenizer=fake_tokenizer)
        text = (
            "Some leading text before the table.\n\n"
            "| Name | Age |\n"
            "| ---- | --- |\n"
            "| Carol | 25 |\n"
        )
        chunks = chunker.chunk(text)
        all_text = " ".join(chunks)
        assert "Carol" in all_text
        assert "leading text" in all_text

    # ---- 5. Nested HTML tables ----
    def test_nested_html_tables(self, fake_tokenizer: FakeTokenizer) -> None:
        """Nested HTML tables should be handled gracefully without errors."""
        chunker = TableAwareChunker(chunk_size=100, tokenizer=fake_tokenizer)
        text = "<table>\n<tr><td>\n  <table><tr><td>Inner</td></tr></table>\n</td></tr>\n</table>\n"
        # Should not raise; content should appear in output
        chunks = chunker.chunk(text)
        assert len(chunks) >= 1
        all_text = " ".join(chunks)
        assert "Inner" in all_text

    # ---- 6. Markdown table with alignment syntax ----
    def test_markdown_table_with_alignment(self, fake_tokenizer: FakeTokenizer) -> None:
        """Tables using alignment syntax |:---|:---:|---:| should be preserved."""
        chunker = TableAwareChunker(chunk_size=50, tokenizer=fake_tokenizer)
        text = (
            "| Left | Center | Right |\n"
            "|:-----|:------:|------:|\n"
            "| a    | b      | c     |\n"
            "| d    | e      | f     |\n"
        )
        chunks = chunker.chunk(text)
        assert len(chunks) >= 1
        all_text = " ".join(chunks)
        assert "Left" in all_text
        assert "Center" in all_text
        assert "Right" in all_text

    # ---- 7. Very wide table (many columns) ----
    def test_very_wide_table(self, fake_tokenizer: FakeTokenizer) -> None:
        """A table with many columns should be preserved as atomic chunk."""
        cols = [f"Col{i}" for i in range(20)]
        header = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join(["---"] * 20) + " |"
        row = "| " + " | ".join([f"v{i}" for i in range(20)]) + " |"
        table = f"{header}\n{sep}\n{row}\n"
        chunker = TableAwareChunker(chunk_size=200, tokenizer=fake_tokenizer)
        chunks = chunker.chunk(table)
        assert len(chunks) >= 1
        all_text = " ".join(chunks)
        assert "Col0" in all_text
        assert "Col19" in all_text
        assert "v0" in all_text
        assert "v19" in all_text

    # ---- 8. Table with empty cells ----
    def test_table_with_empty_cells(self, fake_tokenizer: FakeTokenizer) -> None:
        """Tables with empty cells like | | value | | should be handled."""
        chunker = TableAwareChunker(chunk_size=50, tokenizer=fake_tokenizer)
        text = "| A | B | C |\n| - | - | - |\n| | value | |\n| x | | z |\n"
        chunks = chunker.chunk(text)
        assert len(chunks) >= 1
        all_text = " ".join(chunks)
        assert "value" in all_text

    # ---- 9. Large table row-split preserves header with alignment ----
    def test_large_aligned_table_row_split_preserves_header(
        self, fake_tokenizer: FakeTokenizer
    ) -> None:
        """When a table with alignment syntax is row-split, header is preserved in each chunk."""
        header = "| Name | Score |\n|:-----|------:|\n"
        rows = "".join(f"| Student{i} | {i * 10} |\n" for i in range(15))
        table = header + rows
        chunker = TableAwareChunker(chunk_size=10, tokenizer=fake_tokenizer)
        chunks = chunker.chunk(table)
        assert len(chunks) >= 2
        for c in chunks:
            assert "Name" in c
            assert "Score" in c
