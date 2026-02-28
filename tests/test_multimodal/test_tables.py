"""Tests for table extraction utilities."""

from __future__ import annotations

import tempfile
from pathlib import Path

from astro_context.multimodal.models import ModalityType
from astro_context.multimodal.tables import HTMLTableParser, MarkdownTableParser
from astro_context.protocols.multimodal import TableExtractor


class TestMarkdownTableParser:
    """Tests for MarkdownTableParser."""

    def test_extract_single_table(self) -> None:
        md = "Some text\n\n| A | B |\n| --- | --- |\n| 1 | 2 |\n\nMore text"
        parser = MarkdownTableParser()
        tables = parser.extract_tables(md.encode("utf-8"))
        assert len(tables) == 1
        assert tables[0].modality == ModalityType.TABLE
        assert "| A | B |" in tables[0].content
        assert tables[0].metadata["format"] == "markdown"

    def test_extract_multiple_tables(self) -> None:
        md = (
            "| A | B |\n| --- | --- |\n| 1 | 2 |\n\n"
            "Text between\n\n"
            "| X | Y |\n| --- | --- |\n| 3 | 4 |\n"
        )
        parser = MarkdownTableParser()
        tables = parser.extract_tables(md.encode("utf-8"))
        assert len(tables) == 2

    def test_no_tables(self) -> None:
        parser = MarkdownTableParser()
        tables = parser.extract_tables(b"Just some text with no tables")
        assert tables == []

    def test_from_file(self) -> None:
        md = "| Name | Age |\n| --- | --- |\n| Alice | 30 |\n| Bob | 25 |"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(md)
            f.flush()
            path = Path(f.name)

        parser = MarkdownTableParser()
        tables = parser.extract_tables(path)
        assert len(tables) == 1
        assert "Alice" in tables[0].content

        path.unlink()

    def test_protocol_compliance(self) -> None:
        parser = MarkdownTableParser()
        assert isinstance(parser, TableExtractor)

    def test_empty_input(self) -> None:
        parser = MarkdownTableParser()
        tables = parser.extract_tables(b"")
        assert tables == []


class TestHTMLTableParser:
    """Tests for HTMLTableParser."""

    def test_extract_single_table(self) -> None:
        html = """
        <html><body>
        <table>
            <tr><th>Name</th><th>Age</th></tr>
            <tr><td>Alice</td><td>30</td></tr>
            <tr><td>Bob</td><td>25</td></tr>
        </table>
        </body></html>
        """
        parser = HTMLTableParser()
        tables = parser.extract_tables(html.encode("utf-8"))
        assert len(tables) == 1
        assert tables[0].modality == ModalityType.TABLE
        assert "Alice" in tables[0].content
        assert "Bob" in tables[0].content
        assert tables[0].metadata["format"] == "html"

    def test_extract_multiple_tables(self) -> None:
        html = """
        <table>
            <tr><td>A</td><td>B</td></tr>
        </table>
        <p>Some text</p>
        <table>
            <tr><td>X</td><td>Y</td></tr>
        </table>
        """
        parser = HTMLTableParser()
        tables = parser.extract_tables(html.encode("utf-8"))
        assert len(tables) == 2

    def test_no_tables(self) -> None:
        parser = HTMLTableParser()
        tables = parser.extract_tables(b"<html><body><p>No tables</p></body></html>")
        assert tables == []

    def test_nested_tables_only_outer(self) -> None:
        html = """
        <table>
            <tr><td>Outer</td><td>
                <table><tr><td>Inner</td></tr></table>
            </td></tr>
        </table>
        """
        parser = HTMLTableParser()
        tables = parser.extract_tables(html.encode("utf-8"))
        # Only outer table at depth=1 is captured
        assert len(tables) == 1
        assert "Outer" in tables[0].content

    def test_from_file(self) -> None:
        html = "<table><tr><th>X</th></tr><tr><td>1</td></tr></table>"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
            f.write(html)
            f.flush()
            path = Path(f.name)

        parser = HTMLTableParser()
        tables = parser.extract_tables(path)
        assert len(tables) == 1

        path.unlink()

    def test_converts_to_markdown(self) -> None:
        html = "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>"
        parser = HTMLTableParser()
        tables = parser.extract_tables(html.encode("utf-8"))
        assert len(tables) == 1
        content = tables[0].content
        assert "| A | B |" in content
        assert "| --- | --- |" in content
        assert "| 1 | 2 |" in content

    def test_protocol_compliance(self) -> None:
        parser = HTMLTableParser()
        assert isinstance(parser, TableExtractor)

    def test_empty_input(self) -> None:
        parser = HTMLTableParser()
        tables = parser.extract_tables(b"")
        assert tables == []

    def test_uneven_columns(self) -> None:
        html = "<table><tr><td>A</td><td>B</td><td>C</td></tr><tr><td>1</td></tr></table>"
        parser = HTMLTableParser()
        tables = parser.extract_tables(html.encode("utf-8"))
        assert len(tables) == 1
        # Should handle uneven rows gracefully
        assert "A" in tables[0].content
