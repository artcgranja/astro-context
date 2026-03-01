"""Tests for built-in document parsers."""

from __future__ import annotations

from pathlib import Path

import pytest

from astro_context.exceptions import IngestionError
from astro_context.ingestion.parsers import (
    HTMLParser,
    MarkdownParser,
    PDFParser,
    PlainTextParser,
)
from astro_context.protocols.ingestion import DocumentParser


class TestPlainTextParser:
    """Tests for PlainTextParser."""

    def test_protocol_compliance(self) -> None:
        assert isinstance(PlainTextParser(), DocumentParser)

    def test_parse_bytes(self) -> None:
        parser = PlainTextParser()
        text, meta = parser.parse(b"Hello world\nLine two")
        assert text == "Hello world\nLine two"
        assert meta["line_count"] == 2

    def test_parse_file(self, tmp_path: Path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("Some content here", encoding="utf-8")
        parser = PlainTextParser()
        text, meta = parser.parse(f)
        assert text == "Some content here"
        assert meta["filename"] == "test.txt"
        assert meta["extension"] == ".txt"

    def test_encoding_fallback(self) -> None:
        parser = PlainTextParser()
        # Latin-1 encoded bytes that aren't valid UTF-8
        latin1_bytes = "café résumé".encode("latin-1")
        text, _ = parser.parse(latin1_bytes)
        assert "caf" in text

    def test_supported_extensions(self) -> None:
        assert PlainTextParser().supported_extensions == [".txt"]

    def test_repr(self) -> None:
        assert repr(PlainTextParser()) == "PlainTextParser()"


class TestMarkdownParser:
    """Tests for MarkdownParser."""

    def test_protocol_compliance(self) -> None:
        assert isinstance(MarkdownParser(), DocumentParser)

    def test_parse_bytes(self) -> None:
        parser = MarkdownParser()
        md = b"# My Title\n\nSome content.\n\n## Section\n\nMore content."
        text, meta = parser.parse(md)
        assert "Some content" in text
        assert meta["title"] == "My Title"
        assert len(meta["headings"]) == 2

    def test_heading_extraction(self) -> None:
        parser = MarkdownParser()
        md = b"## Not a title\n\n# Actual Title\n\n### Sub"
        _, meta = parser.parse(md)
        assert meta["title"] == "Actual Title"
        assert len(meta["headings"]) == 3

    def test_frontmatter_detection(self) -> None:
        parser = MarkdownParser()
        md = b"---\ntitle: Test\n---\n\n# Hello\n\nContent."
        text, meta = parser.parse(md)
        assert meta.get("has_frontmatter") is True
        # Frontmatter should be stripped from text
        assert "---" not in text

    def test_parse_file(self, tmp_path: Path) -> None:
        f = tmp_path / "readme.md"
        f.write_text("# Title\n\nContent.", encoding="utf-8")
        parser = MarkdownParser()
        _text, meta = parser.parse(f)
        assert meta["filename"] == "readme.md"
        assert meta["extension"] == ".md"

    def test_supported_extensions(self) -> None:
        exts = MarkdownParser().supported_extensions
        assert ".md" in exts
        assert ".markdown" in exts

    def test_repr(self) -> None:
        assert repr(MarkdownParser()) == "MarkdownParser()"


class TestHTMLParser:
    """Tests for HTMLParser."""

    def test_protocol_compliance(self) -> None:
        assert isinstance(HTMLParser(), DocumentParser)

    def test_strips_tags(self) -> None:
        parser = HTMLParser()
        html = b"<html><body><p>Hello <b>world</b></p></body></html>"
        text, _ = parser.parse(html)
        assert "Hello" in text
        assert "world" in text
        assert "<p>" not in text
        assert "<b>" not in text

    def test_extracts_title(self) -> None:
        parser = HTMLParser()
        html = b"<html><head><title>My Page</title></head><body>Content</body></html>"
        _, meta = parser.parse(html)
        assert meta["title"] == "My Page"

    def test_strips_script_and_style(self) -> None:
        parser = HTMLParser()
        html = (
            b"<html><head><style>body{}</style></head>"
            b"<body><script>alert(1)</script>Text</body></html>"
        )
        text, _ = parser.parse(html)
        assert "alert" not in text
        assert "body{}" not in text
        assert "Text" in text

    def test_parse_file(self, tmp_path: Path) -> None:
        f = tmp_path / "page.html"
        f.write_text("<html><body><p>Content</p></body></html>", encoding="utf-8")
        parser = HTMLParser()
        text, meta = parser.parse(f)
        assert "Content" in text
        assert meta["filename"] == "page.html"

    def test_supported_extensions(self) -> None:
        exts = HTMLParser().supported_extensions
        assert ".html" in exts
        assert ".htm" in exts

    def test_repr(self) -> None:
        assert repr(HTMLParser()) == "HTMLParser()"


class TestPDFParser:
    """Tests for PDFParser."""

    def test_protocol_compliance(self) -> None:
        assert isinstance(PDFParser(), DocumentParser)

    def test_import_error_message(self) -> None:
        """PDFParser.parse raises IngestionError when pypdf is not available."""
        import sys

        # Temporarily hide pypdf to test error handling
        pypdf_module = sys.modules.get("pypdf")
        sys.modules["pypdf"] = None  # type: ignore[assignment]
        try:
            parser = PDFParser()
            with pytest.raises(IngestionError, match="pypdf is required"):
                parser.parse(b"fake pdf bytes")
        finally:
            if pypdf_module is not None:
                sys.modules["pypdf"] = pypdf_module
            else:
                sys.modules.pop("pypdf", None)

    def test_supported_extensions(self) -> None:
        assert PDFParser().supported_extensions == [".pdf"]

    def test_repr(self) -> None:
        assert repr(PDFParser()) == "PDFParser()"
