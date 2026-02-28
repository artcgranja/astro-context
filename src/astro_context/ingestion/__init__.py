"""Document ingestion module for astro-context.

Provides chunkers, parsers, metadata utilities, and an orchestrator
for converting raw documents into ``ContextItem`` objects.
"""

from .chunkers import FixedSizeChunker, RecursiveCharacterChunker, SentenceChunker
from .ingester import DocumentIngester
from .metadata import MetadataEnricher, extract_chunk_metadata, generate_chunk_id, generate_doc_id
from .parsers import HTMLParser, MarkdownParser, PDFParser, PlainTextParser

__all__ = [
    "DocumentIngester",
    "FixedSizeChunker",
    "HTMLParser",
    "MarkdownParser",
    "MetadataEnricher",
    "PDFParser",
    "PlainTextParser",
    "RecursiveCharacterChunker",
    "SentenceChunker",
    "extract_chunk_metadata",
    "generate_chunk_id",
    "generate_doc_id",
]
