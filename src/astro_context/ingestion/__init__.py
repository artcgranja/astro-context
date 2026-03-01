"""Document ingestion module for astro-context.

Provides chunkers, parsers, metadata utilities, and an orchestrator
for converting raw documents into ``ContextItem`` objects.
"""

from .chunkers import FixedSizeChunker, RecursiveCharacterChunker, SemanticChunker, SentenceChunker
from .code_chunker import CodeChunker
from .hierarchical import ParentChildChunker, ParentExpander
from .ingester import DocumentIngester
from .metadata import MetadataEnricher, extract_chunk_metadata, generate_chunk_id, generate_doc_id
from .parsers import HTMLParser, MarkdownParser, PDFParser, PlainTextParser
from .table_chunker import TableAwareChunker

__all__ = [
    "CodeChunker",
    "DocumentIngester",
    "FixedSizeChunker",
    "HTMLParser",
    "MarkdownParser",
    "MetadataEnricher",
    "PDFParser",
    "ParentChildChunker",
    "ParentExpander",
    "PlainTextParser",
    "RecursiveCharacterChunker",
    "SemanticChunker",
    "SentenceChunker",
    "TableAwareChunker",
    "extract_chunk_metadata",
    "generate_chunk_id",
    "generate_doc_id",
]
