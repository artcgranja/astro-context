"""Protocol definitions for multi-modal content handling.

Any object matching these signatures can be used as a modality encoder
or table extractor in the pipeline -- no inheritance required.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from astro_context.multimodal.models import ModalityType, MultiModalContent


@runtime_checkable
class ModalityEncoder(Protocol):
    """Encodes multi-modal content into text for embedding and retrieval.

    Implementations convert modality-specific content (images, tables, etc.)
    into a textual representation suitable for downstream text-based
    processing such as embedding generation or LLM prompting.
    """

    def encode(self, content: MultiModalContent) -> str:
        """Encode multi-modal content into a text representation.

        Parameters:
            content: The multi-modal content to encode.

        Returns:
            A text string representing the content.
        """
        ...

    @property
    def supported_modalities(self) -> list[ModalityType]:
        """Return the list of modality types this encoder can handle.

        Returns:
            A list of ``ModalityType`` values supported by this encoder.
        """
        ...


@runtime_checkable
class TableExtractor(Protocol):
    """Extracts structured table data from documents.

    Implementations parse a document source (file path or raw bytes)
    and return any tables found as ``MultiModalContent`` instances
    with ``modality=TABLE``.
    """

    def extract_tables(self, source: Path | bytes) -> list[MultiModalContent]:
        """Extract tables from a document source.

        Parameters:
            source: Either a file path or raw bytes of the document
                to extract tables from.

        Returns:
            A list of ``MultiModalContent`` objects, each representing
            a single table with ``modality`` set to ``ModalityType.TABLE``.
        """
        ...
