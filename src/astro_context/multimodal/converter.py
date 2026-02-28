"""Conversion utilities between MultiModalItem and ContextItem."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from astro_context.models.context import ContextItem
from astro_context.multimodal.models import ModalityType, MultiModalContent, MultiModalItem

if TYPE_CHECKING:
    from astro_context.protocols.multimodal import ModalityEncoder

logger = logging.getLogger(__name__)


class MultiModalConverter:
    """Converts between ``MultiModalItem`` and ``ContextItem``.

    Bridges the multi-modal content system with the existing text-based
    context pipeline by encoding each modality into text via a
    ``ModalityEncoder``.
    """

    __slots__ = ()

    @staticmethod
    def to_context_item(item: MultiModalItem, encoder: ModalityEncoder) -> ContextItem:
        """Convert a ``MultiModalItem`` to a ``ContextItem``.

        All content pieces are encoded to text and concatenated with
        double newlines.

        Parameters:
            item: The multi-modal item to convert.
            encoder: The encoder used to produce text from each content piece.

        Returns:
            A ``ContextItem`` whose ``content`` is the concatenated
            text encoding of all modalities.
        """
        parts: list[str] = []
        for content in item.contents:
            parts.append(encoder.encode(content))

        combined = "\n\n".join(parts)
        return ContextItem(
            id=item.id,
            content=combined,
            source=item.source,
            score=item.score,
            priority=item.priority,
            token_count=item.token_count,
            metadata={**item.metadata, "multimodal": True},
            created_at=item.created_at,
        )

    @staticmethod
    def to_context_items(
        items: list[MultiModalItem], encoder: ModalityEncoder
    ) -> list[ContextItem]:
        """Convert a list of ``MultiModalItem`` objects to ``ContextItem`` objects.

        Parameters:
            items: The multi-modal items to convert.
            encoder: The encoder used to produce text from each content piece.

        Returns:
            A list of ``ContextItem`` objects.
        """
        return [MultiModalConverter.to_context_item(item, encoder) for item in items]

    @staticmethod
    def from_context_item(
        item: ContextItem, modality: ModalityType = ModalityType.TEXT
    ) -> MultiModalItem:
        """Convert a ``ContextItem`` to a ``MultiModalItem``.

        Wraps the text content in a single ``MultiModalContent`` of the
        specified modality type.

        Parameters:
            item: The context item to convert.
            modality: The modality to assign to the content.  Defaults to
                ``ModalityType.TEXT``.

        Returns:
            A ``MultiModalItem`` containing a single content piece.
        """
        content = MultiModalContent(
            modality=modality,
            content=item.content,
        )
        return MultiModalItem(
            id=item.id,
            contents=[content],
            source=item.source,
            score=item.score,
            priority=item.priority,
            token_count=item.token_count,
            metadata=item.metadata,
            created_at=item.created_at,
        )
