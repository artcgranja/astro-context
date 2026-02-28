"""Zero-dependency modality encoders for multi-modal content."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from astro_context.multimodal.models import ModalityType, MultiModalContent

logger = logging.getLogger(__name__)


class TextEncoder:
    """Pass-through encoder for text content.

    Simply returns the text ``content`` field unchanged.
    Implements the ``ModalityEncoder`` protocol.
    """

    __slots__ = ()

    def encode(self, content: MultiModalContent) -> str:
        """Return the text content as-is.

        Parameters:
            content: A ``MultiModalContent`` with ``modality=TEXT``.

        Returns:
            The ``content`` string unchanged.
        """
        return content.content

    @property
    def supported_modalities(self) -> list[ModalityType]:
        """Return supported modalities.

        Returns:
            A list containing only ``ModalityType.TEXT``.
        """
        return [ModalityType.TEXT]


class TableEncoder:
    """Converts table content into a Markdown text representation.

    If the content already contains a Markdown-formatted table it is
    returned as-is.  Implements the ``ModalityEncoder`` protocol.
    """

    __slots__ = ()

    def encode(self, content: MultiModalContent) -> str:
        """Encode table content as Markdown.

        Parameters:
            content: A ``MultiModalContent`` with ``modality=TABLE``.

        Returns:
            A Markdown representation of the table.
        """
        return content.content

    @property
    def supported_modalities(self) -> list[ModalityType]:
        """Return supported modalities.

        Returns:
            A list containing only ``ModalityType.TABLE``.
        """
        return [ModalityType.TABLE]


class ImageDescriptionEncoder:
    """Encodes image content into text via an optional description callback.

    If a ``describe_fn`` callback is provided it receives the raw image
    bytes and returns a textual description.  When no callback is set
    (or raw data is missing), the encoder falls back to the metadata
    ``description`` key, and finally to the ``content`` field.

    Implements the ``ModalityEncoder`` protocol.
    """

    __slots__ = ("_describe_fn",)

    def __init__(self, describe_fn: Callable[[bytes], str] | None = None) -> None:
        self._describe_fn = describe_fn

    def encode(self, content: MultiModalContent) -> str:
        """Encode image content into text.

        Parameters:
            content: A ``MultiModalContent`` with ``modality=IMAGE``.

        Returns:
            A textual description of the image.
        """
        if self._describe_fn is not None and content.raw_data is not None:
            return self._describe_fn(content.raw_data)

        description: Any = content.metadata.get("description")
        if isinstance(description, str) and description:
            return description

        return content.content

    @property
    def supported_modalities(self) -> list[ModalityType]:
        """Return supported modalities.

        Returns:
            A list containing only ``ModalityType.IMAGE``.
        """
        return [ModalityType.IMAGE]


class CompositeEncoder:
    """Routes encoding to the appropriate encoder based on modality type.

    Holds a mapping of ``ModalityType`` to encoder instances and delegates
    ``encode`` calls accordingly.  Implements the ``ModalityEncoder`` protocol.
    """

    __slots__ = ("_encoders",)

    def __init__(
        self,
        encoders: dict[ModalityType, TextEncoder | TableEncoder | ImageDescriptionEncoder]
        | None = None,
    ) -> None:
        if encoders is not None:
            self._encoders: dict[
                ModalityType, TextEncoder | TableEncoder | ImageDescriptionEncoder
            ] = dict(encoders)
        else:
            self._encoders = {
                ModalityType.TEXT: TextEncoder(),
                ModalityType.TABLE: TableEncoder(),
                ModalityType.IMAGE: ImageDescriptionEncoder(),
                ModalityType.CODE: TextEncoder(),
            }

    def encode(self, content: MultiModalContent) -> str:
        """Encode content by delegating to the appropriate sub-encoder.

        Parameters:
            content: The multi-modal content to encode.

        Returns:
            A text representation of the content.

        Raises:
            ValueError: If no encoder is registered for the content's modality.
        """
        encoder = self._encoders.get(content.modality)
        if encoder is None:
            msg = (
                f"No encoder registered for modality {content.modality!r}. "
                f"Supported: {list(self._encoders)}"
            )
            raise ValueError(msg)
        return encoder.encode(content)

    @property
    def supported_modalities(self) -> list[ModalityType]:
        """Return all modalities this composite encoder can handle.

        Returns:
            A list of all registered ``ModalityType`` values.
        """
        return list(self._encoders)
