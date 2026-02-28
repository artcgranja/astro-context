"""Tests for multi-modal encoders."""

from __future__ import annotations

import pytest

from astro_context.multimodal.encoders import (
    CompositeEncoder,
    ImageDescriptionEncoder,
    TableEncoder,
    TextEncoder,
)
from astro_context.multimodal.models import ModalityType, MultiModalContent
from astro_context.protocols.multimodal import ModalityEncoder


class TestTextEncoder:
    """Tests for TextEncoder."""

    def test_encode_text(self) -> None:
        encoder = TextEncoder()
        content = MultiModalContent(modality=ModalityType.TEXT, content="hello world")
        assert encoder.encode(content) == "hello world"

    def test_encode_empty_text(self) -> None:
        encoder = TextEncoder()
        content = MultiModalContent(modality=ModalityType.TEXT, content="")
        assert encoder.encode(content) == ""

    def test_supported_modalities(self) -> None:
        encoder = TextEncoder()
        assert encoder.supported_modalities == [ModalityType.TEXT]

    def test_protocol_compliance(self) -> None:
        encoder = TextEncoder()
        assert isinstance(encoder, ModalityEncoder)


class TestTableEncoder:
    """Tests for TableEncoder."""

    def test_encode_table(self) -> None:
        md = "| A | B |\n| --- | --- |\n| 1 | 2 |"
        encoder = TableEncoder()
        content = MultiModalContent(modality=ModalityType.TABLE, content=md)
        assert encoder.encode(content) == md

    def test_supported_modalities(self) -> None:
        encoder = TableEncoder()
        assert encoder.supported_modalities == [ModalityType.TABLE]

    def test_protocol_compliance(self) -> None:
        encoder = TableEncoder()
        assert isinstance(encoder, ModalityEncoder)


class TestImageDescriptionEncoder:
    """Tests for ImageDescriptionEncoder."""

    def test_encode_with_callback(self) -> None:
        def describe(data: bytes) -> str:
            return f"Image with {len(data)} bytes"

        encoder = ImageDescriptionEncoder(describe_fn=describe)
        content = MultiModalContent(
            modality=ModalityType.IMAGE,
            content="fallback",
            raw_data=b"\x89PNG\r\n",
        )
        assert encoder.encode(content) == "Image with 6 bytes"

    def test_fallback_to_metadata_description(self) -> None:
        encoder = ImageDescriptionEncoder()
        content = MultiModalContent(
            modality=ModalityType.IMAGE,
            content="fallback text",
            metadata={"description": "A beautiful sunset"},
        )
        assert encoder.encode(content) == "A beautiful sunset"

    def test_fallback_to_content(self) -> None:
        encoder = ImageDescriptionEncoder()
        content = MultiModalContent(
            modality=ModalityType.IMAGE,
            content="fallback text",
        )
        assert encoder.encode(content) == "fallback text"

    def test_callback_with_no_raw_data_falls_back(self) -> None:
        def describe(data: bytes) -> str:
            return "should not reach"

        encoder = ImageDescriptionEncoder(describe_fn=describe)
        content = MultiModalContent(
            modality=ModalityType.IMAGE,
            content="fallback",
            metadata={"description": "from metadata"},
        )
        assert encoder.encode(content) == "from metadata"

    def test_empty_metadata_description_falls_back(self) -> None:
        encoder = ImageDescriptionEncoder()
        content = MultiModalContent(
            modality=ModalityType.IMAGE,
            content="fallback text",
            metadata={"description": ""},
        )
        assert encoder.encode(content) == "fallback text"

    def test_supported_modalities(self) -> None:
        encoder = ImageDescriptionEncoder()
        assert encoder.supported_modalities == [ModalityType.IMAGE]

    def test_protocol_compliance(self) -> None:
        encoder = ImageDescriptionEncoder()
        assert isinstance(encoder, ModalityEncoder)


class TestCompositeEncoder:
    """Tests for CompositeEncoder."""

    def test_default_encoders(self) -> None:
        encoder = CompositeEncoder()
        modalities = encoder.supported_modalities
        assert ModalityType.TEXT in modalities
        assert ModalityType.TABLE in modalities
        assert ModalityType.IMAGE in modalities
        assert ModalityType.CODE in modalities

    def test_encode_text(self) -> None:
        encoder = CompositeEncoder()
        content = MultiModalContent(modality=ModalityType.TEXT, content="hello")
        assert encoder.encode(content) == "hello"

    def test_encode_table(self) -> None:
        encoder = CompositeEncoder()
        md = "| A |\n| --- |\n| 1 |"
        content = MultiModalContent(modality=ModalityType.TABLE, content=md)
        assert encoder.encode(content) == md

    def test_encode_code(self) -> None:
        encoder = CompositeEncoder()
        content = MultiModalContent(modality=ModalityType.CODE, content="print('hi')")
        assert encoder.encode(content) == "print('hi')"

    def test_unsupported_modality_raises(self) -> None:
        encoder = CompositeEncoder()
        content = MultiModalContent(modality=ModalityType.AUDIO, content="audio data")
        with pytest.raises(ValueError, match="No encoder registered for modality"):
            encoder.encode(content)

    def test_custom_encoders(self) -> None:
        encoder = CompositeEncoder(
            encoders={ModalityType.TEXT: TextEncoder()}
        )
        assert encoder.supported_modalities == [ModalityType.TEXT]

    def test_protocol_compliance(self) -> None:
        encoder = CompositeEncoder()
        assert isinstance(encoder, ModalityEncoder)
