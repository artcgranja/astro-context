"""Tests for multi-modal content models."""

from __future__ import annotations

import uuid

import pytest
from pydantic import ValidationError

from astro_context.models.context import SourceType
from astro_context.multimodal.models import ModalityType, MultiModalContent, MultiModalItem


class TestModalityType:
    """Tests for ModalityType enum."""

    def test_values(self) -> None:
        assert ModalityType.TEXT == "text"
        assert ModalityType.IMAGE == "image"
        assert ModalityType.TABLE == "table"
        assert ModalityType.CODE == "code"
        assert ModalityType.AUDIO == "audio"

    def test_is_str(self) -> None:
        assert isinstance(ModalityType.TEXT, str)


class TestMultiModalContent:
    """Tests for MultiModalContent model."""

    def test_create_text_content(self) -> None:
        content = MultiModalContent(modality=ModalityType.TEXT, content="hello")
        assert content.modality == ModalityType.TEXT
        assert content.content == "hello"
        assert content.raw_data is None
        assert content.mime_type is None
        assert content.metadata == {}

    def test_create_image_content(self) -> None:
        data = b"\x89PNG\r\n"
        content = MultiModalContent(
            modality=ModalityType.IMAGE,
            content="A photo of a cat",
            raw_data=data,
            mime_type="image/png",
            metadata={"width": 100, "height": 100},
        )
        assert content.modality == ModalityType.IMAGE
        assert content.raw_data == data
        assert content.mime_type == "image/png"
        assert content.metadata["width"] == 100

    def test_create_table_content(self) -> None:
        md_table = "| A | B |\n| --- | --- |\n| 1 | 2 |"
        content = MultiModalContent(
            modality=ModalityType.TABLE,
            content=md_table,
            metadata={"format": "markdown"},
        )
        assert content.modality == ModalityType.TABLE
        assert "| A | B |" in content.content

    def test_immutability(self) -> None:
        content = MultiModalContent(modality=ModalityType.TEXT, content="hello")
        with pytest.raises(ValidationError):
            content.content = "world"  # type: ignore[misc]

    def test_empty_content(self) -> None:
        content = MultiModalContent(modality=ModalityType.TEXT, content="")
        assert content.content == ""


class TestMultiModalItem:
    """Tests for MultiModalItem model."""

    def test_create_item(self) -> None:
        content = MultiModalContent(modality=ModalityType.TEXT, content="hello")
        item = MultiModalItem(contents=[content], source=SourceType.RETRIEVAL)
        assert len(item.contents) == 1
        assert item.source == SourceType.RETRIEVAL
        assert item.score == 0.0
        assert item.priority == 5
        assert item.token_count == 0
        assert item.metadata == {}
        # ID should be a valid UUID
        uuid.UUID(item.id)

    def test_create_with_multiple_contents(self) -> None:
        text = MultiModalContent(modality=ModalityType.TEXT, content="Description")
        image = MultiModalContent(
            modality=ModalityType.IMAGE,
            content="cat.png",
            raw_data=b"\x89PNG",
            mime_type="image/png",
        )
        item = MultiModalItem(
            contents=[text, image],
            source=SourceType.RETRIEVAL,
            score=0.9,
            priority=8,
        )
        assert len(item.contents) == 2
        assert item.contents[0].modality == ModalityType.TEXT
        assert item.contents[1].modality == ModalityType.IMAGE
        assert item.score == 0.9
        assert item.priority == 8

    def test_immutability(self) -> None:
        content = MultiModalContent(modality=ModalityType.TEXT, content="hello")
        item = MultiModalItem(contents=[content], source=SourceType.RETRIEVAL)
        with pytest.raises(ValidationError):
            item.score = 0.5  # type: ignore[misc]

    def test_score_bounds(self) -> None:
        content = MultiModalContent(modality=ModalityType.TEXT, content="hi")
        with pytest.raises(ValidationError):
            MultiModalItem(contents=[content], source=SourceType.RETRIEVAL, score=1.5)
        with pytest.raises(ValidationError):
            MultiModalItem(contents=[content], source=SourceType.RETRIEVAL, score=-0.1)

    def test_priority_bounds(self) -> None:
        content = MultiModalContent(modality=ModalityType.TEXT, content="hi")
        with pytest.raises(ValidationError):
            MultiModalItem(contents=[content], source=SourceType.RETRIEVAL, priority=0)
        with pytest.raises(ValidationError):
            MultiModalItem(contents=[content], source=SourceType.RETRIEVAL, priority=11)

    def test_token_count_non_negative(self) -> None:
        content = MultiModalContent(modality=ModalityType.TEXT, content="hi")
        with pytest.raises(ValidationError):
            MultiModalItem(contents=[content], source=SourceType.RETRIEVAL, token_count=-1)

    def test_default_created_at(self) -> None:
        content = MultiModalContent(modality=ModalityType.TEXT, content="hello")
        item = MultiModalItem(contents=[content], source=SourceType.RETRIEVAL)
        assert item.created_at is not None

    def test_custom_id(self) -> None:
        content = MultiModalContent(modality=ModalityType.TEXT, content="hello")
        item = MultiModalItem(id="custom-id", contents=[content], source=SourceType.RETRIEVAL)
        assert item.id == "custom-id"
