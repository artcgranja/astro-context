"""Tests for multi-modal converter utilities."""

from __future__ import annotations

from astro_context.models.context import ContextItem, SourceType
from astro_context.multimodal.converter import MultiModalConverter
from astro_context.multimodal.encoders import CompositeEncoder
from astro_context.multimodal.models import ModalityType, MultiModalContent, MultiModalItem


class TestMultiModalConverterToContextItem:
    """Tests for MultiModalConverter.to_context_item."""

    def test_single_text_content(self) -> None:
        content = MultiModalContent(modality=ModalityType.TEXT, content="hello world")
        item = MultiModalItem(
            id="test-1",
            contents=[content],
            source=SourceType.RETRIEVAL,
            score=0.8,
            priority=7,
            token_count=2,
        )
        encoder = CompositeEncoder()
        result = MultiModalConverter.to_context_item(item, encoder)

        assert isinstance(result, ContextItem)
        assert result.id == "test-1"
        assert result.content == "hello world"
        assert result.source == SourceType.RETRIEVAL
        assert result.score == 0.8
        assert result.priority == 7
        assert result.token_count == 2
        assert result.metadata["multimodal"] is True

    def test_multiple_contents(self) -> None:
        text = MultiModalContent(modality=ModalityType.TEXT, content="Description of data")
        table = MultiModalContent(
            modality=ModalityType.TABLE,
            content="| A | B |\n| --- | --- |\n| 1 | 2 |",
        )
        item = MultiModalItem(
            contents=[text, table],
            source=SourceType.RETRIEVAL,
        )
        encoder = CompositeEncoder()
        result = MultiModalConverter.to_context_item(item, encoder)

        assert "Description of data" in result.content
        assert "| A | B |" in result.content
        assert "\n\n" in result.content

    def test_preserves_created_at(self) -> None:
        content = MultiModalContent(modality=ModalityType.TEXT, content="test")
        item = MultiModalItem(contents=[content], source=SourceType.RETRIEVAL)
        encoder = CompositeEncoder()
        result = MultiModalConverter.to_context_item(item, encoder)
        assert result.created_at == item.created_at


class TestMultiModalConverterToContextItems:
    """Tests for MultiModalConverter.to_context_items (batch)."""

    def test_batch_conversion(self) -> None:
        items = [
            MultiModalItem(
                contents=[MultiModalContent(modality=ModalityType.TEXT, content=f"item {i}")],
                source=SourceType.RETRIEVAL,
            )
            for i in range(3)
        ]
        encoder = CompositeEncoder()
        results = MultiModalConverter.to_context_items(items, encoder)
        assert len(results) == 3
        assert all(isinstance(r, ContextItem) for r in results)
        assert results[0].content == "item 0"
        assert results[2].content == "item 2"

    def test_empty_list(self) -> None:
        encoder = CompositeEncoder()
        results = MultiModalConverter.to_context_items([], encoder)
        assert results == []


class TestMultiModalConverterFromContextItem:
    """Tests for MultiModalConverter.from_context_item."""

    def test_from_context_item_default_text(self) -> None:
        ctx = ContextItem(
            id="ctx-1",
            content="hello world",
            source=SourceType.RETRIEVAL,
            score=0.7,
            priority=6,
            token_count=2,
        )
        result = MultiModalConverter.from_context_item(ctx)

        assert isinstance(result, MultiModalItem)
        assert result.id == "ctx-1"
        assert len(result.contents) == 1
        assert result.contents[0].modality == ModalityType.TEXT
        assert result.contents[0].content == "hello world"
        assert result.source == SourceType.RETRIEVAL
        assert result.score == 0.7
        assert result.priority == 6
        assert result.token_count == 2

    def test_from_context_item_with_custom_modality(self) -> None:
        ctx = ContextItem(
            content="SELECT * FROM table",
            source=SourceType.RETRIEVAL,
        )
        result = MultiModalConverter.from_context_item(ctx, modality=ModalityType.CODE)
        assert result.contents[0].modality == ModalityType.CODE

    def test_roundtrip_text(self) -> None:
        """Converting ContextItem -> MultiModalItem -> ContextItem preserves content."""
        original = ContextItem(
            id="rt-1",
            content="round trip test",
            source=SourceType.RETRIEVAL,
            score=0.5,
            priority=5,
            token_count=3,
        )
        mm = MultiModalConverter.from_context_item(original)
        encoder = CompositeEncoder()
        result = MultiModalConverter.to_context_item(mm, encoder)

        assert result.id == original.id
        assert result.content == original.content
        assert result.source == original.source
        assert result.score == original.score
        assert result.priority == original.priority
        assert result.token_count == original.token_count
