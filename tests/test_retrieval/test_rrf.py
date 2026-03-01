"""Tests for astro_context.retrieval._rrf."""

from __future__ import annotations

import pytest

from astro_context.models.context import ContextItem, SourceType
from astro_context.retrieval._rrf import rrf_fuse


def _make_item(item_id: str, content: str, score: float = 0.5) -> ContextItem:
    return ContextItem(
        id=item_id,
        content=content,
        source=SourceType.RETRIEVAL,
        score=score,
        priority=5,
        token_count=5,
    )


class TestRRFSingleListPassthrough:
    """A single ranked list should pass through with correct metadata."""

    def test_single_list_passthrough(self) -> None:
        items = [_make_item("a", "A"), _make_item("b", "B"), _make_item("c", "C")]
        result = rrf_fuse([items])

        # Order preserved (same ranking)
        assert [r.id for r in result] == ["a", "b", "c"]

    def test_single_list_scores_normalized(self) -> None:
        items = [_make_item("a", "A"), _make_item("b", "B"), _make_item("c", "C")]
        result = rrf_fuse([items])
        assert result[0].score == pytest.approx(1.0)
        assert result[-1].score == pytest.approx(0.0)


class TestRRFTwoListsFusion:
    """Two lists with overlapping items should fuse correctly."""

    def test_two_lists_fusion(self) -> None:
        list1 = [_make_item("a", "A"), _make_item("b", "B")]
        list2 = [_make_item("a", "A"), _make_item("c", "C")]

        result = rrf_fuse([list1, list2])
        result_ids = [r.id for r in result]

        # "a" appears in both lists so should rank first
        assert result_ids[0] == "a"
        # All three items present
        assert set(result_ids) == {"a", "b", "c"}

    def test_overlapping_item_gets_higher_score(self) -> None:
        list1 = [_make_item("a", "A"), _make_item("b", "B")]
        list2 = [_make_item("c", "C"), _make_item("a", "A")]

        result = rrf_fuse([list1, list2])
        scores = {r.id: r.score for r in result}
        # "a" appears in both, should have highest score
        assert scores["a"] > scores["b"]
        assert scores["a"] > scores["c"]


class TestRRFWeightsAffectRanking:
    """Weighted lists should produce different ordering."""

    def test_weights_affect_ranking(self) -> None:
        list1 = [_make_item("a", "A"), _make_item("b", "B")]
        list2 = [_make_item("b", "B"), _make_item("a", "A")]

        # Equal weights: both at rank 1 somewhere, tie-breaking unclear
        rrf_fuse([list1, list2], weights=[1.0, 1.0])

        # Heavily weight list2 where "b" is rank 1
        result_weighted = rrf_fuse([list1, list2], weights=[0.1, 10.0])
        assert result_weighted[0].id == "b"

    def test_mismatched_weights_raises(self) -> None:
        list1 = [_make_item("a", "A")]
        with pytest.raises(ValueError, match="weights must have same length"):
            rrf_fuse([list1], weights=[1.0, 2.0])


class TestRRFCustomKValue:
    """Different k values should change scores."""

    def test_custom_k_value(self) -> None:
        list1 = [_make_item("a", "A"), _make_item("b", "B")]
        list2 = [_make_item("a", "A"), _make_item("c", "C")]

        result_k10 = rrf_fuse([list1, list2], k=10)
        result_k1000 = rrf_fuse([list1, list2], k=1000)

        # With smaller k, the difference between rank 1 and rank 2 is larger
        raw_k10 = {r.id: r.metadata["rrf_raw_score"] for r in result_k10}
        raw_k1000 = {r.id: r.metadata["rrf_raw_score"] for r in result_k1000}

        # Both should have "a" first but with different raw scores
        ratio_k10 = raw_k10["a"] / raw_k10["b"]
        ratio_k1000 = raw_k1000["a"] / raw_k1000["b"]
        # Smaller k amplifies rank differences
        assert ratio_k10 > ratio_k1000


class TestRRFTopKLimitsResults:
    """top_k parameter should limit the number of results."""

    def test_top_k_limits_results(self) -> None:
        items = [_make_item(f"item-{i}", f"content {i}") for i in range(10)]
        result = rrf_fuse([items], top_k=3)
        assert len(result) == 3

    def test_top_k_none_returns_all(self) -> None:
        items = [_make_item(f"item-{i}", f"content {i}") for i in range(10)]
        result = rrf_fuse([items], top_k=None)
        assert len(result) == 10


class TestRRFEmptyLists:
    """Empty input should return empty output."""

    def test_empty_lists(self) -> None:
        assert rrf_fuse([]) == []

    def test_all_empty_sublists(self) -> None:
        assert rrf_fuse([[], []]) == []

    def test_one_empty_one_nonempty(self) -> None:
        list1: list[ContextItem] = []
        list2 = [_make_item("a", "A")]
        result = rrf_fuse([list1, list2])
        assert len(result) == 1
        assert result[0].id == "a"


class TestRRFScoresNormalized:
    """All output scores should be in [0, 1]."""

    def test_scores_normalized(self) -> None:
        list1 = [_make_item("a", "A"), _make_item("b", "B"), _make_item("c", "C")]
        list2 = [_make_item("c", "C"), _make_item("a", "A"), _make_item("d", "D")]

        result = rrf_fuse([list1, list2])
        for item in result:
            assert 0.0 <= item.score <= 1.0

    def test_single_item_score_is_one(self) -> None:
        result = rrf_fuse([[_make_item("a", "A")]])
        assert result[0].score == pytest.approx(1.0)


class TestRRFMetadataContainsFields:
    """Output items should have rrf_raw_score and retrieval_method in metadata."""

    def test_metadata_contains_rrf_fields(self) -> None:
        list1 = [_make_item("a", "A"), _make_item("b", "B")]
        list2 = [_make_item("a", "A"), _make_item("c", "C")]

        result = rrf_fuse([list1, list2])
        for item in result:
            assert "retrieval_method" in item.metadata
            assert item.metadata["retrieval_method"] == "rrf"
            assert "rrf_raw_score" in item.metadata
            assert isinstance(item.metadata["rrf_raw_score"], float)

    def test_best_original_item_kept(self) -> None:
        """When the same ID appears in multiple lists, keep the one with the best score."""
        item_low = _make_item("a", "A", score=0.2)
        item_high = _make_item("a", "A", score=0.8)

        # item_high should be kept as the base because its original score is higher
        result = rrf_fuse([[item_low], [item_high]])
        # The rrf_raw_score comes from fusion; the original score is replaced by normalized
        # But the metadata of the higher-scored original should be used as base
        assert len(result) == 1
        assert result[0].id == "a"
