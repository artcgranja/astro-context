"""Tests for astro_context.memory.decay."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from astro_context.memory.decay import (
    EbbinghausDecay,
    ExponentialRecencyScorer,
    LinearDecay,
    LinearRecencyScorer,
)
from astro_context.models.memory import MemoryEntry


def _make_entry(
    hours_ago: float = 0.0,
    access_count: int = 0,
) -> MemoryEntry:
    """Create a MemoryEntry with a last_accessed time *hours_ago* hours in the past."""
    last_accessed = datetime.now(UTC) - timedelta(hours=hours_ago)
    return MemoryEntry(
        content="test content",
        last_accessed=last_accessed,
        access_count=access_count,
    )


class TestEbbinghausDecay:
    """Ebbinghaus forgetting curve retention."""

    def test_fresh_entry_high_retention(self) -> None:
        decay = EbbinghausDecay()
        entry = _make_entry(hours_ago=0.0)
        retention = decay.compute_retention(entry)
        # Just accessed -> retention should be very close to 1.0
        assert retention > 0.95

    def test_old_entry_low_retention(self) -> None:
        decay = EbbinghausDecay(base_strength=1.0)
        entry = _make_entry(hours_ago=24.0, access_count=0)
        retention = decay.compute_retention(entry)
        # 24 hours with strength 1.0: e^(-24/1) ~ 0.0
        assert retention < 0.01

    def test_high_access_count_slower_decay(self) -> None:
        decay = EbbinghausDecay(base_strength=1.0, reinforcement_factor=0.5)
        entry_low_access = _make_entry(hours_ago=5.0, access_count=0)
        entry_high_access = _make_entry(hours_ago=5.0, access_count=10)
        retention_low = decay.compute_retention(entry_low_access)
        retention_high = decay.compute_retention(entry_high_access)
        # More accesses -> higher strength -> slower decay -> higher retention
        assert retention_high > retention_low

    def test_retention_clamped_to_zero_one(self) -> None:
        decay = EbbinghausDecay()
        # Very old entry
        entry = _make_entry(hours_ago=10000.0, access_count=0)
        retention = decay.compute_retention(entry)
        assert 0.0 <= retention <= 1.0
        # Very fresh entry
        entry_fresh = _make_entry(hours_ago=0.0, access_count=100)
        retention_fresh = decay.compute_retention(entry_fresh)
        assert 0.0 <= retention_fresh <= 1.0

    def test_base_strength_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="base_strength"):
            EbbinghausDecay(base_strength=0)

    def test_reinforcement_factor_must_be_non_negative(self) -> None:
        with pytest.raises(ValueError, match="reinforcement_factor"):
            EbbinghausDecay(reinforcement_factor=-1)


class TestLinearDecay:
    """Linear decay from 1.0 to 0.0 over twice the half-life."""

    def test_at_half_life_returns_approximately_half(self) -> None:
        decay = LinearDecay(half_life_hours=10.0)
        entry = _make_entry(hours_ago=10.0)
        retention = decay.compute_retention(entry)
        assert abs(retention - 0.5) < 0.05

    def test_beyond_twice_half_life_returns_zero(self) -> None:
        decay = LinearDecay(half_life_hours=10.0)
        entry = _make_entry(hours_ago=25.0)  # > 2x half_life
        retention = decay.compute_retention(entry)
        assert retention == 0.0

    def test_fresh_entry_returns_approximately_one(self) -> None:
        decay = LinearDecay(half_life_hours=168.0)
        entry = _make_entry(hours_ago=0.0)
        retention = decay.compute_retention(entry)
        assert retention > 0.99

    def test_retention_clamped_at_boundaries(self) -> None:
        decay = LinearDecay(half_life_hours=1.0)
        # Way past the decay window
        entry = _make_entry(hours_ago=1000.0)
        assert decay.compute_retention(entry) == 0.0
        # Just created
        entry_fresh = _make_entry(hours_ago=0.0)
        assert 0.0 <= decay.compute_retention(entry_fresh) <= 1.0

    def test_half_life_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="half_life_hours"):
            LinearDecay(half_life_hours=0)

    def test_half_life_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="half_life_hours"):
            LinearDecay(half_life_hours=-5)


class TestExponentialRecencyScorer:
    """Exponential recency scoring."""

    def test_oldest_item_gets_lowest_score(self) -> None:
        scorer = ExponentialRecencyScorer()
        score_oldest = scorer.score(index=0, total=10)
        score_middle = scorer.score(index=5, total=10)
        assert score_oldest < score_middle

    def test_newest_item_gets_one(self) -> None:
        scorer = ExponentialRecencyScorer()
        score = scorer.score(index=9, total=10)
        assert abs(score - 1.0) < 1e-9

    def test_scores_are_monotonically_increasing(self) -> None:
        scorer = ExponentialRecencyScorer()
        total = 20
        scores = [scorer.score(i, total) for i in range(total)]
        for i in range(1, len(scores)):
            assert scores[i] >= scores[i - 1]

    def test_single_item_returns_one(self) -> None:
        scorer = ExponentialRecencyScorer()
        assert scorer.score(0, 1) == 1.0

    def test_oldest_of_many_close_to_zero(self) -> None:
        scorer = ExponentialRecencyScorer(decay_rate=2.0)
        score = scorer.score(0, 100)
        # With exponential curve and rate=2.0, the oldest score should be very low
        assert score < 0.05

    def test_decay_rate_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="decay_rate"):
            ExponentialRecencyScorer(decay_rate=0)

    def test_scores_in_range_zero_to_one(self) -> None:
        scorer = ExponentialRecencyScorer(decay_rate=3.0)
        for i in range(50):
            s = scorer.score(i, 50)
            assert 0.0 <= s <= 1.0


class TestLinearRecencyScorer:
    """Linear recency scoring."""

    def test_matches_expected_formula(self) -> None:
        scorer = LinearRecencyScorer(min_score=0.5)
        # index=0, total=10: 0.5 + 0.5*(0/9) = 0.5
        score_oldest = scorer.score(0, 10)
        assert abs(score_oldest - 0.5) < 1e-9
        # index=9, total=10: 0.5 + 0.5*(9/9) = 1.0
        score_newest = scorer.score(9, 10)
        assert abs(score_newest - 1.0) < 1e-9

    def test_custom_min_score(self) -> None:
        scorer = LinearRecencyScorer(min_score=0.2)
        score_oldest = scorer.score(0, 10)
        assert abs(score_oldest - 0.2) < 1e-9
        score_newest = scorer.score(9, 10)
        assert abs(score_newest - 1.0) < 1e-9

    def test_single_item_returns_one(self) -> None:
        scorer = LinearRecencyScorer()
        assert scorer.score(0, 1) == 1.0

    def test_min_score_at_zero(self) -> None:
        scorer = LinearRecencyScorer(min_score=0.0)
        assert abs(scorer.score(0, 5) - 0.0) < 1e-9
        assert abs(scorer.score(4, 5) - 1.0) < 1e-9

    def test_invalid_min_score_raises(self) -> None:
        with pytest.raises(ValueError, match="min_score"):
            LinearRecencyScorer(min_score=1.0)
        with pytest.raises(ValueError, match="min_score"):
            LinearRecencyScorer(min_score=-0.1)

    def test_scores_are_monotonically_increasing(self) -> None:
        scorer = LinearRecencyScorer(min_score=0.3)
        total = 15
        scores = [scorer.score(i, total) for i in range(total)]
        for i in range(1, len(scores)):
            assert scores[i] >= scores[i - 1]
