"""Tests for astro_context._math module (cosine_similarity and clamp)."""

from __future__ import annotations

import pytest

from astro_context._math import clamp, cosine_similarity

# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    """Tests for the cosine_similarity function."""

    def test_matching_vectors_returns_approximately_one(self) -> None:
        vec = [1.0, 2.0, 3.0, 4.0]
        assert cosine_similarity(vec, vec) == pytest.approx(1.0)

    def test_matching_unit_vectors(self) -> None:
        vec = [0.0, 1.0, 0.0]
        assert cosine_similarity(vec, vec) == pytest.approx(1.0)

    def test_orthogonal_vectors_returns_zero(self) -> None:
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_orthogonal_vectors_higher_dim(self) -> None:
        a = [1.0, 0.0, 0.0, 0.0]
        b = [0.0, 0.0, 0.0, 1.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors_returns_negative_one(self) -> None:
        a = [1.0, 2.0, 3.0]
        b = [-1.0, -2.0, -3.0]
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_dimension_mismatch_raises_value_error(self) -> None:
        a = [1.0, 2.0, 3.0]
        b = [1.0, 2.0]
        with pytest.raises(ValueError, match="same dimensionality"):
            cosine_similarity(a, b)

    def test_empty_vectors_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            cosine_similarity([], [])

    def test_zero_norm_vector_returns_zero(self) -> None:
        a = [0.0, 0.0, 0.0]
        b = [1.0, 2.0, 3.0]
        assert cosine_similarity(a, b) == 0.0

    def test_both_zero_norm_vectors_returns_zero(self) -> None:
        a = [0.0, 0.0, 0.0]
        b = [0.0, 0.0, 0.0]
        assert cosine_similarity(a, b) == 0.0

    def test_single_element_vectors(self) -> None:
        assert cosine_similarity([3.0], [5.0]) == pytest.approx(1.0)
        assert cosine_similarity([3.0], [-5.0]) == pytest.approx(-1.0)

    def test_result_is_clamped_to_valid_range(self) -> None:
        """Even with floating point noise, result should be in [-1.0, 1.0]."""
        a = [1e15, 1e15, 1e15]
        b = [1e15, 1e15, 1e15]
        result = cosine_similarity(a, b)
        assert -1.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# clamp
# ---------------------------------------------------------------------------


class TestClamp:
    """Tests for the clamp function."""

    def test_value_within_range_unchanged(self) -> None:
        assert clamp(0.5) == 0.5

    def test_value_below_range_clamped_to_lo(self) -> None:
        assert clamp(-0.5) == 0.0

    def test_value_above_range_clamped_to_hi(self) -> None:
        assert clamp(1.5) == 1.0

    def test_value_at_lower_boundary(self) -> None:
        assert clamp(0.0) == 0.0

    def test_value_at_upper_boundary(self) -> None:
        assert clamp(1.0) == 1.0

    def test_custom_range_within(self) -> None:
        assert clamp(5.0, lo=0.0, hi=10.0) == 5.0

    def test_custom_range_below(self) -> None:
        assert clamp(-5.0, lo=0.0, hi=10.0) == 0.0

    def test_custom_range_above(self) -> None:
        assert clamp(15.0, lo=0.0, hi=10.0) == 10.0

    def test_negative_range(self) -> None:
        assert clamp(-0.5, lo=-1.0, hi=0.0) == -0.5
        assert clamp(-2.0, lo=-1.0, hi=0.0) == -1.0
        assert clamp(1.0, lo=-1.0, hi=0.0) == 0.0
