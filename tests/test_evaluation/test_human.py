"""Tests for human-in-the-loop evaluation."""

from __future__ import annotations

import pytest

from astro_context.evaluation.ab_testing import EvaluationDataset
from astro_context.evaluation.human import HumanEvaluationCollector, HumanJudgment
from astro_context.protocols.evaluation import HumanEvaluator


class TestHumanJudgment:
    """Tests for HumanJudgment model."""

    def test_human_judgment_frozen(self) -> None:
        j = HumanJudgment(query="q", item_id="d1", relevance=2, annotator="alice")
        with pytest.raises(Exception):  # noqa: B017
            j.relevance = 3  # type: ignore[misc]

    def test_relevance_validation(self) -> None:
        """Relevance must be 0-3."""
        # Valid boundary values
        HumanJudgment(query="q", item_id="d1", relevance=0, annotator="a")
        HumanJudgment(query="q", item_id="d1", relevance=3, annotator="a")

        # Invalid values
        with pytest.raises(Exception):  # noqa: B017
            HumanJudgment(query="q", item_id="d1", relevance=-1, annotator="a")
        with pytest.raises(Exception):  # noqa: B017
            HumanJudgment(query="q", item_id="d1", relevance=4, annotator="a")


class TestHumanEvaluationCollector:
    """Tests for HumanEvaluationCollector."""

    def test_add_judgment(self) -> None:
        collector = HumanEvaluationCollector()
        j = HumanJudgment(query="q1", item_id="d1", relevance=2, annotator="alice")
        collector.add_judgment(j)
        assert len(collector.judgments) == 1
        assert collector.judgments[0] == j

    def test_add_judgments_bulk(self) -> None:
        collector = HumanEvaluationCollector()
        judgments = [
            HumanJudgment(query="q1", item_id="d1", relevance=2, annotator="alice"),
            HumanJudgment(query="q1", item_id="d2", relevance=1, annotator="alice"),
            HumanJudgment(query="q2", item_id="d1", relevance=3, annotator="bob"),
        ]
        collector.add_judgments(judgments)
        assert len(collector.judgments) == 3

    def test_judgments_property_returns_copy(self) -> None:
        collector = HumanEvaluationCollector()
        j = HumanJudgment(query="q1", item_id="d1", relevance=2, annotator="alice")
        collector.add_judgment(j)

        copy = collector.judgments
        copy.append(HumanJudgment(query="q2", item_id="d2", relevance=0, annotator="bob"))
        # Original should be unchanged
        assert len(collector.judgments) == 1

    def test_compute_agreement_perfect(self) -> None:
        """Two annotators agree on every judgment -> kappa = 1.0."""
        collector = HumanEvaluationCollector()
        collector.add_judgments(
            [
                HumanJudgment(query="q1", item_id="d1", relevance=3, annotator="alice"),
                HumanJudgment(query="q1", item_id="d1", relevance=3, annotator="bob"),
                HumanJudgment(query="q1", item_id="d2", relevance=0, annotator="alice"),
                HumanJudgment(query="q1", item_id="d2", relevance=0, annotator="bob"),
                HumanJudgment(query="q2", item_id="d1", relevance=2, annotator="alice"),
                HumanJudgment(query="q2", item_id="d1", relevance=2, annotator="bob"),
            ]
        )

        kappa = collector.compute_agreement()
        assert kappa == pytest.approx(1.0)

    def test_compute_agreement_none(self) -> None:
        """Single annotator -> agreement = 0.0."""
        collector = HumanEvaluationCollector()
        collector.add_judgments(
            [
                HumanJudgment(query="q1", item_id="d1", relevance=3, annotator="alice"),
                HumanJudgment(query="q1", item_id="d2", relevance=1, annotator="alice"),
            ]
        )

        assert collector.compute_agreement() == 0.0

    def test_compute_agreement_partial(self) -> None:
        """Partial agreement -> 0 < kappa < 1."""
        collector = HumanEvaluationCollector()
        collector.add_judgments(
            [
                HumanJudgment(query="q1", item_id="d1", relevance=3, annotator="alice"),
                HumanJudgment(query="q1", item_id="d1", relevance=3, annotator="bob"),
                HumanJudgment(query="q1", item_id="d2", relevance=2, annotator="alice"),
                HumanJudgment(query="q1", item_id="d2", relevance=1, annotator="bob"),
                HumanJudgment(query="q2", item_id="d1", relevance=0, annotator="alice"),
                HumanJudgment(query="q2", item_id="d1", relevance=0, annotator="bob"),
                HumanJudgment(query="q2", item_id="d2", relevance=1, annotator="alice"),
                HumanJudgment(query="q2", item_id="d2", relevance=2, annotator="bob"),
            ]
        )

        kappa = collector.compute_agreement()
        assert 0.0 < kappa < 1.0

    def test_to_dataset(self) -> None:
        """Converts judgments to an EvaluationDataset correctly."""
        collector = HumanEvaluationCollector()
        collector.add_judgments(
            [
                HumanJudgment(query="q1", item_id="d1", relevance=3, annotator="alice"),
                HumanJudgment(query="q1", item_id="d2", relevance=1, annotator="alice"),
                HumanJudgment(query="q1", item_id="d3", relevance=2, annotator="alice"),
            ]
        )

        ds = collector.to_dataset(threshold=2)
        assert isinstance(ds, EvaluationDataset)
        assert len(ds.samples) == 1
        assert ds.samples[0].query == "q1"
        # d1 (3 >= 2) and d3 (2 >= 2) should be relevant; d2 (1 < 2) should not
        assert sorted(ds.samples[0].relevant_ids) == ["d1", "d3"]

    def test_to_dataset_threshold(self) -> None:
        """Threshold controls which items are considered relevant."""
        collector = HumanEvaluationCollector()
        collector.add_judgments(
            [
                HumanJudgment(query="q1", item_id="d1", relevance=1, annotator="alice"),
                HumanJudgment(query="q1", item_id="d2", relevance=2, annotator="alice"),
                HumanJudgment(query="q1", item_id="d3", relevance=3, annotator="alice"),
            ]
        )

        # threshold=1: d1 (1 >= 1), d2 (2 >= 1), d3 (3 >= 1) -> all relevant
        ds = collector.to_dataset(threshold=1)
        assert sorted(ds.samples[0].relevant_ids) == ["d1", "d2", "d3"]

        # threshold=3: only d3 (3 >= 3) is relevant
        ds = collector.to_dataset(threshold=3)
        assert ds.samples[0].relevant_ids == ["d3"]

    def test_compute_metrics(self) -> None:
        """Returns expected keys and values."""
        collector = HumanEvaluationCollector()
        collector.add_judgments(
            [
                HumanJudgment(query="q1", item_id="d1", relevance=3, annotator="alice"),
                HumanJudgment(query="q1", item_id="d1", relevance=2, annotator="bob"),
                HumanJudgment(query="q2", item_id="d2", relevance=1, annotator="alice"),
            ]
        )

        metrics = collector.compute_metrics()
        assert "mean_relevance" in metrics
        assert "agreement" in metrics
        assert "num_judgments" in metrics
        assert "num_annotators" in metrics
        assert "num_queries" in metrics
        assert metrics["mean_relevance"] == pytest.approx(2.0)
        assert metrics["num_judgments"] == 3.0
        assert metrics["num_annotators"] == 2.0
        assert metrics["num_queries"] == 2.0

    def test_compute_metrics_empty(self) -> None:
        collector = HumanEvaluationCollector()
        metrics = collector.compute_metrics()
        assert metrics["mean_relevance"] == 0.0
        assert metrics["num_judgments"] == 0.0

    def test_repr(self) -> None:
        collector = HumanEvaluationCollector()
        collector.add_judgment(
            HumanJudgment(query="q1", item_id="d1", relevance=2, annotator="alice")
        )
        r = repr(collector)
        assert "HumanEvaluationCollector" in r
        assert "judgments=1" in r

    def test_protocol_compliance(self) -> None:
        """HumanEvaluationCollector satisfies HumanEvaluator protocol."""
        collector = HumanEvaluationCollector()
        assert isinstance(collector, HumanEvaluator)

    # ------------------------------------------------------------------
    # Additional targeted tests (Batch E verification)
    # ------------------------------------------------------------------

    def test_compute_agreement_three_annotators_pairwise(self) -> None:
        """With 3+ annotators, compute_agreement uses first two per pair."""
        collector = HumanEvaluationCollector()
        # Three annotators judge the same items
        for item_id in ["d1", "d2", "d3"]:
            collector.add_judgment(
                HumanJudgment(query="q1", item_id=item_id, relevance=3, annotator="alice")
            )
            collector.add_judgment(
                HumanJudgment(query="q1", item_id=item_id, relevance=3, annotator="bob")
            )
            collector.add_judgment(
                HumanJudgment(query="q1", item_id=item_id, relevance=3, annotator="carol")
            )

        # All three agree on relevance=3; the pairwise kappa (first two)
        # should be 1.0 since alice and bob always agree
        kappa = collector.compute_agreement()
        assert kappa == pytest.approx(1.0)

    def test_compute_agreement_all_disagree(self) -> None:
        """When annotators systematically disagree, kappa should be <= 0."""
        collector = HumanEvaluationCollector()
        # Alice says 0, Bob says 3 for every pair -- maximum disagreement
        for i in range(10):
            collector.add_judgment(
                HumanJudgment(query=f"q{i}", item_id="d1", relevance=0, annotator="alice")
            )
            collector.add_judgment(
                HumanJudgment(query=f"q{i}", item_id="d1", relevance=3, annotator="bob")
            )

        kappa = collector.compute_agreement()
        # Complete disagreement: observed agreement = 0, so kappa <= 0
        assert kappa <= 0.0

    def test_relevance_boundary_zero(self) -> None:
        """Relevance=0 should be valid and treated as not relevant."""
        collector = HumanEvaluationCollector()
        collector.add_judgment(
            HumanJudgment(query="q1", item_id="d1", relevance=0, annotator="alice")
        )

        ds = collector.to_dataset(threshold=1)
        # relevance 0 < threshold 1, so d1 is not relevant
        assert ds.samples[0].relevant_ids == []

    def test_relevance_boundary_three(self) -> None:
        """Relevance=3 should always pass any threshold <= 3."""
        collector = HumanEvaluationCollector()
        collector.add_judgment(
            HumanJudgment(query="q1", item_id="d1", relevance=3, annotator="alice")
        )

        ds = collector.to_dataset(threshold=3)
        assert ds.samples[0].relevant_ids == ["d1"]

    def test_to_dataset_no_relevant_items(self) -> None:
        """When all items are below threshold, relevant_ids should be empty."""
        collector = HumanEvaluationCollector()
        collector.add_judgments(
            [
                HumanJudgment(query="q1", item_id="d1", relevance=0, annotator="alice"),
                HumanJudgment(query="q1", item_id="d2", relevance=1, annotator="alice"),
                HumanJudgment(query="q2", item_id="d3", relevance=0, annotator="bob"),
            ]
        )

        ds = collector.to_dataset(threshold=2)
        assert len(ds.samples) == 2
        for sample in ds.samples:
            assert sample.relevant_ids == []

    def test_to_dataset_all_relevant_items(self) -> None:
        """When all items are at or above threshold, all are relevant."""
        collector = HumanEvaluationCollector()
        collector.add_judgments(
            [
                HumanJudgment(query="q1", item_id="d1", relevance=3, annotator="alice"),
                HumanJudgment(query="q1", item_id="d2", relevance=2, annotator="alice"),
                HumanJudgment(query="q1", item_id="d3", relevance=3, annotator="alice"),
            ]
        )

        ds = collector.to_dataset(threshold=2)
        assert len(ds.samples) == 1
        assert sorted(ds.samples[0].relevant_ids) == ["d1", "d2", "d3"]

    def test_large_number_of_judgments(self) -> None:
        """Collector handles 100+ judgments without error."""
        collector = HumanEvaluationCollector()
        judgments = []
        for i in range(120):
            judgments.append(
                HumanJudgment(
                    query=f"q{i % 20}",
                    item_id=f"d{i % 10}",
                    relevance=i % 4,
                    annotator=f"annotator_{i % 3}",
                )
            )
        collector.add_judgments(judgments)

        assert len(collector.judgments) == 120

        metrics = collector.compute_metrics()
        assert metrics["num_judgments"] == 120.0
        assert metrics["num_annotators"] == 3.0
        assert metrics["num_queries"] == 20.0

        # Agreement should be computable
        kappa = collector.compute_agreement()
        assert isinstance(kappa, float)

        # Dataset conversion should work
        ds = collector.to_dataset(threshold=2)
        assert isinstance(ds, EvaluationDataset)
        assert len(ds.samples) == 20  # 20 unique queries

    def test_compute_metrics_all_keys_present(self) -> None:
        """Verify compute_metrics returns exactly the documented keys."""
        collector = HumanEvaluationCollector()
        collector.add_judgment(
            HumanJudgment(query="q1", item_id="d1", relevance=2, annotator="alice")
        )

        metrics = collector.compute_metrics()
        expected_keys = {
            "mean_relevance",
            "agreement",
            "num_judgments",
            "num_annotators",
            "num_queries",
        }
        assert set(metrics.keys()) == expected_keys
        # All values should be floats
        for key, val in metrics.items():
            assert isinstance(val, float), f"{key} should be float, got {type(val)}"

    def test_compute_metrics_empty_all_keys_present(self) -> None:
        """Even with no judgments, all keys should be present."""
        collector = HumanEvaluationCollector()
        metrics = collector.compute_metrics()
        expected_keys = {
            "mean_relevance",
            "agreement",
            "num_judgments",
            "num_annotators",
            "num_queries",
        }
        assert set(metrics.keys()) == expected_keys
