# Evaluation API Reference

All classes are importable from `astro_context.evaluation`.
For usage examples see the [Evaluation Guide](../guides/evaluation.md).

---

## RetrievalMetrics

Immutable Pydantic model holding metrics for a single retrieval evaluation.
All values are bounded in `[0.0, 1.0]`.

```python
from astro_context.evaluation import RetrievalMetrics
```

| Field | Type | Description |
|-------|------|-------------|
| `precision_at_k` | `float` | Fraction of retrieved items that are relevant |
| `recall_at_k` | `float` | Fraction of relevant items that were retrieved |
| `f1_at_k` | `float` | Harmonic mean of precision and recall |
| `mrr` | `float` | Reciprocal of the rank of the first relevant item |
| `ndcg` | `float` | Normalized Discounted Cumulative Gain |
| `hit_rate` | `float` | 1.0 if at least one relevant item was retrieved |

---

## RAGMetrics

Immutable Pydantic model for RAGAS-style RAG evaluation. All values bounded in `[0.0, 1.0]`.

```python
from astro_context.evaluation import RAGMetrics
```

| Field | Type | Description |
|-------|------|-------------|
| `faithfulness` | `float` | How faithful the answer is to the provided contexts |
| `answer_relevancy` | `float` | How relevant the answer is to the query |
| `context_precision` | `float` | Precision of retrieved contexts for the query |
| `context_recall` | `float` | Recall of retrieved contexts against ground truth |

---

## EvaluationResult

Combines retrieval and RAG metrics into a single evaluation result.

```python
from astro_context.evaluation import EvaluationResult
```

| Field | Type | Description |
|-------|------|-------------|
| `retrieval_metrics` | `RetrievalMetrics \| None` | Retrieval metrics, if computed |
| `rag_metrics` | `RAGMetrics \| None` | RAG metrics, if computed |
| `metadata` | `dict[str, Any]` | Arbitrary metadata for the run |

---

## RetrievalMetricsCalculator

Computes standard IR metrics from ranked results and a known set of relevant
document IDs. No LLM dependencies.

```python
from astro_context.evaluation import RetrievalMetricsCalculator
```

**Constructor:**

```python
RetrievalMetricsCalculator(k: int = 10)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k` | `int` | `10` | Default cutoff for top-k evaluation |

!!! warning
    Raises `ValueError` if `k < 1`.

**`evaluate(retrieved, relevant, k=None) -> RetrievalMetrics`**

```python
def evaluate(
    self,
    retrieved: list[ContextItem],
    relevant: list[str],
    k: int | None = None,
) -> RetrievalMetrics
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `retrieved` | `list[ContextItem]` | -- | Items in ranked order |
| `relevant` | `list[str]` | -- | Ground-truth relevant document IDs |
| `k` | `int \| None` | `None` | Cutoff override; falls back to instance default |

```python
from astro_context.evaluation import RetrievalMetricsCalculator
from astro_context.models.context import ContextItem, SourceType

calc = RetrievalMetricsCalculator(k=5)
items = [ContextItem(id="a", content="x", source=SourceType.RETRIEVAL)]
metrics = calc.evaluate(items, relevant=["a", "b"], k=3)
print(metrics.precision_at_k)  # 1.0
```

---

## LLMRAGEvaluator

RAGAS-style RAG evaluator driven by user-supplied callback functions. Each
callback returns a float in `[0.0, 1.0]`.

```python
from astro_context.evaluation import LLMRAGEvaluator
```

**Constructor:**

```python
LLMRAGEvaluator(
    *,
    faithfulness_fn: Callable[[str, list[str]], float] | None = None,
    relevancy_fn: Callable[[str, str], float] | None = None,
    precision_fn: Callable[[str, list[str]], float] | None = None,
    recall_fn: Callable[[str, list[str], str], float] | None = None,
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `faithfulness_fn` | `(answer, contexts) -> float` | Grounding check |
| `relevancy_fn` | `(query, answer) -> float` | Relevance check |
| `precision_fn` | `(query, contexts) -> float` | Context precision |
| `recall_fn` | `(query, contexts, ground_truth) -> float` | Context recall |

**`evaluate(query, answer, contexts, ground_truth=None) -> RAGMetrics`**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | -- | The original user query |
| `answer` | `str` | -- | The generated answer |
| `contexts` | `list[str]` | -- | Context strings fed to the generator |
| `ground_truth` | `str \| None` | `None` | Reference answer for recall |

Dimensions without registered callbacks return `0.0`.

---

## PipelineEvaluator

Orchestrates retrieval and RAG evaluation into a single result.

```python
from astro_context.evaluation import PipelineEvaluator
```

**Constructor:**

```python
PipelineEvaluator(
    *,
    retrieval_calculator: RetrievalMetricsCalculator | None = None,
    rag_evaluator: LLMRAGEvaluator | None = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `retrieval_calculator` | `RetrievalMetricsCalculator \| None` | `None` | Defaults to a new instance |
| `rag_evaluator` | `LLMRAGEvaluator \| None` | `None` | Optional LLM-based evaluator |

**`evaluate_retrieval(retrieved, relevant, k=10) -> RetrievalMetrics`** -- evaluates retrieval only.

**`evaluate_rag(query, answer, contexts, ground_truth=None) -> RAGMetrics`** -- evaluates RAG only.

!!! warning
    Raises `ValueError` if no `rag_evaluator` was configured.

**`evaluate(...) -> EvaluationResult`** -- runs both evaluations.

```python
def evaluate(
    self,
    query: str,
    answer: str,
    retrieved: list[ContextItem],
    relevant: list[str],
    contexts: list[str],
    ground_truth: str | None = None,
    k: int = 10,
    metadata: dict[str, Any] | None = None,
) -> EvaluationResult
```

---

## EvaluationSample (A/B testing)

A single evaluation sample. Defined in `astro_context.evaluation.ab_testing`.

```python
from astro_context.evaluation import EvaluationSample
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `query` | `str` | -- | The query string |
| `relevant_ids` | `list[str]` | `[]` | Relevant document IDs |
| `metadata` | `dict[str, Any]` | `{}` | Arbitrary metadata |

## EvaluationDataset (A/B testing)

A collection of `EvaluationSample` instances.

```python
from astro_context.evaluation import EvaluationDataset
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `samples` | `list[EvaluationSample]` | `[]` | The evaluation samples |
| `name` | `str` | `""` | Optional dataset name |
| `metadata` | `dict[str, Any]` | `{}` | Arbitrary metadata |

## AggregatedMetrics (A/B testing)

Aggregated retrieval metrics across multiple evaluation samples.

```python
from astro_context.evaluation import AggregatedMetrics
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mean_precision` | `float` | `0.0` | Mean precision@k |
| `mean_recall` | `float` | `0.0` | Mean recall@k |
| `mean_f1` | `float` | `0.0` | Mean F1@k |
| `mean_mrr` | `float` | `0.0` | Mean MRR |
| `mean_ndcg` | `float` | `0.0` | Mean NDCG |
| `num_samples` | `int` | `0` | Number of samples evaluated |

---

## ABTestResult

Result of an A/B test comparing two retrievers.

```python
from astro_context.evaluation import ABTestResult
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `metrics_a` | `AggregatedMetrics` | -- | Metrics for retriever A |
| `metrics_b` | `AggregatedMetrics` | -- | Metrics for retriever B |
| `winner` | `str` | -- | `"a"`, `"b"`, or `"tie"` |
| `p_value` | `float` | -- | Paired t-test p-value |
| `is_significant` | `bool` | -- | Whether the result is statistically significant |
| `significance_level` | `float` | `0.05` | Threshold for significance |
| `per_metric_comparison` | `dict[str, dict[str, Any]]` | `{}` | Per-metric deltas |
| `metadata` | `dict[str, Any]` | `{}` | Arbitrary metadata |

---

## ABTestRunner

Runs an A/B test comparing two retrievers on a shared dataset.

```python
from astro_context.evaluation import ABTestRunner
```

**Constructor:**

```python
ABTestRunner(evaluator: PipelineEvaluator, dataset: EvaluationDataset)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `evaluator` | `PipelineEvaluator` | Evaluator for computing retrieval metrics |
| `dataset` | `EvaluationDataset` | Shared evaluation dataset |

**`run(retriever_a, retriever_b, k=10, significance_level=0.05) -> ABTestResult`**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `retriever_a` | `Retriever` | -- | First retriever |
| `retriever_b` | `Retriever` | -- | Second retriever |
| `k` | `int` | `10` | Top-k cutoff |
| `significance_level` | `float` | `0.05` | p-value threshold |

---

## BatchEvaluator

Runs evaluation over an entire dataset and aggregates results. Importable
from `astro_context.evaluation.batch`.

```python
from astro_context.evaluation.batch import BatchEvaluator
```

**Constructor:**

```python
BatchEvaluator(*, evaluator: PipelineEvaluator, retriever: Retriever, top_k: int = 10)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `evaluator` | `PipelineEvaluator` | -- | Per-sample evaluator |
| `retriever` | `Retriever` | -- | Retriever for fetching items |
| `top_k` | `int` | `10` | Items to retrieve per query |

**`evaluate(dataset, k=10) -> AggregatedMetrics`**

Returns `AggregatedMetrics` (batch module variant) with `count`,
`mean_precision`, `mean_recall`, `mean_f1`, `mean_mrr`, `mean_ndcg`,
`mean_hit_rate`, `p95_precision`, `p95_recall`, `min_precision`,
`min_recall`, and `per_sample_results`.

---

## HumanJudgment

A single human relevance judgment for a query-document pair.

```python
from astro_context.evaluation import HumanJudgment
```

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `query` | `str` | -- | The evaluated query |
| `item_id` | `str` | -- | Document ID being judged |
| `relevance` | `int` | `0 <= x <= 3` | Relevance score |
| `annotator` | `str` | -- | Annotator identifier |
| `metadata` | `dict[str, Any]` | -- | Arbitrary metadata |

---

## HumanEvaluationCollector

Collects human relevance judgments and computes inter-annotator agreement.

```python
from astro_context.evaluation import HumanEvaluationCollector
```

**Constructor:** `HumanEvaluationCollector()` -- no parameters.

**Property:** `judgments -> list[HumanJudgment]` -- copy of all collected judgments.

**`add_judgment(judgment: HumanJudgment) -> None`** -- add a single judgment.

**`add_judgments(judgments: list[HumanJudgment]) -> None`** -- add multiple judgments.

**`compute_agreement() -> float`** -- Cohen's kappa over (query, item_id) pairs judged by at least two annotators. Returns `0.0` if no overlapping judgments exist.

**`to_dataset(threshold: int = 2) -> EvaluationDataset`** -- converts judgments into an `EvaluationDataset`. Items with mean relevance at or above `threshold` are considered relevant.

**`compute_metrics() -> dict[str, float]`** -- returns `mean_relevance`, `agreement`, `num_judgments`, `num_annotators`, `num_queries`.
