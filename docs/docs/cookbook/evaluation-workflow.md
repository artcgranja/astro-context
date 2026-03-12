# Evaluation Workflow

Evaluate retrieval quality with standard metrics and compare retriever
configurations using A/B testing with statistical significance.

---

## Overview

This example demonstrates:

- Creating an `EvaluationDataset` with `EvaluationSample` entries
- Computing retrieval metrics with `RetrievalMetricsCalculator`
- Using `PipelineEvaluator` to evaluate a retriever
- Running `ABTestRunner` to compare two retriever configurations
- Interpreting the results

## Full Example

```python
import math

from astro_context import (
    ContextItem,
    DenseRetriever,
    HybridRetriever,
    InMemoryContextStore,
    InMemoryVectorStore,
    QueryBundle,
    SourceType,
    RetrievalMetricsCalculator,
    PipelineEvaluator,
)
from astro_context.evaluation.ab_testing import (
    ABTestRunner,
    EvaluationDataset,
    EvaluationSample,
)

# ---------------------------------------------------------------
# 1. Deterministic embedding function
# ---------------------------------------------------------------
def embed_fn(text: str) -> list[float]:
    seed = sum(ord(c) for c in text) % 10000
    raw = [math.sin(seed * 1000 + i) for i in range(64)]
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw] if norm else raw


# ---------------------------------------------------------------
# 2. Build a knowledge base with known document IDs
# ---------------------------------------------------------------
knowledge_base = {
    "doc-python": "Python is a versatile language for web and data science.",
    "doc-rag": "RAG combines retrieval with generation for grounded answers.",
    "doc-vectors": "Vector databases store embeddings for similarity search.",
    "doc-context": "Context engineering builds intelligent AI pipelines.",
    "doc-bm25": "BM25 scores documents using term frequency and IDF.",
    "doc-hybrid": "Hybrid search combines dense and sparse retrieval with RRF.",
    "doc-chunking": "Chunking splits documents into smaller pieces for indexing.",
    "doc-reranking": "Reranking re-scores retrieved documents for relevance.",
    "doc-memory": "Sliding window memory manages conversation history.",
    "doc-eval": "Evaluation metrics measure retrieval and generation quality.",
}

items = [
    ContextItem(id=doc_id, content=content, source=SourceType.RETRIEVAL)
    for doc_id, content in knowledge_base.items()
]

# ---------------------------------------------------------------
# 3. Create two retriever configurations to compare
# ---------------------------------------------------------------

# Configuration A: standard dense retriever
retriever_a = DenseRetriever(
    vector_store=InMemoryVectorStore(),
    context_store=InMemoryContextStore(),
    embed_fn=embed_fn,
)
retriever_a.index(items)

# Configuration B: different embedding function (simulates a different model)
def embed_fn_b(text: str) -> list[float]:
    seed = sum(ord(c) for c in text) % 7777
    raw = [math.cos(seed * 500 + i) for i in range(64)]
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw] if norm else raw

retriever_b = DenseRetriever(
    vector_store=InMemoryVectorStore(),
    context_store=InMemoryContextStore(),
    embed_fn=embed_fn_b,
)
retriever_b.index(items)

# ---------------------------------------------------------------
# 4. Create an evaluation dataset with ground-truth relevance
# ---------------------------------------------------------------
dataset = EvaluationDataset(
    name="retrieval-benchmark",
    samples=[
        EvaluationSample(
            query="How does RAG work?",
            relevant_ids=["doc-rag", "doc-vectors", "doc-context"],
        ),
        EvaluationSample(
            query="What is hybrid search?",
            relevant_ids=["doc-hybrid", "doc-bm25", "doc-vectors"],
        ),
        EvaluationSample(
            query="How to evaluate retrieval quality?",
            relevant_ids=["doc-eval", "doc-reranking"],
        ),
        EvaluationSample(
            query="Tell me about Python programming",
            relevant_ids=["doc-python"],
        ),
        EvaluationSample(
            query="How does memory management work in AI?",
            relevant_ids=["doc-memory", "doc-context"],
        ),
        EvaluationSample(
            query="What is document chunking?",
            relevant_ids=["doc-chunking", "doc-rag"],
        ),
    ],
    metadata={"created_by": "evaluation-example"},
)

print(f"Dataset: {dataset.name}")
print(f"Samples: {len(dataset.samples)}\n")

# ---------------------------------------------------------------
# 5. Evaluate a single query with RetrievalMetricsCalculator
# ---------------------------------------------------------------
print("=== Single Query Evaluation ===\n")

calculator = RetrievalMetricsCalculator(k=5)

query = QueryBundle(
    query_str="How does RAG work?",
    embedding=embed_fn("How does RAG work?"),
)
retrieved = retriever_a.retrieve(query, top_k=5)

metrics = calculator.evaluate(
    retrieved=retrieved,
    relevant=["doc-rag", "doc-vectors", "doc-context"],
    k=5,
)

print(f"  Precision@5: {metrics.precision_at_k:.3f}")
print(f"  Recall@5:    {metrics.recall_at_k:.3f}")
print(f"  F1@5:        {metrics.f1_at_k:.3f}")
print(f"  MRR:         {metrics.mrr:.3f}")
print(f"  NDCG:        {metrics.ndcg:.3f}")
print(f"  Hit Rate:    {metrics.hit_rate:.1f}")

print("\n  Retrieved items:")
relevant_set = {"doc-rag", "doc-vectors", "doc-context"}
for i, item in enumerate(retrieved, 1):
    is_relevant = "Y" if item.id in relevant_set else "N"
    print(f"    {i}. [{is_relevant}] {item.id}: {item.content[:50]}...")

# ---------------------------------------------------------------
# 6. Use PipelineEvaluator for structured evaluation
# ---------------------------------------------------------------
print("\n=== PipelineEvaluator ===\n")

evaluator = PipelineEvaluator(retrieval_calculator=calculator)

for sample in dataset.samples[:3]:
    q = QueryBundle(
        query_str=sample.query,
        embedding=embed_fn(sample.query),
    )
    retrieved = retriever_a.retrieve(q, top_k=5)
    m = evaluator.evaluate_retrieval(retrieved, sample.relevant_ids, k=5)
    print(f"  Q: {sample.query}")
    print(f"    P@5={m.precision_at_k:.2f}  R@5={m.recall_at_k:.2f}  "
          f"MRR={m.mrr:.2f}  NDCG={m.ndcg:.2f}")

# ---------------------------------------------------------------
# 7. A/B test two retriever configurations
# ---------------------------------------------------------------
print("\n=== A/B Test: Retriever A vs Retriever B ===\n")

ab_runner = ABTestRunner(evaluator=evaluator, dataset=dataset)
ab_result = ab_runner.run(
    retriever_a=retriever_a,
    retriever_b=retriever_b,
    k=5,
    significance_level=0.05,
)

print("  Retriever A:")
print(f"    Mean Precision: {ab_result.metrics_a.mean_precision:.3f}")
print(f"    Mean Recall:    {ab_result.metrics_a.mean_recall:.3f}")
print(f"    Mean F1:        {ab_result.metrics_a.mean_f1:.3f}")
print(f"    Mean MRR:       {ab_result.metrics_a.mean_mrr:.3f}")
print(f"    Mean NDCG:      {ab_result.metrics_a.mean_ndcg:.3f}")

print("\n  Retriever B:")
print(f"    Mean Precision: {ab_result.metrics_b.mean_precision:.3f}")
print(f"    Mean Recall:    {ab_result.metrics_b.mean_recall:.3f}")
print(f"    Mean F1:        {ab_result.metrics_b.mean_f1:.3f}")
print(f"    Mean MRR:       {ab_result.metrics_b.mean_mrr:.3f}")
print(f"    Mean NDCG:      {ab_result.metrics_b.mean_ndcg:.3f}")

print(f"\n  Winner:         {ab_result.winner}")
print(f"  P-value:        {ab_result.p_value:.4f}")
print(f"  Significant:    {ab_result.is_significant}")
print(f"  Sig. level:     {ab_result.significance_level}")

# ---------------------------------------------------------------
# 8. Inspect per-metric comparison
# ---------------------------------------------------------------
print("\n=== Per-Metric Deltas ===\n")

for metric_name, comparison in ab_result.per_metric_comparison.items():
    delta = comparison["delta"]
    direction = "A > B" if delta > 0 else "B > A" if delta < 0 else "tie"
    print(f"  {metric_name:10s}: A={comparison['a']:.3f}  "
          f"B={comparison['b']:.3f}  delta={delta:+.3f}  ({direction})")
```

## Metrics Reference

### Retrieval Metrics

| Metric | Range | Description |
|--------|-------|-------------|
| `precision_at_k` | 0-1 | Fraction of top-k results that are relevant |
| `recall_at_k` | 0-1 | Fraction of relevant docs found in top-k |
| `f1_at_k` | 0-1 | Harmonic mean of precision and recall |
| `mrr` | 0-1 | Reciprocal rank of first relevant result |
| `ndcg` | 0-1 | Normalized Discounted Cumulative Gain |
| `hit_rate` | 0/1 | Whether any relevant doc appears in top-k |

### Interpreting A/B Results

- **p_value < 0.05**: the difference is statistically significant
- **winner = "tie"**: no significant difference detected
- **winner = "a" or "b"**: that retriever performed significantly better
- **per_metric_comparison**: shows exactly where each retriever wins

!!! tip "Sample Size for A/B Tests"
    For reliable statistical significance, use at least 20-30 evaluation
    samples. Fewer samples may produce high p-values even when there is
    a real difference.

!!! note "No LLM Required for Retrieval Metrics"
    `RetrievalMetricsCalculator` is purely computational -- it compares
    retrieved item IDs against known relevant IDs. No API key or LLM
    call is needed.

!!! warning "Paired T-Test Approximation"
    The A/B test uses a normal approximation for the t-test statistic,
    which is conservative for small sample sizes. For production-grade
    significance testing with small datasets, consider using `scipy`
    for exact t-distribution p-values.

## Next Steps

- [RAG Pipeline](rag-pipeline.md) -- build the pipeline you want to evaluate
- [Custom Retriever](custom-retriever.md) -- create retrievers to compare
- [Document Ingestion](document-ingestion.md) -- prepare evaluation datasets from files
