# Query Classification Guide

Query classifiers inspect a `QueryBundle` and return a string label indicating
the query category. Use classifiers to **route queries to different retrievers**
based on intent, topic, or embedding similarity.

```
Query --> Classify --> label --> Route to Retriever A or B
```

All classifiers implement the `QueryClassifier` protocol with a single method:

```python
def classify(self, query: QueryBundle) -> str: ...
```

## Built-in Classifiers

### KeywordClassifier

Classifies queries by scanning for keywords in the query string. Rules are
evaluated in insertion order; the first matching rule wins. If no rule matches,
the `default` label is returned.

```python
from astro_context.query import KeywordClassifier
from astro_context.models.query import QueryBundle

classifier = KeywordClassifier(
    rules={
        "code": ["function", "class", "def", "import", "error", "bug"],
        "docs": ["documentation", "guide", "tutorial", "how to"],
        "api": ["endpoint", "REST", "GraphQL", "request"],
    },
    default="general",
    case_sensitive=False,
)

query = QueryBundle(query_str="How to fix a bug in my function?")
label = classifier.classify(query)
print(label)  # "code" (matched "function" first, then "bug")
```

!!! tip
    Put the most specific rules first. Since evaluation stops at the first
    match, ordering matters.

### CallbackClassifier

Delegates classification to a user-supplied callback. Use this when you need
custom logic, an LLM-based classifier, or integration with an external service.

```python
from astro_context.query import CallbackClassifier
from astro_context.models.query import QueryBundle

def my_classifier(query: QueryBundle) -> str:
    if "?" in query.query_str:
        return "question"
    return "statement"

classifier = CallbackClassifier(classify_fn=my_classifier)

label = classifier.classify(QueryBundle(query_str="What is RAG?"))
print(label)  # "question"
```

### EmbeddingClassifier

Classifies queries by comparing the query embedding to labelled centroid
embeddings using cosine similarity (or a custom distance function). The query
must have a non-`None` `embedding` field.

```python
import math
from astro_context.query import EmbeddingClassifier
from astro_context.models.query import QueryBundle

# Pre-computed centroid embeddings for each category
centroids = {
    "technical": [0.1, 0.9, 0.3, 0.2],
    "business":  [0.8, 0.1, 0.2, 0.7],
    "general":   [0.5, 0.5, 0.5, 0.5],
}

classifier = EmbeddingClassifier(centroids=centroids)

query = QueryBundle(
    query_str="How does backpropagation work?",
    embedding=[0.15, 0.85, 0.25, 0.18],
)
label = classifier.classify(query)
print(label)  # "technical"
```

!!! warning
    `EmbeddingClassifier.classify()` raises `ValueError` if
    `query.embedding` is `None`. Make sure to compute embeddings before
    classification.

You can supply a custom distance function:

```python
def dot_product(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))

classifier = EmbeddingClassifier(
    centroids=centroids,
    distance_fn=dot_product,
)
```

---

## Routing Queries with classified_retriever_step

The `classified_retriever_step()` function creates a pipeline step that classifies
the query and routes it to the appropriate retriever. This is the primary
integration point between classifiers and the pipeline.

```python
from astro_context.query import KeywordClassifier
from astro_context.pipeline import classified_retriever_step, ContextPipeline

classifier = KeywordClassifier(
    rules={
        "code": ["function", "class", "error", "bug"],
        "docs": ["guide", "tutorial", "documentation"],
    },
    default="general",
)

# Each label maps to a different retriever
step = classified_retriever_step(
    name="routed-retrieval",
    classifier=classifier,
    retrievers={
        "code": code_retriever,       # specialized for code search
        "docs": docs_retriever,       # specialized for documentation
        "general": general_retriever, # fallback
    },
    default="general",
    top_k=10,
)

pipeline = ContextPipeline(steps=[step])
```

### Parameters

| Parameter    | Type                      | Default  | Description                                    |
|------------- |---------------------------|----------|------------------------------------------------|
| `name`       | `str`                     | required | Human-readable step name                       |
| `classifier` | `QueryClassifier`         | required | Classifier that returns a label string         |
| `retrievers` | `dict[str, Retriever]`    | required | Label-to-retriever mapping                     |
| `default`    | `str \| None`             | `None`   | Fallback label when classified label not found |
| `top_k`      | `int`                     | `10`     | Maximum items to retrieve                      |

!!! note
    If the classified label is not in `retrievers` and no `default` is
    configured, a `RetrieverError` is raised.

---

## Full Example: Multi-Index Routing

This example shows a complete workflow where queries about code are routed to a
code-specific retriever and everything else goes to a documentation retriever.

```python
from astro_context.query import KeywordClassifier
from astro_context.models.query import QueryBundle
from astro_context.pipeline import classified_retriever_step

# 1. Define the classifier
classifier = KeywordClassifier(
    rules={
        "code": ["function", "class", "method", "import", "error", "traceback"],
        "docs": ["guide", "tutorial", "overview", "getting started"],
    },
    default="docs",
)

# 2. Verify classification
queries = [
    "How do I fix this traceback?",
    "Getting started with the library",
    "What is the meaning of life?",
]
for q in queries:
    label = classifier.classify(QueryBundle(query_str=q))
    print(f"{q!r:50s} -> {label}")
# "How do I fix this traceback?"                   -> code
# "Getting started with the library"               -> docs
# "What is the meaning of life?"                   -> docs (default)

# 3. Create the routed pipeline step
step = classified_retriever_step(
    name="smart-routing",
    classifier=classifier,
    retrievers={
        "code": code_retriever,
        "docs": docs_retriever,
    },
    default="docs",
    top_k=5,
)
```

!!! tip
    Combine classifiers with [query transformers](query-transform.md) for
    sophisticated pipelines: classify first, then apply domain-specific
    transformations before retrieval.
