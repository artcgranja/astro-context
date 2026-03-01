# Query Transform & Classifier API Reference

API reference for the `astro_context.query` module. For usage patterns and
examples, see the [Query Transform Guide](../guides/query-transform.md) and the
[Classifiers Guide](../guides/classifiers.md).

---

## Query Transformers

All transformers expose a `transform(query: QueryBundle) -> list[QueryBundle]`
method. They accept callback functions for LLM generation so that
`astro-context` never calls an LLM directly.

### HyDETransformer

Hypothetical Document Embeddings. Generates a hypothetical answer and uses it as
the retrieval query.

```python
class HyDETransformer(
    generate_fn: Callable[[str], str],
)
```

| Parameter     | Type                    | Default    | Description                              |
|---------------|-------------------------|------------|------------------------------------------|
| `generate_fn` | `Callable[[str], str]` | **required** | Takes query string, returns hypothetical document |

#### `transform(query)`

Returns a single-element list. The output `QueryBundle` has `query_str` set to
the hypothetical document and metadata keys `original_query` and
`transform = "hyde"`.

---

### MultiQueryTransformer

Generates multiple query variations for broader retrieval coverage.

```python
class MultiQueryTransformer(
    generate_fn: Callable[[str, int], list[str]],
    num_queries: int = 3,
)
```

| Parameter     | Type                              | Default    | Description                              |
|---------------|-----------------------------------|------------|------------------------------------------|
| `generate_fn` | `Callable[[str, int], list[str]]` | **required** | Takes query string and count, returns variations |
| `num_queries` | `int`                             | `3`        | Number of variations to generate         |

#### `transform(query)`

Returns a list of `N+1` `QueryBundle` objects: the original query as the first
element, followed by N generated variations. Each variation carries metadata
keys `original_query`, `transform = "multi_query"`, and `variation_index`.

---

### DecompositionTransformer

Breaks a complex query into simpler sub-questions.

```python
class DecompositionTransformer(
    generate_fn: Callable[[str], list[str]],
)
```

| Parameter     | Type                          | Default    | Description                              |
|---------------|-------------------------------|------------|------------------------------------------|
| `generate_fn` | `Callable[[str], list[str]]` | **required** | Takes query string, returns sub-questions |

#### `transform(query)`

Returns a list of `QueryBundle` objects, one per sub-question. Each carries
metadata keys `parent_query`, `transform = "decomposition"`, and
`sub_question_index`.

---

### StepBackTransformer

Generates a more abstract version of the query alongside the original.

```python
class StepBackTransformer(
    generate_fn: Callable[[str], str],
)
```

| Parameter     | Type                    | Default    | Description                              |
|---------------|-------------------------|------------|------------------------------------------|
| `generate_fn` | `Callable[[str], str]` | **required** | Takes query string, returns abstract version |

#### `transform(query)`

Returns a two-element list: `[original_query, step_back_query]`. The step-back
query carries metadata keys `original_query` and `transform = "step_back"`.

---

## Conversation-Aware Transformers

### ConversationRewriter

Rewrites a query using conversation history via a user-supplied callback. When
`chat_history` is empty, returns the original query unchanged.

```python
class ConversationRewriter(
    rewrite_fn: Callable[[str, list[ConversationTurn]], str],
)
```

| Parameter    | Type                                               | Default    | Description                                           |
|--------------|----------------------------------------------------|------------|-------------------------------------------------------|
| `rewrite_fn` | `Callable[[str, list[ConversationTurn]], str]`     | **required** | Takes query string and history, returns rewritten query |

#### `transform(query)`

Returns a single-element list. If `query.chat_history` is non-empty, the output
`QueryBundle` has metadata keys `original_query` and
`transform = "conversation_rewrite"`. The `embedding` and `chat_history` fields
are preserved from the original query.

---

### ContextualQueryTransformer

Wraps another transformer, prepending conversation context to the query before
delegation.

```python
class ContextualQueryTransformer(
    inner: QueryTransformer,
    context_prefix: str = "Given the conversation context: ",
)
```

| Parameter        | Type               | Default                                | Description                            |
|------------------|--------------------|----------------------------------------|----------------------------------------|
| `inner`          | `QueryTransformer` | **required**                           | Wrapped transformer to delegate to     |
| `context_prefix` | `str`              | `"Given the conversation context: "`   | Text prepended before the summary      |

#### `transform(query)`

When `chat_history` is non-empty, builds a summary string from the conversation
turns in `"role: content | role: content"` format, prepends `context_prefix`,
and delegates the augmented query to the inner transformer. When history is empty,
delegates directly without modification.

**Returns:** The output of `inner.transform()`.

---

## QueryTransformPipeline

Chains multiple query transformers and deduplicates results. Each transformer is
applied to every query produced by the previous stage.

```python
class QueryTransformPipeline(
    transformers: list[QueryTransformer],
)
```

| Parameter      | Type                     | Default    | Description                         |
|----------------|--------------------------|------------|-------------------------------------|
| `transformers` | `list[QueryTransformer]` | **required** | Ordered sequence of transformers  |

### Methods

#### `transform(query)`

Apply all transformers in sequence and deduplicate by `query_str`.

| Parameter | Type          | Default  | Description          |
|-----------|---------------|----------|----------------------|
| `query`   | `QueryBundle` | required | Original query       |

**Returns:** `list[QueryBundle]` -- deduplicated list of transformed queries.

#### `atransform(query)` (async)

Async version. Transformers implementing `AsyncQueryTransformer` are called via
`atransform`; others fall back to synchronous `transform`.

| Parameter | Type          | Default  | Description          |
|-----------|---------------|----------|----------------------|
| `query`   | `QueryBundle` | required | Original query       |

**Returns:** `list[QueryBundle]`

---

## Query Classifiers

All classifiers implement the `QueryClassifier` protocol:

```python
def classify(self, query: QueryBundle) -> str
```

### KeywordClassifier

Classifies queries by matching keywords in the query string. Rules are evaluated
in insertion order; first match wins.

```python
class KeywordClassifier(
    rules: dict[str, list[str]],
    default: str,
    case_sensitive: bool = False,
)
```

| Parameter        | Type                     | Default    | Description                              |
|------------------|--------------------------|------------|------------------------------------------|
| `rules`          | `dict[str, list[str]]`   | **required** | Label-to-keywords mapping              |
| `default`        | `str`                    | **required** | Fallback label when no rule matches    |
| `case_sensitive` | `bool`                   | `False`    | Whether matching is case-sensitive       |

#### `classify(query)`

Scans `query.query_str` for keywords. Returns the label of the first matching
rule, or `default`.

---

### CallbackClassifier

Delegates classification to a user-supplied callback.

```python
class CallbackClassifier(
    classify_fn: Callable[[QueryBundle], str],
)
```

| Parameter     | Type                            | Default    | Description                    |
|---------------|---------------------------------|------------|--------------------------------|
| `classify_fn` | `Callable[[QueryBundle], str]` | **required** | Classification callback      |

#### `classify(query)`

Returns the string label from the callback.

---

### EmbeddingClassifier

Classifies by comparing query embedding to labelled centroid embeddings.

```python
class EmbeddingClassifier(
    centroids: dict[str, list[float]],
    distance_fn: Callable[[list[float], list[float]], float] | None = None,
)
```

| Parameter     | Type                                                    | Default             | Description                              |
|---------------|---------------------------------------------------------|---------------------|------------------------------------------|
| `centroids`   | `dict[str, list[float]]`                               | **required**        | Label-to-centroid embedding mapping      |
| `distance_fn` | `Callable[[list[float], list[float]], float] \| None`   | cosine similarity   | Similarity function (higher = closer)    |

#### `classify(query)`

Compares `query.embedding` to each centroid and returns the label with the
highest similarity score.

**Raises:** `ValueError` if `query.embedding` is `None`.

---

## Pipeline Integration Functions

### `query_transform_step(name, transformer, retriever, top_k=10)`

Create a pipeline step that transforms the query, retrieves for each variant,
and merges results via Reciprocal Rank Fusion (RRF).

```python
from astro_context.pipeline import query_transform_step
```

| Parameter     | Type               | Default  | Description                              |
|---------------|--------------------|----------|------------------------------------------|
| `name`        | `str`              | required | Descriptive name for this step           |
| `transformer` | `QueryTransformer` | required | Transformer to expand the query          |
| `retriever`   | `Retriever`        | required | Retriever to run per expanded query      |
| `top_k`       | `int`              | `10`     | Max items to retrieve per variant        |

**Returns:** `PipelineStep`

### `classified_retriever_step(name, classifier, retrievers, default=None, top_k=10)`

Create a pipeline step that classifies the query and routes to the matching
retriever.

```python
from astro_context.pipeline import classified_retriever_step
```

| Parameter    | Type                    | Default  | Description                                    |
|--------------|-------------------------|----------|------------------------------------------------|
| `name`       | `str`                   | required | Human-readable step name                       |
| `classifier` | `QueryClassifier`       | required | Classifier returning a label string            |
| `retrievers` | `dict[str, Retriever]`  | required | Label-to-retriever mapping                     |
| `default`    | `str \| None`           | `None`   | Fallback label when classified label not found |
| `top_k`      | `int`                   | `10`     | Maximum items to retrieve                      |

**Returns:** `PipelineStep`
**Raises:** `RetrieverError` if classified label has no matching retriever and
no default is configured.
