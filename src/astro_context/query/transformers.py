"""Built-in query transformation strategies.

All transformers accept callback functions for LLM generation so that
``astro-context`` never calls an LLM directly.  Users supply their own
generation functions and the transformers handle orchestration.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from astro_context.models.query import QueryBundle

logger = logging.getLogger(__name__)


class HyDETransformer:
    """Hypothetical Document Embeddings (HyDE) query transformer.

    Generates a hypothetical answer to the query and uses it as the
    retrieval query.  The intuition is that embedding a plausible answer
    is closer in vector space to the real answer than the question itself.

    Parameters:
        generate_fn: A callable ``(str) -> str`` that takes a query
            string and returns a hypothetical document/answer.
    """

    __slots__ = ("_generate_fn",)

    def __init__(self, generate_fn: Callable[[str], str]) -> None:
        self._generate_fn = generate_fn

    def __repr__(self) -> str:
        return "HyDETransformer()"

    def transform(self, query: QueryBundle) -> list[QueryBundle]:
        """Generate a hypothetical document and return it as a query.

        Parameters:
            query: The original query bundle.

        Returns:
            A single-element list containing a ``QueryBundle`` whose
            ``query_str`` is the hypothetical document, with the
            original query preserved in metadata.
        """
        hypothetical_doc = self._generate_fn(query.query_str)
        logger.debug("HyDE generated hypothetical doc for query: %s", query.query_str)
        return [
            QueryBundle(
                query_str=hypothetical_doc,
                metadata={
                    **query.metadata,
                    "original_query": query.query_str,
                    "transform": "hyde",
                },
            )
        ]


class MultiQueryTransformer:
    """Generates multiple query variations for broader retrieval coverage.

    Produces N alternative phrasings of the original query so that
    retrieval can cover different aspects of the user's intent.

    Parameters:
        generate_fn: A callable ``(str, int) -> list[str]`` that takes
            a query string and count, returning that many query variations.
        num_queries: Number of query variations to generate (default 3).
    """

    __slots__ = ("_generate_fn", "_num_queries")

    def __init__(
        self,
        generate_fn: Callable[[str, int], list[str]],
        num_queries: int = 3,
    ) -> None:
        if num_queries < 1:
            msg = "num_queries must be at least 1"
            raise ValueError(msg)
        self._generate_fn = generate_fn
        self._num_queries = num_queries

    def __repr__(self) -> str:
        return f"MultiQueryTransformer(num_queries={self._num_queries})"

    def transform(self, query: QueryBundle) -> list[QueryBundle]:
        """Generate multiple query variations.

        Parameters:
            query: The original query bundle.

        Returns:
            A list of ``QueryBundle`` objects, one per generated variation.
            The original query is included as the first element.
        """
        variations = self._generate_fn(query.query_str, self._num_queries)
        logger.debug(
            "MultiQuery generated %d variations for: %s",
            len(variations),
            query.query_str,
        )
        results = [query]
        for i, variation in enumerate(variations):
            results.append(
                QueryBundle(
                    query_str=variation,
                    metadata={
                        **query.metadata,
                        "original_query": query.query_str,
                        "transform": "multi_query",
                        "variation_index": i,
                    },
                )
            )
        return results


class DecompositionTransformer:
    """Breaks a complex query into simpler sub-questions.

    Useful for multi-hop or compound questions where answering the
    original requires synthesising information from multiple sources.

    Parameters:
        generate_fn: A callable ``(str) -> list[str]`` that takes a
            query string and returns a list of sub-questions.
    """

    __slots__ = ("_generate_fn",)

    def __init__(self, generate_fn: Callable[[str], list[str]]) -> None:
        self._generate_fn = generate_fn

    def __repr__(self) -> str:
        return "DecompositionTransformer()"

    def transform(self, query: QueryBundle) -> list[QueryBundle]:
        """Decompose the query into sub-questions.

        Parameters:
            query: The original query bundle.

        Returns:
            A list of ``QueryBundle`` objects, one per sub-question,
            each carrying the parent query in metadata.
        """
        sub_questions = self._generate_fn(query.query_str)
        logger.debug(
            "Decomposition produced %d sub-questions for: %s",
            len(sub_questions),
            query.query_str,
        )
        return [
            QueryBundle(
                query_str=sub_q,
                metadata={
                    **query.metadata,
                    "parent_query": query.query_str,
                    "transform": "decomposition",
                    "sub_question_index": i,
                },
            )
            for i, sub_q in enumerate(sub_questions)
        ]


class StepBackTransformer:
    """Generates a more abstract version of the query alongside the original.

    The "step-back" technique asks a broader, more general question to
    retrieve high-level context that helps answer the specific query.

    Parameters:
        generate_fn: A callable ``(str) -> str`` that takes a query
            string and returns a more abstract/general version.
    """

    __slots__ = ("_generate_fn",)

    def __init__(self, generate_fn: Callable[[str], str]) -> None:
        self._generate_fn = generate_fn

    def __repr__(self) -> str:
        return "StepBackTransformer()"

    def transform(self, query: QueryBundle) -> list[QueryBundle]:
        """Return the original query plus an abstracted step-back query.

        Parameters:
            query: The original query bundle.

        Returns:
            A two-element list: ``[original_query, step_back_query]``.
        """
        step_back = self._generate_fn(query.query_str)
        logger.debug("StepBack generated abstract query for: %s", query.query_str)
        step_back_bundle = QueryBundle(
            query_str=step_back,
            metadata={
                **query.metadata,
                "original_query": query.query_str,
                "transform": "step_back",
            },
        )
        return [query, step_back_bundle]
