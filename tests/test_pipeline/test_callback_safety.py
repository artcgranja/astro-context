"""Tests for PipelineCallback error swallowing.

Verifies that a crashing callback does not prevent:
- The pipeline from completing successfully
- Subsequent callbacks from firing
- The correct result being returned
"""

from __future__ import annotations

from typing import Any

from astro_context.models.context import ContextItem, ContextResult, SourceType
from astro_context.models.query import QueryBundle
from astro_context.pipeline.pipeline import ContextPipeline
from astro_context.pipeline.step import PipelineStep
from tests.conftest import FakeRetriever, FakeTokenizer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pipeline() -> ContextPipeline:
    """Create a ContextPipeline with a FakeTokenizer."""
    return ContextPipeline(max_tokens=8192, tokenizer=FakeTokenizer())


class ExplodingCallback:
    """A callback that raises RuntimeError on every method."""

    def on_pipeline_start(self, query: QueryBundle) -> None:
        msg = "ExplodingCallback: on_pipeline_start"
        raise RuntimeError(msg)

    def on_step_start(self, step_name: str, items: list[ContextItem]) -> None:
        msg = "ExplodingCallback: on_step_start"
        raise RuntimeError(msg)

    def on_step_end(
        self, step_name: str, items: list[ContextItem], time_ms: float
    ) -> None:
        msg = "ExplodingCallback: on_step_end"
        raise RuntimeError(msg)

    def on_step_error(self, step_name: str, error: Exception) -> None:
        msg = "ExplodingCallback: on_step_error"
        raise RuntimeError(msg)

    def on_pipeline_end(self, result: ContextResult) -> None:
        msg = "ExplodingCallback: on_pipeline_end"
        raise RuntimeError(msg)


class RecordingCallback:
    """A callback that records which methods were called."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.args: dict[str, list[Any]] = {}

    def _record(self, method: str, *args: Any) -> None:
        self.calls.append(method)
        self.args.setdefault(method, []).append(args)

    def on_pipeline_start(self, query: QueryBundle) -> None:
        self._record("on_pipeline_start", query)

    def on_step_start(self, step_name: str, items: list[ContextItem]) -> None:
        self._record("on_step_start", step_name, items)

    def on_step_end(
        self, step_name: str, items: list[ContextItem], time_ms: float
    ) -> None:
        self._record("on_step_end", step_name, items, time_ms)

    def on_step_error(self, step_name: str, error: Exception) -> None:
        self._record("on_step_error", step_name, error)

    def on_pipeline_end(self, result: ContextResult) -> None:
        self._record("on_pipeline_end", result)


# ---------------------------------------------------------------------------
# Crashing callback does not prevent pipeline from completing
# ---------------------------------------------------------------------------


class TestCallbackSafetyPipelineCompletion:
    """Crashing callback does not prevent the pipeline from completing."""

    def test_pipeline_completes_with_exploding_callback(self) -> None:
        pipeline = _make_pipeline()
        pipeline.add_callback(ExplodingCallback())
        pipeline.add_system_prompt("System prompt.")

        result = pipeline.build(QueryBundle(query_str="test"))

        assert isinstance(result, ContextResult)
        assert len(result.window.items) == 1

    def test_pipeline_with_exploding_callback_and_step(self) -> None:
        tokenizer = FakeTokenizer()
        items = [
            ContextItem(
                id="r1",
                content="Retrieved doc.",
                source=SourceType.RETRIEVAL,
                score=0.9,
                priority=5,
                token_count=tokenizer.count_tokens("Retrieved doc."),
            ),
        ]
        pipeline = _make_pipeline()
        pipeline.add_callback(ExplodingCallback())
        pipeline.add_system_prompt("System prompt.")
        pipeline.add_step(
            PipelineStep(
                name="search",
                fn=lambda items_in, q: items_in + items,
            )
        )

        result = pipeline.build(QueryBundle(query_str="test"))

        assert isinstance(result, ContextResult)
        # System prompt + retrieved doc
        assert len(result.window.items) == 2


# ---------------------------------------------------------------------------
# Subsequent callbacks still fire
# ---------------------------------------------------------------------------


class TestCallbackSafetySubsequentCallbacksFire:
    """Crashing callback does not prevent subsequent callbacks from firing."""

    def test_recording_callback_fires_after_exploding(self) -> None:
        recorder = RecordingCallback()

        pipeline = _make_pipeline()
        pipeline.add_callback(ExplodingCallback())
        pipeline.add_callback(recorder)
        pipeline.add_system_prompt("System prompt.")

        pipeline.build(QueryBundle(query_str="test"))

        assert "on_pipeline_start" in recorder.calls
        assert "on_pipeline_end" in recorder.calls

    def test_recording_callback_fires_before_exploding(self) -> None:
        recorder = RecordingCallback()

        pipeline = _make_pipeline()
        pipeline.add_callback(recorder)
        pipeline.add_callback(ExplodingCallback())
        pipeline.add_system_prompt("System prompt.")

        pipeline.build(QueryBundle(query_str="test"))

        assert "on_pipeline_start" in recorder.calls
        assert "on_pipeline_end" in recorder.calls

    def test_all_step_callbacks_fire_with_exploding(self) -> None:
        recorder = RecordingCallback()

        pipeline = _make_pipeline()
        pipeline.add_callback(ExplodingCallback())
        pipeline.add_callback(recorder)
        pipeline.add_step(
            PipelineStep(name="noop", fn=lambda items, q: items)
        )

        pipeline.build(QueryBundle(query_str="test"))

        assert "on_step_start" in recorder.calls
        assert "on_step_end" in recorder.calls


# ---------------------------------------------------------------------------
# Correct result returned despite crashing callbacks
# ---------------------------------------------------------------------------


class TestCallbackSafetyCorrectResult:
    """Correct result is returned despite crashing callbacks."""

    def test_result_contains_correct_items(self) -> None:
        recorder = RecordingCallback()

        pipeline = _make_pipeline()
        pipeline.add_callback(ExplodingCallback())
        pipeline.add_callback(recorder)
        pipeline.add_system_prompt("You are a helpful assistant.")

        result = pipeline.build(QueryBundle(query_str="test"))

        assert result.window.items[0].content == "You are a helpful assistant."
        assert result.format_type == "generic"

    def test_diagnostics_are_correct(self) -> None:
        tokenizer = FakeTokenizer()
        items = [
            ContextItem(
                id="r1",
                content="doc one",
                source=SourceType.RETRIEVAL,
                score=0.8,
                priority=5,
                token_count=tokenizer.count_tokens("doc one"),
            ),
        ]
        retriever = FakeRetriever(items)

        pipeline = _make_pipeline()
        pipeline.add_callback(ExplodingCallback())
        from astro_context.pipeline.step import retriever_step

        pipeline.add_step(retriever_step("search", retriever))

        result = pipeline.build(QueryBundle(query_str="test"))

        assert "steps" in result.diagnostics
        assert len(result.diagnostics["steps"]) == 1
        assert result.diagnostics["steps"][0]["name"] == "search"

    def test_recorder_receives_result_in_on_pipeline_end(self) -> None:
        recorder = RecordingCallback()

        pipeline = _make_pipeline()
        pipeline.add_callback(recorder)
        pipeline.add_system_prompt("System.")

        pipeline.build(QueryBundle(query_str="test"))

        # Verify recorder got the result via callback
        assert len(recorder.args["on_pipeline_end"]) == 1
        recorded_result = recorder.args["on_pipeline_end"][0][0]
        assert isinstance(recorded_result, ContextResult)
        assert recorded_result.window.items[0].content == "System."


# ---------------------------------------------------------------------------
# Async pipeline with crashing callbacks
# ---------------------------------------------------------------------------


class TestCallbackSafetyAsync:
    """Crashing callbacks in async pipeline."""

    async def test_async_pipeline_completes_with_exploding_callback(self) -> None:
        recorder = RecordingCallback()

        pipeline = _make_pipeline()
        pipeline.add_callback(ExplodingCallback())
        pipeline.add_callback(recorder)
        pipeline.add_system_prompt("Async system prompt.")

        result = await pipeline.abuild(QueryBundle(query_str="test"))

        assert isinstance(result, ContextResult)
        assert len(result.window.items) == 1
        assert "on_pipeline_start" in recorder.calls
        assert "on_pipeline_end" in recorder.calls
