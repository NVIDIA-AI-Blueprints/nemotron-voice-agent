# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Tests for correlated Frontend/Backend Agent RTVI stage metrics."""

# ruff: noqa: D101, D102, D103

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

from pipecat.processors.aggregators.llm_context import LLMContext

from examples.frontend_backend_agent.src.stage_metrics import (
    StageMetricsCoordinator,
    run_streamed_inference,
)


class _TestStream:
    def __init__(self, chunks: list[object]) -> None:
        self._chunks = iter(chunks)
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._chunks)
        except StopIteration as exc:
            raise StopAsyncIteration from exc

    async def aclose(self) -> None:
        self.closed = True


class _StreamingLLM:
    def __init__(self, stream: _TestStream) -> None:
        self.stream = stream
        self.context: LLMContext | None = None

    async def get_chat_completions(self, context: LLMContext):
        self.context = context
        return self.stream


class _Adapter:
    def get_llm_invocation_params(self, context, *, system_instruction, convert_developer_to_user):
        return {
            "context": context,
            "system_instruction": system_instruction,
            "convert_developer_to_user": convert_developer_to_user,
        }


class _Completions:
    def __init__(self, stream: _TestStream) -> None:
        self.stream = stream
        self.params: dict | None = None

    async def create(self, **params):
        self.params = params
        return self.stream


class _OutOfBandLLM:
    def __init__(self, stream: _TestStream) -> None:
        self._settings = SimpleNamespace(system_instruction="hidden system")
        self.completions = _Completions(stream)
        self._client = SimpleNamespace(chat=SimpleNamespace(completions=self.completions))
        self.supports_developer_role = True
        self.pipeline_stream_called = False

    def get_llm_adapter(self):
        return _Adapter()

    def build_chat_completion_params(self, invocation_params):
        return {"model": "thinker", "max_completion_tokens": 999, "invocation": invocation_params}

    async def get_chat_completions(self, context):
        self.pipeline_stream_called = True
        raise AssertionError("out-of-band Thinker inference must bypass pipeline stream processing")


def _chunk(*, content: str | None = None, reasoning_content: str | None = None):
    delta = SimpleNamespace(content=content, reasoning_content=reasoning_content, reasoning=None, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


class FrontendBackendStageMetricsTests(unittest.IsolatedAsyncioTestCase):
    async def test_coordinator_emits_correlated_pipecat_metrics_and_tool_lifecycle(self) -> None:
        frames = []
        events = []

        async def emit_metric(frame) -> None:
            frames.append(frame)

        async def emit_event(event: dict) -> None:
            events.append(event)

        coordinator = StageMetricsCoordinator(emit_metric, emit_event)
        initial = await coordinator.start_frontend_initial("talker")
        initial.relabel("frontend_tool_selection")
        await coordinator.bind_tool_call("tool-call-1", initial.turn_id)
        await initial.mark_ttft()
        await initial.mark_ttft()
        await initial.finish()
        await initial.finish()

        await coordinator.bind_backend_call("tool-call-1", "backend-call-1")
        backend = await coordinator.start_backend("backend-call-1", model="thinker", attempt=2)
        await backend.mark_ttft()
        await backend.finish("success")

        tool = await coordinator.start_tool("backend-call-1", tool_name="web_search", ordinal=0)
        await coordinator.finish_tool(tool, "timeout")

        final = await coordinator.start_frontend_final("tool-call-1", "talker")
        await final.mark_ttft()
        await final.finish("fallback")

        metrics = [frame.data[0] for frame in frames]
        self.assertEqual(
            [metric.processor for metric in metrics],
            [
                "frontend_tool_selection_llm",
                "frontend_tool_selection_llm",
                "backend_thinker_llm",
                "backend_thinker_llm",
                "backend_tool_call.web_search",
                "frontend_final_response_llm",
                "frontend_final_response_llm",
            ],
        )
        self.assertEqual({metric.turn_id for metric in metrics}, {initial.turn_id})
        self.assertEqual(backend.parent_invocation_id, "backend-call-1")
        self.assertEqual(final.parent_invocation_id, "tool-call-1")
        self.assertEqual(backend.attempt, 2)
        self.assertEqual(metrics[4].metric, "latency")
        self.assertEqual(metrics[4].outcome, "timeout")
        self.assertEqual(metrics[-1].outcome, "fallback")
        self.assertEqual([event["type"] for event in events], ["tool-call", "tool-call-done"])

        await coordinator.cleanup_tool_call("tool-call-1")
        orphan_final = await coordinator.start_frontend_final("tool-call-1", "talker")
        self.assertNotEqual(orphan_final.turn_id, initial.turn_id)

    async def test_streamed_thinker_marks_true_ttft_and_collects_only_visible_content(self) -> None:
        frames = []

        async def emit_metric(frame) -> None:
            frames.append(frame)

        coordinator = StageMetricsCoordinator(emit_metric)
        span = await coordinator.start_backend("backend-call-1", model="thinker", attempt=1)
        stream = _TestStream(
            [
                _chunk(reasoning_content="private reasoning token"),
                _chunk(content='{"tool":'),
                _chunk(content='"web_search"}'),
            ]
        )
        llm = _StreamingLLM(stream)
        context = LLMContext([{"role": "user", "content": "Plan this request."}])

        result = await run_streamed_inference(llm, context, span)

        self.assertEqual(result, '{"tool":"web_search"}')
        self.assertTrue(stream.closed)
        self.assertIs(llm.context, context)
        metrics = [frame.data[0] for frame in frames]
        self.assertEqual(
            [metric.processor for metric in metrics],
            ["backend_thinker_llm", "backend_thinker_llm"],
        )
        self.assertEqual([metric.metric for metric in metrics], ["ttft", "processing"])

    async def test_out_of_band_stream_bypasses_pipeline_side_effects_and_bounds_tokens(self) -> None:
        stream = _TestStream([_chunk(content='{"tool":"web_search"}')])
        llm = _OutOfBandLLM(stream)
        context = LLMContext([{"role": "user", "content": "Plan this request."}])

        result = await run_streamed_inference(llm, context, None, max_tokens=256)

        self.assertEqual(result, '{"tool":"web_search"}')
        self.assertFalse(llm.pipeline_stream_called)
        self.assertTrue(stream.closed)
        self.assertIsNotNone(llm.completions.params)
        self.assertTrue(llm.completions.params["stream"])
        self.assertEqual(llm.completions.params["max_completion_tokens"], 256)


def test_scaling_benchmark_exports_all_recovered_stage_columns() -> None:
    path = Path("benchmarking_tools/scaling-perf/benchmark.py")
    spec = importlib.util.spec_from_file_location("scaling_perf_benchmark", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    expected = (
        "frontend_tool_selection_ttft",
        "frontend_tool_selection_processing_time",
        "backend_llm_ttft",
        "backend_llm_processing_time",
        "backend_tool_call_latency",
        "frontend_final_response_ttft",
        "frontend_final_response_processing_time",
    )
    assert module.SERVER_METRIC_KEYS[-7:] == expected
