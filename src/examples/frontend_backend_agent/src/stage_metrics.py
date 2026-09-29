# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Correlated, RTVI-compatible metrics for Frontend/Backend Agent stages."""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Awaitable, Callable
from typing import Literal

from openai.types.chat import ChatCompletionChunk
from pipecat.frames.frames import MetricsFrame
from pipecat.metrics.metrics import ProcessingMetricsData, TTFBMetricsData
from pipecat.processors.aggregators.llm_context import LLMContext

MetricOutcome = Literal["success", "timeout", "error", "cancelled", "fallback"]
MetricEmitter = Callable[[MetricsFrame], Awaitable[None]]
ServerEventEmitter = Callable[[dict], Awaitable[None]]

_PROCESSORS = {
    "frontend_tool_selection": "frontend_tool_selection_llm",
    "backend_thinker": "backend_thinker_llm",
    "frontend_final_response": "frontend_final_response_llm",
}


class StageTTFTMetricsData(TTFBMetricsData):
    """A normal Pipecat TTFB metric with additive stage correlation."""

    metric: Literal["ttft"] = "ttft"
    stage: str
    turn_id: str
    invocation_id: str
    parent_invocation_id: str | None = None
    attempt: int = 1


class StageProcessingMetricsData(ProcessingMetricsData):
    """A normal Pipecat processing metric with additive stage correlation."""

    metric: Literal["processing", "latency"] = "processing"
    stage: str
    turn_id: str
    invocation_id: str
    parent_invocation_id: str | None = None
    attempt: int = 1
    outcome: MetricOutcome = "success"
    tool_name: str | None = None


class StageSpan:
    """One immutable-clock logical stage whose frames are emitted at most once."""

    def __init__(
        self,
        coordinator: StageMetricsCoordinator,
        *,
        stage: str,
        model: str | None,
        turn_id: str,
        invocation_id: str,
        parent_invocation_id: str | None = None,
        attempt: int = 1,
        tool_name: str | None = None,
    ) -> None:
        """Start an unreported span using a monotonic clock."""
        self._coordinator = coordinator
        self.stage = stage
        self.model = model
        self.turn_id = turn_id
        self.invocation_id = invocation_id
        self.parent_invocation_id = parent_invocation_id
        self.attempt = attempt
        self.tool_name = tool_name
        self._started_ns = time.perf_counter_ns()
        self._ttft_emitted = False
        self._finished = False

    def relabel(self, stage: str) -> None:
        """Assign the semantic stage before its first metric is emitted."""
        if self._ttft_emitted or self._finished:
            return
        self.stage = stage

    async def mark_ttft(self) -> None:
        """Emit TTFT immediately on the first meaningful streamed token."""
        if self._ttft_emitted or self.stage not in _PROCESSORS:
            return
        self._ttft_emitted = True
        value = _elapsed_seconds(self._started_ns)
        await self._coordinator.emit_metric(
            StageTTFTMetricsData(
                processor=_PROCESSORS[self.stage],
                model=self.model,
                value=value,
                stage=self.stage,
                turn_id=self.turn_id,
                invocation_id=self.invocation_id,
                parent_invocation_id=self.parent_invocation_id,
                attempt=self.attempt,
            )
        )

    async def finish(self, outcome: MetricOutcome = "success") -> None:
        """Emit one terminal processing or latency measurement."""
        if self._finished:
            return
        self._finished = True
        processor = _processor_for(self.stage, self.tool_name)
        if processor is None:
            return
        await self._coordinator.emit_metric(
            StageProcessingMetricsData(
                processor=processor,
                model=self.model,
                value=_elapsed_seconds(self._started_ns),
                metric="latency" if self.stage == "backend_tool_call" else "processing",
                stage=self.stage,
                turn_id=self.turn_id,
                invocation_id=self.invocation_id,
                parent_invocation_id=self.parent_invocation_id,
                attempt=self.attempt,
                outcome=outcome,
                tool_name=self.tool_name,
            )
        )


class StageMetricsCoordinator:
    """Session-local correlation and emission for nested agent stages."""

    def __init__(self, emit_metric: MetricEmitter, emit_server_event: ServerEventEmitter | None = None) -> None:
        """Create a coordinator for one pipeline session."""
        self._emit_metric = emit_metric
        self._emit_server_event = emit_server_event
        self._lock = asyncio.Lock()
        self._turn_counter = 0
        self._invocation_counter = 0
        self._tool_turns: dict[str, str] = {}
        self._backend_turns: dict[str, str] = {}
        self._tool_backends: dict[str, str] = {}

    async def emit_metric(self, metric: TTFBMetricsData | ProcessingMetricsData) -> None:
        """Send one metric through the normal Pipecat/RTVI metrics path."""
        await self._emit_metric(MetricsFrame(data=[metric]))

    async def start_frontend_initial(self, model: str | None) -> StageSpan:
        """Start an unclassified initial Talker request for one user turn."""
        async with self._lock:
            self._turn_counter += 1
            turn_id = f"turn-{self._turn_counter}"
            invocation_id = self._next_invocation_locked("frontend")
        return StageSpan(
            self,
            stage="frontend_initial",
            model=model,
            turn_id=turn_id,
            invocation_id=invocation_id,
        )

    async def start_frontend_final(self, tool_call_id: str, model: str | None) -> StageSpan:
        """Start the post-tool Talker request correlated to its initial call."""
        async with self._lock:
            turn_id = self._tool_turns.get(tool_call_id)
            if turn_id is None:
                turn_id = self._next_orphan_turn_locked()
            invocation_id = self._next_invocation_locked("frontend-final")
        return StageSpan(
            self,
            stage="frontend_final_response",
            model=model,
            turn_id=turn_id,
            invocation_id=invocation_id,
            parent_invocation_id=tool_call_id,
        )

    async def bind_tool_call(self, tool_call_id: str, turn_id: str) -> None:
        """Bind Pipecat's native tool id to the originating frontend turn."""
        if not tool_call_id:
            return
        async with self._lock:
            self._tool_turns[tool_call_id] = turn_id

    async def bind_backend_call(self, tool_call_id: str, backend_call_id: str) -> None:
        """Bind the internal Thinker call to the native frontend tool call."""
        async with self._lock:
            turn_id = self._tool_turns.get(tool_call_id)
            if turn_id is None:
                turn_id = self._next_orphan_turn_locked()
            self._backend_turns[backend_call_id] = turn_id
            self._tool_backends[tool_call_id] = backend_call_id

    async def start_backend(
        self,
        backend_call_id: str,
        *,
        model: str | None,
        attempt: int,
    ) -> StageSpan:
        """Start one streamed Thinker attempt."""
        async with self._lock:
            turn_id = self._backend_turns.get(backend_call_id)
            if turn_id is None:
                turn_id = self._next_orphan_turn_locked()
            invocation_id = self._next_invocation_locked("thinker")
        return StageSpan(
            self,
            stage="backend_thinker",
            model=model,
            turn_id=turn_id,
            invocation_id=invocation_id,
            parent_invocation_id=backend_call_id,
            attempt=attempt,
        )

    async def start_tool(
        self,
        backend_call_id: str,
        *,
        tool_name: str,
        ordinal: int,
    ) -> StageSpan:
        """Start one allowlisted internal tool execution."""
        async with self._lock:
            turn_id = self._backend_turns.get(backend_call_id)
            if turn_id is None:
                turn_id = self._next_orphan_turn_locked()
            invocation_id = f"{backend_call_id}:tool-{ordinal}"
        if self._emit_server_event is not None:
            await self._emit_server_event(
                {
                    "type": "tool-call",
                    "tool": tool_name,
                    "invocation_id": invocation_id,
                    "parent_invocation_id": backend_call_id,
                }
            )
        return StageSpan(
            self,
            stage="backend_tool_call",
            model=None,
            turn_id=turn_id,
            invocation_id=invocation_id,
            parent_invocation_id=backend_call_id,
            tool_name=tool_name,
        )

    async def finish_tool(self, span: StageSpan, outcome: MetricOutcome) -> None:
        """Finish one tool span and close its UI lifecycle marker."""
        await span.finish(outcome)
        if self._emit_server_event is not None:
            await self._emit_server_event(
                {
                    "type": "tool-call-done",
                    "tool": span.tool_name,
                    "invocation_id": span.invocation_id,
                    "parent_invocation_id": span.parent_invocation_id,
                    "outcome": outcome,
                }
            )

    async def cleanup_tool_call(self, tool_call_id: str) -> None:
        """Bound correlation state after success, failure, cancellation, or timeout."""
        async with self._lock:
            self._tool_turns.pop(tool_call_id, None)
            backend_call_id = self._tool_backends.pop(tool_call_id, None)
            if backend_call_id:
                self._backend_turns.pop(backend_call_id, None)

    def _next_invocation_locked(self, prefix: str) -> str:
        self._invocation_counter += 1
        return f"{prefix}-{self._invocation_counter}"

    def _next_orphan_turn_locked(self) -> str:
        self._turn_counter += 1
        return f"turn-{self._turn_counter}"


async def run_streamed_inference(
    llm,
    context: LLMContext,
    span: StageSpan | None,
    *,
    max_tokens: int | None = None,
) -> str:
    """Collect an out-of-pipeline streamed response while measuring true TTFT."""
    stream = await _open_out_of_band_stream(llm, context, max_tokens=max_tokens)
    parts: list[str] = []
    outcome: MetricOutcome = "success"
    try:
        async for chunk in stream:
            if span is not None and _chunk_has_semantic_token(chunk):
                await span.mark_ttft()
            content = _chunk_content(chunk)
            if content:
                parts.append(content)
    except asyncio.CancelledError:
        outcome = "cancelled"
        raise
    except TimeoutError:
        outcome = "timeout"
        raise
    except Exception:
        outcome = "error"
        raise
    finally:
        if span is not None:
            await span.finish(outcome)
        await _close_stream(stream)
    return "".join(parts)


async def _open_out_of_band_stream(llm, context: LLMContext, *, max_tokens: int | None):
    """Open a stream without invoking pipeline-only reasoning-frame side effects."""
    adapter_getter = getattr(llm, "get_llm_adapter", None)
    settings = getattr(llm, "_settings", None)
    client = getattr(llm, "_client", None)
    if adapter_getter is None or settings is None or client is None:
        return await llm.get_chat_completions(context)

    adapter = adapter_getter()
    invocation_params = adapter.get_llm_invocation_params(
        context,
        system_instruction=settings.system_instruction,
        convert_developer_to_user=not llm.supports_developer_role,
    )
    params = llm.build_chat_completion_params(invocation_params)
    params["stream"] = True
    if max_tokens is not None:
        token_key = "max_completion_tokens" if "max_completion_tokens" in params else "max_tokens"
        params[token_key] = max_tokens
    return await client.chat.completions.create(**params)


def _chunk_has_semantic_token(chunk: ChatCompletionChunk) -> bool:
    choices = getattr(chunk, "choices", None)
    if not choices:
        return False
    delta = getattr(choices[0], "delta", None)
    if delta is None:
        return False
    return bool(
        str(getattr(delta, "content", "") or "").strip()
        or str(getattr(delta, "reasoning_content", "") or "").strip()
        or str(getattr(delta, "reasoning", "") or "").strip()
        or getattr(delta, "tool_calls", None)
    )


def _chunk_content(chunk: ChatCompletionChunk) -> str:
    choices = getattr(chunk, "choices", None)
    delta = getattr(choices[0], "delta", None) if choices else None
    content = getattr(delta, "content", None) if delta is not None else None
    return content if isinstance(content, str) else ""


async def _close_stream(stream) -> None:
    close = getattr(stream, "aclose", None) or getattr(stream, "close", None)
    if close is None:
        return
    result = close()
    if inspect.isawaitable(result):
        await result


def _processor_for(stage: str, tool_name: str | None) -> str | None:
    if stage == "backend_tool_call":
        return f"backend_tool_call.{tool_name or 'unknown'}"
    return _PROCESSORS.get(stage)


def _elapsed_seconds(started_ns: int) -> float:
    return max(0.0, (time.perf_counter_ns() - started_ns) / 1_000_000_000)
