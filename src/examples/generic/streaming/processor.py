# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pipecat adapter for one native vLLM StreamingInput session per user turn."""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections.abc import Awaitable, Callable, Coroutine
from copy import deepcopy
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from loguru import logger
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from websockets.asyncio.client import ClientConnection, connect

from examples.generic.streaming.transcript import normalize_transcript

ResponseCallback = Callable[[], Awaitable[None]]
DeltaCallback = Callable[[str], Awaitable[None]]
UsageCallback = Callable[[int, int], Awaitable[None]]


def merge_finalized_segment(committed: str, segment: str) -> str:
    """Merge an independent or cumulative finalized ASR segment."""
    current = normalize_transcript(committed)
    incoming = normalize_transcript(segment)
    if not incoming:
        return current
    if not current or incoming.startswith(f"{current} "):
        return incoming
    if incoming == current or current.endswith(f" {incoming}"):
        return current
    return f"{current} {incoming}"


def combine_interim_segment(committed: str, interim: str) -> str:
    """Build the latest cumulative prefix without committing interim text."""
    current = normalize_transcript(committed)
    incoming = normalize_transcript(interim)
    if not incoming:
        return current
    if not current or incoming == current or incoming.startswith(f"{current} "):
        return incoming
    if current.endswith(f" {incoming}"):
        return current
    return f"{current} {incoming}"


def word_prefixes(previous: str, current: str) -> list[str]:
    """Expand a cumulative ASR jump into one update per newly observed word."""
    before = normalize_transcript(previous)
    after = normalize_transcript(current)
    if not after or after == before:
        return []
    if before and not after.startswith(f"{before} "):
        return [after]
    words = after.split()
    start = len(before.split()) + 1
    return [" ".join(words[:index]) for index in range(start, len(words) + 1)]


@dataclass(frozen=True, slots=True)
class StreamingResult:
    """Completed final response from the native model session."""

    content: str
    ttft_ms: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0


class VLLMStreamingClient:
    """Use one model WebSocket and one vLLM request for one spoken turn."""

    def __init__(
        self,
        *,
        url: str,
        system_prompt: str,
        history: list[dict[str, str]],
        hold_words: int,
        final_max_tokens: int,
        on_first_token: ResponseCallback,
        on_delta: DeltaCallback,
    ) -> None:
        """Store immutable session settings until the connection opens."""
        self._url = url
        self._system_prompt = system_prompt
        self._history = history
        self._hold_words = hold_words
        self._final_max_tokens = final_max_tokens
        self._on_first_token = on_first_token
        self._on_delta = on_delta
        self._websocket: ClientConnection | None = None
        self._reader: asyncio.Task[None] | None = None
        self._pending: dict[tuple[str, str], asyncio.Future[dict[str, Any]]] = {}
        self._prefilled: asyncio.Future[dict[str, Any]] | None = None
        self._send_lock = asyncio.Lock()

    async def _send(self, payload: dict[str, Any]) -> None:
        if self._websocket is None:
            raise RuntimeError("Streaming LLM session is not connected")
        async with self._send_lock:
            await self._websocket.send(json.dumps(payload))

    def _expect(self, event_type: str, command_id: str) -> asyncio.Future[dict[str, Any]]:
        future = asyncio.get_running_loop().create_future()
        self._pending[(event_type, command_id)] = future
        return future

    async def start(self) -> None:
        """Open the turn session and send its prompt and conversation history."""
        self._websocket = await connect(self._url, open_timeout=10, max_size=4 * 1024 * 1024)
        self._reader = asyncio.create_task(self._read_events())
        command_id = str(uuid4())
        started = self._expect("session.started", command_id)
        self._prefilled = self._expect("session.prefilled", command_id)
        await self._send(
            {
                "type": "session.start",
                "command_id": command_id,
                "session_id": str(uuid4()),
                "system_prompt": self._system_prompt,
                "history": self._history,
                "hold_words": self._hold_words,
                "final_max_tokens": self._final_max_tokens,
            }
        )
        await asyncio.wait_for(started, timeout=10)

    async def wait_prefilled(self, *, timeout: float = 10) -> dict[str, Any]:
        """Wait until the model has materialized the stable prompt prefix."""
        if self._prefilled is None:
            raise RuntimeError("Streaming LLM prefill was not started")
        return await asyncio.wait_for(asyncio.shield(self._prefilled), timeout=timeout)

    async def append(self, transcript: str) -> None:
        """Append the latest cumulative ASR hypothesis."""
        await self._send(
            {
                "type": "input.append",
                "command_id": str(uuid4()),
                "transcript": transcript,
            }
        )

    async def finalize(self, transcript: str) -> StreamingResult:
        """Finalize the current turn and await its complete response."""
        command_id = str(uuid4())
        completed = self._expect("response.done", command_id)
        await self._send(
            {
                "type": "input.finalize",
                "command_id": command_id,
                "transcript": transcript,
            }
        )
        payload = await asyncio.wait_for(completed, timeout=120)
        return StreamingResult(
            content=str(payload.get("response", "")).strip(),
            ttft_ms=float(payload["ttft_ms"]) if payload.get("ttft_ms") is not None else None,
            prompt_tokens=int(payload.get("prompt_tokens", 0)),
            completion_tokens=int(payload.get("completion_tokens", 0)),
        )

    async def _read_events(self) -> None:
        try:
            assert self._websocket is not None
            async for raw in self._websocket:
                message = json.loads(raw)
                if not isinstance(message, dict):
                    continue
                event_type = str(message.get("type", ""))
                command_id = str(message.get("command_id", ""))
                if event_type == "response.first_token":
                    await self._on_first_token()
                elif event_type == "response.delta" and message.get("delta"):
                    await self._on_delta(str(message["delta"]))
                future = self._pending.pop((event_type, command_id), None)
                if future is not None and not future.done():
                    future.set_result(message)
                if event_type == "session.error":
                    raise RuntimeError(str(message.get("message", "Streaming LLM session failed")))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(exc)
            self._pending.clear()

    async def close(self) -> None:
        """Close the model session and resolve no further callbacks."""
        if self._websocket is not None:
            with contextlib.suppress(Exception):
                await self._send({"type": "session.close", "command_id": str(uuid4())})
            with contextlib.suppress(Exception):
                await self._websocket.close()
        if self._reader is not None and not self._reader.done():
            self._reader.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reader
        for future in self._pending.values():
            if not future.done():
                future.cancel()
        self._pending.clear()
        self._prefilled = None


class StreamingReasoner:
    """Maintain turn-local model sessions and bounded multi-turn history."""

    def __init__(
        self,
        *,
        url: str,
        system_prompt: str,
        hold_words: int,
        final_max_tokens: int,
        history_turns: int,
        prefill_next_turn: bool,
        on_response_start: ResponseCallback,
        on_first_token: ResponseCallback,
        on_response_delta: DeltaCallback,
        on_usage: UsageCallback,
        on_response_end: ResponseCallback,
    ) -> None:
        """Configure the streaming model connection and response callbacks."""
        self._url = url
        self._system_prompt = system_prompt
        self._hold_words = hold_words
        self._final_max_tokens = final_max_tokens
        self._history_turns = history_turns
        self._prefill_next_turn = prefill_next_turn
        self._on_response_start = on_response_start
        self._on_first_token = on_first_token
        self._on_response_delta = on_response_delta
        self._on_usage = on_usage
        self._on_response_end = on_response_end
        self._history: list[dict[str, str]] = []
        self._client: VLLMStreamingClient | None = None
        self._client_lock = asyncio.Lock()
        self._client_prepared = False
        self._last_partial = ""
        self._active_response_parts: list[str] = []
        self._generation = 0

    @property
    def history(self) -> list[dict[str, str]]:
        """Return a copy of committed conversation history."""
        return deepcopy(self._history)

    async def begin_turn(self) -> None:
        """Reuse a prepared session or create one from committed history."""
        self._generation += 1
        async with self._client_lock:
            if self._client is not None and self._client_prepared:
                self._client_prepared = False
                self._last_partial = ""
                logger.info("Reusing prefilled vLLM session for the new user turn")
                return
            await self._close_client_locked()
            client = self._new_client()
            try:
                await client.start()
            except asyncio.CancelledError:
                await client.close()
                raise
            except Exception:
                await client.close()
                raise
            self._client = client
            self._client_prepared = False
            self._last_partial = ""

    def _new_client(self) -> VLLMStreamingClient:
        return VLLMStreamingClient(
            url=self._url,
            system_prompt=self._system_prompt,
            history=self.history,
            hold_words=self._hold_words,
            final_max_tokens=self._final_max_tokens,
            on_first_token=self._on_first_token,
            on_delta=self._handle_response_delta,
        )

    async def _prepare_next_turn(self, generation: int) -> None:
        """Prefill updated system/history state while the current answer plays."""
        client: VLLMStreamingClient | None = None
        try:
            async with self._client_lock:
                if generation != self._generation:
                    return
                await self._close_client_locked()
                client = self._new_client()
                await client.start()
                payload = await client.wait_prefilled()
                if generation != self._generation:
                    await client.close()
                    return
                self._client = client
                self._client_prepared = True
                self._last_partial = ""
                logger.info(
                    "Prefilled next-turn system/history prefix: {} tokens, model TTFT={} ms",
                    payload.get("prefix_tokens", 0),
                    payload.get("ttft_ms"),
                )
        except asyncio.CancelledError:
            if client is not None and client is not self._client:
                await client.close()
            raise
        except Exception as exc:
            if client is not None and client is not self._client:
                await client.close()
            logger.warning(f"Next-turn prefix prefill failed; falling back to on-demand startup: {exc}")

    async def _handle_response_delta(self, delta: str) -> None:
        """Remember exactly what was emitted before forwarding it to TTS."""
        self._active_response_parts.append(delta)
        await self._on_response_delta(delta)

    async def handle_partial(self, transcript: str) -> None:
        """Send one cumulative model update for every newly observed ASR word."""
        normalized = normalize_transcript(transcript)
        if not normalized or normalized == self._last_partial:
            return
        if self._client is None:
            await self.begin_turn()
        self._client_prepared = False
        assert self._client is not None
        for prefix in word_prefixes(self._last_partial, normalized):
            await self._client.append(prefix)
            self._last_partial = prefix

    async def finalize(self, transcript: str) -> StreamingResult | None:
        """Generate and stream the response after semantic turn finalization."""
        normalized = normalize_transcript(transcript)
        if not normalized:
            return None
        if self._client is None:
            await self.begin_turn()
        self._client_prepared = False
        generation = self._generation
        self._active_response_parts.clear()
        await self._on_response_start()
        try:
            assert self._client is not None
            result = await self._client.finalize(normalized)
            await self._on_usage(result.prompt_tokens, result.completion_tokens)
        except asyncio.CancelledError:
            spoken_prefix = "".join(self._active_response_parts).strip()
            interrupted_response = (
                f"{spoken_prefix}\n[Response interrupted by the user.]"
                if spoken_prefix
                else "[Response interrupted before any answer was spoken.]"
            )
            self._append_history(normalized, interrupted_response)
            logger.info("Committed interrupted streaming turn to conversation history")
            raise
        finally:
            self._active_response_parts.clear()
            await self._on_response_end()
        if generation != self._generation:
            return None
        if result.content:
            self._append_history(normalized, result.content)
        logger.debug(f"vLLM engine-internal final-chunk TTFT: {result.ttft_ms} ms")
        if result.content and self._prefill_next_turn:
            await self._prepare_next_turn(generation)
        return result

    def _append_history(self, transcript: str, response: str) -> None:
        if self._history_turns == 0:
            return
        self._history.extend(
            [
                {"role": "user", "content": transcript},
                {"role": "assistant", "content": response},
            ]
        )
        del self._history[: max(0, len(self._history) - self._history_turns * 2)]

    async def close_client(self) -> None:
        """Close only the current turn while preserving conversation history."""
        async with self._client_lock:
            await self._close_client_locked()

    async def _close_client_locked(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None
        self._client_prepared = False

    async def close(self) -> None:
        """Invalidate active work and close the current turn."""
        self._generation += 1
        await self.close_client()


class StreamingLLMProcessor(FrameProcessor):
    """Convert ASR partials and VAD-final turns into streaming LLM/TTS frames."""

    def __init__(
        self,
        *,
        url: str,
        system_prompt: str,
        hold_words: int,
        final_max_tokens: int,
        history_turns: int,
        prefill_next_turn: bool = True,
        name: str = "generic-streaming-llm",
    ) -> None:
        """Create a Pipecat processor around the native streaming client."""
        super().__init__(name=name)
        self._committed_transcript = ""
        self._latest_interim = ""
        self._turn_finalized = False
        self._ttft_started_at: float | None = None
        self._response_open = False
        self._response_frame_lock = asyncio.Lock()
        self._background: set[asyncio.Task[Any]] = set()
        self._reasoner = StreamingReasoner(
            url=url,
            system_prompt=system_prompt,
            hold_words=hold_words,
            final_max_tokens=final_max_tokens,
            history_turns=history_turns,
            prefill_next_turn=prefill_next_turn,
            on_response_start=self._response_start,
            on_first_token=self._response_first_token,
            on_response_delta=self._response_delta,
            on_usage=self._response_usage,
            on_response_end=self._response_end,
        )

    def can_generate_metrics(self) -> bool:
        """Report native streaming LLM timing and token usage metrics."""
        return True

    async def _response_start(self) -> None:
        async with self._response_frame_lock:
            self._response_open = True
            await self.push_frame(LLMFullResponseStartFrame())
            await self.start_processing_metrics()
            # Match Pipecat's built-in LLM boundary: start immediately before the
            # inference request is sent, after the response-start frame is emitted.
            started_at = time.time()
            self._ttft_started_at = started_at
            await self.start_ttfb_metrics(start_time=started_at)

    async def _stop_ttfb_on_first_output(self) -> None:
        """Stop and log TTFT at the first actual model output event."""
        if self._ttft_started_at is None:
            return
        first_output_at = time.time()
        started_at = self._ttft_started_at
        self._ttft_started_at = None
        await self.stop_ttfb_metrics(end_time=first_output_at)
        logger.info(
            "LLM TTFT (Pipecat-aligned request to first model output): {:.3f} ms",
            (first_output_at - started_at) * 1000,
        )

    async def _response_first_token(self) -> None:
        async with self._response_frame_lock:
            if self._response_open:
                await self._stop_ttfb_on_first_output()

    async def _response_delta(self, delta: str) -> None:
        async with self._response_frame_lock:
            if not self._response_open:
                return
            # Fallback for a model server that predates response.first_token.
            await self._stop_ttfb_on_first_output()
            await self.push_frame(LLMTextFrame(text=delta))

    async def _response_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        await self.start_llm_usage_metrics(
            LLMTokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            )
        )

    async def _response_end(self) -> None:
        async with self._response_frame_lock:
            if not self._response_open:
                return
            self._response_open = False
            if self._ttft_started_at is not None:
                self._ttft_started_at = None
                cancel_ttfb_metrics = getattr(self, "cancel_ttfb_metrics", None)
                if cancel_ttfb_metrics is not None:
                    await cancel_ttfb_metrics()
                logger.warning("LLM response ended before any model output; TTFT was not reported")
            await self.stop_processing_metrics()
            await self.push_frame(LLMFullResponseEndFrame())

    async def _abort_response(self) -> None:
        """Close local response state without emitting a stale end after interruption."""
        async with self._response_frame_lock:
            if not self._response_open:
                return
            self._response_open = False
            if self._ttft_started_at is not None:
                self._ttft_started_at = None
                cancel_ttfb_metrics = getattr(self, "cancel_ttfb_metrics", None)
                if cancel_ttfb_metrics is not None:
                    await cancel_ttfb_metrics()
            await self.stop_processing_metrics()

    def _start_background(self, coroutine: Coroutine[Any, Any, Any]) -> None:
        task = asyncio.create_task(coroutine)
        self._background.add(task)
        task.add_done_callback(self._background_done)

    def _background_done(self, task: asyncio.Task[Any]) -> None:
        self._background.discard(task)
        if task.cancelled():
            return
        try:
            task.result()
        except Exception as exc:
            logger.error(f"Streaming LLM background task failed: {exc}")

    async def _cancel_background(self) -> None:
        tasks = list(self._background)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Process speech lifecycle frames while forwarding them to the existing UI."""
        await super().process_frame(frame, direction)

        if isinstance(frame, InterruptionFrame):
            # Close the response gate first. This guarantees that the reader and
            # the cancelled finalize task cannot emit text/end frames after TTS
            # has torn down the interrupted audio context.
            response_was_open = self._response_open
            await self._abort_response()
            await self.push_frame(frame, direction)
            # UserTurnProcessor broadcasts an InterruptionFrame at the start of
            # every normal turn, even after the previous answer has completed.
            # Preserve the already-running/ready next-turn history prefill in
            # that case. Only tear down model work when an answer was genuinely
            # interrupted while it was still being generated.
            if response_was_open:
                await self._cancel_background()
                await self._reasoner.close_client()
            return

        if isinstance(frame, UserStartedSpeakingFrame):
            # Do not make barge-in wait for the model WebSocket handshake.
            response_was_open = self._response_open
            await self._abort_response()
            await self.push_frame(frame, direction)
            if response_was_open:
                await self._cancel_background()
                await self._reasoner.close_client()
            self._committed_transcript = ""
            self._latest_interim = ""
            self._turn_finalized = False
            try:
                await self._reasoner.begin_turn()
            except Exception as exc:
                logger.error(f"Unable to open streaming LLM turn: {exc}")
            return

        elif isinstance(frame, InterimTranscriptionFrame):
            self._latest_interim = normalize_transcript(frame.text)
            prefix = combine_interim_segment(self._committed_transcript, self._latest_interim)
            if prefix:
                try:
                    await self._reasoner.handle_partial(prefix)
                except Exception as exc:
                    logger.error(f"Unable to append streaming ASR update: {exc}")

        elif isinstance(frame, TranscriptionFrame):
            self._committed_transcript = merge_finalized_segment(self._committed_transcript, frame.text)
            self._latest_interim = ""
            if self._committed_transcript:
                try:
                    await self._reasoner.handle_partial(self._committed_transcript)
                except Exception as exc:
                    logger.error(f"Unable to append finalized ASR segment: {exc}")

        elif isinstance(frame, UserStoppedSpeakingFrame) and not self._turn_finalized:
            transcript = combine_interim_segment(self._committed_transcript, self._latest_interim)
            self._turn_finalized = True
            # Preserve Pipecat's normal turn ordering: every downstream service
            # observes the user-stop boundary before response-start/text frames.
            await self.push_frame(frame, direction)
            if transcript:
                self._start_background(self._reasoner.finalize(transcript))
            return

        elif isinstance(frame, (CancelFrame, EndFrame)):
            await self._abort_response()
            await self._cancel_background()
            await self._reasoner.close()

        await self.push_frame(frame, direction)

    async def cleanup(self) -> None:
        """Cancel final generation and release the model WebSocket."""
        await self._abort_response()
        await self._cancel_background()
        await self._reasoner.close()
        await super().cleanup()
