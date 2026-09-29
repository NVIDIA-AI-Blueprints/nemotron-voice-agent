# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Turn-scoped NVIDIA ASR streams for the OpenAI Realtime transport."""

from __future__ import annotations

import asyncio
import math
import threading
from collections import deque
from collections.abc import AsyncGenerator, Generator, Iterator
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import grpc
from loguru import logger
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.nvidia.stt import NvidiaSTTService
from pipecat.services.settings import assert_given
from pipecat.services.stt_service import STTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from riva.client.proto import riva_asr_pb2 as rasr
from riva.client.proto import riva_common_pb2 as rcommon

from examples.shared.frames import USER_TRANSCRIPT_TURN_FRAME_ID_METADATA
from realtime.audio import MAX_PENDING_INPUT_BYTES, MAX_PENDING_INPUT_SECONDS
from realtime.frames import (
    RealtimeASRTurnEndedFrame,
    RealtimeInputTranscriptionErrorFrame,
    RealtimeManualUserStartedSpeakingFrame,
    RealtimeManualUserStoppedSpeakingFrame,
)
from realtime.vad import (
    RealtimeVADConfigurationFrame,
    RealtimeVADUserStartedSpeakingFrame,
    RealtimeVADUserStoppedSpeakingFrame,
)

_STREAMING_AUDIO_CHUNK_SECONDS = 0.1


class RealtimeASROwnershipError(ValueError):
    """Report an invalid Realtime ASR turn boundary."""

    def __init__(self, *, code: str, message: str) -> None:
        """Store a stable public error code and message."""
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class _RealtimeAudioChunk:
    audio: bytes
    bootstrap: bool = False


class _RealtimeAudioBufferFull(BufferError):
    """The native ASR iterator reached one of its byte-exact budgets."""


class _RealtimeAudioIterator(Iterator[_RealtimeAudioChunk]):
    """Bridge ordered Pipecat audio into one synchronous NVIDIA request."""

    _SENTINEL = object()

    def __init__(
        self,
        *,
        max_bootstrap_bytes: int = 0,
        max_stream_pending_bytes: int = MAX_PENDING_INPUT_BYTES,
    ) -> None:
        for name, value, allow_zero in (
            ("max_bootstrap_bytes", max_bootstrap_bytes, True),
            ("max_stream_pending_bytes", max_stream_pending_bytes, False),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0 or (not allow_zero and value == 0):
                qualifier = "non-negative" if allow_zero else "positive"
                raise ValueError(f"{name} must be a {qualifier} integer")
        self._max_bootstrap_bytes = max_bootstrap_bytes
        self._max_stream_pending_bytes = max_stream_pending_bytes
        self._pending_bootstrap_bytes = 0
        self._pending_stream_bytes = 0
        self._queue: deque[_RealtimeAudioChunk | object] = deque()
        self._condition = threading.Condition()
        self._accepting = True
        self._exhausted = False

    def put(self, chunk: _RealtimeAudioChunk) -> None:
        """Append one chunk unless the request has already been closed."""
        self.put_many((chunk,))

    def put_many(self, chunks: tuple[_RealtimeAudioChunk, ...]) -> None:
        """Atomically append chunks within their independent byte budgets."""
        bootstrap_bytes = sum(len(chunk.audio) for chunk in chunks if chunk.bootstrap)
        stream_bytes = sum(len(chunk.audio) for chunk in chunks if not chunk.bootstrap)
        with self._condition:
            if not self._accepting:
                raise RuntimeError("Realtime ASR turn stream is already closed")
            if self._pending_bootstrap_bytes + bootstrap_bytes > self._max_bootstrap_bytes:
                raise _RealtimeAudioBufferFull("Realtime ASR bootstrap audio exceeded its exact prefix budget")
            if self._pending_stream_bytes + stream_bytes > self._max_stream_pending_bytes:
                raise _RealtimeAudioBufferFull("Realtime ASR pending stream audio exceeded its buffer limit")
            self._pending_bootstrap_bytes += bootstrap_bytes
            self._pending_stream_bytes += stream_bytes
            self._queue.extend(chunks)
            self._condition.notify()

    def close(self) -> None:
        """End the request iterator exactly once."""
        with self._condition:
            if not self._accepting:
                return
            self._accepting = False
            self._queue.append(self._SENTINEL)
            self._condition.notify()

    def __iter__(self) -> _RealtimeAudioIterator:
        return self

    def __next__(self) -> _RealtimeAudioChunk:
        with self._condition:
            if self._exhausted:
                raise StopIteration
            while not self._queue:
                self._condition.wait()
            value = self._queue.popleft()
            if value is self._SENTINEL:
                self._exhausted = True
                raise StopIteration
            chunk = cast("_RealtimeAudioChunk", value)
            if chunk.bootstrap:
                self._pending_bootstrap_bytes -= len(chunk.audio)
            else:
                self._pending_stream_bytes -= len(chunk.audio)
            return chunk


def _streaming_requests(
    chunks: Iterator[_RealtimeAudioChunk],
    streaming_config: rasr.StreamingRecognitionConfig,
    request_id: str,
) -> Generator[rasr.StreamingRecognizeRequest, None, None]:
    """Build one native NVIDIA request stream for one Realtime audio turn."""
    yield rasr.StreamingRecognizeRequest(
        streaming_config=streaming_config,
        id=rcommon.RequestId(value=request_id),
    )
    for chunk in chunks:
        yield rasr.StreamingRecognizeRequest(audio_content=chunk.audio)


@dataclass(slots=True)
class _ProviderEvent:
    kind: str
    value: Any = None


@dataclass(slots=True)
class _RealtimeASRTurn:
    token: int
    request_id: str
    chunks: _RealtimeAudioIterator
    boundary_kind: str = "vad"
    stop_token: int | None = None
    audio_bytes: int = 0
    request_closed: bool = False
    cancel_requested: bool = False
    discard_provider_output: bool = False
    suppress_errors: bool = False
    error_emitted: bool = False
    saw_text: bool = False
    saw_interim: bool = False
    pending_final: Any | None = None
    provider_terminal_queued: bool = False
    provider_terminal: bool = False
    failure_code: str | None = None
    failure_message: str | None = None
    failure_fatal: bool = False
    metrics_closed: bool = False
    terminal_emitted: bool = False
    retired: bool = False
    call: Any | None = None
    worker_thread: threading.Thread | None = None
    worker_task: asyncio.Task[None] | None = None
    publisher_task: asyncio.Task[None] | None = None
    deadline_task: asyncio.Task[None] | None = None
    done: asyncio.Event = field(default_factory=asyncio.Event)
    worker_exited: threading.Event = field(default_factory=threading.Event)
    provider_events: deque[_ProviderEvent] = field(default_factory=deque)
    provider_event_ready: asyncio.Event = field(default_factory=asyncio.Event)
    state_guard: threading.Lock = field(default_factory=threading.Lock)


class RealtimeNvidiaSTTService(NvidiaSTTService):
    """NVIDIA STT with one independently cancellable stream per audio turn.

    Realtime VAD runs immediately before this service and supplies absolute
    sample boundaries. A provider response therefore inherits one immutable
    turn token without relying on provider timing or word-offset heuristics.
    """

    def __init__(
        self,
        *args,
        vad_prefix_padding_secs: float = 0.0,
        turn_drain_timeout_secs: float = 9.0,
        worker_cancel_timeout_secs: float = 1.0,
        max_concurrent_turns: int = 8,
        max_pending_provider_events: int = 128,
        **kwargs,
    ) -> None:
        """Create bounded Realtime-only ASR state."""
        for name, value, allow_zero in (
            ("vad_prefix_padding_secs", vad_prefix_padding_secs, True),
            ("turn_drain_timeout_secs", turn_drain_timeout_secs, False),
            ("worker_cancel_timeout_secs", worker_cancel_timeout_secs, False),
        ):
            if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be a finite number")
            if value < 0 or (not allow_zero and value == 0):
                qualifier = "non-negative" if allow_zero else "positive"
                raise ValueError(f"{name} must be {qualifier}")
        if vad_prefix_padding_secs > MAX_PENDING_INPUT_SECONDS:
            raise ValueError(f"vad_prefix_padding_secs must be at most {MAX_PENDING_INPUT_SECONDS} seconds")
        for name, value in (
            ("max_concurrent_turns", max_concurrent_turns),
            ("max_pending_provider_events", max_pending_provider_events),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

        # Realtime input mixes system-frame audio/boundaries with control-frame
        # configuration and release signals. Pipecat's normal dual-queue path
        # can prioritize a later system frame over an already accepted control
        # frame, so preserve the upstream sequencer's serialized call order.
        kwargs["enable_direct_mode"] = True
        super().__init__(*args, **kwargs)
        self._realtime_prefix_samples = 0
        self._realtime_vad_prefix_padding_secs = float(vad_prefix_padding_secs)
        self._realtime_turn_drain_timeout_secs = float(turn_drain_timeout_secs)
        self._realtime_worker_cancel_timeout_secs = float(worker_cancel_timeout_secs)
        self._realtime_max_concurrent_turns = max_concurrent_turns
        self._realtime_max_pending_provider_events = max_pending_provider_events
        self._realtime_audio_cursor = 0
        self._realtime_pre_roll_start = 0
        self._realtime_pre_roll = bytearray()
        self._realtime_active_turn: _RealtimeASRTurn | None = None
        self._realtime_turns: dict[str, _RealtimeASRTurn] = {}
        self._realtime_stream_sequence = 0
        self._realtime_shutting_down = False

    async def start(self, frame: StartFrame) -> None:
        """Initialize NVIDIA configuration without opening a shared stream."""
        await STTService.start(self, frame)
        self._initialize_client()
        self._config = self._create_recognition_config()
        self._realtime_prefix_samples = round(self._realtime_vad_prefix_padding_secs * self.sample_rate)
        self._realtime_audio_cursor = 0
        self._realtime_pre_roll_start = 0
        self._realtime_pre_roll.clear()
        self._realtime_shutting_down = False
        logger.debug(f"Initialized RealtimeNvidiaSTTService with model: {self._settings.model}")

    async def stop(self, frame: EndFrame) -> None:
        """Cancel native turn streams and flush STT usage."""
        await self._shutdown_turns()
        await STTService.stop(self, frame)

    async def cancel(self, frame: CancelFrame) -> None:
        """Cancel native turn streams and flush STT usage."""
        await self._shutdown_turns()
        await STTService.cancel(self, frame)

    async def cleanup(self) -> None:
        """Release every native turn stream."""
        await self._shutdown_turns()
        await STTService.cleanup(self)

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Open and close native streams at typed Realtime boundaries."""
        try:
            if direction == FrameDirection.DOWNSTREAM and isinstance(
                frame,
                (RealtimeManualUserStartedSpeakingFrame, RealtimeVADUserStartedSpeakingFrame),
            ):
                turn = await self._start_turn(frame)
                await STTService.process_frame(self, frame, direction)
                if turn is not None:
                    self._launch_turn(turn)
                return

            if direction == FrameDirection.DOWNSTREAM and isinstance(frame, RealtimeVADConfigurationFrame):
                if frame.sample_rate != self.sample_rate:
                    raise RealtimeASROwnershipError(
                        code="asr_turn_boundary_invalid",
                        message="Realtime VAD and NVIDIA ASR use different sample rates",
                    )
                maximum_lookback_samples = MAX_PENDING_INPUT_SECONDS * 2 * self.sample_rate
                if frame.start_lookback_samples > maximum_lookback_samples:
                    raise RealtimeASROwnershipError(
                        code="input_buffer_overflow",
                        message="Realtime VAD prefix exceeds the pipeline audio buffer limit",
                    )
                self._realtime_prefix_samples = frame.start_lookback_samples
                await STTService.process_frame(self, frame, direction)
                return

            if direction == FrameDirection.DOWNSTREAM and isinstance(frame, RealtimeManualUserStoppedSpeakingFrame):
                turn = self._validate_active_turn_stop(frame=frame, required=True)
                await STTService.process_frame(self, frame, direction)
                self._close_active_turn(frame=frame, required=True, validated_turn=turn)
                return

            if direction == FrameDirection.DOWNSTREAM and isinstance(frame, RealtimeVADUserStoppedSpeakingFrame):
                turn = self._validate_active_turn_stop(frame=frame, required=True)
                await STTService.process_frame(self, frame, direction)
                self._close_active_turn(frame=frame, required=True, validated_turn=turn)
                return

            await STTService.process_frame(self, frame, direction)
        except RealtimeASROwnershipError as exc:
            await self._emit_error(exc.code, exc.message)

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        """Submit each PCM byte to its exact active turn or prefix buffer."""
        if not audio:
            yield None
            return
        if len(audio) % self._bytes_per_sample():
            await self._emit_error("asr_audio_invalid", "Realtime ASR received an incomplete PCM sample")
            yield None
            return

        sample_count = len(audio) // self._bytes_per_sample()
        turn = self._realtime_active_turn
        if turn is None:
            self._append_pre_roll(audio, sample_count=sample_count)
        else:
            try:
                self._queue_turn_audio(turn, audio)
            except RealtimeASROwnershipError as exc:
                self._queue_turn_failure(turn, code=exc.code, message=exc.message)
        self._realtime_audio_cursor += sample_count
        yield None

    def _record_stt_audio_usage(self, audio: bytes | bytearray) -> None:
        """Defer usage accounting to the exact native turn lifecycle."""
        # The base service counts every input frame before ``run_stt``. Realtime
        # also receives prefix and trailing audio that may never reach NVIDIA,
        # so ``_close_turn_metrics`` accounts only bytes owned by a native turn.
        return

    async def _start_turn(
        self,
        frame: RealtimeManualUserStartedSpeakingFrame | RealtimeVADUserStartedSpeakingFrame,
    ) -> _RealtimeASRTurn | None:
        if self._realtime_shutting_down:
            return None
        if self._realtime_active_turn is not None:
            raise RealtimeASROwnershipError(
                code="asr_turn_boundary_invalid",
                message="NVIDIA ASR received a new Realtime audio start before the active turn stopped",
            )
        if len(self._realtime_turns) >= self._realtime_max_concurrent_turns:
            raise RealtimeASROwnershipError(
                code="asr_turn_capacity_exceeded",
                message="NVIDIA ASR exceeded its bounded concurrent Realtime turn capacity",
            )

        if isinstance(frame, RealtimeVADUserStartedSpeakingFrame):
            if frame.sample_rate != self.sample_rate:
                raise RealtimeASROwnershipError(
                    code="asr_turn_boundary_invalid",
                    message="Realtime VAD and NVIDIA ASR use different sample rates",
                )
            start_sample = frame.audio_start_sample
        else:
            if frame.sample_rate != self.sample_rate or frame.audio_sample_cursor != self._realtime_audio_cursor:
                raise RealtimeASROwnershipError(
                    code="asr_turn_boundary_invalid",
                    message="Manual Realtime audio start does not match the NVIDIA ASR input cursor",
                )
            start_sample = frame.audio_sample_cursor

        if start_sample < self._realtime_pre_roll_start or start_sample > self._realtime_audio_cursor:
            raise RealtimeASROwnershipError(
                code="asr_turn_boundary_invalid",
                message="Realtime VAD start is outside the retained NVIDIA ASR audio prefix",
            )
        byte_offset = (start_sample - self._realtime_pre_roll_start) * self._bytes_per_sample()
        prefix = bytes(self._realtime_pre_roll[byte_offset:])

        self._realtime_stream_sequence += 1
        request_id = f"realtime-asr-stream-{self._realtime_stream_sequence}"
        turn = _RealtimeASRTurn(
            token=frame.id,
            request_id=request_id,
            chunks=_RealtimeAudioIterator(max_bootstrap_bytes=len(prefix)),
            boundary_kind="manual" if isinstance(frame, RealtimeManualUserStartedSpeakingFrame) else "vad",
        )
        self._realtime_active_turn = turn
        self._realtime_turns[request_id] = turn
        await self.start_processing_metrics()
        self._realtime_pre_roll.clear()
        self._realtime_pre_roll_start = self._realtime_audio_cursor
        if prefix:
            self._queue_turn_audio(turn, prefix, bootstrap=True)
        return turn

    def _launch_turn(self, turn: _RealtimeASRTurn) -> None:
        """Launch native publication only after its start boundary is downstream."""
        turn.publisher_task = self.create_task(
            self._publish_provider_events(turn),
            name=f"{turn.request_id}-publisher",
        )
        turn.worker_task = self.create_task(
            self._run_turn_worker(turn),
            name=f"{turn.request_id}-worker",
        )
        turn.worker_task.add_done_callback(lambda _task, turn=turn: self._release_retired_turn_if_exited(turn))

    def _validate_active_turn_stop(
        self,
        *,
        frame: RealtimeManualUserStoppedSpeakingFrame | RealtimeVADUserStoppedSpeakingFrame,
        required: bool,
    ) -> _RealtimeASRTurn | None:
        """Validate a stop without allowing the native producer to finish yet."""
        turn = self._realtime_active_turn
        if turn is None:
            if required:
                raise RealtimeASROwnershipError(
                    code="asr_turn_boundary_invalid",
                    message="NVIDIA ASR received a Realtime audio stop without a matching start",
                )
            return None
        if frame.sample_rate != self.sample_rate:
            raise RealtimeASROwnershipError(
                code="asr_turn_boundary_invalid",
                message="Realtime audio stop and NVIDIA ASR use different sample rates",
            )
        stop_kind = "manual" if isinstance(frame, RealtimeManualUserStoppedSpeakingFrame) else "vad"
        if turn.boundary_kind != stop_kind:
            raise RealtimeASROwnershipError(
                code="asr_turn_boundary_invalid",
                message="Realtime audio stop type does not match its NVIDIA ASR start",
            )
        stop_sample = (
            frame.audio_sample_cursor
            if isinstance(frame, RealtimeManualUserStoppedSpeakingFrame)
            else frame.audio_stop_sample
        )
        if stop_sample != self._realtime_audio_cursor:
            raise RealtimeASROwnershipError(
                code="asr_turn_boundary_invalid",
                message="Realtime audio stop does not match the NVIDIA ASR input cursor",
            )
        return turn

    def _close_active_turn(
        self,
        *,
        frame: RealtimeManualUserStoppedSpeakingFrame | RealtimeVADUserStoppedSpeakingFrame,
        required: bool,
        validated_turn: _RealtimeASRTurn | None = None,
    ) -> _RealtimeASRTurn | None:
        turn = validated_turn or self._validate_active_turn_stop(frame=frame, required=required)
        if turn is None:
            return None
        if self._realtime_active_turn is not turn:
            raise RealtimeASROwnershipError(
                code="asr_turn_boundary_invalid",
                message="Realtime audio stop no longer owns the active NVIDIA ASR stream",
            )
        self._realtime_active_turn = None
        self._realtime_pre_roll.clear()
        self._realtime_pre_roll_start = self._realtime_audio_cursor
        turn.request_closed = True
        turn.stop_token = frame.id
        turn.chunks.close()
        turn.deadline_task = self.create_task(
            self._enforce_turn_deadline(turn),
            name=f"{turn.request_id}-deadline",
        )
        return turn

    def _queue_turn_audio(self, turn: _RealtimeASRTurn, audio: bytes, *, bootstrap: bool = False) -> None:
        if not audio:
            return
        if turn.request_closed or turn.cancel_requested:
            raise RealtimeASROwnershipError(
                code="asr_turn_boundary_invalid",
                message="NVIDIA ASR received audio for a closed Realtime turn",
            )
        chunk_samples = max(1, round(self.sample_rate * _STREAMING_AUDIO_CHUNK_SECONDS))
        chunk_bytes = chunk_samples * self._bytes_per_sample()
        chunks = tuple(
            _RealtimeAudioChunk(audio[offset : offset + chunk_bytes], bootstrap=bootstrap)
            for offset in range(0, len(audio), chunk_bytes)
        )
        try:
            turn.chunks.put_many(chunks)
        except _RealtimeAudioBufferFull as exc:
            raise RealtimeASROwnershipError(
                code="input_buffer_overflow",
                message="Realtime ASR pending audio exceeded the pipeline buffer limit",
            ) from exc
        turn.audio_bytes += len(audio)

    def _append_pre_roll(self, audio: bytes, *, sample_count: int) -> None:
        limit = self._realtime_prefix_samples
        if limit <= 0:
            self._realtime_pre_roll.clear()
            self._realtime_pre_roll_start = self._realtime_audio_cursor + sample_count
            return
        self._realtime_pre_roll.extend(audio)
        retained_samples = len(self._realtime_pre_roll) // self._bytes_per_sample()
        if retained_samples > limit:
            discard_samples = retained_samples - limit
            discard_bytes = discard_samples * self._bytes_per_sample()
            del self._realtime_pre_roll[:discard_bytes]
            self._realtime_pre_roll_start += discard_samples

    def _bytes_per_sample(self) -> int:
        return 2 * max(1, int(self._audio_channel_count))

    async def _run_turn_worker(self, turn: _RealtimeASRTurn) -> None:
        loop = asyncio.get_running_loop()
        completed = loop.create_future()

        def consume() -> None:
            try:
                self._consume_turn_stream(turn)
            finally:
                with suppress(RuntimeError):
                    loop.call_soon_threadsafe(self._complete_worker_future, completed)

        turn.worker_thread = threading.Thread(
            target=consume,
            name=f"{turn.request_id}-grpc",
            daemon=True,
        )
        turn.worker_thread.start()
        try:
            await asyncio.shield(completed)
        except asyncio.CancelledError:
            self._request_turn_cancel(turn, suppress_errors=True)
            raise
        finally:
            loop.call_soon(self._release_retired_turn_if_exited, turn)

    @staticmethod
    def _complete_worker_future(future: asyncio.Future[None]) -> None:
        if not future.done():
            future.set_result(None)

    def _consume_turn_stream(self, turn: _RealtimeASRTurn) -> None:
        terminal = _ProviderEvent("completed")
        try:
            asr_service = self._asr_service
            config = self._config
            if asr_service is None or config is None:
                raise RuntimeError("NVIDIA ASR client is not initialized")
            requests = _streaming_requests(turn.chunks, config, turn.request_id)
            call = asr_service.stub.StreamingRecognize(
                requests,
                metadata=asr_service.auth.get_auth_metadata(),
            )
            with turn.state_guard:
                turn.call = call
                cancel_requested = turn.cancel_requested
            if cancel_requested:
                call.cancel()
                terminal = _ProviderEvent("cancelled")
                return
            for response in call:
                with turn.state_guard:
                    if turn.cancel_requested:
                        terminal = _ProviderEvent("cancelled")
                        return
                self.get_event_loop().call_soon_threadsafe(
                    self._enqueue_provider_event,
                    turn,
                    _ProviderEvent("response", response),
                )
            with turn.state_guard:
                if turn.cancel_requested:
                    terminal = _ProviderEvent("cancelled")
                elif not turn.request_closed:
                    terminal = _ProviderEvent(
                        "error",
                        (
                            "asr_provider_stream_closed",
                            "NVIDIA ASR closed a Realtime turn before its audio stop boundary",
                        ),
                    )
        except grpc.RpcError as exc:
            with turn.state_guard:
                cancelled = turn.cancel_requested
            if cancelled:
                terminal = _ProviderEvent("cancelled")
            else:
                status = exc.code().name if hasattr(exc, "code") else "UNKNOWN"
                terminal = _ProviderEvent(
                    "error",
                    (
                        "asr_provider_error",
                        f"NVIDIA ASR failed the Realtime turn stream ({status})",
                    ),
                )
        except Exception:  # noqa: BLE001
            with turn.state_guard:
                cancelled = turn.cancel_requested
            if cancelled:
                terminal = _ProviderEvent("cancelled")
            else:
                logger.exception("Realtime NVIDIA ASR turn stream failed")
                terminal = _ProviderEvent(
                    "error",
                    ("asr_provider_error", "NVIDIA ASR failed the Realtime turn stream"),
                )
        finally:
            with turn.state_guard:
                turn.call = None
            turn.worker_exited.set()
            loop = self.get_event_loop()
            with suppress(RuntimeError):
                loop.call_soon_threadsafe(self._enqueue_provider_event, turn, terminal)
            with suppress(RuntimeError):
                loop.call_soon_threadsafe(self._release_retired_turn_if_exited, turn)

    def _enqueue_provider_event(self, turn: _RealtimeASRTurn, event: _ProviderEvent) -> None:
        if turn.retired:
            return
        if event.kind == "response" and (turn.provider_terminal_queued or turn.provider_terminal):
            return
        if event.kind in {"completed", "error", "deadline", "cancelled", "overflow"}:
            if turn.provider_terminal_queued or turn.provider_terminal:
                return
            turn.provider_terminal_queued = True
        if event.kind == "response" and len(turn.provider_events) >= self._realtime_max_pending_provider_events:
            if not turn.provider_terminal_queued:
                turn.provider_terminal_queued = True
                turn.provider_events.append(_ProviderEvent("overflow"))
                self._request_turn_cancel(turn, discard_provider_output=False)
        else:
            turn.provider_events.append(event)
        turn.provider_event_ready.set()

    def _queue_turn_failure(self, turn: _RealtimeASRTurn, *, code: str, message: str) -> None:
        """Cancel a turn and let its sole publisher emit the ordered failure."""
        if (
            turn.retired
            or turn.provider_terminal
            or turn.provider_terminal_queued
            or turn.error_emitted
            or turn.discard_provider_output
            or turn.suppress_errors
        ):
            return
        turn.provider_terminal_queued = True
        self._request_turn_cancel(turn, discard_provider_output=False)
        turn.provider_events.append(_ProviderEvent("error", (code, message)))
        turn.provider_event_ready.set()

    async def _publish_provider_events(self, turn: _RealtimeASRTurn) -> None:
        try:
            while not turn.retired:
                await turn.provider_event_ready.wait()
                turn.provider_event_ready.clear()
                while turn.provider_events and not turn.retired:
                    event = turn.provider_events.popleft()
                    if event.kind == "response":
                        await self._handle_turn_response(turn, event.value)
                    elif event.kind == "completed":
                        turn.provider_terminal = True
                        await self._finish_turn(turn)
                        status: Literal["completed", "empty", "failed"]
                        if turn.error_emitted:
                            status = "failed"
                        elif turn.saw_text:
                            status = "completed"
                        else:
                            status = "empty"
                        await self._emit_turn_terminal(
                            turn,
                            status=status,
                            code=turn.failure_code if status == "failed" else None,
                            message=turn.failure_message if status == "failed" else None,
                        )
                    elif event.kind == "overflow":
                        await self._flush_pending_final(turn, finalized=True)
                        await self._fail_turn(
                            turn,
                            code="asr_provider_backpressure",
                            message="NVIDIA ASR exceeded its bounded Realtime response capacity",
                        )
                        turn.provider_terminal = True
                        await self._emit_turn_terminal(
                            turn,
                            status="failed",
                            code=turn.failure_code,
                            message=turn.failure_message,
                        )
                    elif event.kind in {"error", "deadline"}:
                        turn.provider_terminal = True
                        code, message = event.value
                        await self._flush_pending_final(turn, finalized=True)
                        await self._fail_turn(turn, code=code, message=message)
                        await self._emit_turn_terminal(
                            turn,
                            status="failed",
                            code=code,
                            message=message,
                        )
                    elif event.kind == "cancelled":
                        turn.provider_terminal = True
                        if turn.error_emitted:
                            await self._flush_pending_final(turn, finalized=True)
                            await self._emit_turn_terminal(
                                turn,
                                status="failed",
                                code=turn.failure_code,
                                message=turn.failure_message,
                            )
                        else:
                            await self._emit_turn_terminal(turn, status="cancelled")
                    if turn.provider_terminal:
                        await self._retire_turn(turn)
                        return
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            logger.exception("Realtime NVIDIA ASR result publication failed")
            self._request_turn_cancel(turn)
            try:
                await self._fail_turn(
                    turn,
                    code="asr_publication_error",
                    message="NVIDIA ASR failed to publish the Realtime transcription",
                    fatal=True,
                )
            except Exception:  # noqa: BLE001
                logger.exception("Realtime NVIDIA ASR error publication failed")
            finally:
                with suppress(Exception):
                    await self._emit_turn_terminal(
                        turn,
                        status="failed",
                        code=turn.failure_code or "asr_publication_error",
                        message=turn.failure_message or "NVIDIA ASR failed to publish the Realtime transcription",
                    )
                await self._retire_turn(turn)

    async def _handle_turn_response(self, turn: _RealtimeASRTurn, response: Any) -> None:
        if turn.discard_provider_output or turn.error_emitted:
            return
        response_id = getattr(getattr(response, "id", None), "value", "")
        if response_id != turn.request_id:
            self._request_turn_cancel(turn)
            await self._fail_turn(
                turn,
                code="asr_turn_owner_mismatch",
                message="NVIDIA ASR returned a response without the expected Realtime turn identifier",
                fatal=True,
            )
            return
        for result in response.results:
            if not result or not result.alternatives:
                continue
            transcript = result.alternatives[0].transcript
            if not transcript or not transcript.strip():
                continue
            turn.saw_text = True
            if result.is_final:
                turn.saw_interim = False
                await self._flush_pending_final(turn, finalized=False)
                turn.pending_final = result
            else:
                await self._flush_pending_final(turn, finalized=False)
                turn.saw_interim = True
                await self._emit_interim_result(turn, result)

    async def _flush_pending_final(self, turn: _RealtimeASRTurn, *, finalized: bool) -> None:
        result = turn.pending_final
        if result is None:
            return
        turn.pending_final = None
        if turn.discard_provider_output or turn.suppress_errors:
            return
        await self._emit_final_result(turn, result, finalized=finalized)

    async def _emit_interim_result(self, turn: _RealtimeASRTurn, result: Any) -> None:
        frame = InterimTranscriptionFrame(
            result.alternatives[0].transcript,
            self._user_id,
            time_now_iso8601(),
            cast("Language | None", assert_given(self._settings.language)),
            result=result,
        )
        frame.metadata[USER_TRANSCRIPT_TURN_FRAME_ID_METADATA] = turn.token
        await self.push_frame(frame)

    async def _emit_final_result(self, turn: _RealtimeASRTurn, result: Any, *, finalized: bool) -> None:
        transcript = result.alternatives[0].transcript
        language = cast("Language | None", assert_given(self._settings.language))
        logger.debug(f"Transcription: [{transcript}]")
        frame = TranscriptionFrame(
            transcript,
            self._user_id,
            time_now_iso8601(),
            language,
            result=result,
            finalized=finalized,
        )
        frame.metadata[USER_TRANSCRIPT_TURN_FRAME_ID_METADATA] = turn.token
        await self.push_frame(frame)
        await self._handle_transcription(transcript=transcript, is_final=True, language=language)

    async def _finish_turn(self, turn: _RealtimeASRTurn) -> None:
        if turn.cancel_requested or turn.error_emitted:
            return
        if turn.saw_interim:
            await self._fail_turn(
                turn,
                code="asr_final_transcription_missing",
                message="NVIDIA ASR ended a Realtime turn without a final transcription",
            )
            return
        await self._flush_pending_final(turn, finalized=True)

    async def _fail_turn(
        self,
        turn: _RealtimeASRTurn,
        *,
        code: str,
        message: str,
        fatal: bool = False,
    ) -> None:
        if turn.error_emitted or turn.suppress_errors or turn.retired:
            return
        turn.error_emitted = True
        turn.failure_code = code
        turn.failure_message = message
        turn.failure_fatal = turn.failure_fatal or fatal
        self._request_turn_cancel(turn)
        if fatal or turn.stop_token is None:
            await self._emit_error(code, message)

    async def _emit_turn_terminal(
        self,
        turn: _RealtimeASRTurn,
        *,
        status: Literal["completed", "empty", "failed", "cancelled"],
        code: str | None = None,
        message: str | None = None,
    ) -> None:
        """Publish one ordered terminal downstream, then release input upstream."""
        if turn.terminal_emitted or turn.stop_token is None or turn.suppress_errors or self._realtime_shutting_down:
            return
        await self._close_turn_metrics(turn)
        terminal = RealtimeASRTurnEndedFrame(
            turn_start_frame_id=turn.token,
            turn_stop_frame_id=turn.stop_token,
            status=status,
            code=code,
            message=message,
            fatal=turn.failure_fatal if status == "failed" else False,
        )
        turn.terminal_emitted = True
        downstream_error: BaseException | None = None
        try:
            await self.push_frame(terminal)
        except BaseException as exc:  # pragma: no cover - defensive direct-mode edge
            downstream_error = exc
        await self.push_frame(terminal, FrameDirection.UPSTREAM)
        if downstream_error is not None:
            raise downstream_error

    async def _close_turn_metrics(self, turn: _RealtimeASRTurn) -> None:
        """Finalize processing and usage metrics once for the native turn."""
        if turn.metrics_closed:
            return
        turn.metrics_closed = True
        await self.stop_processing_metrics()
        self._stt_usage_pending_seconds = turn.audio_bytes / (self.sample_rate * self._bytes_per_sample())
        await self.emit_stt_usage_metrics()

    async def _emit_error(self, code: str, message: str) -> None:
        await self.push_frame(RealtimeInputTranscriptionErrorFrame(code=code, message=message))

    def _request_turn_cancel(
        self,
        turn: _RealtimeASRTurn,
        *,
        suppress_errors: bool = False,
        discard_provider_output: bool = True,
    ) -> None:
        with turn.state_guard:
            turn.cancel_requested = True
            turn.discard_provider_output = turn.discard_provider_output or discard_provider_output
            turn.suppress_errors = turn.suppress_errors or suppress_errors
            call = turn.call
        turn.chunks.close()
        if call is not None:
            try:
                call.cancel()
            except Exception:  # noqa: BLE001
                logger.exception("Failed to cancel the native Realtime NVIDIA ASR call")

    async def _enforce_turn_deadline(self, turn: _RealtimeASRTurn) -> None:
        try:
            await asyncio.wait_for(turn.done.wait(), timeout=self._realtime_turn_drain_timeout_secs)
            return
        except TimeoutError:
            self._queue_turn_failure(
                turn,
                code="asr_provider_timeout",
                message="NVIDIA ASR did not finish the Realtime audio turn before its deadline",
            )

    async def _wait_for_worker_exit(self, turn: _RealtimeASRTurn) -> bool:
        """Wait on the actual worker task without creating a second executor waiter."""
        task = turn.worker_task
        if task is None:
            return turn.worker_exited.is_set()
        try:
            await asyncio.wait_for(
                asyncio.shield(task),
                timeout=self._realtime_worker_cancel_timeout_secs,
            )
        except TimeoutError:
            return turn.worker_exited.is_set()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            logger.exception(f"Realtime NVIDIA ASR worker task failed request_id={turn.request_id}")
        return turn.worker_exited.is_set()

    async def _retire_turn(self, turn: _RealtimeASRTurn) -> None:
        """Close metrics and release one turn after all native publication."""
        try:
            await self._close_turn_metrics(turn)
        except Exception:  # noqa: BLE001
            logger.exception(f"Realtime NVIDIA ASR metric finalization failed request_id={turn.request_id}")
        if not turn.retired:
            turn.retired = True
            turn.pending_final = None
            turn.provider_events.clear()
            turn.provider_event_ready.set()
            if self._realtime_active_turn is turn:
                self._realtime_active_turn = None
            if turn.deadline_task is not None and turn.deadline_task is not asyncio.current_task():
                turn.deadline_task.cancel()
            turn.done.set()
        self._release_retired_turn_if_exited(turn)

    def _release_retired_turn_if_exited(self, turn: _RealtimeASRTurn) -> None:
        """Forget a retired native call only after its worker task has exited."""
        task = turn.worker_task
        if not turn.retired:
            return
        if task is not None and (not turn.worker_exited.is_set() or not task.done()):
            return
        self._realtime_turns.pop(turn.request_id, None)

    async def _shutdown_turns(self) -> None:
        if self._realtime_shutting_down:
            return
        self._realtime_shutting_down = True
        turns = tuple(self._realtime_turns.values())
        self._realtime_active_turn = None
        self._realtime_pre_roll.clear()
        for turn in turns:
            self._request_turn_cancel(turn, suppress_errors=True)

        exits: tuple[object, ...] = ()
        if turns:
            exits = tuple(
                await asyncio.gather(
                    *(self._wait_for_worker_exit(turn) for turn in turns),
                    return_exceptions=True,
                )
            )
        for turn, exited in zip(turns, exits, strict=True):
            if exited is not True:
                logger.error(f"Realtime NVIDIA ASR worker did not exit during shutdown request_id={turn.request_id}")
        for turn in turns:
            await self._retire_turn(turn)
            for task in (turn.publisher_task, turn.deadline_task):
                if task is not None and task is not asyncio.current_task() and not task.done():
                    await self.cancel_task(task)
