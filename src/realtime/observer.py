# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Translate Pipecat frames into canonical Realtime lifecycle events."""

from __future__ import annotations

import asyncio
import json
import math
from collections import OrderedDict, deque
from collections.abc import Awaitable, Callable, Mapping
from typing import Any
from weakref import WeakKeyDictionary

from loguru import logger
from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    FunctionCallCancelFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    FunctionCallsStartedFrame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    MetricsFrame,
    TranscriptionFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import (
    LLMUsageMetricsData,
    ProcessingMetricsData,
    TTFBMetricsData,
    TurnMetricsData,
)
from pipecat.observers.base_observer import BaseObserver, FrameProcessed, FramePushed
from pipecat.processors.frame_processor import FrameDirection
from pipecat.processors.frameworks.rtvi.frames import RTVIServerMessageFrame
from pipecat.services.tts_service import TTSService
from pipecat.transports.base_output import BaseOutputTransport

from examples.shared.frames import (
    USER_TRANSCRIPT_TURN_FRAME_ID_METADATA,
    LLMProviderCompletionReasonFrame,
    UserTranscriptProducerEndedFrame,
)
from realtime.client_tools import (
    ClientToolCancelledResult,
    ClientToolTimeoutResult,
)
from realtime.controller import RealtimeSessionController
from realtime.events import EmitBatchFn, EmitFn, error_event
from realtime.frames import (
    RealtimeASREndpointSilenceFrame,
    RealtimeInputTranscriptionErrorFrame,
    RealtimeManualUserStartedSpeakingFrame,
    RealtimeManualUserStoppedSpeakingFrame,
    RealtimeOwnedLLMFullResponseStartFrame,
)
from realtime.lifecycle import announce_response, emit_events
from realtime.mcp import RealtimeMCPRuntime
from realtime.protocol import RealtimeProtocolError

_OBSERVED_FRAME_TYPES = (
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    LLMFullResponseStartFrame,
    LLMProviderCompletionReasonFrame,
    TTSTextFrame,
    LLMTextFrame,
    FunctionCallsStartedFrame,
    FunctionCallCancelFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    LLMFullResponseEndFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    ErrorFrame,
    InterruptionFrame,
    MetricsFrame,
    RTVIServerMessageFrame,
    UserTranscriptProducerEndedFrame,
    RealtimeInputTranscriptionErrorFrame,
)

FlushAudioHook = Callable[[], Awaitable[None]]
ResetAudioHook = Callable[[], None]
CancelAudioHook = Callable[[int], None]
FailInputAudioTurnHook = Callable[[], Awaitable[None]]
ClientToolContextHook = Callable[[str], Awaitable[None]]


class RealtimeLifecycleObserver(BaseObserver):
    """Own the ordered Realtime view of Pipecat response and transcript frames."""

    def __init__(
        self,
        *,
        emit: EmitFn,
        emit_batch: EmitBatchFn,
        controller: RealtimeSessionController,
        flush_output_audio: FlushAudioHook | None = None,
        reset_output_audio: ResetAudioHook | None = None,
        cancel_output_audio: CancelAudioHook | None = None,
        fail_input_audio_turn: FailInputAudioTurnHook | None = None,
        wait_client_tool_context: ClientToolContextHook | None = None,
        mcp_runtime: RealtimeMCPRuntime | None = None,
        input_transcription_timeout_secs: float = 8.0,
        max_frames: int = 4096,
        **kwargs: Any,
    ) -> None:
        """Bind frame observation to one controller and serialized emitter."""
        super().__init__(**kwargs)
        if (
            isinstance(input_transcription_timeout_secs, bool)
            or not isinstance(input_transcription_timeout_secs, int | float)
            or not math.isfinite(float(input_transcription_timeout_secs))
            or input_transcription_timeout_secs <= 0
        ):
            raise ValueError("input_transcription_timeout_secs must be a finite positive number")
        self._emit = emit
        self._emit_batch = emit_batch
        self._controller = controller
        self._flush_output_audio = flush_output_audio
        self._reset_output_audio = reset_output_audio
        self._cancel_output_audio = cancel_output_audio
        self._fail_input_audio_turn = fail_input_audio_turn
        self._wait_client_tool_context = wait_client_tool_context
        self._mcp_runtime = mcp_runtime
        self._shutdown = False
        self._input_transcription_timeout_secs = float(input_transcription_timeout_secs)
        self._input_transcription_lock = asyncio.Lock()
        self._input_transcription_changed = asyncio.Event()
        self._pending_input_transcriptions: deque[tuple[str, float]] = deque()
        self._input_item_by_turn_frame_id: OrderedDict[int, str] = OrderedDict()
        self._retired_input_turn_frame_ids: OrderedDict[int, None] = OrderedDict()
        self._manual_input_item_by_turn_frame_id: OrderedDict[int, str] = OrderedDict()
        self._input_transcription_watchdog: asyncio.Task[None] | None = None
        self._input_transcription_terminal = False
        self._input_transcription_shutdown = False
        self._processed_frames: set[int] = set()
        self._frame_history: deque[int] = deque(maxlen=max_frames)
        self._max_frames = max_frames
        self._bot_transcript_from_tts = False
        self._llm_text_buffer = ""
        self._llm_text_response_id: str | None = None
        self._emitted_function_calls: set[str] = set()
        self._tool_response_id: str | None = None
        self._pending_function_results: deque[FunctionCallResultFrame] = deque()
        self._llm_response_owners: WeakKeyDictionary[Any, deque[str]] = WeakKeyDictionary()
        self._llm_start_response_by_frame: OrderedDict[int, str] = OrderedDict()
        self._llm_end_response_by_frame: OrderedDict[int, str] = OrderedDict()
        self._completed_llm_end_frames: OrderedDict[int, None] = OrderedDict()
        self._tts_turn_response: WeakKeyDictionary[Any, str] = WeakKeyDictionary()
        self._tts_context_response: dict[str, str] = {}
        self._pending_tts_contexts: dict[str, set[str]] = {}
        self._sealed_tts_responses: set[str] = set()
        self._pending_llm_completion_tokens: dict[str, float] = {}
        self._pending_llm_processing_seconds: dict[str, float] = {}
        self._provider_terminal_by_response: dict[str, tuple[str, str | None]] = {}
        self._input_audio_cursors: WeakKeyDictionary[Any, tuple[int, int]] = WeakKeyDictionary()
        self._vad_wire_item_id: str | None = None
        self._vad_raw_stop_cursor: tuple[int, int] | None = None

    def _remember_frame(self, frame_id: int) -> bool:
        if frame_id in self._processed_frames:
            return False
        self._processed_frames.add(frame_id)
        self._frame_history.append(frame_id)
        if len(self._processed_frames) > len(self._frame_history):
            self._processed_frames = set(self._frame_history)
        return True

    def _clear_response_buffers(self) -> None:
        self._bot_transcript_from_tts = False
        self._llm_text_buffer = ""
        self._llm_text_response_id = None

    def on_response_cancelled(self) -> None:
        """Discard model text that was never rendered to client audio."""
        self._clear_response_buffers()

    async def on_output_audio_failure(self, response_id: str, failure: RealtimeProtocolError) -> None:
        """Fail the exact active response whose output audio could not be converted."""
        if self._shutdown:
            return
        async with self._controller.response_transition_lock:
            if response_id != self._controller.active_response_id:
                logger.debug(f"Ignoring stale output audio failure response_id={response_id}")
                return
            if self._reset_output_audio is not None:
                self._reset_output_audio()
            await self._finish_response(
                status="failed",
                reason=failure.code,
                prefix_events=[failure.to_event()],
            )

    def _discard_tool_buffers(self) -> None:
        self._tool_response_id = None
        self._pending_function_results.clear()

    def shutdown(self) -> None:
        """Release per-connection buffers on disconnect."""
        if self._shutdown:
            return
        # The observer remains attached to Pipecat until pipeline cancellation
        # finishes. Establish the connection teardown boundary before clearing
        # ownership state so late output frames cannot be interpreted as new
        # protocol activity against an already-closed WebSocket.
        self._shutdown = True
        self._input_transcription_shutdown = True
        self._input_transcription_changed.set()
        if self._input_transcription_watchdog is not None:
            self._input_transcription_watchdog.cancel()
            self._input_transcription_watchdog = None
        self._pending_input_transcriptions.clear()
        self._input_item_by_turn_frame_id.clear()
        self._retired_input_turn_frame_ids.clear()
        self._manual_input_item_by_turn_frame_id.clear()
        self._clear_response_buffers()
        self._processed_frames.clear()
        self._frame_history.clear()
        self._emitted_function_calls.clear()
        self._discard_tool_buffers()
        self._llm_response_owners.clear()
        self._llm_start_response_by_frame.clear()
        self._llm_end_response_by_frame.clear()
        self._completed_llm_end_frames.clear()
        self._tts_turn_response.clear()
        self._tts_context_response.clear()
        self._pending_tts_contexts.clear()
        self._sealed_tts_responses.clear()
        self._pending_llm_completion_tokens.clear()
        self._pending_llm_processing_seconds.clear()
        self._provider_terminal_by_response.clear()
        self._input_audio_cursors.clear()
        self._vad_wire_item_id = None
        self._vad_raw_stop_cursor = None

    async def on_process_frame(self, data: FrameProcessed) -> None:
        """Advance an input-audio clock at each processor's real dequeue edge."""
        if self._shutdown:
            return
        frame = data.frame
        if (
            data.direction != FrameDirection.DOWNSTREAM
            or not isinstance(frame, InputAudioRawFrame)
            or isinstance(frame, RealtimeASREndpointSilenceFrame)
        ):
            return
        sample_rate = max(1, int(frame.sample_rate))
        samples, previous_rate = self._input_audio_cursors.get(data.processor, (0, sample_rate))
        if sample_rate != previous_rate:
            samples = round(samples * sample_rate / previous_rate)
        channels = max(1, int(frame.num_channels))
        samples += len(frame.audio or b"") // (2 * channels)
        self._input_audio_cursors[data.processor] = (samples, sample_rate)

    async def on_push_frame(self, data: FramePushed) -> None:
        """Map relevant downstream frames and interruption/error system frames."""
        if self._shutdown:
            return
        frame = data.frame
        if isinstance(frame, ErrorFrame):
            if not self._remember_frame(frame.id):
                return
            await self._handle_pipeline_error(frame, source=data.source)
            return

        if isinstance(frame, InterruptionFrame):
            if data.direction != FrameDirection.DOWNSTREAM:
                return
            if not self._remember_frame(frame.id):
                return
            async with self._controller.response_transition_lock:
                interruption_generation, _ = self._controller.observe_interruption(frame.id)
                interrupted_response_id, cancel_reason = self._controller.interruption_target(frame.id)
                self._controller.cancel_pending_pipeline_response(
                    older_than_generation=interruption_generation,
                )
                if interrupted_response_id is None or interrupted_response_id != self._controller.active_response_id:
                    return
                self._clear_response_buffers()
                if self._cancel_output_audio is not None:
                    self._cancel_output_audio(frame.id)
                await self._finish_response(
                    status="cancelled",
                    reason=cancel_reason,
                )
            return

        if isinstance(frame, FunctionCallCancelFrame):
            # Observe cancellation only after the assistant aggregator has
            # replaced its IN_PROGRESS context entry. The output edge is the
            # first unambiguous post-context boundary for this system frame.
            if data.direction != FrameDirection.DOWNSTREAM or not isinstance(data.source, BaseOutputTransport):
                return
            if not self._remember_frame(frame.id):
                return
            record = self._controller.tool_call(frame.tool_call_id or "")
            if record is not None and record.owner == "mcp" and self._mcp_runtime is not None:
                await emit_events(
                    self._emit_batch,
                    self._mcp_runtime.cancel_call_events(frame.tool_call_id or ""),
                )
                self._emitted_function_calls.discard(frame.tool_call_id or "")
            return

        if isinstance(frame, LLMFullResponseStartFrame) and isinstance(data.source, TTSService):
            if data.direction != FrameDirection.DOWNSTREAM:
                return
            response_id = self._llm_start_response_by_frame.pop(frame.id, None)
            if response_id is None:
                await self._emit(
                    error_event(
                        "TTS received an uncorrelated LLM response start",
                        code="llm_response_owner_missing",
                        error_type="server_error",
                    )
                )
                return
            self._tts_turn_response[data.source] = response_id
            return

        if isinstance(frame, LLMFullResponseEndFrame) and not isinstance(data.source, TTSService):
            if data.direction != FrameDirection.DOWNSTREAM:
                return
            # Pipecat forwards the same sync frame through every downstream
            # processor. Correlate it at the first raw LLM edge. Audio keeps
            # that mapping until TTS consumes the frame; text-only pipelines
            # have no TTS processor, so their terminal edge is the output
            # transport after it has accepted every preceding text frame.
            if frame.id in self._completed_llm_end_frames:
                return
            if frame.id not in self._llm_end_response_by_frame:
                response_id = self._pop_llm_response_owner(data.source)
                if response_id is None:
                    await self._emit(
                        error_event(
                            "LLM response end has no correlated response start",
                            code="llm_response_owner_missing",
                            error_type="server_error",
                        )
                    )
                    return
                self._remember_response_owner(self._llm_end_response_by_frame, frame.id, response_id)
            if self._controller.output_kind != "text" or not isinstance(data.source, BaseOutputTransport):
                return

        # The output transport re-pushes TTSTextFrame in playout order and
        # TTSStoppedFrame after queued audio is actually drained. Only those
        # output edges may publish an audio transcript or close its response.
        if isinstance(frame, (TTSTextFrame, TTSStoppedFrame)) and not isinstance(data.source, BaseOutputTransport):
            return
        is_input_transcript = isinstance(frame, (InterimTranscriptionFrame, TranscriptionFrame))
        is_raw_vad_boundary = isinstance(frame, (VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame))
        if data.direction != FrameDirection.DOWNSTREAM and not (
            data.direction == FrameDirection.UPSTREAM
            and (is_input_transcript or is_raw_vad_boundary or isinstance(frame, UserTranscriptProducerEndedFrame))
        ):
            return
        if not isinstance(frame, _OBSERVED_FRAME_TYPES) or not self._remember_frame(frame.id):
            return

        try:
            await self._handle_frame(frame, source=data.source)
        except RealtimeProtocolError as exc:
            await self._finish_response(
                status="failed",
                reason=exc.code,
                prefix_events=[
                    RealtimeProtocolError(
                        message=exc.message,
                        code=exc.code,
                        param=exc.param,
                        error_type="server_error",
                    ).to_event()
                ],
            )
        except Exception:
            logger.exception(f"Realtime lifecycle failed for frame={type(frame).__name__}")
            await self._finish_response(
                status="failed",
                reason="lifecycle_error",
                prefix_events=[
                    error_event(
                        "Realtime lifecycle processing failed",
                        code="lifecycle_error",
                        error_type="server_error",
                    )
                ],
            )

    async def _handle_pipeline_error(self, frame: ErrorFrame, *, source: Any) -> None:
        """Fail only the response generation unambiguously owned by an error."""
        try:
            logger.warning(f"Realtime pipeline error source={type(source).__name__}: {frame.error or 'unknown'}")
            owners = self._llm_response_owners.get(source)
            llm_owner = owners[-1] if owners else None
            tts_owner = self._tts_turn_response.get(source) if isinstance(source, TTSService) else None
            active_response_id = self._controller.active_response_id
            if llm_owner == active_response_id:
                if isinstance(frame.exception, RealtimeProtocolError):
                    code = frame.exception.code
                    message = frame.exception.message
                    error_type = frame.exception.error_type
                    error_param = frame.exception.param
                    error_event_id = frame.exception.event_id
                else:
                    code = "llm_provider_error"
                    message = "The language model failed while generating this response"
                    error_type = "server_error"
                    error_param = None
                    error_event_id = None
                correlated = True
            elif tts_owner == active_response_id:
                code = "tts_provider_error"
                message = "The speech synthesizer failed while generating this response"
                error_type = "server_error"
                error_param = None
                error_event_id = None
                correlated = True
            elif frame.fatal and active_response_id is not None:
                code = "pipeline_error"
                message = "The voice pipeline failed while generating this response"
                error_type = "server_error"
                error_param = None
                error_event_id = None
                correlated = True
            else:
                code = "pipeline_error"
                message = "A voice pipeline service reported an uncorrelated error"
                error_type = "server_error"
                error_param = None
                error_event_id = None
                correlated = False

            event = error_event(
                message,
                code=code,
                error_type=error_type,
                param=error_param,
                event_id=error_event_id,
            )
            if correlated:
                if self._cancel_output_audio is not None:
                    self._cancel_output_audio(frame.id)
                elif self._reset_output_audio is not None:
                    self._reset_output_audio()
                await self._finish_response(
                    status="failed",
                    reason=code,
                    prefix_events=[event],
                )
            else:
                await self._emit_batch([event])
        except Exception:
            logger.exception("Realtime lifecycle failed while reporting a pipeline error")

    async def _handle_frame(self, frame: Frame, *, source: Any) -> None:
        if isinstance(frame, RealtimeInputTranscriptionErrorFrame):
            if self._input_transcription_configured():
                await self._handle_input_transcription_error(frame)
            return

        if isinstance(frame, MetricsFrame):
            await self._handle_metrics(frame, source=source)
            return

        if isinstance(frame, RTVIServerMessageFrame):
            await self._handle_server_message(frame)
            return

        if isinstance(frame, UserTranscriptProducerEndedFrame):
            await self._handle_transcript_producer_end(frame)
            return

        if isinstance(frame, VADUserStartedSpeakingFrame):
            if self._controller.manual_input_mode:
                return
            async with self._input_transcription_lock:
                if self._input_transcription_terminal or self._input_transcription_shutdown:
                    return
                audio_sample_cursor, sample_rate = self._input_cursor(source)
                # VAD announces the start only after ``start_secs`` of confirmed
                # speech. Project the event back to the actual speech onset on
                # the pipeline's processed-PCM clock.
                onset_sample = max(
                    0,
                    audio_sample_cursor - round(max(0.0, float(frame.start_secs)) * sample_rate),
                )
                if self._controller.turn_detection_type == "server_vad":
                    onset_sample = max(
                        0,
                        onset_sample - round(self._controller.server_vad_prefix_padding_ms * sample_rate / 1000),
                    )
                item_id, _ = self._controller.begin_user_turn(
                    new_turn=False,
                    audio_sample_cursor=onset_sample,
                    sample_rate=sample_rate,
                )
                if self._tracks_cascaded_input_transcription():
                    self._remember_input_turn_owner(frame.id, item_id)
                self._vad_raw_stop_cursor = None
                if self._vad_wire_item_id == item_id:
                    return
                self._vad_wire_item_id = item_id
                await self._emit(
                    self._server_event(
                        "input_audio_buffer.speech_started",
                        item_id=item_id,
                        audio_start_ms=round(onset_sample * 1000 / max(sample_rate, 1)),
                    )
                )
            return

        if isinstance(frame, VADUserStoppedSpeakingFrame):
            if self._controller.manual_input_mode:
                return
            async with self._input_transcription_lock:
                if self._input_transcription_terminal or self._input_transcription_shutdown:
                    return
                audio_sample_cursor, sample_rate = self._input_cursor(source)
                self._controller.begin_user_turn(
                    new_turn=False,
                    audio_sample_cursor=audio_sample_cursor,
                    sample_rate=sample_rate,
                )
                self._vad_raw_stop_cursor = (audio_sample_cursor, sample_rate)
            return

        if isinstance(frame, UserStartedSpeakingFrame):
            async with self._input_transcription_lock:
                if self._input_transcription_terminal or self._input_transcription_shutdown:
                    return
                if isinstance(frame, RealtimeManualUserStartedSpeakingFrame):
                    audio_sample_cursor, sample_rate = frame.audio_sample_cursor, frame.sample_rate
                else:
                    audio_sample_cursor, sample_rate = self._input_cursor(source)
                item_id, audio_start_ms = self._controller.begin_user_turn(
                    new_turn=False,
                    audio_sample_cursor=audio_sample_cursor,
                    sample_rate=sample_rate,
                )
                if (
                    isinstance(frame, RealtimeManualUserStartedSpeakingFrame)
                    and self._tracks_cascaded_input_transcription()
                ):
                    self._remember_input_turn_owner(frame.id, item_id)
                if not self._controller.manual_input_mode and self._vad_wire_item_id != item_id:
                    # Some provider-owned/direct pipelines expose only semantic
                    # user-turn frames. Keep their turn-detection surface complete,
                    # while Pipecat VAD-backed pipelines publish from the raw
                    # VAD frame above.
                    self._vad_wire_item_id = item_id
                    self._vad_raw_stop_cursor = None
                    await self._emit(
                        self._server_event(
                            "input_audio_buffer.speech_started",
                            item_id=item_id,
                            audio_start_ms=audio_start_ms,
                        )
                    )
            return

        if isinstance(frame, UserStoppedSpeakingFrame):
            close_for_transcription_failure = False
            async with self._input_transcription_lock:
                if self._input_transcription_terminal or self._input_transcription_shutdown:
                    return
                if isinstance(frame, RealtimeManualUserStoppedSpeakingFrame):
                    audio_sample_cursor, sample_rate = frame.audio_sample_cursor, frame.sample_rate
                else:
                    audio_sample_cursor, sample_rate = self._input_cursor(source)
                if (
                    not self._controller.manual_input_mode
                    and self._controller.turn_detection_type == "server_vad"
                    and self._vad_raw_stop_cursor is not None
                ):
                    audio_sample_cursor, sample_rate = self._vad_raw_stop_cursor
                _, audio_start_ms = self._controller.begin_user_turn(
                    new_turn=False,
                    audio_sample_cursor=audio_sample_cursor,
                    sample_rate=sample_rate,
                )
                if self._controller.manual_input_mode and self._input_transcription_configured():
                    item_id, audio_end_ms, item_events = self._controller.commit_user_turn(
                        audio_sample_cursor=audio_sample_cursor,
                        sample_rate=sample_rate,
                    )
                    self._remember_manual_input_turn_owner(frame.id, item_id)
                else:
                    item_id, audio_end_ms, item_events = self._controller.stop_user_turn(
                        audio_sample_cursor=audio_sample_cursor,
                        sample_rate=sample_rate,
                    )
                if self._input_transcription_configured():
                    self._remember_input_turn_owner(frame.id, item_id)
                item_events, close_for_transcription_failure = self._prepare_input_transcription_events(
                    item_id,
                    item_events,
                )
                previous_item_id = item_events[0].get("previous_item_id") if item_events else None
                events = []
                if not self._controller.manual_input_mode:
                    if self._vad_wire_item_id != item_id:
                        events.append(
                            self._server_event(
                                "input_audio_buffer.speech_started",
                                item_id=item_id,
                                audio_start_ms=audio_start_ms,
                            )
                        )
                    events.append(
                        self._server_event(
                            "input_audio_buffer.speech_stopped",
                            item_id=item_id,
                            audio_end_ms=audio_end_ms,
                        )
                    )
                events.extend(
                    [
                        self._server_event(
                            "input_audio_buffer.committed",
                            item_id=item_id,
                            previous_item_id=previous_item_id,
                        ),
                        *item_events,
                    ]
                )
                await emit_events(self._emit_batch, events)
                self._vad_wire_item_id = None
                self._vad_raw_stop_cursor = None
                if not any(
                    event.get("type")
                    in {
                        "conversation.item.input_audio_transcription.completed",
                        "conversation.item.input_audio_transcription.failed",
                    }
                    for event in item_events
                ):
                    self._arm_input_transcription_watchdog(item_id)
            if close_for_transcription_failure and self._fail_input_audio_turn is not None:
                await self._fail_input_audio_turn()
            return

        if isinstance(frame, InterimTranscriptionFrame):
            if not self._input_transcription_configured():
                return
            close_for_transcription_failure = False
            async with self._input_transcription_lock:
                if self._input_transcription_terminal or self._input_transcription_shutdown:
                    return
                self._controller.set_input_audio_cursor(*self._input_cursor(source))
                raw_turn_frame_id = frame.metadata.get(USER_TRANSCRIPT_TURN_FRAME_ID_METADATA)
                transcript_item_id = None
                if _is_frame_id(raw_turn_frame_id):
                    transcript_item_id = self._input_item_by_turn_frame_id.get(raw_turn_frame_id)
                    if transcript_item_id is None:
                        if raw_turn_frame_id in self._retired_input_turn_frame_ids:
                            return
                        await self._emit_input_transcription_owner_missing(raw_turn_frame_id)
                        close_for_transcription_failure = True
                elif self._tracks_cascaded_input_transcription():
                    return
                if not close_for_transcription_failure:
                    await emit_events(
                        self._emit_batch,
                        self._controller.user_transcript_delta(
                            frame.text or "",
                            item_id=transcript_item_id,
                        ),
                    )
            if close_for_transcription_failure and self._fail_input_audio_turn is not None:
                await self._fail_input_audio_turn()
            return

        if isinstance(frame, TranscriptionFrame):
            if not self._input_transcription_configured():
                return
            close_for_transcription_failure = False
            async with self._input_transcription_lock:
                if self._input_transcription_terminal or self._input_transcription_shutdown:
                    return
                self._controller.set_input_audio_cursor(*self._input_cursor(source))
                raw_turn_frame_id = frame.metadata.get(USER_TRANSCRIPT_TURN_FRAME_ID_METADATA)
                fused_transcription = self._tracks_fused_input_transcription()
                if fused_transcription and not _is_frame_id(raw_turn_frame_id):
                    await self._emit_fused_transcription_owner_missing(raw_turn_frame_id)
                    close_for_transcription_failure = True
                    transcript_item_id = None
                elif self._tracks_cascaded_input_transcription() and not _is_frame_id(raw_turn_frame_id):
                    return
                elif _is_frame_id(raw_turn_frame_id):
                    transcript_item_id = self._input_item_by_turn_frame_id.get(raw_turn_frame_id)
                    if transcript_item_id is None:
                        if raw_turn_frame_id in self._retired_input_turn_frame_ids:
                            return
                        if fused_transcription:
                            await self._emit_fused_transcription_owner_missing(raw_turn_frame_id)
                        else:
                            await self._emit_input_transcription_owner_missing(raw_turn_frame_id)
                        close_for_transcription_failure = True
                else:
                    transcript_item_id = None
                if not close_for_transcription_failure:
                    item_id, events = self._controller.set_user_transcript(
                        frame.text or "",
                        includes_inter_frame_spaces=frame.includes_inter_frame_spaces,
                        item_id=transcript_item_id,
                    )
                    events, close_for_transcription_failure = self._prepare_input_transcription_events(
                        item_id,
                        events,
                    )
                    await emit_events(self._emit_batch, events)
            if close_for_transcription_failure and self._fail_input_audio_turn is not None:
                await self._fail_input_audio_turn()
            return

        if isinstance(frame, LLMFullResponseStartFrame):
            owned_frame = frame if isinstance(frame, RealtimeOwnedLLMFullResponseStartFrame) else None
            owned_response_id = owned_frame.response_id or None if owned_frame is not None else None
            owned_run_id = owned_frame.run_owner_id or None if owned_frame is not None else None
            response_started = False
            if owned_response_id is not None:
                events = []
                response_id = owned_response_id
                response_started = owned_response_id == self._controller.active_response_id
            elif owned_run_id is not None:
                async with self._controller.response_transition_lock:
                    activated = self._controller.start_pending_pipeline_response(
                        owner_id=owned_run_id,
                        expected_generation=owned_frame.activation_generation,
                    )
                    if activated is None:
                        response_id = f"cancelled_run_{frame.id}"
                        events = []
                    else:
                        response_id, events = activated
                        response_started = True
                    await emit_events(self._emit_batch, events)
            else:
                # Every Realtime provider run is bound either to an immutable
                # pending-run owner or to an already-created response. A raw
                # Pipecat start has no generation identity, so it must never
                # attach to whichever response happens to be active when the
                # asynchronous observer sees it. Keep a cancelled sentinel in
                # the FIFO so its later text/end frames drain without touching
                # a newer response.
                response_id = f"cancelled_run_{frame.id}"
                events = [
                    error_event(
                        "Pipeline emitted an LLM response start without a Realtime owner",
                        code="llm_response_owner_missing",
                        error_type="server_error",
                    )
                ]
                await emit_events(self._emit_batch, events)
            if response_started:
                self._clear_response_buffers()
                self._llm_text_response_id = response_id
            if response_id is None:
                raise RuntimeError("LLM response started without a Realtime response")
            self._llm_response_owners.setdefault(source, deque()).append(response_id)
            self._remember_response_owner(self._llm_start_response_by_frame, frame.id, response_id)
            return

        if isinstance(frame, LLMProviderCompletionReasonFrame):
            owners = self._llm_response_owners.get(source)
            response_id = owners[0] if owners else None
            if response_id is None:
                logger.debug(
                    "Ignoring provider completion reason without an active LLM owner "
                    f"finish_reason={frame.finish_reason!r}"
                )
                return
            if response_id != self._controller.active_response_id:
                # An interruption can terminalize Response A before the
                # cancelled provider task publishes its final metadata frame.
                # Never recreate terminal state for a response that no longer
                # owns the public response slot.
                self._provider_terminal_by_response.pop(response_id, None)
                logger.debug(f"Ignoring late provider completion reason response_id={response_id}")
                return
            if frame.finish_reason == "length":
                terminal = ("incomplete", "max_output_tokens")
            elif frame.finish_reason == "content_filter":
                terminal = ("incomplete", "content_filter")
            else:
                terminal = ("completed", None)
            previous = self._provider_terminal_by_response.get(response_id)
            if previous is not None and previous != terminal:
                raise RealtimeProtocolError(
                    message="The language model emitted conflicting completion reasons",
                    code="llm_finish_reason_conflict",
                    param="response.status_details",
                )
            self._provider_terminal_by_response[response_id] = terminal
            return

        if isinstance(frame, LLMTextFrame):
            owner = self._current_llm_response_owner(source)
            if owner is not None and owner != self._controller.active_response_id:
                logger.debug(f"Ignoring stale LLM text for terminal response_id={owner}")
                return
            text = frame.text or ""
            if not text:
                return
            if self._controller.output_kind == "text":
                await emit_events(self._emit_batch, self._controller.append_assistant_text(text))
            else:
                await emit_events(self._emit_batch, self._controller.ensure_response())
                self._llm_text_buffer += text
                self._llm_text_response_id = self._controller.active_response_id
            return

        if isinstance(frame, TTSStartedFrame):
            if self._controller.output_kind == "text":
                return
            context_id = frame.context_id
            if not context_id:
                raise RealtimeProtocolError(
                    message="TTSStartedFrame is missing its context ID",
                    code="tts_context_missing",
                    param="response.audio",
                )
            response_id = self._tts_turn_response.get(source)
            if response_id is None:
                await self._emit(
                    error_event(
                        "TTS context has no correlated LLM response",
                        code="tts_response_owner_missing",
                        error_type="server_error",
                    )
                )
                return
            if response_id != self._controller.active_response_id:
                logger.debug(f"Ignoring stale TTS start for terminal response_id={response_id}")
                return
            mapped_response = self._tts_context_response.get(context_id)
            if mapped_response is not None and mapped_response != response_id:
                raise RealtimeProtocolError(
                    message="TTS context was reused across Realtime responses",
                    code="tts_context_conflict",
                    param="response.audio",
                )
            self._tts_context_response[context_id] = response_id
            self._pending_tts_contexts.setdefault(response_id, set()).add(context_id)
            await announce_response(self._controller, self._emit_batch, kind="audio")
            return

        if isinstance(frame, TTSTextFrame):
            if self._controller.output_kind == "text":
                return
            text = frame.text or ""
            if not text:
                return
            response_id = self._tts_context_response.get(frame.context_id or "")
            if response_id is None or response_id != self._controller.active_response_id:
                logger.debug(f"Ignoring stale or uncorrelated TTS text context_id={frame.context_id or '-'}")
                return
            announced_response_id, _ = await announce_response(
                self._controller,
                self._emit_batch,
                kind="audio",
            )
            if announced_response_id != response_id or response_id != self._controller.active_response_id:
                logger.debug(f"Ignoring stale TTS text after announcement response_id={response_id}")
                return
            # Pipecat emits each synthesized text fragment as its own frame. The
            # frame-ID guard in ``on_push_frame`` already removes duplicate
            # observations; comparing text would corrupt legitimate adjacent
            # repetitions such as "very very".
            await emit_events(
                self._emit_batch,
                self._controller.append_assistant_audio_transcript(
                    text,
                    includes_inter_frame_spaces=frame.includes_inter_frame_spaces,
                    context_text=frame.raw_text,
                ),
            )
            self._bot_transcript_from_tts = True
            return

        if isinstance(frame, FunctionCallsStartedFrame):
            owner = self._current_llm_response_owner(source)
            if owner is not None and owner != self._controller.active_response_id:
                logger.debug(f"Ignoring stale function-call start for terminal response_id={owner}")
                return
            for call in frame.function_calls or []:
                await self._announce_function_call(
                    call_id=getattr(call, "tool_call_id", "") or "",
                    name=getattr(call, "function_name", "") or "",
                    arguments=getattr(call, "arguments", None),
                )
            # NvidiaLLMService reports final token usage after scheduling the
            # handlers. Keep Response A open until LLMFullResponseEndFrame and
            # hold any very-fast result behind that terminal boundary.
            self._tool_response_id = self._controller.active_response_id
            return

        if isinstance(frame, FunctionCallInProgressFrame):
            owner = self._current_llm_response_owner(source)
            if owner is not None and owner != self._controller.active_response_id:
                logger.debug(f"Ignoring stale function call for terminal response_id={owner}")
                return
            await self._announce_function_call(
                call_id=frame.tool_call_id or "",
                name=frame.function_name or "",
                arguments=frame.arguments,
            )
            return

        if isinstance(frame, FunctionCallResultFrame):
            if frame.properties is not None and not frame.properties.is_final:
                # OpenAI Realtime has one terminal function_call_output item,
                # not Pipecat's optional intermediate-result stream. Leave the
                # call open and wait for its final callback.
                logger.debug(
                    f"Ignoring non-terminal tool update name={frame.function_name} call_id={frame.tool_call_id}"
                )
                return
            record = self._controller.tool_call(frame.tool_call_id or "")
            if (
                record is not None
                and self._tool_response_id == record.response_id
                and self._controller.active_response_id == record.response_id
            ):
                self._pending_function_results.append(frame)
                return
            await self._record_function_result(frame)
            return

        if isinstance(frame, LLMFullResponseEndFrame):
            response_id = self._llm_end_response_by_frame.pop(frame.id, None)
            tts_turn_response = self._tts_turn_response.get(source)
            if tts_turn_response == response_id:
                self._tts_turn_response.pop(source, None)
            if response_id is None:
                await self._emit(
                    error_event(
                        "Pipeline emitted an uncorrelated LLM response end",
                        code="llm_response_owner_missing",
                        error_type="server_error",
                    )
                )
                return
            self._remember_completed_llm_end(frame.id)
            if response_id != self._controller.active_response_id:
                logger.debug(f"Ignoring stale LLM response end response_id={response_id}")
                self._clear_tts_response(response_id)
                self._provider_terminal_by_response.pop(response_id, None)
                return
            self._sealed_tts_responses.add(response_id)
            if not self._pending_tts_contexts.get(response_id):
                status, reason = self._provider_terminal(response_id)
                await self._finish_response(status=status, reason=reason)
            return

        if isinstance(frame, TTSStoppedFrame):
            if self._controller.output_kind == "text":
                return
            context_id = frame.context_id
            if not context_id:
                raise RealtimeProtocolError(
                    message="TTSStoppedFrame is missing its context ID",
                    code="tts_context_missing",
                    param="response.audio",
                )
            response_id = self._tts_context_response.pop(context_id, None)
            if response_id is None:
                logger.debug(f"Ignoring uncorrelated or stale TTS stop context_id={context_id}")
                return
            pending = self._pending_tts_contexts.get(response_id)
            if pending is not None:
                pending.discard(context_id)
                if not pending:
                    self._pending_tts_contexts.pop(response_id, None)
            if (
                response_id != self._controller.active_response_id
                or response_id not in self._sealed_tts_responses
                or self._pending_tts_contexts.get(response_id)
            ):
                return
            if self._flush_output_audio is not None:
                await self._flush_output_audio()
            status, reason = self._provider_terminal(response_id)
            await self._finish_response(status=status, reason=reason)

    def _input_transcription_configured(self) -> bool:
        """Return whether the public session promises input transcription."""
        session = self._controller.public_session()
        audio = session.get("audio")
        input_audio = audio.get("input") if isinstance(audio, Mapping) else None
        transcription = input_audio.get("transcription") if isinstance(input_audio, Mapping) else None
        return transcription is not None

    def _tracks_cascaded_input_transcription(self) -> bool:
        """Return whether this connection has a separately routed ASR stream."""
        asr_server = self._controller.runtime_config.get("asr_server")
        return self._input_transcription_configured() and isinstance(asr_server, str) and bool(asr_server.strip())

    def _tracks_fused_input_transcription(self) -> bool:
        """Return whether transcript ownership comes from a fused model turn."""
        return self._input_transcription_configured() and not self._tracks_cascaded_input_transcription()

    async def _emit_fused_transcription_owner_missing(self, turn_frame_id: object) -> None:
        """Terminate a fused session whose transcript cannot target an exact item."""
        logger.warning(f"Fused input transcription owner missing turn_frame_id={turn_frame_id!r}")
        await self._terminate_input_transcription(
            code="omni_transcription_owner_missing",
            message="Fused input transcription has no correlated Realtime item",
        )

    async def _emit_input_transcription_owner_missing(self, turn_frame_id: object) -> None:
        """Terminate a cascaded session whose stamped owner is unknown."""
        logger.warning(f"Cascaded input transcription owner missing turn_frame_id={turn_frame_id!r}")
        await self._terminate_input_transcription(
            code="asr_transcription_owner_missing",
            message="ASR input transcription has no correlated Realtime item",
        )

    async def _handle_input_transcription_error(self, frame: RealtimeInputTranscriptionErrorFrame) -> None:
        """Publish a typed ASR ownership failure and close the affected connection."""
        async with self._input_transcription_lock:
            if self._input_transcription_terminal or self._input_transcription_shutdown:
                return
            await self._terminate_input_transcription(code=frame.code, message=frame.message)
        if self._fail_input_audio_turn is not None:
            await self._fail_input_audio_turn()

    async def _terminate_input_transcription(self, *, code: str, message: str) -> None:
        """Clear transcript ownership and emit one terminal Realtime error."""
        self._input_transcription_terminal = True
        self._pending_input_transcriptions.clear()
        self._input_item_by_turn_frame_id.clear()
        self._retired_input_turn_frame_ids.clear()
        self._manual_input_item_by_turn_frame_id.clear()
        self._input_transcription_changed.set()
        await self._emit_batch(
            [
                error_event(
                    message,
                    code=code,
                    param="session.audio.input.transcription",
                    error_type="server_error",
                )
            ]
        )

    def _arm_input_transcription_watchdog(self, item_id: str) -> None:
        """Bound one committed cascaded-ASR turn after its item is announced."""
        if (
            self._input_transcription_terminal
            or self._input_transcription_shutdown
            or not self._tracks_cascaded_input_transcription()
        ):
            return
        loop = asyncio.get_running_loop()
        self._pending_input_transcriptions.append((item_id, loop.time() + self._input_transcription_timeout_secs))
        self._input_transcription_changed.set()
        if self._input_transcription_watchdog is None or self._input_transcription_watchdog.done():
            self._input_transcription_watchdog = asyncio.create_task(
                self._watch_input_transcriptions(),
                name="realtime-input-transcription-watchdog",
            )

    def _complete_input_transcription_watchdog(self, item_id: str) -> None:
        """Release the watchdog entry owned by an exactly correlated item."""
        if not self._pending_input_transcriptions:
            return
        pending_item_id, _deadline = self._pending_input_transcriptions[0]
        if pending_item_id == item_id:
            self._pending_input_transcriptions.popleft()
            self._input_transcription_changed.set()
            return

        # The controller is authoritative. This branch should only be reached
        # if an external producer completed a turn outside this observer; drop
        # only the matching timer and never transfer it to a different item.
        for pending in self._pending_input_transcriptions:
            if pending[0] == item_id:
                self._pending_input_transcriptions.remove(pending)
                self._input_transcription_changed.set()
                logger.warning(f"Realtime input transcription completed out of watchdog order item_id={item_id}")
                return

    def _complete_input_transcription(self, item_id: str) -> None:
        """Release every observer-side owner for one completed transcript item."""
        self._complete_input_transcription_watchdog(item_id)
        self._forget_manual_input_turn_owner(item_id)
        if self._tracks_cascaded_input_transcription():
            self._forget_input_turn_owner(item_id)

    def _remember_input_turn_owner(self, turn_frame_id: int, item_id: str) -> None:
        """Correlate an immutable input-boundary frame token with one public item."""
        self._retired_input_turn_frame_ids.pop(turn_frame_id, None)
        self._input_item_by_turn_frame_id[turn_frame_id] = item_id
        self._input_item_by_turn_frame_id.move_to_end(turn_frame_id)
        while len(self._input_item_by_turn_frame_id) > self._max_frames:
            self._input_item_by_turn_frame_id.popitem(last=False)

    def _retire_input_turn_frame_id(self, turn_frame_id: int) -> None:
        """Remember a bounded set of valid owners whose items are terminal."""
        self._retired_input_turn_frame_ids[turn_frame_id] = None
        self._retired_input_turn_frame_ids.move_to_end(turn_frame_id)
        while len(self._retired_input_turn_frame_ids) > self._max_frames:
            self._retired_input_turn_frame_ids.popitem(last=False)

    def _forget_input_turn_owner(self, item_id: str) -> None:
        """Release any internal stop-frame token owned by one terminal item."""
        for turn_frame_id, owned_item_id in tuple(self._input_item_by_turn_frame_id.items()):
            if owned_item_id == item_id:
                self._input_item_by_turn_frame_id.pop(turn_frame_id, None)
                self._retire_input_turn_frame_id(turn_frame_id)

    def _remember_manual_input_turn_owner(self, turn_frame_id: int, item_id: str) -> None:
        """Correlate a manual commit with its later semantic turn barrier."""
        self._manual_input_item_by_turn_frame_id[turn_frame_id] = item_id
        self._manual_input_item_by_turn_frame_id.move_to_end(turn_frame_id)
        while len(self._manual_input_item_by_turn_frame_id) > self._max_frames:
            self._manual_input_item_by_turn_frame_id.popitem(last=False)

    def _forget_manual_input_turn_owner(self, item_id: str) -> None:
        """Release the exact manual barrier token owned by one terminal item."""
        for turn_frame_id, owned_item_id in tuple(self._manual_input_item_by_turn_frame_id.items()):
            if owned_item_id == item_id:
                self._manual_input_item_by_turn_frame_id.pop(turn_frame_id, None)

    def _prepare_input_transcription_events(
        self,
        item_id: str,
        events: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], bool]:
        """Apply observer bookkeeping and append a terminal fused-producer error."""
        if any(event.get("type") == "conversation.item.input_audio_transcription.completed" for event in events):
            self._complete_input_transcription(item_id)
            return events, False

        failure = next(
            (event for event in events if event.get("type") == "conversation.item.input_audio_transcription.failed"),
            None,
        )
        if failure is None:
            return events, False

        self._complete_input_transcription_watchdog(item_id)
        failure_detail = failure.get("error")
        code = (
            str(failure_detail.get("code") or "omni_transcription_failed")
            if isinstance(failure_detail, Mapping)
            else "omni_transcription_failed"
        )
        message = (
            str(failure_detail.get("message") or "Fused input transcription failed")
            if isinstance(failure_detail, Mapping)
            else "Fused input transcription failed"
        )
        if code == "omni_transcription_skipped":
            return events, False
        input_overflow = code == "input_buffer_overflow"
        return [
            *events,
            error_event(
                message,
                code=code,
                param="audio" if input_overflow else "session.audio.input.transcription",
                error_type="invalid_request_error" if input_overflow else "server_error",
            ),
        ], False

    async def _watch_input_transcriptions(self) -> None:
        """Fail the oldest committed audio item if final ASR never arrives."""
        try:
            while True:
                async with self._input_transcription_lock:
                    if self._input_transcription_terminal or self._input_transcription_shutdown:
                        self._input_transcription_watchdog = None
                        return
                    if not self._pending_input_transcriptions:
                        self._input_transcription_watchdog = None
                        return
                    item_id, deadline = self._pending_input_transcriptions[0]
                    self._input_transcription_changed.clear()

                remaining = deadline - asyncio.get_running_loop().time()
                if remaining > 0:
                    try:
                        await asyncio.wait_for(self._input_transcription_changed.wait(), timeout=remaining)
                        continue
                    except TimeoutError:
                        pass
                if await self._fail_timed_out_input_transcription(item_id):
                    return
        except asyncio.CancelledError:
            return
        except Exception:
            logger.exception("Realtime input transcription watchdog failed")
            await self._abort_input_transcription_watchdog()

    async def _fail_timed_out_input_transcription(self, item_id: str) -> bool:
        """Atomically release one timed-out turn, report it, then close its session."""
        async with self._input_transcription_lock:
            if self._input_transcription_terminal or self._input_transcription_shutdown:
                self._input_transcription_watchdog = None
                return True
            if not self._pending_input_transcriptions or self._pending_input_transcriptions[0][0] != item_id:
                return False

            message = "Input audio transcription did not complete before the server deadline"
            failure_events = self._controller.fail_user_transcription(
                code="transcription_timeout",
                message=message,
                item_id=item_id,
            )
            self._pending_input_transcriptions.popleft()
            self._input_transcription_changed.set()
            if not failure_events:
                # A final transcript won the controller race outside this
                # observer. Do not manufacture a failure for an unknown item.
                return False

            failed_item_id = failure_events[0].get("item_id")
            if failed_item_id != item_id:
                logger.warning(
                    "Realtime input transcription watchdog/controller order differed "
                    f"watchdog_item_id={item_id} controller_item_id={failed_item_id}"
                )
            self._input_transcription_terminal = True
            self._input_transcription_watchdog = None
            self._pending_input_transcriptions.clear()
            events = [
                *failure_events,
                error_event(
                    message,
                    code="asr_transcription_timeout",
                    param="session.audio.input.transcription",
                    error_type="server_error",
                ),
            ]
            if self._controller.response_in_progress:
                events.extend(
                    self._controller.finish_response(
                        status="failed",
                        reason="asr_transcription_timeout",
                    )
                )

        # One serialized batch preserves the item failure before the terminal
        # connection error. Closing only after it is sent prevents a delayed
        # reconnect transcript from being correlated to a later user turn.
        await self._emit_batch(events)
        if self._fail_input_audio_turn is not None:
            await self._fail_input_audio_turn()
        return True

    async def _abort_input_transcription_watchdog(self) -> None:
        """Terminate safely if the watchdog itself encounters an internal fault."""
        async with self._input_transcription_lock:
            if self._input_transcription_terminal or self._input_transcription_shutdown:
                return
            self._input_transcription_terminal = True
            self._input_transcription_watchdog = None
            self._pending_input_transcriptions.clear()
            events = [
                error_event(
                    "Input audio transcription lifecycle failed",
                    code="input_transcription_lifecycle_error",
                    param="session.audio.input.transcription",
                    error_type="server_error",
                )
            ]
            if self._controller.response_in_progress:
                events.extend(
                    self._controller.finish_response(
                        status="failed",
                        reason="input_transcription_lifecycle_error",
                    )
                )
        await self._emit_batch(events)
        if self._fail_input_audio_turn is not None:
            await self._fail_input_audio_turn()

    async def _handle_metrics(self, frame: MetricsFrame, *, source: Any) -> None:
        """Expose Pipecat metrics without requiring the RTVI processor."""
        llm_response_owner = self._current_llm_response_owner(source)
        for datum in frame.data:
            processor = str(getattr(datum, "processor", ""))
            category = _processor_category(processor)
            if category == "llm":
                if llm_response_owner is None:
                    logger.debug("Ignoring LLM metrics with no correlated response generation")
                    continue
                if llm_response_owner != self._controller.active_response_id:
                    logger.debug(f"Ignoring stale LLM metrics for terminal response_id={llm_response_owner}")
                    continue
            metrics: dict[str, float] = {}
            if isinstance(datum, TTFBMetricsData):
                key = {"asr": "asr_ttfb", "llm": "llm_ttft", "tts": "tts_ttfb"}.get(category)
                if key:
                    value = _finite_nonnegative(datum.value)
                    if value is not None:
                        metrics[key] = value
            elif isinstance(datum, ProcessingMetricsData) and category == "llm":
                processing_seconds = _finite_nonnegative(datum.value)
                if processing_seconds is not None:
                    metrics["llm_processing_time"] = processing_seconds
                    self._pending_llm_processing_seconds[llm_response_owner] = processing_seconds
            elif isinstance(datum, TurnMetricsData):
                # Pipecat's Smart Turn implementation measures only the model
                # prediction call here. Keep it distinct from ``user_turn_secs``
                # below, which also includes VAD silence and turn-release work.
                inference_ms = _finite_nonnegative(datum.e2e_processing_time_ms)
                if inference_ms is not None:
                    metrics["smart_turn_inference"] = inference_ms / 1000.0
            elif isinstance(datum, LLMUsageMetricsData) and category == "llm":
                usage = datum.value
                completion_tokens = _finite_nonnegative(usage.completion_tokens)
                if completion_tokens is not None:
                    self._pending_llm_completion_tokens[llm_response_owner] = completion_tokens
                if llm_response_owner == self._controller.active_response_id:
                    self._controller.set_response_usage(_realtime_usage(usage))

            if metrics:
                await self._emit(
                    self._server_event(
                        "nvidia.metrics.updated",
                        metrics=metrics,
                        source={
                            "processor": processor,
                            "model": getattr(datum, "model", None),
                            "kind": type(datum).__name__,
                        },
                    )
                )

        complete_response_ids = self._pending_llm_completion_tokens.keys() & self._pending_llm_processing_seconds.keys()
        for response_id in sorted(complete_response_ids):
            completion_tokens = self._pending_llm_completion_tokens.pop(response_id)
            processing_seconds = self._pending_llm_processing_seconds.pop(response_id)
            if processing_seconds <= 0:
                continue
            await self._emit(
                self._server_event(
                    "nvidia.metrics.updated",
                    metrics={"llm_tokens_per_sec": completion_tokens / processing_seconds},
                    source={"kind": "derived", "from": ["LLMUsageMetricsData", "ProcessingMetricsData"]},
                )
            )

    async def _handle_server_message(self, frame: RTVIServerMessageFrame) -> None:
        """Handle explicit turn barriers and project latency metric messages."""
        data = frame.data
        if not isinstance(data, Mapping):
            return
        message_type = data.get("type")
        if message_type == "user-turn-finalized":
            if not self._controller.manual_input_mode:
                return
            raw_turn_frame_id = data.get("turn_frame_id")
            item_id = (
                self._manual_input_item_by_turn_frame_id.get(raw_turn_frame_id)
                if _is_frame_id(raw_turn_frame_id)
                else None
            )
            if item_id is None:
                # Never transfer an uncorrelated or stale semantic marker to the
                # oldest FIFO item. The commit watchdog remains authoritative
                # for an item whose internal marker was lost.
                logger.warning(f"Ignoring uncorrelated manual turn barrier turn_frame_id={raw_turn_frame_id!r}")
                return
            transcript = data.get("transcript")
            if not isinstance(transcript, str) or not transcript.strip():
                await self._fail_empty_manual_transcription(item_id=item_id)
            else:
                await self._finalize_manual_transcription(item_id=item_id, transcript=transcript)
            return

        metrics: dict[str, float] = {}
        if message_type == "user-bot-latency" and not bool(data.get("first")):
            latency = _finite_nonnegative(data.get("latency"))
            if latency is not None:
                metrics["server_e2e"] = latency
        elif message_type == "latency-breakdown":
            latency = _finite_nonnegative(data.get("vad_smart_turn"))
            if latency is not None:
                metrics["vad_smart_turn"] = latency
        if metrics:
            await self._emit(
                self._server_event(
                    "nvidia.metrics.updated",
                    metrics=metrics,
                    source={"kind": "UserBotLatencyObserver", "message_type": message_type},
                )
            )

    async def _finalize_manual_transcription(self, *, item_id: str, transcript: str) -> None:
        """Publish one exact manual transcript at Pipecat's semantic turn barrier."""
        async with self._input_transcription_lock:
            if self._input_transcription_terminal or self._input_transcription_shutdown:
                return
            finalized_item_id, events = self._controller.finalize_user_transcript(
                item_id=item_id,
                transcript=transcript,
            )
            if not events:
                logger.warning(f"Ignoring stale manual turn barrier item_id={item_id}")
                return
            events, _close_for_transcription_failure = self._prepare_input_transcription_events(
                finalized_item_id,
                events,
            )
            await emit_events(self._emit_batch, events)

    async def _fail_empty_manual_transcription(self, *, item_id: str) -> None:
        """Fail the sole manual commit when ASR settles without usable text."""
        message = "Committed input audio did not produce a non-empty transcript"
        async with self._input_transcription_lock:
            if self._input_transcription_terminal or self._input_transcription_shutdown:
                return
            events = self._controller.fail_user_transcription(
                code="empty_transcript",
                message=message,
                item_id=item_id,
            )
            if not events:
                return
            item_id = events[0].get("item_id")
            if isinstance(item_id, str) and item_id:
                self._complete_input_transcription(item_id)
            events.append(
                error_event(
                    message,
                    code="input_audio_transcription_empty",
                    param="session.audio.input.transcription",
                    error_type="server_error",
                )
            )
            if self._controller.response_in_progress:
                events.extend(
                    self._controller.finish_response(
                        status="failed",
                        reason="input_audio_transcription_empty",
                    )
                )
        await self._emit_batch(events)

    async def _handle_transcript_producer_end(self, frame: UserTranscriptProducerEndedFrame) -> None:
        """Fail a fused transcript only after its ordered producer terminal arrives."""
        if frame.status == "failed":
            code = "omni_transcription_provider_error"
            message = "The fused model failed before producing the input audio transcript"
        elif frame.status == "cancelled":
            code = "omni_transcription_cancelled"
            message = "The fused model was cancelled before producing the input audio transcript"
        elif frame.status == "skipped":
            code = "omni_transcription_skipped"
            message = "The fused audio turn was empty or shorter than the configured minimum"
        elif frame.status == "overflowed":
            code = "input_buffer_overflow"
            message = "The fused input audio exceeded the configured maximum turn duration"
        else:
            code = "omni_transcription_missing"
            message = "The fused model response ended without an input audio transcript"

        close_for_transcription_failure = False
        async with self._input_transcription_lock:
            if self._input_transcription_terminal or self._input_transcription_shutdown:
                return
            turn_frame_id = frame.turn_frame_id
            item_id = (
                self._input_item_by_turn_frame_id.pop(turn_frame_id, None) if _is_frame_id(turn_frame_id) else None
            )
            if item_id is None:
                if _is_frame_id(turn_frame_id) and turn_frame_id in self._retired_input_turn_frame_ids:
                    return
                await self._emit_fused_transcription_owner_missing(turn_frame_id)
                close_for_transcription_failure = True
            else:
                self._retire_input_turn_frame_id(turn_frame_id)
                item_id, events = self._controller.end_user_transcript_producer(
                    item_id=item_id,
                    code=code,
                    message=message,
                )
                events, close_for_transcription_failure = self._prepare_input_transcription_events(
                    item_id,
                    events,
                )
                await emit_events(self._emit_batch, events)
        if close_for_transcription_failure and self._fail_input_audio_turn is not None:
            await self._fail_input_audio_turn()

    async def _announce_function_call(self, *, call_id: str, name: str, arguments: Any) -> None:
        if call_id and call_id in self._emitted_function_calls:
            return
        events = self._controller.start_function_call(call_id=call_id, name=name, arguments=arguments)
        await emit_events(self._emit_batch, events)
        if call_id:
            self._emitted_function_calls.add(call_id)
            record = self._controller.tool_call(call_id)
            if self._mcp_runtime is not None and record is not None and record.owner == "mcp":
                self._mcp_runtime.notify_call_announced(call_id)

    async def _record_function_result(self, frame: FunctionCallResultFrame) -> None:
        """Project a late/direct terminal tool result as one atomic batch."""
        if isinstance(frame.result, ClientToolTimeoutResult) and self._wait_client_tool_context is not None:
            await self._wait_client_tool_context(frame.tool_call_id or "")
        await emit_events(self._emit_batch, self._function_result_events(frame))

    def _function_result_events(self, frame: FunctionCallResultFrame) -> list[dict[str, Any]]:
        """Apply one terminal tool result synchronously and return its wire events.

        Keeping controller mutation await-free lets ``_finish_response`` commit
        Response A and every already-buffered function output before another
        pipeline task can start Response B.
        """
        call_id = frame.tool_call_id or ""
        record = self._controller.tool_call(call_id)
        if record is None:
            return [
                error_event(
                    f"Tool result has no correlated function call: {call_id}",
                    code="call_not_found",
                    param="item.call_id",
                    error_type="server_error",
                )
            ]
        if record.retired:
            # Pipecat handlers can finish after barge-in has already retired
            # Response A. That internal completion is expected cleanup, not a
            # second client-visible failure. Client-submitted late outputs go
            # through the serializer/controller and still receive the
            # deterministic ``tool_call_not_active`` protocol error.
            self._emitted_function_calls.discard(call_id)
            logger.debug(
                f"Discarding internal tool result for retired response_id={record.response_id} call_id={call_id}"
            )
            return []
        if record.owner == "mcp":
            # Realtime MCP terminal publication belongs to the result's
            # on_context_updated callback. Observing this internal frame here
            # happens before Pipecat has committed it to the shared context.
            self._emitted_function_calls.discard(call_id)
            return []
        if record.owner == "client" and isinstance(frame.result, ClientToolCancelledResult):
            # This typed internal result exists only to replace Pipecat's
            # in-progress context entry during cancellation. It is never a
            # client-authored function output and must not appear on the wire.
            self._emitted_function_calls.discard(call_id)
            logger.debug(f"Discarding internal client-tool cancellation call_id={call_id}")
            return []
        if (
            record.owner == "client"
            and frame.properties is not None
            and frame.properties.run_llm is False
            and frame.properties.on_context_updated is not None
            and not isinstance(frame.result, ClientToolTimeoutResult)
        ):
            # The broker uses this property pair when applying an already
            # accepted client output to Pipecat context. The serializer owns
            # its wire item; a very fast context callback can reach this edge
            # just before that controller transition completes.
            self._emitted_function_calls.discard(call_id)
            logger.debug(f"Discarding internal client-tool context confirmation call_id={call_id}")
            return []
        if record.owner == "client":
            if record.completed:
                # Client output was already published by the serializer. This
                # frame only confirms Pipecat applied it to LLM context.
                self._emitted_function_calls.discard(call_id)
                return []
            if not isinstance(frame.result, ClientToolTimeoutResult):
                return [
                    error_event(
                        f"Client-owned tool {record.name!r} was executed by the pipeline",
                        code="tool_owner_mismatch",
                        param="item.call_id",
                        error_type="server_error",
                    )
                ]
        if record.completed:
            return [
                error_event(
                    f"Tool {record.name!r} ({call_id}) produced more than one terminal result",
                    code="duplicate_tool_output",
                    param="item.call_id",
                    metadata={
                        "kind": "tool_failure",
                        "call_id": call_id,
                        "tool_name": record.name,
                    },
                    error_type="server_error",
                )
            ]
        result, serialization_failure = _serialize_tool_result(frame.result)
        try:
            events = self._controller.add_function_output(
                call_id=call_id,
                output=result,
                owner=record.owner,
            )
        except RealtimeProtocolError as exc:
            return [
                RealtimeProtocolError(
                    message=exc.message,
                    code=exc.code,
                    param=exc.param,
                    error_type="server_error",
                ).to_event()
            ]
        failure = serialization_failure or _tool_failure(frame.result)
        if failure is not None:
            code, message = failure
            events.append(
                error_event(
                    f"Tool {record.name!r} ({call_id}) failed: {message}",
                    code=code,
                    param="item.call_id",
                    metadata={
                        "kind": "tool_failure",
                        "call_id": call_id,
                        "tool_name": record.name,
                    },
                    error_type="server_error",
                )
            )
        self._emitted_function_calls.discard(call_id)
        return events

    async def _finish_response(
        self,
        *,
        status: str,
        reason: str | None = None,
        prefix_events: list[dict[str, Any]] | None = None,
    ) -> None:
        events = list(prefix_events or [])
        if not self._controller.response_in_progress:
            self._clear_response_buffers()
            await emit_events(self._emit_batch, events)
            return
        finishing_response_id = self._controller.active_response_id
        finishes_tool_response = self._tool_response_id == finishing_response_id
        if (
            self._controller.output_kind == "audio"
            and not self._bot_transcript_from_tts
            and self._llm_text_buffer
            and self._llm_text_response_id == self._controller.active_response_id
            and self._controller.assistant_item_id is not None
            and not self._controller.assistant_text
        ):
            events.extend(self._controller.append_assistant_text(self._llm_text_buffer))
        events.extend(
            self._controller.finish_response(
                status=status,  # type: ignore[arg-type]
                reason=reason,
            )
        )
        if finishes_tool_response:
            self._tool_response_id = None
            deferred_client_timeouts: list[FunctionCallResultFrame] = []
            while self._pending_function_results:
                result_frame = self._pending_function_results.popleft()
                if (
                    isinstance(result_frame.result, ClientToolTimeoutResult)
                    and self._wait_client_tool_context is not None
                ):
                    deferred_client_timeouts.append(result_frame)
                else:
                    events.extend(self._function_result_events(result_frame))
        await emit_events(
            self._emit_batch,
            events,
        )
        if finishes_tool_response:
            for result_frame in deferred_client_timeouts:
                await self._record_function_result(result_frame)
        self._clear_response_buffers()
        if finishing_response_id is not None:
            self._clear_tts_response(finishing_response_id)
            self._pending_llm_completion_tokens.pop(finishing_response_id, None)
            self._pending_llm_processing_seconds.pop(finishing_response_id, None)
            self._provider_terminal_by_response.pop(finishing_response_id, None)

    def _input_cursor(self, processor: Any) -> tuple[int, int]:
        """Return the PCM cursor seen when the emitting processor dequeued audio."""
        return self._input_audio_cursors.get(processor, (0, 16000))

    def _clear_tts_response(self, response_id: str) -> None:
        """Forget every physical TTS context owned by a terminal response."""
        for context_id in self._pending_tts_contexts.pop(response_id, set()):
            self._tts_context_response.pop(context_id, None)
        self._sealed_tts_responses.discard(response_id)

    def _provider_terminal(self, response_id: str) -> tuple[str, str | None]:
        """Return the native Realtime terminal mapped from provider metadata."""
        return self._provider_terminal_by_response.get(response_id, ("completed", None))

    def _current_llm_response_owner(self, processor: Any) -> str | None:
        """Return the generation opened by the newest start-frame boundary.

        Cancelled generations can remain queued until their delayed end frame
        arrives. Content emitted after a newer start belongs to that newer
        generation, while end frames still drain the queue in FIFO order.
        """
        owners = self._llm_response_owners.get(processor)
        return owners[-1] if owners else None

    def _pop_llm_response_owner(self, processor: Any) -> str | None:
        owners = self._llm_response_owners.get(processor)
        owner_processor = processor
        if not owners:
            candidates = [(candidate, queue) for candidate, queue in self._llm_response_owners.items() if queue]
            if len(candidates) != 1:
                return None
            owner_processor, owners = candidates[0]
        response_id = owners.popleft()
        if not owners:
            self._llm_response_owners.pop(owner_processor, None)
        return response_id

    @staticmethod
    def _remember_response_owner(owners: OrderedDict[int, str], frame_id: int, response_id: str) -> None:
        owners[frame_id] = response_id
        owners.move_to_end(frame_id)
        if len(owners) > 4096:
            owners.popitem(last=False)

    def _remember_completed_llm_end(self, frame_id: int) -> None:
        """Keep terminal LLM sync-frame IDs independent of high-volume PCM."""
        self._completed_llm_end_frames[frame_id] = None
        self._completed_llm_end_frames.move_to_end(frame_id)
        if len(self._completed_llm_end_frames) > self._max_frames:
            self._completed_llm_end_frames.popitem(last=False)

    @staticmethod
    def _server_event(event_type: str, **payload: Any) -> dict[str, Any]:
        from realtime.protocol import build_server_event

        return build_server_event(event_type, **payload)


def _tool_failure(result: Any) -> tuple[str, str] | None:
    """Extract a safe client-visible failure from a terminal tool envelope."""
    if not isinstance(result, Mapping):
        return None
    raw_error = result.get("error")
    if result.get("ok") is not False and raw_error is None:
        return None
    if isinstance(raw_error, Mapping):
        code = str(raw_error.get("code") or "tool_execution_error")
        message = str(raw_error.get("message") or "Tool execution failed")
        return code, message
    return "tool_execution_error", str(raw_error or "Tool execution failed")


def _serialize_tool_result(result: Any) -> tuple[str, tuple[str, str] | None]:
    """Return an opaque function output string and an optional safe failure."""
    if isinstance(result, str):
        return result, None
    try:
        return json.dumps(result, ensure_ascii=False, allow_nan=False), None
    except (TypeError, ValueError, OverflowError):
        code = "tool_result_serialization_error"
        message = "Tool result could not be serialized as finite JSON"
        envelope = {
            "ok": False,
            "error": {
                "code": code,
                "message": message,
            },
        }
        return json.dumps(envelope), (code, message)


def _processor_category(processor: str) -> str:
    name = processor.lower()
    if "asr" in name or "stt" in name:
        return "asr"
    if "tts" in name:
        return "tts"
    if "llm" in name:
        return "llm"
    return ""


def _is_frame_id(value: object) -> bool:
    """Return whether a value is a Pipecat frame identifier, excluding bool."""
    return isinstance(value, int) and not isinstance(value, bool)


def _finite_nonnegative(value: Any) -> float | None:
    """Return an interoperable metric value, excluding booleans and NaN/Inf."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    try:
        numeric = float(value)
    except OverflowError:
        return None
    return numeric if math.isfinite(numeric) and numeric >= 0 else None


def _realtime_usage(usage: Any) -> dict[str, Any]:
    """Map Pipecat usage without double-counting audio or cached tokens.

    Pipecat normalizes provider usage such that ``total_tokens`` is gross, but
    ``prompt_tokens`` may be net of a separately reported cache read. Realtime
    requires cached tokens to remain a subset of gross input tokens, while the
    text/audio detail fields must partition their parent counts.
    """
    completion_tokens = _nonnegative_int(usage.completion_tokens)
    reported_total = _nonnegative_int(usage.total_tokens)
    prompt_tokens = _nonnegative_int(usage.prompt_tokens)
    cached_tokens = _bounded_tokens(usage.cache_read_input_tokens, reported_total)

    # Prefer the provider's gross total. If it is inconsistent, recover a
    # conservative gross input from the prompt and cache components.
    if reported_total >= completion_tokens and reported_total - completion_tokens >= prompt_tokens:
        input_tokens = reported_total - completion_tokens
    else:
        input_tokens = prompt_tokens
        reported_total = input_tokens + completion_tokens
    cached_tokens = min(cached_tokens, input_tokens)

    reported_audio_tokens = _bounded_tokens(usage.input_audio_tokens, input_tokens)
    cached_audio_tokens = _bounded_tokens(
        usage.cache_read_input_audio_tokens,
        cached_tokens,
    )
    cache_creation_tokens = _bounded_tokens(usage.cache_creation_input_tokens, input_tokens)
    cache_is_reported_separately = input_tokens == (prompt_tokens + cached_tokens + cache_creation_tokens)
    input_audio_tokens = min(
        input_tokens,
        reported_audio_tokens + cached_audio_tokens
        if cache_is_reported_separately
        else max(reported_audio_tokens, cached_audio_tokens),
    )
    output_audio_tokens = _bounded_tokens(usage.output_audio_tokens, completion_tokens)

    input_details: dict[str, Any] = {
        "text_tokens": input_tokens - input_audio_tokens,
        "audio_tokens": input_audio_tokens,
        "cached_tokens": cached_tokens,
    }
    if usage.cache_read_input_audio_tokens is not None:
        input_details["cached_tokens_details"] = {
            "text_tokens": cached_tokens - cached_audio_tokens,
            "audio_tokens": cached_audio_tokens,
        }
    output_details: dict[str, int] = {
        "text_tokens": completion_tokens - output_audio_tokens,
        "audio_tokens": output_audio_tokens,
    }
    return {
        "total_tokens": reported_total,
        "input_tokens": input_tokens,
        "output_tokens": completion_tokens,
        "input_token_details": input_details,
        "output_token_details": output_details,
    }


def _nonnegative_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError, OverflowError):
        return 0


def _bounded_tokens(value: Any, upper_bound: int) -> int:
    return min(_nonnegative_int(value), max(0, upper_bound))
