# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pipeline observer that emits Realtime transcript / lifecycle events."""

from __future__ import annotations

from collections import deque
from typing import Any

from loguru import logger
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    ErrorFrame,
    Frame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    FunctionCallsStartedFrame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMTextFrame,
    MetricsFrame,
    TranscriptionFrame,
    TTSStoppedFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import TTFBMetricsData
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.processors.frame_processor import FrameDirection
from pipecat.processors.frameworks.rtvi.frames import RTVIServerMessageFrame
from pipecat.transports.base_output import BaseOutputTransport

from realtime.conversation import ConversationState
from realtime.events import (
    SERVER_AUDIO_COMMITTED,
    SERVER_INPUT_TRANSCRIPT_COMPLETED,
    SERVER_INPUT_TRANSCRIPT_DELTA,
    SERVER_ITEM_CREATED,
    SERVER_NVIDIA_TOOL_COMPLETED,
    SERVER_NVIDIA_TOOL_STARTED,
    SERVER_OUTPUT_AUDIO_TRANSCRIPT_DELTA,
    SERVER_SPEECH_STARTED,
    SERVER_SPEECH_STOPPED,
    EmitFn,
    emit_with_aliases,
    error_event,
    server_event,
)
from realtime.lifecycle import announce_response, finish_response
from realtime.metrics import RealtimeMetricsAccumulator, RealtimeResponseTelemetry

# Frames this observer maps to Realtime events (ignore the rest for dedupe).
_OBSERVED_FRAME_TYPES = (
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    BotStartedSpeakingFrame,
    TTSTextFrame,
    LLMTextFrame,
    FunctionCallsStartedFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    LLMFullResponseEndFrame,
    TTSStoppedFrame,
    ErrorFrame,
    InterruptionFrame,
)


class RealtimeLifecycleObserver(BaseObserver):
    """Emit OpenAI Realtime–shaped events from Pipecat frame traffic.

    Spoken turns finish on the output transport's re-pushed ``TTSStoppedFrame``
    after queued audio has been serialized. Pipeline barge-in finishes on
    ``InterruptionFrame``.
    """

    def __init__(
        self,
        *,
        emit: EmitFn,
        conversation: ConversationState,
        max_frames: int = 4096,
        **kwargs: Any,
    ) -> None:
        """Create an observer bound to shared conversation state + emit callback."""
        super().__init__(**kwargs)
        self._emit_fn = emit
        self._conversation = conversation
        self._processed_frames: set[int] = set()
        self._frame_history: deque[int] = deque(maxlen=max_frames)
        self._processed_metric_frames: set[int] = set()
        self._metric_frame_history: deque[int] = deque(maxlen=max_frames)
        self._bot_transcript_from_tts = False
        self._emitted_function_calls: set[str] = set()
        self._llm_text_buffer = ""
        self._llm_text_generation = 0
        self._metrics = RealtimeMetricsAccumulator()
        self._waiting_for_metrics_scope_after_cancel = False
        self._speech_started_emitted = False
        self._speech_stopped_emitted = False

    def _clear_frame_dedupe(self) -> None:
        self._processed_frames.clear()
        self._frame_history.clear()

    def _remember_metric_frame(self, frame_id: int) -> bool:
        """Deduplicate metric frames without changing lifecycle dedupe state."""
        if frame_id in self._processed_metric_frames:
            return False
        self._processed_metric_frames.add(frame_id)
        self._metric_frame_history.append(frame_id)
        if len(self._processed_metric_frames) > len(self._metric_frame_history):
            self._processed_metric_frames = set(self._metric_frame_history)
        return True

    def _remember_frame(self, frame_id: int) -> bool:
        """Return True if this frame id is new and should be handled."""
        if frame_id in self._processed_frames:
            return False
        self._processed_frames.add(frame_id)
        self._frame_history.append(frame_id)
        if len(self._processed_frames) > len(self._frame_history):
            self._processed_frames = set(self._frame_history)
        return True

    async def _emit_event(self, event: dict[str, Any]) -> None:
        await emit_with_aliases(self._emit_fn, event)

    async def _emit_transcript_delta(self, text: str) -> None:
        await self._emit_event(
            server_event(
                SERVER_OUTPUT_AUDIO_TRANSCRIPT_DELTA,
                response_id=self._conversation.response_id,
                item_id=self._conversation.assistant_item_id,
                output_index=0,
                content_index=0,
                delta=text,
            )
        )

    async def _emit_user_item_if_needed(self, item_id: str) -> None:
        """Create the committed user audio item exactly once."""
        if not self._conversation.announce_user_item():
            return
        transcript = self._conversation.pending_user_transcript or ""
        await self._emit_event(
            server_event(
                SERVER_ITEM_CREATED,
                previous_item_id=None,
                item={
                    "id": item_id,
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_audio", "transcript": transcript}],
                },
            )
        )

    async def _emit_user_transcript_if_ready(self) -> None:
        """Complete transcription after stop/commit and item creation."""
        ready = self._conversation.ready_user_transcript()
        if ready is None:
            return
        item_id, transcript = ready
        await self._emit_user_item_if_needed(item_id)
        await self._emit_event(
            server_event(
                SERVER_INPUT_TRANSCRIPT_COMPLETED,
                item_id=item_id,
                content_index=0,
                transcript=transcript,
            )
        )
        self._conversation.clear_user_item()

    def _input_audio_ms(self) -> int:
        """Return the current input-audio clock position."""
        return int(self._conversation.input_audio_samples * 1000 / max(self._conversation.input_sample_rate, 1))

    async def _emit_speech_started(self, item_id: str, audio_start_ms: int) -> None:
        if self._speech_started_emitted:
            return
        self._speech_started_emitted = True
        await self._emit_event(
            server_event(
                SERVER_SPEECH_STARTED,
                item_id=item_id,
                audio_start_ms=audio_start_ms,
            )
        )

    async def _emit_speech_stopped(self, item_id: str, audio_end_ms: int) -> None:
        if not self._speech_started_emitted or self._speech_stopped_emitted:
            return
        self._speech_stopped_emitted = True
        await self._emit_event(
            server_event(
                SERVER_SPEECH_STOPPED,
                item_id=item_id,
                audio_end_ms=audio_end_ms,
            )
        )

    def _reset_speech_events(self) -> None:
        self._speech_started_emitted = False
        self._speech_stopped_emitted = False

    def take_response_telemetry(self) -> RealtimeResponseTelemetry:
        """Snapshot passively collected metrics for the next ``response.done``."""
        try:
            return self._metrics.finish()
        except Exception:
            logger.exception("Realtime metrics snapshot failed; emitting response without telemetry")
            self._metrics.clear()
            return RealtimeResponseTelemetry()

    def on_response_cancelled(self) -> str:
        """Invalidate buffered LLM text for a cancelled turn; return drained text."""
        text = self._llm_text_buffer
        self._llm_text_buffer = ""
        self._llm_text_generation = 0
        self._bot_transcript_from_tts = False
        self._waiting_for_metrics_scope_after_cancel = True
        self._clear_frame_dedupe()
        return text

    def shutdown(self) -> None:
        """Clear buffers on transport / session teardown."""
        self._llm_text_buffer = ""
        self._llm_text_generation = 0
        self._metrics.clear()
        self._waiting_for_metrics_scope_after_cancel = False
        self._processed_metric_frames.clear()
        self._metric_frame_history.clear()
        self._reset_speech_events()
        self._clear_frame_dedupe()

    async def on_push_frame(self, data: FramePushed) -> None:
        """Map relevant frames to Realtime server events.

        ``InterruptionFrame`` is handled in either direction (barge-in often
        travels upstream). Other lifecycle frames are downstream-only.
        """
        frame = data.frame
        if isinstance(frame, ErrorFrame):
            if not self._remember_frame(frame.id):
                return
            try:
                await self._emit_event(error_event(frame.error, code="pipeline_error"))
                if frame.fatal:
                    if self._conversation.response_requested:
                        await self._ensure_response_announced()
                    await self._finish_response(status="failed")
                else:
                    self._conversation.release_response_request()
            except Exception:
                logger.exception("RealtimeLifecycleObserver failed while handling pipeline error")
            return

        if isinstance(frame, InterruptionFrame):
            if not self._remember_frame(frame.id):
                return
            try:
                await self._finish_response(status="cancelled")
            except Exception:
                logger.exception("RealtimeLifecycleObserver failed while handling interruption")
            return

        # Pipecat pushes the same TTSStoppedFrame twice: first from TTS into the
        # output transport, then from BaseOutputTransport after its FIFO audio
        # queue has drained. Ignore the first edge without consuming the frame id.
        if isinstance(frame, TTSStoppedFrame) and not isinstance(data.source, BaseOutputTransport):
            return

        if data.direction != FrameDirection.DOWNSTREAM:
            return
        if isinstance(frame, (MetricsFrame, RTVIServerMessageFrame)):
            if self._remember_metric_frame(frame.id):
                try:
                    if self._waiting_for_metrics_scope_after_cancel:
                        response_started = (
                            self._conversation.response_requested or self._conversation.response_status == "in_progress"
                        )
                        new_asr_started = isinstance(frame, MetricsFrame) and any(
                            isinstance(item, TTFBMetricsData)
                            and ("asr" in item.processor.lower() or "stt" in item.processor.lower())
                            for item in frame.data
                        )
                        if not response_started and not new_asr_started:
                            return
                        self._metrics.clear()
                        self._waiting_for_metrics_scope_after_cancel = False
                    self._metrics.consume(frame)
                except Exception:
                    logger.exception("Realtime metrics collection failed; ignoring metric frame")
            return
        if not isinstance(frame, _OBSERVED_FRAME_TYPES):
            return
        if not self._remember_frame(frame.id):
            return

        try:
            await self._handle_frame(frame)
        except Exception:
            logger.exception("RealtimeLifecycleObserver failed while handling frame")

    async def _handle_frame(self, frame: Frame) -> None:
        if isinstance(frame, VADUserStartedSpeakingFrame):
            item_id = self._conversation.begin_user_item()
            await self._emit_speech_started(item_id, self._input_audio_ms())
            return

        if isinstance(frame, VADUserStoppedSpeakingFrame):
            if not self._speech_started_emitted:
                return
            item_id = self._conversation.begin_user_item()
            await self._emit_speech_stopped(item_id, self._input_audio_ms())
            return

        if isinstance(frame, UserStartedSpeakingFrame):
            item_id, audio_start_ms = self._conversation.begin_user_turn()
            # Providers and pipelines without raw VAD frames retain the
            # finalized-turn fallback.
            await self._emit_speech_started(item_id, audio_start_ms)
            return

        if isinstance(frame, UserStoppedSpeakingFrame):
            item_id, audio_end_ms = self._conversation.stop_user_turn()
            await self._emit_speech_stopped(item_id, audio_end_ms)
            await self._emit_event(
                server_event(
                    SERVER_AUDIO_COMMITTED,
                    item_id=item_id,
                    previous_item_id=None,
                )
            )
            await self._emit_user_item_if_needed(item_id)
            await self._emit_user_transcript_if_ready()
            self._reset_speech_events()
            return

        if isinstance(frame, InterimTranscriptionFrame):
            if self._conversation.user_turn_start_sample is None or self._conversation.user_turn_stopped:
                return
            item_id = self._conversation.begin_user_item()
            await self._emit_event(
                server_event(
                    SERVER_INPUT_TRANSCRIPT_DELTA,
                    item_id=item_id,
                    content_index=0,
                    delta=frame.text or "",
                )
            )
            return

        if isinstance(frame, TranscriptionFrame):
            transcript = frame.text or ""
            self._conversation.set_user_transcript(transcript)
            await self._emit_user_transcript_if_ready()
            return

        if isinstance(frame, BotStartedSpeakingFrame):
            _, created = await self._ensure_response_announced()
            if created:
                self._bot_transcript_from_tts = False
                self._llm_text_buffer = ""
                self._llm_text_generation = 0
            return

        if isinstance(frame, TTSTextFrame):
            text = frame.text or ""
            if not text:
                return
            await self._ensure_response_announced()
            if self._conversation.assistant_transcript.endswith(text):
                self._bot_transcript_from_tts = True
                self._llm_text_buffer = ""
                self._llm_text_generation = 0
                return
            self._conversation.append_assistant_transcript(text)
            self._bot_transcript_from_tts = True
            self._llm_text_buffer = ""
            self._llm_text_generation = 0
            await self._emit_transcript_delta(text)
            return

        if isinstance(frame, LLMTextFrame):
            text = frame.text or ""
            if not text:
                return
            await self._ensure_response_announced()
            self._llm_text_buffer += text
            self._llm_text_generation = self._conversation.response_generation
            return

        if isinstance(frame, LLMFullResponseEndFrame):
            if frame.skip_tts is True:
                await self._finish_response(status="completed")
            return

        if isinstance(frame, FunctionCallsStartedFrame):
            for call in frame.function_calls or []:
                await self._emit_server_tool_started(
                    tool_call_id=getattr(call, "tool_call_id", "") or "",
                    function_name=getattr(call, "function_name", "") or "",
                    arguments=getattr(call, "arguments", None),
                )
            return

        if isinstance(frame, FunctionCallInProgressFrame):
            await self._emit_server_tool_started(
                tool_call_id=frame.tool_call_id or "",
                function_name=frame.function_name or "",
                arguments=frame.arguments,
            )
            return

        if isinstance(frame, FunctionCallResultFrame):
            await self._emit_event(
                server_event(
                    SERVER_NVIDIA_TOOL_COMPLETED,
                    tool_call_id=frame.tool_call_id or "",
                    name=frame.function_name or "",
                    arguments=frame.arguments,
                    result=frame.result,
                )
            )
            if frame.tool_call_id:
                self._emitted_function_calls.discard(frame.tool_call_id)
            return

        if isinstance(frame, TTSStoppedFrame):
            await self._finish_response(status="completed")
            return

    async def _ensure_response_announced(self) -> tuple[str, bool]:
        return await announce_response(self._conversation, self._emit_fn)

    async def _emit_server_tool_started(
        self,
        *,
        tool_call_id: str,
        function_name: str,
        arguments: Any,
    ) -> None:
        call_id = tool_call_id or ""
        if call_id and call_id in self._emitted_function_calls:
            return
        if call_id:
            self._emitted_function_calls.add(call_id)
        await self._emit_event(
            server_event(
                SERVER_NVIDIA_TOOL_STARTED,
                tool_call_id=call_id,
                name=str(function_name or ""),
                arguments=arguments if arguments is not None else {},
            )
        )

    async def _finish_response(self, *, status: str) -> None:
        if self._conversation.response_status != "in_progress":
            self._llm_text_buffer = ""
            self._llm_text_generation = 0
            self._bot_transcript_from_tts = False
            return

        buffer = self._llm_text_buffer
        if self._llm_text_generation != self._conversation.response_generation:
            buffer = ""

        if not self._bot_transcript_from_tts and buffer and not self._conversation.assistant_transcript:
            self._conversation.append_assistant_transcript(buffer)
            await self._emit_transcript_delta(buffer)

        await finish_response(
            self._conversation,
            self._emit_fn,
            status=status,
            telemetry=self.take_response_telemetry(),
        )
        self._bot_transcript_from_tts = False
        self._llm_text_buffer = ""
        self._llm_text_generation = 0
        self._clear_frame_dedupe()
