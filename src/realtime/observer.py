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
    FunctionCallCancelFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    FunctionCallsStartedFrame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    MetricsFrame,
    TranscriptionFrame,
    TTSStoppedFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.processors.frame_processor import FrameDirection
from pipecat.processors.frameworks.rtvi.frames import RTVIServerMessageFrame
from pipecat.services.llm_service import LLMService
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.utils.errors import ErrorCategory

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
from realtime.lifecycle import announce_response, announce_response_created, finish_response
from realtime.metrics import RealtimeMetricsAccumulator, RealtimeResponseTelemetry

# Frames this observer maps to Realtime events (ignore the rest for dedupe).
_OBSERVED_FRAME_TYPES = (
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
    FunctionCallCancelFrame,
    LLMFullResponseStartFrame,
    LLMFullResponseEndFrame,
    TTSStoppedFrame,
    ErrorFrame,
    InterruptionFrame,
)


def _is_speakable(text: str) -> bool:
    """Return whether TTS would voice ``text`` (same rule as ``NvidiaTTSService``)."""
    return any(char.isalnum() for char in text)


def _tool_result_runs_llm(frame: FunctionCallResultFrame) -> bool:
    """Return whether the assistant aggregator runs the LLM again for this result."""
    if not frame.result:
        return False
    if frame.properties is not None and frame.properties.run_llm is not None:
        return bool(frame.properties.run_llm)
    if frame.run_llm is not None:
        return bool(frame.run_llm)
    return True


class RealtimeLifecycleObserver(BaseObserver):
    """Emit OpenAI Realtime–shaped events from Pipecat frame traffic.

    Spoken turns finish on the output transport's re-pushed ``TTSStoppedFrame``
    after queued audio has been serialized. A response that produced no speech
    (for example an empty reply to ``response.create``) finishes when its LLM
    run ends and no tool follow-up is pending. Pipeline barge-in finishes on
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
        # LLM runs and tool calls are tied to the response generation that was in
        # progress when they started (0 when none was), so that a run or tool
        # result left over from an earlier turn cannot finish a newer response.
        self._active_llm_runs = 0
        self._run_generation = 0
        self._tool_calls: dict[str, int] = {}
        self._followup_generation: int | None = None
        self._failed_generation: int | None = None

    def _clear_frame_dedupe(self) -> None:
        self._processed_frames.clear()
        self._frame_history.clear()

    def _open_generation(self) -> int:
        """Return the in-progress response generation, or 0 when none is open."""
        if self._conversation.response_status == "in_progress":
            return self._conversation.response_generation
        return 0

    def _reset_turn_tracking(self) -> None:
        """Forget LLM runs and tool calls cancelled by an interruption."""
        self._active_llm_runs = 0
        self._run_generation = 0
        self._tool_calls.clear()
        self._followup_generation = None
        self._failed_generation = None
        self._conversation.reply_pending = False

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
        if len(self._frame_history) == self._frame_history.maxlen:
            self._processed_frames.discard(self._frame_history[0])
        self._processed_frames.add(frame_id)
        self._frame_history.append(frame_id)
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
        self._reset_turn_tracking()
        return text

    def shutdown(self) -> None:
        """Clear buffers on transport / session teardown."""
        self._llm_text_buffer = ""
        self._llm_text_generation = 0
        self._metrics.clear()
        self._processed_metric_frames.clear()
        self._metric_frame_history.clear()
        self._clear_frame_dedupe()
        self._reset_turn_tracking()

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
                        await announce_response_created(self._conversation, self._emit_fn)
                    await self._finish_response(status="failed")
                else:
                    if self._is_completion_failure(frame):
                        # The LLM run still ends normally; a reply without speech then fails.
                        self._failed_generation = self._run_generation
                    self._conversation.release_response_request()
            except Exception:
                logger.exception("RealtimeLifecycleObserver failed while handling pipeline error")
            return

        if isinstance(frame, InterruptionFrame):
            if not self._remember_frame(frame.id):
                return
            try:
                if frame.id in self._conversation.applied_interruption_ids:
                    # response.cancel / truncate already finished its response; a
                    # newer response.create must not be cancelled by this frame.
                    self._conversation.applied_interruption_ids.discard(frame.id)
                else:
                    await self._finish_response(status="cancelled")
                # Interruption cancels in-flight tool calls and the LLM run.
                self._reset_turn_tracking()
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
        if isinstance(frame, UserStartedSpeakingFrame):
            item_id, audio_start_ms = self._conversation.begin_user_turn()
            await self._emit_event(
                server_event(
                    SERVER_SPEECH_STARTED,
                    item_id=item_id,
                    audio_start_ms=audio_start_ms,
                )
            )
            return

        if isinstance(frame, UserStoppedSpeakingFrame):
            item_id, audio_end_ms = self._conversation.stop_user_turn()
            await self._emit_event(
                server_event(
                    SERVER_SPEECH_STOPPED,
                    item_id=item_id,
                    audio_end_ms=audio_end_ms,
                )
            )
            await self._emit_event(
                server_event(
                    SERVER_AUDIO_COMMITTED,
                    item_id=item_id,
                    previous_item_id=None,
                )
            )
            # The committed turn's reply is on its way; it becomes a response on first speech.
            self._conversation.reply_pending = True
            await self._emit_user_item_if_needed(item_id)
            await self._emit_user_transcript_if_ready()
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
            if _is_speakable(text):
                await self._ensure_response_announced()
            elif not self._conversation.output_item_announced or self._conversation.response_status != "in_progress":
                return
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
            if _is_speakable(text):
                await self._ensure_response_announced()
            elif self._conversation.response_status != "in_progress":
                # Whitespace or punctuation alone is never voiced, so it opens no response.
                return
            self._llm_text_buffer += text
            self._llm_text_generation = self._conversation.response_generation
            return

        if isinstance(frame, LLMFullResponseStartFrame):
            generation = self._open_generation()
            self._active_llm_runs += 1
            self._run_generation = generation
            if self._followup_generation == generation and generation not in self._tool_calls.values():
                self._followup_generation = None
            return

        if isinstance(frame, LLMFullResponseEndFrame):
            self._active_llm_runs = max(0, self._active_llm_runs - 1)
            if frame.skip_tts is True:
                await self._finish_response(status="completed")
                return
            await self._settle_generation(self._run_generation)
            return

        if isinstance(frame, FunctionCallsStartedFrame):
            for call in frame.function_calls or []:
                self._track_tool_call(getattr(call, "tool_call_id", "") or "")
                await self._emit_server_tool_started(
                    tool_call_id=getattr(call, "tool_call_id", "") or "",
                    function_name=getattr(call, "function_name", "") or "",
                    arguments=getattr(call, "arguments", None),
                )
            return

        if isinstance(frame, FunctionCallInProgressFrame):
            self._track_tool_call(frame.tool_call_id or "")
            await self._emit_server_tool_started(
                tool_call_id=frame.tool_call_id or "",
                function_name=frame.function_name or "",
                arguments=frame.arguments,
            )
            return

        if isinstance(frame, FunctionCallCancelFrame):
            await self._settle_tool_call(frame.tool_call_id or "", runs_llm=bool(frame.run_llm))
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
            if frame.properties is not None and not frame.properties.is_final:
                return
            if frame.tool_call_id:
                self._emitted_function_calls.discard(frame.tool_call_id)
            await self._settle_tool_call(frame.tool_call_id or "", runs_llm=_tool_result_runs_llm(frame))
            return

        if isinstance(frame, TTSStoppedFrame):
            await self._finish_response(status="completed")
            return

    def _track_tool_call(self, tool_call_id: str) -> None:
        if tool_call_id:
            self._tool_calls.setdefault(tool_call_id, self._open_generation())

    def _is_completion_failure(self, frame: ErrorFrame) -> bool:
        """Return whether ``frame`` reports a failed LLM completion (not a tool error)."""
        return (
            self._active_llm_runs > 0
            and isinstance(getattr(frame, "processor", None), LLMService)
            and getattr(frame, "category", None) != ErrorCategory.APPLICATION
        )

    async def _settle_tool_call(self, tool_call_id: str, *, runs_llm: bool) -> None:
        """Account for a tool call that finished or was cancelled."""
        generation = self._tool_calls.pop(tool_call_id, None)
        # Unknown ids belong to a turn this observer already moved past.
        if generation is None or generation in self._tool_calls.values():
            return
        if runs_llm:
            self._followup_generation = generation
            return
        await self._settle_generation(generation)

    async def _settle_generation(self, generation: int) -> None:
        """React to the end of an LLM run or tool round owned by ``generation``.

        A response with an announced assistant item finishes on
        ``TTSStoppedFrame``. One without an item (nothing speakable yet) has no
        audio to wait for, so it finishes once its own LLM runs and tool calls
        are done and no tool follow-up run is due.
        """
        if self._active_llm_runs or generation in self._tool_calls.values() or self._followup_generation == generation:
            return
        if generation == 0:
            # A user turn's reply ended without speech; nothing is pending any more.
            self._conversation.reply_pending = False
            return
        if generation != self._open_generation() or self._conversation.output_item_announced:
            return
        await self._finish_response(status="failed" if self._failed_generation == generation else "completed")

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

        if (
            not self._bot_transcript_from_tts
            and buffer
            and not self._conversation.assistant_transcript
            and self._conversation.output_item_announced
        ):
            self._conversation.append_assistant_transcript(buffer)
            await self._emit_transcript_delta(buffer)

        generation = self._conversation.response_generation
        await finish_response(
            self._conversation,
            self._emit_fn,
            status=status,
            telemetry=self.take_response_telemetry(),
        )
        self._bot_transcript_from_tts = False
        self._llm_text_buffer = ""
        self._llm_text_generation = 0
        # Keep frame dedupe across responses: a finished turn's frames can still
        # reach later pipeline edges and must not open a new response.
        self._tool_calls = {call_id: owner for call_id, owner in self._tool_calls.items() if owner != generation}
        if self._followup_generation == generation:
            self._followup_generation = None
        if self._failed_generation == generation:
            self._failed_generation = None
