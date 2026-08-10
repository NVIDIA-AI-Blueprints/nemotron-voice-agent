# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Pipeline observer that emits Realtime transcript / lifecycle events."""

from __future__ import annotations

import json
from collections import deque
from typing import Any

from loguru import logger
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    Frame,
    FunctionCallInProgressFrame,
    FunctionCallsStartedFrame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    LLMTextFrame,
    TranscriptionFrame,
    TTSStoppedFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.processors.frame_processor import FrameDirection

from realtime.conversation import ConversationState, new_item_id
from realtime.events import (
    SERVER_FUNCTION_CALL_ARGUMENTS_DELTA,
    SERVER_FUNCTION_CALL_ARGUMENTS_DONE,
    SERVER_INPUT_TRANSCRIPT_COMPLETED,
    SERVER_INPUT_TRANSCRIPT_DELTA,
    SERVER_ITEM_CREATED,
    SERVER_OUTPUT_AUDIO_TRANSCRIPT_DELTA,
    SERVER_OUTPUT_ITEM_ADDED,
    SERVER_OUTPUT_ITEM_DONE,
    SERVER_OUTPUT_TEXT_DELTA,
    SERVER_RESPONSE_CREATED,
    SERVER_RESPONSE_DONE,
    SERVER_SPEECH_STARTED,
    SERVER_SPEECH_STOPPED,
    EmitFn,
    emit_with_aliases,
    response_created_body,
    server_event,
)
from realtime.lifecycle import announce_response, finish_response

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
    TTSStoppedFrame,
    InterruptionFrame,
)


class RealtimeLifecycleObserver(BaseObserver):
    """Emit OpenAI Realtime–shaped events from Pipecat frame traffic.

    Spoken turns finish on ``TTSStoppedFrame``. Pipeline barge-in finishes on
    ``InterruptionFrame`` (``response.cancel`` / truncate already finish in the
    serializer before pushing the interrupt).
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
        self._bot_transcript_from_tts = False
        self._emitted_function_calls: set[str] = set()
        self._llm_text_buffer = ""
        self._llm_text_generation = 0
        self._pending_fc_output_items: list[dict[str, Any]] = []

    def _clear_frame_dedupe(self) -> None:
        self._processed_frames.clear()
        self._frame_history.clear()

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

    def on_response_cancelled(self) -> str:
        """Invalidate buffered LLM text for a cancelled turn; return drained text."""
        self._pending_fc_output_items.clear()
        text = self._llm_text_buffer
        self._llm_text_buffer = ""
        self._llm_text_generation = 0
        self._bot_transcript_from_tts = False
        self._clear_frame_dedupe()
        return text

    def shutdown(self) -> None:
        """Clear buffers on transport / session teardown."""
        self._pending_fc_output_items.clear()
        self._llm_text_buffer = ""
        self._llm_text_generation = 0
        self._clear_frame_dedupe()

    async def on_push_frame(self, data: FramePushed) -> None:
        """Map relevant frames to Realtime server events.

        ``InterruptionFrame`` is handled in either direction (barge-in often
        travels upstream). Other lifecycle frames are downstream-only.
        """
        frame = data.frame
        if isinstance(frame, InterruptionFrame):
            if not self._remember_frame(frame.id):
                return
            try:
                await self._finish_response(status="cancelled")
            except Exception:
                logger.exception("RealtimeLifecycleObserver failed while handling interruption")
            return

        if data.direction != FrameDirection.DOWNSTREAM:
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
            await self._emit_event(server_event(SERVER_SPEECH_STARTED))
            return

        if isinstance(frame, UserStoppedSpeakingFrame):
            await self._emit_event(server_event(SERVER_SPEECH_STOPPED))
            return

        if isinstance(frame, InterimTranscriptionFrame):
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
            item_id = self._conversation.begin_user_item()
            transcript = frame.text or ""
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
            await self._emit_event(
                server_event(
                    SERVER_INPUT_TRANSCRIPT_COMPLETED,
                    item_id=item_id,
                    content_index=0,
                    transcript=transcript,
                )
            )
            self._conversation.clear_user_item()
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
            self._conversation.output_text_emitted = True
            await self._emit_event(
                server_event(
                    SERVER_OUTPUT_TEXT_DELTA,
                    response_id=self._conversation.response_id,
                    item_id=self._conversation.assistant_item_id,
                    output_index=0,
                    content_index=0,
                    delta=text,
                )
            )
            return

        if isinstance(frame, FunctionCallsStartedFrame):
            calls = list(frame.function_calls or [])
            for index, call in enumerate(calls):
                await self._emit_function_call(
                    tool_call_id=getattr(call, "tool_call_id", "") or "",
                    function_name=getattr(call, "function_name", "") or "",
                    arguments=getattr(call, "arguments", None),
                    output_index=index,
                    finish_response=(index == len(calls) - 1),
                )
            return

        if isinstance(frame, FunctionCallInProgressFrame):
            await self._emit_function_call(
                tool_call_id=frame.tool_call_id or "",
                function_name=frame.function_name or "",
                arguments=frame.arguments,
                output_index=0,
                finish_response=True,
            )
            return

        if isinstance(frame, TTSStoppedFrame):
            await self._finish_response(status="completed")
            return

    async def _ensure_response_announced(self) -> tuple[str, bool]:
        return await announce_response(self._conversation, self._emit_fn)

    async def _emit_function_call(
        self,
        *,
        tool_call_id: str,
        function_name: str,
        arguments: Any,
        output_index: int = 0,
        finish_response: bool = True,
    ) -> None:
        call_id = tool_call_id or ""
        if call_id and call_id in self._emitted_function_calls:
            return
        if call_id:
            self._emitted_function_calls.add(call_id)

        response_id, created = self._conversation.begin_response()
        if created:
            await self._emit_event(
                server_event(
                    SERVER_RESPONSE_CREATED,
                    response=response_created_body(response_id),
                )
            )
        else:
            response_id = self._conversation.response_id or response_id

        if isinstance(arguments, str):
            args_json = arguments
            args_for_store: Any = arguments
        else:
            try:
                args_json = json.dumps(arguments if arguments is not None else {})
            except TypeError:
                args_json = "{}"
            args_for_store = arguments if arguments is not None else {}

        name = str(function_name or "")
        if call_id:
            self._conversation.remember_function_call(
                call_id,
                name=name,
                arguments=args_for_store,
            )

        fc_item_id = new_item_id()
        fc_item: dict[str, Any] = {
            "id": fc_item_id,
            "object": "realtime.item",
            "type": "function_call",
            "status": "in_progress",
            "name": name,
            "call_id": call_id,
            "arguments": "",
        }
        await self._emit_event(server_event(SERVER_ITEM_CREATED, previous_item_id=None, item=fc_item))
        await self._emit_event(
            server_event(
                SERVER_OUTPUT_ITEM_ADDED,
                response_id=response_id,
                output_index=output_index,
                item=fc_item,
            )
        )
        await self._emit_event(
            server_event(
                SERVER_FUNCTION_CALL_ARGUMENTS_DELTA,
                response_id=response_id,
                item_id=fc_item_id,
                output_index=output_index,
                call_id=call_id,
                delta=args_json,
            )
        )
        await self._emit_event(
            server_event(
                SERVER_FUNCTION_CALL_ARGUMENTS_DONE,
                response_id=response_id,
                item_id=fc_item_id,
                output_index=output_index,
                call_id=call_id,
                name=name,
                arguments=args_json,
            )
        )
        done_item = {
            **fc_item,
            "status": "completed",
            "arguments": args_json,
        }
        await self._emit_event(
            server_event(
                SERVER_OUTPUT_ITEM_DONE,
                response_id=response_id,
                output_index=output_index,
                item=done_item,
            )
        )
        self._pending_fc_output_items.append(done_item)

        if not finish_response:
            return

        snap = self._conversation.complete_response("completed")
        if snap is None:
            self._pending_fc_output_items.clear()
            return
        await self._emit_event(
            server_event(
                SERVER_RESPONSE_DONE,
                response={
                    "id": snap.response_id,
                    "status": "completed",
                    "output": list(self._pending_fc_output_items),
                },
            )
        )
        self._pending_fc_output_items.clear()
        self._conversation.reset_response_slot(generation=snap.generation)
        self._llm_text_buffer = ""
        self._llm_text_generation = 0
        self._bot_transcript_from_tts = False
        self._clear_frame_dedupe()

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

        output_text = buffer or self._conversation.assistant_transcript
        await finish_response(
            self._conversation,
            self._emit_fn,
            status=status,
            output_text=output_text,
        )
        self._bot_transcript_from_tts = False
        self._llm_text_buffer = ""
        self._llm_text_generation = 0
        self._clear_frame_dedupe()
