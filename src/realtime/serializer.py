# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Pipecat FrameSerializer for OpenAI Realtime–shaped JSON WebSocket frames."""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import Callable
from typing import Any

from loguru import logger
from pipecat.frames.frames import (
    Frame,
    FunctionCallResultFrame,
    InputAudioRawFrame,
    InterruptionFrame,
    LLMMessagesAppendFrame,
    LLMRunFrame,
    OutputAudioRawFrame,
    StartFrame,
    TTSUpdateSettingsFrame,
)
from pipecat.serializers.base_serializer import FrameSerializer

from realtime.audio import (
    DEFAULT_CLIENT_PCM_RATE,
    MAX_PENDING_INPUT_BYTES,
    PIPELINE_PCM_RATE,
    AudioResampler,
    decode_base64_audio,
    encode_base64_audio,
    extract_client_input_format_type,
    extract_client_output_pcm_rate,
    extract_client_pcm_rate,
    validate_session_audio_config,
)
from realtime.conversation import ConversationState, new_item_id
from realtime.events import (
    CLIENT_AUDIO_APPEND,
    CLIENT_AUDIO_CLEAR,
    CLIENT_AUDIO_COMMIT,
    CLIENT_ITEM_CREATE,
    CLIENT_ITEM_TRUNCATE,
    CLIENT_RESPONSE_CANCEL,
    CLIENT_RESPONSE_CREATE,
    CLIENT_SESSION_UPDATE,
    SERVER_AUDIO_CLEARED,
    SERVER_AUDIO_COMMITTED,
    SERVER_ITEM_CREATED,
    SERVER_OUTPUT_AUDIO_DELTA,
    SERVER_SESSION_UPDATED,
    EmitFn,
    emit_with_aliases,
    error_event,
    server_event,
    with_beta_aliases,
)
from realtime.lifecycle import announce_response, finish_response
from realtime.session import (
    live_session_patch,
    merge_session_patch,
    nvidia_public_view,
    unsupported_live_session_fields,
)
from realtime.voice import resolve_realtime_tts_voice

CancelHook = Callable[[], str]

# Base64 is ~4/3 of raw bytes; reject oversized appends before decode/resample.
_MAX_APPEND_B64_CHARS = (MAX_PENDING_INPUT_BYTES * 4) // 3 + 64


class RealtimeFrameSerializer(FrameSerializer):
    """Convert Realtime JSON events ↔ Pipecat frames on ``FastAPIWebsocketTransport``.

    Extra server events go through :meth:`set_emit` (Pipecat returns one
    ``serialize()`` value per outbound frame). Transcript lifecycle is owned by
    :class:`realtime.observer.RealtimeLifecycleObserver`.
    """

    def __init__(
        self,
        *,
        session_view: dict[str, Any] | None = None,
        conversation: ConversationState | None = None,
        params: FrameSerializer.InputParams | None = None,
    ) -> None:
        """Create a serializer bound to the current Realtime session view."""
        super().__init__(params or FrameSerializer.InputParams())
        self._session_view: dict[str, Any] = dict(session_view or {})
        self._conversation = conversation or ConversationState()
        self._resampler = AudioResampler()
        self._pipeline_rate = PIPELINE_PCM_RATE
        self._client_in_rate = extract_client_pcm_rate(self._session_view)
        self._client_out_rate = extract_client_output_pcm_rate(self._session_view)
        self._emit: EmitFn | None = None
        self._on_response_cancel: CancelHook | None = None
        # When turn_detection is null (push-to-talk), buffer PCM until commit.
        self._pending_input_pcm = bytearray()
        self._bytes_since_commit = 0

    @property
    def conversation(self) -> ConversationState:
        """Shared conversation / response id state for the observer."""
        return self._conversation

    @property
    def emit(self) -> EmitFn | None:
        """Registered Realtime event emit callback, if any."""
        return self._emit

    def set_emit(self, emit: EmitFn) -> None:
        """Register async callback used to send Realtime JSON events on the WS."""
        self._emit = emit

    def set_on_response_cancel(self, hook: CancelHook | None) -> None:
        """Register a sync hook that drains observer state on ``response.cancel``."""
        self._on_response_cancel = hook

    def update_session_view(self, session_view: dict[str, Any]) -> None:
        """Refresh rates / voice from the latest Realtime session object."""
        prev_in = self._client_in_rate
        prev_out = self._client_out_rate
        self._session_view = dict(session_view or {})
        self._client_in_rate = extract_client_pcm_rate(self._session_view)
        self._client_out_rate = extract_client_output_pcm_rate(self._session_view)
        # SOXR stream resamplers cannot switch rates after the first chunk.
        if prev_in != self._client_in_rate or prev_out != self._client_out_rate:
            self._resampler.reset()
            logger.info(
                f"Realtime PCM rates changed in={prev_in}->{self._client_in_rate} "
                f"out={prev_out}->{self._client_out_rate}; resampler reset"
            )

    async def _emit_event(self, event: dict[str, Any]) -> None:
        if self._emit is None:
            logger.warning(f"Realtime emit not configured; dropping event type={event.get('type')}")
            return
        await emit_with_aliases(self._emit, event)

    async def setup(self, frame: StartFrame) -> None:
        """Capture pipeline sample rate from StartFrame when provided."""
        await super().setup(frame)
        if getattr(frame, "audio_in_sample_rate", None):
            self._pipeline_rate = int(frame.audio_in_sample_rate)
        elif getattr(frame, "audio_out_sample_rate", None):
            self._pipeline_rate = int(frame.audio_out_sample_rate)

    async def serialize(self, frame: Frame) -> str | bytes | None:
        """Serialize outbound audio frames to Realtime JSON text."""
        if self.should_ignore_frame(frame):
            return None
        if isinstance(frame, OutputAudioRawFrame):
            return await self._serialize_output_audio(frame)
        return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        """Deserialize an inbound WebSocket message to a Pipecat frame."""
        if isinstance(data, bytes):
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                await self._emit_event(
                    error_event(
                        "Binary frames are not supported; send JSON text",
                        code="invalid_frame",
                    )
                )
                return None
        else:
            text = data

        try:
            message = json.loads(text)
        except json.JSONDecodeError:
            await self._emit_event(error_event("Invalid JSON", code="invalid_json"))
            return None

        if not isinstance(message, dict):
            await self._emit_event(error_event("Event must be a JSON object", code="invalid_event"))
            return None

        event_type = message.get("type")
        client_event_id = message.get("event_id")
        echo_id = client_event_id if isinstance(client_event_id, str) else None

        if event_type == CLIENT_AUDIO_APPEND:
            return await self._deserialize_append(message, echo_id)

        if event_type == CLIENT_AUDIO_COMMIT:
            return await self._deserialize_commit(echo_id)

        if event_type == CLIENT_AUDIO_CLEAR:
            self._pending_input_pcm.clear()
            self._bytes_since_commit = 0
            await self._emit_event(server_event(SERVER_AUDIO_CLEARED))
            return None

        if event_type == CLIENT_SESSION_UPDATE:
            return await self._deserialize_session_update(message, echo_id)

        if event_type == CLIENT_RESPONSE_CREATE:
            # Welcome gate: reject until first assistant response.done (RTVI parity).
            if not self._conversation.assistant_has_responded:
                logger.info("Rejecting response.create during welcome-message window")
                await self._emit_event(
                    error_event(
                        "response.create is not accepted before the first assistant response",
                        code="response_create_rejected_pre_intro",
                        event_id=echo_id,
                        param="type",
                    )
                )
                return None
            # Empty response:{} is accepted; non-empty overrides are not.
            response_override = message.get("response") if "response" in message else None
            if response_override is not None and (not isinstance(response_override, dict) or response_override):
                await self._emit_event(
                    error_event(
                        "response.create.response overrides are not supported in v1; "
                        "omit the response field (or send {}) and use session-level config",
                        code="unsupported_response_override",
                        event_id=echo_id,
                        param="response",
                    )
                )
                return None
            return LLMRunFrame()

        if event_type == CLIENT_RESPONSE_CANCEL:
            response_id = message.get("response_id")
            active_id = self._conversation.response_id
            # Idle cancel is a no-op (clients often send this on speech_started).
            if self._conversation.response_status != "in_progress" or not active_id:
                return None
            if isinstance(response_id, str) and response_id and response_id != active_id:
                await self._emit_event(
                    error_event(
                        "response_id does not match the active response",
                        code="invalid_value",
                        event_id=echo_id,
                        param="response_id",
                    )
                )
                return None
            await self._emit_cancelled_response()
            return InterruptionFrame()

        if event_type == CLIENT_ITEM_TRUNCATE:
            return await self._deserialize_truncate(message, echo_id)

        if event_type == CLIENT_ITEM_CREATE:
            return await self._deserialize_item_create(message, echo_id)

        await self._emit_event(
            error_event(
                f"Event type '{event_type}' is not supported in v1",
                code="unsupported_event",
                event_id=echo_id,
                param="type",
            )
        )
        return None

    def _manual_turn_detection(self) -> bool:
        """True when the client disabled server VAD (push-to-talk / commit-driven)."""
        audio = self._session_view.get("audio")
        if isinstance(audio, dict):
            inp = audio.get("input")
            if isinstance(inp, dict) and "turn_detection" in inp:
                return inp.get("turn_detection") is None
        if "turn_detection" in self._session_view:
            return self._session_view.get("turn_detection") is None
        return False

    async def _deserialize_session_update(self, message: dict[str, Any], echo_id: str | None) -> Frame | None:
        session_patch = message.get("session")
        if not isinstance(session_patch, dict):
            await self._emit_event(
                error_event(
                    "session.update requires a session object",
                    code="invalid_session",
                    event_id=echo_id,
                    param="session",
                )
            )
            return None

        # Live: voice/turn_detection/audio format only. Unchanged agent fields
        # (full-session client echoes) are ignored; real agent changes are rejected.
        unsupported = unsupported_live_session_fields(session_patch, self._session_view)
        if unsupported:
            fields = ", ".join(unsupported)
            await self._emit_event(
                error_event(
                    f"Post-handoff session.update cannot apply [{fields}]; "
                    "reconnect to change instructions/tools/temperature/nvidia. "
                    "Live: voice, turn_detection, audio format/rate only",
                    code="unsupported_live_session_update",
                    event_id=echo_id,
                    param="session",
                )
            )
            return None

        try:
            validate_session_audio_config(session_patch)
        except ValueError as exc:
            await self._emit_event(error_event(str(exc), code="invalid_session", event_id=echo_id, param="session"))
            return None

        patch = live_session_patch(session_patch)

        voice = ""
        if isinstance(patch.get("voice"), str):
            voice = patch["voice"].strip()
        audio = patch.get("audio")
        if isinstance(audio, dict):
            output = audio.get("output")
            if isinstance(output, dict) and isinstance(output.get("voice"), str):
                voice = output["voice"].strip()
        if voice:
            # Soft catalog check (same list path as RTVI UI); unknown → default.
            nvidia = self._session_view.get("nvidia") if isinstance(self._session_view.get("nvidia"), dict) else {}
            voice_config = {
                "tts_voice_id": voice,
                "tts_server": nvidia.get("tts_server", ""),
                "tts_function_id": nvidia.get("tts_function_id", ""),
                "tts_model": nvidia.get("tts_model", ""),
            }
            resolved = await asyncio.to_thread(resolve_realtime_tts_voice, voice_config, voice_was_set=True) or voice
            voice = resolved
            if isinstance(patch.get("voice"), str):
                patch["voice"] = voice
            audio_patch = patch.get("audio")
            if isinstance(audio_patch, dict):
                output_patch = audio_patch.get("output")
                if isinstance(output_patch, dict) and "voice" in output_patch:
                    output_patch["voice"] = voice

        self._session_view = merge_session_patch(self._session_view, patch)
        if voice:
            audio = dict(self._session_view.get("audio") or {})
            output = dict(audio.get("output") or {})
            output["voice"] = voice
            audio["output"] = output
            self._session_view["audio"] = audio
            self._session_view["voice"] = voice
        nvidia = self._session_view.get("nvidia")
        if isinstance(nvidia, dict):
            self._session_view["nvidia"] = nvidia_public_view(nvidia)
        self.update_session_view(self._session_view)
        await self._emit_event(server_event(SERVER_SESSION_UPDATED, session=dict(self._session_view)))

        if voice:
            return TTSUpdateSettingsFrame(settings={"voice": voice})
        return None

    async def _deserialize_truncate(self, message: dict[str, Any], echo_id: str | None) -> Frame | None:
        item_id = message.get("item_id")
        audio_end_ms = message.get("audio_end_ms")
        if not isinstance(item_id, str) or not item_id:
            await self._emit_event(
                error_event(
                    "conversation.item.truncate requires item_id",
                    code="invalid_truncate",
                    event_id=echo_id,
                    param="item_id",
                )
            )
            return None
        if isinstance(audio_end_ms, bool) or not isinstance(audio_end_ms, int) or audio_end_ms < 0:
            await self._emit_event(
                error_event(
                    "conversation.item.truncate requires non-negative integer audio_end_ms",
                    code="invalid_truncate",
                    event_id=echo_id,
                    param="audio_end_ms",
                )
            )
            return None

        # Active truncate → barge-in; never emit conversation.item.truncated. Idle → no-op.
        known = item_id == self._conversation.assistant_item_id or item_id in self._conversation.item_transcripts
        if not known:
            await self._emit_event(
                error_event(
                    f"unknown item_id for truncate: {item_id}",
                    code="invalid_truncate",
                    event_id=echo_id,
                    param="item_id",
                )
            )
            return None

        if self._conversation.response_status != "in_progress":
            return None

        await self._emit_cancelled_response(status_if_active="cancelled")
        return InterruptionFrame()

    async def _deserialize_commit(self, echo_id: str | None) -> Frame | None:
        if self._bytes_since_commit <= 0 and not self._pending_input_pcm:
            await self._emit_event(
                error_event(
                    "input_audio_buffer is empty; append audio before commit",
                    code="input_audio_buffer_commit_empty",
                    event_id=echo_id,
                )
            )
            return None

        item_id = self._conversation.begin_user_item()
        await self._emit_event(
            server_event(
                SERVER_AUDIO_COMMITTED,
                item_id=item_id,
                previous_item_id=None,
            )
        )
        self._bytes_since_commit = 0
        if not self._manual_turn_detection():
            # Server VAD path: audio already streamed; commit is an ack only.
            return None
        pcm = bytes(self._pending_input_pcm)
        self._pending_input_pcm.clear()
        return InputAudioRawFrame(
            audio=pcm,
            sample_rate=self._pipeline_rate,
            num_channels=1,
        )

    async def _deserialize_append(self, message: dict[str, Any], echo_id: str | None) -> Frame | None:
        fmt = extract_client_input_format_type(self._session_view)
        if fmt != "audio/pcm":
            await self._emit_event(
                error_event(
                    f"v1 supports audio/pcm only (got {fmt})",
                    code="unsupported_audio_format",
                    event_id=echo_id,
                    param="session.audio.input.format.type",
                )
            )
            return None

        audio_b64 = message.get("audio")
        if not isinstance(audio_b64, str):
            await self._emit_event(
                error_event("audio must be a base64 string", code="invalid_audio", event_id=echo_id, param="audio")
            )
            return None
        if len(audio_b64) > _MAX_APPEND_B64_CHARS:
            await self._emit_event(
                error_event(
                    "input_audio_buffer.append payload too large",
                    code="input_buffer_overflow",
                    event_id=echo_id,
                    param="audio",
                )
            )
            return None
        try:
            raw = decode_base64_audio(audio_b64)
            pcm = await self._resampler.to_pipeline(
                raw,
                self._client_in_rate or DEFAULT_CLIENT_PCM_RATE,
                pipeline_rate=self._pipeline_rate,
            )
        except ValueError as exc:
            await self._emit_event(error_event(str(exc), code="invalid_audio", event_id=echo_id, param="audio"))
            return None

        if not pcm:
            return None

        if len(pcm) > MAX_PENDING_INPUT_BYTES:
            await self._emit_event(
                error_event(
                    "input_audio_buffer.append decoded audio exceeds max size",
                    code="input_buffer_overflow",
                    event_id=echo_id,
                    param="audio",
                )
            )
            return None

        if self._manual_turn_detection():
            if len(self._pending_input_pcm) + len(pcm) > MAX_PENDING_INPUT_BYTES:
                await self._emit_event(
                    error_event(
                        "input_audio_buffer exceeded max pending size; commit or clear before appending more",
                        code="input_buffer_overflow",
                        event_id=echo_id,
                        param="audio",
                    )
                )
                return None
            self._pending_input_pcm.extend(pcm)
            self._bytes_since_commit += len(pcm)
            return None

        self._bytes_since_commit += len(pcm)
        return InputAudioRawFrame(
            audio=pcm,
            sample_rate=self._pipeline_rate,
            num_channels=1,
        )

    async def _deserialize_item_create(self, message: dict[str, Any], echo_id: str | None) -> Frame | None:
        item = message.get("item")
        if not isinstance(item, dict):
            await self._emit_event(
                error_event(
                    "conversation.item.create requires an item object",
                    code="invalid_item",
                    event_id=echo_id,
                    param="item",
                )
            )
            return None

        item_type = item.get("type") or "message"
        if item_type == "function_call_output":
            return await self._deserialize_function_call_output(item, echo_id)

        role = item.get("role") or "user"
        if item_type != "message" or role != "user":
            await self._emit_event(
                error_event(
                    "v1 only supports conversation.item.create for user text messages or function_call_output",
                    code="unsupported_item",
                    event_id=echo_id,
                    param="item",
                )
            )
            return None

        text = _extract_item_text(item)
        if text is None:
            await self._emit_event(
                error_event(
                    "conversation.item.create requires text content",
                    code="invalid_item",
                    event_id=echo_id,
                    param="item.content",
                )
            )
            return None

        # Drop client text until the first assistant response when welcome is enabled.
        # Audio append/commit is unaffected; paired response.create uses the same gate.
        if not self._conversation.assistant_has_responded:
            logger.info(f"Dropping client user text before first assistant response (len={len(text)})")
            await self._emit_event(
                error_event(
                    "Client text is not accepted before the first assistant response",
                    code="item_rejected_pre_intro",
                    event_id=echo_id,
                    param="item",
                )
            )
            return None

        item_id = item.get("id") if isinstance(item.get("id"), str) else new_item_id()
        await self._emit_event(
            server_event(
                SERVER_ITEM_CREATED,
                previous_item_id=None,
                item={
                    "id": item_id,
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}],
                },
            )
        )
        # Match OpenAI: create does not auto-run; client sends response.create.
        return LLMMessagesAppendFrame(
            messages=[{"role": "user", "content": text}],
            run_llm=False,
        )

    async def _deserialize_function_call_output(self, item: dict[str, Any], echo_id: str | None) -> Frame | None:
        call_id = item.get("call_id")
        output = item.get("output")
        if not isinstance(call_id, str) or not call_id.strip():
            await self._emit_event(
                error_event(
                    "function_call_output requires call_id",
                    code="invalid_item",
                    event_id=echo_id,
                    param="item.call_id",
                )
            )
            return None
        if output is None:
            await self._emit_event(
                error_event(
                    "function_call_output requires output",
                    code="invalid_item",
                    event_id=echo_id,
                    param="item.output",
                )
            )
            return None

        meta = self._conversation.pop_function_call(call_id)
        if meta is None:
            await self._emit_event(
                error_event(
                    f"unknown call_id for function_call_output: {call_id}",
                    code="invalid_item",
                    event_id=echo_id,
                    param="item.call_id",
                )
            )
            return None

        name = str(meta.get("name") or "")
        arguments = meta.get("arguments") if "arguments" in meta else {}
        result: Any = output
        if isinstance(output, str):
            with contextlib.suppress(json.JSONDecodeError, TypeError):
                result = json.loads(output)

        item_id = item.get("id") if isinstance(item.get("id"), str) else new_item_id()
        await self._emit_event(
            server_event(
                SERVER_ITEM_CREATED,
                previous_item_id=None,
                item={
                    "id": item_id,
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output if isinstance(output, str) else json.dumps(output),
                },
            )
        )
        # Do not auto-run LLM — client follows with response.create (OpenAI semantics).
        return FunctionCallResultFrame(
            function_name=name or "unknown",
            tool_call_id=call_id,
            arguments=arguments,
            result=result,
            run_llm=False,
        )

    async def _serialize_output_audio(self, frame: OutputAudioRawFrame) -> str | None:
        pcm = frame.audio or b""
        if not pcm:
            return None

        try:
            client_pcm = await self._resampler.from_pipeline(
                pcm,
                self._client_out_rate or DEFAULT_CLIENT_PCM_RATE,
                pipeline_rate=self._pipeline_rate,
            )
        except ValueError as exc:
            logger.warning(f"Realtime output resample failed: {exc}")
            return None

        if self._emit is None:
            logger.warning("Realtime emit not configured; dropping output audio")
            return None

        # Prefer an in-progress response. Early PCM may arrive before the observer
        # announces — allow opening. After response.done, stream trailing PCM on the
        # closed item only (do not create an empty assistant row).
        if self._conversation.response_status == "in_progress":
            response_id, _created = await announce_response(self._conversation, self._emit)
            item_id = self._conversation.assistant_item_id
        elif self._conversation.closed_response_id and self._conversation.closed_item_id:
            response_id = self._conversation.closed_response_id
            item_id = self._conversation.closed_item_id
        else:
            response_id, _created = await announce_response(self._conversation, self._emit)
            item_id = self._conversation.assistant_item_id

        event = server_event(
            SERVER_OUTPUT_AUDIO_DELTA,
            response_id=response_id,
            item_id=item_id,
            output_index=0,
            content_index=0,
            delta=encode_base64_audio(client_pcm),
        )
        # serialize() returns one WS frame; emit any pre-GA alias separately.
        for payload in with_beta_aliases(event)[1:]:
            await self._emit(payload)
        return json.dumps(event)

    async def _emit_cancelled_response(self, *, status_if_active: str = "cancelled") -> None:
        if self._emit is None:
            return
        # Drain observer LLM text before finish so late TTSStopped cannot revive the turn.
        buffered = self._on_response_cancel() if self._on_response_cancel is not None else ""
        await finish_response(
            self._conversation,
            self._emit,
            status=status_if_active,
            output_text=buffered or self._conversation.assistant_transcript,
        )


def _extract_item_text(item: dict[str, Any]) -> str | None:
    content = item.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return None
    parts: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") in {"input_text", "text"} and isinstance(part.get("text"), str):
            parts.append(part["text"])
    if not parts:
        return None
    return "".join(parts)
