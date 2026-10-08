# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A Pipecat processor that is the caller-facing frontend: a remote realtime model over WebSocket.

Caller audio goes in as ``input_audio_buffer.append`` and the model's speech comes out as the frames a TTS service
pushes (audio, word text, start/stop), so the live pipeline helpers (input clock, transcript events, output audio)
work unchanged. The model's turn-taking, interruption and transcription are the remote server's; this processor
only translates them.

Delegation is a function call: the model calls ``delegate``, :class:`RealtimeFrontendProcessor` hands the request to
its ``on_delegate`` hook and answers the call at once so the model keeps talking. Backend results and application
updates come back through :meth:`inject`.
"""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from loguru import logger
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    InputAudioRawFrame,
    StartFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.utils.time import time_now_iso8601
from websockets.asyncio.client import connect

from examples.frontend_backend_live.prompts.realtime import DELEGATE_TOOL

SAMPLE_RATE = 24000
READY_TIMEOUT_SECONDS = 20
IDLE_TIMEOUT_SECONDS = 10
# The server reports these when a request raced its own state; neither is a failure of the call.
CANCEL_RACE_CODE = "response_cancel_not_active"
BUSY_CODE = "conversation_already_has_active_response"

AUDIO_DELTA = {"response.output_audio.delta", "response.audio.delta"}
AUDIO_DONE = {"response.output_audio.done", "response.audio.done"}
TRANSCRIPT_DELTA = {"response.output_audio_transcript.delta", "response.audio_transcript.delta"}
TRANSCRIPT_DONE = {"response.output_audio_transcript.done", "response.audio_transcript.done"}

Delegate = Callable[[str], Awaitable[str]]
Hear = Callable[[str], None]


@dataclass
class RealtimeSettings:
    """Everything needed to open and configure one realtime session."""

    url: str
    api_key: str
    model: str
    instructions: str
    voice: str = "marin"
    history: list[dict] = field(default_factory=list)
    headers: dict[str, str] = field(default_factory=dict)
    tools: list[dict] = field(default_factory=lambda: [DELEGATE_TOOL])
    transcription_model: str = "gpt-4o-mini-transcribe"
    # Deep-merged into the ``session`` object, to adapt to servers that differ from the default shape.
    session_overrides: dict = field(default_factory=dict)

    def session(self) -> dict:
        """The ``session`` object of the opening ``session.update``."""
        session = {
            "type": "realtime",
            "model": self.model,
            "instructions": self.instructions,
            "output_modalities": ["audio"],
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
                    "transcription": {"model": self.transcription_model},
                    "turn_detection": {"type": "server_vad", "create_response": True, "interrupt_response": True},
                },
                "output": {"format": {"type": "audio/pcm", "rate": SAMPLE_RATE}, "voice": self.voice},
            },
            "tools": self.tools,
            "tool_choice": "auto",
        }
        return merge(session, self.session_overrides)


def merge(base: dict, patch: dict) -> dict:
    """Return ``base`` with ``patch`` merged in; nested mappings merge, everything else is replaced."""
    merged = dict(base)
    for key, value in patch.items():
        merged[key] = (
            merge(merged[key], value) if isinstance(value, dict) and isinstance(merged.get(key), dict) else value
        )
    return merged


def message_item(role: str, text: str) -> dict:
    """A ``conversation.item.create`` event carrying one text message."""
    kind = "output_text" if role == "assistant" else "input_text"
    return {
        "type": "conversation.item.create",
        "item": {"type": "message", "role": role, "content": [{"type": kind, "text": text}]},
    }


class RealtimeFrontendProcessor(FrameProcessor):
    """Connects the pipeline to a remote realtime model and translates between the two."""

    def __init__(
        self,
        settings: RealtimeSettings,
        *,
        on_delegate: Delegate,
        on_user_text: Hear = lambda text: None,
        on_assistant_text: Hear = lambda text: None,
        on_closed: Callable[[], Awaitable[None]] | None = None,
        **kwargs,
    ):
        """Create the processor; the connection opens when the pipeline starts.

        Args:
            settings: The session settings.
            on_delegate: Awaited with the ``request`` of each ``delegate`` call; returns the function output.
            on_user_text: Called with each finished caller utterance.
            on_assistant_text: Called with each finished spoken reply.
            on_closed: Awaited when the remote connection ends unexpectedly.
            **kwargs: Passed to :class:`FrameProcessor`.
        """
        super().__init__(**kwargs)
        self.settings, self.on_delegate = settings, on_delegate
        self.on_user_text, self.on_assistant_text, self.on_closed = on_user_text, on_assistant_text, on_closed
        self.ready = asyncio.Event()
        self.failure: str | None = None
        self._ws = None
        self._receiver: asyncio.Task | None = None
        self._tasks: set[asyncio.Task] = set()
        self._closing = False
        self._response_active = False
        self._response_idle = asyncio.Event()
        self._response_idle.set()
        self._cancelled: set[str] = set()
        self._response_id = ""
        self._tts_open = False
        self._words = ""
        self._spoken = ""
        self._calls: list[dict] = []
        self._user_idle = asyncio.Event()
        self._user_idle.set()
        self._send_lock = asyncio.Lock()
        # Response requests go one at a time, and a request the server refused as busy is repeated when it is idle.
        self._respond_lock = asyncio.Lock()
        self._last_request: dict | None = None
        self._repeat_request = False

    # -------------------------------------------------------------------------------- frames
    async def process_frame(self, frame, direction: FrameDirection) -> None:
        """Forward caller audio to the model; connect on start and disconnect on end."""
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame):
            await self.push_frame(frame, direction)
            await self._connect()
            return
        if isinstance(frame, (EndFrame, CancelFrame)):
            await self._disconnect()
        elif isinstance(frame, InputAudioRawFrame):
            if self.ready.is_set() and frame.sample_rate == SAMPLE_RATE:
                await self.send({"type": "input_audio_buffer.append", "audio": base64.b64encode(frame.audio).decode()})
            return
        await self.push_frame(frame, direction)

    # ----------------------------------------------------------------------------- connection
    async def _connect(self) -> None:
        headers = {"Authorization": f"Bearer {self.settings.api_key}", **self.settings.headers}
        try:
            self._ws = await connect(self.settings.url, additional_headers=headers, max_size=None)
        except Exception as exc:
            self.failure = f"Cannot connect to the realtime model: {type(exc).__name__}: {exc}"
            self.ready.set()
            return
        self._receiver = asyncio.create_task(self._receive(), name="realtime-frontend")
        await self.send({"type": "session.update", "session": self.settings.session()})

    async def _disconnect(self) -> None:
        self._closing = True
        for task in tuple(self._tasks):
            task.cancel()
        if self._ws is not None:
            await self._ws.close()
        if self._receiver is not None:
            self._receiver.cancel()
            await asyncio.gather(self._receiver, return_exceptions=True)
            self._receiver = None

    async def send(self, message: dict) -> None:
        """Send one client event; a send on a closed connection is dropped."""
        if self._ws is None or self._closing:
            return
        async with self._send_lock:
            try:
                await self._ws.send(json.dumps(message))
            except Exception as exc:
                if not self._closing:
                    logger.warning(f"Realtime send failed: {type(exc).__name__}: {exc}")

    def _spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _receive(self) -> None:
        try:
            async for raw in self._ws:
                try:
                    await self._handle(json.loads(raw))
                except Exception as exc:
                    logger.exception(f"Realtime event failed: {exc}")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not self._closing:
                logger.warning(f"Realtime connection lost: {type(exc).__name__}: {exc}")
        if not self._closing:
            self.failure = self.failure or "The realtime connection closed."
            self.ready.set()
            if self.on_closed:
                await self.on_closed()

    # ------------------------------------------------------------------------- server events
    async def _handle(self, event: dict) -> None:
        kind = event.get("type", "")
        if kind == "session.updated":
            await self._on_session_updated()
        elif kind == "input_audio_buffer.speech_started":
            await self._on_speech_started()
        elif kind == "input_audio_buffer.speech_stopped":
            self._user_idle.set()
            await self.push_frame(UserStoppedSpeakingFrame())
        elif kind == "conversation.item.input_audio_transcription.completed":
            text = (event.get("transcript") or "").strip()
            if text:
                self.on_user_text(text)
                await self.push_frame(
                    TranscriptionFrame(text=text, user_id="caller", timestamp=time_now_iso8601(), finalized=True)
                )
        elif kind == "response.created":
            self._response_id = (event.get("response") or {}).get("id", "")
            self._set_response_active(True)
        elif kind in AUDIO_DELTA:
            await self._on_audio(event)
        elif kind in TRANSCRIPT_DELTA:
            await self._on_words(event)
        elif kind in TRANSCRIPT_DONE:
            await self._flush_words()
            if self._spoken.strip():
                self.on_assistant_text(self._spoken.strip())
            self._spoken = ""
        elif kind in AUDIO_DONE:
            await self._close_speech()
        elif kind == "response.output_item.done":
            item = event.get("item") or {}
            if item.get("type") == "function_call":
                self._calls.append(item)
        elif kind == "response.done":
            await self._on_response_done(event)
        elif kind == "error":
            await self._on_error(event.get("error") or {})

    async def _on_session_updated(self) -> None:
        if self.ready.is_set():
            return
        for item in self.settings.history:
            await self.send(message_item(item["role"], item["text"]))
        self.ready.set()

    async def _on_speech_started(self) -> None:
        self._user_idle.clear()
        if self._response_active or self._tts_open:
            # The server cancels its own response when it hears the caller; drop what it already sent.
            self._cancelled.add(self._response_id)
            await self._close_speech()
            await self.broadcast_interruption()
        await self.push_frame(UserStartedSpeakingFrame())

    def _drop(self, event: dict) -> bool:
        response_id = event.get("response_id") or self._response_id
        return response_id in self._cancelled

    async def _on_audio(self, event: dict) -> None:
        if self._drop(event):
            return
        if not self._tts_open:
            self._tts_open = True
            await self.push_frame(TTSStartedFrame())
        audio = base64.b64decode(event.get("delta") or "")
        if audio:
            await self.push_frame(TTSAudioRawFrame(audio=audio, sample_rate=SAMPLE_RATE, num_channels=1))

    async def _on_words(self, event: dict) -> None:
        """Re-assemble the model's transcript deltas into whole words (the output events assume word units)."""
        if self._drop(event):
            return
        delta = event.get("delta") or ""
        self._spoken += delta
        self._words += delta
        *complete, self._words = self._words.split(" ")
        for word in complete:
            if word.strip():
                await self.push_frame(TTSTextFrame(text=word, aggregated_by="word"))

    async def _flush_words(self) -> None:
        if self._words.strip():
            await self.push_frame(TTSTextFrame(text=self._words.strip(), aggregated_by="word"))
        self._words = ""

    async def _close_speech(self) -> None:
        await self._flush_words()
        if self._tts_open:
            self._tts_open = False
            await self.push_frame(TTSStoppedFrame())

    async def _on_response_done(self, event: dict) -> None:
        await self._close_speech()
        self._set_response_active(False)
        calls, self._calls = self._calls, []
        if calls:
            self._spawn(self._answer_calls(calls))
        elif self._repeat_request and self._last_request is not None:
            self._repeat_request = False
            self._spawn(self.create_response((self._last_request.get("response") or {}).get("instructions")))

    async def _on_error(self, error: dict) -> None:
        code = error.get("code")
        if code == BUSY_CODE:
            # The model was already replying (to the caller, say) when we asked. Ask again once it is done, so the
            # answer we wanted spoken is not lost.
            self._repeat_request = True
            return
        if code == CANCEL_RACE_CODE:
            return
        message = error.get("message") or "The realtime model reported an error."
        if not self.ready.is_set():
            self.failure = message
            self.ready.set()
            return
        await self.push_error(error_msg=str(message)[:500])

    # --------------------------------------------------------------------------- delegation
    async def _answer_calls(self, calls: list[dict]) -> None:
        for call in calls:
            output = await self._run_call(call)
            await self.send(
                {
                    "type": "conversation.item.create",
                    "item": {"type": "function_call_output", "call_id": call.get("call_id"), "output": output},
                }
            )
        await self.create_response()

    async def _run_call(self, call: dict) -> str:
        if call.get("name") != "delegate":
            return json.dumps({"error": f"Unknown function {call.get('name')!r}."})
        try:
            request = str(json.loads(call.get("arguments") or "{}").get("request", "")).strip()
        except (ValueError, AttributeError):
            request = ""
        if not request:
            return json.dumps({"error": "delegate needs a non-empty request."})
        return await self.on_delegate(request)

    # -------------------------------------------------------------------------- the commands
    def _set_response_active(self, active: bool) -> None:
        self._response_active = active
        if active:
            self._response_idle.clear()
        else:
            self._response_idle.set()

    async def create_response(self, instructions: str | None = None) -> None:
        """Ask the model to speak, once it is not mid-response and the caller is not mid-sentence.

        Requests are serialized: when several wait for the same reply to end, they go out one after another, each
        after the reply the previous one started.
        """
        async with self._respond_lock:
            for gate in (self._response_idle, self._user_idle):
                try:
                    await asyncio.wait_for(gate.wait(), IDLE_TIMEOUT_SECONDS)
                except TimeoutError:
                    logger.warning("Realtime model still busy; asking for a response anyway")
            self._set_response_active(True)
            message: dict = {"type": "response.create"}
            if instructions:
                message["response"] = {"instructions": instructions}
            self._last_request = message
            await self.send(message)

    async def inject(self, text: str, *, respond: bool, instructions: str | None = None, interrupt: bool = False):
        """Add a system message to the conversation, optionally interrupting speech and asking for a reply."""
        if interrupt and (self._response_active or self._tts_open):
            self._cancelled.add(self._response_id)
            await self.send({"type": "response.cancel"})
            await self._close_speech()
            await self.broadcast_interruption()
            self._set_response_active(False)
        await self.send(message_item("system", text))
        if respond:
            await self.create_response(instructions)
