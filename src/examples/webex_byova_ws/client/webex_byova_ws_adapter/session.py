"""Per-socket Cisco conversation state machine."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from enum import Enum, auto
from typing import Any

from aiohttp import WSMsgType, web

from webex_byova_ws_adapter.audio import CallerAudioConverter, WebexAudioChunker, encode_audio
from webex_byova_ws_adapter.bridge import NemotronBridge
from webex_byova_ws_adapter.completion import ReplyLifecycle
from webex_byova_ws_adapter.config import AdapterConfig
from webex_byova_ws_adapter.protocol import (
    OutboundSequence,
    ProtocolError,
    error,
    parse_request,
    parse_text,
    pong,
    response,
)

LOGGER = logging.getLogger(__name__)


class SessionPhase(Enum):
    """Canonical phases for one persistent Cisco WebSocket."""

    WAITING_SESSION_START = auto()
    WELCOME_REPLY = auto()
    LISTENING = auto()
    CALLER_ACTIVE = auto()
    ASSISTANT_REPLY = auto()
    TERMINAL = auto()


class Conversation:
    """Handle one Webex WebSocket and one stable conversation ID."""

    def __init__(self, config: AdapterConfig, websocket: web.WebSocketResponse) -> None:
        """Create isolated state for one logical Webex conversation."""
        self.config = config
        self.websocket = websocket
        self.conversation_id = ""
        self.last_inbound_seq = 0
        self.sequence = OutboundSequence()
        self.bridge: NemotronBridge | None = None
        self.bridge_pump: asyncio.Task[None] | None = None
        self.reply_timeout_task: asyncio.Task[None] | None = None
        self.send_lock = asyncio.Lock()
        self.attach_lock = asyncio.Lock()
        self.caller_audio = CallerAudioConverter()
        self.bot_audio = WebexAudioChunker()
        self.reply = ReplyLifecycle()
        self.phase = SessionPhase.WAITING_SESSION_START
        self.pending_transcript = ""
        self.closed = False
        self.closed_event = asyncio.Event()
        self.reply_complete = asyncio.Event()
        self.reply_complete.set()
        self.idle_close_task: asyncio.Task[None] | None = None
        self.audio_chunks_sent = 0
        self.audio_bytes_sent = 0

    async def attach(
        self,
        websocket: web.WebSocketResponse,
        initial_envelope: dict[str, Any],
    ) -> None:
        """Attach a Cisco continuation socket to this logical conversation."""
        async with self.attach_lock:
            if self.closed:
                raise ProtocolError("conversation is already closed")
            if self.idle_close_task:
                self.idle_close_task.cancel()
                self.idle_close_task = None
            try:
                await asyncio.wait_for(
                    self.reply_complete.wait(),
                    self.config.first_audio_timeout_seconds + self.config.completion_timeout_seconds,
                )
            except TimeoutError:
                await self.close()
                raise
            async with self.send_lock:
                previous = self.websocket
                self.websocket = websocket
                self.last_inbound_seq = 0
                self.sequence = OutboundSequence()
                if previous is not websocket and not previous.closed:
                    await previous.close(code=1000, message=b"conversation continued")
            await self._process_envelope(initial_envelope)
            LOGGER.info(
                "Attached Cisco continuation socket conversation_id=%s phase=%s",
                self.conversation_id,
                self.phase.name,
            )

    async def run(
        self,
        initial_envelope: dict[str, Any] | None = None,
        websocket: web.WebSocketResponse | None = None,
    ) -> None:
        """Serve the currently attached physical Cisco socket."""
        attached_websocket = websocket or self.websocket
        try:
            if initial_envelope is not None:
                await self._process_envelope(initial_envelope)
            async for message in attached_websocket:
                if message.type != WSMsgType.TEXT:
                    LOGGER.warning(
                        "Rejected non-text Cisco frame conversation_id=%s type=%s",
                        self.conversation_id or "unknown",
                        message.type.name,
                    )
                    await self._send_error("bad_request", "only JSON text frames are accepted")
                    break
                try:
                    await self._process_envelope(parse_text(message.data))
                except (ProtocolError, ValueError):
                    break
                if self.closed:
                    break
        finally:
            if self.websocket is attached_websocket and not self.closed:
                self._schedule_idle_close(attached_websocket)

    def _schedule_idle_close(self, websocket: web.WebSocketResponse) -> None:
        """Retain state briefly while Cisco opens a continuation socket."""
        if self.idle_close_task:
            self.idle_close_task.cancel()
        self.idle_close_task = asyncio.create_task(
            self._close_if_not_continued(websocket),
            name=f"continuation-grace-{self.conversation_id}",
        )

    async def _close_if_not_continued(self, websocket: web.WebSocketResponse) -> None:
        try:
            await asyncio.sleep(self.config.continuation_grace_seconds)
            if self.websocket is websocket and not self.closed:
                await self.close()
        except asyncio.CancelledError:
            pass

    async def _process_envelope(self, envelope: dict[str, Any]) -> None:
        try:
            await self._handle(envelope)
        except ProtocolError as exc:
            LOGGER.warning(
                "Rejected Cisco request conversation_id=%s code=%s detail=%s",
                self.conversation_id or "unknown",
                exc.code,
                exc.detail,
            )
            await self._send_error(exc.code, exc.detail)
            raise
        except ValueError as exc:
            LOGGER.warning(
                "Rejected Cisco media conversation_id=%s detail=%s",
                self.conversation_id or "unknown",
                exc,
            )
            await self._send_error("unsupported_media", str(exc))
            raise

    async def _handle(self, envelope: dict[str, Any]) -> None:
        message_type = envelope.get("type")
        if message_type == "PING":
            if not self.conversation_id:
                raise ProtocolError("SESSION_START is required before PING")
            result = pong(envelope, self.conversation_id)
            if result["seq"] <= self.last_inbound_seq:
                raise ProtocolError("PING seq must increase monotonically")
            self.last_inbound_seq = result["seq"]
            await self._send(result)
            return
        request = parse_request(envelope)
        if self.config.auth_enabled and request.payload["customer_org_id"] != self.config.org_uuid:
            raise ProtocolError("customer_org_id does not match the authenticated organization", "unauthorized")
        if request.seq <= self.last_inbound_seq:
            raise ProtocolError("request seq must increase monotonically")
        self.last_inbound_seq = request.seq
        if self.conversation_id and request.conversation_id != self.conversation_id:
            raise ProtocolError("one WebSocket cannot carry multiple conversations")
        event_type = request.input_value.get("event_type") if request.input_kind == "event_input" else None
        if self.phase is SessionPhase.WAITING_SESSION_START:
            if request.input_kind != "event_input" or event_type != "SESSION_START":
                LOGGER.warning(
                    "First Cisco request was not SESSION_START conversation_id=%s seq=%s input_kind=%s event_type=%s",
                    request.conversation_id,
                    request.seq,
                    request.input_kind,
                    event_type,
                )
                raise ProtocolError("SESSION_START must be the first application request")
            await self._start(request)
            return
        if event_type == "SESSION_START":
            raise ProtocolError("SESSION_START may only be sent once")
        if self.phase is SessionPhase.TERMINAL:
            raise ProtocolError("session is already terminal")
        if event_type == "SESSION_END":
            await self._session_end()
        elif request.input_kind == "audio_input":
            value = request.input_value
            encoded_audio = value.get("caller_audio_b64")
            if not encoded_audio:
                return
            pcm = self.caller_audio.convert(
                encoded_audio,
                value.get("encoding"),
                value.get("sample_rate_hertz"),
            )
            assert self.bridge is not None
            await self.bridge.send_audio(pcm)
        else:
            self._begin_reply(request.input_kind)
            self._arm_timeout(self.config.first_audio_timeout_seconds)
            assert self.bridge is not None
            if request.input_kind == "dtmf_input":
                await self.bridge.send_input("dtmf", request.input_value)
            elif event_type == "NO_INPUT":
                await self.bridge.send_input("no-input", request.input_value)
            elif event_type == "CUSTOM_EVENT":
                await self.bridge.send_input("custom-event", request.input_value)
            elif event_type == "START_OF_DTMF":
                await self.bridge.send_input("start-of-dtmf", request.input_value)

    async def _start(self, request: Any) -> None:
        self.conversation_id = request.conversation_id
        self.bridge = NemotronBridge(self.config, self.conversation_id)
        try:
            await asyncio.wait_for(self.bridge.start(), self.config.send_timeout_seconds)
        except TimeoutError as exc:
            raise ProtocolError("Nemotron connection timed out", "timeout") from exc
        except Exception as exc:
            raise ProtocolError("Nemotron connection failed", "upstream_error") from exc
        self.phase = SessionPhase.WELCOME_REPLY
        LOGGER.info("Started Cisco conversation conversation_id=%s", self.conversation_id)
        self._begin_reply("welcome")
        self._arm_timeout(self.config.first_audio_timeout_seconds)
        self.bridge_pump = asyncio.create_task(self._pump_bridge(), name=f"bridge-pump-{self.conversation_id}")

    async def _pump_bridge(self) -> None:
        assert self.bridge is not None
        while not self.closed:
            item = await self.bridge.events.get()
            kind = item.get("kind")
            if kind == "audio":
                if not self.reply.active:
                    self._begin_reply("assistant-output")
                if self.phase is not SessionPhase.WELCOME_REPLY:
                    self.phase = SessionPhase.ASSISTANT_REPLY
                self.reply.note_audio()
                self._arm_timeout(self.config.completion_timeout_seconds)
                for chunk in self.bot_audio.feed(
                    item["audio"],
                    item["sample_rate"],
                    item["num_channels"],
                ):
                    await self._send_audio(chunk)
            elif kind == "transcript":
                self.pending_transcript = item["text"]
            elif kind == "event":
                await self._handle_backend_event(item["type"])
            elif kind == "output-complete":
                await self._finish_reply()
            elif kind == "error":
                await self._send_error("upstream_error", item.get("detail", "Nemotron failed"))
                self.closed = True
                await self.websocket.close(code=1011, message=b"upstream failure")
                return

    async def _handle_backend_event(self, event_type: str) -> None:
        if event_type == "user-started-speaking":
            if self.phase is SessionPhase.CALLER_ACTIVE:
                return
            if not self.reply.active:
                self._begin_reply("caller-turn")
            self.phase = SessionPhase.CALLER_ACTIVE
            await self._send_event("START_OF_INPUT")
        elif event_type == "user-turn-finalized":
            if self.phase is not SessionPhase.CALLER_ACTIVE:
                return
            if not self.reply.active:
                self._begin_reply("caller-turn")
            await self._send_event("END_OF_INPUT")
            self.phase = SessionPhase.ASSISTANT_REPLY
            self._arm_timeout(self.config.first_audio_timeout_seconds)
        elif event_type == "bot-started-speaking":
            if not self.reply.active:
                self._begin_reply("assistant-output")
            if self.phase is not SessionPhase.WELCOME_REPLY:
                self.phase = SessionPhase.ASSISTANT_REPLY

    async def _finish_reply(self) -> None:
        if not self.reply.finish():
            return
        self._cancel_reply_timeout()
        for chunk in self.bot_audio.flush():
            await self._send_audio(chunk)
        await self._send_audio_final()
        if self.phase not in {SessionPhase.CALLER_ACTIVE, SessionPhase.TERMINAL}:
            self.phase = SessionPhase.LISTENING

    async def _send_audio_final(self) -> None:
        await self._send_payload(
            {
                "prompts": [{"audio_content_b64": "", "is_barge_in_enabled": True}],
                "input_mode": "INPUT_VOICE_DTMF",
                "input_sensitive": False,
                "response_type": "FINAL",
            }
        )
        self.reply_complete.set()

    def _begin_reply(self, label: str) -> None:
        if self.reply.active:
            return
        self.reply.begin(label)
        self.reply_complete.clear()
        LOGGER.debug(
            "Opened Cisco reply conversation_id=%s reply=%s label=%s phase=%s",
            self.conversation_id,
            self.reply.ordinal,
            label,
            self.phase.name,
        )

    def _arm_timeout(self, delay: float) -> None:
        self._cancel_reply_timeout()
        self.reply_timeout_task = asyncio.create_task(self._reply_timeout(delay))

    def _cancel_reply_timeout(self) -> None:
        if self.reply_timeout_task is not None and self.reply_timeout_task is not asyncio.current_task():
            self.reply_timeout_task.cancel()
        self.reply_timeout_task = None

    async def _reply_timeout(self, delay: float) -> None:
        try:
            await asyncio.sleep(delay)
            if self.reply.finish() and not self.closed:
                for chunk in self.bot_audio.flush():
                    await self._send_audio(chunk)
                await self._send_error("timeout", "Nemotron reply timed out")
                await self._send_audio_final()
                self.phase = SessionPhase.LISTENING
        except asyncio.CancelledError:
            raise
        finally:
            if self.reply_timeout_task is asyncio.current_task():
                self.reply_timeout_task = None

    async def _send_audio(self, chunk: bytes) -> None:
        if not 320 <= len(chunk) <= 32 * 1024:
            raise RuntimeError("internal audio chunk bound violation")
        self.audio_chunks_sent += 1
        self.audio_bytes_sent += len(chunk)
        if self.audio_chunks_sent == 1:
            LOGGER.info(
                "Sending first audio chunk conversation_id=%s bytes=%s",
                self.conversation_id,
                len(chunk),
            )
        prompt: dict[str, Any] = {
            "audio_content_b64": encode_audio(chunk),
            "is_barge_in_enabled": True,
        }
        if self.pending_transcript:
            prompt["text"] = self.pending_transcript
            self.pending_transcript = ""
        await self._send_payload(
            {
                "prompts": [prompt],
                "output_events": [],
                "input_mode": "INPUT_VOICE_DTMF",
                "input_sensitive": False,
                "response_type": "CHUNK",
            }
        )

    async def _send_event(self, event_type: str) -> None:
        await self._send_payload(
            {
                "output_events": [{"event_type": event_type}],
                "response_type": "FINAL" if event_type == "TRANSFER_TO_AGENT" else "CHUNK",
            }
        )
        if event_type == "TRANSFER_TO_AGENT":
            self._cancel_reply_timeout()
            self.reply.reset()
            self.reply_complete.set()
            self.phase = SessionPhase.TERMINAL

    async def _session_end(self) -> None:
        self._cancel_reply_timeout()
        self.reply.reset()
        self.reply_complete.set()
        self.phase = SessionPhase.TERMINAL
        await self._send_payload(
            {
                "output_events": [{"event_type": "SESSION_END"}],
                "response_type": "FINAL",
            }
        )
        await self.websocket.close(code=1000, message=b"session ended")
        await self.close()

    async def _send_payload(self, payload: dict[str, Any]) -> None:
        await self._send(response(self.sequence.next(), self.conversation_id, payload))

    async def _send_error(self, code: str, detail: str) -> None:
        await self._send(error(self.sequence.next(), self.conversation_id, code, detail))

    async def _send(self, value: dict[str, Any]) -> None:
        async with self.send_lock:
            await asyncio.wait_for(self.websocket.send_json(value), self.config.send_timeout_seconds)

    async def close(self) -> None:
        """Release all per-conversation resources exactly once."""
        if self.closed and self.bridge is None and self.bridge_pump is None:
            return
        self.closed = True
        self.phase = SessionPhase.TERMINAL
        self.reply.reset()
        self.reply_complete.set()
        if self.idle_close_task is not None and self.idle_close_task is not asyncio.current_task():
            self.idle_close_task.cancel()
        self.idle_close_task = None
        bridge, self.bridge = self.bridge, None
        pump, self.bridge_pump = self.bridge_pump, None
        timeout_task, self.reply_timeout_task = self.reply_timeout_task, None
        if bridge is not None:
            await bridge.close()
        if pump is not None and pump is not asyncio.current_task():
            pump.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await pump
        if timeout_task is not None and timeout_task is not asyncio.current_task():
            timeout_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await timeout_task
        if not self.websocket.closed:
            await self.websocket.close()
        LOGGER.info(
            "Closed Cisco conversation conversation_id=%s audio_chunks=%s audio_bytes=%s",
            self.conversation_id or "unknown",
            self.audio_chunks_sent,
            self.audio_bytes_sent,
        )
        self.closed_event.set()
