"""Per-socket Cisco conversation state machine."""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

from aiohttp import WSMsgType, web

from webex_byova_ws_adapter.audio import CallerAudioConverter, WebexAudioChunker, encode_audio
from webex_byova_ws_adapter.bridge import NemotronBridge
from webex_byova_ws_adapter.completion import BaselineCompletionPolicy
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


class Conversation:
    """Handle one Webex WebSocket and one stable conversation ID."""

    def __init__(self, config: AdapterConfig, websocket: web.WebSocketResponse) -> None:
        """Create isolated state for one accepted Webex socket."""
        self.config = config
        self.websocket = websocket
        self.conversation_id = ""
        self.last_inbound_seq = 0
        self.sequence = OutboundSequence()
        self.bridge: NemotronBridge | None = None
        self.bridge_pump: asyncio.Task[None] | None = None
        self.reply_timeout_task: asyncio.Task[None] | None = None
        self.send_lock = asyncio.Lock()
        self.caller_audio = CallerAudioConverter()
        self.bot_audio = WebexAudioChunker()
        self.completion = BaselineCompletionPolicy()
        self.session_started = False
        self.pending_transcript = ""
        self.closed = False

    async def run(self) -> None:
        """Run until SESSION_END, protocol failure, or transport close."""
        try:
            async for message in self.websocket:
                if message.type != WSMsgType.TEXT:
                    await self._send_error("bad_request", "only JSON text frames are accepted")
                    break
                try:
                    await self._handle(parse_text(message.data))
                except ProtocolError as exc:
                    await self._send_error(exc.code, exc.detail)
                    break
                except ValueError as exc:
                    await self._send_error("unsupported_media", str(exc))
                    break
                if self.closed:
                    break
        finally:
            await self.close()

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
        if not self.session_started:
            if request.input_kind != "event_input" or event_type != "SESSION_START":
                raise ProtocolError("SESSION_START must be the first application request")
            await self._start(request)
            return
        if event_type == "SESSION_START":
            raise ProtocolError("SESSION_START may only be sent once")
        if event_type == "SESSION_END":
            await self._session_end()
        elif request.input_kind == "audio_input":
            self._begin_reply()
            value = request.input_value
            pcm = self.caller_audio.convert(
                value.get("caller_audio_b64"),
                value.get("encoding"),
                value.get("sample_rate_hertz"),
            )
            assert self.bridge is not None
            await self.bridge.send_audio(pcm)
        else:
            self._begin_reply()
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
        self.session_started = True
        self._begin_reply()
        self.bridge_pump = asyncio.create_task(self._pump_bridge(), name=f"bridge-pump-{self.conversation_id}")

    async def _pump_bridge(self) -> None:
        assert self.bridge is not None
        while not self.closed:
            item = await self.bridge.events.get()
            kind = item.get("kind")
            if kind == "audio":
                self.completion.note_audio()
                self._arm_timeout(self.config.completion_timeout_seconds)
                for chunk in self.bot_audio.feed(item["audio"]):
                    await self._send_audio(chunk)
            elif kind == "transcript":
                self.pending_transcript = item["text"]
            elif kind == "event":
                await self._handle_backend_event(item["type"])
            elif kind == "error":
                await self._send_error("upstream_error", item.get("detail", "Nemotron failed"))
                self.closed = True
                await self.websocket.close(code=1011, message=b"upstream failure")
                return

    async def _handle_backend_event(self, event_type: str) -> None:
        if event_type == "user-started-speaking":
            await self._send_event("START_OF_INPUT")
        elif event_type == "user-turn-finalized":
            await self._send_event("END_OF_INPUT")
        elif self.completion.should_finalize(event_type):
            self._cancel_reply_timeout()
            for chunk in self.bot_audio.flush():
                await self._send_audio(chunk)
            await self._send_payload({"response_type": "FINAL", "input_mode": "INPUT_VOICE_DTMF"})

    def _begin_reply(self) -> None:
        if self.completion.reply_open:
            return
        self.completion.begin_reply()
        self._arm_timeout(self.config.first_audio_timeout_seconds)

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
            if self.completion.force_finalize() and not self.closed:
                for chunk in self.bot_audio.flush():
                    await self._send_audio(chunk)
                await self._send_error("timeout", "Nemotron reply timed out")
                await self._send_payload({"response_type": "FINAL", "input_mode": "INPUT_VOICE_DTMF"})
        except asyncio.CancelledError:
            raise
        finally:
            if self.reply_timeout_task is asyncio.current_task():
                self.reply_timeout_task = None

    async def _send_audio(self, chunk: bytes) -> None:
        if not 320 <= len(chunk) <= 32 * 1024:
            raise RuntimeError("internal audio chunk bound violation")
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
                "response_type": "CHUNK",
            }
        )

    async def _session_end(self) -> None:
        self._cancel_reply_timeout()
        self.completion.force_finalize()
        await self._send_payload(
            {
                "output_events": [{"event_type": "SESSION_END"}],
                "response_type": "FINAL",
            }
        )
        self.closed = True
        await self.websocket.close(code=1000, message=b"session ended")

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
