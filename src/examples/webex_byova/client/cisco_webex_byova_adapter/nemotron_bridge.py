# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Nemotron websocket session bridge used by the Cisco Webex BYOVA adapter."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import ssl
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode

import websockets
from pipecat.frames.protobufs import frames_pb2

from cisco_webex_byova_adapter.audio import TARGET_SAMPLE_RATE
from cisco_webex_byova_adapter.config import AdapterConfig

logger = logging.getLogger(__name__)


_TRACE_PATH = os.environ.get("NEMOTRON_BYOVA_ADAPTER_TRACE_FILE", "")
_PCM16_BYTES_PER_SAMPLE = 2
_NEMOTRON_INPUT_CHUNK_MS = 32
_NEMOTRON_INPUT_CHUNK_BYTES = TARGET_SAMPLE_RATE * _NEMOTRON_INPUT_CHUNK_MS // 1000 * _PCM16_BYTES_PER_SAMPLE


def _trace(message: str) -> None:
    if not _TRACE_PATH:
        return
    with open(_TRACE_PATH, "a", encoding="utf-8") as trace_file:
        trace_file.write(message + "\n")


def _ssl_context(insecure: bool) -> ssl.SSLContext | None:
    if not insecure:
        return None
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _ssl_for_uri(uri: str, insecure: bool) -> ssl.SSLContext | None:
    # The websockets / urllib libraries reject an SSL context on plain ws:// or
    # http:// targets. Only attach one when the scheme is actually TLS.
    scheme = uri.split("://", 1)[0].lower() if "://" in uri else ""
    if scheme in ("wss", "https"):
        return _ssl_context(insecure)
    return None


@dataclass(slots=True)
class NemotronSession:
    """Own the Nemotron HTTP and websocket state for one Cisco conversation."""

    config: AdapterConfig
    conversation_id: str
    vendor_specific_config: str = ""
    outbound_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    websocket: Any | None = None
    session_id: str = ""
    reader_task: asyncio.Task | None = None
    finalizer_task: asyncio.Task | None = None
    closed: bool = False
    vendor_config: dict[str, Any] = field(default_factory=dict)
    transcripts: list[str] = field(default_factory=list)
    user_turns: list[str] = field(default_factory=list)
    bot_turns: list[str] = field(default_factory=list)
    last_audio_send_monotonic: float = 0.0
    first_audio_send_monotonic: float = 0.0
    # Caller-audio (INPUT) instrumentation: count frames + measure inter-arrival
    # gaps so we can tell streaming-vs-bursty input apart from the adapter side.
    caller_audio_frames: int = 0
    caller_audio_bytes: int = 0
    last_inbound_activity_monotonic: float = field(default_factory=time.monotonic)
    response_started: bool = False
    pending_bot_prompt_text: str = ""
    last_spoken_bot_text: str = ""
    bot_speaking: bool = False
    transfer_requested: bool = False
    end_session_requested: bool = False
    caller_resample_state: Any = None

    async def start(self) -> None:
        """Create a Nemotron session and open its websocket channel."""
        self.vendor_config = self._parse_vendor_config()
        self.session_id = await asyncio.to_thread(self._create_session)
        params = urlencode({"session_id": self.session_id})
        uri = f"{self.config.nemotron_voice_agent_ws.rstrip('/')}/api/ws?{params}"
        logger.info(
            "Opening Nemotron websocket for conversation_id=%s session_id=%s",
            self.conversation_id,
            self.session_id,
        )
        _trace(f"open websocket session_id={self.session_id}")
        self.websocket = await websockets.connect(uri, ssl=_ssl_for_uri(uri, self.config.allow_insecure_tls))
        self.reader_task = asyncio.create_task(self._reader_loop(), name=f"nemotron-reader-{self.conversation_id}")
        # Announce the adapter as a ready RTVI client so the backend can send
        # the opening bot turn.
        await self._send_client_ready()

    async def _send_client_ready(self) -> None:
        if self.websocket is None:
            raise RuntimeError(f"websocket is not connected for conversation {self.conversation_id}")
        payload = {
            "label": "rtvi-ai",
            "type": "client-ready",
            "id": f"client-ready-{self.conversation_id}",
            "data": {
                "version": "1.2.0",
                "about": {"library": "cisco-webex-byova-adapter"},
            },
        }
        frame = frames_pb2.Frame(message=frames_pb2.MessageFrame(data=json.dumps(payload)))
        await self.websocket.send(frame.SerializeToString())
        _trace("sent client-ready")

    def _create_session(self) -> str:
        payload: dict[str, Any] = {"pipeline_mode": self.config.pipeline_mode}
        if self.vendor_config:
            payload.update(self.vendor_config)

        body = json.dumps(payload).encode("utf-8")
        url = f"{self.config.nemotron_voice_agent_http.rstrip('/')}/api/session-config"
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(
            request,
            context=_ssl_for_uri(url, self.config.allow_insecure_tls),
            timeout=30,
        ) as response:
            data = json.load(response)
        session_id = data.get("session_id", "")
        if not session_id:
            raise RuntimeError("Nemotron session-config response did not include session_id")
        return session_id

    def _parse_vendor_config(self) -> dict[str, Any]:
        if not self.vendor_specific_config.strip():
            return {}
        try:
            return json.loads(self.vendor_specific_config)
        except json.JSONDecodeError:
            return {"vendor_specific_config": self.vendor_specific_config}

    async def send_audio(self, pcm_audio: bytes) -> None:
        """Split caller audio into unpaced 32 ms frames and send immediately."""
        if not self.websocket:
            raise RuntimeError("Nemotron websocket is not connected")
        for offset in range(0, len(pcm_audio), _NEMOTRON_INPUT_CHUNK_BYTES):
            chunk = pcm_audio[offset : offset + _NEMOTRON_INPUT_CHUNK_BYTES]
            now = time.monotonic()
            prev = self.last_audio_send_monotonic
            gap_ms = (now - prev) * 1000.0 if prev else 0.0
            self.caller_audio_frames += 1
            self.caller_audio_bytes += len(chunk)
            if not self.first_audio_send_monotonic:
                self.first_audio_send_monotonic = now
            self.last_audio_send_monotonic = now
            elapsed_ms = (now - self.first_audio_send_monotonic) * 1000.0
            _trace(
                f"caller_audio cid={self.conversation_id} n={self.caller_audio_frames} "
                f"bytes={len(chunk)} gap_ms={gap_ms:.0f}"
            )
            if self.caller_audio_frames <= 20 or self.caller_audio_frames % 50 == 0 or (prev and gap_ms > 150.0):
                log = logger.warning if prev and gap_ms > 500.0 else logger.info
                log(
                    (
                        "[IN] send_to_nemotron conversation_id=%s frame_n=%d bytes=%d "
                        "total_bytes=%d gap_ms=%.0f elapsed_ms=%.0f"
                    ),
                    self.conversation_id,
                    self.caller_audio_frames,
                    len(chunk),
                    self.caller_audio_bytes,
                    gap_ms,
                    elapsed_ms,
                )
            frame = frames_pb2.Frame(
                audio=frames_pb2.AudioRawFrame(audio=chunk, sample_rate=TARGET_SAMPLE_RATE, num_channels=1)
            )
            await self.websocket.send(frame.SerializeToString())

    async def _reader_loop(self) -> None:
        if self.websocket is None:
            raise RuntimeError("WebSocket is not initialized in _reader_loop")
        try:
            async for message in self.websocket:
                frame = frames_pb2.Frame.FromString(message)
                frame_type = frame.WhichOneof("frame")
                logger.info("Nemotron frame_type=%s conversation_id=%s", frame_type, self.conversation_id)
                _trace(f"nemotron frame_type={frame_type}")
                self.last_inbound_activity_monotonic = time.monotonic()
                self.response_started = True
                if frame_type == "audio" and frame.audio.audio:
                    item = {"kind": "audio", "audio": frame.audio.audio}
                    if self.pending_bot_prompt_text:
                        # Cisco can surface both audio and transcript text for a
                        # prompt, but Nemotron emits them as separate frames.
                        item["text"] = self.pending_bot_prompt_text
                        self.pending_bot_prompt_text = ""
                    await self.outbound_queue.put(item)
                    # A fresh audio chunk resets turn-final detection.
                    self._schedule_final_marker()
                elif frame_type == "text":
                    text = getattr(frame.text, "text", "").strip()
                    if text:
                        self.transcripts.append(text)
                elif frame_type == "message":
                    try:
                        message_data = getattr(frame.message, "data", "")
                    except Exception:
                        message_data = ""
                    _trace(f"message data={message_data[:300]}")
                    self._handle_message_payload(message_data)
                    await self.outbound_queue.put({"kind": "message", "message": frame.message})
        except Exception as exc:
            logger.exception("Nemotron websocket reader failed conversation_id=%s", self.conversation_id)
            await self.outbound_queue.put({"kind": "error", "error": str(exc)})
        finally:
            self.closed = True
            if self.finalizer_task:
                self.finalizer_task.cancel()

    def _handle_message_payload(self, message_data: str) -> None:
        if not message_data:
            return
        try:
            payload = json.loads(message_data)
        except json.JSONDecodeError:
            return

        message_type = payload.get("type")
        data = payload.get("data")
        if message_type == "server-message" and isinstance(data, dict):
            payload = data
            message_type = payload.get("type")
            data = payload

        # Pipecat RTVI VAD events. We forward these to the adapter so it
        # can emit Cisco BYoVA START_OF_INPUT / END_OF_INPUT events per
        # spec §3.2 — no need for adapter-side VAD because Nemotron's
        # Silero/Smart-Turn already detects this.
        if message_type == "user-started-speaking":
            now = time.monotonic()
            since_last_audio_ms = (
                (now - self.last_audio_send_monotonic) * 1000.0 if self.last_audio_send_monotonic else -1.0
            )
            elapsed_ms = (now - self.first_audio_send_monotonic) * 1000.0 if self.first_audio_send_monotonic else -1.0
            logger.info(
                (
                    "[IN] nemotron_user_started conversation_id=%s frame_n=%d "
                    "total_bytes=%d since_last_audio_ms=%.0f elapsed_ms=%.0f"
                ),
                self.conversation_id,
                self.caller_audio_frames,
                self.caller_audio_bytes,
                since_last_audio_ms,
                elapsed_ms,
            )
            self.outbound_queue.put_nowait({"kind": "user_started_speaking"})
            return
        if message_type == "user-stopped-speaking":
            now = time.monotonic()
            since_last_audio_ms = (
                (now - self.last_audio_send_monotonic) * 1000.0 if self.last_audio_send_monotonic else -1.0
            )
            elapsed_ms = (now - self.first_audio_send_monotonic) * 1000.0 if self.first_audio_send_monotonic else -1.0
            logger.info(
                (
                    "[IN] nemotron_user_stopped conversation_id=%s frame_n=%d "
                    "total_bytes=%d since_last_audio_ms=%.0f elapsed_ms=%.0f"
                ),
                self.conversation_id,
                self.caller_audio_frames,
                self.caller_audio_bytes,
                since_last_audio_ms,
                elapsed_ms,
            )
            self.outbound_queue.put_nowait({"kind": "user_stopped_speaking"})
            return
        if message_type == "bot-started-speaking":
            self.bot_speaking = True
            self.outbound_queue.put_nowait({"kind": "bot_started_speaking"})
            return
        if message_type == "bot-stopped-speaking":
            self.bot_speaking = False
            self.outbound_queue.put_nowait({"kind": "bot_stopped_speaking"})
            self._schedule_final_marker()
            return

        if not isinstance(data, dict):
            return

        text = str(data.get("text", "")).strip()
        if message_type == "user-llm-text" and text:
            self.user_turns.append(text)
            self.transcripts.append(f"User: {text}")
            if self._matches_any(text, self._transfer_keywords()):
                self.transfer_requested = True
            elif self._matches_any(text, self._end_session_keywords()):
                self.end_session_requested = True
            return

        if message_type == "bot-output" and text and bool(data.get("spoken")) and text != self.last_spoken_bot_text:
            self.last_spoken_bot_text = text
            self.bot_turns.append(text)
            self.transcripts.append(f"Assistant: {text}")
            # Attach the next spoken transcript to the next outbound audio
            # chunk so Webex sees aligned prompt text and audio.
            self.pending_bot_prompt_text = text
            if self._matches_any(text, self._transfer_keywords()):
                self.transfer_requested = True
            elif self._matches_any(text, self._end_session_keywords()):
                self.end_session_requested = True

    def _transfer_keywords(self) -> list[str]:
        keywords = self.vendor_config.get("transfer_keywords")
        if isinstance(keywords, list):
            return [str(item).strip().lower() for item in keywords if str(item).strip()]
        return [item.strip().lower() for item in self.config.default_transfer_keywords.split(",") if item.strip()]

    def _end_session_keywords(self) -> list[str]:
        keywords = self.vendor_config.get("end_session_keywords")
        if isinstance(keywords, list):
            return [str(item).strip().lower() for item in keywords if str(item).strip()]
        return [item.strip().lower() for item in self.config.default_end_session_keywords.split(",") if item.strip()]

    @staticmethod
    def _matches_any(text: str, keywords: list[str]) -> bool:
        normalized = text.lower()
        return any(keyword in normalized for keyword in keywords)

    def transfer_metadata(self) -> dict[str, Any]:
        """Return transfer metadata for a transfer-to-agent output event."""
        metadata = self.vendor_config.get("transfer_metadata")
        if isinstance(metadata, dict):
            return metadata
        try:
            return json.loads(self.config.default_transfer_metadata_json)
        except json.JSONDecodeError:
            return {"route": "live-agent"}

    def _schedule_final_marker(self) -> None:
        if self.finalizer_task:
            self.finalizer_task.cancel()
        self.finalizer_task = asyncio.create_task(self._emit_final_after_idle())

    async def _emit_final_after_idle(self) -> None:
        await asyncio.sleep(self.config.output_idle_timeout_ms / 1000)
        await self.outbound_queue.put({"kind": "final"})

    async def wait_for_response_settle(self) -> None:
        """Wait for outbound activity to settle before closing the websocket."""
        start = time.monotonic()
        while not self.closed:
            now = time.monotonic()
            if self.response_started:
                idle_for = now - self.last_inbound_activity_monotonic
                if idle_for >= self.config.response_idle_timeout_secs:
                    _trace(f"response settled idle_for={idle_for:.3f}")
                    return
            elif now - start >= self.config.response_settle_timeout_secs:
                _trace("response settle timeout before outbound activity")
                return
            await asyncio.sleep(0.1)

    async def close(self) -> None:
        """Close the websocket and background tasks for this session."""
        if self.closed:
            return
        logger.info(
            "Closing Nemotron websocket conversation_id=%s session_id=%s",
            self.conversation_id,
            self.session_id,
        )
        self.closed = True
        if self.finalizer_task:
            self.finalizer_task.cancel()
        if self.websocket is not None:
            await self.websocket.close()
        if self.reader_task is not None:
            self.reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.reader_task
