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
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import websockets
from pipecat.frames.protobufs import frames_pb2

from cisco_webex_byova_adapter.audio import TARGET_SAMPLE_RATE
from cisco_webex_byova_adapter.config import AdapterConfig

logger = logging.getLogger(__name__)


_TRACE_PATH = os.environ.get("NEMOTRON_BYOVA_ADAPTER_TRACE_FILE", "")
_PCM16_BYTES_PER_SAMPLE = 2
_NEMOTRON_INPUT_CHUNK_MS = 32
_NEMOTRON_INPUT_CHUNK_BYTES = TARGET_SAMPLE_RATE * _NEMOTRON_INPUT_CHUNK_MS // 1000 * _PCM16_BYTES_PER_SAMPLE
_ADAPTER_VENDOR_CONFIG_KEYS = {"transfer_metadata"}
_DTMF_FIELD_LENGTHS = {"phone_number": 10, "date_of_birth": 8}
_TERMINAL_ACTIONS = {"transfer_to_human", "end_call"}


def _dtmf_validation_error(field_name: str, value: str) -> str | None:
    """Return a caller-facing reason when collected digits are unusable."""
    expected_length = _DTMF_FIELD_LENGTHS.get(field_name)
    if expected_length is None:
        return "that keypad field is not supported"
    if len(value) != expected_length:
        return f"it was not exactly {expected_length} digits"
    if field_name == "phone_number":
        return None
    try:
        parsed = datetime.strptime(value, "%d%m%Y").date()
    except ValueError:
        return "it was not a valid date in day month year format"
    if parsed > date.today() or parsed.year < 1900:
        return "it was not a valid past date"
    return None


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
    # websockets rejects an SSL context on plain ws:// targets.
    scheme = uri.split("://", 1)[0].lower() if "://" in uri else ""
    if scheme == "wss":
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
    reader_task: asyncio.Task | None = None
    finalizer_task: asyncio.Task | None = None
    closed: bool = False
    vendor_config: dict[str, Any] = field(default_factory=dict)
    last_audio_send_monotonic: float = 0.0
    first_audio_send_monotonic: float = 0.0
    # Caller-audio (INPUT) instrumentation: count frames + measure inter-arrival
    # gaps so we can tell streaming-vs-bursty input apart from the adapter side.
    caller_audio_frames: int = 0
    caller_audio_bytes: int = 0
    pending_bot_prompt_text: str = ""
    last_spoken_bot_text: str = ""
    bot_speaking: bool = False
    terminal_action: str | None = None
    terminal_reason: str = ""
    control_transfer_metadata: dict[str, Any] = field(default_factory=dict)
    pending_dtmf_field: str | None = None
    caller_resample_state: Any = None

    async def start(self) -> None:
        """Open a stateless Nemotron websocket channel for this conversation."""
        self.vendor_config = self._parse_vendor_config()
        uri = f"{self.config.nemotron_voice_agent_ws.rstrip('/')}/api/ws"
        unsupported_keys = sorted(set(self.vendor_config) - _ADAPTER_VENDOR_CONFIG_KEYS)
        if unsupported_keys:
            logger.warning(
                "Ignoring unsupported vendor_specific_config keys for conversation_id=%s keys=%s",
                self.conversation_id,
                unsupported_keys,
            )
        logger.info(
            "Opening stateless Nemotron websocket for conversation_id=%s",
            self.conversation_id,
        )
        _trace("open websocket")
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

    async def _send_client_message(self, message_type: str, data: dict[str, Any]) -> None:
        if self.websocket is None:
            raise RuntimeError(f"websocket is not connected for conversation {self.conversation_id}")
        payload = {
            "label": "rtvi-ai",
            "type": message_type,
            "id": f"{message_type}-{self.conversation_id}-{time.monotonic_ns()}",
            "data": data,
        }
        frame = frames_pb2.Frame(message=frames_pb2.MessageFrame(data=json.dumps(payload)))
        await self.websocket.send(frame.SerializeToString())

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
                logger.debug("Nemotron frame_type=%s conversation_id=%s", frame_type, self.conversation_id)
                _trace(f"nemotron frame_type={frame_type}")
                if frame_type == "audio" and frame.audio.audio:
                    item = {
                        "kind": "audio",
                        "audio": frame.audio.audio,
                        "sample_rate": frame.audio.sample_rate,
                        "num_channels": frame.audio.num_channels,
                    }
                    if self.pending_bot_prompt_text:
                        # Cisco can surface both audio and transcript text for a
                        # prompt, but Nemotron emits them as separate frames.
                        item["text"] = self.pending_bot_prompt_text
                        self.pending_bot_prompt_text = ""
                    await self.outbound_queue.put(item)
                    # A fresh audio chunk resets turn-final detection.
                    self._schedule_final_marker()
                elif frame_type == "message":
                    try:
                        message_data = getattr(frame.message, "data", "")
                    except Exception:
                        message_data = ""
                    _trace(f"message data={message_data[:300]}")
                    self._handle_message_payload(message_data)
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

        if message_type == "webex-call-control":
            action = str(payload.get("action", "")).strip()
            if action in _TERMINAL_ACTIONS:
                reason = str(payload.get("reason", "")).strip()[:240]
                metadata = payload.get("metadata")
                self.request_terminal_action(
                    action,
                    reason=reason,
                    metadata=metadata if isinstance(metadata, dict) else {},
                )
            elif action == "request_keypad_input":
                self.request_keypad_input(str(payload.get("field", "")).strip())
            return

        if not isinstance(data, dict):
            return

        text = str(data.get("text", "")).strip()
        if message_type == "bot-output" and text and bool(data.get("spoken")) and text != self.last_spoken_bot_text:
            self.last_spoken_bot_text = text
            # Attach the next spoken transcript to the next outbound audio
            # chunk so Webex sees aligned prompt text and audio.
            self.pending_bot_prompt_text = text

    def request_terminal_action(self, action: str, *, reason: str, metadata: dict[str, Any]) -> bool:
        """Record exactly one LLM-authorized terminal action."""
        if action not in _TERMINAL_ACTIONS or self.terminal_action is not None:
            return False
        self.terminal_action = action
        self.terminal_reason = reason
        self.control_transfer_metadata = metadata
        self.pending_dtmf_field = None
        logger.info(
            "Accepted LLM Webex terminal action=%s conversation_id=%s",
            action,
            self.conversation_id,
        )
        return True

    def request_keypad_input(self, field_name: str) -> bool:
        """Arm one supported sensitive DTMF collection."""
        if self.terminal_action or field_name not in _DTMF_FIELD_LENGTHS:
            return False
        if self.pending_dtmf_field not in (None, field_name):
            return False
        self.pending_dtmf_field = field_name
        logger.info(
            "Armed secure Webex keypad collection field=%s conversation_id=%s",
            field_name,
            self.conversation_id,
        )
        return True

    @property
    def dtmf_input_length(self) -> int:
        """Return the exact digit count that also completes collection."""
        return _DTMF_FIELD_LENGTHS.get(self.pending_dtmf_field or "", 0)

    async def ingest_dtmf(self, digits: list[str]) -> str:
        """Validate one completed Cisco DTMF interaction and notify the pipeline."""
        field_name = self.pending_dtmf_field
        if field_name is None or self.terminal_action:
            return "ignored"
        value = "".join(digit for digit in digits if digit.isdigit())
        if not value:
            # Cisco completed collection on digit count, so a trailing
            # terminator can arrive as its own interaction.
            return "ignored"
        validation_error = _dtmf_validation_error(field_name, value)
        if validation_error:
            await self._send_client_message(
                "webex-dtmf-error",
                {"field": field_name, "reason": validation_error},
            )
            logger.info(
                "Rejected secure Webex keypad length field=%s conversation_id=%s",
                field_name,
                self.conversation_id,
            )
            return "invalid"

        self.pending_dtmf_field = None
        await self._send_client_message(
            "webex-dtmf",
            {"field": field_name, "value": value},
        )
        logger.info(
            "Forwarded completed secure Webex keypad field=%s conversation_id=%s",
            field_name,
            self.conversation_id,
        )
        return "complete"

    async def retry_dtmf(self, reason: str) -> bool:
        """Clear partial digits and ask the pipeline to retry the active field."""
        field_name = self.pending_dtmf_field
        if field_name is None:
            return False
        await self._send_client_message("webex-dtmf-error", {"field": field_name, "reason": reason})
        logger.info(
            "Cancelled secure Webex keypad field=%s conversation_id=%s",
            field_name,
            self.conversation_id,
        )
        return True

    def transfer_metadata(self) -> dict[str, Any]:
        """Return transfer metadata for a transfer-to-agent output event."""
        metadata = self.vendor_config.get("transfer_metadata")
        if not isinstance(metadata, dict):
            try:
                metadata = json.loads(self.config.default_transfer_metadata_json)
            except json.JSONDecodeError:
                metadata = {"route": "live-agent"}
        merged = {**metadata, **self.control_transfer_metadata}
        if self.terminal_reason:
            merged["reason"] = self.terminal_reason
        return {str(key)[:64]: value for key, value in list(merged.items())[:16]}

    def _schedule_final_marker(self) -> None:
        if self.finalizer_task:
            self.finalizer_task.cancel()
        self.finalizer_task = asyncio.create_task(self._emit_final_after_idle())

    async def _emit_final_after_idle(self) -> None:
        await asyncio.sleep(self.config.output_idle_timeout_ms / 1000)
        await self.outbound_queue.put({"kind": "final"})

    async def close(self) -> None:
        """Close the websocket and background tasks for this session."""
        if self.closed:
            return
        logger.info(
            "Closing Nemotron websocket conversation_id=%s",
            self.conversation_id,
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
