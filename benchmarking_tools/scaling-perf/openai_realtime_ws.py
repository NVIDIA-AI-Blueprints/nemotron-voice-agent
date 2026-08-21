# SPDX-FileCopyrightText: Copyright (c) 2024–2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Small OpenAI Realtime WebSocket client used by scaling-perf.

The client deliberately sends a minimal ``session.update``. Declaring only the
input PCM rate preserves server-tuned VAD settings. After each utterance, the
benchmark keeps streaming silence until the response completes so streaming
ASR and server VAD remain active.
"""

# ruff: noqa: D102,D103,D105,D107

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import ssl
import time
from collections.abc import Mapping
from typing import Any

import websockets
from websockets.exceptions import ConnectionClosed

DEFAULT_OUTPUT_RATE = 24_000
DEFAULT_AUTH_SCHEME = "Bearer"
DEFAULT_CONNECT_TIMEOUT = 60.0
DEFAULT_READY_TIMEOUT = 60.0
API_KEY_ENV = "OPENAI_REALTIME_API_KEY"
AUTH_SCHEME_ENV = "OPENAI_REALTIME_AUTH_SCHEME"
WS_URL_ENV = "OPENAI_REALTIME_WS_URL"

SESSION_READY_TYPES = frozenset({"session.created", "session.updated"})


class RealtimeProtocolError(RuntimeError):
    """Server sent an OpenAI Realtime error event."""


class EndOfRealtimeResponse(Exception):
    """The current assistant response completed without more audio."""


class RealtimeTurnError(RealtimeProtocolError):
    """The current Realtime response failed without ending the session."""

    def __init__(self, message: str, *, terminal: bool = False):
        super().__init__(message)
        self.terminal = terminal


def resolve_protocol(*, protocol: str = "", ws_url: str = "") -> str:
    """Resolve the selected wire protocol."""
    value = (protocol or "").strip().lower()
    if value:
        if value not in {"rtvi", "realtime"}:
            raise ValueError(f"unsupported protocol {protocol!r}; expected rtvi or realtime")
        return value
    return "realtime" if (ws_url or "").strip() else "rtvi"


def resolve_api_key(explicit: str | None = None) -> str:
    """Resolve an explicit or environment-provided Realtime API key."""
    return (explicit or os.environ.get(API_KEY_ENV) or "").strip()


def resolve_ws_url(explicit: str | None = None) -> str:
    """Resolve an explicit or environment-provided Realtime WebSocket URL."""
    return (explicit or os.environ.get(WS_URL_ENV) or "").strip()


def resolve_auth_scheme(explicit: str | None = None) -> str:
    """Resolve the authorization scheme for a Realtime API key."""
    return (explicit or os.environ.get(AUTH_SCHEME_ENV) or DEFAULT_AUTH_SCHEME).strip()


def auth_headers(api_key: str, *, auth_scheme: str = DEFAULT_AUTH_SCHEME) -> dict[str, str]:
    """Build the optional Realtime authorization header."""
    key = (api_key or "").strip()
    if not key:
        return {}
    scheme = (auth_scheme or DEFAULT_AUTH_SCHEME).strip() or DEFAULT_AUTH_SCHEME
    return {"Authorization": f"{scheme} {key}"}


def input_format_session_update(sample_rate: int) -> dict[str, Any]:
    """Build a minimal session update without replacing server VAD settings."""
    return {
        "type": "session.update",
        "session": {
            "audio": {
                "input": {"format": {"type": "audio/pcm", "rate": int(sample_rate)}},
            },
        },
    }


def encode_audio(pcm: bytes) -> str:
    """Encode PCM bytes for ``input_audio_buffer.append``."""
    return base64.b64encode(pcm).decode("ascii")


def decode_audio(delta: str) -> bytes:
    """Decode a Realtime output-audio delta."""
    return base64.b64decode(delta) if delta else b""


def is_session_ready(event: Mapping[str, Any]) -> bool:
    return str(event.get("type") or "") in SESSION_READY_TYPES


def is_response_done(event: Mapping[str, Any]) -> bool:
    return str(event.get("type") or "") == "response.done"


def is_error_event(event: Mapping[str, Any]) -> bool:
    kind = str(event.get("type") or "")
    return kind == "error" or (kind.startswith("session.") and kind.endswith(".failed"))


def parse_output_audio(event: Mapping[str, Any]) -> bytes:
    if str(event.get("type") or "") != "response.output_audio.delta":
        return b""
    return decode_audio(str(event.get("delta") or ""))


def session_output_rate(event: Mapping[str, Any], default: int = DEFAULT_OUTPUT_RATE) -> int:
    session = event.get("session")
    if not isinstance(session, Mapping):
        return default
    audio = session.get("audio")
    if not isinstance(audio, Mapping):
        return default
    output = audio.get("output")
    if not isinstance(output, Mapping):
        return default
    audio_format = output.get("format")
    if not isinstance(audio_format, Mapping):
        return default
    rate = audio_format.get("rate")
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or rate <= 0:
        return default
    return int(rate)


def error_message(event: Mapping[str, Any]) -> str:
    err = event.get("error")
    if isinstance(err, Mapping):
        message = err.get("message") or err.get("code")
        if message:
            return str(message)
    response = event.get("response")
    if isinstance(response, Mapping):
        status_details = response.get("status_details")
        if isinstance(status_details, Mapping):
            response_error = status_details.get("error")
            if isinstance(response_error, Mapping):
                message = response_error.get("message") or response_error.get("code")
                if message:
                    return str(message)
        status = response.get("status")
        if status:
            return f"response status: {status}"
    return json.dumps(dict(event), default=str)[:400]


class OpenAIRealtimeSocket:
    """One OpenAI Realtime WebSocket session."""

    def __init__(
        self,
        url: str,
        *,
        api_key: str = "",
        auth_scheme: str = DEFAULT_AUTH_SCHEME,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
        input_sample_rate: int = 16_000,
        output_sample_rate: int = DEFAULT_OUTPUT_RATE,
        verify_tls: bool = True,
    ):
        self.url = url
        self.api_key = api_key
        self.auth_scheme = auth_scheme
        self.connect_timeout = connect_timeout
        self.input_sample_rate = int(input_sample_rate)
        self.output_sample_rate = int(output_sample_rate)
        self.verify_tls = verify_tls
        self.ws: Any = None
        self.events: list[dict[str, Any]] = []
        self._send_lock = asyncio.Lock()

    async def __aenter__(self) -> OpenAIRealtimeSocket:
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb
        await self.close()

    async def connect(self) -> None:
        headers = auth_headers(self.api_key, auth_scheme=self.auth_scheme)
        kwargs: dict[str, Any] = {"max_size": None, "open_timeout": self.connect_timeout}
        if headers:
            kwargs["additional_headers"] = headers
        if self.url.startswith("wss://") and not self.verify_tls:
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            kwargs["ssl"] = ssl_context
        self.ws = await websockets.connect(self.url, **kwargs)

    async def close(self) -> None:
        ws = self.ws
        self.ws = None
        if ws is None:
            return
        with contextlib.suppress(Exception):
            await ws.close()

    async def send_event(self, payload: Mapping[str, Any]) -> None:
        if self.ws is None:
            raise RuntimeError("realtime socket is not connected")
        async with self._send_lock:
            await self.ws.send(json.dumps(dict(payload)))

    async def send_pcm(self, pcm: bytes) -> None:
        await self.send_event({"type": "input_audio_buffer.append", "audio": encode_audio(pcm)})

    async def recv_event(self, timeout: float | None = None) -> dict[str, Any]:
        if self.ws is None:
            raise RuntimeError("realtime socket is not connected")
        raw = await asyncio.wait_for(self.ws.recv(), timeout=timeout)
        if isinstance(raw, (bytes, bytearray)):
            event = {"type": "_binary", "data": bytes(raw)}
        else:
            decoded = json.loads(raw)
            if not isinstance(decoded, dict):
                raise RealtimeProtocolError(f"expected a JSON object event, got {type(decoded).__name__}")
            event = decoded

        kind = str(event.get("type") or "")
        summary: dict[str, Any] = {"type": kind, "received_at": time.monotonic()}
        if kind == "response.output_audio.delta":
            summary["audio_bytes"] = len(parse_output_audio(event))
        elif kind.endswith("input_audio_transcription.completed"):
            summary["transcript"] = event.get("transcript", "")
        elif kind == "response.done":
            response = event.get("response")
            if isinstance(response, Mapping):
                summary["status"] = response.get("status")
                if response.get("usage") is not None:
                    summary["usage"] = response["usage"]
        if kind == "error" or (kind == "response.done" and summary.get("status") == "failed"):
            summary["error"] = error_message(event)
        self.events.append(summary)
        return event

    def _apply_session_event(self, event: Mapping[str, Any]) -> None:
        if is_session_ready(event):
            self.output_sample_rate = session_output_rate(event, default=self.output_sample_rate)

    def _raise_if_error(self, event: Mapping[str, Any]) -> None:
        if is_error_event(event):
            raise RealtimeProtocolError(error_message(event))

    async def configure_input(
        self,
        sample_rate: int | None = None,
        *,
        timeout: float = DEFAULT_READY_TIMEOUT,
    ) -> dict[str, Any]:
        if sample_rate is not None:
            self.input_sample_rate = int(sample_rate)
        await self.send_event(input_format_session_update(self.input_sample_rate))
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise TimeoutError("timed out waiting for session.updated")
            event = await self.recv_event(timeout=remaining)
            self._raise_if_error(event)
            self._apply_session_event(event)
            if str(event.get("type") or "") == "session.updated":
                return event

    async def cancel_response(self, timeout: float) -> dict[str, Any]:
        """Cancel an active response and wait for its terminal event."""
        await self.send_event({"type": "response.cancel"})
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise TimeoutError("timed out waiting for cancelled response.done")
            event = await self.recv_event(timeout=remaining)
            self._raise_if_error(event)
            self._apply_session_event(event)
            if is_response_done(event):
                return event

    async def recv_audio(self, timeout: float | None = None) -> bytes:
        """Return the next output PCM chunk or signal response completion."""
        deadline = None if timeout is None else asyncio.get_running_loop().time() + timeout
        while True:
            wait = None if deadline is None else max(0.0, deadline - asyncio.get_running_loop().time())
            if wait == 0:
                raise TimeoutError
            try:
                event = await self.recv_event(timeout=wait)
            except ConnectionClosed as exc:
                raise EndOfRealtimeResponse from exc
            if str(event.get("type") or "") == "error":
                raise RealtimeTurnError(error_message(event))
            self._raise_if_error(event)
            self._apply_session_event(event)
            if is_response_done(event):
                response = event.get("response")
                if isinstance(response, Mapping):
                    status = response.get("status")
                    if status and status != "completed":
                        raise RealtimeTurnError(error_message(event), terminal=True)
                raise EndOfRealtimeResponse
            pcm = parse_output_audio(event)
            if pcm:
                return pcm
