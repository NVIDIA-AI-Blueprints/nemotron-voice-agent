#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Serve deterministic private Realtime and response-only artifact fixtures."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import signal
import threading
import time
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed

_RESULT_DISPLAY = "## Verified artifact\n\nThe arbitrary delegated request completed through the normalized backend."
_RESULT_SPEECH = "The delegated result is ready in the display."
_RESULT_ENVELOPE = json.dumps(
    {"schema": "voiceclaw.result.v1", "speech": _RESULT_SPEECH, "display": _RESULT_DISPLAY},
    separators=(",", ":"),
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-port", required=True, type=int)
    parser.add_argument("--backend-port", required=True, type=int)
    parser.add_argument("--bearer-file", required=True, type=Path)
    parser.add_argument("--grant-file", required=True, type=Path)
    parser.add_argument("--expected-query", required=True)
    parser.add_argument("--evidence-file", required=True, type=Path)
    parser.add_argument("--ready-file", required=True, type=Path)
    return parser


class _State:
    """Keep bounded, non-secret observations from the two private fixtures."""

    def __init__(self, *, bearer: str, grant: str, expected_query: str, evidence_file: Path) -> None:
        self.bearer = bearer
        self.expected_query = expected_query
        self.evidence_file = evidence_file
        self.grant = grant
        self.lock = threading.Lock()
        self.values: dict[str, Any] = {
            "schema": "voiceclaw.artifact_fixture.v1",
            "backend_health_authenticated": False,
            "backend_sessions_admitted": 0,
            "backend_turns_received": 0,
            "backend_sessions_deleted": 0,
            "arbitrary_query_observed": False,
            "upstream_connections": 0,
            "model_selected_delegate": False,
            "speech_purposes": [],
        }
        self.write()

    def update(self, **values: Any) -> None:
        """Update evidence and replace its file atomically."""
        with self.lock:
            self.values.update(values)
            self._write_locked()

    def increment(self, name: str) -> None:
        """Increment one integer evidence counter."""
        with self.lock:
            self.values[name] = int(self.values[name]) + 1
            self._write_locked()

    def append(self, name: str, value: str) -> None:
        """Append one unique string observation."""
        with self.lock:
            current = list(self.values[name])
            if value not in current:
                current.append(value)
            self.values[name] = current
            self._write_locked()

    def write(self) -> None:
        """Write the current evidence snapshot."""
        with self.lock:
            self._write_locked()

    def _write_locked(self) -> None:
        self.evidence_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.evidence_file.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(self.values, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        temporary.replace(self.evidence_file)


class _Gateway(ThreadingHTTPServer):
    """Threaded loopback response-only fixture."""

    daemon_threads = True

    def __init__(self, port: int, state: _State) -> None:
        self.fixture_state = state
        super().__init__(("127.0.0.1", port), _GatewayHandler)


class _GatewayHandler(BaseHTTPRequestHandler):
    """Implement only the normalized endpoints consumed by the adapter."""

    server: _Gateway

    def log_message(self, _format: str, *_arguments: object) -> None:
        """Keep credentials and request content out of process logs."""

    def _body(self) -> dict[str, Any]:
        size = int(self.headers.get("Content-Length", "0"))
        if size <= 0 or size > 256 * 1024:
            return {}
        value = json.loads(self.rfile.read(size))
        return value if isinstance(value, dict) else {}

    def _send(self, status: int, body: bytes = b"", *, content_type: str | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        if content_type is not None:
            self.send_header("Content-Type", content_type)
        self.end_headers()
        if body:
            self.wfile.write(body)

    def do_GET(self) -> None:
        """Authenticate the adapter readiness probe."""
        state = self.server.fixture_state
        if self.path != "/healthz" or self.headers.get("Authorization") != f"Bearer {state.bearer}":
            self._send(401)
            return
        state.update(backend_health_authenticated=True)
        self._send(204)

    def do_POST(self) -> None:
        """Admit one session or stream one deterministic result envelope."""
        state = self.server.fixture_state
        body = self._body()
        if self.path == "/v1/voice/sessions":
            if self.headers.get("Authorization") != f"Bearer {state.bearer}":
                self._send(401)
                return
            runtime_conversation_id = body.get("runtimeConversationId")
            if (
                set(body) != {"runtimeConversationId"}
                or not isinstance(runtime_conversation_id, str)
                or not runtime_conversation_id
            ):
                self._send(400)
                return
            state.increment("backend_sessions_admitted")
            response = json.dumps(
                {
                    "voiceSessionId": "artifact-session",
                    "grant": state.grant,
                    "expiresAt": (datetime.now(UTC) + timedelta(minutes=4)).isoformat().replace("+00:00", "Z"),
                },
                separators=(",", ":"),
            ).encode()
            self._send(201, response, content_type="application/json")
            return
        if self.path != "/v1/voice/sessions/artifact-session/turns":
            self._send(404)
            return
        if self.headers.get("Authorization") != f"Bearer {state.grant}":
            self._send(401)
            return
        commit_id = body.get("commitId")
        text = body.get("text")
        if (
            set(body) != {"commitId", "text"}
            or not isinstance(commit_id, str)
            or not commit_id
            or not isinstance(text, str)
        ):
            self._send(400)
            return
        observed = isinstance(text, str) and state.expected_query in text
        state.increment("backend_turns_received")
        state.update(arbitrary_query_observed=observed)
        identity = {
            "voiceSessionId": "artifact-session",
            "turnId": "artifact-turn",
            "responseId": "artifact-response",
        }
        display_start = _RESULT_ENVELOPE.index('"display":"') + len('"display":"')
        display_end = _RESULT_ENVELOPE.rindex('"}')
        midpoint = display_start + ((display_end - display_start) // 2)
        events = (
            {"type": "response.started", **identity},
            {"type": "response.text.delta", **identity, "sequence": 0, "text": _RESULT_ENVELOPE[:midpoint]},
            {"type": "response.text.delta", **identity, "sequence": 1, "text": _RESULT_ENVELOPE[midpoint:]},
            {"type": "response.completed", **identity},
        )
        records = tuple(json.dumps(event, separators=(",", ":")).encode() + b"\n" for event in events)
        self.send_response(200)
        self.send_header("Content-Length", str(sum(len(record) for record in records)))
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        for record in records:
            self.wfile.write(record)
            self.wfile.flush()
            time.sleep(0.01)

    def do_DELETE(self) -> None:
        """Release the short-lived fixture session."""
        state = self.server.fixture_state
        if (
            self.path != "/v1/voice/sessions/artifact-session"
            or self.headers.get("Authorization") != f"Bearer {state.grant}"
        ):
            self._send(401)
            return
        state.increment("backend_sessions_deleted")
        self._send(204)


class _RealtimeFixture:
    """Act as the private model edge while preserving VoiceClaw authority."""

    def __init__(self, state: _State) -> None:
        self.state = state
        self.event_number = 0
        self.response_number = 0
        self.last_user_text: str | None = None

    def _event(self, event_type: str, **values: Any) -> dict[str, Any]:
        self.event_number += 1
        return {"event_id": f"fixture-event-{self.event_number}", "type": event_type, **values}

    async def _send(self, websocket: ServerConnection, event_type: str, **values: Any) -> None:
        await websocket.send(json.dumps(self._event(event_type, **values), separators=(",", ":")))

    async def _session_update(self, websocket: ServerConnection, event: dict[str, Any]) -> None:
        supplied = event.get("session")
        session = dict(supplied) if isinstance(supplied, dict) else {}
        audio = dict(session.get("audio")) if isinstance(session.get("audio"), dict) else {}
        input_audio = dict(audio.get("input")) if isinstance(audio.get("input"), dict) else {}
        input_audio["turn_detection"] = None
        audio["input"] = input_audio
        session["audio"] = audio
        session.update({"id": "fixture-private-session", "object": "realtime.session", "model": "fixture-model"})
        await self._send(websocket, "session.updated", session=session)

    async def _select_delegate(self, websocket: ServerConnection) -> None:
        if self.last_user_text is None:
            raise RuntimeError("selector response arrived before a finalized user turn")
        self.response_number += 1
        response_id = f"fixture-selector-{self.response_number}"
        item_id = f"fixture-call-item-{self.response_number}"
        call_id = f"fixture-call-{self.response_number}"
        arguments = json.dumps({"goal": self.last_user_text}, separators=(",", ":"))
        item = {
            "id": item_id,
            "object": "realtime.item",
            "type": "function_call",
            "status": "in_progress",
            "name": "voiceclaw_work_delegate",
            "call_id": call_id,
            "arguments": arguments,
        }
        await self._send(
            websocket,
            "response.created",
            response={
                "id": response_id,
                "object": "realtime.response",
                "status": "in_progress",
                "conversation_id": "fixture-private-conversation",
                "output": [],
                "metadata": {},
            },
        )
        await self._send(
            websocket,
            "response.output_item.added",
            response_id=response_id,
            output_index=0,
            item=item,
        )
        completed = {**item, "status": "completed"}
        await self._send(
            websocket,
            "response.output_item.done",
            response_id=response_id,
            output_index=0,
            item=completed,
        )
        await self._send(
            websocket,
            "response.done",
            response={
                "id": response_id,
                "object": "realtime.response",
                "status": "completed",
                "conversation_id": "fixture-private-conversation",
                "output": [completed],
                "metadata": {},
            },
        )
        self.state.update(model_selected_delegate=True)

    @staticmethod
    def _response_context(event: dict[str, Any]) -> dict[str, Any] | None:
        response = event.get("response")
        instructions = response.get("instructions") if isinstance(response, dict) else None
        if not isinstance(instructions, str):
            return None
        opening = "<voiceclaw_response_context>\n"
        closing = "\n</voiceclaw_response_context>"
        if opening not in instructions or closing not in instructions:
            return None
        value = instructions.split(opening, 1)[1].split(closing, 1)[0]
        decoded = json.loads(value)
        return decoded if isinstance(decoded, dict) else None

    async def _speak(self, websocket: ServerConnection, *, purpose: str, payload: str) -> None:
        self.response_number += 1
        response_id = f"fixture-speech-{self.response_number}"
        item_id = f"fixture-speech-item-{self.response_number}"
        transcript = (
            "I’m working on that request now."
            if purpose == "delegation_ack"
            else payload.strip() or "The delegated result is ready."
        )
        item = {
            "id": item_id,
            "object": "realtime.item",
            "type": "message",
            "status": "in_progress",
            "role": "assistant",
            "content": [],
        }
        await self._send(
            websocket,
            "response.created",
            response={
                "id": response_id,
                "object": "realtime.response",
                "status": "in_progress",
                "conversation_id": "fixture-private-conversation",
                "output": [],
                "metadata": {},
            },
        )
        await self._send(
            websocket,
            "response.output_item.added",
            response_id=response_id,
            output_index=0,
            item=item,
        )
        fields = {
            "response_id": response_id,
            "item_id": item_id,
            "output_index": 0,
            "content_index": 0,
        }
        await self._send(
            websocket,
            "response.content_part.added",
            **fields,
            part={"type": "audio", "transcript": ""},
        )
        await self._send(
            websocket,
            "response.output_audio.delta",
            **fields,
            delta=base64.b64encode(b"\x00" * 4_800).decode("ascii"),
        )
        await self._send(websocket, "response.output_audio_transcript.delta", **fields, delta=transcript)
        await self._send(websocket, "response.output_audio.done", **fields)
        await self._send(websocket, "response.output_audio_transcript.done", **fields, transcript=transcript)
        await self._send(
            websocket,
            "response.content_part.done",
            **fields,
            part={"type": "audio", "transcript": transcript},
        )
        completed = {
            **item,
            "status": "completed",
            "content": [{"type": "output_audio", "transcript": transcript}],
        }
        await self._send(
            websocket,
            "response.output_item.done",
            response_id=response_id,
            output_index=0,
            item=completed,
        )
        await self._send(
            websocket,
            "response.done",
            response={
                "id": response_id,
                "object": "realtime.response",
                "status": "completed",
                "conversation_id": "fixture-private-conversation",
                "output": [completed],
                "metadata": {},
            },
        )
        self.state.append("speech_purposes", purpose)

    async def __call__(self, websocket: ServerConnection) -> None:
        """Serve one private Realtime connection."""
        self.state.increment("upstream_connections")
        await self._send(
            websocket,
            "session.created",
            session={
                "id": "fixture-private-session",
                "object": "realtime.session",
                "model": "fixture-model",
                "audio": {"input": {"turn_detection": None}},
            },
        )
        await self._send(
            websocket,
            "conversation.created",
            conversation={"id": "fixture-private-conversation", "object": "realtime.conversation"},
        )
        try:
            async for raw in websocket:
                if not isinstance(raw, str):
                    continue
                event = json.loads(raw)
                if not isinstance(event, dict):
                    continue
                event_type = event.get("type")
                if event_type == "session.update":
                    await self._session_update(websocket, event)
                    continue
                if event_type == "conversation.item.create":
                    item = event.get("item")
                    if isinstance(item, dict) and item.get("role") == "user":
                        for part in item.get("content", []):
                            if isinstance(part, dict) and part.get("type") == "input_text":
                                text = part.get("text")
                                if isinstance(text, str) and text.strip():
                                    self.last_user_text = text
                    continue
                if event_type == "conversation.item.truncate":
                    await self._send(
                        websocket,
                        "conversation.item.truncated",
                        item_id=event.get("item_id"),
                        content_index=event.get("content_index"),
                        audio_end_ms=event.get("audio_end_ms"),
                    )
                    continue
                if event_type != "response.create":
                    continue
                context = self._response_context(event)
                if context is None:
                    await self._select_delegate(websocket)
                    continue
                purpose = context.get("response_purpose")
                payload = context.get("payload_text")
                if not isinstance(purpose, str) or not isinstance(payload, str):
                    raise RuntimeError("VoiceClaw response context is malformed")
                await self._speak(websocket, purpose=purpose, payload=payload)
        except ConnectionClosed:
            return


async def _run(arguments: argparse.Namespace) -> None:
    bearer = arguments.bearer_file.read_text(encoding="ascii").strip()
    grant = arguments.grant_file.read_text(encoding="ascii").strip()
    if len(bearer) < 32 or len(grant) < 32 or bearer == grant:
        raise ValueError("fixture credentials must be distinct and at least 32 characters")
    state = _State(
        bearer=bearer,
        grant=grant,
        expected_query=arguments.expected_query,
        evidence_file=arguments.evidence_file,
    )
    gateway = _Gateway(arguments.backend_port, state)
    gateway_thread = threading.Thread(target=gateway.serve_forever, daemon=True)
    gateway_thread.start()
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, stop.set)
    try:
        async with serve(_RealtimeFixture(state), "127.0.0.1", arguments.upstream_port, max_size=16 * 1024 * 1024):
            arguments.ready_file.parent.mkdir(parents=True, exist_ok=True)
            arguments.ready_file.write_text("ready\n", encoding="ascii")
            await stop.wait()
    finally:
        gateway.shutdown()
        gateway.server_close()
        gateway_thread.join(timeout=5)
        state.write()


def main() -> int:
    """Run both private fixture services until terminated."""
    arguments = _parser().parse_args()
    if not all(1 <= value <= 65535 for value in (arguments.upstream_port, arguments.backend_port)):
        raise SystemExit("fixture ports must be from 1 through 65535")
    if arguments.expected_query != arguments.expected_query.strip() or not arguments.expected_query:
        raise SystemExit("expected query must be non-empty and unpadded")
    try:
        asyncio.run(_run(arguments))
    except (OSError, UnicodeError, ValueError) as error:
        raise SystemExit(f"artifact fixture failed: {error}") from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
