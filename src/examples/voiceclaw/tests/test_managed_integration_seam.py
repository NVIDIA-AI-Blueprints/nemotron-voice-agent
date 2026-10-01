# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D101, D102, D103

from __future__ import annotations

import asyncio
import json
import threading
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import httpx

from voiceclaw.adapters.nemoclaw.managed_projection import (
    MANAGED_LISTENER_PORT,
    ManagedProjectionPaths,
    load_managed_projection,
    materialize_managed_config,
)
from voiceclaw.composition import compose_backends
from voiceclaw.domain.models import BackendOperation, CapabilitySource, Durability, EventDelivery
from voiceclaw.domain.response_only import RESULT_ENVELOPE_SCHEMA
from voiceclaw.ports.runtime import SessionSnapshot
from voiceclaw.ports.turns import CommittedTurnRequest, CommittedTurnResult
from voiceclaw.server import create_app

_SPEECH_CREDENTIAL = "nvapi-managed-" + "s" * 40
_FIRST_AGENT_CREDENTIAL = "nemoclaw-agent-" + "a" * 40
_SECOND_AGENT_CREDENTIAL = "nemoclaw-agent-" + "b" * 40
_SESSION_GRANT = "nemoclaw-session-" + "g" * 40
_SESSION_ID = "voice-session-managed"
_TURN_ID = "turn-managed"
_RESPONSE_ID = "response-managed"
_DISPLAY_TEXT = "## Managed result\n\nThe composed consumer seam completed."
_SPEECH_TEXT = "The managed result is ready."


class _ReadinessGateway(BaseHTTPRequestHandler):
    requests: list[tuple[str | None, str | None]] = []

    def do_GET(self) -> None:
        if self.path != "/healthz":
            self.send_response(404)
            self.end_headers()
            return
        type(self).requests.append(
            (
                self.headers.get("Host"),
                self.headers.get("X-NemoClaw-Authorization"),
            )
        )
        self.send_response(204)
        self.end_headers()

    def log_message(self, _format: str, *_args: object) -> None:
        return None


class _CommittedTurnGateway(BaseHTTPRequestHandler):
    requests: list[dict[str, Any]] = []

    def log_message(self, _format: str, *_args: object) -> None:
        return None

    def _body(self) -> object | None:
        size = int(self.headers.get("Content-Length", "0"))
        return None if size == 0 else json.loads(self.rfile.read(size))

    def _record(self, body: object | None) -> None:
        type(self).requests.append(
            {
                "method": self.command,
                "path": self.path,
                "host": self.headers.get("Host"),
                "authorization": self.headers.get("Authorization"),
                "nemoclaw_authorization": self.headers.get("X-NemoClaw-Authorization"),
                "body": body,
            }
        )

    def _send(self, status: int, body: bytes = b"", *, content_type: str | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        if content_type is not None:
            self.send_header("Content-Type", content_type)
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _authorized(self, credential: str) -> bool:
        return (
            self.headers.get("X-NemoClaw-Authorization") == f"Bearer {credential}"
            and self.headers.get("Authorization") is None
        )

    def do_GET(self) -> None:
        self._record(None)
        if self.path != "/healthz" or not self._authorized(_FIRST_AGENT_CREDENTIAL):
            self._send(401)
            return
        self._send(204)

    def do_POST(self) -> None:
        body = self._body()
        self._record(body)
        if self.path == "/v1/voice/sessions":
            if not self._authorized(_SECOND_AGENT_CREDENTIAL):
                self._send(401)
                return
            expires_at = (datetime.now(UTC) + timedelta(minutes=5)).isoformat().replace("+00:00", "Z")
            payload = json.dumps(
                {"voiceSessionId": _SESSION_ID, "grant": _SESSION_GRANT, "expiresAt": expires_at},
                separators=(",", ":"),
            ).encode()
            self._send(201, payload, content_type="application/json")
            return
        if self.path != f"/v1/voice/sessions/{_SESSION_ID}/turns" or not self._authorized(_SESSION_GRANT):
            self._send(401)
            return
        envelope = json.dumps(
            {"schema": RESULT_ENVELOPE_SCHEMA, "speech": _SPEECH_TEXT, "display": _DISPLAY_TEXT},
            separators=(",", ":"),
        )
        midpoint = len(envelope) // 2
        identity = {"voiceSessionId": _SESSION_ID, "turnId": _TURN_ID, "responseId": _RESPONSE_ID}
        events = [
            {"type": "response.started", **identity},
            {"type": "response.text.delta", **identity, "sequence": 0, "text": envelope[:midpoint]},
            {"type": "response.text.delta", **identity, "sequence": 1, "text": envelope[midpoint:]},
            {"type": "response.completed", **identity},
        ]
        payload = b"".join(json.dumps(event, separators=(",", ":")).encode() + b"\n" for event in events)
        self._send(200, payload, content_type="application/x-ndjson")

    def do_DELETE(self) -> None:
        self._record(None)
        if self.path != f"/v1/voice/sessions/{_SESSION_ID}" or not self._authorized(_SESSION_GRANT):
            self._send(401)
            return
        self._send(204)


def _write_protected(path: Path, payload: str) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    temporary = path.with_name(f".{path.name}.next")
    temporary.write_text(payload, encoding="ascii")
    temporary.chmod(0o600)
    temporary.replace(path)


def test_managed_projection_composition_and_readiness_use_route_and_rotated_credential(tmp_path: Path) -> None:
    gateway = ThreadingHTTPServer(("127.0.0.1", 0), _ReadinessGateway)
    gateway_thread = threading.Thread(target=gateway.serve_forever, daemon=True)
    gateway_thread.start()
    _ReadinessGateway.requests = []
    gateway_port = gateway.server_address[1]
    route_host = f"agent.sandbox.openshell.localhost:{gateway_port}"

    paths = ManagedProjectionPaths(root=tmp_path / "projection")
    paths.root.mkdir(mode=0o700)
    paths.root.chmod(0o700)
    projection_document = {
        "integration": "voiceclaw",
        "sandbox": "sandbox",
        "agent": "agent",
        "port": MANAGED_LISTENER_PORT,
        "speechProvider": "nvidia",
        "speechCredentialPath": str(paths.speech_credential),
        "agentCredentialPath": str(paths.agent_credential),
        "agentEndpoint": f"http://127.0.0.1:{gateway_port}",
        "agentRouteHost": route_host,
    }
    _write_protected(paths.configuration, json.dumps(projection_document, separators=(",", ":")))
    _write_protected(paths.speech_credential, _SPEECH_CREDENTIAL)
    _write_protected(paths.agent_credential, _FIRST_AGENT_CREDENTIAL)

    config_path = tmp_path / "runtime" / "voiceclaw.yaml"
    config = materialize_managed_config(load_managed_projection(paths), config_path)
    composition = compose_backends(config, environ={})
    app = create_app(
        config,
        environ={"REALTIME_UPSTREAM_API_KEY": "voiceclaw-managed-internal-upstream-key"},
        composition=composition,
    )

    async def exercise() -> tuple[httpx.Response, httpx.Response]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://voiceclaw") as client:
            first = await client.get("/readyz")
            _write_protected(paths.agent_credential, _SECOND_AGENT_CREDENTIAL)
            second = await client.get("/readyz")
            return first, second

    try:
        first, second = asyncio.run(exercise())
    finally:
        app.state.voiceclaw_state_store.close()
        gateway.shutdown()
        gateway.server_close()
        gateway_thread.join(timeout=1)

    assert first.status_code == 200
    assert second.status_code == 200
    assert _ReadinessGateway.requests == [
        (route_host, f"Bearer {_FIRST_AGENT_CREDENTIAL}"),
        (route_host, f"Bearer {_SECOND_AGENT_CREDENTIAL}"),
    ]


def test_managed_composed_consumer_seam_and_public_capabilities(tmp_path: Path) -> None:
    gateway = ThreadingHTTPServer(("127.0.0.1", 0), _CommittedTurnGateway)
    gateway_thread = threading.Thread(target=gateway.serve_forever, daemon=True)
    gateway_thread.start()
    _CommittedTurnGateway.requests = []
    gateway_port = gateway.server_address[1]
    route_host = f"agent.sandbox.openshell.localhost:{gateway_port}"

    paths = ManagedProjectionPaths(root=tmp_path / "projection")
    paths.root.mkdir(mode=0o700)
    paths.root.chmod(0o700)
    projection_document = {
        "integration": "voiceclaw",
        "sandbox": "sandbox",
        "agent": "agent",
        "port": MANAGED_LISTENER_PORT,
        "speechProvider": "nvidia",
        "speechCredentialPath": str(paths.speech_credential),
        "agentCredentialPath": str(paths.agent_credential),
        "agentEndpoint": f"http://127.0.0.1:{gateway_port}",
        "agentRouteHost": route_host,
    }
    _write_protected(paths.configuration, json.dumps(projection_document, separators=(",", ":")))
    _write_protected(paths.speech_credential, _SPEECH_CREDENTIAL)
    _write_protected(paths.agent_credential, _FIRST_AGENT_CREDENTIAL)

    config = materialize_managed_config(
        load_managed_projection(paths),
        tmp_path / "runtime" / "voiceclaw.yaml",
    )
    composition = compose_backends(config, environ={})
    app = create_app(
        config,
        environ={"REALTIME_UPSTREAM_API_KEY": "voiceclaw-managed-internal-upstream-key"},
        composition=composition,
    )

    async def exercise() -> tuple[SessionSnapshot, CommittedTurnResult]:
        snapshot = await app.state.voiceclaw_runtime.open_session("session-managed", "conversation-managed")
        _write_protected(paths.agent_credential, _SECOND_AGENT_CREDENTIAL)
        assert composition.turn_backend is not None
        result = await composition.turn_backend.commit_turn(
            CommittedTurnRequest(
                runtime_conversation_id="conversation-managed",
                commit_id="commit-managed",
                text="Explain the managed consumer seam.",
            )
        )
        await app.state.voiceclaw_runtime.close_session("session-managed", "test_complete")
        return snapshot, result

    try:
        snapshot, result = asyncio.run(exercise())
    finally:
        asyncio.run(composition.shutdown())
        app.state.voiceclaw_state_store.close()
        gateway.shutdown()
        gateway.server_close()
        gateway_thread.join(timeout=1)

    assert result.backend_session_id == _SESSION_ID
    assert result.turn_id == _TURN_ID
    assert result.response_id == _RESPONSE_ID
    assert result.display_text == _DISPLAY_TEXT
    assert result.speak_text == _SPEECH_TEXT

    assert snapshot.gateway_reachable is True
    assert snapshot.capabilities == (BackendOperation.SUBMIT.value,)
    assert snapshot.durability is Durability.NONE
    assert snapshot.event_delivery is EventDelivery.RESPONSE_ONLY
    assert snapshot.recoverable_inflight is False
    assert snapshot.max_parallel_work == 1
    assert snapshot.capability_source is CapabilitySource.COMPATIBILITY_PROJECTION
    assert tuple(tool.name for tool in snapshot.frontend_tools) == ("work.delegate",)
    public_projection = json.loads(snapshot.projection)
    assert public_projection["contract"]["operations"] == [BackendOperation.SUBMIT.value]
    assert public_projection["contract"]["durability"] == Durability.NONE.value
    assert public_projection["contract"]["event_delivery"] == EventDelivery.RESPONSE_ONLY.value
    assert public_projection["contract"]["sessionful"] is False
    assert public_projection["contract"]["supports_parallel_work"] is False
    assert public_projection["contract"]["max_parallel_requests"] == 1
    assert public_projection["connection"]["attachment_mode"] == "ephemeral_per_request"
    assert public_projection["connection"]["resume_supported"] is False

    requests = _CommittedTurnGateway.requests
    assert [(request["method"], request["path"]) for request in requests] == [
        ("GET", "/healthz"),
        ("POST", "/v1/voice/sessions"),
        ("POST", f"/v1/voice/sessions/{_SESSION_ID}/turns"),
        ("DELETE", f"/v1/voice/sessions/{_SESSION_ID}"),
    ]
    assert all(request["host"] == route_host for request in requests)
    assert all(request["authorization"] is None for request in requests)
    assert [request["nemoclaw_authorization"] for request in requests] == [
        f"Bearer {_FIRST_AGENT_CREDENTIAL}",
        f"Bearer {_SECOND_AGENT_CREDENTIAL}",
        f"Bearer {_SESSION_GRANT}",
        f"Bearer {_SESSION_GRANT}",
    ]
    assert requests[1]["body"] == {"runtimeConversationId": "conversation-managed"}
    turn_body = requests[2]["body"]
    assert isinstance(turn_body, dict)
    assert turn_body["commitId"] == "commit-managed"
    assert "Explain the managed consumer seam." in turn_body["text"]
    assert RESULT_ENVELOPE_SCHEMA in turn_body["text"]
