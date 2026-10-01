# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D101, D102, D103, D107

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from voiceclaw.adapters.nemoclaw.committed_turn import (
    DEFAULT_RESULT_DISPLAY_BUDGET_BYTES,
    DeploymentAuthorization,
    EndpointPolicy,
    NemoClawBackendFailure,
    NemoClawCommittedTurnAdapter,
    NemoClawEndpointError,
    NemoClawProtocolError,
    NemoClawRequestRejected,
    NemoClawRequestValidationError,
    NemoClawTransportError,
)
from voiceclaw.adapters.result_envelope import build_result_envelope_prompt
from voiceclaw.adapters.state.sqlite import SqliteStateStore
from voiceclaw.application.runtime import RealtimeInteractionManager
from voiceclaw.domain.capabilities import CapabilityToolRegistry
from voiceclaw.domain.models import (
    BackendCapabilities,
    BackendOperation,
    CapabilityEvidence,
    CapabilitySource,
    Durability,
    EventDelivery,
)
from voiceclaw.domain.response_only import RESULT_ENVELOPE_SCHEMA
from voiceclaw.interaction_profiles import (
    SessionScope,
    WorkCardinality,
    load_interaction_profile_catalog,
)
from voiceclaw.ports.readiness import SelectedAgentReadinessError
from voiceclaw.ports.turns import (
    MAX_COMMITTED_TURN_GOAL_BYTES,
    CommittedTurnCompleted,
    CommittedTurnDisplayDelta,
    CommittedTurnRequest,
)

_DEPLOYMENT_BEARER = "deployment-bearer-kept-on-server"
_DISPLAY_TEXT = "## Test command\n\nRun `npm test`."
_SPEECH_TEXT = "Run npm test."
_RESULT_ENVELOPE = json.dumps(
    {"schema": RESULT_ENVELOPE_SCHEMA, "speech": _SPEECH_TEXT, "display": _DISPLAY_TEXT},
    separators=(",", ":"),
)


@dataclass
class _RecordedRequest:
    method: str
    path: str
    authorization: str | None
    nemoclaw_authorization: str | None
    host: str | None
    body: object | None


@dataclass
class _GatewayState:
    mode: str = "success"
    delete_status: int = 204
    requests: list[_RecordedRequest] = field(default_factory=list)
    sessions: dict[str, str] = field(default_factory=dict)
    session_expiries: dict[str, float] = field(default_factory=dict)
    admissions: int = 0
    managed: bool = False
    deployment_bearer: str = _DEPLOYMENT_BEARER
    session_lifetime: timedelta = timedelta(minutes=5)
    admission_headers_sent: threading.Event = field(default_factory=threading.Event)

    def events(self, session_id: str) -> list[dict[str, Any]]:
        identity = {
            "voiceSessionId": session_id,
            "turnId": f"turn-{self.admissions}",
            "responseId": f"response-{self.admissions}",
        }
        if self.mode == "gap":
            return [
                {"type": "response.started", **identity},
                {"type": "response.text.delta", **identity, "sequence": 1, "text": "wrong"},
                {"type": "response.completed", **identity},
            ]
        if self.mode == "failed":
            return [{"type": "response.failed", **identity, "reason": "agent_failed"}]
        if self.mode == "truncated_after_start":
            return [{"type": "response.started", **identity}]
        if self.mode == "failed_after_display":
            split = _RESULT_ENVELOPE.index("## Test") + len("## Test")
            return [
                {"type": "response.started", **identity},
                {"type": "response.text.delta", **identity, "sequence": 0, "text": _RESULT_ENVELOPE[:split]},
                {"type": "response.failed", **identity, "reason": "agent_failed"},
            ]
        if self.mode == "oversized":
            oversized_envelope = json.dumps(
                {"schema": RESULT_ENVELOPE_SCHEMA, "speech": None, "display": "x" * 128_001},
                separators=(",", ":"),
            )
            return [
                {"type": "response.started", **identity},
                {"type": "response.text.delta", **identity, "sequence": 0, "text": oversized_envelope},
                {"type": "response.completed", **identity},
            ]
        if self.mode in {"empty", "whitespace"}:
            events = [{"type": "response.started", **identity}]
            if self.mode == "whitespace":
                events.append({"type": "response.text.delta", **identity, "sequence": 0, "text": " \n\t "})
            events.append({"type": "response.completed", **identity})
            return events
        if self.mode == "plain_text":
            return [
                {"type": "response.started", **identity},
                {"type": "response.text.delta", **identity, "sequence": 0, "text": "npm test passed"},
                {"type": "response.completed", **identity},
            ]
        if self.mode == "outer_json_expansion":
            result_envelope = json.dumps(
                {"schema": RESULT_ENVELOPE_SCHEMA, "speech": None, "display": "\x01" * 30_000},
                separators=(",", ":"),
            )
            return [
                {"type": "response.started", **identity},
                {"type": "response.text.delta", **identity, "sequence": 0, "text": result_envelope},
                {"type": "response.completed", **identity},
            ]
        result_envelope = (
            json.dumps(
                {"schema": RESULT_ENVELOPE_SCHEMA, "speech": None, "display": " \t"},
                separators=(",", ":"),
            )
            if self.mode == "malformed_envelope"
            else _RESULT_ENVELOPE
        )
        if self.mode == "duplicate_member_replay":
            members = _RESULT_ENVELOPE[1:-1]
            result_envelope = "{" + members + "," + members + "}"
        midpoint = (
            result_envelope.index("## Test") + len("## Test")
            if self.mode == "stream_pause"
            else len(result_envelope) // 2
        )
        return [
            {"type": "response.started", **identity},
            {
                "type": "response.text.delta",
                **identity,
                "sequence": 0,
                "text": result_envelope[:midpoint],
            },
            {
                "type": "response.text.delta",
                **identity,
                "sequence": 1,
                "text": result_envelope[midpoint:],
            },
            {"type": "response.completed", **identity},
        ]


class _GatewayServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, state: _GatewayState) -> None:
        self.state = state
        super().__init__(("127.0.0.1", 0), _GatewayHandler)


class _GatewayHandler(BaseHTTPRequestHandler):
    server: _GatewayServer

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def _body(self) -> object | None:
        size = int(self.headers.get("Content-Length", "0"))
        if size == 0:
            return None
        return json.loads(self.rfile.read(size))

    def _record(self, body: object | None) -> None:
        self.server.state.requests.append(
            _RecordedRequest(
                method=self.command,
                path=self.path,
                authorization=self.headers.get("Authorization"),
                nemoclaw_authorization=self.headers.get("X-NemoClaw-Authorization"),
                host=self.headers.get("Host"),
                body=body,
            )
        )

    def _authorized(self, bearer: str) -> bool:
        if self.server.state.managed:
            return (
                self.headers.get("X-NemoClaw-Authorization") == f"Bearer {bearer}"
                and self.headers.get("Authorization") is None
            )
        return self.headers.get("Authorization") == f"Bearer {bearer}"

    def _send(self, status: int, body: bytes = b"", *, content_type: str | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        if content_type is not None:
            self.send_header("Content-Type", content_type)
        self.end_headers()
        if body:
            self.wfile.write(body)

    def do_GET(self) -> None:
        self._record(None)
        if self.path != "/healthz":
            self._send(404)
            return
        if not self._authorized(self.server.state.deployment_bearer):
            self._send(401, b'{"error":"authentication_failed"}', content_type="application/json")
            return
        self._send(204)

    def do_POST(self) -> None:
        body = self._body()
        self._record(body)
        state = self.server.state
        if self.path == "/v1/voice/sessions":
            if state.mode == "redirect":
                self.send_response(307)
                self.send_header("Location", "/must-not-follow")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if not self._authorized(state.deployment_bearer):
                self._send(401, b'{"error":"authentication_failed"}', content_type="application/json")
                return
            state.admissions += 1
            session_id = f"session-{state.admissions}"
            grant = f"private-session-grant-{state.admissions}"
            state.sessions[session_id] = grant
            expiry_offset = {
                "expired_admission": timedelta(seconds=-1),
                "far_future_admission": timedelta(minutes=5, seconds=6),
            }.get(state.mode, state.session_lifetime)
            state.session_expiries[session_id] = time.monotonic() + expiry_offset.total_seconds()
            expires_at = (datetime.now(UTC) + expiry_offset).isoformat().replace("+00:00", "Z")
            payload = json.dumps(
                {
                    "voiceSessionId": session_id,
                    "grant": grant,
                    "expiresAt": expires_at,
                },
                separators=(",", ":"),
            ).encode()
            content_type = "text/plain" if state.mode == "bad_admission_content_type" else "application/json"
            if state.mode == "delayed_admission_body":
                self.send_response(201)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Content-Type", content_type)
                self.end_headers()
                self.wfile.flush()
                state.admission_headers_sent.set()
                time.sleep(0.2)
                try:
                    self.wfile.write(payload)
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self._send(201, payload, content_type=content_type)
            return

        parts = self.path.split("/")
        if len(parts) == 6 and parts[:4] == ["", "v1", "voice", "sessions"] and parts[5] == "turns":
            session_id = parts[4]
            if not self._authorized(str(state.sessions.get(session_id))):
                self._send(401, b'{"error":"authentication_failed"}', content_type="application/json")
                return
            payload = b"".join(
                json.dumps(event, separators=(",", ":")).encode() + b"\n" for event in state.events(session_id)
            )
            if state.mode == "stream_pause":
                first_events = state.events(session_id)[:2]
                first_payload = b"".join(
                    json.dumps(event, separators=(",", ":")).encode() + b"\n" for event in first_events
                )
                remaining_payload = payload[len(first_payload) :]
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(first_payload)
                self.wfile.flush()
                time.sleep(5)
                try:
                    self.wfile.write(remaining_payload)
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return
            if state.mode == "slow_trickle":
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.send_header("Connection", "close")
                self.end_headers()
                for byte in payload:
                    try:
                        self.wfile.write(bytes((byte,)))
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        return
                    time.sleep(0.01)
                return
            self._send(200, payload, content_type="application/x-ndjson")
            return
        self._send(404)

    def do_DELETE(self) -> None:
        body = self._body()
        self._record(body)
        state = self.server.state
        session_id = self.path.rsplit("/", 1)[-1]
        if not self._authorized(str(state.sessions.get(session_id))):
            self._send(401)
            return
        if time.monotonic() >= state.session_expiries.get(session_id, 0.0):
            self._send(401)
            return
        self._send(state.delete_status)


@contextmanager
def _gateway(
    *,
    mode: str = "success",
    delete_status: int = 204,
    managed: bool = False,
    session_lifetime: timedelta = timedelta(minutes=5),
):
    state = _GatewayState(
        mode=mode,
        delete_status=delete_status,
        managed=managed,
        session_lifetime=session_lifetime,
    )
    server = _GatewayServer(state)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield state, f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _adapter(origin: str) -> NemoClawCommittedTurnAdapter:
    return NemoClawCommittedTurnAdapter(origin=origin, deployment_bearer=_DEPLOYMENT_BEARER)


def _request(
    commit_id: str = "commit-1",
    *,
    text: str = "Which command runs the tests?",
) -> CommittedTurnRequest:
    return CommittedTurnRequest(
        runtime_conversation_id="runtime-conversation-1",
        commit_id=commit_id,
        text=text,
    )


def test_one_fresh_session_per_turn_and_runtime_conversation_can_be_reused() -> None:
    with _gateway() as (state, origin):
        adapter = _adapter(origin)
        first = asyncio.run(adapter.commit_turn(_request("commit-1")))
        second = asyncio.run(adapter.commit_turn(_request("commit-2")))

    assert first.display_text == second.display_text == _DISPLAY_TEXT
    assert first.speak_text == second.speak_text == _SPEECH_TEXT
    assert (first.backend_session_id, second.backend_session_id) == ("session-1", "session-2")
    assert (first.turn_id, first.response_id) == ("turn-1", "response-1")
    assert (second.turn_id, second.response_id) == ("turn-2", "response-2")
    assert [(item.method, item.path) for item in state.requests] == [
        ("POST", "/v1/voice/sessions"),
        ("POST", "/v1/voice/sessions/session-1/turns"),
        ("DELETE", "/v1/voice/sessions/session-1"),
        ("POST", "/v1/voice/sessions"),
        ("POST", "/v1/voice/sessions/session-2/turns"),
        ("DELETE", "/v1/voice/sessions/session-2"),
    ]
    admissions = [item for item in state.requests if item.path == "/v1/voice/sessions"]
    assert [item.body for item in admissions] == [
        {"runtimeConversationId": "runtime-conversation-1"},
        {"runtimeConversationId": "runtime-conversation-1"},
    ]
    turns = [item for item in state.requests if item.path.endswith("/turns")]
    assert [item.body for item in turns] == [
        {
            "commitId": "commit-1",
            "text": build_result_envelope_prompt(
                "Which command runs the tests?",
                display_budget_bytes=DEFAULT_RESULT_DISPLAY_BUDGET_BYTES,
            ),
        },
        {
            "commitId": "commit-2",
            "text": build_result_envelope_prompt(
                "Which command runs the tests?",
                display_budget_bytes=DEFAULT_RESULT_DISPLAY_BUDGET_BYTES,
            ),
        },
    ]
    assert admissions[0].authorization == f"Bearer {_DEPLOYMENT_BEARER}"
    assert turns[0].authorization == "Bearer private-session-grant-1"


def test_committed_goal_is_wrapped_without_trimming_or_rewriting() -> None:
    goal = '  Preserve "quotes", newlines\n,and Unicode ☃ exactly.  '
    with _gateway() as (state, origin):
        asyncio.run(_adapter(origin).commit_turn(_request(text=goal)))

    turn = next(item for item in state.requests if item.path.endswith("/turns"))
    assert turn.body == {
        "commitId": "commit-1",
        "text": build_result_envelope_prompt(
            goal,
            display_budget_bytes=DEFAULT_RESULT_DISPLAY_BUDGET_BYTES,
        ),
    }
    encoded_goal = turn.body["text"].rsplit("user_goal_json=", 1)[1].splitlines()[0]
    assert json.loads(encoded_goal) == goal


def test_wrapped_turn_must_fit_the_backend_text_limit() -> None:
    one_character_contract_size = len(
        build_result_envelope_prompt(
            "a",
            display_budget_bytes=DEFAULT_RESULT_DISPLAY_BUDGET_BYTES,
        ).encode("utf-8")
    )
    exact_goal = "a" * (MAX_COMMITTED_TURN_GOAL_BYTES - one_character_contract_size + 1)

    adapter = _adapter("http://127.0.0.1:18800")
    wrapped = adapter._validate_request(
        _request(text=exact_goal),
        display_budget_bytes=DEFAULT_RESULT_DISPLAY_BUDGET_BYTES,
    )

    assert len(wrapped.encode("utf-8")) == MAX_COMMITTED_TURN_GOAL_BYTES
    with pytest.raises(NemoClawRequestValidationError, match="turn_payload_too_large"):
        adapter._validate_request(
            _request(text=exact_goal + "a"),
            display_budget_bytes=DEFAULT_RESULT_DISPLAY_BUDGET_BYTES,
        )


def test_goal_over_public_limit_is_rejected_before_wire_wrapping() -> None:
    request = _request(text="a" * (MAX_COMMITTED_TURN_GOAL_BYTES + 1))

    with pytest.raises(NemoClawRequestValidationError, match="goal_too_large"):
        _adapter("http://127.0.0.1:18800")._validate_request(request)


def test_stream_turn_emits_provisional_display_then_one_terminal_result() -> None:
    async def collect(adapter: NemoClawCommittedTurnAdapter):
        return [event async for event in adapter.stream_turn(_request())]

    with _gateway() as (_state, origin):
        events = asyncio.run(collect(_adapter(origin)))

    delta_events = [event for event in events if isinstance(event, CommittedTurnDisplayDelta)]
    terminals = [event for event in events if isinstance(event, CommittedTurnCompleted)]
    assert "".join(event.delta for event in delta_events) == _DISPLAY_TEXT
    assert [event.sequence for event in delta_events] == list(range(len(delta_events)))
    assert {(event.backend_session_id, event.turn_id, event.response_id) for event in delta_events} == {
        ("session-1", "turn-1", "response-1")
    }
    assert len(terminals) == 1
    assert events[-1] is terminals[0]
    assert terminals[0].result.display_text == _DISPLAY_TEXT
    assert terminals[0].result.speak_text == _SPEECH_TEXT


def test_configured_display_budget_is_enforced_on_the_received_envelope() -> None:
    exact_budget = len(_DISPLAY_TEXT.encode("utf-8"))
    with _gateway() as (_state, origin):
        adapter = NemoClawCommittedTurnAdapter(
            origin=origin,
            deployment_bearer=_DEPLOYMENT_BEARER,
            result_display_budget_bytes=exact_budget,
        )
        result = asyncio.run(adapter.commit_turn(_request()))
    assert result.display_text == _DISPLAY_TEXT

    with (
        _gateway() as (_state, origin),
        pytest.raises(NemoClawProtocolError, match="response_limit"),
    ):
        adapter = NemoClawCommittedTurnAdapter(
            origin=origin,
            deployment_bearer=_DEPLOYMENT_BEARER,
            result_display_budget_bytes=exact_budget - 1,
        )
        asyncio.run(adapter.commit_turn(_request()))


def test_outer_ndjson_duplicate_members_fail_closed() -> None:
    async def collect(adapter: NemoClawCommittedTurnAdapter):
        return [event async for event in adapter.stream_turn(_request())]

    with (
        _gateway(mode="duplicate_member_replay") as (_state, origin),
        pytest.raises(NemoClawProtocolError, match="protocol_error"),
    ):
        asyncio.run(collect(_adapter(origin)))


def test_outer_ndjson_escaping_does_not_narrow_the_decoded_result_limit() -> None:
    display = "\x01" * 30_000
    with _gateway(mode="outer_json_expansion") as (_state, origin):
        adapter = NemoClawCommittedTurnAdapter(
            origin=origin,
            deployment_bearer=_DEPLOYMENT_BEARER,
            result_display_budget_bytes=len(display.encode("utf-8")),
        )
        result = asyncio.run(adapter.commit_turn(_request()))

    assert result.display_text == display
    assert result.speak_text is None


@pytest.mark.parametrize(
    ("mode", "error_type"),
    [
        ("malformed_envelope", NemoClawProtocolError),
        ("failed_after_display", NemoClawBackendFailure),
    ],
)
def test_provisional_display_never_becomes_terminal_evidence_after_failure(
    mode: str,
    error_type: type[Exception],
) -> None:
    async def collect(adapter: NemoClawCommittedTurnAdapter):
        events = []
        with pytest.raises(error_type):
            async for event in adapter.stream_turn(_request()):
                events.append(event)
        return events

    with _gateway(mode=mode) as (state, origin):
        events = asyncio.run(collect(_adapter(origin)))

    assert any(isinstance(event, CommittedTurnDisplayDelta) for event in events)
    assert not any(isinstance(event, CommittedTurnCompleted) for event in events)
    assert [item.method for item in state.requests] == ["POST", "POST", "DELETE"]


def test_closing_stream_abandons_transport_cleans_up_and_reopens_admission() -> None:
    async def exercise(adapter: NemoClawCommittedTurnAdapter) -> None:
        stream = adapter.stream_turn(_request("commit-stream-abandoned"))
        first = await anext(stream)
        assert isinstance(first, CommittedTurnDisplayDelta)
        assert first.delta == "## Test"
        started = time.monotonic()
        await stream.aclose()
        assert time.monotonic() - started < 2.0

        state.mode = "success"
        result = await adapter.commit_turn(_request("commit-after-abandon"))
        assert result.display_text == _DISPLAY_TEXT

    with _gateway(mode="stream_pause") as (state, origin):
        asyncio.run(exercise(_adapter(origin)))

    assert [(item.method, item.path) for item in state.requests] == [
        ("POST", "/v1/voice/sessions"),
        ("POST", "/v1/voice/sessions/session-1/turns"),
        ("DELETE", "/v1/voice/sessions/session-1"),
        ("POST", "/v1/voice/sessions"),
        ("POST", "/v1/voice/sessions/session-2/turns"),
        ("DELETE", "/v1/voice/sessions/session-2"),
    ]


def test_authenticated_readiness_check_reports_only_bounded_capabilities() -> None:
    with _gateway() as (state, origin):
        backend = asyncio.run(_adapter(origin).inspect())

    assert backend.label == "NemoClaw"
    assert backend.mode == "response_only"
    assert backend.target_ref == "backend-selected-per-request"
    assert backend.capabilities == BackendCapabilities(
        backend_kind="response_only",
        target_label="NemoClaw",
        revision="nemoclaw-committed-turn-v1",
        operations=frozenset({BackendOperation.SUBMIT}),
        durability=Durability.NONE,
        event_delivery=EventDelivery.RESPONSE_ONLY,
        max_parallel_work=1,
    )
    assert backend.capability_source is CapabilitySource.COMPATIBILITY_PROJECTION
    assert backend.capability_source_id == "nemoclaw_committed_turn"
    assert backend.capability_evidence == CapabilityEvidence.capture(
        backend.capabilities,
        source=CapabilitySource.COMPATIBILITY_PROJECTION,
        source_id="nemoclaw_committed_turn",
    )
    profile = load_interaction_profile_catalog().resolve("stateless")
    assert profile.session_scope is SessionScope.NONE
    assert profile.work_cardinality is WorkCardinality.MANY
    assert [tool.name for tool in CapabilityToolRegistry(profile=profile).project(backend.capabilities)] == [
        "work.delegate"
    ]
    assert [(item.method, item.path) for item in state.requests] == [("GET", "/healthz")]


def test_managed_transport_routes_every_request_and_reloads_the_deployment_credential() -> None:
    rotated_bearer = "rotated-deployment-bearer-kept-on-server"
    credential = [_DEPLOYMENT_BEARER]
    with _gateway(managed=True) as (state, origin):
        port = int(origin.rsplit(":", 1)[1])
        route_host = f"voice-main.openshell.localhost:{port}"
        adapter = NemoClawCommittedTurnAdapter(
            origin=origin,
            deployment_bearer_loader=lambda: credential[0],
            deployment_authorization=DeploymentAuthorization.NEMOCLAW_SCOPED,
            route_host=route_host,
            target_ref="voice:assistant:main",
            endpoint_policy=EndpointPolicy.MANAGED_PRIVATE_HTTP,
        )

        backend = asyncio.run(adapter.inspect())
        credential[0] = rotated_bearer
        state.deployment_bearer = rotated_bearer
        result = asyncio.run(adapter.commit_turn(_request()))

    assert backend.target_ref == "voice:assistant:main"
    assert result.display_text == _DISPLAY_TEXT
    assert [request.authorization for request in state.requests] == [None, None, None, None]
    assert [request.host for request in state.requests] == [route_host] * 4
    assert [request.nemoclaw_authorization for request in state.requests] == [
        f"Bearer {_DEPLOYMENT_BEARER}",
        f"Bearer {rotated_bearer}",
        "Bearer private-session-grant-1",
        "Bearer private-session-grant-1",
    ]


def test_managed_endpoint_and_route_ports_must_match() -> None:
    with pytest.raises(NemoClawEndpointError, match="ports must match"):
        NemoClawCommittedTurnAdapter(
            origin="http://127.0.0.1:17681",
            deployment_bearer_loader=lambda: _DEPLOYMENT_BEARER,
            deployment_authorization=DeploymentAuthorization.NEMOCLAW_SCOPED,
            route_host="voice-main.openshell.localhost:17682",
            endpoint_policy=EndpointPolicy.MANAGED_PRIVATE_HTTP,
        )


def test_managed_endpoint_rejects_ipv6_unique_local_addresses() -> None:
    with pytest.raises(NemoClawEndpointError, match="loopback or private"):
        NemoClawCommittedTurnAdapter(
            origin="http://[fd00::1]:17681",
            deployment_bearer_loader=lambda: _DEPLOYMENT_BEARER,
            deployment_authorization=DeploymentAuthorization.NEMOCLAW_SCOPED,
            route_host="voice-main.openshell.localhost:17681",
            endpoint_policy=EndpointPolicy.MANAGED_PRIVATE_HTTP,
        )


def test_total_exchange_deadline_stops_a_peer_that_continuously_trickles_bytes() -> None:
    with _gateway(mode="slow_trickle") as (state, origin):
        adapter = NemoClawCommittedTurnAdapter(
            origin=origin,
            deployment_bearer=_DEPLOYMENT_BEARER,
            exchange_deadline_seconds=0.08,
        )
        started = time.monotonic()
        with pytest.raises(NemoClawTransportError, match="timeout"):
            asyncio.run(adapter.commit_turn(_request()))
        elapsed = time.monotonic() - started

    assert elapsed < 0.75
    assert [(item.method, item.path) for item in state.requests] == [
        ("POST", "/v1/voice/sessions"),
        ("POST", "/v1/voice/sessions/session-1/turns"),
        ("DELETE", "/v1/voice/sessions/session-1"),
    ]


def test_accepted_turn_lost_before_terminal_is_never_retried() -> None:
    with (
        _gateway(mode="truncated_after_start") as (state, origin),
        pytest.raises(NemoClawProtocolError, match="truncated_stream"),
    ):
        asyncio.run(_adapter(origin).commit_turn(_request()))

    assert state.admissions == 1
    assert sum(item.path.endswith("/turns") for item in state.requests) == 1
    assert [(item.method, item.path) for item in state.requests] == [
        ("POST", "/v1/voice/sessions"),
        ("POST", "/v1/voice/sessions/session-1/turns"),
        ("DELETE", "/v1/voice/sessions/session-1"),
    ]


def test_concurrent_adapter_admission_allows_one_exchange_and_rejects_one_locally() -> None:
    async def exercise(state: _GatewayState, origin: str) -> None:
        adapter = NemoClawCommittedTurnAdapter(
            origin=origin,
            deployment_bearer=_DEPLOYMENT_BEARER,
            exchange_deadline_seconds=30,
        )
        first = asyncio.create_task(adapter.commit_turn(_request("commit-first")))
        while not any(item.path.endswith("/turns") for item in state.requests):
            await asyncio.sleep(0.005)
        with pytest.raises(NemoClawTransportError, match="backend_busy"):
            await adapter.commit_turn(_request("commit-second"))
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first

    with _gateway(mode="stream_pause") as (state, origin):
        asyncio.run(exercise(state, origin))

    assert state.admissions == 1
    assert sum(item.path.endswith("/turns") for item in state.requests) == 1
    assert [item.method for item in state.requests] == ["POST", "POST", "DELETE"]


def test_cancellation_abandons_active_exchange_and_runs_bounded_cleanup() -> None:
    async def exercise() -> None:
        adapter = NemoClawCommittedTurnAdapter(
            origin=origin,
            deployment_bearer=_DEPLOYMENT_BEARER,
            exchange_deadline_seconds=30,
        )
        task = asyncio.create_task(adapter.commit_turn(_request("commit-abandoned")))
        while not any(item.path.endswith("/turns") for item in state.requests):
            await asyncio.sleep(0.005)
        started = time.monotonic()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert time.monotonic() - started < 2.0

    with _gateway(mode="slow_trickle") as (state, origin):
        asyncio.run(exercise())

    assert [(item.method, item.path) for item in state.requests] == [
        ("POST", "/v1/voice/sessions"),
        ("POST", "/v1/voice/sessions/session-1/turns"),
        ("DELETE", "/v1/voice/sessions/session-1"),
    ]


def test_cancellation_during_admission_recovers_authority_and_deletes_session() -> None:
    async def exercise(state: _GatewayState, origin: str) -> None:
        adapter = NemoClawCommittedTurnAdapter(
            origin=origin,
            deployment_bearer=_DEPLOYMENT_BEARER,
            exchange_deadline_seconds=30,
        )
        task = asyncio.create_task(adapter.commit_turn(_request("commit-abandoned-admission")))
        assert await asyncio.to_thread(state.admission_headers_sent.wait, 1.0)
        started = time.monotonic()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert time.monotonic() - started < 2.0

    with _gateway(mode="delayed_admission_body") as (state, origin):
        asyncio.run(exercise(state, origin))

    assert [(item.method, item.path) for item in state.requests] == [
        ("POST", "/v1/voice/sessions"),
        ("DELETE", "/v1/voice/sessions/session-1"),
    ]


def test_sequence_gap_is_protocol_error_and_session_is_still_deleted() -> None:
    with (
        _gateway(mode="gap") as (state, origin),
        pytest.raises(NemoClawProtocolError, match="protocol_error"),
    ):
        asyncio.run(_adapter(origin).commit_turn(_request()))

    assert [item.method for item in state.requests] == ["POST", "POST", "DELETE"]


def test_terminal_failure_is_typed_and_session_is_deleted_without_exposing_grant() -> None:
    with (
        _gateway(mode="failed") as (state, origin),
        pytest.raises(NemoClawBackendFailure) as captured,
    ):
        asyncio.run(_adapter(origin).commit_turn(_request()))

    assert captured.value.reason == "agent_failed"
    assert str(captured.value) == "agent_failed"
    assert _DEPLOYMENT_BEARER not in repr(captured.value)
    assert "private-session-grant" not in str(captured.value)
    assert "private-session-grant" not in repr(captured.value)
    assert [item.method for item in state.requests] == ["POST", "POST", "DELETE"]


def test_terminal_backend_reason_reaches_interaction_projection_and_persisted_request() -> None:
    async def exercise(origin: str) -> None:
        with SqliteStateStore(":memory:") as store:
            runtime = RealtimeInteractionManager(
                backend_profile="nemoclaw",
                state_store=store,
                committed_turns=_adapter(origin),
            )
            await runtime.open_session("session-agent-failure", "conversation-agent-failure")
            updates = [
                update
                async for update in runtime.execute_tool(
                    session_id="session-agent-failure",
                    commit_id="commit-agent-failure",
                    call_id="call-agent-failure",
                    tool_name="work.delegate",
                    arguments={"goal": "complete the delegated request"},
                    finalized_user_text="complete the delegated request",
                )
            ]

            terminal = updates[-1]
            assert terminal.phase == "failed"
            assert terminal.correlation["error_code"] == "agent_failed"
            projection = json.loads(runtime.projection("session-agent-failure"))
            assert projection["execution"]["latest_failure_code"] == "agent_failed"
            assert projection["local_requests"][0]["failure_code"] == "agent_failed"

    with _gateway(mode="failed") as (state, origin):
        asyncio.run(exercise(origin))

    assert [item.method for item in state.requests] == ["GET", "POST", "POST", "DELETE"]


def test_result_larger_than_realtime_projection_limit_is_rejected_and_deleted() -> None:
    with (
        _gateway(mode="oversized") as (state, origin),
        pytest.raises(NemoClawProtocolError, match="response_limit"),
    ):
        asyncio.run(_adapter(origin).commit_turn(_request()))

    assert [item.method for item in state.requests] == ["POST", "POST", "DELETE"]


@pytest.mark.parametrize("mode", ["empty", "whitespace"])
def test_completed_result_without_display_content_is_protocol_error_and_session_is_deleted(mode: str) -> None:
    with (
        _gateway(mode=mode) as (state, origin),
        pytest.raises(NemoClawProtocolError, match="protocol_error"),
    ):
        asyncio.run(_adapter(origin).commit_turn(_request()))

    assert [item.method for item in state.requests] == ["POST", "POST", "DELETE"]


def test_completed_plain_text_result_is_protocol_error_and_session_is_deleted() -> None:
    with (
        _gateway(mode="plain_text") as (state, origin),
        pytest.raises(NemoClawProtocolError, match="protocol_error"),
    ):
        asyncio.run(_adapter(origin).commit_turn(_request()))

    assert [item.method for item in state.requests] == ["POST", "POST", "DELETE"]


def test_admission_metadata_error_revokes_a_recoverable_private_grant() -> None:
    with (
        _gateway(mode="bad_admission_content_type") as (state, origin),
        pytest.raises(NemoClawProtocolError, match="protocol_error"),
    ):
        asyncio.run(_adapter(origin).commit_turn(_request()))

    assert [(item.method, item.path) for item in state.requests] == [
        ("POST", "/v1/voice/sessions"),
        ("DELETE", "/v1/voice/sessions/session-1"),
    ]


@pytest.mark.parametrize("mode", ["expired_admission", "far_future_admission"])
def test_admission_rejects_expired_or_overlong_grant_and_revokes_it(mode: str) -> None:
    with (
        _gateway(mode=mode) as (state, origin),
        pytest.raises(NemoClawProtocolError, match="protocol_error"),
    ):
        asyncio.run(_adapter(origin).commit_turn(_request()))

    assert [(item.method, item.path) for item in state.requests] == [
        ("POST", "/v1/voice/sessions"),
        ("DELETE", "/v1/voice/sessions/session-1"),
    ]


def test_delete_is_best_effort_and_does_not_create_a_false_durability_claim() -> None:
    with _gateway(delete_status=500) as (state, origin):
        result = asyncio.run(_adapter(origin).commit_turn(_request()))

    assert result.display_text == _DISPLAY_TEXT
    assert result.speak_text == _SPEECH_TEXT
    assert state.requests[-1].method == "DELETE"


def test_failed_cleanup_is_retried_by_readiness_before_new_work_is_admitted() -> None:
    with _gateway(delete_status=500) as (state, origin):
        adapter = _adapter(origin)
        result = asyncio.run(adapter.commit_turn(_request("commit-cleanup-failed")))

        with pytest.raises(SelectedAgentReadinessError, match="selected_agent_unavailable"):
            asyncio.run(adapter.check_selected_agent())

        state.delete_status = 204
        asyncio.run(adapter.check_selected_agent())
        recovered = asyncio.run(adapter.commit_turn(_request("commit-after-cleanup")))

    assert result.display_text == _DISPLAY_TEXT
    assert result.speak_text == _SPEECH_TEXT
    assert recovered.display_text == _DISPLAY_TEXT
    assert [(item.method, item.path) for item in state.requests] == [
        ("POST", "/v1/voice/sessions"),
        ("POST", "/v1/voice/sessions/session-1/turns"),
        ("DELETE", "/v1/voice/sessions/session-1"),
        ("DELETE", "/v1/voice/sessions/session-1"),
        ("DELETE", "/v1/voice/sessions/session-1"),
        ("GET", "/healthz"),
        ("POST", "/v1/voice/sessions"),
        ("POST", "/v1/voice/sessions/session-2/turns"),
        ("DELETE", "/v1/voice/sessions/session-2"),
    ]


def test_readiness_cleans_the_old_session_grant_then_uses_the_rotated_deployment_credential() -> None:
    rotated_bearer = "rotated-deployment-bearer-kept-on-server"
    credential = [_DEPLOYMENT_BEARER]
    with _gateway(delete_status=500, managed=True) as (state, origin):
        port = int(origin.rsplit(":", 1)[1])
        route_host = f"voice-main.openshell.localhost:{port}"
        adapter = NemoClawCommittedTurnAdapter(
            origin=origin,
            deployment_bearer_loader=lambda: credential[0],
            deployment_authorization=DeploymentAuthorization.NEMOCLAW_SCOPED,
            route_host=route_host,
            target_ref="voice:assistant:main",
            endpoint_policy=EndpointPolicy.MANAGED_PRIVATE_HTTP,
        )
        asyncio.run(adapter.commit_turn(_request("commit-before-rotation")))

        credential[0] = rotated_bearer
        state.deployment_bearer = rotated_bearer
        state.delete_status = 204
        asyncio.run(adapter.check_selected_agent())

    assert [(item.method, item.path) for item in state.requests] == [
        ("POST", "/v1/voice/sessions"),
        ("POST", "/v1/voice/sessions/session-1/turns"),
        ("DELETE", "/v1/voice/sessions/session-1"),
        ("DELETE", "/v1/voice/sessions/session-1"),
        ("GET", "/healthz"),
    ]
    assert [item.nemoclaw_authorization for item in state.requests] == [
        f"Bearer {_DEPLOYMENT_BEARER}",
        "Bearer private-session-grant-1",
        "Bearer private-session-grant-1",
        "Bearer private-session-grant-1",
        f"Bearer {rotated_bearer}",
    ]


def test_expired_cleanup_authority_is_discarded_without_treating_unauthorized_delete_as_success() -> None:
    with _gateway(
        delete_status=401,
        session_lifetime=timedelta(seconds=1),
    ) as (state, origin):
        adapter = _adapter(origin)
        asyncio.run(adapter.commit_turn(_request("commit-before-session-expiry")))

        with pytest.raises(SelectedAgentReadinessError, match="selected_agent_unavailable"):
            asyncio.run(adapter.check_selected_agent())

        time.sleep(1.1)
        asyncio.run(adapter.check_selected_agent())
        state.delete_status = 204
        recovered = asyncio.run(adapter.commit_turn(_request("commit-after-session-expiry")))

    assert recovered.display_text == _DISPLAY_TEXT
    assert [(item.method, item.path) for item in state.requests] == [
        ("POST", "/v1/voice/sessions"),
        ("POST", "/v1/voice/sessions/session-1/turns"),
        ("DELETE", "/v1/voice/sessions/session-1"),
        ("DELETE", "/v1/voice/sessions/session-1"),
        ("GET", "/healthz"),
        ("POST", "/v1/voice/sessions"),
        ("POST", "/v1/voice/sessions/session-2/turns"),
        ("DELETE", "/v1/voice/sessions/session-2"),
    ]


def test_shutdown_retries_pending_cleanup_once_then_discards_authority() -> None:
    with _gateway(delete_status=500) as (state, origin):
        adapter = _adapter(origin)
        asyncio.run(adapter.commit_turn(_request("commit-before-shutdown")))

        asyncio.run(adapter.shutdown())
        asyncio.run(adapter.shutdown())
        with pytest.raises(NemoClawTransportError, match="adapter_closed"):
            asyncio.run(adapter.commit_turn(_request("commit-after-shutdown")))

    assert [(item.method, item.path) for item in state.requests] == [
        ("POST", "/v1/voice/sessions"),
        ("POST", "/v1/voice/sessions/session-1/turns"),
        ("DELETE", "/v1/voice/sessions/session-1"),
        ("DELETE", "/v1/voice/sessions/session-1"),
    ]


def test_shutdown_abandons_active_exchange_and_deletes_its_session() -> None:
    async def exercise(state: _GatewayState, origin: str) -> None:
        adapter = NemoClawCommittedTurnAdapter(
            origin=origin,
            deployment_bearer=_DEPLOYMENT_BEARER,
            exchange_deadline_seconds=30,
        )
        task = asyncio.create_task(adapter.commit_turn(_request("commit-active-at-shutdown")))
        while not any(item.path.endswith("/turns") for item in state.requests):
            await asyncio.sleep(0.005)

        started = time.monotonic()
        await adapter.shutdown()
        assert time.monotonic() - started < 2.0
        with pytest.raises(NemoClawTransportError, match="adapter_closed"):
            await task

    with _gateway(mode="slow_trickle") as (state, origin):
        asyncio.run(exercise(state, origin))

    assert [(item.method, item.path) for item in state.requests] == [
        ("POST", "/v1/voice/sessions"),
        ("POST", "/v1/voice/sessions/session-1/turns"),
        ("DELETE", "/v1/voice/sessions/session-1"),
    ]


def test_cancelled_async_consumer_cannot_reopen_admission_before_worker_exits(monkeypatch) -> None:
    async def exercise() -> None:
        adapter = _adapter("http://127.0.0.1:18800")
        entered = threading.Event()
        release = threading.Event()

        def blocked_exchange(*_args, **_kwargs):
            entered.set()
            release.wait(10)
            raise NemoClawTransportError("transport_error")

        monkeypatch.setattr(adapter, "_commit_turn_sync", blocked_exchange)
        consumer = asyncio.create_task(adapter.commit_turn(_request("commit-blocked-worker")))
        while not entered.is_set():
            await asyncio.sleep(0.005)

        started = time.monotonic()
        consumer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await consumer
        assert time.monotonic() - started < 2.0

        with pytest.raises(NemoClawTransportError, match="backend_busy"):
            await adapter.commit_turn(_request("commit-must-not-overlap"))

        release.set()
        for _ in range(200):
            if adapter._operation_gate.acquire(blocking=False):
                adapter._operation_gate.release()
                break
            await asyncio.sleep(0.005)
        else:
            pytest.fail("worker did not release the operation gate")
        await adapter.shutdown()

    asyncio.run(exercise())


def test_cancelled_inspection_with_blocked_loader_does_not_hold_process_exit() -> None:
    script = """
import asyncio
import contextlib
import sys
import threading

sys.path.insert(0, "src/examples/voiceclaw/src")
import voiceclaw.adapters.nemoclaw.committed_turn as committed_turn

committed_turn._ADAPTER_SHUTDOWN_SECONDS = 0.2
committed_turn._ABANDONED_SESSION_CLEANUP_SECONDS = 0.05
entered = threading.Event()
blocked = threading.Event()

def loader():
    entered.set()
    blocked.wait()
    return "deployment-bearer-kept-on-server"

async def main():
    adapter = committed_turn.NemoClawCommittedTurnAdapter(
        origin="http://127.0.0.1:18800",
        deployment_bearer_loader=loader,
    )
    inspection = asyncio.create_task(adapter.inspect())
    while not entered.is_set():
        await asyncio.sleep(0.005)
    inspection.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await inspection
    await adapter.shutdown()

asyncio.run(main())
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=3,
    )
    assert completed.returncode == 0, completed.stderr


def test_shutdown_prevents_cancelled_inspection_from_sending_late_authenticated_request() -> None:
    async def exercise(origin: str) -> None:
        entered = threading.Event()
        release = threading.Event()

        def loader() -> str:
            entered.set()
            release.wait(10)
            return _DEPLOYMENT_BEARER

        adapter = NemoClawCommittedTurnAdapter(
            origin=origin,
            deployment_bearer_loader=loader,
        )
        inspection = asyncio.create_task(adapter.inspect())
        while not entered.is_set():
            await asyncio.sleep(0.005)
        inspection.cancel()
        with pytest.raises(asyncio.CancelledError):
            await inspection

        shutdown = asyncio.create_task(adapter.shutdown())
        await asyncio.sleep(0.05)
        release.set()
        await shutdown

    with _gateway() as (state, origin):
        asyncio.run(exercise(origin))

    assert state.requests == []


def test_http_redirect_is_rejected_without_following_it() -> None:
    with (
        _gateway(mode="redirect") as (state, origin),
        pytest.raises(NemoClawRequestRejected) as captured,
    ):
        asyncio.run(_adapter(origin).commit_turn(_request()))

    assert captured.value.status == 307
    assert [(item.method, item.path) for item in state.requests] == [("POST", "/v1/voice/sessions")]


def test_ambient_proxy_variables_cannot_redirect_authenticated_gateway_calls(monkeypatch) -> None:
    for name in ("ALL_PROXY", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "http_proxy", "https_proxy"):
        monkeypatch.setenv(name, "http://127.0.0.1:1")
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.setenv(name, "")

    with _gateway() as (state, origin):
        result = asyncio.run(_adapter(origin).commit_turn(_request()))

    assert result.display_text == _DISPLAY_TEXT
    assert [item.method for item in state.requests] == ["POST", "POST", "DELETE"]


@pytest.mark.parametrize(
    "origin,policy",
    [
        ("http://8.8.8.8:18800", EndpointPolicy.PRIVATE_NETWORK),
        ("http://10.1.2.3:18800", EndpointPolicy.LOOPBACK_ONLY),
        ("http://10.1.2.3:18800", EndpointPolicy.PRIVATE_NETWORK),
        ("http://localhost:18800", EndpointPolicy.LOOPBACK_ONLY),
        ("http://user:password@127.0.0.1:18800", EndpointPolicy.LOOPBACK_ONLY),
        ("http://127.0.0.1:18800/path", EndpointPolicy.LOOPBACK_ONLY),
    ],
)
def test_endpoint_policy_fails_closed(origin: str, policy: EndpointPolicy) -> None:
    with pytest.raises(NemoClawEndpointError):
        NemoClawCommittedTurnAdapter(
            origin=origin,
            deployment_bearer=_DEPLOYMENT_BEARER,
            endpoint_policy=policy,
        )


def test_private_literal_address_requires_explicit_private_network_policy() -> None:
    adapter = NemoClawCommittedTurnAdapter(
        origin="https://10.1.2.3:18800",
        deployment_bearer=_DEPLOYMENT_BEARER,
        endpoint_policy=EndpointPolicy.PRIVATE_NETWORK,
    )

    assert isinstance(adapter, NemoClawCommittedTurnAdapter)
