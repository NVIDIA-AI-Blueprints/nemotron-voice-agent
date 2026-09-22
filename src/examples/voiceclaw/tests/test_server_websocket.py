# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D101, D102, D103

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from voiceclaw.composition import BackendComposition
from voiceclaw.config import ServerConfig, StateConfig, ToolCopyOverride, load_config
from voiceclaw.domain.models import BackendCapabilities, BackendOperation, CapabilitySource
from voiceclaw.interaction_profiles import load_interaction_profile_catalog
from voiceclaw.ports.turns import CommittedTurnBackend
from voiceclaw.server import create_app

EXAMPLE_CONFIG = Path(__file__).parents[1] / "src" / "voiceclaw" / "resources" / "voiceclaw.example.yaml"
ENVIRONMENT = {
    "NEMOCLAW_VOICE_GATEWAY_BEARER_FILE": "/run/secrets/test-nemoclaw-deployment-bearer",
    "REALTIME_UPSTREAM_ENDPOINT": "ws://127.0.0.1:7861/v1/realtime",
    "REALTIME_UPSTREAM_API_KEY": "private-realtime-key",
}


class _TurnBackend:
    async def inspect(self):
        return CommittedTurnBackend(
            label="NemoClaw",
            target_ref="server-selected agent",
            mode="response_only",
            capabilities=BackendCapabilities(
                backend_kind="response_only",
                target_label="NemoClaw",
                revision="test-server-v1",
                operations=frozenset({BackendOperation.SUBMIT}),
            ),
            capability_source=CapabilitySource.OPERATOR_CONFIGURED,
            capability_source_id="test_committed_turn",
        )

    async def commit_turn(self, _request):  # pragma: no cover - bootstrap test only
        raise AssertionError("no turn expected")


class _FakeUpstream:
    instances: list["_FakeUpstream"] = []

    def __init__(self, **_kwargs) -> None:
        self.sent: list[str] = []
        self.closed = False
        self.incoming = [
            json.dumps(
                {
                    "type": "session.created",
                    "session": {
                        "id": "private-session",
                        "object": "realtime.session",
                        "model": "private-model",
                        "client_secret": {"value": "must-not-cross-the-facade"},
                    },
                }
            ),
            json.dumps(
                {
                    "type": "conversation.created",
                    "conversation": {"id": "private-conversation", "object": "realtime.conversation"},
                }
            ),
        ]
        self.policy_acknowledged = False
        self.waiter: asyncio.Event | None = None
        self.instances.append(self)

    async def __aenter__(self):
        self.waiter = asyncio.Event()
        return self

    async def __aexit__(self, *_args) -> None:
        self.closed = True

    async def receive(self) -> str:
        if self.incoming:
            return self.incoming.pop(0)
        if self.sent and not self.policy_acknowledged:
            self.policy_acknowledged = True
            policy_update = json.loads(self.sent[-1])
            return json.dumps(
                {
                    "type": "session.updated",
                    "session": {
                        "id": "private-session",
                        "object": "realtime.session",
                        **policy_update["session"],
                    },
                }
            )
        assert self.waiter is not None
        await self.waiter.wait()
        raise EOFError

    async def send(self, message: str) -> None:
        self.sent.append(message)


class WebSocketDisconnect(Exception):
    pass


class _FakeWebSocket:
    def __init__(self, *, subprotocols: str = "realtime") -> None:
        self.headers = {"host": "testserver", "sec-websocket-protocol": subprotocols}
        self.query_params = {"model": "nvidia/voiceclaw"}
        self.sent: list[str] = []
        self.accepted_subprotocol: str | None = None
        self.close_code: int | None = None
        self.close_reason = ""

    async def accept(self, *, subprotocol: str | None = None) -> None:
        self.accepted_subprotocol = subprotocol

    async def receive_text(self) -> str:
        raise WebSocketDisconnect

    async def send_text(self, message: str) -> None:
        self.sent.append(message)

    async def close(self, *, code: int, reason: str = "") -> None:
        self.close_code = code
        self.close_reason = reason


def _websocket_endpoint(app):
    return next(route.endpoint for route in app.routes if getattr(route, "path", None) == "/v1/realtime")


def test_public_websocket_bootstrap_is_sanitized_and_keeps_protected_tools_server_side() -> None:
    config = replace(
        load_config(EXAMPLE_CONFIG, environ=ENVIRONMENT),
        server=ServerConfig(host="127.0.0.1", port=7860),
        state=StateConfig(path=":memory:"),
    )
    app = create_app(
        config,
        environ=ENVIRONMENT,
        composition=BackendComposition(
            turn_backend=_TurnBackend(),
            turn_status="response_only",
        ),
    )
    _FakeUpstream.instances.clear()

    websocket = _FakeWebSocket()
    try:
        with patch("voiceclaw.server.WebSocketRealtimeUpstream", _FakeUpstream):
            asyncio.run(_websocket_endpoint(app)(websocket))
    finally:
        app.state.voiceclaw_state_store.close()

    events = [json.loads(message) for message in websocket.sent]
    session_event = events[0]
    conversation_event = events[1]
    attachment_events = events[2:10]

    assert session_event["type"] == "session.created"
    assert session_event["session"]["id"].startswith("sess_vc_")
    assert session_event["session"]["model"] == "nvidia/voiceclaw"
    assert "client_secret" not in session_event["session"]
    assert conversation_event["type"] == "conversation.created"
    assert conversation_event["conversation"]["id"].startswith("conv_vc_")
    attachment_done = attachment_events[-1]
    assert attachment_done["type"] == "response.done"
    assert attachment_done["response"]["metadata"]["voiceclaw_kind"] == "backend_target"
    assert attachment_done["response"]["metadata"]["voiceclaw_phase"] == "reachable"

    upstream = _FakeUpstream.instances[0]
    policy_update = json.loads(upstream.sent[0])
    assert policy_update["type"] == "session.update"
    assert {tool["name"] for tool in policy_update["session"]["tools"]} == {
        "voiceclaw_conversation_respond",
        "voiceclaw_work_delegate",
    }
    delegate = next(tool for tool in policy_update["session"]["tools"] if tool["name"] == "voiceclaw_work_delegate")
    expected_delegate_schema = (
        load_interaction_profile_catalog().resolve("stateless").tool("work.delegate").render_input_schema()
    )
    assert delegate["parameters"] == expected_delegate_schema
    assert "private-realtime-key" not in "".join(upstream.sent)
    assert upstream.closed is True
    assert websocket.accepted_subprotocol == "realtime"
    assert websocket.close_code == 1000


def test_backend_tool_copy_reaches_the_private_selector_schema() -> None:
    delegate_description = "Send substantial work to the configured research backend."
    goal_description = "Complete standalone research goal with all confirmed constraints."
    loaded = load_config(EXAMPLE_CONFIG, environ=ENVIRONMENT)
    backend_name = loaded.default_backend
    backend = replace(
        loaded.backend_profiles[backend_name],
        tool_copy={
            "work.delegate": ToolCopyOverride(
                description=delegate_description,
                properties={"goal": goal_description},
            )
        },
    )
    config = replace(
        loaded,
        server=ServerConfig(host="127.0.0.1", port=7860),
        state=StateConfig(path=":memory:"),
        backend_profiles={**loaded.backend_profiles, backend_name: backend},
    )
    app = create_app(
        config,
        environ=ENVIRONMENT,
        composition=BackendComposition(
            turn_backend=_TurnBackend(),
            turn_status="response_only",
        ),
    )
    _FakeUpstream.instances.clear()

    try:
        with patch("voiceclaw.server.WebSocketRealtimeUpstream", _FakeUpstream):
            asyncio.run(_websocket_endpoint(app)(_FakeWebSocket()))
    finally:
        app.state.voiceclaw_state_store.close()

    policy_update = json.loads(_FakeUpstream.instances[0].sent[0])
    delegate = next(tool for tool in policy_update["session"]["tools"] if tool["name"] == "voiceclaw_work_delegate")
    assert delegate["description"] == delegate_description
    assert delegate["parameters"]["properties"]["goal"]["description"] == goal_description


def test_auth_disabled_rejects_a_supplied_browser_credential() -> None:
    config = replace(
        load_config(EXAMPLE_CONFIG, environ=ENVIRONMENT),
        server=ServerConfig(host="127.0.0.1", port=7860),
        state=StateConfig(path=":memory:"),
    )
    app = create_app(
        config,
        environ=ENVIRONMENT,
        composition=BackendComposition(turn_backend=None, turn_status="disabled"),
    )
    websocket = _FakeWebSocket(subprotocols="realtime, openai-insecure-api-key.ek_not-used")

    try:
        asyncio.run(_websocket_endpoint(app)(websocket))
    finally:
        app.state.voiceclaw_state_store.close()

    assert websocket.accepted_subprotocol is None
    assert websocket.close_code == 1008
    assert websocket.close_reason == "authentication failed"
