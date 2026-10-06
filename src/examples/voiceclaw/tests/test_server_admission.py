# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D101, D102, D103

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import httpx

from voiceclaw.composition import BackendComposition
from voiceclaw.config import ServerConfig, StateConfig, load_config
from voiceclaw.server import RealtimeAdmission, create_app

EXAMPLE_CONFIG = Path(__file__).parents[1] / "src" / "voiceclaw" / "resources" / "voiceclaw.example.yaml"


class _SelectedAgentReadiness:
    def __init__(self, failure: Exception | None = None) -> None:
        self.calls = 0
        self.failure = failure

    async def check_selected_agent(self) -> None:
        self.calls += 1
        if self.failure is not None:
            raise self.failure


class _TurnBackend:
    def __init__(self) -> None:
        self.shutdown_calls = 0

    async def inspect(self):  # pragma: no cover - no backend work is created in these tests
        raise AssertionError("admission must not inspect backend work")

    async def commit_turn(self, _request):  # pragma: no cover - no backend work is created in these tests
        raise AssertionError("admission must not commit backend work")

    async def stream_turn(self, _request):  # pragma: no cover - no backend work is created in these tests
        raise AssertionError("admission must not stream backend work")

    async def shutdown(self) -> None:
        self.shutdown_calls += 1


class _FakeUpstream:
    created = 0

    def __init__(self, **_kwargs) -> None:
        type(self).created += 1

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args) -> None:
        return None


class _BlockingFacade:
    def __init__(self, *, downstream, **_kwargs) -> None:
        self._websocket = downstream._websocket

    async def serve(self) -> None:
        self._websocket.facade_started.set()
        await self._websocket.closed.wait()


class _RuntimeOpeningFacade:
    def __init__(self, *, downstream, runtime, **_kwargs) -> None:
        self._websocket = downstream._websocket
        self._runtime = runtime

    async def serve(self) -> None:
        session_id = "session-backend-unavailable"
        snapshot = await self._runtime.open_session(session_id, "conversation-backend-unavailable")
        self._websocket.session_snapshot = snapshot
        self._websocket.runtime_projection = json.loads(self._runtime.projection(session_id))
        self._websocket.facade_started.set()
        try:
            await self._websocket.closed.wait()
        finally:
            await self._runtime.close_session(session_id, "client_disconnected")


class _FakeWebSocket:
    def __init__(
        self,
        *,
        model: str = "nvidia/voiceclaw",
        subprotocols: str = "realtime",
    ) -> None:
        self.headers = {"host": "testserver", "sec-websocket-protocol": subprotocols}
        self.query_params = {"model": model}
        self.accepted = asyncio.Event()
        self.facade_started = asyncio.Event()
        self.closed = asyncio.Event()
        self.close_calls: list[tuple[int, str]] = []
        self.session_snapshot = None
        self.runtime_projection = None

    async def accept(self, *, subprotocol: str | None = None) -> None:
        assert subprotocol == "realtime"
        self.accepted.set()

    async def receive_text(self) -> str:  # pragma: no cover - the fake facade controls lifetime
        await self.closed.wait()
        raise EOFError

    async def send_text(self, _message: str) -> None:  # pragma: no cover - the fake facade sends nothing
        return None

    async def close(self, *, code: int, reason: str = "") -> None:
        self.close_calls.append((code, reason))
        self.closed.set()


def _app(tmp_path: Path, *, max_sessions: int = 1):
    credential_file = tmp_path / "openshell-client-secret"
    credential_file.write_text("voiceclaw-test-deployment-bearer-0001\n", encoding="ascii")
    credential_file.chmod(0o600)
    environ = {
        "VOICECLAW_OPENSHELL_CLIENT_SECRET_FILE": str(credential_file),
        "VOICECLAW_OPENSHELL_ENDPOINT": "127.0.0.1:8080",
        "VOICECLAW_OPENSHELL_WORKSPACE": "test-workspace",
        "VOICECLAW_OPENSHELL_SANDBOX": "test-sandbox",
        "VOICECLAW_FABRIC_ADAPTER_ID": "nvidia.fabric.openclaw",
        "VOICECLAW_FABRIC_AGENT": "main",
        "VOICECLAW_NATIVE_AGENT": "main",
        "VOICECLAW_OPENSHELL_ISSUER": "https://identity.example.test",
        "VOICECLAW_OPENSHELL_CLIENT_ID": "voiceclaw-test",
        "REALTIME_UPSTREAM_ENDPOINT": "ws://127.0.0.1:7861/v1/realtime",
        "REALTIME_UPSTREAM_API_KEY": "internal-upstream-key",
    }
    loaded = load_config(EXAMPLE_CONFIG, environ=environ)
    config = replace(
        loaded,
        server=ServerConfig(host="127.0.0.1", port=7860, max_sessions=max_sessions),
        state=StateConfig(path=":memory:"),
    )
    readiness = _SelectedAgentReadiness()
    backend = _TurnBackend()
    app = create_app(
        config,
        environ=environ,
        composition=BackendComposition(
            turn_backend=backend,
            turn_status="response_only",
            selected_agent_readiness=readiness,
        ),
    )
    app.state.test_turn_backend = backend
    return app, readiness


def _websocket_endpoint(app):
    return next(route.endpoint for route in app.routes if getattr(route, "path", None) == "/v1/realtime")


def test_concurrent_acquires_cannot_oversubscribe_one_session() -> None:
    async def exercise() -> None:
        admission = RealtimeAdmission(1)

        async def close() -> None:
            return None

        leases = await asyncio.gather(admission.acquire(close), admission.acquire(close))
        admitted = [lease for lease in leases if lease is not None]
        assert len(admitted) == 1
        assert leases.count(None) == 1
        await admission.release(admitted[0])
        replacement = await admission.acquire(close)
        assert replacement is not None
        await admission.release(replacement)

    asyncio.run(exercise())


def test_capacity_is_atomic_without_changing_deployment_readiness(tmp_path: Path) -> None:
    app, readiness = _app(tmp_path)
    endpoint = _websocket_endpoint(app)
    _FakeUpstream.created = 0

    async def exercise() -> None:
        async with app.router.lifespan_context(app):
            with (
                patch("voiceclaw.server.WebSocketRealtimeUpstream", _FakeUpstream),
                patch("voiceclaw.server.VoiceClawRealtimeFacade", _BlockingFacade),
            ):
                first = _FakeWebSocket()
                first_task = asyncio.create_task(endpoint(first))
                await asyncio.wait_for(first.facade_started.wait(), timeout=1)

                second = _FakeWebSocket()
                await endpoint(second)
                assert second.accepted.is_set() is False
                assert second.close_calls == [(1013, "capacity unavailable")]
                assert _FakeUpstream.created == 1

                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                    full = await client.get("/readyz")
                assert full.status_code == 200
                assert full.content == b""
                assert full.headers["x-voiceclaw-reason"] == "ready"

                await first.close(code=1000, reason="client finished")
                await asyncio.wait_for(first_task, timeout=1)

                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                    available = await client.get("/readyz")
                assert available.status_code == 200
                assert available.headers["x-voiceclaw-reason"] == "ready"

                third = _FakeWebSocket()
                third_task = asyncio.create_task(endpoint(third))
                await asyncio.wait_for(third.facade_started.wait(), timeout=1)
                assert _FakeUpstream.created == 2
                await third.close(code=1000, reason="client finished")
                await asyncio.wait_for(third_task, timeout=1)

    asyncio.run(exercise())
    assert readiness.calls == 2


def test_rejected_requests_do_not_consume_capacity_or_open_upstream(tmp_path: Path) -> None:
    app, _readiness = _app(tmp_path)
    endpoint = _websocket_endpoint(app)
    _FakeUpstream.created = 0

    async def exercise() -> None:
        async with app.router.lifespan_context(app):
            with (
                patch("voiceclaw.server.WebSocketRealtimeUpstream", _FakeUpstream),
                patch("voiceclaw.server.VoiceClawRealtimeFacade", _BlockingFacade),
            ):
                credential_when_auth_is_disabled = _FakeWebSocket(
                    subprotocols="realtime, openai-insecure-api-key.ek_not-used"
                )
                await endpoint(credential_when_auth_is_disabled)
                assert credential_when_auth_is_disabled.close_calls == [(1008, "authentication failed")]

                wrong_model = _FakeWebSocket(model="private-model")
                await endpoint(wrong_model)
                assert wrong_model.close_calls == [(1008, "unknown VoiceClaw model")]
                assert _FakeUpstream.created == 0

                valid = _FakeWebSocket()
                valid_task = asyncio.create_task(endpoint(valid))
                await asyncio.wait_for(valid.facade_started.wait(), timeout=1)
                assert _FakeUpstream.created == 1
                await valid.close(code=1000, reason="client finished")
                await asyncio.wait_for(valid_task, timeout=1)

    asyncio.run(exercise())


def test_selected_agent_failure_does_not_gate_realtime_session(tmp_path: Path) -> None:
    app, readiness = _app(tmp_path)
    readiness.failure = RuntimeError("private backend detail")
    endpoint = _websocket_endpoint(app)
    _FakeUpstream.created = 0

    async def exercise() -> None:
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                unavailable = await client.get("/readyz")
            assert unavailable.status_code == 503
            assert unavailable.headers["x-voiceclaw-reason"] == "agent-unavailable"

            with (
                patch("voiceclaw.server.WebSocketRealtimeUpstream", _FakeUpstream),
                patch("voiceclaw.server.VoiceClawRealtimeFacade", _RuntimeOpeningFacade),
            ):
                accepted = _FakeWebSocket()
                accepted_task = asyncio.create_task(endpoint(accepted))
                await asyncio.wait_for(accepted.facade_started.wait(), timeout=1)
                assert accepted.accepted.is_set() is True
                assert accepted.close_calls == []
                assert _FakeUpstream.created == 1
                snapshot = accepted.session_snapshot
                projection = accepted.runtime_projection
                assert snapshot is not None
                assert projection is not None
                assert snapshot.target_binding_verified is False
                assert snapshot.capabilities == ()
                assert projection["connection"]["target_binding_state"] == "unavailable"
                assert projection["contract"]["frontend_tools"] == []
                await accepted.close(code=1000, reason="client finished")
                await asyncio.wait_for(accepted_task, timeout=1)

    asyncio.run(exercise())
    assert readiness.calls == 1


def test_lifespan_shutdown_closes_and_drains_active_sessions(tmp_path: Path) -> None:
    app, _readiness = _app(tmp_path)
    endpoint = _websocket_endpoint(app)
    _FakeUpstream.created = 0

    async def exercise() -> _FakeWebSocket:
        active = _FakeWebSocket()
        with (
            patch("voiceclaw.server.WebSocketRealtimeUpstream", _FakeUpstream),
            patch("voiceclaw.server.VoiceClawRealtimeFacade", _BlockingFacade),
        ):
            async with app.router.lifespan_context(app):
                active_task = asyncio.create_task(endpoint(active))
                await asyncio.wait_for(active.facade_started.wait(), timeout=1)
            await asyncio.wait_for(active_task, timeout=1)
        return active

    websocket = asyncio.run(exercise())
    assert websocket.close_calls[0] == (1001, "server shutting down")
    assert websocket.close_calls[-1] == (1000, "")
    assert app.state.test_turn_backend.shutdown_calls == 1
