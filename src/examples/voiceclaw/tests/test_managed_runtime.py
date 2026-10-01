# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D101, D102, D103

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from voiceclaw.adapters.nemoclaw.managed_projection import (
    ManagedProjection,
    ManagedProjectionError,
    ManagedProjectionPaths,
)
from voiceclaw.frontend_runtime import FrontendRuntimePlan
from voiceclaw.managed_runtime import (
    ManagedApplication,
    ManagedRuntimeController,
    ManagedRuntimeError,
    _require_managed_identity,
    _terminate_process,
    _wait_for_realtime_frontend,
)


class _StartingController:
    def __init__(self) -> None:
        self.started = False
        self.stopped = False

    async def start(self) -> None:
        self.started = True

    async def shutdown(self) -> None:
        self.stopped = True

    async def snapshot(self) -> tuple[None, str]:
        return None, "starting"

    async def readiness_snapshot(self) -> tuple[None, str]:
        return await self.snapshot()


class _ReadyDelegate:
    def __init__(self, admission: "_Admission | None" = None) -> None:
        self.state = SimpleNamespace(voiceclaw_admission=admission or _Admission())

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        del receive
        if scope["type"] == "websocket":
            await send({"type": "websocket.accept"})
            await send({"type": "websocket.close", "code": 1000})
            return
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})


class _LifecycleController(ManagedRuntimeController):
    def __init__(self, *, stop_after_activations: int) -> None:
        super().__init__(poll_seconds=0)
        self.stop_after_activations = stop_after_activations
        self.activations: list[str] = []
        self.deactivations: list[str] = []

    async def _activate(self, projection: ManagedProjection) -> None:
        self.activations.append(projection.projection_digest)
        async with self._lock:
            self._delegate = _ReadyDelegate()
            self._projection_digest = projection.projection_digest
            self._reason = "ready"
        if len(self.activations) >= self.stop_after_activations:
            self._stopping = True

    async def readiness_snapshot(self) -> tuple[Any | None, str]:
        return await self.snapshot()

    async def _deactivate(self, reason: str, *, expected_process: object | None = None) -> bool:
        del expected_process
        self.deactivations.append(reason)
        async with self._lock:
            self._delegate = None
            self._delegate_lifespan = None
            self._nva = None
            self._projection_digest = None
            self._reason = reason
        return True


class _Admission:
    def __init__(self, *, accepting: bool = True) -> None:
        self.drains = 0
        self.accepting = accepting
        self.leases: set[object] = set()

    async def can_accept(self) -> bool:
        return self.accepting and not self.leases

    async def acquire(self, _close: Any) -> object | None:
        if not await self.can_accept():
            return None
        lease = object()
        self.leases.add(lease)
        return lease

    async def release(self, lease: object) -> None:
        self.leases.remove(lease)

    async def begin_drain(self) -> None:
        self.drains += 1
        self.accepting = False


class _Lifespan:
    def __init__(self) -> None:
        self.exits = 0

    async def __aexit__(self, exception_type: Any, exception: Any, traceback: Any) -> None:
        del exception_type, exception, traceback
        self.exits += 1


class _Process:
    def __init__(self, returncode: int | None = None) -> None:
        self.returncode = returncode
        self.terminated = False
        self.killed = False
        self.waits = 0

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True

    async def wait(self) -> int:
        self.waits += 1
        self.returncode = 0
        return 0


class _ExitRacingProcess(_Process):
    def terminate(self) -> None:
        self.returncode = 0
        raise ProcessLookupError


class _BootstrapUpstream:
    def __init__(self, **_arguments: Any) -> None:
        self._events = iter(
            (
                '{"type":"session.created"}',
                '{"type":"conversation.created"}',
            )
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_arguments: Any) -> None:
        return None

    async def receive_text(self) -> str:
        return next(self._events)


class _SlowClosingBootstrapUpstream(_BootstrapUpstream):
    async def __aexit__(self, *_arguments: Any) -> None:
        await asyncio.sleep(60)


class _EnteredLifespan:
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.exits = 0

    async def __aenter__(self) -> None:
        self.entered.set()

    async def __aexit__(self, *_arguments: Any) -> None:
        self.exits += 1


def _projection(digest: str) -> ManagedProjection:
    return ManagedProjection(
        integration="voice",
        sandbox="sandbox",
        agent="agent",
        agent_endpoint="http://127.0.0.1:8652",
        agent_route_host="agent.localhost:8652",
        paths=ManagedProjectionPaths(root=Path("/unused")),
        projection_digest=digest,
    )


def _frontend_plan() -> FrontendRuntimePlan:
    return FrontendRuntimePlan(
        kind="bundled_nva",
        launch_bundled_nva=True,
        public_model="nvidia/voiceclaw",
        upstream_endpoint="ws://127.0.0.1:7861/v1/realtime",
        upstream_model="nvidia/voiceclaw-cascade",
    )


async def _invoke(app: ManagedApplication, scope: dict[str, Any]) -> list[dict[str, Any]]:
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    await app(scope, receive, send)
    return sent


async def _inline_to_thread(function: Any, /, *args: Any, **kwargs: Any) -> Any:
    return function(*args, **kwargs)


def test_managed_application_liveness_does_not_wait_for_projection() -> None:
    controller = _StartingController()
    app = ManagedApplication(controller)

    messages = asyncio.run(_invoke(app, {"type": "http", "path": "/livez"}))

    assert messages == [
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"cache-control", b"no-store")],
        },
        {"type": "http.response.body", "body": b""},
    ]


def test_managed_application_is_not_ready_before_projection_activation() -> None:
    controller = _StartingController()
    app = ManagedApplication(controller)

    messages = asyncio.run(_invoke(app, {"type": "http", "path": "/readyz"}))

    assert messages == [
        {
            "type": "http.response.start",
            "status": 503,
            "headers": [
                (b"cache-control", b"no-store"),
                (b"x-voiceclaw-reason", b"starting"),
            ],
        },
        {"type": "http.response.body", "body": b""},
    ]


def test_managed_application_rejects_websocket_before_projection_activation() -> None:
    controller = _StartingController()
    app = ManagedApplication(controller)

    messages = asyncio.run(_invoke(app, {"type": "websocket", "path": "/v1/realtime"}))

    assert messages == [
        {
            "type": "websocket.close",
            "code": 1013,
            "reason": "service unavailable",
        }
    ]


def test_managed_application_lifespan_controls_mock_controller() -> None:
    controller = _StartingController()
    app = ManagedApplication(controller)
    received = iter(
        (
            {"type": "lifespan.startup"},
            {"type": "lifespan.shutdown"},
        )
    )
    sent: list[dict[str, Any]] = []

    async def exercise() -> None:
        async def receive() -> dict[str, Any]:
            return next(received)

        async def send(message: dict[str, Any]) -> None:
            sent.append(message)

        await app({"type": "lifespan"}, receive, send)

    asyncio.run(exercise())

    assert controller.started is True
    assert controller.stopped is True
    assert sent == [
        {"type": "lifespan.startup.complete"},
        {"type": "lifespan.shutdown.complete"},
    ]


def test_managed_runtime_requires_projected_service_identity() -> None:
    with (
        patch("voiceclaw.managed_runtime.os.geteuid", return_value=65_532),
        patch("voiceclaw.managed_runtime.os.getegid", return_value=65_532),
    ):
        _require_managed_identity()

    for uid, gid in ((0, 65_532), (65_532, 0)):
        with (
            patch("voiceclaw.managed_runtime.os.geteuid", return_value=uid),
            patch("voiceclaw.managed_runtime.os.getegid", return_value=gid),
            pytest.raises(ManagedRuntimeError, match="managed-identity-invalid"),
        ):
            _require_managed_identity()


def test_frontend_readiness_requires_the_realtime_bootstrap_sequence() -> None:
    process = _Process()

    with patch("voiceclaw.managed_runtime.WebSocketRealtimeUpstream", _BootstrapUpstream):
        asyncio.run(
            _wait_for_realtime_frontend(
                process,
                plan=_frontend_plan(),
                bearer="internal-key",
                maximum_event_bytes=4096,
                timeout_seconds=1,
            )
        )

    assert process.returncode is None


def test_frontend_cleanup_tolerates_child_exit_before_terminate_signal() -> None:
    process = _ExitRacingProcess()

    asyncio.run(_terminate_process(process))

    assert process.waits == 1
    assert process.killed is False


def test_frontend_readiness_timeout_includes_upstream_close() -> None:
    process = _Process()

    with (
        patch("voiceclaw.managed_runtime.WebSocketRealtimeUpstream", _SlowClosingBootstrapUpstream),
        pytest.raises(ManagedRuntimeError, match="speech-unavailable"),
    ):
        asyncio.run(
            _wait_for_realtime_frontend(
                process,
                plan=_frontend_plan(),
                bearer="internal-key",
                maximum_event_bytes=4096,
                timeout_seconds=0.01,
            )
        )


def test_controller_activates_after_projection_arrives_without_replacing_outer_app() -> None:
    controller = _LifecycleController(stop_after_activations=1)
    app = ManagedApplication(controller)

    async def exercise() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        before = await _invoke(app, {"type": "http", "path": "/livez"})
        await controller._run()
        after = await _invoke(app, {"type": "http", "path": "/readyz"})
        return before, after

    with (
        patch(
            "voiceclaw.managed_runtime.load_managed_projection",
            side_effect=(ManagedProjectionError("projection-unavailable"), _projection("first")),
        ) as load_projection,
        patch("voiceclaw.managed_runtime.asyncio.to_thread", _inline_to_thread),
    ):
        before, after = asyncio.run(exercise())

    assert before[0]["status"] == 200
    assert after[0]["status"] == 204
    assert controller.activations == ["first"]
    assert controller.deactivations == ["starting", "starting"]
    assert load_projection.call_count == 2
    assert app.controller is controller


def test_controller_reactivates_only_when_projection_digest_changes() -> None:
    controller = _LifecycleController(stop_after_activations=2)

    with (
        patch(
            "voiceclaw.managed_runtime.load_managed_projection",
            side_effect=(_projection("first"), _projection("first"), _projection("second")),
        ) as load_projection,
        patch("voiceclaw.managed_runtime.asyncio.to_thread", _inline_to_thread),
    ):
        asyncio.run(controller._run())

    assert controller.activations == ["first", "second"]
    assert controller.deactivations == ["starting", "starting"]
    assert load_projection.call_count == 3


def test_controller_polling_fails_closed_and_recovers_after_unexpected_projection_error() -> None:
    controller = _LifecycleController(stop_after_activations=1)
    controller._delegate = _ReadyDelegate()
    controller._projection_digest = "old"
    controller._reason = "ready"

    with (
        patch(
            "voiceclaw.managed_runtime.load_managed_projection",
            side_effect=(OSError("projection read failed"), _projection("replacement")),
        ) as load_projection,
        patch("voiceclaw.managed_runtime.asyncio.to_thread", _inline_to_thread),
    ):
        asyncio.run(controller._run())

    assert controller.activations == ["replacement"]
    assert controller.deactivations == ["configuration-invalid", "starting"]
    assert load_projection.call_count == 2


def test_controller_snapshot_deactivates_when_frontend_process_dies() -> None:
    controller = ManagedRuntimeController()
    admission = _Admission()
    process = _Process(returncode=17)
    controller._delegate = SimpleNamespace(state=SimpleNamespace(voiceclaw_admission=admission))
    controller._nva = process
    controller._projection_digest = "active"
    controller._reason = "ready"

    delegate, reason = asyncio.run(controller.snapshot())

    assert delegate is None
    assert reason == "speech-unavailable"
    assert admission.drains == 1
    assert process.terminated is False
    assert controller._projection_digest is None


def test_conditional_deactivation_does_not_tear_down_replacement_process() -> None:
    controller = ManagedRuntimeController()
    old_process = _Process(returncode=17)
    replacement_process = _Process()
    replacement_delegate = SimpleNamespace(state=SimpleNamespace(voiceclaw_admission=_Admission()))
    controller._delegate = replacement_delegate
    controller._nva = replacement_process
    controller._projection_digest = "replacement"
    controller._reason = "ready"

    deactivated = asyncio.run(controller._deactivate("speech-unavailable", expected_process=old_process))

    assert deactivated is False
    assert controller._delegate is replacement_delegate
    assert controller._nva is replacement_process
    assert controller._projection_digest == "replacement"
    assert replacement_process.terminated is False


def test_only_readyz_probes_frontend_before_delegation() -> None:
    controller = ManagedRuntimeController()
    process = _Process()
    controller._delegate = _ReadyDelegate()
    controller._nva = process
    controller._frontend_plan_state = _frontend_plan()
    controller._frontend_bearer = "internal-key"
    controller._frontend_maximum_event_bytes = 4096
    controller._projection_digest = "active"
    controller._reason = "ready"
    app = ManagedApplication(controller)
    probes: list[tuple[_Process, str, int]] = []

    async def probe(
        observed_process: _Process,
        *,
        plan: FrontendRuntimePlan,
        bearer: str,
        maximum_event_bytes: int,
        timeout_seconds: float,
    ) -> None:
        del plan, timeout_seconds
        probes.append((observed_process, bearer, maximum_event_bytes))

    async def exercise() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        ready = await _invoke(app, {"type": "http", "path": "/readyz"})
        ordinary = await _invoke(app, {"type": "http", "path": "/v1/config"})
        websocket = await _invoke(app, {"type": "websocket", "path": "/v1/realtime"})
        return ready, ordinary, websocket

    with patch("voiceclaw.managed_runtime._probe_realtime_frontend", probe):
        ready, ordinary, websocket = asyncio.run(exercise())

    assert ready[0]["status"] == 204
    assert ordinary[0]["status"] == 204
    assert websocket == [
        {"type": "websocket.accept"},
        {"type": "websocket.close", "code": 1000},
    ]
    assert probes == [(process, "internal-key", 4096)]


def test_readyz_tolerates_two_transient_frontend_probe_failures_before_restart() -> None:
    controller = ManagedRuntimeController()
    admission = _Admission()
    process = _Process()
    controller._delegate = SimpleNamespace(state=SimpleNamespace(voiceclaw_admission=admission))
    controller._nva = process
    controller._frontend_plan_state = _frontend_plan()
    controller._frontend_bearer = "internal-key"
    controller._frontend_maximum_event_bytes = 4096
    controller._projection_digest = "active"
    controller._reason = "ready"
    app = ManagedApplication(controller)

    async def unavailable(*_arguments: Any, **_keywords: Any) -> None:
        raise ManagedRuntimeError("speech-unavailable")

    with patch("voiceclaw.managed_runtime._probe_realtime_frontend", unavailable):
        first = asyncio.run(_invoke(app, {"type": "http", "path": "/readyz"}))
        second = asyncio.run(_invoke(app, {"type": "http", "path": "/readyz"}))

        assert first[0]["status"] == 503
        assert second[0]["status"] == 503
        assert admission.drains == 0
        assert process.terminated is False
        assert controller._delegate is not None
        assert controller._nva is process

        messages = asyncio.run(_invoke(app, {"type": "http", "path": "/readyz"}))

    assert messages == [
        {
            "type": "http.response.start",
            "status": 503,
            "headers": [
                (b"cache-control", b"no-store"),
                (b"x-voiceclaw-reason", b"speech-unavailable"),
            ],
        },
        {"type": "http.response.body", "body": b""},
    ]
    assert admission.drains == 1
    assert process.terminated is True
    assert controller._delegate is None
    assert controller._nva is None


def test_successful_readyz_probe_resets_transient_failure_count() -> None:
    controller = ManagedRuntimeController()
    admission = _Admission()
    process = _Process()
    controller._delegate = SimpleNamespace(state=SimpleNamespace(voiceclaw_admission=admission))
    controller._nva = process
    controller._frontend_plan_state = _frontend_plan()
    controller._frontend_bearer = "internal-key"
    controller._frontend_maximum_event_bytes = 4096
    controller._projection_digest = "active"
    controller._reason = "ready"
    outcomes = iter((False, True, False, False))

    async def probe(*_arguments: Any, **_keywords: Any) -> None:
        if not next(outcomes):
            raise ManagedRuntimeError("speech-unavailable")

    with patch("voiceclaw.managed_runtime._probe_realtime_frontend", probe):
        observations = [asyncio.run(controller.readiness_snapshot()) for _index in range(4)]

    assert [delegate is not None for delegate, _reason in observations] == [False, True, False, False]
    assert admission.drains == 0
    assert process.terminated is False
    assert controller._delegate is not None
    assert controller._frontend_readiness_failures == 2


def test_readyz_skips_frontend_probe_when_public_session_capacity_is_full() -> None:
    controller = ManagedRuntimeController()
    admission = _Admission(accepting=False)
    process = _Process()
    delegate = SimpleNamespace(state=SimpleNamespace(voiceclaw_admission=admission))
    controller._delegate = delegate
    controller._nva = process
    controller._frontend_plan_state = _frontend_plan()
    controller._frontend_bearer = "internal-key"
    controller._frontend_maximum_event_bytes = 4096
    controller._projection_digest = "active"
    controller._reason = "ready"

    async def unexpected_probe(*_arguments: Any, **_keywords: Any) -> None:
        raise AssertionError("frontend must not be probed while admission is full")

    with patch("voiceclaw.managed_runtime._probe_realtime_frontend", unexpected_probe):
        observed_delegate, reason = asyncio.run(controller.readiness_snapshot())

    assert observed_delegate is delegate
    assert reason == "ready"


def test_controller_shutdown_exits_nested_lifespan_and_terminates_frontend_process() -> None:
    controller = ManagedRuntimeController()
    admission = _Admission()
    lifespan = _Lifespan()
    process = _Process()
    controller._delegate = SimpleNamespace(state=SimpleNamespace(voiceclaw_admission=admission))
    controller._delegate_lifespan = lifespan
    controller._nva = process
    controller._projection_digest = "active"
    controller._reason = "ready"

    asyncio.run(controller.shutdown())

    assert admission.drains == 1
    assert lifespan.exits == 1
    assert process.terminated is True
    assert process.killed is False
    assert process.waits == 1
    assert controller._delegate is None
    assert controller._nva is None
    assert controller._reason == "starting"


def test_activation_cancellation_before_publication_closes_lifespan_and_child(tmp_path: Path) -> None:
    controller = ManagedRuntimeController(runtime_root=tmp_path / "runtime", source_environment={})
    process = _Process()
    lifespan = _EnteredLifespan()
    config = SimpleNamespace(realtime=SimpleNamespace(max_event_bytes=4096))
    app = SimpleNamespace(router=SimpleNamespace(lifespan_context=lambda _app: lifespan))

    async def create_process(*_arguments: Any, **_keywords: Any) -> _Process:
        return process

    async def frontend_ready(*_arguments: Any, **_keywords: Any) -> None:
        return None

    async def exercise() -> None:
        await controller._lock.acquire()
        try:
            task = asyncio.create_task(controller._activate(_projection("first")))
            await asyncio.wait_for(lifespan.entered.wait(), timeout=1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            controller._lock.release()

    with (
        patch("voiceclaw.managed_runtime.materialize_managed_config", return_value=config),
        patch.object(controller, "_frontend_plan", return_value=_frontend_plan()),
        patch("voiceclaw.managed_runtime.read_managed_credential", return_value="s" * 32),
        patch("voiceclaw.managed_runtime.asyncio.create_subprocess_exec", create_process),
        patch("voiceclaw.managed_runtime.asyncio.to_thread", _inline_to_thread),
        patch("voiceclaw.managed_runtime._wait_for_realtime_frontend", frontend_ready),
        patch("voiceclaw.managed_runtime.create_app", return_value=app),
    ):
        asyncio.run(exercise())

    assert lifespan.exits == 1
    assert process.terminated is True
    assert process.waits == 1
    assert controller._delegate is None
