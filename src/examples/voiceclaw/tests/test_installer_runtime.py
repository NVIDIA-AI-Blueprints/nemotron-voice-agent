# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause
# ruff: noqa: D100, D101, D102, D103, D107

import asyncio
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import yaml
from test_installer_inputs import connection, environment, settings

from voiceclaw import installer_runtime
from voiceclaw.config import load_config
from voiceclaw.frontend_runtime import materialize_frontend_runtime
from voiceclaw.installer_executor import compose_installer_backend
from voiceclaw.installer_runtime import InstallerApplication, run_healthcheck
from voiceclaw.model_contracts import load_model_contract_catalog
from voiceclaw.ports.readiness import SelectedAgentReadinessError
from voiceclaw.server import create_app


class Process:
    pid = 54321
    returncode = None
    terminated = False
    killed = False

    def terminate(self):
        self.terminated = True
        self.returncode = 0

    def kill(self):
        self.killed = True
        self.returncode = -9

    async def wait(self):
        return self.returncode


class AgentHealth:
    def __init__(self, error=None):
        self.error = error
        self.calls = 0

    async def check_selected_agent_readiness(self):
        self.calls += 1
        if self.error:
            raise self.error


def runtime(error=None, *, no_auth=False, **overrides):
    app = InstallerApplication(settings(environment(no_auth=no_auth, **overrides)))
    app.process = Process()
    app.initialized = True
    app.delegate = SimpleNamespace(state=SimpleNamespace(voiceclaw_adapters=AgentHealth(error)))
    return app


async def request(app, path):
    messages = []

    async def send(message):
        messages.append(message)

    async def receive():
        return {}

    await app({"type": "http", "path": path}, receive, send)
    return messages


@pytest.mark.parametrize("no_auth", [False, True])
@pytest.mark.parametrize("fault", [None, "initialization", "frontend", "probe", "probe-error", "stopping"])
def test_installation_health_is_local_observation_and_fail_closed(fault, no_auth):
    app = runtime(SelectedAgentReadinessError("selected_agent_readiness_unsupported"), no_auth=no_auth)
    if fault == "initialization":
        app.initialized = False
    if fault == "frontend":
        app.process.returncode = 1
    if fault == "stopping":
        app.stopping = True
    original_delegate = app.delegate
    with patch.object(
        app,
        "_frontend_available",
        new=AsyncMock(
            return_value=fault != "probe",
            side_effect=OSError("private-sentinel") if fault == "probe-error" else None,
        ),
    ):
        messages = asyncio.run(request(app, "/health"))
    assert messages[0]["status"] == (200 if fault is None else 503)
    assert messages[1]["body"] == b""
    assert b"private-sentinel" not in repr(messages).encode()
    assert app.delegate is original_delegate
    assert app.process.terminated is False
    assert app.delegate.state.voiceclaw_adapters.calls == 0


def test_busy_does_not_fail_readiness_and_total_deadline():
    app = runtime()
    app.delegate.state.voiceclaw_admission = SimpleNamespace(can_accept=lambda: False)
    with patch.object(app, "_frontend_available", new=AsyncMock(return_value=True)):
        assert asyncio.run(request(app, "/health"))[0]["status"] == 200
    assert app.delegate.state.voiceclaw_adapters.calls == 0

    async def stalled():
        await asyncio.sleep(10)

    with (
        patch.object(app, "_frontend_available", new=stalled),
        patch("voiceclaw.installer_runtime.READINESS_SECONDS", 0.01),
    ):
        assert asyncio.run(request(app, "/health"))[0]["status"] == 503


def test_child_dying_during_probe_fails_readiness():
    app = runtime()

    async def probe():
        app.process.returncode = 1
        return True

    with patch.object(app, "_frontend_available", new=probe):
        assert asyncio.run(app.readiness()) == "frontend-unavailable"


@pytest.mark.parametrize("ready", [False, True])
def test_installer_websocket_admission_checks_local_health_before_upstream(ready):
    app = runtime()
    app.readiness = AsyncMock(return_value=None if ready else "frontend-unavailable")
    app.delegate = AsyncMock()
    messages = []

    async def send(message):
        messages.append(message)

    receive = AsyncMock()
    scope = {"type": "websocket", "path": "/v1/realtime"}
    asyncio.run(app(scope, receive, send))
    app.readiness.assert_awaited_once()
    if ready:
        app.delegate.assert_awaited_once_with(scope, receive, send)
        assert messages == []
    else:
        app.delegate.assert_not_awaited()
        assert messages == [{"type": "websocket.close", "code": 1013}]


@pytest.mark.parametrize("no_auth", [False, True])
def test_installer_main_constructs_the_explicit_application_before_serving(monkeypatch, no_auth):
    values = environment(no_auth=no_auth, VOICECLAW_PORT="18791")
    monkeypatch.setattr(installer_runtime.os, "geteuid", lambda: 65532)
    monkeypatch.setattr(installer_runtime.os, "getegid", lambda: 65532)
    with (
        patch.dict(os.environ, values, clear=True),
        patch("voiceclaw.installer_runtime.os.umask"),
        patch("uvicorn.run") as run,
    ):
        installer_runtime.main()
    run.assert_called_once()
    (application,) = run.call_args.args
    assert application.settings == settings(values)
    assert run.call_args.kwargs["port"] == 18791


@pytest.mark.parametrize(
    "overrides,identity",
    [
        ({"VOICECLAW_SPEECH_KEY": None}, (65532, 65532)),
        ({"VOICECLAW_GATEWAY_TOKEN": None}, (65532, 65532)),
        ({"VOICECLAW_SANDBOX_ID": "ACTUAL_SANDBOX_ID"}, (65532, 65532)),
        ({}, (0, 0)),
    ],
)
def test_incomplete_settings_or_unexpected_identity_prevent_activation(monkeypatch, capsys, overrides, identity):
    monkeypatch.setattr(installer_runtime.os, "geteuid", lambda: identity[0])
    monkeypatch.setattr(installer_runtime.os, "getegid", lambda: identity[1])
    with (
        patch.dict(os.environ, environment(**overrides), clear=True),
        patch("voiceclaw.installer_runtime.os.umask"),
        patch("uvicorn.run") as run,
        pytest.raises(SystemExit, match="^container-input-invalid$"),
    ):
        installer_runtime.main()
    run.assert_not_called()
    assert "fixture.bearer" not in repr(capsys.readouterr())


def test_installer_shutdown_cancels_its_executor_even_if_upstream_draining_fails():
    from unittest.mock import Mock

    app = runtime()
    app.executor = Mock()
    app.lifespan = SimpleNamespace(__aexit__=AsyncMock(side_effect=RuntimeError("fixture")))
    with patch("voiceclaw.installer_runtime._terminate_process", new=AsyncMock()) as terminate:
        asyncio.run(app.shutdown())
    app.executor.close.assert_called_once_with()
    app.executor.wait_closed.assert_called_once_with(2)
    terminate.assert_awaited_once()


def test_installer_profile_uses_model_routing_and_disables_provider_tokenization(tmp_path):
    resource = Path(installer_runtime.__file__).parent / "resources" / "nemoclaw_container_v1.yaml"
    config = load_config(resource, environ={})
    profile = config.selected_frontend
    assert profile.services.llm.supports_tokenize is False
    assert config.interaction.turn_routing_mode == "model"
    assert "work.delegate" in config.backend_profiles[config.default_backend].tool_copy

    plan = materialize_frontend_runtime(
        profile, tmp_path / "frontend", internal_endpoint="ws://127.0.0.1:7861/v1/realtime"
    )
    catalog = yaml.safe_load(plan.services_cloud_path.read_text())
    llm = catalog["llm"][profile.services.llm.id]
    assert llm["supports_tokenize"] is False
    assert llm["base_url"] == "https://integrate.api.nvidia.com/v1"
    assert llm["realtime_max_output_tokens"] == 2048


def test_installer_preset_constructs_the_real_upstream_facade_without_network(tmp_path):
    resource = Path(installer_runtime.__file__).parent / "resources" / "nemoclaw_container_v1.yaml"
    raw = yaml.safe_load(resource.read_text())
    state = tmp_path / "state"
    state.mkdir(mode=0o700)
    for role in ("llm", "asr", "tts"):
        assert raw["frontend_profiles"]["managed_nvidia"]["services"][role]["credential"] == {
            "env": "VOICECLAW_SPEECH_KEY"
        }
    raw["state"]["path"] = str(state / "state.db")
    config_file = tmp_path / "voiceclaw.yaml"
    config_file.write_text(yaml.safe_dump(raw))
    config = load_config(config_file, environ={})
    composition = compose_installer_backend(connection(), load_model_contract_catalog())
    with patch("voiceclaw.installer_executor.InstallerSandboxExecutor.execute") as execute:
        app = create_app(
            config,
            environ={"VOICECLAW_SPEECH_KEY": "speech-fixture", "REALTIME_UPSTREAM_API_KEY": "internal-fixture"},
            composition=composition,
        )
        assert app.state.voiceclaw_runtime is not None
        assert app.state.voiceclaw_state_store is not None
        execute.assert_not_called()
        assert "/v1/realtime" in {route.path for route in app.routes}

        async def close():
            async with app.router.lifespan_context(app):
                pass

        asyncio.run(close())


@pytest.mark.parametrize(
    "response,available",
    [
        (b'HTTP/1.1 200 OK\r\nContent-Length: 15\r\n\r\n{"status":"ok"}', True),
        (b'HTTP/1.1 503 Unavailable\r\nContent-Length: 15\r\n\r\n{"status":"ok"}', False),
        (b'HTTP/1.1 200 OK\r\nContent-Length: 16\r\n\r\n{"status":"bad"}', False),
        (b"HTTP/1.1 200 OK\r\nContent-Length: 2048\r\n\r\n", False),
        (b"HTTP/1.1 200 OK\r\nContent-Length: 15\r\nContent-Length: 15\r\n\r\n", False),
        (b"HTTP/1.1 200 OK\r\nX: " + b"x" * 4096 + b"\r\n\r\n", False),
        (b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n", False),
        (b"HTTP/1.1 200 OK\r\nContent-Length: 15\r\n\r\nshort", False),
    ],
)
def test_frontend_probe_is_bounded_and_only_gets_local_health(response, available):
    async def scenario():
        reader = asyncio.StreamReader(limit=4096)
        reader.feed_data(response)
        reader.feed_eof()
        writer = SimpleNamespace(
            write=lambda data: writes.append(data), drain=AsyncMock(), close=lambda: closes.append(True)
        )
        writes, closes = [], []
        with patch(
            "voiceclaw.installer_runtime.asyncio.open_connection", new=AsyncMock(return_value=(reader, writer))
        ) as connect:
            assert await runtime()._frontend_available() is available
        assert connect.call_args.args == ("127.0.0.1", 7861)
        assert connect.call_args.kwargs == {"limit": 4096}
        assert writes == [b"GET /health HTTP/1.1\r\nHost: 127.0.0.1:7861\r\nConnection: close\r\n\r\n"]
        assert closes == [True]

    asyncio.run(scenario())


def test_frontend_probe_deadline_cancels_stalled_read_and_closes_socket():
    async def scenario():
        reader = asyncio.StreamReader(limit=4096)
        closes = []
        writer = SimpleNamespace(write=lambda _: None, drain=AsyncMock(), close=lambda: closes.append(True))
        with (
            patch("voiceclaw.installer_runtime.asyncio.open_connection", new=AsyncMock(return_value=(reader, writer))),
            patch("voiceclaw.installer_runtime.FRONTEND_PROBE_SECONDS", 0.01),
        ):
            assert await runtime()._frontend_available() is False
        assert closes == [True]

    asyncio.run(scenario())


def test_healthcheck_probes_health_on_the_configured_port_with_empty_output(capsys):
    class Response:
        status = 200

        def read(self, maximum):
            assert maximum == 1
            return b""

    with patch("voiceclaw.installer_runtime.http.client.HTTPConnection") as client:
        client.return_value.getresponse.return_value = Response()
        with pytest.raises(SystemExit) as status:
            run_healthcheck(environment(VOICECLAW_PORT="18791"))
    assert status.value.code == 0
    assert client.call_args.args == ("127.0.0.1", 18791)
    assert client.call_args.kwargs["timeout"] == 15
    assert client.return_value.request.call_args.args == ("GET", "/health")
    assert capsys.readouterr() == ("", "")
    with (
        patch("voiceclaw.installer_runtime.http.client.HTTPConnection", side_effect=OSError("private-sentinel")),
        pytest.raises(SystemExit) as status,
    ):
        run_healthcheck(environment())
    assert status.value.code == 1
    assert capsys.readouterr() == ("", "")


def test_shutdown_stops_and_reaps_only_local_child():
    app = runtime()
    process = app.process

    def kill_group(pid, signal):
        assert pid == process.pid
        process.terminated = True
        process.returncode = 0

    with patch("voiceclaw.installer_runtime.os.killpg", side_effect=kill_group) as signals:
        asyncio.run(app.shutdown())
    assert [call.args[1] for call in signals.call_args_list] == [
        installer_runtime.signal.SIGTERM,
        installer_runtime.signal.SIGKILL,
    ]
    assert process.terminated
    assert app.process is None and app.delegate is None
    assert app.stopping


@pytest.mark.parametrize("initialization_fails", [False, True])
def test_bootstrap_uses_bundled_preset_and_limits_speech_key_to_trusted_child(
    tmp_path, monkeypatch, initialization_fails
):
    monkeypatch.setattr(installer_runtime, "RUNTIME_ROOT", tmp_path / "run")
    monkeypatch.setattr(installer_runtime, "STATE_PATH", tmp_path / "state" / "state.db")
    monkeypatch.setenv("OPERATOR_TOKEN", "unrelated-fixture-sentinel")
    app = runtime(VOICECLAW_SPEECH_KEY="speech-fixture-sentinel", VOICECLAW_PORT="18791")
    selected = {}

    @asynccontextmanager
    async def lifespan(_app):
        if initialization_fails:
            raise RuntimeError("private-init-sentinel")
        yield

    def create_app(config, *, environ, ui, composition):
        selected.update(config=config, environment=environ, ui=ui, composition=composition)
        return SimpleNamespace(router=SimpleNamespace(lifespan_context=lifespan))

    monkeypatch.setattr(installer_runtime, "create_app", create_app)
    with (
        patch("voiceclaw.installer_runtime.asyncio.create_subprocess_exec", return_value=Process()) as spawn,
        patch("voiceclaw.installer_runtime._terminate_process", new=AsyncMock()) as terminate,
    ):
        if initialization_fails:
            with pytest.raises(RuntimeError, match="private-init-sentinel"):
                asyncio.run(app.start())
            terminate.assert_awaited_once()
            assert app.initialized is False
            assert app.process is None and app.delegate is None
            assert asyncio.run(app.readiness()) == "frontend-unavailable"
            return
        asyncio.run(app.start())
    assert app.initialized
    from voiceclaw.adapters.openshell_fabric.committed_turn import OpenShellFabricAdapter

    assert type(selected["composition"].turn_backend) is OpenShellFabricAdapter
    assert selected["ui"] is True
    assert selected["config"].backend_profiles["nemoclaw"].kind == "openshell_fabric"
    assert selected["config"].backend_profiles["nemoclaw"].settings == {}
    assert selected["config"].server.port == 18791
    assert selected["config"].state.path == str(tmp_path / "state" / "state.db")
    assert (tmp_path / "state").is_dir()
    assert set(selected["environment"]) == {"VOICECLAW_SPEECH_KEY", "REALTIME_UPSTREAM_API_KEY"}
    child_env = spawn.call_args.kwargs["env"]
    assert child_env["NVIDIA_API_KEY"] == "speech-fixture-sentinel"
    assert "OPERATOR_TOKEN" not in child_env
    assert "fixture.bearer" not in repr(child_env) and "fixture.bearer" not in repr(selected)
    assert "speech-fixture-sentinel" not in repr({k: v for k, v in selected.items() if k != "environment"})
    assert spawn.call_args.args[0] == "/app/.venv/bin/python"
    assert spawn.call_args.kwargs["start_new_session"] is True
    assert spawn.call_args.kwargs["stdout"] == asyncio.subprocess.DEVNULL
    assert spawn.call_args.kwargs["stderr"] == asyncio.subprocess.DEVNULL
    for generated in tmp_path.rglob("*"):
        if generated.is_file():
            assert b"speech-fixture-sentinel" not in generated.read_bytes()


@pytest.mark.parametrize("no_auth", [False, True])
def test_installer_serves_packaged_browser_without_exposing_credentials_or_invoking_agent(
    tmp_path, monkeypatch, no_auth
):
    monkeypatch.setattr(installer_runtime, "RUNTIME_ROOT", tmp_path / "run")
    monkeypatch.setattr(installer_runtime, "STATE_PATH", tmp_path / "state.db")
    app = runtime(no_auth=no_auth, VOICECLAW_SPEECH_KEY="speech-fixture-sentinel")

    async def scenario():
        with (
            patch("voiceclaw.installer_runtime.asyncio.create_subprocess_exec", return_value=Process()),
            patch("voiceclaw.installer_runtime._terminate_process", new=AsyncMock()),
        ):
            await app.start()
            try:
                with patch.object(app.executor, "execute") as execute:
                    transport = httpx.ASGITransport(app=app)
                    async with httpx.AsyncClient(transport=transport, base_url="http://localhost:18790") as client:
                        for path, content_type in (
                            ("/", "text/html"),
                            ("/app.js", "text/javascript"),
                            ("/styles.css", "text/css"),
                            ("/marked.min.js", "text/javascript"),
                        ):
                            response = await client.get(path)
                            assert response.status_code == 200
                            assert response.headers["content-type"].startswith(content_type)
                            assert response.headers["cache-control"] == "no-store"
                            assert "connect-src 'self'" in response.headers["content-security-policy"]
                            assert response.headers["x-content-type-options"] == "nosniff"
                            assert response.content
                            assert "speech-fixture-sentinel" not in response.text
                            assert "fixture.bearer" not in response.text
                        for path in (
                            "/state.db",
                            "/voiceclaw.yaml",
                            "/missing.js",
                        ):
                            assert (await client.get(path)).status_code == 404
                    execute.assert_not_called()
            finally:
                await app.shutdown()

    asyncio.run(scenario())


def test_local_child_ignoring_sigterm_is_killed_and_reaped():
    async def scenario():
        child = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
            "print('ready',flush=True); time.sleep(60)",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            start_new_session=True,
        )
        assert await child.stdout.readline() == b"ready\n"
        async with asyncio.timeout(4):
            await installer_runtime._terminate_process(child)
        assert child.returncode == -installer_runtime.signal.SIGKILL

    asyncio.run(scenario())
