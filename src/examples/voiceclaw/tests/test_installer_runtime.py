# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause
# ruff: noqa: D100, D101, D102, D103, D107

import asyncio
import errno
import os
import stat
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
import yaml
from test_installer_inputs import connection, connection_value

from voiceclaw import installer_runtime
from voiceclaw.config import load_config
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


def runtime(error=None, *, no_auth=False):
    app = InstallerApplication.__new__(InstallerApplication)
    app.inputs = SimpleNamespace(
        connection_file=Path("/var/lib/voiceclaw/config/agent-connection.json"),
        speech_file=Path("/var/lib/voiceclaw/credentials/speech"),
    )
    app.connection = connection(connection_value(no_auth=no_auth))
    app.process = Process()
    app.stopping = False
    app.initialized = True
    app.delegate = SimpleNamespace(state=SimpleNamespace(voiceclaw_adapters=AgentHealth(error)))
    app.lifespan = None
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
@pytest.mark.parametrize(
    "fault", [None, "inputs", "changed", "initialization", "frontend", "probe", "probe-error", "stopping"]
)
def test_installation_health_is_local_observation_and_fail_closed(fault, no_auth):
    app = runtime(SelectedAgentReadinessError("selected_agent_readiness_unsupported"), no_auth=no_auth)
    if fault == "initialization":
        app.initialized = False
    if fault == "frontend":
        app.process.returncode = 1
    if fault == "stopping":
        app.stopping = True
    original_delegate = app.delegate
    with (
        patch(
            "voiceclaw.installer_runtime.load_connection", return_value=None if fault == "changed" else app.connection
        ),
        patch("voiceclaw.installer_runtime.read_protected", return_value=b"valid-key") as read,
        patch(
            "voiceclaw.installer_runtime.validate_speech_key",
            side_effect=installer_runtime.ContainerInputError() if fault == "inputs" else None,
        ),
        patch.object(
            app,
            "_frontend_available",
            new=AsyncMock(
                return_value=fault != "probe",
                side_effect=OSError("private-sentinel") if fault == "probe-error" else None,
            ),
        ),
    ):
        messages = asyncio.run(request(app, "/readyz"))
        live = asyncio.run(request(app, "/livez"))
    assert messages[0]["status"] == (200 if fault is None else 503)
    assert messages[1]["body"] == b""
    assert b"private-sentinel" not in repr(messages).encode()
    assert live[0]["status"] == 200
    assert app.delegate is original_delegate
    assert app.process.terminated is False
    assert app.delegate.state.voiceclaw_adapters.calls == 0

    if no_auth:
        assert all(call.args[0] == str(app.inputs.speech_file) for call in read.call_args_list)


@pytest.mark.parametrize("no_auth", [False, True])
def test_auth_mode_change_does_not_downgrade_or_rebind_running_application(no_auth):
    app = runtime(no_auth=no_auth)
    changed = connection(connection_value(no_auth=not no_auth))
    with (
        patch("voiceclaw.installer_runtime.load_connection", return_value=changed),
        patch("voiceclaw.installer_runtime.read_protected") as read,
        patch.object(app, "_frontend_available", new=AsyncMock()) as probe,
    ):
        assert asyncio.run(app.readiness()) == "configuration-invalid"
    assert app.connection.authentication_mode == ("none" if no_auth else "oidcBearer")
    read.assert_not_called()
    probe.assert_not_called()


def test_busy_does_not_fail_readiness_and_total_deadline():
    app = runtime()
    app.delegate.state.voiceclaw_admission = SimpleNamespace(can_accept=lambda: False)
    with (
        patch("voiceclaw.installer_runtime.load_connection", return_value=app.connection),
        patch("voiceclaw.installer_runtime.read_protected", return_value=b"valid-key"),
        patch.object(app, "_frontend_available", new=AsyncMock(return_value=True)),
    ):
        assert asyncio.run(request(app, "/readyz"))[0]["status"] == 200
    assert app.delegate.state.voiceclaw_adapters.calls == 0

    async def stalled():
        await asyncio.sleep(10)

    with (
        patch.object(app, "_frontend_available", new=stalled),
        patch("voiceclaw.installer_runtime.load_connection", return_value=app.connection),
        patch("voiceclaw.installer_runtime.read_protected", return_value=b"valid-key"),
        patch("voiceclaw.installer_runtime.READINESS_SECONDS", 0.01),
    ):
        assert asyncio.run(request(app, "/readyz"))[0]["status"] == 503


def test_child_dying_during_probe_fails_readiness():
    app = runtime()

    async def probe():
        app.process.returncode = 1
        return True

    with (
        patch("voiceclaw.installer_runtime.load_connection", return_value=app.connection),
        patch("voiceclaw.installer_runtime.read_protected", return_value=b"valid-key"),
        patch.object(app, "_frontend_available", new=probe),
    ):
        assert asyncio.run(app.readiness()) == "frontend-unavailable"


@pytest.mark.parametrize("ready", [False, True])
def test_installer_websocket_admission_checks_local_inputs_before_upstream(ready):
    app = runtime()
    app.readiness = AsyncMock(return_value=None if ready else "configuration-invalid")
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


def test_installer_main_constructs_the_explicit_application_before_serving(monkeypatch):
    values = {
        "VOICECLAW_RUNTIME_PROFILE": "nemoclaw-container-v1",
        "VOICECLAW_INSTALL_CONTRACT": "voiceclaw.nemoclaw.container.v1",
        "VOICECLAW_AGENT_CONNECTION_FILE": "/var/lib/voiceclaw/config/agent-connection.json",
        "VOICECLAW_SPEECH_CREDENTIAL_FILE": "/var/lib/voiceclaw/credentials/speech",
        "VOICECLAW_STATE_PATH": "/var/lib/voiceclaw/state/state.db",
    }
    monkeypatch.setattr(installer_runtime.os, "geteuid", lambda: 65532)
    monkeypatch.setattr(installer_runtime.os, "getegid", lambda: 65532)
    with (
        patch.dict(os.environ, values, clear=True),
        patch("voiceclaw.installer_runtime.os.umask"),
        patch("voiceclaw.installer_runtime.InstallerApplication") as application,
        patch("uvicorn.run") as run,
    ):
        installer_runtime.main()
    assert application.call_args.kwargs["environment"] == values
    run.assert_called_once()
    assert run.call_args.args == (application.return_value,)
    assert run.call_args.kwargs["port"] == 18790


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


def test_installer_preset_constructs_the_real_upstream_facade_without_network(tmp_path):
    resource = Path(installer_runtime.__file__).parent / "resources" / "nemoclaw_container_v1.yaml"
    raw = yaml.safe_load(resource.read_text())
    speech = tmp_path / "speech"
    speech.write_text("speech-fixture-sentinel")
    speech.chmod(0o600)
    state = tmp_path / "state"
    state.mkdir(mode=0o700)
    for role in ("llm", "asr", "tts"):
        raw["frontend_profiles"]["managed_nvidia"]["services"][role]["credential"]["file"] = str(speech)
    raw["state"]["path"] = str(state / "state.db")
    config_file = tmp_path / "voiceclaw.yaml"
    config_file.write_text(yaml.safe_dump(raw))
    config = load_config(config_file, environ={})
    composition = compose_installer_backend(connection(), load_model_contract_catalog())
    with patch("voiceclaw.installer_executor.InstallerSandboxExecutor.execute") as execute:
        app = create_app(config, environ={"REALTIME_UPSTREAM_API_KEY": "internal-fixture"}, composition=composition)
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


def test_healthcheck_probes_readyz_with_empty_output(capsys):
    class Response:
        status = 200

        def read(self, maximum):
            assert maximum == 1
            return b""

    with patch("voiceclaw.installer_runtime.http.client.HTTPConnection") as client:
        client.return_value.getresponse.return_value = Response()
        with pytest.raises(SystemExit) as status:
            run_healthcheck()
    assert status.value.code == 0
    assert client.call_args.kwargs["timeout"] == 15
    assert client.return_value.request.call_args.args == ("GET", "/readyz")
    assert capsys.readouterr() == ("", "")
    with (
        patch("voiceclaw.installer_runtime.http.client.HTTPConnection", side_effect=OSError("private-sentinel")),
        pytest.raises(SystemExit) as status,
    ):
        run_healthcheck()
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


@pytest.mark.parametrize("no_auth", [False, True])
@pytest.mark.parametrize("bad_speech", [False, True])
def test_initial_inputs_keep_speech_protected_and_only_require_selected_bearer(no_auth, bad_speech):
    c = connection(connection_value(no_auth=no_auth))
    inputs = SimpleNamespace(
        connection_file=Path("/var/lib/voiceclaw/config/agent-connection.json"),
        speech_file=Path("/var/lib/voiceclaw/credentials/speech"),
        state=Path("/var/lib/voiceclaw/state/state.db"),
    )
    reads = []

    def read(path, **_kwargs):
        reads.append(path)
        if path == str(inputs.speech_file):
            if bad_speech:
                raise installer_runtime.ContainerInputError
            return b"speech-fixture"
        assert not no_auth and path == str(c.credential_file)
        return b"fixture.bearer"

    with (
        patch("voiceclaw.installer_runtime.load_connection", return_value=c),
        patch("voiceclaw.installer_runtime.read_protected", side_effect=read),
        patch("voiceclaw.installer_runtime._state_path"),
    ):
        if bad_speech:
            with pytest.raises(installer_runtime.ContainerInputError):
                InstallerApplication(inputs, environment={})
        else:
            app = InstallerApplication(inputs, environment={})
            assert app.initialized is False
    assert str(inputs.speech_file) in reads
    assert reads == ([str(inputs.speech_file)] if no_auth else [str(c.credential_file), str(inputs.speech_file)])


def test_missing_complete_inputs_prevent_activation():
    with (
        patch("voiceclaw.installer_runtime.load_connection", side_effect=installer_runtime.ContainerInputError()),
        pytest.raises(installer_runtime.ContainerInputError),
    ):
        InstallerApplication(SimpleNamespace(connection_file=Path("/missing")), environment={})


@pytest.fixture
def application_state_root(tmp_path, monkeypatch):
    from voiceclaw.installer_inputs import protected_directory

    def directory(path, *, mutable=False):
        return protected_directory(path, root=tmp_path, uid=os.getuid(), gid=os.getgid(), mutable=mutable)

    def no_acl(*_args, **_kwargs):
        raise OSError(errno.ENOTSUP, "fixture ACL unavailable")

    monkeypatch.setattr(installer_runtime, "DATA_ROOT", tmp_path, raising=False)
    monkeypatch.setattr(installer_runtime, "protected_directory", directory)
    monkeypatch.setattr(os, "getxattr", no_acl, raising=False)
    return tmp_path


@pytest.mark.parametrize("relative", ["state/state.db", "state/history/state.db"])
def test_bootstrap_creates_missing_state_parents_without_rewriting_inputs(application_state_root, relative):
    root = application_state_root
    protected = []
    for name in ("config/agent-connection.json", "credentials/speech"):
        path = root / name
        path.parent.mkdir(mode=0o700)
        path.write_bytes(b"public-test-fixture")
        path.chmod(0o600)
        protected.append((path, path.stat(), path.read_bytes()))
    state = root / relative
    installer_runtime._state_path(state)
    assert not state.exists()
    for parent in (root / "state", state.parent):
        metadata = parent.stat()
        assert (metadata.st_uid, metadata.st_gid, stat.S_IMODE(metadata.st_mode)) == (
            os.getuid(),
            os.getgid(),
            0o700,
        )
    before = state.parent.stat()
    installer_runtime._state_path(state)
    assert state.parent.stat() == before
    for path, metadata, contents in protected:
        assert path.stat() == metadata
        assert path.read_bytes() == contents


@pytest.mark.parametrize("variant", ["symlink", "permissive", "regular-file"])
def test_bootstrap_rejects_unsafe_existing_state_parent_without_repair(application_state_root, tmp_path, variant):
    root = application_state_root
    parent = root / "state"
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o700)
    if variant == "symlink":
        parent.symlink_to(outside, target_is_directory=True)
    elif variant == "permissive":
        parent.mkdir(mode=0o755)
        parent.chmod(0o755)
    else:
        parent.write_bytes(b"preserve-invalid-parent")
    before = parent.lstat()
    with pytest.raises(installer_runtime.ContainerInputError, match="^container-input-invalid$"):
        installer_runtime._state_path(parent / "nested" / "state.db")
    assert parent.lstat() == before
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize("mode,valid", [(0o600, True), (0o644, False)])
def test_bootstrap_preserves_existing_state_and_rejects_unsafe_file(application_state_root, monkeypatch, mode, valid):
    root = application_state_root
    parent = root / "state"
    parent.mkdir(mode=0o700)
    state = parent / "state.db"
    state.write_bytes(b"existing-public-state-fixture")
    state.chmod(mode)
    before = state.stat()
    original_fstat = os.fstat

    def fixed_file_owner(fd):
        metadata = original_fstat(fd)
        if stat.S_ISREG(metadata.st_mode):
            return SimpleNamespace(st_mode=metadata.st_mode, st_uid=65532, st_gid=65532)
        return metadata

    # Only the file's expected container identity is modeled. Directory traversal,
    # no-follow opens, contents, and modes use the real host filesystem.
    monkeypatch.setattr(os, "fstat", fixed_file_owner)
    if valid:
        installer_runtime._state_path(state)
    else:
        with pytest.raises(installer_runtime.ContainerInputError):
            installer_runtime._state_path(state)
    assert state.stat() == before
    assert state.read_bytes() == b"existing-public-state-fixture"


@pytest.mark.parametrize("initialization_fails", [False, True])
def test_bootstrap_uses_bundled_preset_and_limits_speech_key_to_trusted_child(
    tmp_path, monkeypatch, initialization_fails
):
    from voiceclaw.installer_inputs import protected_directory

    def directory(path, *, root=None, mutable=False):
        return protected_directory(path, root=root or tmp_path, uid=os.getuid(), gid=os.getgid(), mutable=mutable)

    def no_acl(*_args, **_kwargs):
        raise OSError(errno.ENOTSUP, "fixture ACL unavailable")

    monkeypatch.setattr(installer_runtime, "RUNTIME_ROOT", tmp_path)
    monkeypatch.setattr(installer_runtime, "protected_directory", directory)
    monkeypatch.setattr(os, "getxattr", no_acl, raising=False)
    monkeypatch.setattr(installer_runtime, "read_protected", lambda _: b"speech-fixture-sentinel")
    monkeypatch.setenv("OPERATOR_TOKEN", "unrelated-fixture-sentinel")
    app = runtime()
    app.inputs.state = tmp_path / "state" / "state.db"
    app.environment = {"VOICECLAW_RUNTIME_PROFILE": "nemoclaw-container-v1"}
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
    assert selected["ui"] is False
    assert selected["config"].backend_profiles["nemoclaw"].kind == "openshell_fabric"
    assert selected["config"].backend_profiles["nemoclaw"].settings == {}
    child_env = spawn.call_args.kwargs["env"]
    assert child_env["NVIDIA_API_KEY"] == "speech-fixture-sentinel"
    assert "OPERATOR_TOKEN" not in child_env and "VOICECLAW_SPEECH_CREDENTIAL_FILE" not in child_env
    assert "speech-fixture-sentinel" not in repr(selected)
    assert spawn.call_args.args[0] == "/app/.venv/bin/python"
    assert spawn.call_args.kwargs["start_new_session"] is True
    assert spawn.call_args.kwargs["stdout"] == asyncio.subprocess.DEVNULL
    assert spawn.call_args.kwargs["stderr"] == asyncio.subprocess.DEVNULL
    for generated in tmp_path.rglob("*"):
        if generated.is_file():
            assert b"speech-fixture-sentinel" not in generated.read_bytes()


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
