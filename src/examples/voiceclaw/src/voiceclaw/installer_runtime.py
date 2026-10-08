# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Foreground container-v1 bootstrap with observation-only readiness."""

from __future__ import annotations

import asyncio
import http.client
import os
import secrets
import signal
import stat
from contextlib import suppress
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from voiceclaw.config import load_config
from voiceclaw.frontend_runtime import materialize_frontend_runtime
from voiceclaw.installer_executor import InstallerSandboxExecutor, compose_installer_backend
from voiceclaw.installer_inputs import (
    CREDENTIAL_LIMIT,
    DATA_ROOT,
    ContainerInputError,
    ProfileInputs,
    _acl,
    load_connection,
    protected_directory,
    read_protected,
    strict_json,
    validate_bearer,
    validate_speech_key,
)
from voiceclaw.model_contracts import load_model_contract_catalog
from voiceclaw.server import create_app

PORT = 18790
RUNTIME_ROOT = Path("/run/voiceclaw-managed")
READINESS_SECONDS = 15
FRONTEND_PROBE_SECONDS = 2
FRONTEND_HEADER_LIMIT = 4096
FRONTEND_BODY_LIMIT = 1024


async def _terminate_process(process: asyncio.subprocess.Process | None) -> None:
    if process is None:
        return
    # The bundled frontend is launched in its own local process group. Reap
    # descendants even if their parent exits first; never signal the sandbox.
    if process.returncode is None:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        with suppress(TimeoutError):
            await asyncio.wait_for(process.wait(), 2)
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    await process.wait()


def _runtime_directory(path: Path) -> None:
    if path != RUNTIME_ROOT:
        with protected_directory(path.parent, root=RUNTIME_ROOT, mutable=True) as parent, suppress(FileExistsError):
            os.mkdir(path.name, 0o700, dir_fd=parent)
    with protected_directory(path, root=RUNTIME_ROOT):
        pass


def _state_path(path: Path) -> None:
    # A populated installer volume hides the image's state directory. Create only
    # missing application-owned parents; never repair unsafe existing inputs.
    current = DATA_ROOT
    for component in path.parent.relative_to(DATA_ROOT).parts:
        with protected_directory(current, mutable=True) as parent, suppress(FileExistsError):
            os.mkdir(component, 0o700, dir_fd=parent)
        current = current / component
    with protected_directory(path.parent) as parent:
        try:
            fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=parent)
        except FileNotFoundError:
            return
        try:
            m = os.fstat(fd)
            if (
                not stat.S_ISREG(m.st_mode)
                or (m.st_uid, m.st_gid, stat.S_IMODE(m.st_mode)) != (65532, 65532, 0o600)
                or _acl(fd)
            ):
                raise ContainerInputError
        finally:
            os.close(fd)


class InstallerApplication:
    """One explicit application; health cannot activate, repair or stop it."""

    def __init__(self, inputs: ProfileInputs, *, environment: dict[str, str]) -> None:
        """Require a complete protected input set before activation."""
        self.inputs = inputs
        self.environment = environment
        self.connection = load_connection(str(inputs.connection_file))
        if self.connection.credential_file == inputs.speech_file:
            raise ContainerInputError
        if self.connection.credential_file is not None:
            validate_bearer(read_protected(str(self.connection.credential_file)))
        validate_speech_key(read_protected(str(inputs.speech_file), maximum=CREDENTIAL_LIMIT))
        _state_path(inputs.state)
        self.process: asyncio.subprocess.Process | None = None
        self.delegate: Any | None = None
        self.lifespan: Any | None = None
        self.stopping = False
        self.initialized = False
        self.executor: InstallerSandboxExecutor | None = None

    async def start(self) -> None:
        """Start the bundled child once; never poll missing legacy projections."""
        for path in (RUNTIME_ROOT, RUNTIME_ROOT / "home", RUNTIME_ROOT / "cache", RUNTIME_ROOT / "config"):
            _runtime_directory(path)
        raw = yaml.safe_load(resources.files("voiceclaw.resources").joinpath("nemoclaw_container_v1.yaml").read_text())
        for role in ("llm", "asr", "tts"):
            raw["frontend_profiles"]["managed_nvidia"]["services"][role]["credential"]["file"] = str(
                self.inputs.speech_file
            )
        raw["state"]["path"] = str(self.inputs.state)
        effective = RUNTIME_ROOT / "config" / "voiceclaw.yaml"
        # Scratch is image/application-owned, separate from installer inputs.
        fd = os.open(effective, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
        with os.fdopen(fd, "w") as output:
            yaml.safe_dump(raw, output, sort_keys=False)
        config = load_config(effective, environ={})
        contracts = load_model_contract_catalog(config.model_contracts.path, profile=config.model_contracts.profile)
        plan = materialize_frontend_runtime(
            config.selected_frontend,
            RUNTIME_ROOT / "frontend",
            internal_endpoint="ws://127.0.0.1:7861/v1/realtime",
            model_contracts=contracts,
        )
        speech = validate_speech_key(read_protected(str(self.inputs.speech_file)))
        internal_key = secrets.token_urlsafe(32)
        child_environment = {
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "LANG": "C.UTF-8",
            "PYTHONUNBUFFERED": "1",
            "NLTK_DATA": "/usr/local/share/nltk_data",
            "HOME": str(RUNTIME_ROOT / "home"),
            "XDG_CACHE_HOME": str(RUNTIME_ROOT / "cache"),
            **plan.nva_environment,
            "NVIDIA_API_KEY": speech,
            "REALTIME_API_KEY": internal_key,
            "PIPELINE_TLS": "false",
            "UVICORN_WORKERS": "1",
        }
        try:
            self.executor = InstallerSandboxExecutor(self.connection)
            self.process = await asyncio.create_subprocess_exec(
                "/app/.venv/bin/python",
                "/app/src/realtime_server.py",
                "--host",
                "127.0.0.1",
                "--port",
                "7861",
                "--workers",
                "1",
                env=child_environment,
                start_new_session=True,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            self.delegate = create_app(
                config,
                environ={**self.environment, "REALTIME_UPSTREAM_API_KEY": internal_key},
                ui=False,
                composition=compose_installer_backend(self.connection, contracts, executor=self.executor),
            )
            self.lifespan = self.delegate.router.lifespan_context(self.delegate)
            await self.lifespan.__aenter__()
            self.initialized = True
        except BaseException:
            self.initialized = False
            if self.executor is not None:
                self.executor.close()
            await _terminate_process(self.process)
            self.process = None
            self.delegate = None
            raise

    async def _frontend_available(self) -> bool:
        """Observe the bundled listener without sessions, credentials or repair."""
        writer = None
        try:
            async with asyncio.timeout(FRONTEND_PROBE_SECONDS):
                reader, writer = await asyncio.open_connection("127.0.0.1", 7861, limit=FRONTEND_HEADER_LIMIT)
                writer.write(b"GET /health HTTP/1.1\r\nHost: 127.0.0.1:7861\r\nConnection: close\r\n\r\n")
                await writer.drain()
                header = await reader.readuntil(b"\r\n\r\n")
                if len(header) > FRONTEND_HEADER_LIMIT:
                    return False
                lines = header[:-4].split(b"\r\n")
                status = lines[0].split(b" ", 2)
                if len(status) < 2 or status[:2] not in ([b"HTTP/1.1", b"200"], [b"HTTP/1.0", b"200"]):
                    return False
                fields = {}
                for line in lines[1:]:
                    key, separator, value = line.partition(b":")
                    key = key.lower()
                    if not separator or key in fields:
                        return False
                    fields[key] = value.strip()
                length = fields.get(b"content-length", b"")
                if b"transfer-encoding" in fields or not length.isdigit() or len(length) > 4:
                    return False
                size = int(length)
                if not 0 < size <= FRONTEND_BODY_LIMIT:
                    return False
                body = await reader.readexactly(size)
                return strict_json(body, FRONTEND_BODY_LIMIT) == {"status": "ok"}
        except (OSError, TimeoutError, asyncio.IncompleteReadError, asyncio.LimitOverrunError, ContainerInputError):
            return False
        finally:
            if writer is not None:
                writer.close()

    def _local_initialized(self) -> bool:
        return (
            self.initialized
            and not self.stopping
            and self.delegate is not None
            and self.process is not None
            and self.process.returncode is None
        )

    async def readiness(self) -> str | None:
        """Check installation readiness; native agent qualification is separate."""
        if not self._local_initialized():
            return "frontend-unavailable"
        try:
            async with asyncio.timeout(READINESS_SECONDS):
                current = await asyncio.to_thread(load_connection, str(self.inputs.connection_file))
                if current != self.connection:
                    return "configuration-invalid"
                validate_speech_key(await asyncio.to_thread(read_protected, str(self.inputs.speech_file)))
                if current.credential_file is not None:
                    validate_bearer(await asyncio.to_thread(read_protected, str(current.credential_file)))
                if not await self._frontend_available() or not self._local_initialized():
                    return "frontend-unavailable"
        except ContainerInputError:
            return "configuration-invalid"
        except Exception:
            return "readiness-unknown"
        return None

    async def shutdown(self) -> None:
        """Bound facade draining plus local child cleanup inside 15 seconds."""
        self.stopping = True
        self.initialized = False
        executor = getattr(self, "executor", None)
        if executor is not None:
            executor.close()
        try:
            async with asyncio.timeout(5):
                if self.lifespan is not None:
                    await self.lifespan.__aexit__(None, None, None)
        except (Exception, asyncio.CancelledError):
            pass
        finally:
            if executor is not None:
                await asyncio.to_thread(executor.wait_closed, 2)
            await _terminate_process(self.process)
            self.process = None
            self.delegate = None

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        """Keep health separate from delegate admission and session capacity."""
        if scope["type"] == "lifespan":
            while True:
                message = await receive()
                if message["type"] == "lifespan.startup":
                    try:
                        await self.start()
                    except Exception:
                        await send({"type": "lifespan.startup.failed", "message": "container-startup-failed"})
                        return
                    await send({"type": "lifespan.startup.complete"})
                elif message["type"] == "lifespan.shutdown":
                    await self.shutdown()
                    await send({"type": "lifespan.shutdown.complete"})
                    return
        elif scope["type"] == "http" and scope.get("path") in {"/livez", "/readyz"}:
            reason = None if scope["path"] == "/livez" else await self.readiness()
            headers = [(b"cache-control", b"no-store")]
            if reason:
                headers.append((b"x-voiceclaw-reason", reason.encode("ascii")))
            await send({"type": "http.response.start", "status": 503 if reason else 200, "headers": headers})
            await send({"type": "http.response.body", "body": b""})
        elif scope["type"] == "websocket" and await self.readiness() is not None:
            await send({"type": "websocket.close", "code": 1013})
        elif self.delegate is not None:
            await self.delegate(scope, receive, send)
        elif scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1013})
        else:
            await send({"type": "http.response.start", "status": 503, "headers": []})
            await send({"type": "http.response.body", "body": b""})


def run_healthcheck() -> None:
    """Probe localhost readiness; Docker retains no diagnostic output."""
    connection = None
    code = 1
    try:
        connection = http.client.HTTPConnection("127.0.0.1", PORT, timeout=READINESS_SECONDS)
        connection.request("GET", "/readyz", headers={"Connection": "close"})
        response = connection.getresponse()
        if response.status == 200 and response.read(1) == b"":
            code = 0
    except Exception:
        pass
    finally:
        if connection is not None:
            connection.close()
    raise SystemExit(code)


def main() -> None:
    """Start only the complete explicit nonroot container profile."""
    os.umask(0o077)
    try:
        if (os.geteuid(), os.getegid()) != (65532, 65532):
            raise ContainerInputError
        inputs = ProfileInputs.from_environment(os.environ)
        environment = {
            name: os.environ[name]
            for name in (
                "VOICECLAW_RUNTIME_PROFILE",
                "VOICECLAW_INSTALL_CONTRACT",
                "VOICECLAW_AGENT_CONNECTION_FILE",
                "VOICECLAW_SPEECH_CREDENTIAL_FILE",
                "VOICECLAW_STATE_PATH",
            )
        }
        application = InstallerApplication(inputs, environment=environment)
    except Exception:
        raise SystemExit("container-input-invalid") from None
    import uvicorn

    uvicorn.run(
        application, host="0.0.0.0", port=PORT, access_log=False, log_level="warning", timeout_graceful_shutdown=5
    )
