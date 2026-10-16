# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Foreground Container Service Contract v1 runtime with local health."""

from __future__ import annotations

import asyncio
import http.client
import os
import secrets
import signal
from contextlib import suppress
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from voiceclaw.config import parse_config
from voiceclaw.frontend_runtime import materialize_frontend_runtime
from voiceclaw.installer_executor import InstallerSandboxExecutor, compose_installer_backend
from voiceclaw.installer_inputs import ContainerInputError, ContainerSettings, strict_json
from voiceclaw.model_contracts import load_model_contract_catalog
from voiceclaw.server import create_app

RUNTIME_ROOT = Path("/run/voiceclaw-managed")
STATE_PATH = Path("/var/lib/voiceclaw/state/state.db")
SPEECH_KEY_ENV = "VOICECLAW_SPEECH_KEY"
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


class InstallerApplication:
    """One explicit application; health cannot activate, repair or stop it."""

    def __init__(self, settings: ContainerSettings) -> None:
        """Bind the validated environment settings before activation."""
        self.settings = settings
        self.connection = settings.connection
        self.process: asyncio.subprocess.Process | None = None
        self.delegate: Any | None = None
        self.lifespan: Any | None = None
        self.stopping = False
        self.initialized = False
        self.executor: InstallerSandboxExecutor | None = None

    async def start(self) -> None:
        """Start the bundled child once and compose the upstream adapter."""
        for path in (RUNTIME_ROOT / "home", RUNTIME_ROOT / "cache", STATE_PATH.parent):
            os.makedirs(path, mode=0o700, exist_ok=True)
        raw = yaml.safe_load(resources.files("voiceclaw.resources").joinpath("nemoclaw_container_v1.yaml").read_text())
        raw["server"]["port"] = self.settings.port
        raw["state"]["path"] = str(STATE_PATH)
        config = parse_config(raw)
        contracts = load_model_contract_catalog(config.model_contracts.path, profile=config.model_contracts.profile)
        plan = materialize_frontend_runtime(
            config.selected_frontend,
            RUNTIME_ROOT / "frontend",
            internal_endpoint="ws://127.0.0.1:7861/v1/realtime",
            model_contracts=contracts,
        )
        internal_key = secrets.token_urlsafe(32)
        # The frontend child receives the speech key only, never the gateway token.
        child_environment = {
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "LANG": "C.UTF-8",
            "PYTHONUNBUFFERED": "1",
            "NLTK_DATA": "/usr/local/share/nltk_data",
            "HOME": str(RUNTIME_ROOT / "home"),
            "XDG_CACHE_HOME": str(RUNTIME_ROOT / "cache"),
            **plan.nva_environment,
            "NVIDIA_API_KEY": self.settings.speech_key,
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
                environ={SPEECH_KEY_ENV: self.settings.speech_key, "REALTIME_UPSTREAM_API_KEY": internal_key},
                ui=True,
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
        """Check local readiness; native agent qualification is separate."""
        if not self._local_initialized():
            return "frontend-unavailable"
        try:
            async with asyncio.timeout(READINESS_SECONDS):
                if not await self._frontend_available() or not self._local_initialized():
                    return "frontend-unavailable"
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
        elif scope["type"] == "http" and scope.get("path") == "/health":
            reason = await self.readiness()
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


def run_healthcheck(environ: dict[str, str] | None = None) -> None:
    """Probe localhost health; Docker retains no diagnostic output."""
    connection = None
    code = 1
    try:
        port = ContainerSettings.from_environment(os.environ if environ is None else environ).port
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=READINESS_SECONDS)
        connection.request("GET", "/health", headers={"Connection": "close"})
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
        application = InstallerApplication(ContainerSettings.from_environment(os.environ))
    except Exception:
        raise SystemExit("container-input-invalid") from None
    import uvicorn

    uvicorn.run(
        application,
        host="0.0.0.0",
        port=application.settings.port,
        access_log=False,
        log_level="warning",
        timeout_graceful_shutdown=5,
    )
