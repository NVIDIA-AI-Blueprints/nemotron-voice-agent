# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Non-root VoiceClaw bootstrap for the NemoClaw managed-service contract."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import secrets
from collections.abc import Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any

from voiceclaw.adapters.nemoclaw.managed_projection import (
    MANAGED_LISTENER_PORT,
    ManagedProjection,
    ManagedProjectionError,
    ManagedProjectionPaths,
    load_managed_projection,
    materialize_managed_config,
)
from voiceclaw.adapters.nemoclaw.security import read_managed_credential
from voiceclaw.config import VoiceClawConfig
from voiceclaw.frontend_runtime import FrontendRuntimePlan, materialize_frontend_runtime
from voiceclaw.model_contracts import load_model_contract_catalog
from voiceclaw.realtime.upstream import RealtimeUpstreamError, WebSocketRealtimeUpstream
from voiceclaw.server import create_app

_MANAGED_UID = 65_532
_MANAGED_GID = 65_532
_INTERNAL_REALTIME_PORT = 7_861
_RUNTIME_ROOT = Path("/run/voiceclaw-managed")
_NVA_PYTHON = Path("/app/.venv/bin/python")
_NVA_SERVER = Path("/app/src/realtime_server.py")
_ACTIVATION_POLL_SECONDS = 1.0
_NVA_STARTUP_SECONDS = 180.0
_NVA_READINESS_SECONDS = 1.0
_NVA_READINESS_FAILURE_LIMIT = 3
_READINESS_REASON_HEADER = b"x-voiceclaw-reason"
_UNCONDITIONAL_DEACTIVATION = object()
_PROCESS_ENVIRONMENT = frozenset(
    {
        "CURL_CA_BUNDLE",
        "GRPC_DEFAULT_SSL_ROOTS_FILE_PATH",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "LD_LIBRARY_PATH",
        "NLTK_DATA",
        "PATH",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONUNBUFFERED",
        "REQUESTS_CA_BUNDLE",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "TZ",
    }
)
_LOGGER = logging.getLogger(__name__)


async def _close_readiness_lease() -> None:
    return None


class ManagedRuntimeError(RuntimeError):
    """A content-free managed bootstrap failure."""

    def __init__(self, code: str) -> None:
        """Create an error carrying one stable, content-free reason code."""
        self.code = code
        super().__init__(code)


def _require_managed_identity(
    *,
    expected_uid: int = _MANAGED_UID,
    expected_gid: int = _MANAGED_GID,
) -> None:
    if os.geteuid() != expected_uid or os.getegid() != expected_gid:
        raise ManagedRuntimeError("managed-identity-invalid")


def _safe_process_environment(source: Mapping[str, str]) -> dict[str, str]:
    return {name: source[name] for name in _PROCESS_ENVIRONMENT if name in source}


async def _wait_for_realtime_frontend(
    process: asyncio.subprocess.Process,
    *,
    plan: FrontendRuntimePlan,
    bearer: str,
    maximum_event_bytes: int,
    timeout_seconds: float,
) -> None:
    """Wait for a non-generative Realtime protocol bootstrap from the child."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_seconds
    retry_delay = 0.5
    while loop.time() < deadline:
        remaining = deadline - loop.time()
        try:
            await _probe_realtime_frontend(
                process,
                plan=plan,
                bearer=bearer,
                maximum_event_bytes=maximum_event_bytes,
                timeout_seconds=remaining,
            )
            return
        except asyncio.CancelledError:
            raise
        except ManagedRuntimeError:
            if process.returncode is not None:
                raise
            remaining = deadline - loop.time()
            if remaining > 0:
                await asyncio.sleep(min(retry_delay, remaining))
                retry_delay = min(retry_delay * 2, 5.0)
    raise ManagedRuntimeError("speech-unavailable")


async def _probe_realtime_frontend(
    process: asyncio.subprocess.Process,
    *,
    plan: FrontendRuntimePlan,
    bearer: str,
    maximum_event_bytes: int,
    timeout_seconds: float,
) -> None:
    """Perform one non-generative Realtime bootstrap against the active child."""
    if process.returncode is not None or timeout_seconds <= 0:
        raise ManagedRuntimeError("speech-unavailable")

    async def exchange() -> None:
        async with WebSocketRealtimeUpstream(
            endpoint=plan.upstream_endpoint,
            model=plan.upstream_model,
            bearer=bearer,
            connect_timeout_seconds=min(3.0, timeout_seconds),
            max_event_bytes=maximum_event_bytes,
        ) as upstream:
            events = []
            for _index in range(2):
                raw = await upstream.receive_text()
                event = json.loads(raw)
                if not isinstance(event, dict) or not isinstance(event.get("type"), str):
                    raise RealtimeUpstreamError("realtime bootstrap event is malformed")
                events.append(event["type"])
            if events != ["session.created", "conversation.created"]:
                raise RealtimeUpstreamError("realtime bootstrap sequence is invalid")

    try:
        await asyncio.wait_for(exchange(), timeout=timeout_seconds)
        if process.returncode is not None:
            raise ManagedRuntimeError("speech-unavailable")
    except asyncio.CancelledError:
        raise
    except ManagedRuntimeError:
        raise
    except Exception as error:
        raise ManagedRuntimeError("speech-unavailable") from error


async def _terminate_process(process: asyncio.subprocess.Process | None) -> None:
    if process is None or process.returncode is not None:
        return
    try:
        process.terminate()
    except ProcessLookupError:
        await process.wait()
        return
    try:
        await asyncio.wait_for(process.wait(), timeout=10.0)
    except TimeoutError:
        with suppress(ProcessLookupError):
            process.kill()
        await process.wait()


class ManagedRuntimeController:
    """Activate and replace the product app behind one stable ASGI listener."""

    def __init__(
        self,
        *,
        paths: ManagedProjectionPaths | None = None,
        runtime_root: Path = _RUNTIME_ROOT,
        source_environment: Mapping[str, str] | None = None,
        nva_command: Sequence[str] | None = None,
        poll_seconds: float = _ACTIVATION_POLL_SECONDS,
        nva_startup_seconds: float = _NVA_STARTUP_SECONDS,
    ) -> None:
        """Create a controller around fixed projection and runtime paths."""
        self._paths = paths or ManagedProjectionPaths()
        self._runtime_root = runtime_root
        self._effective_config = runtime_root / "config/voiceclaw.yaml"
        self._frontend_runtime = runtime_root / "frontend"
        self._source_environment = dict(os.environ if source_environment is None else source_environment)
        self._nva_command = tuple(
            nva_command
            or (
                str(_NVA_PYTHON),
                str(_NVA_SERVER),
                "--host",
                "127.0.0.1",
                "--port",
                str(_INTERNAL_REALTIME_PORT),
                "--workers",
                "1",
            )
        )
        self._poll_seconds = poll_seconds
        self._nva_startup_seconds = nva_startup_seconds
        self._lock = asyncio.Lock()
        self._frontend_probe_lock = asyncio.Lock()
        self._delegate: Any | None = None
        self._delegate_lifespan: Any | None = None
        self._nva: asyncio.subprocess.Process | None = None
        self._frontend_plan_state: FrontendRuntimePlan | None = None
        self._frontend_bearer: str | None = None
        self._frontend_maximum_event_bytes: int | None = None
        self._frontend_readiness_failures = 0
        self._projection_digest: str | None = None
        self._reason = "starting"
        self._task: asyncio.Task[None] | None = None
        self._stopping = False

    async def start(self) -> None:
        """Start activation without delaying the outer listener startup."""
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="voiceclaw-managed-activation")

    async def shutdown(self) -> None:
        """Stop activation and close the nested app and frontend process."""
        self._stopping = True
        task = self._task
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        await self._deactivate("starting")

    async def snapshot(self) -> tuple[Any | None, str]:
        """Return the active app and stable readiness reason atomically."""
        while True:
            async with self._lock:
                delegate = self._delegate
                reason = self._reason
                nva = self._nva
            if delegate is None or nva is None or nva.returncode is None:
                return delegate, reason
            if await self._deactivate("speech-unavailable", expected_process=nva):
                return None, "speech-unavailable"

    async def readiness_snapshot(self) -> tuple[Any | None, str]:
        """Return an app only after an ongoing non-generative frontend probe."""
        async with self._frontend_probe_lock:
            for _attempt in range(2):
                async with self._lock:
                    delegate = self._delegate
                    reason = self._reason
                    nva = self._nva
                    plan = self._frontend_plan_state
                    bearer = self._frontend_bearer
                    maximum_event_bytes = self._frontend_maximum_event_bytes
                if delegate is None:
                    return None, reason
                if nva is None or plan is None or bearer is None or maximum_event_bytes is None:
                    if await self._deactivate("speech-unavailable", expected_process=nva):
                        return None, "speech-unavailable"
                    continue
                if nva.returncode is not None:
                    if await self._deactivate("speech-unavailable", expected_process=nva):
                        return None, "speech-unavailable"
                    continue
                admission = delegate.state.voiceclaw_admission
                lease = await admission.acquire(_close_readiness_lease)
                if lease is None:
                    return delegate, reason
                probe_failed = False
                failure_limit_reached = False
                try:
                    await _probe_realtime_frontend(
                        nva,
                        plan=plan,
                        bearer=bearer,
                        maximum_event_bytes=maximum_event_bytes,
                        timeout_seconds=_NVA_READINESS_SECONDS,
                    )
                except asyncio.CancelledError:
                    raise
                except ManagedRuntimeError:
                    probe_failed = True
                    async with self._lock:
                        if self._delegate is not delegate or self._nva is not nva:
                            continue
                        self._frontend_readiness_failures += 1
                        failure_limit_reached = self._frontend_readiness_failures >= _NVA_READINESS_FAILURE_LIMIT
                finally:
                    await admission.release(lease)
                if probe_failed:
                    if failure_limit_reached:
                        await self._deactivate("speech-unavailable", expected_process=nva)
                    return None, "speech-unavailable"
                async with self._lock:
                    if self._delegate is delegate and self._nva is nva:
                        self._frontend_readiness_failures = 0
                        return delegate, reason
            return None, "starting"

    async def _run(self) -> None:
        while not self._stopping:
            try:
                await self._poll_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                try:
                    await self._deactivate("configuration-invalid")
                except Exception:
                    _LOGGER.error("VoiceClaw managed fail-closed cleanup failed")
                _LOGGER.error("VoiceClaw managed projection poll failed: configuration-invalid")
            if not self._stopping:
                await asyncio.sleep(self._poll_seconds)

    async def _poll_once(self) -> None:
        await self.snapshot()
        try:
            projection = await asyncio.to_thread(load_managed_projection, self._paths)
        except ManagedProjectionError as error:
            reason = "starting" if error.code == "projection-unavailable" else "configuration-invalid"
            await self._deactivate(reason)
            return
        if self._projection_digest == projection.projection_digest and self._delegate is not None:
            return
        await self._deactivate("starting")
        try:
            await self._activate(projection)
        except asyncio.CancelledError:
            raise
        except ManagedRuntimeError as error:
            await self._deactivate(error.code)
            _LOGGER.error("VoiceClaw managed activation failed: %s", error.code)
        except Exception:
            await self._deactivate("speech-unavailable")
            _LOGGER.error("VoiceClaw managed activation failed: speech-unavailable")

    async def _activate(self, projection: ManagedProjection) -> None:
        self._prepare_runtime_root()
        try:
            config = await asyncio.to_thread(materialize_managed_config, projection, self._effective_config)
            plan = await asyncio.to_thread(self._frontend_plan, config)
        except ManagedProjectionError as error:
            reason = "runtime-path-invalid" if error.code.startswith("runtime-path-") else "configuration-invalid"
            raise ManagedRuntimeError(reason) from error
        except Exception as error:
            raise ManagedRuntimeError("configuration-invalid") from error
        if not plan.launch_bundled_nva:
            raise ManagedRuntimeError("configuration-invalid")
        try:
            speech_credential = await asyncio.to_thread(read_managed_credential, projection.paths.speech_credential)
        except Exception as error:
            raise ManagedRuntimeError("configuration-invalid") from error
        internal_key = secrets.token_urlsafe(32)
        environment = _safe_process_environment(self._source_environment)
        environment.update(plan.nva_environment)
        environment.update(
            {
                "HOME": str(self._runtime_root / "home"),
                "NVIDIA_API_KEY": speech_credential,
                "PIPELINE_TLS": "false",
                "REALTIME_API_KEY": internal_key,
                "UVICORN_WORKERS": "1",
                "XDG_CACHE_HOME": str(self._runtime_root / "cache"),
            }
        )
        try:
            process = await asyncio.create_subprocess_exec(
                *self._nva_command,
                env=environment,
                stdin=asyncio.subprocess.DEVNULL,
            )
        except Exception as error:
            raise ManagedRuntimeError("speech-unavailable") from error
        lifespan: Any | None = None
        lifespan_entered = False
        try:
            await _wait_for_realtime_frontend(
                process,
                plan=plan,
                bearer=internal_key,
                maximum_event_bytes=config.realtime.max_event_bytes,
                timeout_seconds=self._nva_startup_seconds,
            )
            try:
                app = create_app(config, environ={"REALTIME_UPSTREAM_API_KEY": internal_key}, ui=False)
            except Exception as error:
                raise ManagedRuntimeError("configuration-invalid") from error
            lifespan = app.router.lifespan_context(app)
            await lifespan.__aenter__()
            lifespan_entered = True
            async with self._lock:
                if self._stopping:
                    raise asyncio.CancelledError
                self._delegate = app
                self._delegate_lifespan = lifespan
                self._nva = process
                self._frontend_plan_state = plan
                self._frontend_bearer = internal_key
                self._frontend_maximum_event_bytes = config.realtime.max_event_bytes
                self._frontend_readiness_failures = 0
                self._projection_digest = projection.projection_digest
                self._reason = "ready"
        except BaseException:
            if lifespan is not None and lifespan_entered:
                with suppress(Exception):
                    await lifespan.__aexit__(None, None, None)
            await _terminate_process(process)
            raise

    def _frontend_plan(self, config: VoiceClawConfig) -> FrontendRuntimePlan:
        profile = config.selected_frontend
        if profile is None:
            raise ManagedRuntimeError("configuration-invalid")
        model_contracts = load_model_contract_catalog(
            config.model_contracts.path,
            profile=config.model_contracts.profile,
        )
        return materialize_frontend_runtime(
            profile,
            self._frontend_runtime,
            internal_endpoint=f"ws://127.0.0.1:{_INTERNAL_REALTIME_PORT}/v1/realtime",
            model_contracts=model_contracts,
        )

    def _prepare_runtime_root(self) -> None:
        for path in (self._runtime_root, self._runtime_root / "home", self._runtime_root / "cache"):
            path.mkdir(mode=0o700, parents=True, exist_ok=True)
            metadata = path.stat(follow_symlinks=False)
            if path.is_symlink() or metadata.st_uid != os.geteuid() or metadata.st_gid != os.getegid():
                raise ManagedRuntimeError("runtime-path-invalid")
            os.chmod(path, 0o700)

    async def _deactivate(
        self,
        reason: str,
        *,
        expected_process: object = _UNCONDITIONAL_DEACTIVATION,
    ) -> bool:
        async with self._lock:
            if expected_process is not _UNCONDITIONAL_DEACTIVATION and self._nva is not expected_process:
                return False
            app = self._delegate
            lifespan = self._delegate_lifespan
            process = self._nva
            self._delegate = None
            self._delegate_lifespan = None
            self._nva = None
            self._frontend_plan_state = None
            self._frontend_bearer = None
            self._frontend_maximum_event_bytes = None
            self._frontend_readiness_failures = 0
            self._projection_digest = None
            self._reason = reason
        if app is not None:
            admission = app.state.voiceclaw_admission
            await admission.begin_drain()
        if lifespan is not None:
            with suppress(Exception):
                await lifespan.__aexit__(None, None, None)
        await _terminate_process(process)
        return True


class ManagedApplication:
    """Stable liveness surface with an atomically replaceable product app."""

    def __init__(self, controller: ManagedRuntimeController | None = None) -> None:
        """Create a stable ASGI surface around one managed controller."""
        self.controller = controller or ManagedRuntimeController()

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        """Serve liveness directly and delegate all product traffic when ready."""
        scope_type = scope["type"]
        if scope_type == "lifespan":
            await self._lifespan(receive, send)
            return
        if scope_type == "http" and scope.get("path") == "/livez":
            await _empty_http(send, 200)
            return
        if scope_type == "http" and scope.get("path") == "/readyz":
            delegate, reason = await self.controller.readiness_snapshot()
        else:
            delegate, reason = await self.controller.snapshot()
        if delegate is None:
            if scope_type == "websocket":
                await send({"type": "websocket.close", "code": 1013, "reason": "service unavailable"})
            elif scope_type == "http":
                await _empty_http(send, 503, reason)
            return
        await delegate(scope, receive, send)

    async def _lifespan(self, receive: Any, send: Any) -> None:
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                await self.controller.start()
                await send({"type": "lifespan.startup.complete"})
            elif message["type"] == "lifespan.shutdown":
                await self.controller.shutdown()
                await send({"type": "lifespan.shutdown.complete"})
                return


async def _empty_http(send: Any, status: int, reason: str | None = None) -> None:
    headers = [(b"cache-control", b"no-store")]
    if reason is not None:
        headers.append((_READINESS_REASON_HEADER, reason.encode("ascii")))
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": headers,
        }
    )
    await send({"type": "http.response.body", "body": b""})


def run_healthcheck() -> None:
    """Probe the fixed managed liveness endpoint without proxy discovery."""
    import http.client

    connection = http.client.HTTPConnection("127.0.0.1", MANAGED_LISTENER_PORT, timeout=2.5)
    try:
        connection.request("GET", "/livez", headers={"Connection": "close"})
        response = connection.getresponse()
        response.read()
        if response.status != 200:
            raise ManagedRuntimeError("managed-liveness-failed")
    finally:
        connection.close()


def main(argv: Sequence[str] | None = None) -> None:
    """Run the fixed managed listener; projection arrives after process start."""
    parser = argparse.ArgumentParser(prog="voiceclaw-runtime")
    parser.add_argument("command", choices=("serve",))
    parser.parse_args(argv)
    try:
        _require_managed_identity()
    except ManagedRuntimeError as error:
        raise SystemExit(f"VoiceClaw managed runtime failed: {error.code}") from error
    import uvicorn

    uvicorn.run(
        ManagedApplication(),
        host="0.0.0.0",
        port=MANAGED_LISTENER_PORT,
        access_log=False,
        log_level="info",
        timeout_graceful_shutdown=10,
    )


__all__ = [
    "ManagedApplication",
    "ManagedRuntimeController",
    "ManagedRuntimeError",
    "main",
    "run_healthcheck",
]


if __name__ == "__main__":
    main()
