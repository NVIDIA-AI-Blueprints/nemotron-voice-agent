# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Runnable VoiceClaw Realtime facade with an optional packaged UI."""

from __future__ import annotations

import argparse
import asyncio
import ipaddress
import json
import logging
import os
import sqlite3
import tempfile
from collections.abc import Mapping
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from voiceclaw.adapters.state.sqlite import SqliteStateStore
from voiceclaw.composition import BackendComposition, compose_backends
from voiceclaw.config import ConfigurationError, CredentialReference, ListenerSecurity, VoiceClawConfig, load_config
from voiceclaw.frontend_runtime import (
    bind_nva_credential,
    materialize_frontend_runtime,
    resolve_credential_value,
)
from voiceclaw.interaction_profiles import (
    InteractionProfile,
    InteractionProfileCatalog,
    InteractionProfileError,
    load_interaction_profile_catalog,
)
from voiceclaw.model_contracts import ModelContractCatalog, ModelContractError, load_model_contract_catalog
from voiceclaw.ports.readiness import SelectedAgentReadinessError
from voiceclaw.realtime.auth import (
    ClientSecretIssuer,
    RealtimeAuthenticationError,
    bearer_from_headers,
    realtime_credential_from_headers,
)
from voiceclaw.realtime.facade import RealtimeTransport, VoiceClawRealtimeFacade
from voiceclaw.realtime.upstream import RealtimeUpstreamError, WebSocketRealtimeUpstream
from voiceclaw.ui import read_asset

try:
    from fastapi import FastAPI, Request, WebSocket
    from fastapi.responses import HTMLResponse, JSONResponse, Response
except ImportError:  # pragma: no cover - core-only installations do not import this module
    FastAPI = None  # type: ignore[assignment,misc]
    Request = Any  # type: ignore[assignment,misc]
    WebSocket = Any  # type: ignore[assignment,misc]
    HTMLResponse = Any  # type: ignore[assignment,misc]
    JSONResponse = Any  # type: ignore[assignment,misc]
    Response = Any  # type: ignore[assignment,misc]

_UI_ASSETS = {
    "app.js": "text/javascript; charset=utf-8",
    "marked.min.js": "text/javascript; charset=utf-8",
    "styles.css": "text/css; charset=utf-8",
}
_READINESS_TIMEOUT_SECONDS = 4.0
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _RuntimePrerequisites:
    model_contracts: ModelContractCatalog
    interaction_profiles: InteractionProfileCatalog
    interaction_profile: InteractionProfile
    adapters: BackendComposition
    issuer: ClientSecretIssuer | None
    upstream_bearer: str | None


class _DownstreamTransport(RealtimeTransport):
    """Adapt a Starlette WebSocket without leaking it into application code."""

    def __init__(self, websocket: Any) -> None:
        self._websocket = websocket

    async def receive(self) -> str:
        try:
            return await self._websocket.receive_text()
        except Exception as error:
            if type(error).__name__ == "WebSocketDisconnect":
                raise EOFError from error
            raise

    async def send(self, message: str) -> None:
        await self._websocket.send_text(message)


def _same_origin(headers: Mapping[str, str]) -> bool:
    origin = headers.get("origin")
    if origin is None:
        return True
    host = headers.get("host", "").lower()
    try:
        parsed = urlsplit(origin)
    except ValueError:
        return False
    return parsed.scheme in {"http", "https"} and parsed.netloc.lower() == host


def _public_key(config: VoiceClawConfig, environ: Mapping[str, str]) -> str | None:
    if config.server.auth_mode == "none":
        return None
    configured_sources = sum(value is not None for value in (config.server.api_key_env, config.server.api_key_file))
    if configured_sources != 1:
        raise ConfigurationError(
            "ephemeral auth must configure exactly one of server.api_key_env or server.api_key_file"
        )
    reference = CredentialReference(env=config.server.api_key_env, file=config.server.api_key_file)
    return resolve_credential_value(
        reference,
        source_environment=environ,
        label="public Realtime master key",
    )


def _public_issuer(config: VoiceClawConfig, environ: Mapping[str, str]) -> ClientSecretIssuer | None:
    """Resolve and validate the public credential exactly as serving will use it."""
    master_key = _public_key(config, environ)
    if master_key is None:
        return None
    try:
        return ClientSecretIssuer(master_key)
    except ValueError as error:
        raise ConfigurationError(str(error)) from error


def _upstream_bearer(config: VoiceClawConfig, environ: Mapping[str, str]) -> str | None:
    """Resolve the selected private frontend credential from its trusted reference."""
    realtime = config.realtime
    if realtime is None:
        return None
    if realtime.credential_env is not None and realtime.credential_file is not None:
        raise ConfigurationError("realtime upstream credential must select exactly one of env or file")
    reference = None
    if realtime.credential_env is not None or realtime.credential_file is not None:
        reference = CredentialReference(env=realtime.credential_env, file=realtime.credential_file)
    return resolve_credential_value(
        reference,
        source_environment=environ,
        label="realtime upstream credential",
    )


def _speech_delivery_capabilities(config: VoiceClawConfig) -> dict[str, str]:
    """Describe model-mediated speech and its validated client receipt boundary."""
    return {
        "acknowledgement": "model_mediated",
        "result": "model_mediated",
        "failure": "model_mediated",
        "playback_receipt": "conversation.item.truncate.v1",
        "playback_receipt_authority": "client_reported_validated",
        "response_done": "generation_only",
        "speech_floor": "waits_for_playback_receipt",
    }


def _with_listener_host(config: VoiceClawConfig, host: str | None) -> VoiceClawConfig:
    """Apply a CLI listener override before evaluating the authentication policy."""
    effective = config.server.host if host is None else host.strip()
    if not effective:
        raise ConfigurationError("server host must not be empty")
    return replace(config, server=replace(config.server, host=effective))


def _listener_port(config: VoiceClawConfig, port: int | None) -> int:
    """Resolve a CLI listener override with the same bounds as YAML configuration."""
    effective = config.server.port if port is None else port
    if isinstance(effective, bool) or not isinstance(effective, int) or not 1 <= effective <= 65535:
        raise ConfigurationError("server port must be between 1 and 65535")
    return effective


def _tls_listener_files(
    config: VoiceClawConfig,
    certfile: str,
    keyfile: str,
) -> tuple[str | None, str | None]:
    """Enforce the configured loopback, private-bridge, or TLS boundary."""
    certificate = certfile.strip()
    private_key = keyfile.strip()
    if bool(certificate) != bool(private_key):
        raise ConfigurationError("both TLS certificate and private key are required")
    host = config.server.host.strip().strip("[]")
    try:
        loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = host.lower() == "localhost"
    security = ListenerSecurity(config.server.listener_security)
    if security is ListenerSecurity.LOOPBACK and not loopback:
        raise ConfigurationError("server.listener_security loopback requires a loopback listener")
    if security is ListenerSecurity.TLS and not certificate:
        raise ConfigurationError("TLS certificate and private key are required when server.listener_security is tls")
    if certificate and (not Path(certificate).is_file() or not Path(private_key).is_file()):
        raise ConfigurationError("TLS certificate or private key file does not exist")
    return (certificate or None, private_key or None)


def _runtime_prerequisites(
    config: VoiceClawConfig,
    runtime_environ: Mapping[str, str],
    *,
    composition: BackendComposition | None = None,
    resolve_upstream: bool = True,
) -> _RuntimePrerequisites:
    try:
        model_contracts = load_model_contract_catalog(
            config.model_contracts.path,
            profile=config.model_contracts.profile,
        )
    except ModelContractError as error:
        raise ConfigurationError(str(error)) from error
    selected_backend = config.backend_profiles[config.default_backend]
    prose_overrides: dict[str, dict[str, Any]] = {}
    for tool_name, tool_copy in selected_backend.tool_copy.items():
        override: dict[str, Any] = {}
        if tool_copy.description is not None:
            override["description"] = tool_copy.description
        if tool_copy.properties:
            override["property_descriptions"] = dict(tool_copy.properties)
        prose_overrides[tool_name] = override
    try:
        interaction_profiles = config.interaction_profiles.inline_catalog or load_interaction_profile_catalog(
            config.interaction_profiles.path
        )
        interaction_profile = interaction_profiles.resolve(
            selected_backend.interaction_profile,
            prose_overrides=prose_overrides or None,
        )
    except InteractionProfileError as error:
        raise ConfigurationError(str(error)) from error
    adapters = composition or compose_backends(
        config,
        environ=runtime_environ,
        model_contracts=model_contracts,
    )
    adapters.require_realtime_runtime()
    adapters.validate_interaction_profile(interaction_profile)
    return _RuntimePrerequisites(
        model_contracts=model_contracts,
        interaction_profiles=interaction_profiles,
        interaction_profile=interaction_profile,
        adapters=adapters,
        issuer=_public_issuer(config, runtime_environ),
        upstream_bearer=_upstream_bearer(config, runtime_environ) if resolve_upstream else None,
    )


def validate_configuration(
    config: VoiceClawConfig,
    *,
    environ: Mapping[str, str] | None = None,
    bundled_nva_supervised: bool = False,
) -> None:
    """Validate the selected runtime path without mutating state or opening network resources.

    ``bundled_nva_supervised`` is reserved for the image supervisor after it
    has independently materialized and bound the bundled frontend. Ordinary
    package validation must resolve both the provider and private Realtime
    credentials that serving will require.
    """
    runtime_environ = dict(os.environ if environ is None else environ)
    selected_frontend = config.selected_frontend
    resolve_upstream = True
    if selected_frontend is not None:
        try:
            model_contracts = load_model_contract_catalog(
                config.model_contracts.path,
                profile=config.model_contracts.profile,
            )
        except ModelContractError as error:
            raise ConfigurationError(str(error)) from error
        with tempfile.TemporaryDirectory(prefix="voiceclaw-check-") as runtime_directory:
            plan = materialize_frontend_runtime(
                selected_frontend,
                Path(runtime_directory),
                internal_endpoint="ws://127.0.0.1:7861/v1/realtime",
                model_contracts=model_contracts,
            )
            if plan.launch_bundled_nva and not bundled_nva_supervised:
                bind_nva_credential(
                    {},
                    plan.nva_credential,
                    source_environment=runtime_environ,
                )
    _runtime_prerequisites(
        config,
        runtime_environ,
        resolve_upstream=resolve_upstream,
    )
    _validate_state_configuration(config.state.path)


def _state_parent_for_new_path(path: Path) -> Path:
    candidate = path.parent
    while not candidate.exists() and not candidate.is_symlink():
        parent = candidate.parent
        if parent == candidate:
            break
        candidate = parent
    return candidate


def _reject_state_symlinks(path: Path) -> None:
    absolute = path.absolute()
    for component in (absolute, *absolute.parents):
        if component.is_symlink():
            raise ConfigurationError(f"state.path must not contain symbolic links: {component}")


def _require_state_parent_access(path: Path) -> None:
    parent = path.parent if path.exists() else _state_parent_for_new_path(path)
    if not parent.is_dir():
        raise ConfigurationError(f"state.path parent is not a directory: {parent}")
    if not os.access(parent, os.W_OK | os.X_OK):
        raise ConfigurationError(f"state.path parent is not writable: {parent}")


def _validate_state_configuration(raw_path: str) -> None:
    """Validate SQLite readability/schema using only a disposable copy."""
    if raw_path == ":memory:":
        return
    path = Path(raw_path)
    _reject_state_symlinks(path)
    if not path.exists():
        _require_state_parent_access(path)
        return
    if not path.is_file():
        raise ConfigurationError(f"state.path must be a regular file: {path}")
    if not os.access(path, os.R_OK | os.W_OK):
        raise ConfigurationError(f"state.path must be readable and writable: {path}")
    try:
        resolved_path = path.resolve(strict=True)
    except OSError as error:
        raise ConfigurationError(f"state.path could not be resolved: {path}: {error}") from error
    _require_state_parent_access(resolved_path)

    source: sqlite3.Connection | None = None
    target: sqlite3.Connection | None = None
    try:
        source_uri = f"{resolved_path.as_uri()}?mode=ro"
        source = sqlite3.connect(source_uri, uri=True, timeout=5.0)
        integrity = source.execute("PRAGMA quick_check(1)").fetchone()
        if integrity is None or integrity[0] != "ok":
            detail = "no result" if integrity is None else str(integrity[0])
            raise ConfigurationError(f"state.path failed SQLite integrity validation: {detail}")
        with tempfile.TemporaryDirectory(prefix="voiceclaw-state-check-") as directory:
            snapshot_path = Path(directory) / "state.db"
            target = sqlite3.connect(snapshot_path)
            source.backup(target)
            target.close()
            target = None
            snapshot = SqliteStateStore(str(snapshot_path))
            snapshot.close()
    except ConfigurationError:
        raise
    except (OSError, RuntimeError, sqlite3.Error, ValueError) as error:
        raise ConfigurationError(f"state.path is not a usable VoiceClaw SQLite database: {error}") from error
    finally:
        if target is not None:
            target.close()
        if source is not None:
            source.close()


def create_app(
    config: VoiceClawConfig,
    *,
    environ: Mapping[str, str] | None = None,
    composition: BackendComposition | None = None,
    ui: bool = False,
) -> Any:
    """Create the Realtime API, optionally including the packaged browser UI."""
    if FastAPI is None:  # pragma: no cover - exercised from a core-only installation
        raise RuntimeError("install nemotron-voiceclaw[server] to run the facade")

    if config.realtime is None:
        raise ConfigurationError("realtime configuration is required to run the VoiceClaw facade")
    runtime_environ = dict(os.environ if environ is None else environ)
    prerequisites = _runtime_prerequisites(config, runtime_environ, composition=composition)
    model_contracts = prerequisites.model_contracts
    interaction_profiles = prerequisites.interaction_profiles
    interaction_profile = prerequisites.interaction_profile
    adapters = prerequisites.adapters
    issuer = prerequisites.issuer
    upstream_bearer = prerequisites.upstream_bearer

    _validate_state_configuration(config.state.path)
    try:
        state_store = SqliteStateStore(config.state.path)
    except (OSError, RuntimeError, sqlite3.Error, ValueError) as error:
        raise ConfigurationError(f"state.path could not be opened: {error}") from error
    runtime = adapters.create_runtime(
        backend_profile=config.default_backend,
        state_store=state_store,
        turn_routing_mode=config.interaction.turn_routing_mode,
        request_summary_character_limit=config.interaction.request_summary_character_limit,
        retained_request_limit=config.interaction.retained_request_limit,
        model_contracts=model_contracts,
        interaction_profile=interaction_profile,
        interaction_profile_schema=interaction_profiles.schema_version,
        interaction_profile_hash=interaction_profile.resolved_digest,
    )

    @asynccontextmanager
    async def lifespan(_app: Any):
        try:
            yield
        finally:
            state_store.close()

    app = FastAPI(title="VoiceClaw", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.state.voiceclaw_runtime = runtime
    app.state.voiceclaw_state_store = state_store

    def protected_headers(media_type: str) -> dict[str, str]:
        return {
            "Cache-Control": "no-store",
            "Content-Security-Policy": (
                "default-src 'self'; connect-src 'self'; img-src 'self' data:; "
                "media-src 'self' blob:; style-src 'self'; script-src 'self'; base-uri 'none'; frame-ancestors 'none'"
            ),
            "Referrer-Policy": "no-referrer",
            "X-Content-Type-Options": "nosniff",
            "Content-Type": media_type,
        }

    @app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse(
            {
                "status": "ok",
                "realtime_upstream": {
                    "configuration": "loaded",
                    "readiness": "checked_per_session",
                    "speech_delivery": _speech_delivery_capabilities(config),
                },
                "turn_backend": adapters.turn_status,
                "model_contracts": {
                    "schema": model_contracts.schema_version,
                    "profile": model_contracts.profile,
                    "digest": model_contracts.digest,
                },
                "interaction_profiles": {
                    "schema": interaction_profiles.schema_version,
                    "profile": interaction_profile.name,
                    "digest": interaction_profiles.digest,
                    "resolved_digest": interaction_profile.resolved_digest,
                },
                "authentication": {
                    "mode": config.server.auth_mode,
                    "client_secret_required": issuer is not None,
                },
                "state": "running",
            },
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/livez")
    async def live() -> Response:
        """Return a content-free process-liveness signal for service managers."""
        return Response(status_code=200, headers={"Cache-Control": "no-store"})

    @app.get("/readyz")
    async def ready() -> Response:
        """Attest scoped selected-agent access without starting user work."""
        try:
            async with asyncio.timeout(_READINESS_TIMEOUT_SECONDS):
                await adapters.check_selected_agent_readiness()
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            _LOGGER.warning("VoiceClaw readiness failed: selected_agent_readiness_timeout")
            return Response(status_code=503, headers={"Cache-Control": "no-store"})
        except SelectedAgentReadinessError as error:
            _LOGGER.warning("VoiceClaw readiness failed: %s", error.code)
            return Response(status_code=503, headers={"Cache-Control": "no-store"})
        except Exception:
            # Never log the exception object: adapter/provider failures can
            # carry credentials, target identities, or conversational data.
            _LOGGER.error("VoiceClaw readiness failed: readiness_internal_error")
            return Response(status_code=503, headers={"Cache-Control": "no-store"})
        return Response(status_code=200, headers={"Cache-Control": "no-store"})

    if ui:

        @app.get("/")
        async def index() -> HTMLResponse:
            return HTMLResponse(read_asset("index.html"), headers=protected_headers("text/html; charset=utf-8"))

        @app.get("/{asset_name}")
        async def asset(asset_name: str) -> Response:
            media_type = _UI_ASSETS.get(asset_name)
            if media_type is None:
                return Response(status_code=404)
            return Response(read_asset(asset_name), headers=protected_headers(media_type))

    @app.post("/v1/realtime/client_secrets")
    async def client_secrets(request: Request) -> JSONResponse:
        if issuer is None:
            return JSONResponse({"error": "authentication_disabled"}, status_code=404)
        try:
            credential = bearer_from_headers(request.headers)
        except RealtimeAuthenticationError:
            credential = None
        if credential is None or not issuer.is_master(credential):
            return JSONResponse({"error": "unauthorized"}, status_code=401, headers={"Cache-Control": "no-store"})
        secret = issuer.issue(lifetime_seconds=config.server.client_secret_lifetime_seconds)
        return JSONResponse(
            {"value": secret.value, "expires_at": secret.expires_at},
            headers={"Cache-Control": "no-store"},
        )

    @app.websocket("/v1/realtime")
    async def realtime(websocket: WebSocket) -> None:
        if not _same_origin(websocket.headers):
            await websocket.close(code=1008, reason="origin rejected")
            return
        try:
            supplied_credential = realtime_credential_from_headers(websocket.headers)
            if issuer is None:
                if supplied_credential is not None:
                    raise RealtimeAuthenticationError("authentication is disabled")
            else:
                issuer.authenticate_websocket(websocket.headers)
        except RealtimeAuthenticationError:
            await websocket.close(code=1008, reason="authentication failed")
            return
        requested_model = websocket.query_params.get("model")
        if requested_model not in {None, "", config.realtime.public_model}:
            await websocket.close(code=1008, reason="unknown VoiceClaw model")
            return
        offered = websocket.headers.get("sec-websocket-protocol", "")
        subprotocol = "realtime" if "realtime" in {part.strip() for part in offered.split(",")} else None
        await websocket.accept(subprotocol=subprotocol)
        downstream = _DownstreamTransport(websocket)
        try:
            async with WebSocketRealtimeUpstream(
                endpoint=config.realtime.upstream_endpoint,
                model=config.realtime.upstream_model,
                bearer=upstream_bearer,
                connect_timeout_seconds=config.realtime.connect_timeout_seconds,
                max_event_bytes=config.realtime.max_event_bytes,
            ) as upstream:
                facade = VoiceClawRealtimeFacade(
                    downstream=downstream,
                    upstream=upstream,
                    runtime=runtime,
                    model_contracts=model_contracts,
                    interaction_profile=interaction_profile,
                    public_model=config.realtime.public_model,
                    max_event_bytes=config.realtime.max_event_bytes,
                    bootstrap_timeout_seconds=config.realtime.bootstrap_timeout_seconds,
                    max_pending_speech=config.interaction.max_pending_speech,
                    context_character_budget=config.interaction.context_character_budget,
                )
                await facade.serve()
        except RealtimeUpstreamError:
            event = {
                "type": "error",
                "error": {
                    "type": "upstream_unavailable",
                    "code": "upstream_unavailable",
                    "message": "The configured realtime frontend is unavailable.",
                    "param": None,
                },
            }
            with suppress(Exception):
                await websocket.send_text(json.dumps(event, separators=(",", ":")))
        finally:
            with suppress(Exception):
                await websocket.close(code=1000)

    return app


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the VoiceClaw OpenAI Realtime facade")
    parser.add_argument("--config", default=os.getenv("VOICECLAW_CONFIG", ""), help="VoiceClaw YAML configuration")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--ssl-certfile", default=os.getenv("VOICECLAW_TLS_CERTFILE", ""))
    parser.add_argument("--ssl-keyfile", default=os.getenv("VOICECLAW_TLS_KEYFILE", ""))
    parser.add_argument("--log-level", default="info")
    parser.add_argument("--ui", action="store_true", help="serve the optional VoiceClaw browser UI")
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="validate the selected frontend, catalogs, credentials, and backend without serving",
    )
    return parser


def main() -> None:
    """Load strict YAML and run a single-process facade server."""
    arguments = _parser().parse_args()
    if not arguments.config:
        raise SystemExit("--config or VOICECLAW_CONFIG is required")
    try:
        config = _with_listener_host(load_config(Path(arguments.config)), arguments.host)
        port = _listener_port(config, arguments.port)
        certfile, keyfile = _tls_listener_files(config, arguments.ssl_certfile, arguments.ssl_keyfile)
        if arguments.check_config:
            validate_configuration(config)
    except (ConfigurationError, OSError, ValueError) as error:
        raise SystemExit(f"VoiceClaw configuration failed: {error}") from None
    if arguments.check_config:
        selected_frontend = config.default_frontend or "legacy_realtime"
        print(f"VoiceClaw configuration is valid: frontend={selected_frontend}, backend={config.default_backend}")
        return
    try:
        import uvicorn
    except ImportError as error:
        raise SystemExit("install nemotron-voiceclaw[server] to run the facade") from error
    try:
        app = create_app(config, ui=arguments.ui)
    except (ConfigurationError, OSError, ValueError) as error:
        raise SystemExit(f"VoiceClaw configuration failed: {error}") from None
    uvicorn.run(
        app,
        host=config.server.host,
        port=port,
        ssl_certfile=certfile,
        ssl_keyfile=keyfile,
        log_level=arguments.log_level,
        ws_max_size=config.realtime.max_event_bytes if config.realtime is not None else 16 * 1024 * 1024,
    )


if __name__ == "__main__":
    main()
