# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Process-wide state for the Frontend/Backend Verdict sessions.

This is the prototype server's lifespan (``voice/server.py`` ``build_app``), run once
per configuration inside this repository's server instead of in its own FastAPI app:
one event log, routing sink and filler log, one set of speech services, one LLM
client per role, the speech warm-up, and the ``server.max_sessions`` cap. It calls
the copied prototype functions (``build_clients``, ``_agent_factory``, ``_AppState``)
rather than re-implementing them.

Two values come from this repository instead of the prototype YAML:

* the LLM endpoints (``model`` and ``base_url`` of each role) come from the service
  catalog entries the Realtime route selected (``llm`` = frontend, ``thinker-llm`` =
  backend); every request parameter stays the prototype's (``text/config/agent.yaml``);
* the WebSocket keepalive is uvicorn's, so it is checked against the profile.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from loguru import logger

from examples.frontend_backend_verdict.voice.agent.filler import FillerLog
from examples.frontend_backend_verdict.voice.agent.runner import AgentClients
from examples.frontend_backend_verdict.voice.agent.sinks import EventLog, SessionRoutingSink
from examples.frontend_backend_verdict.voice.config import PACKAGE_DIR, VoiceConfig, load_voice_config
from examples.frontend_backend_verdict.voice.server import ServerOptions, _agent_factory, _AppState, build_clients
from examples.frontend_backend_verdict.voice.speech.factory import build_speech_services
from examples.frontend_backend_verdict.voice.speech.ports import SpeechServices
from examples.frontend_backend_verdict.voice.wire import server_events as ev

PROFILES_DIR = PACKAGE_DIR / "config" / "profiles"
PROFILE_ENV = "FBV_PROFILE"
#: Prototype profiles this example serves (plan D11). Each extends tau3_eval.yaml, so each
#: has the 800 ms, server-owned turn detection of the registry's session defaults.
SHIPPED_PROFILES = (
    "tau3_eval_frontend_verdict_speak_history",
    "tau3_eval_frontend_verdict_speak",
    "tau3_eval_frontend_verdict",
    "tau3_eval_backend_history",
    "tau3_eval_backend_history_noguide",
    "tau3_eval_normalization",
    "tau3_eval",
)
DEFAULT_PROFILE = SHIPPED_PROFILES[0]
#: uvicorn's defaults, used by src/server.py when UVICORN_WS_PING_* are unset.
UVICORN_DEFAULT_PING_S = 20.0


class ProfileError(RuntimeError):
    """The selected profile cannot be served by this deployment."""


def profile_name(environ: Mapping[str, str] | None = None) -> str:
    """The profile selected by ``FBV_PROFILE`` (default: the frontend-verdict + history arm)."""
    env = os.environ if environ is None else environ
    name = (env.get(PROFILE_ENV) or DEFAULT_PROFILE).strip()
    if name.endswith(".yaml"):
        name = name[: -len(".yaml")]
    if name not in SHIPPED_PROFILES:
        raise ProfileError(f"{PROFILE_ENV}={name!r} is not served by this example; choose one of {SHIPPED_PROFILES}")
    return name


def profile_path(name: str) -> Path:
    """The YAML file of a shipped profile."""
    return PROFILES_DIR / f"{name}.yaml"


def _float_env(env: Mapping[str, str], key: str, default: float) -> float:
    raw = (env.get(key) or "").strip()
    return float(raw) if raw else default


def keepalive_mismatch(config: VoiceConfig, environ: Mapping[str, str] | None = None) -> str:
    """A message when the server's WebSocket keepalive differs from the profile's, else ``""``.

    The prototype passes ``ws_ping_interval=server.ws_ping_interval_s or None`` and
    ``ws_ping_timeout=server.ws_ping_timeout_s`` to uvicorn; ``src/server.py`` passes
    ``UVICORN_WS_PING_INTERVAL`` / ``UVICORN_WS_PING_TIMEOUT`` the same way.
    """
    env = os.environ if environ is None else environ
    interval = _float_env(env, "UVICORN_WS_PING_INTERVAL", UVICORN_DEFAULT_PING_S) or None
    timeout = _float_env(env, "UVICORN_WS_PING_TIMEOUT", UVICORN_DEFAULT_PING_S)
    wanted_interval = config.server.ws_ping_interval_s or None
    wanted_timeout = config.server.ws_ping_timeout_s
    if (interval, timeout) == (wanted_interval, wanted_timeout):
        return ""
    return (
        f"WebSocket keepalive differs from the profile: server ping interval={interval} timeout={timeout}, "
        f"profile ws_ping_interval_s={config.server.ws_ping_interval_s} ws_ping_timeout_s={wanted_timeout}; "
        f"set UVICORN_WS_PING_INTERVAL={config.server.ws_ping_interval_s:g} "
        f"UVICORN_WS_PING_TIMEOUT={wanted_timeout:g}"
    )


@dataclass(frozen=True, slots=True)
class LLMEndpoints:
    """Model and base URL per LLM role, from the Realtime route's catalog selection."""

    frontend_model: str = ""
    frontend_base_url: str = ""
    backend_model: str = ""
    backend_base_url: str = ""

    @classmethod
    def from_body(cls, body: Mapping[str, Any]) -> LLMEndpoints:
        """Read the hydrated catalog fields (``model_id``/``base_url``, ``thinker_*``)."""
        return cls(
            frontend_model=str(body.get("model_id") or ""),
            frontend_base_url=str(body.get("base_url") or ""),
            backend_model=str(body.get("thinker_model_id") or ""),
            backend_base_url=str(body.get("thinker_base_url") or ""),
        )


def apply_llm_endpoints(config: VoiceConfig, endpoints: LLMEndpoints) -> VoiceConfig:
    """Return ``config`` with the catalog's model and base URL per role; other LLM settings unchanged."""
    agent = config.agent

    def _llm(llm: Any, model: str, base_url: str) -> Any:
        return replace(llm, model=model or llm.model, base_url=base_url or llm.base_url)

    frontend = replace(
        agent.frontend, llm=_llm(agent.frontend.llm, endpoints.frontend_model, endpoints.frontend_base_url)
    )
    backend = replace(agent.backend, llm=_llm(agent.backend.llm, endpoints.backend_model, endpoints.backend_base_url))
    return replace(config, agent=replace(agent, frontend=frontend, backend=backend))


class VerdictRuntime:
    """The prototype app state for one (profile, LLM endpoints) configuration."""

    def __init__(
        self,
        config: VoiceConfig,
        *,
        profile: str,
        options: ServerOptions | None = None,
        services: SpeechServices | None = None,
        clients: AgentClients | None = None,
    ) -> None:
        """Build the prototype's shared objects exactly as ``voice/server.py`` ``build_app`` does.

        ``options``, ``services`` and ``clients`` may be injected, as ``build_app`` allows (tests).
        """
        self.profile = profile
        self.config = config
        self.state = _AppState(config=config, options=options or ServerOptions(), services=services, clients=clients)
        event_log = EventLog(config.logging.event_log, redact_content=config.logging.redact_content)
        self.state.routing_sink = SessionRoutingSink(event_log)
        self.state.filler_log = FillerLog(config.filler.log_path, event_log=event_log)
        self.agent_factory = _agent_factory(self.state)
        self._lock = asyncio.Lock()

    async def ensure_ready(self) -> bool:
        """Build speech services and LLM clients, then warm up ASR and TTS once (prototype lifespan)."""
        if self.state.ready:
            return True
        async with self._lock:
            if self.state.ready:
                return True
            config = self.config
            for warning in config.warnings:
                logger.warning(warning)
            if self.state.services is None:
                self.state.services = build_speech_services(config, stub=self.state.options.stub_speech)
            if self.state.clients is None and self.state.options.stub_agent != "scripted":
                self.state.clients = build_clients(config)
            if config.server.warmup and not self.state.options.stub_speech:
                logger.info("warming up ASR and TTS ...")
                try:
                    await self.state.services.warmup()
                except Exception as exc:  # noqa: BLE001 - reported to the client as server_busy; retried next session
                    self.state.startup_error = str(exc)
                    logger.error(f"speech warm-up failed: {exc}")
                    return False
            self.state.startup_error = ""
            self.state.ready = True
            logger.info(
                f"frontend/backend verdict agent ready (profile={self.profile}, filler={config.filler.mode}, "
                f"barge_in={config.barge_in.while_thinking}, tools={config.tools.source}, "
                f"frontend={config.agent.frontend.llm.model}, backend={config.agent.backend.llm.model})"
            )
            logger.info(f"ASR {config.asr.endpoint.describe()}")
            logger.info(f"TTS {config.tts.endpoint.describe()}")
            return True

    async def admit(self, websocket: Any) -> bool:
        """Apply the prototype's admission rule: not ready or at ``max_sessions`` -> ``server_busy`` + 1013."""
        ready = await self.ensure_ready()
        if not ready or self.state.sessions >= self.config.server.max_sessions:
            reason = "server is starting" if not ready else "server at capacity (server.max_sessions)"
            await websocket.send_text(json.dumps(ev.error(reason, code="server_busy")))
            await websocket.close(code=1013, reason=reason)
            return False
        self.state.sessions += 1
        self.state.total_sessions += 1
        return True

    def release(self) -> None:
        """One admitted session ended."""
        self.state.sessions = max(0, self.state.sessions - 1)


_RUNTIMES: dict[tuple[str, LLMEndpoints], VerdictRuntime] = {}


def runtime_for(body: Mapping[str, Any], environ: Mapping[str, str] | None = None) -> VerdictRuntime:
    """The shared runtime for the selected profile and the route's LLM endpoints."""
    name = profile_name(environ)
    endpoints = LLMEndpoints.from_body(body)
    key = (name, endpoints)
    runtime = _RUNTIMES.get(key)
    if runtime is None:
        config = apply_llm_endpoints(load_voice_config(profile_path(name)), endpoints)
        mismatch = keepalive_mismatch(config, environ)
        if mismatch:
            raise ProfileError(mismatch)
        runtime = VerdictRuntime(config, profile=name)
        _RUNTIMES[key] = runtime
        for warning in speech_selection_warnings(config, body):
            logger.warning(warning)
    return runtime


def speech_selection_warnings(config: VoiceConfig, body: Mapping[str, Any]) -> list[str]:
    """Differences between the route's ASR/TTS catalog selection and the prototype's resolved endpoints.

    The prototype loader resolves ASR and TTS itself (``asr``/``tts.catalog`` in the profile,
    with ``FBA_ASR_SERVER``/``FBA_TTS_SERVER`` overrides); the registry selection is shown to
    clients. A difference is logged so a deployment that changes one also changes the other.
    """
    warnings = []
    for kind, endpoint in (("asr", config.asr.endpoint), ("tts", config.tts.endpoint)):
        selected = str(body.get(f"{kind}_server") or "")
        if selected and selected != endpoint.server:
            warnings.append(
                f"{kind}: the Realtime route selected server {selected!r}, but the prototype speech loader uses "
                f"{endpoint.server!r} ({endpoint.source}); set FBA_{kind.upper()}_SERVER to change it"
            )
    return warnings
