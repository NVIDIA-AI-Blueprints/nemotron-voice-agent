# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Process-wide state for the Pipecat adapter's sessions (S12).

The same rules as v1's ``VerdictRuntime`` (``adapters/prototype/runtime.py`` at tag
``fbv-v1-baseline``), with the shared Pipecat services in place of v1's own speech clients:
one runtime per (profile, LLM endpoints) configuration, with one event log, routing sink and
filler log, one LLM client per role (``core/agent/factory.py``), the speech warm-up
(``warmup.py``) and the ``server.max_sessions`` cap (``server_busy`` and close code 1013).
Profiles, LLM endpoints, the speech platform remap and the keepalive check come from
``core/config/profiles.py``.

OpenAI Realtime sessions always get the ``eval`` profile's runtime (:func:`runtime_for`), whichever
profile ``FBV_PROFILE`` selects; browser UI sessions have their own runtimes and admission
(``ui_session.py``).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from examples.frontend_backend_verdict.adapters.pipecat.settings import PipecatSettings, load_pipecat_settings
from examples.frontend_backend_verdict.adapters.pipecat.warmup import warm_up
from examples.frontend_backend_verdict.core.agent.factory import ServerOptions, _agent_factory, build_clients
from examples.frontend_backend_verdict.core.config.profiles import (
    DEFAULT_PROFILE,
    PROFILE_ENV,
    REALTIME_PROFILE,
    LLMEndpoints,
    ProfileError,
    apply_llm_endpoints,
    apply_speech_platform,
    keepalive_mismatch,
    pipecat_settings_file,
    profile_name,
    profile_path,
    speech_selection_warnings,
)
from examples.frontend_backend_verdict.core.config.voice import VoiceConfig, load_voice_config
from examples.frontend_backend_verdict.core.engine import server_events as ev
from examples.frontend_backend_verdict.core.engine.event_log import EventLog, SessionRoutingSink
from examples.frontend_backend_verdict.core.engine.filler import FillerLog
from examples.frontend_backend_verdict.core.engine.runner import AgentClients

if TYPE_CHECKING:
    from examples.frontend_backend_verdict.adapters.pipecat.pipeline import SpeechStack
    from examples.frontend_backend_verdict.adapters.pipecat.session import PipecatSession

#: Builds one session's speech stack (``None``: the shared Pipecat services).
SpeechFactory = Callable[["PipecatSession"], "SpeechStack"]


def prompt_catalog(config: VoiceConfig) -> dict[str, str]:
    """The prompt catalog a profile resolved to (file name and SHA-256), for the logs (plan §7.3)."""
    path = Path(config.agent.prompts_path)
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        digest = ""
    return {"catalog": path.name, "sha256": digest}


@dataclass(slots=True)
class PipecatState:
    """What the agent factory and the sessions share (satisfies ``AgentFactoryState``)."""

    config: VoiceConfig
    options: ServerOptions
    clients: AgentClients | None = None
    routing_sink: SessionRoutingSink = field(default_factory=SessionRoutingSink)
    filler_log: FillerLog = field(default_factory=FillerLog)
    ready: bool = False
    sessions: int = 0
    total_sessions: int = 0
    startup_error: str = ""


class PipecatRuntime:
    """The adapter's state for one (profile, LLM endpoints) configuration."""

    def __init__(
        self,
        config: VoiceConfig,
        *,
        profile: str,
        settings: PipecatSettings | None = None,
        options: ServerOptions | None = None,
        clients: AgentClients | None = None,
        speech_factory: SpeechFactory | None = None,
    ) -> None:
        """Build the shared objects; ``clients`` and ``speech_factory`` may be injected (tests)."""
        self.profile = profile
        self.config = config
        self.settings = settings or load_pipecat_settings()
        self.speech_factory = speech_factory
        self.state = PipecatState(config=config, options=options or ServerOptions(), clients=clients)
        event_log = EventLog(config.logging.event_log, redact_content=config.logging.redact_content)
        self.state.routing_sink = SessionRoutingSink(event_log)
        self.state.filler_log = FillerLog(config.filler.log_path, event_log=event_log)
        self.agent_factory = _agent_factory(self.state)
        self.prompts = prompt_catalog(config)
        self._lock = asyncio.Lock()
        if self.settings.audio.in_sample_rate != config.audio.engine_rate:
            logger.warning(
                f"Pipecat input rate {self.settings.audio.in_sample_rate} differs from the profile's "
                f"audio.engine_rate {config.audio.engine_rate}"
            )

    async def ensure_ready(self) -> bool:
        """Build the LLM clients, then warm up ASR and TTS once."""
        if self.state.ready:
            return True
        async with self._lock:
            if self.state.ready:
                return True
            config = self.config
            for warning in config.warnings:
                logger.warning(warning)
            if self.state.clients is None and self.state.options.stub_agent != "scripted":
                self.state.clients = build_clients(config)
            if config.server.warmup and not self.state.options.stub_speech and self.speech_factory is None:
                logger.info("warming up ASR and TTS ...")
                try:
                    await warm_up(config)
                except Exception as exc:  # noqa: BLE001 - reported to the client as server_busy; retried next session
                    self.state.startup_error = str(exc)
                    logger.error(f"speech warm-up failed: {exc}")
                    return False
            self.state.startup_error = ""
            self.state.ready = True
            logger.info(
                f"frontend/backend verdict agent ready (adapter=pipecat, profile={self.profile}, "
                f"filler={config.filler.mode}, barge_in={config.barge_in.while_thinking}, "
                f"tools={config.tools.source}, frontend={config.agent.frontend.llm.model}, "
                f"backend={config.agent.backend.llm.model}, pipecat={self.settings.summary()})"
            )
            logger.info(f"prompts {config.agent.prompts_path} (sha256 {self.prompts['sha256'] or 'unreadable'})")
            logger.info(f"ASR {config.asr.endpoint.describe()}")
            logger.info(f"TTS {config.tts.endpoint.describe()}")
            return True

    async def admit(self, websocket: Any) -> bool:
        """v1's admission rule: not ready or at ``max_sessions`` -> ``server_busy`` + 1013."""
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


_RUNTIMES: dict[tuple[str, LLMEndpoints], PipecatRuntime] = {}
_NOTED_DEFAULT_CHANGE = False


def _note_default_change(environ: Mapping[str, str] | None) -> None:
    """Say once that an explicit ``FBV_PROFILE=default`` no longer changes Realtime sessions."""
    global _NOTED_DEFAULT_CHANGE  # noqa: PLW0603 - one log line per process
    env = os.environ if environ is None else environ
    if _NOTED_DEFAULT_CHANGE or (env.get(PROFILE_ENV) or "").strip() != DEFAULT_PROFILE:
        return
    _NOTED_DEFAULT_CHANGE = True
    logger.info(
        f"{PROFILE_ENV}={DEFAULT_PROFILE} selects the browser UI profile; OpenAI Realtime sessions use the "
        f"{REALTIME_PROFILE!r} profile (the Tau prompt catalog)"
    )


def runtime_for(body: Mapping[str, Any], environ: Mapping[str, str] | None = None) -> PipecatRuntime:
    """The shared runtime for OpenAI Realtime sessions and the route's LLM endpoints.

    Realtime sessions always run with the ``eval`` profile and its Pipecat settings
    (``settings/eval.yaml``). ``FBV_PROFILE`` is still checked, so an unknown value is refused.
    """
    profile_name(environ)
    _note_default_change(environ)
    name = REALTIME_PROFILE
    endpoints = LLMEndpoints.from_body(body)
    key = (name, endpoints)
    runtime = _RUNTIMES.get(key)
    if runtime is None:
        config = apply_llm_endpoints(load_voice_config(profile_path(name)), endpoints)
        config = apply_speech_platform(config, environ)
        mismatch = keepalive_mismatch(config, environ)
        if mismatch:
            raise ProfileError(mismatch)
        runtime = PipecatRuntime(config, profile=name, settings=load_pipecat_settings(pipecat_settings_file(name)))
        _RUNTIMES[key] = runtime
        for warning in speech_selection_warnings(config, body):
            logger.warning(warning)
    return runtime
