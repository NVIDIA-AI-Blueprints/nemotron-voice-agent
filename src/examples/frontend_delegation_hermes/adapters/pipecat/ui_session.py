# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Browser UI sessions (WebRTC or the UI's WebSocket) with the ``default`` profile.

A UI session is today's demo session (``profiles/pipecat_demo.yaml``, ``adapters/pipecat.yaml``: demo
tools run in this process, the greeting, the demo backend delay, Pipecat's default turn settings) on
the repository's UI transport (``create_transport``) and RTVI instead of the Realtime wire protocol.
:class:`UiSession` subclasses :class:`PipecatSession`, so the turn manager, turn bridge, output
tagger, delegation and backend link are the same objects an OpenAI Realtime session uses:

* the session is configured when the browser is ready, with the ``session.update`` FDH's own demo
  page sent (:data:`UI_SESSION_UPDATE`), through the same Realtime protocol code; the backend is
  configured and the greeting spoken exactly as before;
* the UI's choices come in the request body with the keys every example reads (``llm_id``,
  ``asr_id``, ``tts_id``, ``tts_voice_id``): the frontend model, ASR and TTS endpoints are resolved
  from this example's own catalogs; the backend (Hermes) model stays the gateway's. The UI's prompt
  editor is not applied: FDH's frontend prompt is ``config/prompts/<profile>/frontend.yaml`` (``frontend_system``);
* runtimes (frontend model client, gateway check) are cached per frontend model and speech
  endpoints, so their number is bounded by the catalogs; the voice applies per session;
* all UI runtimes share one admission counter (the ``default`` profile's ``server.max_sessions``),
  separate from the Realtime runtime's, so browser sessions never take an evaluation slot.
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
from pipecat.frames.frames import TTSUpdateSettingsFrame
from pipecat.processors.frameworks.rtvi.frames import RTVIServerMessageFrame
from pipecat.runner.types import RunnerArguments
from pipecat.workers.runner import WorkerRunner

import examples_registry
from examples.frontend_delegation_hermes.adapters.pipecat.bot import EXAMPLE_KEY
from examples.frontend_delegation_hermes.adapters.pipecat.pipeline import SpeechStack
from examples.frontend_delegation_hermes.adapters.pipecat.profiles import DEFAULT_PROFILE, PROFILES
from examples.frontend_delegation_hermes.adapters.pipecat.runtime import FdhRuntime, SpeechFactory
from examples.frontend_delegation_hermes.adapters.pipecat.session import PipecatSession
from examples.frontend_delegation_hermes.adapters.pipecat.settings import load_pipecat_settings
from examples.frontend_delegation_hermes.adapters.pipecat.ui_pipeline import UiSessionPipeline
from examples.frontend_delegation_hermes.core.catalog import SpeechEndpoint, build_endpoint, load_catalog_entry
from examples.frontend_delegation_hermes.core.config import DelegationConfig, load_delegation_config
from examples.frontend_delegation_hermes.core.engine.filler import FillerTap
from examples.frontend_delegation_hermes.core.errors import VoiceConfigError
from examples.frontend_delegation_hermes.core.voice_config import VoiceConfig

#: The example's root: ``services.local.yaml`` and ``services.cloud.yaml``.
EXAMPLE_DIR = Path(__file__).resolve().parents[2]
#: The ``services.local.yaml`` section of the single-GPU recipe (the only FDH recipe).
LOCAL_SECTION = "singlegpu"
#: Replace the selected catalog server (the same variables as ``config/voice_agent.yaml``).
SPEECH_SERVER_ENV = {"asr": "FBA_ASR_SERVER", "tts": "FBA_TTS_SERVER"}

#: The ``session.update`` FDH's own demo page (``web/index.html``, removed) sent on ``session.created``.
UI_SESSION_UPDATE: dict[str, Any] = {
    "type": "session.update",
    "session": {
        "type": "realtime",
        "output_modalities": ["audio"],
        "audio": {
            "input": {
                "format": {"type": "audio/pcm", "rate": 24000},
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500,
                },
            },
            "output": {"format": {"type": "audio/pcm", "rate": 24000}, "voice": "alloy"},
        },
    },
}


# -- selections ---------------------------------------------------------------------------


def _catalog_ref(entry_id: str) -> tuple[str, str, str] | None:
    """``(services, section, key)`` for a UI service id, or ``None`` when it is not a catalog id.

    Accepted ids: ``cloud-nim:<key>``, ``self-hosted:<platform>:<key>``, ``self-hosted:<key>`` and a
    bare ``<key>`` (the single-GPU section). ``custom-*`` ids and unknown sources are not catalog ids.
    """
    if not entry_id or entry_id.startswith("custom-"):
        return None
    if ":" not in entry_id:
        return "local", LOCAL_SECTION, entry_id
    source, key = entry_id.split(":", 1)
    if source == "cloud-nim" and key:
        return "cloud", "", key
    if source == "self-hosted" and key:
        if ":" in key:
            section, key = key.split(":", 1)
            return ("local", section, key) if section and key else None
        return "local", LOCAL_SECTION, key
    return None


def _catalog_entry(kind: str, entry_id: str) -> tuple[dict[str, Any], str] | None:
    ref = _catalog_ref(entry_id)
    if ref is None:
        return None
    services, section, key = ref
    try:
        entry = load_catalog_entry(kind, example_dir=EXAMPLE_DIR, services=services, platform=section, key=key)
    except VoiceConfigError as exc:
        logger.warning(f"UI selection {kind}={entry_id!r} ignored: {exc}")
        return None
    return entry, f"catalog {EXAMPLE_DIR}/services.{services}.yaml [{section or '-'}] {key}"


def speech_selection(kind: str, body: Mapping[str, Any], env: Mapping[str, str]) -> SpeechEndpoint | None:
    """The ASR or TTS endpoint the UI selected (``<kind>_id``), or ``None`` for the profile's default."""
    entry_id = str(body.get(f"{kind}_id") or "")
    if not entry_id:
        return None
    if entry_id.startswith("custom-"):
        server = str(body.get(f"{kind}_server") or "").strip()
        if not server:
            return None
        entry: dict[str, Any] = {
            "server": server,
            "model": str(body.get(f"{kind}_model") or ""),
            "function_id": str(body.get(f"{kind}_function_id") or ""),
        }
        source = f"ui custom {entry_id}"
    else:
        found = _catalog_entry(kind, entry_id)
        if found is None:
            return None
        entry, source = found
    try:
        return build_endpoint(
            kind,
            entry,
            source=source,
            server_override=str(env.get(SPEECH_SERVER_ENV[kind]) or "").strip(),
            environ=dict(env),
        )
    except VoiceConfigError as exc:
        logger.warning(f"UI selection {kind}={entry_id!r} ignored: {exc}")
        return None


def llm_selection(body: Mapping[str, Any]) -> tuple[str, str]:
    """The frontend model and base URL the UI selected (``llm_id``); ``("", "")`` keeps the profile's."""
    entry_id = str(body.get("llm_id") or "")
    if entry_id.startswith("custom-"):
        return str(body.get("model_id") or ""), str(body.get("base_url") or "")
    found = _catalog_entry("llm", entry_id) if entry_id else None
    if found is None:
        return "", ""
    entry, _ = found
    return str(entry.get("model_id") or ""), str(entry.get("base_url") or "")


@dataclass(frozen=True, slots=True)
class UiSelection:
    """One UI session's choices, resolved."""

    model: str
    base_url: str
    asr: SpeechEndpoint | None
    tts: SpeechEndpoint | None
    voice: str
    prompt_edited: bool

    @classmethod
    def from_body(cls, body: Mapping[str, Any], environ: Mapping[str, str] | None = None) -> UiSelection:
        """Resolve the request body the UI sends (``client/src/components/Header.tsx``)."""
        env = os.environ if environ is None else environ
        model, base_url = llm_selection(body)
        content = body.get("prompt_content")
        return cls(
            model=model,
            base_url=base_url,
            asr=speech_selection("asr", body, env),
            tts=speech_selection("tts", body, env),
            voice=str(body.get("tts_voice_id") or "").strip(),
            prompt_edited=isinstance(content, str) and bool(content.strip()),
        )


# -- runtimes and admission ------------------------------------------------------------


def ui_config(selection: UiSelection) -> DelegationConfig:
    """The ``default`` profile with the selected frontend model and speech endpoints."""
    config = load_delegation_config(PROFILES[DEFAULT_PROFILE].delegation_path)
    voice = config.voice
    if selection.asr is not None:
        voice = replace(voice, asr=replace(voice.asr, endpoint=selection.asr))
    if selection.tts is not None:
        voice = replace(voice, tts=replace(voice.tts, endpoint=selection.tts))
    if not examples_registry.welcome_message_enabled(EXAMPLE_KEY):
        # ENABLE_WELCOME_MESSAGE=false turns the greeting off, as in the other examples.
        voice = replace(voice, protocol=replace(voice.protocol, greeting_enabled=False))
    llm = config.frontend.llm
    if selection.model:
        llm = replace(llm, model=selection.model, base_url=selection.base_url or llm.base_url)
    return replace(config, voice=voice, frontend=replace(config.frontend, llm=llm))


def _speech_key(endpoint: SpeechEndpoint) -> tuple[Any, ...]:
    """What a runtime binds of an endpoint (the voice is applied per session)."""
    return (endpoint.server, endpoint.model, endpoint.function_id, endpoint.language_code, endpoint.use_ssl)


_UI_RUNTIMES: dict[tuple[Any, ...], FdhRuntime] = {}


class UiAdmission:
    """One session counter for every UI runtime, capped by the ``default`` profile's ``max_sessions``."""

    def __init__(self) -> None:
        """Start with no sessions."""
        self.sessions = 0

    def admit(self, limit: int) -> bool:
        """Count one more session, unless ``limit`` sessions are already open."""
        if self.sessions >= limit:
            return False
        self.sessions += 1
        return True

    def release(self) -> None:
        """One admitted session ended."""
        self.sessions = max(0, self.sessions - 1)


UI_ADMISSION = UiAdmission()


def ui_runtime_for(selection: UiSelection, *, speech_factory: SpeechFactory | None = None) -> FdhRuntime:
    """The cached runtime for a selection's frontend model and speech endpoints (voice excluded)."""
    config = ui_config(selection)
    voice = config.voice
    key = (config.frontend.llm.model, config.frontend.llm.base_url, voice.protocol.greeting_enabled)
    key += _speech_key(voice.asr.endpoint) + _speech_key(voice.tts.endpoint)
    runtime = _UI_RUNTIMES.get(key)
    if runtime is None:
        settings = load_pipecat_settings(PROFILES[DEFAULT_PROFILE].pipecat_settings)
        runtime = FdhRuntime(config, settings, profile=DEFAULT_PROFILE, speech_factory=speech_factory)
        _UI_RUNTIMES[key] = runtime
    return runtime


def session_voice_config(runtime: FdhRuntime, selection: UiSelection) -> VoiceConfig:
    """The runtime's voice config with this session's TTS voice."""
    voice_config = runtime.config.voice
    voice = selection.voice or (selection.tts.voice_id if selection.tts is not None else "")
    endpoint = voice_config.tts.endpoint
    if voice and voice != endpoint.voice_id:
        tts = replace(voice_config.tts, endpoint=replace(endpoint, voice_id=voice))
        voice_config = replace(voice_config, tts=tts)
    return voice_config


# -- the session --------------------------------------------------------------------------


class UiSession(PipecatSession):
    """A browser UI session: the Realtime session's turn handling on the UI transport, with RTVI."""

    def __init__(
        self,
        *,
        runtime: FdhRuntime,
        config: VoiceConfig,
        transport: Any,
        speech: SpeechStack | None = None,
        idle_timeout_secs: float | None = None,
        clock: Any = None,
        session_id: str | None = None,
    ) -> None:
        """Wire the session on ``transport`` (from ``create_transport``); nothing runs until :meth:`run`."""
        self._transport = transport
        self._idle_timeout_secs = idle_timeout_secs
        super().__init__(
            config=config,
            settings=runtime.settings,
            websocket=None,  # type: ignore[arg-type]
            routing_sink=runtime.routing_sink,
            filler_log=runtime.filler_log,
            turn_manager_factory=runtime.delegation.turn_manager,
            clock=clock,
            model="",
            session_id=session_id,
            speech=speech,
        )
        self.pipeline.gate.bind(self.playout)

    def _build_pipeline(self, websocket: Any, speech: SpeechStack) -> UiSessionPipeline:  # noqa: ARG002
        return UiSessionPipeline(self, self._transport, speech, idle_timeout_secs=self._idle_timeout_secs)

    async def configure_session(self) -> bool:
        """Configure the session as FDH's demo page did (backend tools, greeting); whether it was accepted."""
        await self.protocol.handle_message(json.dumps(UI_SESSION_UPDATE))
        if not self.protocol.configured:
            logger.error(f"[{self.session_id}] the browser session could not be configured (see the error above)")
        return self.protocol.configured

    def _register_rtvi_handlers(self) -> None:
        worker = self.pipeline.worker
        configured = False

        @worker.rtvi.event_handler("on_client_ready")
        async def on_client_ready(_rtvi: Any) -> None:
            nonlocal configured
            if configured:
                return
            configured = True
            await self.configure_session()

        @worker.rtvi.event_handler("on_client_message")
        async def on_client_message(_rtvi: Any, message: Any) -> None:
            if message.type == "set-voice":
                await self.apply_set_voice(message.data if isinstance(message.data, dict) else {})

        latency = self.pipeline.latency_observer

        @latency.event_handler("on_first_bot_speech_latency")
        async def on_first(_observer: Any, value: float) -> None:
            await worker.queue_frame(
                RTVIServerMessageFrame(data={"type": "user-bot-latency", "latency": round(value, 3), "first": True})
            )

        @latency.event_handler("on_latency_measured")
        async def on_latency(_observer: Any, value: float) -> None:
            await worker.queue_frame(
                RTVIServerMessageFrame(data={"type": "user-bot-latency", "latency": round(value, 3), "first": False})
            )

    async def apply_set_voice(self, payload: Mapping[str, Any]) -> None:
        """The UI's ``set-voice`` message: the TTS uses ``voice_id`` (and ``language``) from the next sentence on."""
        from pipecat.services.nvidia.tts import NvidiaTTSSettings  # noqa: PLC0415

        from utils import normalize_lang_code  # noqa: PLC0415

        voice_id = str(payload.get("voice_id") or "")
        language = str(payload.get("language") or "")
        if not voice_id:
            return
        settings: dict[str, Any] = {"voice": voice_id}
        if language:
            settings["language"] = normalize_lang_code(language)
        await self.pipeline.worker.queue_frame(
            TTSUpdateSettingsFrame(delta=NvidiaTTSSettings(**settings), service=self.pipeline.tts)
        )
        logger.info(
            f"[{self.session_id}] voice switched to {voice_id}, language={settings.get('language', '(unchanged)')}"
        )

    async def run(self) -> None:
        """Serve until the browser disconnects."""
        loop = asyncio.get_running_loop()
        self._routing_sink.register(self.session_id, FillerTap(loop, self.turns.on_filler, clock=self.clock))
        self.log.session_start(model="", config=self.config, adapter="pipecat-ui", pipecat=self.settings.summary())
        logger.info(f"[{self.session_id}] browser UI session started (profile={DEFAULT_PROFILE}, adapter=pipecat)")
        await self.turns.start()
        self._register_rtvi_handlers()
        self._pump = loop.create_task(self._pump_events(), name=f"fdh-ui-events-{self.session_id}")
        runner = WorkerRunner(handle_sigint=False, handle_sigterm=False)
        try:
            await runner.add_workers(self.pipeline.worker)
            await runner.run()
        finally:
            await self.close()


# -- entry point ------------------------------------------------------------------------


async def serve_ui(runner_args: RunnerArguments, *, speech: SpeechFactory | None = None) -> None:
    """Serve one browser UI session (WebRTC or the UI WebSocket) with the ``default`` profile."""
    from examples.shared.pipeline_utils import create_transport  # noqa: PLC0415

    body = getattr(runner_args, "body", None) or {}
    selection = UiSelection.from_body(body)
    if selection.prompt_edited:
        logger.info(
            f"{EXAMPLE_KEY}: the UI's prompt is not applied; "
            "the frontend prompt is config/prompts/<profile>/frontend.yaml (frontend_system)"
        )
    runtime = ui_runtime_for(selection, speech_factory=speech)
    if not await runtime.ensure_ready():
        logger.error(f"{EXAMPLE_KEY}: browser session refused: {runtime.startup_error}")
        return
    limit = runtime.config.voice.server.max_sessions
    if not UI_ADMISSION.admit(limit):
        logger.warning(f"{EXAMPLE_KEY}: browser session refused: {limit} UI sessions are open")
        return
    try:
        transport = create_transport(runner_args)
        factory = speech or runtime.speech_factory
        session = UiSession(
            runtime=runtime,
            config=session_voice_config(runtime, selection),
            transport=transport,
            speech=factory(None) if factory is not None else None,
            idle_timeout_secs=getattr(runner_args, "pipeline_idle_timeout_secs", None),
        )
        await session.run()
    finally:
        UI_ADMISSION.release()
        logger.info(f"{EXAMPLE_KEY}: browser session finished")
