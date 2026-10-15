# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Browser UI sessions (WebRTC or the UI's WebSocket) with the ``default`` profile.

Plan: ``misc/frontend-backend-verdict/ui-transport-plan.md``, sections 4.1, 4.4 and 4.5. A UI
session runs the same turn manager, turn bridge and output tagger as an OpenAI Realtime session
(:class:`UiSession` subclasses :class:`PipecatSession`), on the repository's UI transport
(``create_transport``) and RTVI instead of the Realtime wire protocol:

* the agent is configured when the session starts (no ``session.update``), with the profile's
  server-run tools and policy, and greets once the browser is ready;
* the UI's choices come in the request body with the keys every example reads (``llm_id``,
  ``asr_id``, ``tts_id``, ``tts_voice_id``, ``prompt_key``/``prompt_content``); speech and LLM ids
  are resolved by FBV's own catalog reader, so a selection always gives the same endpoint (no
  reachability probing);
* runtimes (LLM clients, speech warm-up) are cached per LLM and speech endpoints only, so their
  number is bounded by the catalogs; the edited prompt and the voice apply per session;
* all UI runtimes share one admission counter (the ``default`` profile's ``server.max_sessions``),
  separate from the Realtime runtimes', so browser sessions never take an evaluation slot.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from loguru import logger
from pipecat.processors.frameworks.rtvi.frames import RTVIServerMessageFrame
from pipecat.runner.types import RunnerArguments
from pipecat.workers.runner import WorkerRunner

from examples.frontend_backend_verdict.adapters.pipecat.pipeline import SpeechStack, real_speech
from examples.frontend_backend_verdict.adapters.pipecat.runtime import PipecatRuntime, SpeechFactory
from examples.frontend_backend_verdict.adapters.pipecat.session import PipecatSession
from examples.frontend_backend_verdict.adapters.pipecat.settings import load_pipecat_settings
from examples.frontend_backend_verdict.adapters.pipecat.ui_pipeline import UiSessionPipeline
from examples.frontend_backend_verdict.core.agent.factory import _agent_factory
from examples.frontend_backend_verdict.core.config.catalog import SpeechEndpoint, build_endpoint, load_catalog_entry
from examples.frontend_backend_verdict.core.config.profiles import (
    DEFAULT_PROFILE,
    EXAMPLE_DIR,
    SPEECH_PLATFORM_ENV,
    SPEECH_SERVER_ENV,
    LLMEndpoints,
    apply_llm_endpoints,
    apply_speech_platform,
    pipecat_settings_file,
    profile_path,
)
from examples.frontend_backend_verdict.core.config.voice import VoiceConfig, load_voice_config
from examples.frontend_backend_verdict.core.engine.filler import FillerTap
from examples.frontend_backend_verdict.core.errors import VoiceConfigError

#: The only public prompt key; the agent's other prompt keys stay hidden (registry agent_prompt_keys).
PUBLIC_PROMPT_KEY = "frontend"


# -- selections ---------------------------------------------------------------------------


def _local_section(env: Mapping[str, str]) -> str:
    """The services.local.yaml section of this deployment (single-GPU unless it says ``server``)."""
    return "server" if str(env.get(SPEECH_PLATFORM_ENV) or "").strip() == "server" else "singlegpu"


def _catalog_ref(entry_id: str, env: Mapping[str, str]) -> tuple[str, str, str] | None:
    """``(services, section, key)`` for a UI or Realtime service id, or ``None`` when it is not a catalog id.

    Accepted ids: ``cloud-nim:<key>``, ``self-hosted:<platform>:<key>``, ``self-hosted:<key>``
    (this deployment's local section) and a bare ``<key>`` (the same). ``custom-*`` ids and
    unknown sources are not catalog ids.
    """
    if not entry_id or entry_id.startswith("custom-"):
        return None
    if ":" not in entry_id:
        return "local", _local_section(env), entry_id
    source, key = entry_id.split(":", 1)
    if source == "cloud-nim" and key:
        return "cloud", "", key
    if source == "self-hosted" and key:
        if ":" in key:
            section, key = key.split(":", 1)
            return ("local", section, key) if section and key else None
        return "local", _local_section(env), key
    return None


def _catalog_entry(kind: str, entry_id: str, env: Mapping[str, str]) -> tuple[dict[str, Any], str] | None:
    ref = _catalog_ref(entry_id, env)
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
    """The ASR or TTS endpoint the UI selected (``<kind>_id``), or ``None`` for the deployment default.

    A catalog id is read from FBV's catalogs; a custom service (``custom-*``) uses the body's
    ``<kind>_server``/``<kind>_model``/``<kind>_function_id``. Either way the endpoint is built by
    FBV's ``build_endpoint`` (``FBA_ASR_SERVER``/``FBA_TTS_SERVER``, NVCF TLS and key checks).
    """
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
        found = _catalog_entry(kind, entry_id, env)
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


def llm_selection(body: Mapping[str, Any], env: Mapping[str, str]) -> LLMEndpoints:
    """The frontend LLM the UI selected (``llm_id``); the backend (thinker) LLM stays the profile's.

    A catalog id is read from FBV's catalogs (``model_id``, ``base_url``); a custom LLM uses the
    body's ``model_id``/``base_url``. The UI has no thinker control: the backend LLM is the
    profile's (core/config/agent.yaml), which matches the catalogs' ``thinker-llm`` default.
    """
    entry_id = str(body.get("llm_id") or "")
    if entry_id.startswith("custom-"):
        return LLMEndpoints(
            frontend_model=str(body.get("model_id") or ""), frontend_base_url=str(body.get("base_url") or "")
        )
    found = _catalog_entry("llm", entry_id, env) if entry_id else None
    if found is None:
        return LLMEndpoints()
    entry, _ = found
    return LLMEndpoints(
        frontend_model=str(entry.get("model_id") or ""), frontend_base_url=str(entry.get("base_url") or "")
    )


def _speech_key(endpoint: SpeechEndpoint) -> tuple[Any, ...]:
    """What a runtime binds of an endpoint (the voice is applied per session)."""
    return (
        endpoint.kind,
        endpoint.server,
        endpoint.model,
        endpoint.function_id,
        endpoint.language_code,
        endpoint.use_ssl,
    )


@dataclass(frozen=True, slots=True)
class UiSelection:
    """One UI session's choices, resolved."""

    llm: LLMEndpoints
    asr: SpeechEndpoint | None
    tts: SpeechEndpoint | None
    voice: str
    prompt: str

    @classmethod
    def from_body(cls, body: Mapping[str, Any], environ: Mapping[str, str] | None = None) -> UiSelection:
        """Resolve the request body the UI sends (``client/src/components/Header.tsx``)."""
        env = os.environ if environ is None else environ
        prompt = ""
        key = str(body.get("prompt_key") or "")
        content = body.get("prompt_content")
        if key in ("", PUBLIC_PROMPT_KEY) and isinstance(content, str) and content.strip():
            prompt = content
        return cls(
            llm=llm_selection(body, env),
            asr=speech_selection("asr", body, env),
            tts=speech_selection("tts", body, env),
            voice=str(body.get("tts_voice_id") or "").strip(),
            prompt=prompt,
        )


# -- runtimes and admission ------------------------------------------------------------


def ui_config(selection: UiSelection, environ: Mapping[str, str] | None = None) -> VoiceConfig:
    """The ``default`` profile with the deployment's speech platform and the selected endpoints."""
    config = apply_speech_platform(load_voice_config(profile_path(DEFAULT_PROFILE)), environ)
    if selection.asr is not None:
        config = replace(config, asr=replace(config.asr, endpoint=selection.asr))
    if selection.tts is not None:
        config = replace(config, tts=replace(config.tts, endpoint=selection.tts))
    return apply_llm_endpoints(config, selection.llm)


_UI_RUNTIMES: dict[tuple[Any, ...], PipecatRuntime] = {}


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


def ui_runtime_for(selection: UiSelection, environ: Mapping[str, str] | None = None) -> PipecatRuntime:
    """The cached runtime for a selection's LLM and speech endpoints (prompt and voice excluded)."""
    config = ui_config(selection, environ)
    key = (config.agent.frontend.llm.model, config.agent.frontend.llm.base_url)
    key += _speech_key(config.asr.endpoint) + _speech_key(config.tts.endpoint)
    runtime = _UI_RUNTIMES.get(key)
    if runtime is None:
        settings = load_pipecat_settings(pipecat_settings_file(DEFAULT_PROFILE))
        runtime = PipecatRuntime(config, profile=DEFAULT_PROFILE, settings=settings)
        _UI_RUNTIMES[key] = runtime
    return runtime


def session_config(runtime: PipecatRuntime, selection: UiSelection) -> VoiceConfig:
    """The runtime's config with this session's prompt and voice."""
    config = runtime.config
    voice = selection.voice or (selection.tts.voice_id if selection.tts is not None else "")
    if voice and voice != config.tts.endpoint.voice_id:
        config = replace(config, tts=replace(config.tts, endpoint=replace(config.tts.endpoint, voice_id=voice)))
    if selection.prompt:
        prompts = config.agent.prompts
        inline = {**prompts.inline, PUBLIC_PROMPT_KEY: selection.prompt}
        config = replace(config, agent=replace(config.agent, prompts=replace(prompts, inline=inline)))
    return config


@dataclass
class _SessionRuntime:
    """What :class:`PipecatSession` reads from a runtime, with this session's config and agent factory."""

    config: VoiceConfig
    settings: Any
    prompts: dict[str, str]
    state: Any
    agent_factory: Any


def _session_runtime(runtime: PipecatRuntime, config: VoiceConfig) -> _SessionRuntime:
    if config is runtime.config:
        factory = runtime.agent_factory
    else:
        factory = _agent_factory(replace(runtime.state, config=config))
    return _SessionRuntime(
        config=config, settings=runtime.settings, prompts=runtime.prompts, state=runtime.state, agent_factory=factory
    )


# -- the session --------------------------------------------------------------------------


class UiSession(PipecatSession):
    """A browser UI session: the Realtime session's turn handling on the UI transport, with RTVI."""

    def __init__(
        self,
        *,
        runtime: Any,
        transport: Any,
        speech: SpeechFactory | None = None,
        idle_timeout_secs: float | None = None,
        clock: Any = None,
        session_id: str | None = None,
    ) -> None:
        """Wire the session on ``transport`` (from ``create_transport``); nothing runs until :meth:`run`."""
        self._transport = transport
        self._idle_timeout_secs = idle_timeout_secs
        super().__init__(
            runtime=runtime,
            websocket=None,  # type: ignore[arg-type]
            model="",
            speech=speech,
            clock=clock,
            session_id=session_id,
        )

    def _build_pipeline(self, websocket: Any, speech: SpeechStack) -> UiSessionPipeline:  # noqa: ARG002
        return UiSessionPipeline(self, self._transport, speech, idle_timeout_secs=self._idle_timeout_secs)

    def configure_agent(self) -> None:
        """Give the agent the profile's tools and policy (what a Realtime client does with ``session.update``)."""
        self.agent.configure(tools=[], instructions="")
        self._configured = True
        self.log.write(
            "session_updated",
            input_format=None,
            output_format=None,
            turn_detection=self.view.settings.turn_detection.effective() | {"mode": "ui"},
            tools=[tool.name for tool in self.config.tools.config_tool_specs],
            instructions_chars=0,
        )

    def _register_rtvi_handlers(self) -> None:
        worker = self.pipeline.worker
        greeted = False

        @worker.rtvi.event_handler("on_client_ready")
        async def on_client_ready(_rtvi: Any) -> None:
            nonlocal greeted
            if greeted:
                return
            greeted = True
            if self.config.protocol.greeting_enabled:
                self.turns.speak_greeting()

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

    async def run(self) -> None:
        """Serve until the browser disconnects."""
        loop = asyncio.get_running_loop()
        self._routing_sink.register(self.session_id, FillerTap(loop, self.turns.on_filler, clock=self.clock))
        history = self.config.agent.backend.conversation_history
        self.log.write(
            "session_start",
            model="",
            backend_history=history.effective_include,
            backend_history_guidance=history.resolved_guidance_key,
            normalization=self.config.normalization.summary(),
            adapter="pipecat-ui",
            pipecat=self.settings.summary(),
            prompts=self._runtime_prompts,
        )
        logger.info(f"[{self.session_id}] browser UI session started (profile={DEFAULT_PROFILE}, adapter=pipecat)")
        self.configure_agent()
        self._register_rtvi_handlers()
        self._pump = loop.create_task(self._pump_events(), name=f"fbv-ui-events-{self.session_id}")
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
    runtime = ui_runtime_for(selection)
    if not await runtime.ensure_ready():
        logger.error(f"frontend-backend-verdict-agent: browser session refused: {runtime.state.startup_error}")
        return
    limit = runtime.config.server.max_sessions
    if not UI_ADMISSION.admit(limit):
        logger.warning(f"frontend-backend-verdict-agent: browser session refused: {limit} UI sessions are open")
        return
    try:
        transport = create_transport(runner_args)
        session = UiSession(
            runtime=_session_runtime(runtime, session_config(runtime, selection)),
            transport=transport,
            speech=speech or runtime.speech_factory or real_speech,
            idle_timeout_secs=getattr(runner_args, "pipeline_idle_timeout_secs", None),
        )
        await session.run()
    finally:
        UI_ADMISSION.release()
        logger.info("frontend/backend verdict browser session finished")
