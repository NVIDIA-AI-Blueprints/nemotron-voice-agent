# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The cascade frontend adapter: serves a live session with speech in, the talker, and speech out."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, ClassVar, Self

from loguru import logger
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.utils.context.llm_context_summarization import LLMContextSummarizationUtil

from examples.frontend_backend_live.adapter.pipeline_host import PipelineHost
from examples.frontend_backend_live.adapter.readiness import EndpointReadiness
from examples.frontend_backend_live.config_manager.schema import LiveConfig, ModelEndpoint
from examples.frontend_backend_live.frontend import FrontendSpec, build_frontend
from examples.frontend_backend_live.media.services import build_stt, build_tts, named_mode, speech_server
from examples.frontend_backend_live.media.turns import SmartTurn
from examples.frontend_backend_live.prompts import Prompts, load_prompts
from examples.frontend_backend_live.talker import (
    LLM_SERVICES,
    RESPONSE_FORMATS,
    JsonSchema,
    PromptPart,
    ResponseFormat,
    TalkerChat,
)
from examples.frontend_backend_live.voice import Voice
from examples.shared.pipeline_utils import PIPELINE_AUDIO_IN_SAMPLE_RATE
from examples.shared.prewarm import get_tts_config
from gateways.live.protocol import SessionConfig
from gateways.realtime.voice import resolve_realtime_tts_voice

if TYPE_CHECKING:
    from examples.frontend_backend_live.adapter.contract import ServiceCatalog, ServiceReadiness
    from examples.frontend_backend_live.frontend import Frontend
    from gateways.live.ports import DelegationPort, MediaConnection, SessionPort

LIVE_AUDIO_IN_RATE = PIPELINE_AUDIO_IN_SAMPLE_RATE


def own_instructions(config: SessionConfig) -> str | None:
    """Return the session's instructions, or ``None`` when they are unset or blank."""
    return config.instructions if (config.instructions or "").strip() else None


def seed_messages(config: SessionConfig, instructions: str) -> list[dict]:
    """Return the talker's starting context: its instructions, then the client's input history."""
    messages = [{"role": "system", "content": instructions}]
    for item in config.input:
        part = item["content"][0]
        messages.append({"role": item["role"], "content": part["text"]})
    return messages


def listed_voice(voice_id: str, tts: Mapping[str, Any]) -> str:
    """Return the voice to speak for the custom ``voice_id``, checked with ``resolve_realtime_tts_voice``.

    ``tts`` is the text-to-speech's catalog entry; when its voice list is empty, the entry's own voice, if any, wins.
    """
    server, function_id, model = speech_server(tts), str(tts.get("function_id") or ""), str(tts.get("model") or "")
    own = str(tts.get("voice_id") or "")
    # Listed with its own voice first, so the cached names are this model's even when the catalog's first TTS is not.
    if not get_tts_config(server, own, function_id, model).get("voices"):
        spoken = own or voice_id
        if voice_id != own:
            logger.warning(f"TTS voice catalog empty at {server}; cannot verify {voice_id!r}; using {spoken!r}")
        return spoken
    selection = {"tts_voice_id": voice_id, "tts_server": server, "tts_function_id": function_id, "tts_model": model}
    return resolve_realtime_tts_voice(selection, voice_was_set=True) or voice_id


@dataclass(frozen=True)
class CascadeAdapterSettings:
    """The validated ``frontend.cascade`` block and what its names pick: ``CascadeAdapter.Settings``."""

    live: LiveConfig
    talkers: Callable[[str], type[TalkerChat]]
    turn_end: SmartTurn
    response_formats: Callable[[str], ResponseFormat]
    message_text: Callable[[dict], str]

    @classmethod
    def parse(cls, block: Mapping[str, Any], *, pick: Callable[..., Any], message_text: Callable[[dict], str]) -> Self:
        """Validate the ``frontend.cascade`` block and pick each name it gives, so a bad value fails the root's read."""
        live = LiveConfig.model_validate(block)
        talkers = partial(pick, LLM_SERVICES, kind="talker_style")
        for style in live.talker_style.values():
            talkers(style)
        load_prompts().text("Talker", live.prompt)
        return cls(
            live=live,
            talkers=talkers,
            turn_end=SmartTurn(live.speech),
            response_formats=partial(pick, RESPONSE_FORMATS, kind="response_format"),
            message_text=message_text,
        )


class CascadeAdapter(PipelineHost):
    """Serves one live session with the cascade."""

    Settings: ClassVar[type[CascadeAdapterSettings]] = CascadeAdapterSettings

    def __init__(
        self,
        config: SessionConfig,
        transport: str,
        connection,
        *,
        live_config: LiveConfig,
        frontend_endpoint: ModelEndpoint,
        instructions: str,
        prompts: Prompts,
        stt,
        tts,
        item_text: Callable[[dict], str],
        user_params,
        talker: type[TalkerChat] = TalkerChat,
        response_format: ResponseFormat = RESPONSE_FORMATS[JsonSchema.name],
    ):
        """Hold everything the session needs; nothing runs until :meth:`start`."""
        super().__init__()
        self.config, self.live_config = config, live_config
        self.delegation: DelegationPort | None = None
        self.transport, self.connection, self.frontend_endpoint = transport, connection, frontend_endpoint
        self.instructions, self.stt, self.tts, self.user_params = instructions, stt, tts, user_params
        self.prompts, self.talker = prompts, talker
        self.item_text, self.response_format = item_text, response_format
        self.frontend: Frontend | None = None
        self.voice: Voice | None = None
        self._closed = False

    @classmethod
    def from_config(
        cls,
        session: SessionConfig,
        transport: str,
        connection: MediaConnection | None,
        settings: CascadeAdapterSettings,
        catalog: ServiceCatalog,
    ) -> CascadeAdapter:
        """Build one session's adapter: its talker, instructions and speech services, with a custom voice checked."""
        live = settings.live
        prompts = load_prompts()
        voice, tts_entry = session.audio.output.voice, catalog.service("tts")
        overrides = {"tts_voice_id": listed_voice(voice["id"], tts_entry)} if isinstance(voice, dict) else {}
        endpoint = catalog.model(live.slot, overrides)
        return cls(
            session,
            transport,
            connection,
            live_config=live,
            frontend_endpoint=endpoint,
            instructions=own_instructions(session) or prompts.text("Talker", live.prompt),
            prompts=prompts,
            stt=build_stt(overrides, catalog.service("asr")),
            tts=build_tts(overrides, tts_entry),
            item_text=settings.message_text,
            user_params=settings.turn_end.user_params(welcome_enabled=False),
            talker=settings.talkers(live.style_of(endpoint.service_key)),
            response_format=settings.response_formats(endpoint.response_format),
        )

    @classmethod
    def describe(cls, settings: CascadeAdapterSettings, catalog: ServiceCatalog) -> dict:
        """Return the adapter's half of ``GET /v1/live/info``: its talker, and its speech settings and TTS mode."""
        live = settings.live
        endpoint = catalog.model(live.slot)
        tts_mode = named_mode(catalog.service("tts").get("synthesis_mode")).synthesis_mode
        return {
            "frontend": {
                "talker_style": live.style_of(endpoint.service_key),
                "slot": live.slot,
                "model": endpoint.model,
                "base_url": endpoint.base_url,
                "prompt": live.prompt,
            },
            "speech": {"tts_mode": tts_mode, **settings.turn_end.describe()},
        }

    @classmethod
    def readiness(cls, settings: CascadeAdapterSettings, catalog: ServiceCatalog) -> ServiceReadiness:
        """Return the readiness check of the speech-to-text, the text-to-speech and the talker's model."""
        slot = settings.live.slot
        return EndpointReadiness(
            {
                "asr": speech_server(catalog.service("asr")),
                "tts": speech_server(catalog.service("tts")),
                slot: catalog.model(slot).base_url,
            }
        )

    async def start(self, session: SessionPort, delegation: DelegationPort) -> None:
        """Build the frontend on the session's edges, start the voice, and return once the pipeline runs."""
        self.session, self.delegation = session, delegation
        live = self.live_config
        talker = self.talker.from_config(self.frontend_endpoint, live.reliability, self.response_format)
        spec = FrontendSpec(
            config=live,
            talker=talker,
            prompts=self.prompts.cascade,
            lines=self.prompts.lines,
            seed=seed_messages(self.config, self.instructions),
            stt=self.stt,
            tts=self.tts,
            user_params=self.user_params,
            delegation=delegation,
            item_text=self.item_text,
        )
        edges = self.edges(
            transport=self.transport,
            connection=self.connection,
            audio_in_rate=LIVE_AUDIO_IN_RATE,
            stt=self.stt,
        )
        self.frontend = build_frontend(spec, edges)
        self.voice = self.frontend.voice
        self.voice.start()
        await self.run_pipeline(self.frontend.worker, self.frontend.processors)

    def context_tokens(self) -> int | None:
        """Return Pipecat's token estimate of the talker's next routing request, for the session's context meter."""
        if self.voice is None:
            return None
        voice = self.voice
        routing = voice.layout.routing("\n\n".join(voice.instructions), voice.worknote.line(), "")
        messages = routing.messages(voice.conversation(labeled=True))
        return LLMContextSummarizationUtil.estimate_context_tokens(LLMContext(messages))

    def context_window_tokens(self) -> int | None:
        """Return the talker's memory budget (``memory.context_tokens``), for the session's context meter."""
        if self.voice is None:
            return None
        return self.live_config.memory.context_tokens

    async def history(self) -> dict | None:
        """Return the adapter's half of ``GET /v1/live/sessions/{id}/history``; ``None`` before the start."""
        frontend = self.frontend_history()
        if frontend is None:
            return None
        backend = None
        if self.delegation.reports_events:
            backend = {
                "busy": bool(self.voice.worknote.open()),
                "pending": [{"kind": kind, "text": text} for kind, text in self.voice.heard_log.lines()],
            }
        return json.loads(json.dumps({"frontend": frontend, "backend": backend}, default=str))

    def frontend_history(self) -> dict | None:
        """Return the talker's model, routing instructions and labeled history; ``None`` before the start."""
        if self.voice is None:
            return None
        return {
            "model": self.frontend_endpoint.model,
            "prompt": self._talker_prompt(),
            "messages": self.voice.conversation(labeled=True),
        }

    def _talker_prompt(self) -> list[dict]:
        """Return the routing call's instructions part by part, each with its source, in the order they are sent."""
        own = own_instructions(self.config)
        base = "session.instructions" if own else f"agent_prompts.Talker.{self.live_config.prompt}"
        first, *appended = self.voice.instructions
        instructions = [PromptPart(base, first)]
        instructions += [PromptPart("session.instructions.append", text, "\n\n") for text in appended]
        layout = self.voice.layout
        parts = [{"source": part.source, "text": part.text} for part in layout.routing_parts(instructions)]
        rules = layout.routing_texts.rules
        parts.append({"source": rules.source, "text": rules.text})
        parts.append({"source": "runtime data, after the conversation", "text": self.voice.runtime_data()})
        return parts

    async def close(self) -> None:
        """Stop the pipeline and close the voice; a repeat call stops only a pipeline that started after the first."""
        if self._closed:
            if self.runner_task is not None and not self.runner_task.done():
                await self.stop_pipeline()
            return
        self._closed = True
        await self.stop_pipeline()
        if self.voice is not None:
            await self.voice.close()

    async def append_instructions(
        self, text: str, client_event_id: str | None = None, delegation_id: str | None = None
    ) -> None:
        """Interrupt current speech and decide on the new direction for the call; ``delegation_id`` is unused."""
        self._spawn(self.voice.instruction(text))

    async def append_thinking(
        self, text: str, client_event_id: str | None = None, delegation_id: str | None = None
    ) -> None:
        """Keep silent context for the call and its next delegation; ``delegation_id`` is unused."""
        self.voice.thinking(text)

    async def append_commentary(
        self, text: str, client_event_id: str | None = None, delegation_id: str | None = None
    ) -> None:
        """Keep the facts and word them for the caller; ``delegation_id`` is the client delegation they answer."""
        self.voice.commentary(text, delegation_id)

    def typed(self, text: str) -> None:
        """Put the application's typed question into the talker's context, never as a caller turn."""
        if self.voice is not None:
            self.voice.typed(text)
