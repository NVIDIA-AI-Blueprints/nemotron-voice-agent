# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The cascade adapter: serves a live session with speech in, the talker, commentary, and speech out.

It builds the Pipecat pipeline (speech-to-text, user aggregator, :class:`LiveTalkerProcessor`, text-to-speech) around a
:class:`CascadeSession`. The client supplies the prompts and tools; the server supplies the models and the speech
services from the registry. The session hands the adapter its delegation part (``live.ports.DelegationPort``): the
server delegation in ``responses`` mode, whose official events are the cascade's only view of its delegations, or the
client delegation in client mode, which announces each delegation to the client app. The session opens and closes the
delegation part; the adapter owns its pipeline, its presenter's task and its silence timer.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Self

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair

from examples.frontend_backend_live.adapter.pipeline_host import PipelineHost
from examples.frontend_backend_live.adapter.preroll import SpeechPreroll
from examples.frontend_backend_live.adapter.readiness import EndpointReadiness
from examples.frontend_backend_live.adapter.services import (
    SPEECH_MODES,
    SpeechMode,
    build_stt,
    build_tts,
    speech_mode,
    speech_server,
)
from examples.frontend_backend_live.cascade.agent import LiveAgent, build_agent
from examples.frontend_backend_live.cascade.memory import ContextConversation, message_text, split_messages
from examples.frontend_backend_live.cascade.processor import LiveTalkerProcessor
from examples.frontend_backend_live.cascade.session import CascadeSession
from examples.frontend_backend_live.cascade.settings import CascadeSettings
from examples.frontend_backend_live.cascade.talker.talker import PromptPart
from examples.frontend_backend_live.config_manager.schema import LiveConfig, ModelEndpoint
from examples.frontend_backend_live.tables import pick
from examples.shared.pipeline_utils import PIPELINE_AUDIO_IN_SAMPLE_RATE, build_user_aggregator_params
from live.protocol import SessionConfig, approx_tokens

if TYPE_CHECKING:
    from examples.frontend_backend_live.adapter.contract import ServiceCatalog, ServiceReadiness
    from live.ports import DelegationPort, MediaConnection, SessionPort

LIVE_AUDIO_IN_RATE = PIPELINE_AUDIO_IN_SAMPLE_RATE  # the rate the cascade hears the caller at: NVA's shared input rate
SEED_ROLES = {"developer": "developer", "user": "user", "assistant": "assistant"}


def seed_messages(config: SessionConfig, instructions: str) -> list[dict]:
    """The talker's starting context: its instructions, then the client's input history."""
    messages = [{"role": "system", "content": instructions}]
    for item in config.input:
        part = item["content"][0]
        messages.append({"role": SEED_ROLES[item["role"]], "content": part["text"]})
    return messages


# ------------------------------------------------------------------------------------------- the settings
@dataclass(frozen=True)
class CascadeAdapterSettings:
    """``CascadeAdapter.Settings``: the validated ``frontend.cascade`` block and the text-to-speech mode.

    The cascade's own share of the block is checked here too, so a bad value fails the root's read.
    """

    live: LiveConfig
    tts_mode: str = "per_sentence"

    @classmethod
    def parse(cls, block: Mapping[str, Any]) -> Self:
        """Validate the class's settings: the ``frontend.cascade`` block, read once (``LiveConfig.parse``).

        Raises:
            ConfigError: The block, or a part of it, is malformed, or a key belongs elsewhere (the message says where).
            ValueError: ``speech.tts_mode`` names no TTS mode (the message lists them), or the cascade refuses its share
                of the block (an unknown ``result_wording``, a ``compact_at`` above 1).
        """
        live = LiveConfig.parse(block)
        pick(SPEECH_MODES, live.speech.tts_mode, "speech.tts_mode")
        CascadeSettings.from_live_config(live)  # the cascade's own rules, checked now rather than at a session's start
        return cls(live=live, tts_mode=live.speech.tts_mode)


# ------------------------------------------------------------------------------------------- the adapter
class CascadeAdapter(PipelineHost):
    """Serves one live session with the cascade."""

    Settings: ClassVar[type[CascadeAdapterSettings]] = CascadeAdapterSettings
    hosts_rtvi: ClassVar[bool] = True  # the repository UI bot hosts the cascade in its own RTVI pipeline

    def __init__(
        self,
        config: SessionConfig,
        transport: str,
        connection,
        *,
        live_config: LiveConfig,
        frontend_endpoint: ModelEndpoint,
        instructions: str,
        stt,
        tts,
        user_params=None,
        frontend_client=None,
        speech: SpeechMode | None = None,
    ):
        """Hold everything the session needs; nothing runs until :meth:`start`.

        Args:
            config: The session's startup configuration.
            transport: ``webrtc`` or ``websocket``.
            connection: The WebRTC connection (``webrtc`` only).
            live_config: The validated example configuration.
            frontend_endpoint: The talker's model endpoint.
            instructions: The talker's instructions.
            stt: The speech-to-text service.
            tts: The text-to-speech service.
            user_params: Optional user-aggregator parameters (tests supply a transcript-driven turn strategy).
            frontend_client: Optional injected chat client for the frontend.
            speech: How ``tts`` takes a line, which must match its synthesis mode. By default it is the mode ``tts``
                already runs in (``services.speech_mode``), for a host that built the text-to-speech itself.
        """
        super().__init__()
        self.config, self.live_config = config, live_config
        self.delegation: DelegationPort | None = None  # the session's delegation part, from start
        self.transport, self.connection, self.frontend_endpoint = transport, connection, frontend_endpoint
        self.instructions, self.stt, self.tts, self.user_params = instructions, stt, tts, user_params
        self.frontend_client = frontend_client
        self.speech = speech or speech_mode(tts)
        self.agent: LiveAgent | None = None
        self.cascade: CascadeSession | None = None
        self.live: LiveTalkerProcessor | None = None
        self._closed = False

    # ------------------------------------------------------------------------------ the class side
    @classmethod
    def from_config(
        cls,
        session: SessionConfig,
        transport: str,
        connection: MediaConnection | None,
        settings: CascadeAdapterSettings,
        catalog: ServiceCatalog,
    ) -> CascadeAdapter:
        """Build one session's adapter: the talker's endpoint and instructions, and its own speech services.

        The session's voice reaches the text-to-speech, and the TTS mode its synthesis mode, through the services'
        body overrides. Nothing connects until :meth:`start`.
        """
        live = settings.live
        voice = session.audio.output.voice
        overrides = {"tts_voice_id": voice["id"]} if isinstance(voice, dict) else {}
        speech = pick(SPEECH_MODES, settings.tts_mode, "speech.tts_mode")()
        return cls(
            session,
            transport,
            connection,
            live_config=live,
            frontend_endpoint=catalog.model(live.slot, overrides),
            instructions=session.instructions or catalog.prompt(live.prompt),
            stt=build_stt(overrides, catalog.service("asr")),
            tts=build_tts({**overrides, "tts_synthesis_mode": speech.synthesis_mode}, catalog.service("tts")),
            speech=speech,
        )

    @classmethod
    def describe(cls, settings: CascadeAdapterSettings, catalog: ServiceCatalog) -> dict:
        """The adapter's half of ``GET /v1/live/info``: its talker, prompt set, guards, speech and instructions."""
        live = settings.live
        endpoint = catalog.model(live.slot)
        return {
            "frontend": {
                "provider": live.provider,
                "slot": live.slot,
                "model": endpoint.model,
                "base_url": endpoint.base_url,
                "prompt": live.prompt,
            },
            "prompt_version": live.prompt_version,
            "guards": asdict(live.guards),
            "speech": {"tts_mode": settings.tts_mode},
            "instructions": {"frontend": catalog.prompt(live.prompt)},
        }

    @classmethod
    def readiness(cls, settings: CascadeAdapterSettings, catalog: ServiceCatalog) -> ServiceReadiness:
        """Whether the deployment's speech-to-text, text-to-speech and talker model accept connections."""
        slot = settings.live.slot
        return EndpointReadiness(
            {
                "asr": speech_server(catalog.service("asr")),
                "tts": speech_server(catalog.service("tts")),
                slot: catalog.model(slot).base_url,
            }
        )

    # ---------------------------------------------------------------------------------- start
    async def start(self, session: SessionPort, delegation: DelegationPort) -> None:
        """Build the cascade and the pipeline, start the cascade (its presenter and silence timer) and the pipeline.

        The cascade delegates to ``delegation``, the session's delegation part (its server delegation, or in client
        mode its client delegation). It returns once the pipeline runs.
        """
        self.session, self.delegation = session, delegation
        self.agent = build_agent(self.live_config, self.frontend_endpoint, frontend_client=self.frontend_client)
        context = LLMContext(seed_messages(self.config, self.instructions))
        # Only the caller's turns come from Pipecat's aggregator; the cascade writes its own lines (cascade.utterances).
        user_aggregator = LLMContextAggregatorPair(
            context, user_params=self.user_params or build_user_aggregator_params(False)
        ).user()
        self.live = LiveTalkerProcessor()
        self.cascade = CascadeSession(
            self.agent,
            self.live_config,
            ContextConversation(context),
            delegation=self.delegation,
            say=self.live.speak,
            interrupt=self.live.interrupt,
            reports_playout=True,  # the playout tracker reports what each line played
            reports_sentences=self.speech.reports_sentences,
        )
        self.live.attach(self.cascade)
        self.cascade.start()
        speech_out = [self.tts, *self.speech.text_hold(), SpeechPreroll(self.live_config.tts_preroll_ms)]
        processors = self.audio_pipeline(
            [self.stt, user_aggregator, self.live, *speech_out],
            transport=self.transport,
            connection=self.connection,
            audio_in_rate=LIVE_AUDIO_IN_RATE,
        )
        self.playout.listen(self.cascade.playout)
        await self.run_pipeline(processors, stt=self.stt, tts=self.tts)

    def context_tokens(self) -> int | None:
        """Return the tokens the talker's conversation holds now, for the session's context meter."""
        if self.cascade is None:
            return None
        return approx_tokens(json.dumps(self.cascade.conversation.messages(), default=str))

    def context_window_tokens(self) -> int | None:
        """The talker's memory budget (``memory.context_tokens``), for the session's context meter."""
        if self.cascade is None:
            return None
        return self.cascade.memory.context_tokens

    async def history(self) -> dict | None:
        """The adapter's half of ``GET /v1/live/sessions/{id}/history``, as plain JSON; ``None`` before the start.

        ``frontend`` is what the talker holds (:meth:`frontend_history`). ``backend`` holds what the adapter knows of
        the backend's work: whether any delegation is open (``busy``) and what the next delegation will carry
        (``pending``). The delegation part adds the backend's model, prompt and items. A delegation part that reports
        no events (client mode) has no server-side backend, so ``backend`` is ``None``.
        """
        frontend = self.frontend_history()
        if frontend is None:
            return None
        backend = None
        if self.delegation.reports_events:
            backend = {
                "busy": bool(self.cascade.worknote.open()),
                "pending": self.cascade.heard_log.transcript(),  # the labeled lines the next delegation carries
            }
        return json.loads(json.dumps({"frontend": frontend, "backend": backend}, default=str))

    def frontend_history(self) -> dict | None:
        """What the talker holds now, for the ``frontend`` part of the session's history; ``None`` before the start.

        ``messages`` is the talker's voice history as its next call sends it (the set labels a caller turn that follows
        reference data), and ``prompt`` the instructions that call sends, as ``{source, text}`` parts in order, the
        provider's reply format last. The delegation part adds the backend's part.
        """
        if self.cascade is None:
            return None
        messages = self.cascade.conversation.messages()
        _, history = split_messages(messages)
        return {
            "model": self.frontend_endpoint.model,
            "prompt": self._talker_prompt(messages),
            "messages": self.agent.talker.labeled(history),
        }

    def _talker_prompt(self, messages: list[dict]) -> list[dict]:
        """The talker's routing instructions by source, as the talker lays them out (``Talker.routing_parts``).

        The client's instructions are the seed's system message, then each instruction append, joined as the talker's
        memory joins them; the provider's own reply format, when it adds one, comes last.
        """
        base = (
            "session.instructions" if self.config.instructions else f"prompts.yaml: {self.live_config.prompt} + channel"
        )
        instructions = [PromptPart(base, self.instructions)]
        instructions += [
            PromptPart("session.instructions.append", message_text(m), "\n\n")
            for m in messages[1:]
            if m.get("role") in ("system", "developer")
        ]
        parts = [
            {"source": part.source, "text": part.text}
            for part in self.agent.talker.routing_parts(instructions, self.cascade.turns.work_line())
        ]
        frontend = self.agent.frontend
        if frontend.reply_format:
            parts.append({"source": f"reply format ({frontend.name} frontend)", "text": frontend.reply_format})
        return parts

    # ----------------------------------------------------------------------------- shutdown
    async def close(self) -> None:
        """Stop the pipeline, then close the cascade (its presenter, the silence timer and the talker's model), once.

        This is the cascade's only close: its Pipecat processor closes nothing, and the session closes its delegation
        part itself. A later call only stops a pipeline that started after the first call: a session closed while
        it was starting closes its parts again once the start returns.
        """
        if self._closed:
            if self.runner_task is not None and not self.runner_task.done():
                await self.stop_pipeline()
            return
        self._closed = True
        await self.stop_pipeline()
        if self.cascade is not None:
            await self.cascade.close()

    # ------------------------------------------------------------------- the frontend adapter
    async def append_instructions(
        self, text: str, client_event_id: str | None = None, delegation_id: str | None = None
    ) -> None:
        """Interrupt current speech and evaluate the new direction, for the whole call (``delegation_id`` unread)."""
        self._spawn(self.cascade.on_instruction(text))

    async def append_thinking(
        self, text: str, client_event_id: str | None = None, delegation_id: str | None = None
    ) -> None:
        """Retain silent context for the call and its next delegation (``delegation_id`` unread)."""
        self.cascade.on_thinking(text)

    async def append_commentary(
        self, text: str, client_event_id: str | None = None, delegation_id: str | None = None
    ) -> None:
        """Retain the facts and phrase them for the caller; ``delegation_id`` is the client delegation they answer."""
        self._spawn(self.cascade.on_commentary(text, delegation_id))

    def typed(self, text: str) -> None:
        """The application typed a question (``TypedReader``): the talker's memory holds it, never as a turn."""
        if self.cascade is not None:
            self.cascade.on_typed(text)
