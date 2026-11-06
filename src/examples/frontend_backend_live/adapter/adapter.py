# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The cascade adapter: serves a live session with speech in, the talker, and speech out.

With the frontend's build (``examples.frontend_backend_live.frontend``), it runs the Pipecat pipeline whose talker is
Pipecat's LLM service, and the voice that acts on its decisions (``examples.frontend_backend_live.voice``). The client
supplies the prompts and tools; the server supplies the models and the speech services from the registry. The session
hands the adapter its delegation part (``gateways.live.ports.DelegationPort``): the server delegation in ``responses``
mode, whose official events are the voice's only view of its delegations, or the client delegation in client mode,
which announces each delegation to the client app. The session opens and closes the delegation part; the adapter owns
its pipeline and its voice.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, ClassVar, Self

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
from examples.shared.pipeline_utils import PIPELINE_AUDIO_IN_SAMPLE_RATE, build_user_aggregator_params
from gateways.live.protocol import SessionConfig

if TYPE_CHECKING:
    from examples.frontend_backend_live.adapter.contract import ServiceCatalog, ServiceReadiness
    from examples.frontend_backend_live.frontend import Frontend
    from gateways.live.ports import DelegationPort, MediaConnection, SessionPort

LIVE_AUDIO_IN_RATE = PIPELINE_AUDIO_IN_SAMPLE_RATE  # the rate the cascade hears the caller at: the shared input rate
SEED_ROLES = {"developer": "developer", "user": "user", "assistant": "assistant"}


def own_instructions(config: SessionConfig) -> str | None:
    """The session's own instructions; ``None`` when it sets none or blank ones, which get the server's default."""
    return config.instructions if (config.instructions or "").strip() else None


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
    """``CascadeAdapter.Settings``: the validated ``frontend.cascade`` block and what its names pick.

    The voice's own names in the block are picked here too, so a bad value fails the root's read.
    """

    live: LiveConfig
    talkers: Callable[[str], type[TalkerChat]]  # the talker's service, by the style the block names for its model
    turn_end: SmartTurn  # the caller's turns, with the speech block's settings
    response_formats: Callable[[str], ResponseFormat]  # the talker's level, by the name its services entry gives
    message_text: Callable[[dict], str]  # a message item's text, which the voice reads results with

    @classmethod
    def parse(cls, block: Mapping[str, Any], *, pick: Callable[..., Any], message_text: Callable[[dict], str]) -> Self:
        """Validate the class's settings: the ``frontend.cascade`` block, read once (:class:`LiveConfig`).

        ``pick`` and ``message_text`` are the delegation core's (``examples.frontend_backend_live.delegation_core``),
        which the composition root hands in.

        Raises:
            ValueError: The block, or a part of it, is malformed (a pydantic ``ValidationError`` names each key at
                fault), ``talker_style`` names a style the talker does not have (the message lists the known ones),
                or the caller's turns refuse the speech block's seconds.
        """
        live = LiveConfig.model_validate(block)
        talkers = partial(pick, LLM_SERVICES, kind="talker_style")
        for style in live.talker_style.values():
            talkers(style)
        return cls(
            live=live,
            talkers=talkers,
            turn_end=SmartTurn(live.speech),
            response_formats=partial(pick, RESPONSE_FORMATS, kind="response_format"),
            message_text=message_text,
        )


# ------------------------------------------------------------------------------------------- the adapter
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
        user_params=None,
        talker: type[TalkerChat] = TalkerChat,
        response_format: ResponseFormat = RESPONSE_FORMATS[JsonSchema.name],
        frontend_client=None,
    ):
        """Hold everything the session needs; nothing runs until :meth:`start`.

        Args:
            config: The session's startup configuration.
            transport: ``webrtc`` or ``websocket``.
            connection: The WebRTC connection (``webrtc`` only).
            live_config: The validated example configuration.
            frontend_endpoint: The talker's model endpoint.
            instructions: The talker's instructions.
            prompts: The session's read of the example's prompt file: the cascade's texts and lines.
            stt: The speech-to-text service.
            tts: The text-to-speech service.
            item_text: A message item's text, which the voice reads results with (the settings' ``message_text``).
            user_params: The user aggregator's parameters, from the caller's turns. Without them the session gets the
                shared ones.
            talker: The talker's service, of the style the block names for its model (``talker_style``).
            response_format: The talker's level, which its services entry names.
            frontend_client: Optional injected chat client for the talker.
        """
        super().__init__()
        self.config, self.live_config = config, live_config
        self.delegation: DelegationPort | None = None  # the session's delegation part, from start
        self.transport, self.connection, self.frontend_endpoint = transport, connection, frontend_endpoint
        self.instructions, self.stt, self.tts, self.user_params = instructions, stt, tts, user_params
        self.prompts, self.talker = prompts, talker
        self.item_text, self.response_format, self.frontend_client = item_text, response_format, frontend_client
        self.frontend: Frontend | None = None
        self.voice: Voice | None = None
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

        The talker's style is the one ``talker_style`` names for its endpoint's services key, and its endpoint names
        its level (``response_format``). The session's voice reaches the text-to-speech through the services' body
        overrides, and the text-to-speech speaks in its catalog entry's synthesis mode. The caller's turns (``speech``)
        build the user aggregator's settings. Nothing connects until :meth:`start`.

        The session reads the example's prompt file once, here: a session that sets no or blank instructions gets the
        talker prompt that ``prompt`` names, and the cascade gets the file's prompt set and lines.

        Raises:
            ValueError: The talker's services entry names a ``response_format`` the talker does not have (the message
                lists the known ones).
        """
        live = settings.live
        prompts = load_prompts()
        voice = session.audio.output.voice
        overrides = {"tts_voice_id": voice["id"]} if isinstance(voice, dict) else {}
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
            tts=build_tts(overrides, catalog.service("tts")),
            item_text=settings.message_text,
            user_params=settings.turn_end.user_params(welcome_enabled=False),
            talker=settings.talkers(live.style_of(endpoint.service_key)),
            response_format=settings.response_formats(endpoint.response_format),
        )

    @classmethod
    def describe(cls, settings: CascadeAdapterSettings, catalog: ServiceCatalog) -> dict:
        """The adapter's half of ``GET /v1/live/info``: its talker and speech, ``tts_mode`` the TTS's synthesis mode."""
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
        """Build the frontend, start the voice (its quiet task) and the pipeline.

        The frontend's build puts the pipeline between the session's edges, and its voice delegates to ``delegation``,
        the session's delegation part (its server delegation, or in client mode its client delegation). It returns once
        the pipeline runs.
        """
        self.session, self.delegation = session, delegation
        live = self.live_config
        talker = self.talker.from_config(
            self.frontend_endpoint, live.reliability, self.response_format, self.frontend_client
        )
        spec = FrontendSpec(
            config=live,
            talker=talker,
            prompts=self.prompts.cascade,
            lines=self.prompts.lines,
            seed=seed_messages(self.config, self.instructions),
            stt=self.stt,
            tts=self.tts,
            user_params=self.user_params or build_user_aggregator_params(False),
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
        """Return the tokens the talker's next routing call sends now, for the session's context meter.

        The count is Pipecat's estimate of a context (``LLMContextSummarizationUtil.estimate_context_tokens``), the one
        its own context summary measures with. The request is the routing call's from the talker's labeled view, which
        reads in the roles the call sends.
        """
        if self.voice is None:
            return None
        voice = self.voice
        routing = voice.layout.routing("\n\n".join(voice.instructions), voice.worknote.line(), "")
        messages = routing.messages(voice.conversation(labeled=True))
        return LLMContextSummarizationUtil.estimate_context_tokens(LLMContext(messages))

    def context_window_tokens(self) -> int | None:
        """The talker's memory budget (``memory.context_tokens``), for the session's context meter."""
        if self.voice is None:
            return None
        return self.live_config.memory.context_tokens

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
                "busy": bool(self.voice.worknote.open()),
                # The lines the next delegation carries, each as its kind and the labeled line the backend reads.
                "pending": [{"kind": kind, "text": text} for kind, text in self.voice.heard_log.lines()],
            }
        return json.loads(json.dumps({"frontend": frontend, "backend": backend}, default=str))

    def frontend_history(self) -> dict | None:
        """What the talker holds now, for the ``frontend`` part of the session's history; ``None`` before the start.

        ``messages`` is the talker's history as its next routing call sends it (the set labels a caller turn that
        follows reference data), and ``prompt`` the instructions that call sends, as ``{source, text}`` parts in order,
        the routing call's rules after them (the decision rules, or the tool-calling talker's ``tool_rules``). Each
        message carries its kind (``examples.frontend_backend_live.voice.Kind``), and a routing tool's call or result
        its tool fields. The delegation part adds the backend's part.
        """
        if self.voice is None:
            return None
        return {
            "model": self.frontend_endpoint.model,
            "prompt": self._talker_prompt(),
            "messages": self.voice.conversation(labeled=True),
        }

    def _talker_prompt(self) -> list[dict]:
        """The talker's routing instructions by source, as the talker lays them out (``Layout.routing_parts``).

        The client's instructions are the seed's system message, then the seed's developer items and each instruction
        append, joined as the voice joins them; the routing call's rules come after them (``Layout.routing_rules``),
        and the runtime data the call sends after the conversation last.
        """
        own = own_instructions(self.config)
        base = "session.instructions" if own else f"agent_prompts.Talker.{self.live_config.prompt}"
        first, *appended = self.voice.instructions
        instructions = [PromptPart(base, first)]
        instructions += [PromptPart("session.instructions.append", text, "\n\n") for text in appended]
        layout = self.voice.layout
        parts = [{"source": part.source, "text": part.text} for part in layout.routing_parts(instructions)]
        parts.append({"source": layout.routing_rules.source, "text": layout.routing_rules.text})
        parts.append({"source": "runtime data, after the conversation", "text": self.voice.runtime_data()})
        return parts

    # ----------------------------------------------------------------------------- shutdown
    async def close(self) -> None:
        """Stop the pipeline, then close the voice (its tasks and timers), once.

        The session closes its delegation part itself. A later call only stops a pipeline that started after the first
        call: a session closed while it was starting closes its parts again once the start returns.
        """
        if self._closed:
            if self.runner_task is not None and not self.runner_task.done():
                await self.stop_pipeline()
            return
        self._closed = True
        await self.stop_pipeline()
        if self.voice is not None:
            await self.voice.close()

    # ------------------------------------------------------------------- the frontend adapter
    async def append_instructions(
        self, text: str, client_event_id: str | None = None, delegation_id: str | None = None
    ) -> None:
        """Interrupt current speech and evaluate the new direction, for the whole call (``delegation_id`` unread)."""
        self._spawn(self.voice.instruction(text))

    async def append_thinking(
        self, text: str, client_event_id: str | None = None, delegation_id: str | None = None
    ) -> None:
        """Retain silent context for the call and its next delegation (``delegation_id`` unread)."""
        self.voice.thinking(text)

    async def append_commentary(
        self, text: str, client_event_id: str | None = None, delegation_id: str | None = None
    ) -> None:
        """Retain the facts and phrase them for the caller; ``delegation_id`` is the client delegation they answer."""
        self.voice.commentary(text, delegation_id)

    def typed(self, text: str) -> None:
        """The application typed a question (``TypedReader``): the talker's context holds it, never as a turn."""
        if self.voice is not None:
            self.voice.typed(text)
