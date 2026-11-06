# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""Shared doubles of the frontend adapter's tests and of whole calls.

Pipecat stand-ins, and what a whole call is served by, as the roots build it: a session's parts (:func:`cascade_parts`)
and the gateway's one core (:func:`chat_core`), its backend LLM on a scripted model client.
"""

import asyncio
import time

from pipecat.frames.frames import (
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMTextFrame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    TextType,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.aggregators.llm_response_universal import LLMUserAggregatorParams
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.turns.user_start import ExternalUserTurnStartStrategy
from pipecat.turns.user_stop import ExternalUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from examples.frontend_backend_live.adapter.adapter import CascadeAdapter
from examples.frontend_backend_live.adapter.connection import LiveWebRTCConnection
from examples.frontend_backend_live.config_manager.schema import LiveConfig, ModelEndpoint, ReliabilityConfig
from examples.frontend_backend_live.delegation_core.api.local import LocalResponsesClient
from examples.frontend_backend_live.delegation_core.api.service import ResponsesService
from examples.frontend_backend_live.delegation_core.app import DelegationCore, build_core
from examples.frontend_backend_live.delegation_core.backend.adapters import ADAPTERS
from examples.frontend_backend_live.delegation_core.backend.llm import LlmSettings, LlmSource
from examples.frontend_backend_live.delegation_core.config import CoreSettings
from examples.frontend_backend_live.delegation_core.im.input import BackendTask
from examples.frontend_backend_live.delegation_core.items import message_text
from examples.frontend_backend_live.prompts import load_prompts
from examples.shared.delegation_contract import ModelEndpoint as CoreEndpoint
from examples.shared.observe import log_observation
from gateways.live.ports import SessionParts
from gateways.live.server_delegation import ServerDelegation
from tests.unit.core_helpers import CORE_BLOCK, CORE_TEXTS, chunk, tool_fragment


class FakeSTT(FrameProcessor):
    """Speech-to-text stand-in: the test says a sentence instead of sending speech audio."""

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)

    async def say(self, text):
        await self.push_frame(UserStartedSpeakingFrame())
        await self.push_frame(
            TranscriptionFrame(text=text, user_id="caller", timestamp=str(time.time()), finalized=True)
        )
        await self.push_frame(UserStoppedSpeakingFrame())

    async def propose(self, text):
        """The caller says ``text`` in a turn proposed as a speech service with turn detection proposes it.

        The user aggregator resolves the proposal: it starts the turn and broadcasts the interruption, as it does for
        a turn its own strategies detect, so a turn said over the bot cuts what it was doing.
        """
        await self.push_frame(ProposedUserStartedSpeakingFrame())
        await asyncio.sleep(0.02)
        await self.push_frame(
            TranscriptionFrame(text=text, user_id="caller", timestamp=str(time.time()), finalized=True)
        )
        await self.push_frame(ProposedUserStoppedSpeakingFrame())


class FakeTTS(FrameProcessor):
    """Text-to-speech stand-in: 100 ms of tone per word, with the word frames a real TTS emits.

    A word the streamed text has not finished yet waits for the rest of it, or for the response's end.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.spoken: list[str] = []
        self._started = False
        self._partial = ""

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMTextFrame):
            text = self._partial + frame.text
            words = text.split()
            self._partial = words.pop() if words and not text[-1].isspace() else ""
            await self._speak(words)
            return
        if isinstance(frame, InterruptionFrame):
            self._partial = ""
        if isinstance(frame, LLMFullResponseEndFrame):
            words, self._partial = self._partial.split(), ""
            await self._speak(words)
            if self._started:
                self._started = False
                await self.push_frame(TTSStoppedFrame())
        await self.push_frame(frame, direction)

    async def _speak(self, words):
        for word in words:
            if not self._started:
                self._started = True
                await self.push_frame(TTSStartedFrame())
            self.spoken.append(word)
            await self.push_frame(TTSAudioRawFrame(audio=b"\x10\x10" * 2400, sample_rate=24000, num_channels=1))
            await self.push_frame(TTSTextFrame(text=word, text_type=TextType.WORD))


def _delegation(backend, *, tools=(), observe=log_observation, **delegation):
    """The call's server delegation, as the gateway builds it, on a core ``build_core`` makes from the shipped settings.

    The core's backend LLM runs every round on ``backend`` (a scripted provider, ``ScriptedBackend``; or a function that
    makes one for each IM session), with ``tools`` as each IM session's server tool sources; the core has the shipped
    settings, with a short close grace, and ``delegation`` sets keys of its ``delegation`` block.
    """
    sources = tuple(tools)
    make = backend if callable(backend) else lambda: backend

    class ScriptedLlm(LlmSource):
        @classmethod
        def from_settings(cls, block, *, endpoints, tools, vocabulary):
            settings = LlmSettings.model_validate(block)
            return cls(settings, endpoints=endpoints, tools=lambda: list(sources), make_provider=lambda _: make())

    settings = CoreSettings.parse(CORE_BLOCK, CORE_TEXTS)
    delegation = settings.delegation.model_copy(update={"close_grace_seconds": 0.01, **delegation})
    settings = settings.model_copy(update={"delegation": delegation})
    core = build_core(
        settings,
        endpoints=lambda model: CoreEndpoint(model=model or "test/model", base_url="http://localhost:1/v1"),
        defaults=BackendTask("test/model", "Do the task.", []),
        adapters={**ADAPTERS, "llm": lambda: ScriptedLlm},
    )
    return ServerDelegation(core_client(core), observe=observe)


def core_client(core: DelegationCore) -> LocalResponsesClient:
    """The in-process client of a core's official endpoints, as the root makes it (``factory.core_client``)."""
    service = ResponsesService(core.session, stateless_session=core.one_shot_session, backend=core)
    return LocalResponsesClient(service, close_seconds=core.close_seconds, close_core=core.aclose)


def core_session(delegation):
    """The IM session that serves the server delegation's conversation in its in-process core."""
    return delegation.client.service.sessions[delegation.conversation]


def _decision_chunks(text, finish="stop"):
    half = len(text) // 2
    return [chunk(text[:half]), chunk(text[half:]), chunk(finish=finish)]


# ------------------------------------------------------------- the end-to-end tests' doubles
ENDPOINT = ModelEndpoint(model="org/org/model", base_url="http://localhost:1/v1")
LIVE_CONFIG = LiveConfig(
    slot="llm",
    reliability=ReliabilityConfig(provider_retries=0),
)
STATIC_BACKEND = "Static backend prompt from config."
MENU_TOOL = {
    "type": "function",
    "name": "get_menu",
    "description": "Look up the menu.",
    "parameters": {"type": "object", "properties": {"category": {"type": "string"}}},
}


def transcript_turns() -> LLMUserAggregatorParams:
    """The user aggregator's settings of whole-call tests: a caller turn starts and ends with the line the test says."""
    return LLMUserAggregatorParams(
        user_turn_strategies=UserTurnStrategies(
            start=[ExternalUserTurnStartStrategy()], stop=[ExternalUserTurnStopStrategy()]
        )
    )


def cascade_parts(
    config,
    transport,
    ice_servers,
    *,
    chat,
    stt,
    tts,
    live_config=LIVE_CONFIG,
    instructions="You are a kiosk assistant.",
) -> SessionParts:
    """One session's parts as the root builds them: the cascade adapter, here with fake speech and a scripted talker.

    ``chat`` is the talker's model client; ``instructions`` are the talker's when the session sets
    none. The gateway reaches the backend through its one core (:func:`chat_core`), never through the parts.
    A WebRTC session's peer connection is made on the gateway's ``ice_servers``, as the root makes it.
    """
    connection = None if ice_servers is None else LiveWebRTCConnection.from_ice_servers(ice_servers)
    adapter = CascadeAdapter(
        config,
        transport,
        connection,
        live_config=live_config,
        frontend_endpoint=ENDPOINT,
        instructions=config.instructions or instructions,
        prompts=load_prompts(),
        stt=stt,
        tts=tts,
        item_text=message_text,
        user_params=transcript_turns(),
        frontend_client=chat,
    )
    return SessionParts(frontend=adapter, connection=connection)


def chat_adapters(chat) -> dict:
    """The core's adapter table with its backend LLM on the scripted model client ``chat``.

    The backend LLM keeps its block's settings and its own provider (chat completions), with ``chat`` as the provider's
    client, so a test scripts the backend model's chunks as it scripts the talker's.
    """

    class ChatLlm(LlmSource):
        @classmethod
        def from_settings(cls, block, *, endpoints, tools, vocabulary):
            settings = LlmSettings.model_validate(block)
            return cls(
                settings,
                endpoints=endpoints,
                tools=tools,
                make_provider=lambda endpoint: settings.provider(endpoint, settings.limits, client=chat),
            )

    return {**ADAPTERS, "llm": lambda: ChatLlm}


def chat_core(chat, *, instructions=STATIC_BACKEND, models=(), core=CORE_BLOCK, **delegation) -> DelegationCore:
    """The one core of a whole-call test: ``build_core`` on the shipped core settings, its backend LLM on ``chat``.

    The backend LLM runs on the scripted model client ``chat`` (:func:`chat_adapters`). The catalog lists
    :data:`ENDPOINT`'s model and ``models``, all at its address, and the core reports any other; a request that sets
    nothing gets the endpoint's model and ``instructions``. ``core`` gives other core settings, and ``delegation`` sets
    keys of its ``delegation`` block; the close grace is short unless set, so a closed session's work stops at once.
    """
    listed = {ENDPOINT.model, *models}

    def endpoints(model):
        if model is not None and model not in listed:
            raise LookupError(f"The catalog lists no model {model!r}")
        return CoreEndpoint(model=model or ENDPOINT.model, base_url=ENDPOINT.base_url)

    settings = CoreSettings.parse(core, CORE_TEXTS)
    delegation = settings.delegation.model_copy(update={"close_grace_seconds": 0.01, **delegation})
    settings = settings.model_copy(update={"delegation": delegation})
    return build_core(
        settings,
        endpoints=endpoints,
        defaults=BackendTask(ENDPOINT.model, instructions, []),
        adapters=chat_adapters(chat),
    )


def core_hook(core: DelegationCore):
    """The gateway's ``core`` hook on ``core``: it gives the core's in-process client, as the root's hook does.

    The hook keeps the client as ``hook.client``, so a test can reach the IM session of a conversation
    (``hook.client.service.sessions``).
    """
    client = core_client(core)

    async def hook():
        return client

    hook.client = client
    return hook


def tool_call_round(call_id="call_1", arguments='{"category":"tea"}'):
    return [
        chunk(tool_calls=[tool_fragment(0, call_id, "get_menu", arguments[:8])]),
        chunk(tool_calls=[tool_fragment(0, arguments=arguments[8:])], finish="tool_calls"),
    ]
