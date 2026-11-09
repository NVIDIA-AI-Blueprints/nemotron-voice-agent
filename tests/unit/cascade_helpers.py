# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""Doubles of whole calls: Pipecat stand-ins, and a session's parts and the gateway's core as the roots build them."""

import asyncio
import base64
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
from examples.frontend_backend_live.adapter.connection import LivePeer
from examples.frontend_backend_live.config_manager.schema import LiveConfig, ModelEndpoint, ReliabilityConfig
from examples.frontend_backend_live.delegation_core.api.local import LocalResponsesClient
from examples.frontend_backend_live.delegation_core.api.service import ResponsesService
from examples.frontend_backend_live.delegation_core.app import DelegationCore, build_core
from examples.frontend_backend_live.delegation_core.backend.adapters import ADAPTERS
from examples.frontend_backend_live.delegation_core.backend.llm import LlmSettings, LlmSource
from examples.frontend_backend_live.delegation_core.backend.providers import provider_class
from examples.frontend_backend_live.delegation_core.config import CoreSettings
from examples.frontend_backend_live.delegation_core.im.input import BackendTask
from examples.frontend_backend_live.delegation_core.items import message_text
from examples.frontend_backend_live.prompts import load_prompts
from examples.frontend_backend_live.talker import TalkerChat
from examples.shared.delegation_contract import ModelEndpoint as CoreEndpoint
from gateways.live.ports import SessionParts
from tests.unit.core_helpers import CORE_BLOCK, CORE_TEXTS

ENDPOINT = ModelEndpoint(model="org/org/model", base_url="http://localhost:1/v1")
LIVE_CONFIG = LiveConfig(slot="llm", reliability=ReliabilityConfig(provider_retries=0))
MENU_TOOL = {
    "type": "function",
    "name": "get_menu",
    "description": "Look up the menu.",
    "parameters": {"type": "object", "properties": {"category": {"type": "string"}}},
}
RESPONSES_SESSION = {
    "model": "live-1",
    "instructions": "You are a kiosk voice assistant. Delegate every request.",
    "delegation": {
        "type": "responses",
        "responses": {"model": "org/org/model", "instructions": "Look things up.", "tools": [MENU_TOOL]},
    },
}
# 100 ms of silence in the default wire format, 24 kHz PCM16.
SILENCE_100MS = base64.b64encode(bytes(4800)).decode()


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
        """Say ``text`` in a proposed turn, which the user aggregator starts with an interruption."""
        await self.push_frame(ProposedUserStartedSpeakingFrame())
        await asyncio.sleep(0.02)
        await self.push_frame(
            TranscriptionFrame(text=text, user_id="caller", timestamp=str(time.time()), finalized=True)
        )
        await self.push_frame(ProposedUserStoppedSpeakingFrame())


class FakeTTS(FrameProcessor):
    """Text-to-speech stand-in: 100 ms of tone and a word frame per word, once the streamed text has ended it."""

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


def transcript_turns() -> LLMUserAggregatorParams:
    """Return the user aggregator's settings: a caller turn starts and ends with the line the test says."""
    strategies = UserTurnStrategies(start=[ExternalUserTurnStartStrategy()], stop=[ExternalUserTurnStopStrategy()])
    return LLMUserAggregatorParams(user_turn_strategies=strategies)


def scripted_talker(service: type[TalkerChat], chat) -> type[TalkerChat]:
    """Return a subclass of ``service`` whose model client is ``chat``."""

    class Scripted(service):
        def create_client(self, api_key=None, base_url=None, **kwargs):
            return chat

    return Scripted


def cascade_parts(config, transport, ice_servers, *, chat, stt, tts, live_config=LIVE_CONFIG) -> SessionParts:
    """Build a session's parts as the root does, with ``chat`` as the talker's client and a peer for WebRTC."""
    connection = None if ice_servers is None else LivePeer.from_ice_servers(ice_servers)
    adapter = CascadeAdapter(
        config,
        transport,
        connection,
        live_config=live_config,
        frontend_endpoint=ENDPOINT,
        instructions=config.instructions or "You are a kiosk assistant.",
        prompts=load_prompts(),
        stt=stt,
        tts=tts,
        item_text=message_text,
        user_params=transcript_turns(),
        talker=scripted_talker(TalkerChat, chat),
    )
    return SessionParts(frontend=adapter, connection=connection)


def chat_adapters(chat) -> dict:
    """Return the core's adapter table with its backend LLM's provider on the model client ``chat``."""

    class ChatLlm(LlmSource):
        @classmethod
        def from_settings(cls, block, *, endpoints, tools, vocabulary):
            settings = LlmSettings.model_validate(block)
            return cls(
                settings,
                endpoints=endpoints,
                tools=tools,
                make_provider=lambda endpoint: provider_class(endpoint)(endpoint, settings.limits, client=chat),
            )

    return {**ADAPTERS, "llm": lambda: ChatLlm}


def chat_core(chat, *, models=(), **delegation) -> DelegationCore:
    """Build the gateway's core on the shipped settings, with its backend LLM on ``chat`` and ``models`` listed."""
    listed = {ENDPOINT.model, *models}

    def endpoints(model):
        if model is not None and model not in listed:
            raise LookupError(f"The catalog lists no model {model!r}")
        return CoreEndpoint(model=model or ENDPOINT.model, base_url=ENDPOINT.base_url)

    settings = CoreSettings.parse(CORE_BLOCK, CORE_TEXTS)
    delegation = settings.delegation.model_copy(update={"close_grace_seconds": 0.01, **delegation})
    settings = settings.model_copy(update={"delegation": delegation})
    defaults = BackendTask(ENDPOINT.model, "Static backend prompt from config.", [])
    return build_core(settings, endpoints=endpoints, defaults=defaults, adapters=chat_adapters(chat))


def core_hook(core: DelegationCore):
    """Return the gateway's ``core`` hook for ``core``'s in-process client, which ``hook.client`` also holds."""
    service = ResponsesService(core.session, stateless_session=core.one_shot_session, backend=core)
    client = LocalResponsesClient(service, close_seconds=core.close_seconds, close_core=core.aclose)

    async def hook():
        return client

    hook.client = client
    return hook
