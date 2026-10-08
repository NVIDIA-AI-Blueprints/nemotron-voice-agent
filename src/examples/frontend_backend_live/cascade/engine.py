# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The cascade engine: serves a live session with speech in, the talker, commentary, and speech out.

It builds the Pipecat pipeline (speech-to-text, user aggregator, :class:`LiveTalkerProcessor`, text-to-speech) around a
:class:`CascadeSession`. The client supplies the prompts and tools (``session.instructions`` and
``delegation.responses``); the server supplies the models and the speech services from the registry.
"""

from __future__ import annotations

import json

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair

from examples.frontend_backend_live.cascade.agent import LiveAgent, build_agent
from examples.frontend_backend_live.cascade.conversation import ContextConversation
from examples.frontend_backend_live.cascade.processor import LiveTalkerProcessor
from examples.frontend_backend_live.cascade.session import CascadeSession
from examples.frontend_backend_live.common.ids import approx_tokens
from examples.frontend_backend_live.config_manager.schema import LiveConfig, ModelEndpoint
from examples.frontend_backend_live.engine.pipeline_host import PipelineLiveEngine
from examples.shared.pipeline_utils import build_user_aggregator_params
from live.media.processors import LiveInputGate
from live.protocol import SessionConfig
from live.session import LiveProtocolSession

LIVE_AUDIO_IN_RATE = 16000
SEED_ROLES = {"developer": "developer", "user": "user", "assistant": "assistant"}


def seed_messages(config: SessionConfig, instructions: str) -> list[dict]:
    """The talker's starting context: its instructions, then the client's input history."""
    messages = [{"role": "system", "content": instructions}]
    for item in config.input:
        part = item["content"][0]
        messages.append({"role": SEED_ROLES[item["role"]], "content": part["text"]})
    return messages


class CascadeLiveEngine(PipelineLiveEngine):
    """Serves one live session with the cascade."""

    def __init__(
        self,
        config: SessionConfig,
        transport: str,
        connection,
        *,
        live_config: LiveConfig,
        frontend_endpoint: ModelEndpoint,
        backend_endpoint: ModelEndpoint,
        instructions: str,
        backend_instructions: str,
        stt,
        tts,
        user_params=None,
        frontend_client=None,
        backend_client=None,
    ):
        """Hold everything the session needs; nothing runs until :meth:`start`.

        Args:
            config: The session's startup configuration.
            transport: ``webrtc`` or ``websocket``.
            connection: The WebRTC connection (``webrtc`` only).
            live_config: The validated example configuration.
            frontend_endpoint: The talker's model endpoint.
            backend_endpoint: The backend's model endpoint.
            instructions: The talker's instructions.
            backend_instructions: The backend's static instructions, used when the session sets none.
            stt: The speech-to-text service.
            tts: The text-to-speech service.
            user_params: Optional user-aggregator parameters (tests supply a transcript-driven turn strategy).
            frontend_client: Optional injected chat client for the frontend.
            backend_client: Optional injected chat client for the backend.
        """
        super().__init__(
            config,
            live_config=live_config,
            backend_endpoint=backend_endpoint,
            backend_instructions=backend_instructions,
        )
        self.transport, self.connection, self.frontend_endpoint = transport, connection, frontend_endpoint
        self.instructions, self.stt, self.tts, self.user_params = instructions, stt, tts, user_params
        self.frontend_client, self.backend_client = frontend_client, backend_client
        self.agent: LiveAgent | None = None
        self.cascade: CascadeSession | None = None
        self.live: LiveTalkerProcessor | None = None

    # ---------------------------------------------------------------------------------- start
    async def start(self, session: LiveProtocolSession) -> None:
        """Build the cascade and the pipeline, start it, and return once it is running."""
        self.session = session
        self.agent = build_agent(
            self.live_config,
            self.frontend_endpoint,
            self.backend_endpoint,
            frontend_client=self.frontend_client,
            backend_client=self.backend_client,
        )
        context = LLMContext(seed_messages(self.config, self.instructions))
        user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
            context, user_params=self.user_params or build_user_aggregator_params(False)
        )
        self.build_client_delegation()
        self.live = LiveTalkerProcessor()
        self.cascade = CascadeSession(
            self.agent,
            self.live_config,
            ContextConversation(context),
            backend_task=self.initial_backend_task(),
            tools=self.tools,
            say=self.live.speak,
            mode=self.mode,
            on_event=self.relay.backend_event,
            on_client_delegation=self.relay.client_delegation,
            interrupt=self.live.interrupt,
        )
        self.coordinator = self.cascade.coordinator
        self.live.attach(self.cascade)
        self.gate = LiveInputGate(session)
        processors = self.audio_pipeline(
            [self.gate, self.stt, user_aggregator, self.live, self.tts],
            transport=self.transport,
            connection=self.connection,
            audio_in_rate=LIVE_AUDIO_IN_RATE,
            tail=[assistant_aggregator],
        )
        await self.run_pipeline(processors, stt=self.stt, tts=self.tts)

    def context_tokens(self) -> int | None:
        """Return the tokens the talker's conversation holds now, for the session's context meter."""
        if self.cascade is None:
            return None
        return approx_tokens(json.dumps(self.cascade.conversation.messages(), default=str))

    # ----------------------------------------------------------------------------- shutdown
    async def close(self) -> None:
        """Stop the pipeline, the delegation worker and the models."""
        await self.stop_pipeline()
        if self.cascade is not None:
            await self.cascade.close()

    # ------------------------------------------------------------------- the LiveEngine API
    async def append_instructions(self, text: str) -> None:
        """Interrupt current speech and evaluate the new direction."""
        self._spawn(self.cascade.on_instruction(text))

    async def append_thinking(self, text: str) -> None:
        """Retain silent context."""
        self.cascade.on_thinking(text)

    async def append_commentary(self, text: str, delegation_id: str | None) -> None:
        """Retain the facts and phrase them for the caller."""
        self._spawn(self.cascade.on_commentary(text, delegation_id))
