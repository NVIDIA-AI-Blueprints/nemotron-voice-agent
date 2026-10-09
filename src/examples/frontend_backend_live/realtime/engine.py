# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The live engine with a remote realtime model as the frontend.

The remote model hears the caller and speaks to them; delegation, the backend, the client's tools and every live
command are the same as for the cascade (:class:`PipelineLiveEngine`). The model reaches the backend by calling a
``delegate`` function. Backend results and application updates are added to its conversation as system messages.
Enable it with ``frontend.provider: realtime`` in ``config.yaml``.
"""

from __future__ import annotations

import asyncio
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from examples.frontend_backend_live.config_manager.schema import LiveConfig, ModelEndpoint
from examples.frontend_backend_live.delegation.coordinator import DelegationCoordinator
from examples.frontend_backend_live.delegation.thinker import build_thinker
from examples.frontend_backend_live.delegation.worker import DelegatedTask
from examples.frontend_backend_live.engine.pipeline_host import PipelineLiveEngine
from examples.frontend_backend_live.prompts.realtime import (
    COMMENTARY_ITEM,
    DELEGATE_INSTRUCTIONS,
    DELEGATION_STARTED,
    INSTRUCTION_ITEM,
    NEWER_NOTE,
    RESULT_ITEM,
    SPEAK_RESULT,
    SPEAK_UPDATE,
    THINKING_ITEM,
    TOLD_NOTE,
)
from examples.frontend_backend_live.realtime.client import (
    READY_TIMEOUT_SECONDS,
    RealtimeFrontendProcessor,
    RealtimeSettings,
)
from live.media.processors import LiveInputGate
from live.protocol import SessionConfig
from live.session import LiveProtocolSession

REALTIME_PROVIDER = "realtime"
REALTIME_AUDIO_IN_RATE = 24000
HISTORY_ROLES = {"developer": "system", "user": "user", "assistant": "assistant"}
DEFAULT_VOICE = "marin"


def realtime_url(base_url: str, model: str) -> str:
    """Return ``base_url`` with the ``model`` query parameter the realtime endpoint expects."""
    parts = urlparse(base_url)
    query = dict(parse_qsl(parts.query))
    query.setdefault("model", model)
    return urlunparse(parts._replace(query=urlencode(query)))


def voice_name(config: SessionConfig) -> str:
    """The session's requested voice: a name, or the id of a custom voice."""
    voice = config.audio.output.voice
    return voice["id"] if isinstance(voice, dict) else voice or DEFAULT_VOICE


def endpoint_problem(live_config: LiveConfig, endpoint: ModelEndpoint) -> str | None:
    """Return what is wrong with the realtime frontend's endpoint, or ``None`` when it can be used.

    A realtime frontend is reached over a WebSocket. The usual mistake is a services entry the active recipe does not
    offer (the ``cloud`` recipe drops entries without an ``nvcf`` key), which leaves the slot on its default HTTPS
    chat model.
    """
    if endpoint.base_url.startswith(("ws://", "wss://")):
        return None
    return (
        f"frontend.provider is 'realtime', but slot {live_config.frontend.slot!r} resolved to "
        f"{endpoint.base_url!r}, not a ws:// or wss:// URL. Check that the slot's services entry exists, is offered by "
        "examples_registry.yaml, and is available under SERVICE_RECIPE (a custom endpoint needs SERVICE_RECIPE=server)."
    )


class RealtimeLiveEngine(PipelineLiveEngine):
    """Serves one live session with a remote realtime model as the frontend."""

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
        backend_client=None,
    ):
        """Hold everything the session needs; nothing runs until :meth:`start`.

        Args:
            config: The session's startup configuration.
            transport: ``webrtc`` or ``websocket``.
            connection: The WebRTC connection (``webrtc`` only).
            live_config: The validated example configuration.
            frontend_endpoint: The realtime model's endpoint (a ``ws(s)://`` base URL).
            backend_endpoint: The backend's model endpoint.
            instructions: The realtime model's instructions.
            backend_instructions: The backend's static instructions, used when the session sets none.
            backend_client: Optional injected chat client for the backend.

        Raises:
            ValueError: The frontend endpoint is not a ``ws://`` or ``wss://`` URL.
        """
        problem = endpoint_problem(live_config, frontend_endpoint)
        if problem:
            raise ValueError(problem)
        super().__init__(
            config,
            live_config=live_config,
            backend_endpoint=backend_endpoint,
            backend_instructions=backend_instructions,
        )
        self.transport, self.connection = transport, connection
        self.frontend_endpoint, self.instructions, self.backend_client = frontend_endpoint, instructions, backend_client
        self.frontend: RealtimeFrontendProcessor | None = None

    def settings(self) -> RealtimeSettings:
        """Return the remote session's settings.

        The endpoint's ``extra_params`` may carry ``headers``, ``session`` and ``transcription_model`` for servers
        that differ from the default.
        """
        extra = dict(self.frontend_endpoint.extra_params)
        history = [
            {"role": HISTORY_ROLES[item["role"]], "text": item["content"][0]["text"]} for item in self.config.input
        ]
        settings = RealtimeSettings(
            url=realtime_url(self.frontend_endpoint.base_url, self.frontend_endpoint.model),
            api_key=self.frontend_endpoint.api_key,
            model=self.frontend_endpoint.model,
            instructions=f"{self.instructions.rstrip()}\n\n{DELEGATE_INSTRUCTIONS}",
            voice=voice_name(self.config),
            history=history,
            headers=dict(extra.get("headers") or {}),
            session_overrides=dict(extra.get("session") or {}),
        )
        if extra.get("transcription_model"):
            settings.transcription_model = str(extra["transcription_model"])
        return settings

    # ---------------------------------------------------------------------------------- start
    async def start(self, session: LiveProtocolSession) -> None:
        """Build the delegation side and the pipeline, connect to the remote model, and return once it is ready."""
        self.session = session
        thinker = build_thinker(self.live_config, self.backend_endpoint, backend_client=self.backend_client)
        self.build_client_delegation()
        self.coordinator = DelegationCoordinator(
            thinker,
            self.live_config,
            backend_task=self.initial_backend_task(),
            tools=self.tools,
            sink=self,
            mode=self.mode,
            on_event=self.relay.backend_event,
            on_client_delegation=self.relay.client_delegation,
        )
        self.gate = LiveInputGate(session)
        self.frontend = RealtimeFrontendProcessor(
            self.settings(),
            on_delegate=self._on_delegate,
            on_user_text=lambda text: self.coordinator.remember_voice("user", text),
            on_assistant_text=lambda text: self.coordinator.remember_voice("assistant", text),
            on_closed=self._on_closed,
        )
        processors = self.audio_pipeline(
            [self.gate, self.frontend],
            transport=self.transport,
            connection=self.connection,
            audio_in_rate=REALTIME_AUDIO_IN_RATE,
        )
        self.coordinator.start()
        await self.run_pipeline(processors, stt=self.frontend, tts=self.frontend)
        async with asyncio.timeout(READY_TIMEOUT_SECONDS):
            await self.frontend.ready.wait()
        if self.frontend.failure:
            raise RuntimeError(self.frontend.failure)

    async def close(self) -> None:
        """Stop the pipeline and the delegation worker, and release the backend."""
        await self.stop_pipeline()
        if self.coordinator is not None:
            await self.coordinator.close()
            await self.coordinator.thinker.backend.close()

    async def _on_closed(self) -> None:
        if self.session is not None and self.session.status == "active":
            self.session.spawn(self.session.close("connection_lost"), "realtime-closed")

    # -------------------------------------------------------------------------- delegation
    async def _on_delegate(self, request: str) -> str:
        """The model called ``delegate``: hand the request to the backend and answer the call at once."""
        delegation_id = await self.coordinator.delegate(request, record=False)
        return DELEGATION_STARTED % delegation_id

    async def on_result(self, text: str, task: DelegatedTask, told: str | None, *, newer_requests: list[str]) -> None:
        """Give the model a backend result; it speaks it, as being updated if a newer request may change it."""
        note = TOLD_NOTE.format(told=told) if told else ""
        if newer_requests:
            note = f"{note} {NEWER_NOTE.format(asked='; '.join(newer_requests))}".strip()
        await self.frontend.inject(
            RESULT_ITEM.format(delegation_id=task.id, text=text),
            respond=True,
            instructions=SPEAK_RESULT.format(note=note).strip(),
        )

    # ------------------------------------------------------------------- the LiveEngine API
    async def append_instructions(self, text: str) -> None:
        """Interrupt current speech and have the model follow the new direction."""
        self._spawn(self.frontend.inject(INSTRUCTION_ITEM.format(text=text), respond=True, interrupt=True))

    async def append_thinking(self, text: str) -> None:
        """Add silent context."""
        self._spawn(self.frontend.inject(THINKING_ITEM.format(text=text), respond=False))

    async def append_commentary(self, text: str, delegation_id: str | None) -> None:
        """Give the model the facts and ask it to say what matters to the caller."""
        told = self.coordinator.told.pop(delegation_id, None) if delegation_id else None
        note = TOLD_NOTE.format(told=told) if told else ""
        self._spawn(
            self.frontend.inject(
                COMMENTARY_ITEM.format(text=text), respond=True, instructions=SPEAK_UPDATE.format(note=note).strip()
            )
        )
