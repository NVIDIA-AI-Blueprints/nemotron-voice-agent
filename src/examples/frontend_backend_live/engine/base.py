# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What every live engine shares: the session's settings, delegation, and the client's tool calls.

A live engine serves one protocol session. The frontend (how the caller is heard and spoken to) varies; the backend
side does not. This base class owns the :class:`ClientToolGateway`, the relay of backend events to the client, and the
commands that touch delegation (``item_create``, ``response_create``, ``update_backend`` ...). A subclass supplies
the frontend: it sets ``self.session`` and ``self.coordinator``, calls :meth:`build_client_delegation` and implements
``start``, ``close``, ``feed_audio``, ``set_muted`` and the three ``append_*`` commands.
"""

from __future__ import annotations

import asyncio

from loguru import logger

from examples.frontend_backend_live.config_manager.schema import LiveConfig, ModelEndpoint
from examples.frontend_backend_live.delegation.coordinator import DelegationCoordinator
from examples.frontend_backend_live.delegation.events import BackendEventRelay
from examples.frontend_backend_live.delegation.tasks import CLIENT, RESPONSES, BackendTask, backend_task_for
from examples.frontend_backend_live.tool_calling.client_tools import ClientToolGateway
from live.protocol import SessionConfig
from live.session import LiveProtocolSession


class BaseLiveEngine:
    """Delegation and client tool calls for one live session; the frontend is a subclass's concern."""

    def __init__(
        self,
        config: SessionConfig,
        *,
        live_config: LiveConfig,
        backend_endpoint: ModelEndpoint,
        backend_instructions: str,
    ):
        """Hold the session's configuration; nothing runs until :meth:`start`.

        Args:
            config: The session's startup configuration.
            live_config: The validated example configuration.
            backend_endpoint: The backend's model endpoint.
            backend_instructions: The backend's static instructions, used when the session sets none.
        """
        self.config, self.live_config = config, live_config
        self.backend_endpoint, self.backend_instructions = backend_endpoint, backend_instructions
        self.session: LiveProtocolSession | None = None
        self.coordinator: DelegationCoordinator | None = None
        self.tools: ClientToolGateway | None = None
        self.relay: BackendEventRelay | None = None
        self.background: set[asyncio.Task] = set()

    # ------------------------------------------------------------------------------- delegation
    @property
    def mode(self) -> str:
        """The delegation mode this session runs in."""
        return RESPONSES if self.config.mode == RESPONSES else CLIENT

    def initial_backend_task(self) -> BackendTask | None:
        """The backend settings the session starts with (``None`` in client mode)."""
        return backend_task_for(self.config, self.backend_endpoint.model, self.backend_instructions)

    def build_client_delegation(self) -> ClientToolGateway:
        """Create the tool gateway the client uses to answer function calls, and the relay that reports to it.

        Call after ``self.session`` is set. Pass ``self.relay.backend_event`` and ``self.relay.client_delegation`` to
        the :class:`DelegationCoordinator`.
        """
        self.tools = ClientToolGateway(
            submit=self._submit_typed,
            busy=lambda: bool(self.coordinator) and self.coordinator.backend_busy,
            tool_timeout_seconds=self.live_config.reliability.tool_timeout_seconds,
        )
        self.relay = BackendEventRelay(self.session, self.tools)
        return self.tools

    async def _submit_typed(self, items: list[dict]) -> None:
        await self.coordinator.submit_items(items)

    # --------------------------------------------------------------- session updates and commands
    def validate_backend_model(self, model: str) -> None:
        """Raise ``ValueError`` if the backend cannot serve ``model``."""
        self.coordinator.validate_model(model)

    def update_backend(self, config: SessionConfig) -> None:
        """Apply a ``session.update`` to the backend settings used from the next round on."""
        self.config = config
        self.coordinator.update_settings(config, self.backend_endpoint.model, self.backend_instructions)

    def knows_delegation(self, delegation_id: str) -> bool:
        """Return True for a client-mode delegation id this session issued."""
        return delegation_id in self.coordinator.client_delegations

    def item_create(self, item: dict) -> None:
        """Accept a function result or typed user message."""
        self.tools.item(item)

    async def response_create(self) -> None:
        """Continue a complete function-result batch, or run queued typed input."""
        await self.tools.create()

    def _spawn(self, coro) -> None:
        """Run a command in the background; a failure is reported to the client without ending the session."""
        task = asyncio.create_task(coro)
        self.background.add(task)

        def done(finished: asyncio.Task) -> None:
            self.background.discard(finished)
            if not finished.cancelled() and finished.exception():
                logger.error(f"Live command failed: {finished.exception()!r}")
                self.session.spawn(self.session.fail("A command could not be processed."), "command-failure")

        task.add_done_callback(done)
