# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What a frontend adapter class offers the composition root, so a second adapter is one class and one table entry.

``frontend.adapter`` in the example's settings names an entry of ``FRONTEND_ADAPTERS``. The root reads the
example's catalog once, waits until the class's services are ready, then builds one adapter per session:

1. ``Settings.parse(block, pick=..., message_text=...)``: the class's own block (``frontend.<name>``), unknown keys
   refused, when the root starts. The root hands in the helpers the frontend shares with the delegation core, whose one
   copy only the composition roots import: ``pick`` reads a name table, ``message_text`` a message item's text;
2. ``readiness(settings, catalog).ensure_ready()``: off the event loop and cached; a service still starting refuses
   the session with ``ServiceStarting`` (the root's "still starting" answer);
3. ``from_config(session, transport, connection, settings, catalog)``: builds the session's ``FrontendAdapter``
   (``gateways.live.ports``) and connects nothing; ``start()`` connects.

``describe(settings, catalog)`` is the adapter's half of ``GET /v1/live/info``. Types only.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, Self

if TYPE_CHECKING:
    from examples.frontend_backend_live.config_manager.schema import ModelEndpoint
    from gateways.live.ports import FrontendAdapter, MediaConnection
    from gateways.live.protocol import SessionConfig


class ServiceStarting(Exception):
    """A speech or model service the adapter needs is still starting: the session is refused until it is ready."""

    def __init__(self, name: str):
        """Name the service that is still starting."""
        super().__init__(f"{name} is still starting")
        self.name = name


class AdapterSettings(Protocol):
    """A frontend adapter class's own settings: a frozen dataclass of its ``frontend.<name>`` block."""

    @classmethod
    def parse(cls, block: Mapping[str, Any], *, pick: Callable[..., Any], message_text: Callable[[dict], str]) -> Self:
        """Validate the block, picking what it names with ``pick``; unknown keys and wrong kinds are refused."""


class ServiceCatalog(Protocol):
    """The example's catalog, read once by the root: models by slot and services by slot."""

    def model(self, slot: str, overrides: Mapping[str, Any] | None = None) -> ModelEndpoint:
        """The model endpoint a slot names; an unlisted slot is an error."""

    def service(self, slot: str) -> dict:
        """The service entry a slot names (speech-to-text, text-to-speech)."""


class ServiceReadiness(Protocol):
    """Whether the services an adapter class needs are up."""

    async def ensure_ready(self) -> None:
        """Return once every needed service answers; raise :class:`ServiceStarting` for one that is still starting."""


class FrontendAdapterClass(Protocol):
    """A frontend adapter class, as ``FRONTEND_ADAPTERS`` holds it."""

    Settings: ClassVar[type[AdapterSettings]]

    @classmethod
    def from_config(
        cls,
        session: SessionConfig,
        transport: str,
        connection: MediaConnection | None,
        settings: AdapterSettings,
        catalog: ServiceCatalog,
    ) -> FrontendAdapter:
        """Build one session's adapter from its startup settings and the class's settings; connect nothing."""

    @classmethod
    def describe(cls, settings: AdapterSettings, catalog: ServiceCatalog) -> dict:
        """The adapter's half of ``GET /v1/live/info``, as plain JSON (no credentials)."""

    @classmethod
    def readiness(cls, settings: AdapterSettings, catalog: ServiceCatalog) -> ServiceReadiness:
        """What tells whether the class's services are ready."""
