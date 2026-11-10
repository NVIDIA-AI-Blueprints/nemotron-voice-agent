# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What a frontend adapter class offers the composition roots, which build each session's adapter with it."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, Self

if TYPE_CHECKING:
    from examples.frontend_backend_live.config_manager.schema import ModelEndpoint
    from gateways.live.ports import FrontendAdapter, MediaConnection
    from gateways.live.protocol import SessionConfig


class ServiceStarting(Exception):
    """Raised while a service the adapter needs is still starting: the session is refused until it is ready."""

    def __init__(self, name: str):
        """Name the service that is still starting."""
        super().__init__(f"{name} is still starting")
        self.name = name


class AdapterSettings(Protocol):
    """A frontend adapter class's own settings: a frozen dataclass of its ``frontend.<name>`` block."""

    @classmethod
    def parse(cls, block: Mapping[str, Any], *, pick: Callable[..., Any], message_text: Callable[[dict], str]) -> Self:
        """Validate ``block``, refusing unknown keys and wrong kinds.

        ``pick`` reads a name table and ``message_text`` a message item's text: the delegation core's copies, which
        only the composition roots import.
        """


class ServiceCatalog(Protocol):
    """The example's catalog, read once by the root: models and services by slot."""

    def model(self, slot: str, overrides: Mapping[str, Any] | None = None) -> ModelEndpoint:
        """Return the model endpoint of ``slot`` with ``overrides``; an unlisted slot raises ``ValueError``."""

    def service(self, slot: str) -> dict:
        """Return the service entry of ``slot`` (speech-to-text or text-to-speech)."""


class ServiceReadiness(Protocol):
    """The readiness check of the services an adapter class needs."""

    async def ensure_ready(self) -> None:
        """Return once every needed service answers; raise :class:`ServiceStarting` for one still starting."""


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
        """Build one session's adapter from its startup settings and the class's settings; connect nothing.

        The root calls it off the event loop once :meth:`readiness` passed, so it may read a service's catalog; the
        adapter's ``start()`` connects.
        """

    @classmethod
    def describe(cls, settings: AdapterSettings, catalog: ServiceCatalog) -> dict:
        """Return the adapter's half of ``GET /v1/live/info`` as plain JSON, with no credentials."""

    @classmethod
    def readiness(cls, settings: AdapterSettings, catalog: ServiceCatalog) -> ServiceReadiness:
        """Return the readiness check of the class's services."""
