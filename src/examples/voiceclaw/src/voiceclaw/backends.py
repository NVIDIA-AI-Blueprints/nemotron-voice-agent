# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Stable composition boundary for installed backend adapters.

Installed integrations register one :class:`BackendAdapterRegistration` in
the ``voiceclaw.backends`` Python entry-point group. The registration is
limited to identity, API compatibility, and a factory. Model instructions and
tool descriptions remain in VoiceClaw's trusted model-contract catalog.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import KW_ONLY, dataclass
from importlib import metadata as importlib_metadata

from voiceclaw.application.interaction import InteractionCoordinator
from voiceclaw.application.routing import build_turn_routing_policy
from voiceclaw.application.runtime import RealtimeInteractionManager
from voiceclaw.config import BackendProfile, ConfigurationError
from voiceclaw.domain.capabilities import CapabilityToolRegistry
from voiceclaw.interaction_profiles import INTERACTION_PROFILE_SCHEMA, InteractionProfile, SessionScope
from voiceclaw.model_contracts import ModelContractCatalog
from voiceclaw.ports.interaction import AgentInteractionPort
from voiceclaw.ports.readiness import (
    SelectedAgentReadinessCode,
    SelectedAgentReadinessError,
    SelectedAgentReadinessPort,
)
from voiceclaw.ports.runtime import RealtimeSessionRuntimePort
from voiceclaw.ports.state import StateStore
from voiceclaw.ports.turns import EphemeralCommittedTurnPort

BACKEND_ENTRY_POINT_GROUP = "voiceclaw.backends"
BACKEND_PLUGIN_API_VERSION = "1"

_BACKEND_NAME = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
_MAX_BACKEND_NAME_CHARACTERS = 64


class BackendPluginError(ValueError):
    """An installed backend registration is ambiguous or incompatible."""


class DurableRuntimeUnavailableError(ConfigurationError):
    """A durable backend was selected before its core Realtime runtime exists."""


def validate_backend_name(value: object, *, field: str = "backend adapter name") -> str:
    """Return one canonical registry name or reject an unsafe identifier."""
    if (
        not isinstance(value, str)
        or len(value) > _MAX_BACKEND_NAME_CHARACTERS
        or _BACKEND_NAME.fullmatch(value) is None
    ):
        raise BackendPluginError(
            f"{field} must be 1..{_MAX_BACKEND_NAME_CHARACTERS} lowercase letters, digits, or single underscores"
        )
    return value


@dataclass(frozen=True, slots=True)
class BackendComposition:
    """Normalized ports selected for one configured backend profile."""

    turn_backend: EphemeralCommittedTurnPort | None
    turn_status: str
    _: KW_ONLY
    agent_backend: AgentInteractionPort | None = None
    selected_agent_readiness: SelectedAgentReadinessPort | None = None

    def __post_init__(self) -> None:
        """Reject malformed adapter results before they reach the server."""
        validate_backend_name(self.turn_status, field="backend turn status")
        if self.turn_backend is not None and self.agent_backend is not None:
            raise BackendPluginError("a backend composition cannot mix ephemeral and durable interaction ports")
        if self.turn_status == "disabled" and (self.turn_backend is not None or self.agent_backend is not None):
            raise BackendPluginError("a disabled backend composition cannot expose an interaction port")
        if self.turn_status != "disabled" and self.turn_backend is None and self.agent_backend is None:
            raise BackendPluginError("an active backend composition must expose an interaction port")
        if self.selected_agent_readiness is not None and self.turn_backend is None and self.agent_backend is None:
            raise BackendPluginError("selected-agent readiness requires an active interaction port")
        if self.agent_backend is not None and not isinstance(self.agent_backend, AgentInteractionPort):
            raise TypeError("agent_backend must implement AgentInteractionPort")
        if self.selected_agent_readiness is not None and not isinstance(
            self.selected_agent_readiness,
            SelectedAgentReadinessPort,
        ):
            raise TypeError("selected_agent_readiness must implement SelectedAgentReadinessPort")

    async def check_selected_agent_readiness(self) -> None:
        """Verify target-specific access or fail closed when no attestor exists."""
        if self.selected_agent_readiness is None:
            raise SelectedAgentReadinessError(SelectedAgentReadinessCode.READINESS_UNSUPPORTED)
        await self.selected_agent_readiness.check_selected_agent()

    def create_interaction_coordinator(
        self,
        *,
        state_store: StateStore,
        model_contracts: ModelContractCatalog,
        interaction_profile: InteractionProfile,
    ) -> InteractionCoordinator:
        """Build the core-owned durable coordinator around the selected port.

        Backend plugins supply protocol translation only. VoiceClaw owns the
        capability registry, command admission, persistence, and recovery
        coordinator that sit above that port.
        """
        if self.agent_backend is None:
            raise ValueError("backend composition does not provide an AgentInteractionPort")
        return InteractionCoordinator(
            backend=self.agent_backend,
            state_store=state_store,
            tools=CapabilityToolRegistry(profile=interaction_profile),
        )

    def require_realtime_runtime(self) -> None:
        """Fail closed when the selected port has no core-owned facade runtime."""
        if self.agent_backend is not None:
            raise DurableRuntimeUnavailableError(
                "durable AgentInteractionPort backends do not yet have a core Realtime session runtime"
            )

    def validate_interaction_profile(self, interaction_profile: InteractionProfile) -> None:
        """Reject a topology whose authoritative state the selected port cannot supply."""
        if self.turn_backend is not None and interaction_profile.session_scope is not SessionScope.NONE:
            raise ConfigurationError(
                "response-only backends require a sessionless interaction profile; "
                "session-scoped profiles require an authoritative Agent Session runtime"
            )

    def create_runtime(
        self,
        *,
        backend_profile: str,
        state_store: StateStore,
        turn_routing_mode: str = "model",
        request_summary_character_limit: int = 512,
        retained_request_limit: int = 8,
        model_contracts: ModelContractCatalog | None = None,
        interaction_profile: InteractionProfile | None = None,
        interaction_profile_schema: str = INTERACTION_PROFILE_SCHEMA,
        interaction_profile_hash: str | None = None,
    ) -> RealtimeSessionRuntimePort:
        """Build the facade-facing runtime without exposing adapter types."""
        self.require_realtime_runtime()
        if interaction_profile is not None:
            self.validate_interaction_profile(interaction_profile)
        return RealtimeInteractionManager(
            backend_profile=backend_profile,
            state_store=state_store,
            committed_turns=self.turn_backend,
            turn_routing_policy=build_turn_routing_policy(turn_routing_mode),
            request_summary_character_limit=request_summary_character_limit,
            retained_request_limit=retained_request_limit,
            model_contracts=model_contracts,
            interaction_profile=interaction_profile,
            interaction_profile_schema=interaction_profile_schema,
            interaction_profile_hash=interaction_profile_hash,
        )


type InstalledBackendFactory = Callable[
    [BackendProfile, Mapping[str, str], ModelContractCatalog],
    BackendComposition,
]


@dataclass(frozen=True, slots=True)
class BackendAdapterRegistration:
    """Versioned installed-adapter registration loaded from an entry point.

    The factory receives VoiceClaw's validated profile, read-only environment
    view, and trusted model-contract catalog. The registration cannot
    contribute prompt or tool prose.
    """

    name: str
    api_version: str
    factory: InstalledBackendFactory

    def __post_init__(self) -> None:
        """Validate the inert registration before the registry consumes it."""
        validate_backend_name(self.name)
        if not isinstance(self.api_version, str) or not self.api_version:
            raise BackendPluginError("backend adapter api_version must be non-empty text")
        if not callable(self.factory):
            raise BackendPluginError("backend adapter factory must be callable")


def discover_backend_registration(
    selected_name: str,
    *,
    reserved_names: frozenset[str] = frozenset(),
) -> BackendAdapterRegistration | None:
    """Load only the configured installed adapter, failing closed on ambiguity.

    Entry-point metadata can enumerate a whole group, but only an entry whose
    name exactly matches ``selected_name`` is imported. Duplicate entries and
    attempts to shadow a trusted registry entry fail before plugin code loads.
    """
    selected_name = validate_backend_name(selected_name)
    discovered = importlib_metadata.entry_points(group=BACKEND_ENTRY_POINT_GROUP)
    matches = tuple(
        entry_point
        for entry_point in discovered
        if entry_point.name == selected_name and entry_point.group == BACKEND_ENTRY_POINT_GROUP
    )
    if len(matches) > 1:
        raise BackendPluginError(f"multiple installed backend adapters register {selected_name!r}")
    if not matches:
        return None
    if selected_name in reserved_names:
        raise BackendPluginError(f"installed backend adapter {selected_name!r} shadows a trusted registry entry")

    try:
        loaded = matches[0].load()
    except Exception as error:
        raise BackendPluginError(f"installed backend adapter {selected_name!r} could not be loaded") from error
    if type(loaded) is not BackendAdapterRegistration:
        raise BackendPluginError(f"installed backend adapter {selected_name!r} must export BackendAdapterRegistration")
    if loaded.name != selected_name:
        raise BackendPluginError(
            f"installed backend adapter entry point {selected_name!r} exported registration {loaded.name!r}"
        )
    if loaded.api_version != BACKEND_PLUGIN_API_VERSION:
        raise BackendPluginError(
            f"installed backend adapter {selected_name!r} uses unsupported API version {loaded.api_version!r}"
        )
    return loaded


__all__ = [
    "BACKEND_ENTRY_POINT_GROUP",
    "BACKEND_PLUGIN_API_VERSION",
    "BackendAdapterRegistration",
    "BackendComposition",
    "BackendPluginError",
    "DurableRuntimeUnavailableError",
    "InstalledBackendFactory",
    "discover_backend_registration",
    "validate_backend_name",
]
