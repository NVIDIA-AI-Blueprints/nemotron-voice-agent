# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Trusted domain-plugin contract for the shared Frontend/Backend Agent pipeline."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Literal, Protocol

from pipecat.adapters.schemas.tools_schema import ToolsSchema

from examples.frontend_backend_agent.src.tools import ToolSpec


class DomainBackend(Protocol):
    """Runtime boundary consumed by the domain-neutral Talker handlers."""

    async def call(self, query: str, slots: dict[str, Any] | None = None, *, on_started=None) -> dict[str, Any]:
        """Plan and execute one delegated request."""

    def cancel_active(self, reason: str = "new_user_query") -> bool:
        """Cancel an active delegated request."""

    def cancel_pending_work(self) -> bool:
        """Clear domain state that remains pending without an active request."""


@dataclass(slots=True, frozen=True)
class DomainBuildContext:
    """Session-scoped dependencies supplied to a trusted domain backend factory."""

    thinker_llm: Any
    thinker_prompt: str
    thinker_max_tokens: int
    tool_names: tuple[str, ...]
    tool_delay_seconds: float
    tool_delay_min_seconds: float
    load_service_entry: Callable[[str, str], dict]


FillerPolicy = Literal["code_authored", "planner_authored"]


@dataclass(slots=True, frozen=True)
class DomainSpec:
    """All domain-specific policy injected into the reusable voice pipeline."""

    key: str
    label: str
    thinker_prompt_key: str
    talker_tools_schema: ToolsSchema
    build_backend: Callable[[DomainBuildContext], DomainBackend]
    runtime_context: Callable[[], str]
    intro_prompt: str = "Please greet the user briefly."
    tts_text_transform: Callable[[str], str] | None = None
    filler_selector: Callable[[str], str] | None = None
    filler_policy: FillerPolicy = "code_authored"
    tool_registry: Mapping[str, ToolSpec] = field(default_factory=dict)
    max_query_chars: int = 4000


# This allowlist is the code-level trust boundary. ``domain_profile`` may arrive
# in a session body, but it can only resolve to one of these repository-owned
# modules. Adding a future domain does not require editing the shared pipeline.
_DOMAIN_FACTORIES: dict[str, str] = {
    "airline": "examples.frontend_backend_agent.airline.domain:create_domain_spec",
    "generic": "examples.frontend_backend_agent.generic.domain:create_domain_spec",
}


def available_domain_profiles() -> tuple[str, ...]:
    """Return the stable, allowlisted domain keys."""
    return tuple(_DOMAIN_FACTORIES)


def resolve_domain_spec(profile: object) -> DomainSpec:
    """Load one repository-owned domain spec, defaulting to the legacy airline behavior."""
    key = str(profile or "airline").strip().lower()
    target = _DOMAIN_FACTORIES.get(key)
    if target is None:
        raise ValueError(f"Unknown Frontend/Backend Agent domain profile: {key!r}")
    module_name, factory_name = target.split(":", 1)
    factory = getattr(import_module(module_name), factory_name, None)
    if not callable(factory):
        raise RuntimeError(f"Invalid Frontend/Backend Agent domain factory: {target}")
    spec = factory()
    if not isinstance(spec, DomainSpec) or spec.key != key:
        raise RuntimeError(f"Domain factory {target} returned an invalid specification")
    return spec
