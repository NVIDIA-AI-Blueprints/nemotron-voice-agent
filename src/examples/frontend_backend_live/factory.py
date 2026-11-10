# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The composition root of live sessions: the hooks that the registry's ``live:`` block names for this example."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import cache, wraps
from typing import TYPE_CHECKING, Any

import examples_registry

# Imported for its setup: it loads .env and, with ENABLE_TRACING on, starts OpenTelemetry.
import tracing  # noqa: F401
from examples.frontend_backend_live.adapter.contract import ServiceStarting
from examples.frontend_backend_live.config_manager.schema import ExampleConfig, ModelEndpoint
from examples.frontend_backend_live.delegation_core.api.local import LocalResponsesClient
from examples.frontend_backend_live.delegation_core.api.service import ResponsesService
from examples.frontend_backend_live.delegation_core.app import DelegationCore, build_core
from examples.frontend_backend_live.delegation_core.backend.port import EndpointLookup
from examples.frontend_backend_live.delegation_core.config import CoreSettings
from examples.frontend_backend_live.delegation_core.im.input import BackendTask
from examples.frontend_backend_live.delegation_core.items import message_text
from examples.frontend_backend_live.delegation_core.tables import pick
from examples.frontend_backend_live.models.endpoints import ExampleCatalog
from examples.frontend_backend_live.prompts import Prompts, load_prompts
from examples.shared.delegation_contract import ModelEndpoint as CoreEndpoint
from gateways.live.ports import SessionParts
from gateways.live.protocol import ProtocolError, SessionConfig

if TYPE_CHECKING:
    from examples.frontend_backend_live.adapter.contract import (
        AdapterSettings,
        FrontendAdapterClass,
        ServiceReadiness,
    )
    from gateways.live.ports import MediaConnection

EXAMPLE_KEY = "frontend-backend-live"

# Reentrant: the core's build reads the root's state.
_BUILDING = threading.RLock()


def once[T](build: Callable[[], T]) -> Callable[[], T]:
    """Return ``build`` run once: later calls get its result, and concurrent first calls wait for the one build.

    A build that raises is not kept, so the next call runs it again; ``cache_clear`` starts over.
    """
    cached = cache(build)

    @wraps(build)
    def built() -> T:
        with _BUILDING:
            return cached()

    built.cache_clear = cached.cache_clear
    return built


@dataclass(frozen=True)
class Root:
    """What the root reads once: the example's settings, the core's, the adapter's class and settings, the catalog."""

    config: ExampleConfig
    core: CoreSettings
    adapter: FrontendAdapterClass
    settings: AdapterSettings
    readiness: ServiceReadiness
    catalog: ExampleCatalog


def example_config() -> ExampleConfig:
    """Return the example's settings: the validated ``config`` block of its ``examples_registry.yaml`` entry."""
    return ExampleConfig.model_validate(examples_registry.example_config(EXAMPLE_KEY))


def core_settings(data: Mapping[str, Any]) -> CoreSettings:
    """Return the core's settings from ``data``, with the core's texts from the example's prompt file."""
    return CoreSettings.parse(data, load_prompts().core)


def ui_bot_core(config: ExampleConfig) -> CoreSettings:
    """Return the UI bot's core settings: the core's, with ``ui_bot.core`` merged over them."""
    return core_settings(config.ui_bot_core)


def frontend_adapters() -> Mapping[str, FrontendAdapterClass]:
    """Return the frontend adapters a deployment chooses by name (``adapter/registry.py``)."""
    from examples.frontend_backend_live.adapter.registry import FRONTEND_ADAPTERS

    return FRONTEND_ADAPTERS


def webrtc_peer(ice_servers: list[dict]) -> MediaConnection:
    """Return a WebRTC session's peer connection on the ICE servers the gateway hands over."""
    from examples.frontend_backend_live.adapter.connection import LivePeer

    return LivePeer.from_ice_servers(ice_servers)


def adapter_block(config: ExampleConfig) -> Mapping[str, Any]:
    """Return the block the chosen frontend adapter reads: ``frontend.<adapter>``."""
    return config.frontend.block


def with_display_label(block: Mapping[str, Any], core: CoreSettings) -> dict[str, Any]:
    """Return ``block`` with the core's display label and app-output deadline, so each is written once."""
    delegation = core.delegation
    return {
        **block,
        "display_label": delegation.display_label,
        "client_tool_timeout_seconds": delegation.client_tool_timeout_seconds,
    }


@once
def root() -> Root:
    """Read the settings and the catalog and settle the frontend adapter, once; it blocks, so callers use a thread."""
    config = example_config()
    core = core_settings(config.core)
    catalog = ExampleCatalog.read()
    adapter = pick(frontend_adapters(), config.frontend.adapter, "frontend adapter")
    block = with_display_label(adapter_block(config), core)
    settings = adapter.Settings.parse(block, pick=pick, message_text=message_text)
    return Root(config, core, adapter, settings, adapter.readiness(settings, catalog), catalog)


def core_endpoint(endpoint: ModelEndpoint) -> CoreEndpoint:
    """Return a resolved endpoint as the delegation core takes it, without the frontend's own fields."""
    return CoreEndpoint(endpoint.model, endpoint.base_url, endpoint.api_key, dict(endpoint.extra_params), endpoint.api)


def endpoint_lookup(catalog: ExampleCatalog, slot: str, overrides: dict | None = None) -> EndpointLookup:
    """Return the core's ``EndpointLookup`` for a backend slot.

    ``None`` or the slot's model gives the slot's endpoint with ``overrides``; another model gives the catalog entry
    that serves it, and an unlisted one raises ``LookupError``, which the backend reports.
    """
    default = catalog.model(slot, overrides)

    def endpoints(model: str | None) -> CoreEndpoint:
        return core_endpoint(default if model in (None, default.model) else catalog.named(slot, model))

    return endpoints


def backend_defaults(core: CoreSettings, endpoints: EndpointLookup, instructions: Callable[[str], str]) -> BackendTask:
    """Return what a request leaves out: the default model of ``backend.slot`` and the ``backend.prompt`` text."""
    return BackendTask(model=endpoints(None).model, instructions=instructions(core.backend.prompt), tools=[])


def live_backend_prompt(name: str) -> str:
    """Return the backend prompt ``name`` as written: the Live core serves every client, so it adds no channel."""
    return load_prompts().text("Backend", name)


def ui_bot_instructions(prompts: Prompts, channel: str) -> Callable[[str], str]:
    """Return the UI bot's backend instructions: the backend prompt ``name``, then the channel context."""
    return lambda name: f"{prompts.text('Backend', name)}\n\n{channel}"


def core_client(core: DelegationCore) -> LocalResponsesClient:
    """Return the in-process client of a core's endpoints, carrying the core's close budget and its close."""
    service = ResponsesService(core.session, stateless_session=core.one_shot_session, backend=core)
    return LocalResponsesClient(service, close_seconds=core.close_seconds, close_core=core.aclose)


@once
def delegation_core() -> DelegationCore:
    """Build the process's one delegation core on first use, from the core's settings; it blocks."""
    state = root()
    endpoints = endpoint_lookup(state.catalog, state.core.backend.slot)
    defaults = backend_defaults(state.core, endpoints, live_backend_prompt)
    return build_core(state.core, endpoints=endpoints, defaults=defaults)


@once
def responses_client() -> LocalResponsesClient:
    """Return the client of the process's one delegation core; it blocks while the core is built."""
    return core_client(delegation_core())


async def create_frontend_adapter(
    config: SessionConfig, transport: str, ice_servers: list[dict] | None
) -> SessionParts:
    """Build one session's parts once its services are ready: its frontend adapter and WebRTC peer (``live.factory``).

    A service still starting raises ``ProtocolError`` with code ``server_error``, so the client can try again.
    """
    state = await asyncio.to_thread(root)
    try:
        await state.readiness.ensure_ready()
    except ServiceStarting as exc:
        raise ProtocolError(f"{exc}. Try again in a moment.", "server_error") from exc
    connection = None if ice_servers is None else webrtc_peer(ice_servers)
    try:
        adapter = await asyncio.to_thread(
            state.adapter.from_config, config, transport, connection, state.settings, state.catalog
        )
    except BaseException:
        if connection is not None:
            await connection.close_after_drain()
        raise
    return SessionParts(frontend=adapter, connection=connection)


async def core() -> LocalResponsesClient:
    """Return the client of the one delegation core, built on first use off the event loop (``live.core``)."""
    return await asyncio.to_thread(responses_client)


async def describe() -> dict:
    """Describe the example as plain JSON, with no credentials (``live.describe``)."""
    return await asyncio.to_thread(_describe)


def _describe() -> dict:
    """Return the adapter's half, the core's half and the backend's models."""
    state, built = root(), delegation_core()
    backend = state.core.backend
    default = endpoint_lookup(state.catalog, backend.slot)(None)
    frontend = state.adapter.describe(state.settings, state.catalog)
    return {
        **frontend,
        "warnings": list(frontend.get("warnings", [])),
        "backend_models": list(dict.fromkeys([default.model, *state.catalog.models(backend.slot)])),
        "backend": {
            **built.describe(),
            "slot": backend.slot,
            "model": default.model,
            "base_url": default.base_url,
            "prompt": backend.prompt,
        },
    }
