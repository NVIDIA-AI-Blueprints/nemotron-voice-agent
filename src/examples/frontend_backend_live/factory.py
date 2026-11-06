# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The composition root of live sessions: the hooks that the registry's ``live:`` block names for this example.

* :func:`create_frontend_adapter` (``live.factory``) builds one session's frontend adapter, of the class that
  ``frontend.adapter`` names, once the class's services are ready, and a WebRTC session's peer connection
  (``adapter/connection.py``). It builds no delegation core, so a client-mode session never needs one.
* :func:`core` (``live.core``) gives the client of the process's one delegation core, built on first use.
* :func:`describe` (``live.describe``) describes the example for ``GET /v1/live/info``.

The root reads the example's settings (the ``config`` block of its ``examples_registry.yaml`` entry,
:func:`example_config`), the core's texts (the ``agent_prompts.Core`` entries of ``prompts.yaml``) and the example's
catalog once, off the event loop (:func:`root`), and resolves what the core names by reference (``backend.slot``, and
``backend.prompt``: an ``agent_prompts.Backend`` entry of ``prompts.yaml``). Nothing of the frontend goes into the core.
The repository UI bot (``pipeline.py``) builds a core of its own with these functions, from the core's settings with
``ui_bot.core`` merged over them, with the channel context after its backend prompt (:func:`ui_bot_instructions`). The
two roots are the only modules that import the core, and each hands the frontend the helpers the two share: the core's
one copy of ``pick`` and ``message_text``.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import cache, wraps
from typing import TYPE_CHECKING, Any

import examples_registry

# Starts OpenTelemetry when ENABLE_TRACING is on, as the UI bot's root does: the frontend adapter traces each session's
# pipeline but may not import the setup itself.
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

# The example's key in examples_registry.yaml, whose config block holds the example's settings.
EXAMPLE_KEY = "frontend-backend-live"

# The root's reads and the core's build happen once, also when several sessions start together: a first call waits
# for the build another thread has begun. Reentrant, because the core's build reads the root's state.
_BUILDING = threading.RLock()


def once[T](build: Callable[[], T]) -> Callable[[], T]:
    """``build``, run once: later calls return its result, and concurrent first calls wait for the one build.

    A build that raises is not kept, so the next call runs it again (a corrected file is read on the next use).
    ``cache_clear`` starts over.
    """
    cached = cache(build)

    @wraps(build)
    def built() -> T:
        with _BUILDING:
            return cached()

    built.cache_clear = cached.cache_clear
    return built


# ---------------------------------------------------------------------------------------------- the root's settings
@dataclass(frozen=True)
class Root:
    """What the root reads once: the example's settings, the core's and its texts, the adapter's class, the catalog."""

    config: ExampleConfig
    core: CoreSettings
    adapter: FrontendAdapterClass  # the class frontend.adapter names
    settings: AdapterSettings  # its own block, parsed by the class, with the core's display label handed in
    readiness: ServiceReadiness  # whether the class's services are up
    catalog: ExampleCatalog


def example_config() -> ExampleConfig:
    """The example's settings: the ``config`` block of its ``examples_registry.yaml`` entry, validated.

    Raises:
        ValueError: The block is malformed (a pydantic ``ValidationError`` names each key at fault).
    """
    return ExampleConfig.model_validate(examples_registry.example_config(EXAMPLE_KEY))


def core_settings(data: Mapping[str, Any]) -> CoreSettings:
    """The core's settings ``data``, with the core's texts from the example's prompt file.

    Raises:
        ValueError: The settings or a text is malformed.
        RuntimeError: The prompt file is missing or malformed (:func:`load_prompts`).
        KeyError: The prompt file lacks a group or an entry that the example reads (:func:`load_prompts`).
    """
    return CoreSettings.parse(data, load_prompts().core)


def ui_bot_core(config: ExampleConfig) -> CoreSettings:
    """The UI bot's core: the core's settings with ``ui_bot.core`` merged over them, on the llm adapter it names.

    Raises:
        ValueError: The merged settings are malformed.
    """
    return core_settings(config.ui_bot_core)


def frontend_adapters() -> Mapping[str, FrontendAdapterClass]:
    """The frontend adapters a deployment chooses by name: ``FRONTEND_ADAPTERS`` (``adapter/registry.py``)."""
    from examples.frontend_backend_live.adapter.registry import FRONTEND_ADAPTERS

    return FRONTEND_ADAPTERS


def webrtc_peer(ice_servers: list[dict]) -> MediaConnection:
    """A WebRTC session's peer connection, on the ICE servers the gateway hands over (``adapter/connection.py``)."""
    from examples.frontend_backend_live.adapter.connection import LiveWebRTCConnection

    return LiveWebRTCConnection.from_ice_servers(ice_servers)


def adapter_block(config: ExampleConfig) -> Mapping[str, Any]:
    """The block of the example's settings that the chosen frontend adapter reads: ``frontend.<adapter>``."""
    return config.frontend.block


def with_display_label(block: Mapping[str, Any], core: CoreSettings) -> dict[str, Any]:
    """``block`` with the core's display label handed in, so the label is written once (the core's settings)."""
    return {**block, "display_label": core.delegation.display_label}


@once
def root() -> Root:
    """Read the settings and the catalog, and settle the frontend adapter, once; it blocks, so callers use a thread.

    Raises:
        ValueError: The example's settings are malformed, ``frontend.adapter`` names no frontend adapter, or the
            adapter's class refuses its block.
        RuntimeError: The prompt file is missing or malformed.
        KeyError: The prompt file lacks a group or an entry that the example reads.
    """
    config = example_config()
    core = core_settings(config.core)
    catalog = ExampleCatalog.read()
    adapter = pick(frontend_adapters(), config.frontend.adapter, "frontend adapter")
    block = with_display_label(adapter_block(config), core)
    settings = adapter.Settings.parse(block, pick=pick, message_text=message_text)
    return Root(config, core, adapter, settings, adapter.readiness(settings, catalog), catalog)


# --------------------------------------------------------------------------------------------- the delegation core
def core_endpoint(endpoint: ModelEndpoint) -> CoreEndpoint:
    """A resolved endpoint as the delegation core takes it (without the frontend's own reply format)."""
    return CoreEndpoint(endpoint.model, endpoint.base_url, endpoint.api_key, dict(endpoint.extra_params))


def endpoint_lookup(catalog: ExampleCatalog, slot: str, overrides: dict | None = None) -> EndpointLookup:
    """The core's ``EndpointLookup`` for a backend slot.

    ``None`` (or the slot's own model) gives the slot's endpoint, with the session's ``overrides``; another model gives
    the catalog entry that serves it, and a model the catalog does not list raises ``LookupError``, which the backend
    reports.

    Raises:
        ValueError: At once, when the slot has no model.
    """
    default = catalog.model(slot, overrides)

    def endpoints(model: str | None) -> CoreEndpoint:
        return core_endpoint(default if model in (None, default.model) else catalog.named(slot, model))

    return endpoints


def backend_defaults(core: CoreSettings, endpoints: EndpointLookup, instructions: Callable[[str], str]) -> BackendTask:
    """What a request leaves out: the default model of ``backend.slot`` and the instructions of ``backend.prompt``.

    ``instructions`` gives the text of the backend prompt that ``backend.prompt`` names: the Live core's
    (:func:`live_backend_prompt`), or the UI bot's (:func:`ui_bot_instructions`).
    """
    return BackendTask(model=endpoints(None).model, instructions=instructions(core.backend.prompt), tools=[])


def live_backend_prompt(name: str) -> str:
    """The Live core's default backend instructions: the backend prompt ``name`` of the prompt file, as written.

    The Live core serves every Live client, whatever its tools, so the prompt is used as written, with no channel. The
    file is read once, when the core is built.
    """
    return load_prompts().text("Backend", name)


def ui_bot_instructions(prompts: Prompts, channel: str) -> Callable[[str], str]:
    """The UI bot's backend instructions: the backend prompt that ``backend.prompt`` names, then the channel context.

    ``prompts`` is the session's read of the prompt file, and ``channel`` the text of the channel context, which also
    follows the session's talker prompt.
    """
    return lambda name: f"{prompts.text('Backend', name)}\n\n{channel}"


def core_client(core: DelegationCore) -> LocalResponsesClient:
    """The in-process client of a core's official endpoints; the core answers the model check and settings check.

    It carries the core's close budget, which the gateway sizes its wait for a session's close from, and the core's
    close, which the gateway runs through the client at server shutdown, after every session.
    """
    service = ResponsesService(core.session, stateless_session=core.one_shot_session, backend=core)
    return LocalResponsesClient(service, close_seconds=core.close_seconds, close_core=core.aclose)


@once
def delegation_core() -> DelegationCore:
    """The one delegation core of this process, built on first use from the core's settings; it blocks.

    It serves every Live client, whatever its tools, so a request with no instructions gets the backend prompt that
    ``backend.prompt`` names as written, with no channel context.
    """
    state = root()
    endpoints = endpoint_lookup(state.catalog, state.core.backend.slot)
    defaults = backend_defaults(state.core, endpoints, live_backend_prompt)
    return build_core(state.core, endpoints=endpoints, defaults=defaults)


@once
def responses_client() -> LocalResponsesClient:
    """The client of the process's one delegation core; it blocks while the core is built."""
    return core_client(delegation_core())


# --------------------------------------------------------------------------------------------------- the hooks
async def create_frontend_adapter(
    config: SessionConfig, transport: str, ice_servers: list[dict] | None
) -> SessionParts:
    """``live.factory``: one session's parts, once the adapter's services are ready: its frontend adapter and peer.

    A WebRTC session's peer connection is made on the ``ice_servers`` that the gateway hands over (:func:`webrtc_peer`),
    and the adapter gets it; a WebSocket session's ``ice_servers`` are ``None``, and it has no peer.

    The adapter is built by its class (``from_config``) and connects nothing until the session starts it. No delegation
    core is built here: the gateway reaches the one core through :func:`core`, and a client-mode session never does.

    Raises:
        ProtocolError: A service the adapter needs is still starting (code ``server_error``, so the client is not
            blamed and can try again).
        ValueError: The root's settings are malformed (the gateway answers that the pipeline could not be created).
    """
    state = await asyncio.to_thread(root)
    try:
        await state.readiness.ensure_ready()
    except ServiceStarting as exc:
        raise ProtocolError(f"{exc}. Try again in a moment.", "server_error") from exc
    connection = None if ice_servers is None else webrtc_peer(ice_servers)
    try:
        adapter = state.adapter.from_config(config, transport, connection, state.settings, state.catalog)
    except BaseException:
        if connection is not None:  # the peer was never negotiated: closing it releases what it holds
            await connection.close_after_drain()
        raise
    return SessionParts(frontend=adapter, connection=connection)


async def core() -> LocalResponsesClient:
    """``live.core``: the client of the process's one delegation core, built on first use, off the event loop.

    Every ``responses`` session and every delegate round reaches the same core; a client-mode session never asks for
    it. A bad backend block fails the first use that needs the core, not the client-mode sessions before it.
    """
    return await asyncio.to_thread(responses_client)


async def describe() -> dict:
    """``live.describe``: the example, as plain JSON (no credentials)."""
    return await asyncio.to_thread(_describe)


def _describe() -> dict:
    """The adapter's half, the core's half and the backend's models."""
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
