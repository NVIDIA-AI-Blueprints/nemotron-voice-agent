# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The composition root of live sessions: the hooks that the registry's ``live:`` block names for this example.

* :func:`create_frontend_adapter` (``live.factory``) builds one session's frontend adapter, of the class that
  ``frontend.adapter`` names, once the class's services are ready. It builds no delegation core, so a client-mode
  session never needs one.
* :func:`core` (``live.core``) gives the client of the process's one delegation core, built on first use.
* :func:`describe` (``live.describe``) describes the example for ``GET /v1/live/info``.

The root reads ``config.yaml`` (``LIVE_CONFIG_PATH``), the core's ``core.yaml`` (``CORE_CONFIG_PATH``) and the
example's catalog once, off the event loop (:func:`root`), and resolves what the core names by reference
(``backend.slot``, ``backend.prompt``). Nothing of the frontend goes into the core. The repository UI bot
(``pipeline.py``) builds a core of its own with these functions, from ``ui_bot.core_config``.
"""

from __future__ import annotations

import asyncio
import os
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import cache, wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Starts OpenTelemetry when ENABLE_TRACING is on, as the UI bot's root does: the frontend adapter traces each session's
# pipeline but may not import the setup itself.
import tracing  # noqa: F401
from delegation_core.api.local import LocalResponsesClient
from delegation_core.api.service import ResponsesService
from delegation_core.app import DelegationCore, build_core, tool_sources
from delegation_core.backend.port import EndpointLookup
from delegation_core.config import CoreSettings
from delegation_core.config import ModelEndpoint as CoreEndpoint
from delegation_core.im.input import BackendTask
from examples.frontend_backend_live.adapter.contract import ServiceStarting
from examples.frontend_backend_live.config_manager.loader import EXAMPLE_DIR, load_live_config
from examples.frontend_backend_live.config_manager.schema import ExampleConfig, ModelEndpoint
from examples.frontend_backend_live.models.endpoints import ExampleCatalog
from examples.frontend_backend_live.prompts.catalog import CATALOG_PATH, catalog_line, catalog_prompt, role_instructions
from examples.frontend_backend_live.tables import pick
from live.ports import SessionParts
from live.protocol import ProtocolError, SessionConfig
from utils import default_prompt_key, load_yaml_file

if TYPE_CHECKING:
    from examples.frontend_backend_live.adapter.contract import (
        AdapterSettings,
        FrontendAdapterClass,
        ServiceReadiness,
    )

# The delegation core's own file, beside config.yaml; CORE_CONFIG_PATH names another.
CORE_CONFIG = EXAMPLE_DIR / "core.yaml"

# The catalog's lines (prompts.yaml ``lines``) that the console sends: in client mode the silent note that work started
# (``app_started``) and a failed delegation's line (``apology``, ``unclear``), the progress line of a slow tool batch
# (``still_working``), and its append presets (``intro``, ``still_working``).
CONSOLE_LINES = ("apology", "unclear", "still_working", "intro", "app_started")
# The catalog's block of the console's other append presets (prompts.yaml ``console_presets``): by append type, a list
# of [label, text]. It has no content, so the prompt list leaves it out.
CONSOLE_PRESETS = "console_presets"

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


# ------------------------------------------------------------------------------------------------- the root's files
@dataclass(frozen=True)
class Root:
    """What the root reads once: the example's settings, the core's, the frontend adapter's class and the catalog."""

    config: ExampleConfig
    core: CoreSettings
    adapter: FrontendAdapterClass  # the class frontend.adapter names
    settings: AdapterSettings  # its own block, parsed by the class, with the core's display label handed in
    readiness: ServiceReadiness  # whether the class's services are up
    catalog: ExampleCatalog


def core_config_path() -> Path:
    """The delegation core's own file: ``CORE_CONFIG_PATH``, or the bundled ``core.yaml``."""
    return Path(os.environ.get("CORE_CONFIG_PATH") or CORE_CONFIG)


def ui_bot_core(config: ExampleConfig) -> CoreSettings:
    """The UI bot's core profile: the file ``ui_bot.core_config`` names, in the example's folder.

    Raises:
        ValueError: The profile is malformed.
    """
    return CoreSettings.load(EXAMPLE_DIR / config.ui_bot.core_config)


def frontend_adapters() -> Mapping[str, FrontendAdapterClass]:
    """The frontend adapters a deployment chooses by name: ``FRONTEND_ADAPTERS`` (``adapter/registry.py``)."""
    from examples.frontend_backend_live.adapter.registry import FRONTEND_ADAPTERS

    return FRONTEND_ADAPTERS


def adapter_block(config: ExampleConfig) -> Mapping[str, Any]:
    """The block of ``config.yaml`` that the chosen frontend adapter reads: ``frontend.<adapter>``."""
    return config.blocks[config.adapter]


def with_display_label(block: Mapping[str, Any], core: CoreSettings) -> dict[str, Any]:
    """``block`` with the core's display label handed in, so the label is written once (``core.yaml``)."""
    return {**block, "display_label": core.delegation.display_label}


@once
def root() -> Root:
    """Read the root's files and the catalog, and settle the frontend adapter, once; it blocks, so callers use a thread.

    Raises:
        ValueError: A file is malformed (``ConfigError`` for ``config.yaml``), ``frontend.adapter`` names no frontend
            adapter, or the adapter's class refuses its block.
    """
    config = load_live_config()
    core = CoreSettings.load(core_config_path())
    catalog = ExampleCatalog.read()
    adapter = pick(frontend_adapters(), config.adapter, "frontend adapter")
    settings = adapter.Settings.parse(with_display_label(adapter_block(config), core))
    return Root(config, core, adapter, settings, adapter.readiness(settings, catalog), catalog)


# --------------------------------------------------------------------------------------------- the delegation core
def core_endpoint(endpoint: ModelEndpoint) -> CoreEndpoint:
    """A resolved endpoint as the delegation core takes it (without the frontend's own reply format)."""
    return CoreEndpoint(endpoint.model, endpoint.base_url, endpoint.api_key, dict(endpoint.extra_params))


def endpoint_lookup(catalog: ExampleCatalog, slot: str, overrides: dict | None = None) -> EndpointLookup:
    """The core's ``EndpointLookup`` for a backend slot.

    ``None`` (or the slot's own model) gives the slot's endpoint, with the session's ``overrides``; another model gives
    the catalog entry that serves it, and a model the catalog does not list raises ``LookupError`` (the backend reports
    it, or falls back to the default endpoint where its settings say so).

    Raises:
        ValueError: At once, when the slot has no model.
    """
    default = catalog.model(slot, overrides)

    def endpoints(model: str | None) -> CoreEndpoint:
        return core_endpoint(default if model in (None, default.model) else catalog.named(slot, model))

    return endpoints


def backend_defaults(
    core: CoreSettings, endpoints: EndpointLookup, instructions: Callable[[str], str] = role_instructions
) -> BackendTask:
    """What a request leaves out: the default model of ``backend.slot`` and the instructions of ``backend.prompt``.

    ``instructions`` reads the catalog entry that ``backend.prompt`` names: by default the entry and then the channel
    context, as the UI bot's backend starts.
    """
    return BackendTask(model=endpoints(None).model, instructions=instructions(core.backend.prompt), tools=[])


def core_client(core: DelegationCore) -> LocalResponsesClient:
    """The in-process client of a core's official endpoints; the core answers the model check and settings check.

    It carries the core's close budget, which the gateway sizes its wait for a session's close from, and the core's
    close, which the gateway runs through the client at server shutdown, after every session.
    """
    service = ResponsesService(core.session, stateless_session=core.one_shot_session, backend=core)
    return LocalResponsesClient(service, close_seconds=core.close_seconds, close_core=core.aclose)


@once
def delegation_core() -> DelegationCore:
    """The one delegation core of this process, built on first use from the core's own file; it blocks.

    It serves every Live client, whatever its tools, so a request with no instructions gets the entry that
    ``backend.prompt`` names as written, with no channel context.
    """
    state = root()
    endpoints = endpoint_lookup(state.catalog, state.core.backend.slot)
    defaults = backend_defaults(state.core, endpoints, catalog_prompt)
    return build_core(state.core, endpoints=endpoints, defaults=defaults)


@once
def responses_client() -> LocalResponsesClient:
    """The client of the process's one delegation core; it blocks while the core is built."""
    return core_client(delegation_core())


# --------------------------------------------------------------------------------------------------- the hooks
async def create_frontend_adapter(config: SessionConfig, transport: str, connection) -> SessionParts:
    """``live.factory``: one session's parts, its frontend adapter, once the adapter's services are ready.

    The adapter is built by its class (``from_config``) and connects nothing until the session starts it. No delegation
    core is built here: the gateway reaches the one core through :func:`core`, and a client-mode session never does.

    Raises:
        ProtocolError: A service the adapter needs is still starting (code ``server_error``, so the client is not
            blamed and can try again).
        ValueError: A root file is malformed (the gateway answers that the pipeline could not be created).
    """
    state = await asyncio.to_thread(root)
    try:
        await state.readiness.ensure_ready()
    except ServiceStarting as exc:
        raise ProtocolError(f"{exc}. Try again in a moment.", "server_error") from exc
    adapter = state.adapter.from_config(config, transport, connection, state.settings, state.catalog)
    return SessionParts(frontend=adapter)


async def core() -> LocalResponsesClient:
    """``live.core``: the client of the process's one delegation core, built on first use, off the event loop.

    Every ``responses`` session and every console round reaches the same core; a client-mode session never asks for
    it. A bad backend block fails the first use that needs the core, not the client-mode sessions before it.
    """
    return await asyncio.to_thread(responses_client)


async def describe() -> dict:
    """``live.describe``: the example for the live console, as plain JSON (no credentials)."""
    return await asyncio.to_thread(_describe)


def _describe() -> dict:
    """The adapter's half, the core's half, the backend's models, and the console's sample.

    The sample is the default talker, the UI bot's backend and its tools, and the lines and presets the console sends.
    """
    state, built, sample = root(), delegation_core(), console_sample()
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
        "instructions": {"frontend": sample.talker, "backend": sample.instructions},
        "sample_tools": sample.tools,
        "lines": sample.lines,
        "presets": sample.presets,
    }


@dataclass(frozen=True)
class ConsoleSample:
    """What the console starts with: the default talker, the backend of ``ui_bot.core_config``, lines and presets."""

    instructions: str  # the backend instructions of the UI bot's profile, as its backend starts
    tools: list[dict]  # the tools of the tool sources that the profile names, built by name
    lines: dict[str, str]  # the catalog's lines that the console sends, by name (CONSOLE_LINES)
    presets: dict[str, list[list[str]]]  # the catalog's other append presets, by append type (CONSOLE_PRESETS)
    talker: str  # the example catalog's default talker prompt, then the channel context


@once
def console_sample() -> ConsoleSample:
    """The console's sample: the catalog's default talker, the UI bot's backend, and the catalog's lines and presets.

    The console runs in the browser, with its own copy of the tools, the demo that the UI bot's profile
    (``ui_bot.core_config``) names, so it starts with the example catalog's default talker prompt (the one that
    ``utils.resolve_prompt`` falls back to), then the channel context, the same backend instructions and the
    same tools. The Live frontend and core run no demo: they give a session or a request with no instructions their own
    defaults.

    Raises:
        KeyError: The catalog has no line of a name in ``CONSOLE_LINES``.
        ValueError: The catalog's ``console_presets`` block is malformed.
    """
    backend = ui_bot_core(root().config).backend
    sources = tool_sources(backend.tools)()
    tools = [tool for source in sources for tool in source.tools()]
    lines = {name: catalog_line(name) for name in CONSOLE_LINES}
    talker = role_instructions(default_prompt_key(load_yaml_file(CATALOG_PATH)))
    return ConsoleSample(role_instructions(backend.prompt), tools, lines, console_presets(), talker)


def console_presets() -> dict[str, list[list[str]]]:
    """The catalog's ``console_presets`` block: by append type, the [label, text] presets the console offers.

    A catalog without the block offers none.

    Raises:
        ValueError: The block is not a mapping of append type to a list of [label, text] pairs of text.
    """
    block = load_yaml_file(CATALOG_PATH).get(CONSOLE_PRESETS) or {}
    if not isinstance(block, dict) or not all(
        isinstance(entries, list) and all(map(_is_preset, entries)) for entries in block.values()
    ):
        raise ValueError(f"prompts.yaml {CONSOLE_PRESETS} must map each append type to a list of [label, text] pairs")
    return block


def _is_preset(entry: object) -> bool:
    """Whether ``entry`` is one append preset: a [label, text] pair of text."""
    return isinstance(entry, list) and len(entry) == 2 and all(isinstance(text, str) for text in entry)
