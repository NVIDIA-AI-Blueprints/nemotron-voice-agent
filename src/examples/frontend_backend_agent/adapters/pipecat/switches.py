# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in switches for OpenAI Realtime sessions, and the eval lock.

Every Realtime session runs the ``eval`` profile, so each Realtime extra is off unless its
``FBA_REALTIME_*`` variable turns it on, and stays off in an eval-mode container
(``FBA_PROFILE=eval``, the mode evaluation runs use) whatever the variables say. Browser UI
sessions read only the shared ``ENABLE_TRACING``.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from loguru import logger

from examples.frontend_backend_agent.core.config.profiles import PROFILE_ENV, REALTIME_PROFILE

#: The registry key of this example.
EXAMPLE_KEY = "frontend-backend-agent"
#: Traces Realtime sessions too (with ``ENABLE_TRACING=true``).
REALTIME_TRACING_ENV = "FBA_REALTIME_TRACING"
#: The Realtime turn modes a client may select, comma separated (``server_vad`` is always allowed).
TURN_MODES_ENV = "FBA_REALTIME_TURN_MODES"
#: MCP tools for Realtime sessions (with ``REALTIME_MCP_ALLOWED_SERVER_URLS``).
REALTIME_MCP_ENV = "FBA_REALTIME_MCP"
#: Every boolean Realtime switch; all are ignored when ``FBA_PROFILE=eval``.
REALTIME_SWITCHES = (REALTIME_TRACING_ENV, REALTIME_MCP_ENV)
#: The shared tracing variable (``src/tracing.py``).
TRACING_ENV = "ENABLE_TRACING"

_WARNED: set[str] = set()


def _env(environ: Mapping[str, str] | None) -> Mapping[str, str]:
    return os.environ if environ is None else environ


def _true(env: Mapping[str, str], name: str) -> bool:
    return str(env.get(name) or "").strip().lower() == "true"


def eval_mode(environ: Mapping[str, str] | None = None) -> bool:
    """Whether this container serves evaluation runs (``FBA_PROFILE=eval``)."""
    return str(_env(environ).get(PROFILE_ENV) or "").strip().removesuffix(".yaml") == REALTIME_PROFILE


def realtime_switch(name: str, environ: Mapping[str, str] | None = None) -> bool:
    """A boolean Realtime switch: ``true`` turns it on, except in an eval-mode container."""
    env = _env(environ)
    if not _true(env, name):
        return False
    if eval_mode(env):
        if name not in _WARNED:
            _WARNED.add(name)
            logger.warning(
                f"{name}=true is ignored: {PROFILE_ENV}={REALTIME_PROFILE} keeps evaluation sessions unchanged"
            )
        return False
    return True


def realtime_turn_modes(environ: Mapping[str, str] | None = None) -> frozenset[str]:
    """The turn modes Realtime clients may select: ``server_vad``, plus ``manual``/``semantic_vad`` if listed.

    Unset, or an eval-mode container: ``server_vad`` only (today's behavior). Unknown names are
    ignored with a warning.
    """
    from examples.frontend_backend_agent.adapters.pipecat.turn_modes import MODES, SERVER_VAD  # noqa: PLC0415

    env = _env(environ)
    raw = str(env.get(TURN_MODES_ENV) or "")
    requested = {name.strip().lower() for name in raw.split(",") if name.strip()}
    unknown = requested - set(MODES)
    if unknown and TURN_MODES_ENV not in _WARNED:
        _WARNED.add(TURN_MODES_ENV)
        logger.warning(f"{TURN_MODES_ENV}: unknown turn modes {sorted(unknown)} are ignored (allowed: {list(MODES)})")
    modes = (requested & set(MODES)) | {SERVER_VAD}
    if modes != {SERVER_VAD} and eval_mode(env):
        if f"{TURN_MODES_ENV}:eval" not in _WARNED:
            _WARNED.add(f"{TURN_MODES_ENV}:eval")
            logger.warning(
                f"{TURN_MODES_ENV}={raw} is ignored: "
                f"{PROFILE_ENV}={REALTIME_PROFILE} keeps evaluation sessions unchanged"
            )
        return frozenset({SERVER_VAD})
    return frozenset(modes)


def eval_llm_refusal(body: Mapping[str, Any], environ: Mapping[str, str] | None = None) -> str | None:
    """In an eval-mode container, refuse a Realtime session whose LLMs are not the registry defaults.

    Today the catalogs hold only the default LLMs, so nothing else can be selected. The optional
    local LLM (services.local.yaml, ``nemotron-3.5-lightning-local``) could otherwise be chosen
    through an ``ek_`` key's ``session.nvidia.llm_id``; evaluation runs keep their LLMs.
    """
    if not eval_mode(environ):
        return None
    import examples_registry  # noqa: PLC0415

    defaults = examples_registry.find(EXAMPLE_KEY).get("defaults") or {}
    for field, slot in (("llm_id", "llm"), ("thinker_llm_id", "thinker-llm")):
        selected = str(body.get(field) or "")
        allowed = {str(key) for key in defaults.get(slot) or ()}
        if selected and selected.rsplit(":", 1)[-1] not in allowed:
            return (
                f"{field}={selected!r} is not available with {PROFILE_ENV}={REALTIME_PROFILE}: "
                f"evaluation sessions use the registry default {sorted(allowed)}"
            )
    return None


def ui_tracing(environ: Mapping[str, str] | None = None) -> bool:
    """``ENABLE_TRACING=true``: trace browser UI sessions (initializes OpenTelemetry once)."""
    if not _true(_env(environ), TRACING_ENV):
        return False
    # Imported only here: importing ``tracing`` loads ``.env`` defaults and sets up the exporter.
    import tracing  # noqa: PLC0415

    return tracing.IS_TRACING_ENABLED


def realtime_tracing(environ: Mapping[str, str] | None = None) -> bool:
    """Trace Realtime sessions: ``ENABLE_TRACING`` and ``FBA_REALTIME_TRACING``, never in eval mode."""
    return realtime_switch(REALTIME_TRACING_ENV, environ) and ui_tracing(environ)
