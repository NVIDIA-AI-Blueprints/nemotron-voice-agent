# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Airline domain adapter for the shared Frontend/Backend Agent pipeline."""

from __future__ import annotations

import os
from datetime import timedelta

from examples.frontend_backend_agent.airline.backend import HTTPBookingBackend
from examples.frontend_backend_agent.airline.thinker import ThinkerBackend
from examples.frontend_backend_agent.airline.tools import TOOLS_SCHEMA
from examples.frontend_backend_agent.airline.tts_filter import apply_frontend_backend_agent_pronunciation_for_tts
from examples.frontend_backend_agent.src.domain import DomainBuildContext, DomainSpec
from examples.frontend_backend_agent.src.planner import NvidiaThinkerPlanner
from examples.frontend_backend_agent.src.runtime_context import runtime_today
from utils import parse_env_float


def default_booking_backend_url() -> str:
    """Return the legacy booking service URL for the current runtime."""
    if os.environ.get("APP_RUNTIME", "").strip().lower() == "container":
        return "http://booking-server:8001"
    return "http://localhost:8001"


def booking_backend_url(default_booking_server: dict) -> str:
    """Resolve the booking URL while preserving explicit operator overrides."""
    explicit_url = os.getenv("BOOKING_BACKEND_URL", "").strip()
    if explicit_url:
        return explicit_url
    configured_url = str(default_booking_server.get("server") or "").strip()
    runtime_default = default_booking_backend_url()
    if runtime_default == "http://localhost:8001" and configured_url == "http://booking-server:8001":
        return runtime_default
    return configured_url or runtime_default


def _runtime_context() -> str:
    today = runtime_today()
    return (
        "\n\nRuntime context:\n"
        f"- Today is {today.isoformat()}.\n"
        f"- Tomorrow is {(today + timedelta(days=1)).isoformat()}.\n"
        "- For travel dates without a year, choose the next upcoming occurrence relative to today.\n"
        "- Always pass travel dates to call_backend as ISO YYYY-MM-DD when the date is known."
    )


def _build_backend(context: DomainBuildContext) -> ThinkerBackend:
    default_booking_server = context.load_service_entry("booking-server", "")
    backend_url = booking_backend_url(default_booking_server)
    planner = NvidiaThinkerPlanner(
        llm=context.thinker_llm,
        system_prompt=context.thinker_prompt,
        max_tokens=context.thinker_max_tokens,
    )
    return ThinkerBackend(
        backend=HTTPBookingBackend(backend_url),
        planner=planner,
        tool_delay_seconds=context.tool_delay_seconds,
        tool_delay_min_seconds=context.tool_delay_min_seconds,
        overall_timeout_seconds=parse_env_float("AIRLINE_BACKEND_TIMEOUT_SECONDS", 30.0, min_value=1.0),
        planner_timeout_seconds=parse_env_float("AIRLINE_PLANNER_TIMEOUT_SECONDS", 30.0, min_value=1.0),
    )


def create_domain_spec() -> DomainSpec:
    """Return the backward-compatible airline domain specification."""
    return DomainSpec(
        key="airline",
        label="G Force Airlines",
        thinker_prompt_key="thinker",
        talker_tools_schema=TOOLS_SCHEMA,
        build_backend=_build_backend,
        filler_policy="planner_authored",
        runtime_context=_runtime_context,
        tts_text_transform=apply_frontend_backend_agent_pronunciation_for_tts,
        max_query_chars=4000,
    )
