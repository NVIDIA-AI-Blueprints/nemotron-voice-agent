# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Generic-assistant domain specification for the shared voice pipeline."""

from __future__ import annotations

from datetime import datetime

from examples.frontend_backend_agent.generic.backend import GenericThinkerBackend
from examples.frontend_backend_agent.generic.client_tools import build_client_tool_specs
from examples.frontend_backend_agent.generic.planner import NvidiaGenericPlanner
from examples.frontend_backend_agent.generic.tools import TOOLS, TOOLS_SCHEMA, resolve_enabled_tools
from examples.frontend_backend_agent.src.domain import DomainBuildContext, DomainSpec
from utils import parse_env_float, parse_env_int


def _runtime_context() -> str:
    now = datetime.now().astimezone()
    return (
        "\n\nRuntime context:\n"
        f"- The local date is {now.date().isoformat()}.\n"
        f"- The local timezone is {now.tzinfo}.\n"
        "- Delegate current, recent, forecast, or otherwise changing facts instead of answering from memory."
    )


def _build_backend(context: DomainBuildContext) -> GenericThinkerBackend:
    server_tools = resolve_enabled_tools(context.tool_names)
    enabled_specs = tuple(TOOLS[name] for name in server_tools)
    client_specs = build_client_tool_specs(context.client_tools)
    enabled_tools = (*server_tools, *client_specs)
    planner = NvidiaGenericPlanner(
        llm=context.thinker_llm,
        system_prompt=context.thinker_prompt,
        enabled_tools=enabled_specs,
        client_tools=context.client_tools,
        client_instructions=context.client_instructions,
        max_tokens=context.thinker_max_tokens,
        stage_metrics=context.stage_metrics,
        model_name=context.thinker_model_name,
    )
    return GenericThinkerBackend(
        planner=planner,
        tools=TOOLS,
        enabled_tools=enabled_tools,
        client_tools=client_specs,
        client_tool_executor=context.client_tool_executor,
        client_tool_timeout_seconds=parse_env_float(
            "GENERIC_CLIENT_TOOL_TIMEOUT_SECONDS",
            25.0,
            min_value=1.0,
        ),
        overall_timeout_seconds=parse_env_float("GENERIC_BACKEND_TIMEOUT_SECONDS", 40.0, min_value=1.0),
        # A plan that must act, rather than look up, spends far more of the
        # reasoning budget: measured round-two plans ran 5-8s where a lookup
        # ran under one. Six seconds sat inside that spread, so every acting
        # turn timed out twice and fell back to failure speech. Two attempts at
        # ten still fit the forty-second overall budget with room for a round.
        planner_timeout_seconds=parse_env_float("GENERIC_PLANNER_TIMEOUT_SECONDS", 10.0, min_value=1.0),
        max_planning_rounds=parse_env_int("GENERIC_MAX_PLANNING_ROUNDS", 8, min_value=1),
        on_tool_started=context.on_tool_started,
        stage_metrics=context.stage_metrics,
    )


def _build_realtime_prompt_coordinator(**kwargs):
    from examples.frontend_backend_agent.generic.realtime_prompts import GenericRealtimePromptCoordinator

    return GenericRealtimePromptCoordinator(**kwargs)


def create_domain_spec() -> DomainSpec:
    """Return the generic domain's prompts, tools, backend, and speech policy."""
    return DomainSpec(
        key="generic",
        label="Generic Assistant",
        thinker_prompt_key="generic_thinker",
        talker_prompt_key="generic_talker",
        talker_tools_schema=TOOLS_SCHEMA,
        build_backend=_build_backend,
        runtime_context=_runtime_context,
        filler_policy="talker_authored",
        tool_registry=TOOLS,
        realtime_prompt_coordinator_factory=_build_realtime_prompt_coordinator,
        max_query_chars=2000,
    )
