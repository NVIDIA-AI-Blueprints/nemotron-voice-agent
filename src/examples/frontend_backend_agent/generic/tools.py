# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Talker-visible and internal-tool contracts for the generic domain."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from pipecat.adapters.schemas.tools_schema import AdapterType, ToolsSchema

from examples.frontend_backend_agent.generic import services, speech
from examples.frontend_backend_agent.src.tools import ParamSpec, ToolContext, ToolSpec

GenericService = Callable[[Mapping[str, Any]], Awaitable[dict[str, Any]]]

CALL_BACKEND_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "call_backend",
        "description": (
            "Delegate one self-contained request that requires current or externally verified data, "
            "a deterministic calculation, or explicit random generation. Do not speak in the same turn."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 2000,
                    "description": (
                        "The complete current request, with necessary conversational context and the user's "
                        "latest corrections."
                    ),
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}

CANCEL_BACKEND_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "cancel_backend",
        "description": (
            "Cancel in-progress delegated work when the user says stop, cancel, never mind, ignore that, "
            "or otherwise withdraws the pending request. Do not speak in the same turn."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
}

TOOLS_SCHEMA = ToolsSchema(
    standard_tools=[],
    custom_tools={AdapterType.OPENAI: [CALL_BACKEND_TOOL, CANCEL_BACKEND_TOOL]},
)


def _stateless(service: GenericService):
    async def run(arguments: Mapping[str, Any], context: ToolContext) -> dict[str, Any]:
        del context
        return await service(arguments)

    return run


def _validate_random_range(arguments: Mapping[str, Any]) -> None:
    low = int(arguments.get("min", 1))
    high = int(arguments.get("max", 100))
    if low > high:
        raise ValueError("invalid random range")


TOOLS: dict[str, ToolSpec] = {
    spec.name: spec
    for spec in (
        ToolSpec(
            name="get_weather",
            contract="live CURRENT conditions; never forecast, history, climate, or weather news",
            capability="check current weather",
            params={
                "city": ParamSpec(str, label="the city or location", max_len=200),
                "units": ParamSpec(
                    str,
                    required=False,
                    choices=frozenset({"celsius", "fahrenheit"}),
                    default="celsius",
                ),
            },
            run=_stateless(services.get_weather),
            speak=speech.weather,
            timeout_s=12.0,
        ),
        ToolSpec(
            name="get_stock_price",
            contract="a live public-company quote; never news, predictions, crypto, commodities, or history",
            capability="check current stock prices",
            params={
                "company_name": ParamSpec(str, label="the company name or stock ticker", max_len=200),
            },
            run=_stateless(services.get_stock_price),
            speak=speech.stock,
            timeout_s=12.0,
        ),
        ToolSpec(
            name="web_search",
            contract="current, recent, forecast, changing, externally verifiable, or uncertain information",
            capability="search the live web",
            params={
                "query": ParamSpec(str, label="what you would like me to look up", max_len=1000),
            },
            run=_stateless(services.web_search),
            speak=speech.search,
            timeout_s=30.0,
        ),
        ToolSpec(
            name="calculate_bmi",
            contract="metric adult BMI; never infer weight or height",
            capability="calculate BMI",
            params={
                "weight_kg": ParamSpec(float, label="your weight in kilograms", bounds=(1, 1000)),
                "height_m": ParamSpec(float, label="your height in metres", bounds=(0.3, 4)),
            },
            run=_stateless(services.calculate_bmi),
            speak=speech.bmi,
            timeout_s=1.0,
        ),
        ToolSpec(
            name="generate_random_number",
            contract="one random inclusive integer; use only when randomness is explicit",
            capability="generate a random number",
            params={
                "min": ParamSpec(int, required=False, bounds=(-1_000_000_000, 1_000_000_000), default=1),
                "max": ParamSpec(int, required=False, bounds=(-1_000_000_000, 1_000_000_000), default=100),
            },
            run=_stateless(services.generate_random_number),
            speak=speech.random_number,
            validate=_validate_random_range,
            timeout_s=1.0,
        ),
    )
}


def resolve_enabled_tools(raw_tools: object) -> tuple[str, ...]:
    """Resolve registry-bound names against the immutable generic ToolSpec registry."""
    if isinstance(raw_tools, str) and raw_tools.strip():
        requested = [] if raw_tools.strip().lower() == "none" else raw_tools.split(",")
    elif isinstance(raw_tools, list | tuple):
        requested = raw_tools
    else:
        requested = ()
    normalized = (str(item).strip() for item in requested)
    return tuple(dict.fromkeys(name for name in normalized if name in TOOLS))
