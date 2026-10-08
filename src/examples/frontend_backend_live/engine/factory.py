# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The entry points ``live.mount`` calls.

They build the engine for a new session, describe the example, and run the backend for a client that owns its
delegation.

``LIVE_ENGINE_FACTORY`` names :func:`create_engine`. It reads ``config.yaml`` to pick the frontend: the cascade, or a
remote realtime model. The models and the speech services come from the registry; the client supplies the prompts
and tools.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict

from pydantic import ValidationError

import examples_registry
from examples.frontend_backend_live.cascade.engine import CascadeLiveEngine
from examples.frontend_backend_live.cascade.speech import build_stt, build_tts
from examples.frontend_backend_live.config_manager.loader import load_live_config
from examples.frontend_backend_live.delegation.thinker import build_thinker
from examples.frontend_backend_live.engine.base import BaseLiveEngine
from examples.frontend_backend_live.models.endpoints import resolve_role_endpoints
from examples.frontend_backend_live.prompts.catalog import role_instructions
from examples.frontend_backend_live.realtime.engine import REALTIME_PROVIDER, RealtimeLiveEngine, endpoint_problem
from examples.frontend_backend_live.tool_calling.cafe.schemas import CAFE_TOOLS
from live.protocol import BackendConfig, SessionConfig
from utils import build_services_api_response, load_service_entry, set_service_context

EXAMPLE_KEY = "frontend-backend-live"


def describe() -> dict:
    """Describe the example for the live console: its models, prompts, guards and sample tools (no credentials)."""
    set_service_context(examples_registry.find(EXAMPLE_KEY))
    live_config = load_live_config()
    frontend, backend = resolve_role_endpoints(live_config, {})

    def role(config, endpoint) -> dict:
        return {
            "provider": config.provider,
            "slot": config.slot,
            "model": endpoint.model,
            "base_url": endpoint.base_url,
            "prompt": config.prompt,
        }

    warnings = []
    if live_config.frontend.provider == REALTIME_PROVIDER and (problem := endpoint_problem(live_config, frontend)):
        warnings.append(problem)
    offered = [e.get("model_id") for e in build_services_api_response().get(live_config.backend.slot, [])]
    return {
        "warnings": warnings,
        "backend_models": list(dict.fromkeys([backend.model, *[m for m in offered if m]])),
        "frontend": role(live_config.frontend, frontend),
        "backend": role(live_config.backend, backend),
        "prompt_version": live_config.prompt_version,
        "guards": asdict(live_config.guards),
        "reliability": asdict(live_config.reliability),
        "instructions": {
            "frontend": role_instructions(live_config.frontend.prompt),
            "backend": role_instructions(live_config.backend.prompt),
        },
        "sample_tools": CAFE_TOOLS,
    }


async def delegate(payload: dict) -> dict:
    """Run one round of the backend for a client that does the delegated work itself (``client`` delegation).

    The client keeps the conversation: ``input`` is the whole history so far, and ``tools`` are the function tools it
    can run. The reply is the round's output items. The client runs any function calls in them and sends the next
    round with their outputs, until a round returns no calls.

    Raises:
        ValueError: ``payload`` is not a valid request.
    """
    items, tools = payload.get("input"), payload.get("tools") or []
    if not isinstance(items, list) or not items or not all(isinstance(item, dict) for item in items):
        raise ValueError("input must be a non-empty list of items")
    try:
        validated = BackendConfig(model="client", tools=tools)
    except ValidationError as exc:
        raise ValueError("tools must be function or web_search tools") from exc
    set_service_context(examples_registry.find(EXAMPLE_KEY))
    live_config = load_live_config()
    _, backend_endpoint = resolve_role_endpoints(live_config, {})
    request = {
        "model": str(payload.get("model") or backend_endpoint.model),
        "instructions": str(payload.get("instructions") or role_instructions(live_config.backend.prompt)),
        "tools": validated.tools,
        "tool_choice": "auto",
        "input": items,
    }
    thinker = build_thinker(live_config, backend_endpoint)
    output, response_id = [], ""
    try:
        async with asyncio.timeout(live_config.reliability.api_timeout_seconds):
            async for event in thinker.stream(request):
                kind = event.get("type")
                if kind == "response.output_item.done":
                    output.append(event["item"])
                elif kind == "response.completed":
                    response_id = event["response"]["id"]
                elif kind in {"response.failed", "response.incomplete"}:
                    raise RuntimeError(f"The backend ended the round with {kind}")
    finally:
        await thinker.backend.close()
    return {
        "response": {"id": response_id, "output": output},
        "sent_request": {"model": request["model"], "tools": [t.get("name") or t.get("type") for t in validated.tools]},
    }


async def create_engine(config: SessionConfig, transport: str, connection) -> BaseLiveEngine:
    """Engine factory for ``live.gateway``: resolve models and speech services from the registry."""
    set_service_context(examples_registry.find(EXAMPLE_KEY))
    live_config = load_live_config()
    body: dict = {}
    voice = config.audio.output.voice
    if isinstance(voice, dict):
        body["tts_voice_id"] = voice["id"]
    frontend_endpoint, backend_endpoint = resolve_role_endpoints(live_config, body)
    shared = {
        "live_config": live_config,
        "frontend_endpoint": frontend_endpoint,
        "backend_endpoint": backend_endpoint,
        "instructions": config.instructions or role_instructions(live_config.frontend.prompt),
        "backend_instructions": role_instructions(live_config.backend.prompt),
    }
    if live_config.frontend.provider == REALTIME_PROVIDER:
        return RealtimeLiveEngine(config, transport, connection, **shared)
    return CascadeLiveEngine(
        config,
        transport,
        connection,
        stt=build_stt(body, load_service_entry("asr", "")),
        tts=build_tts(body, load_service_entry("tts", "")),
        **shared,
    )
