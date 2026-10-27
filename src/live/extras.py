# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Routes beside the live session protocol: ``GET /v1/live/info`` and ``POST /v1/live/delegate``.

``/info`` describes the server for clients such as the live console: the protocol's voices, the session limit, and
what the composition root's optional ``describe`` hook adds. ``/delegate`` runs one stateless backend round for a
client-mode app on the gateway's delegation core client, with the client's whole history as input; it has its own
limits, since it runs backend work for anyone who reaches it (:func:`add_extras`).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, HTTPException, Request
from loguru import logger
from pydantic import ValidationError

from live.gateway import LiveGateway, security
from live.protocol import LIVE_VOICES, BackendConfig, ProtocolError

Describe = Callable[[], Awaitable[dict]]


def console_request(payload: dict) -> dict:
    """The stateless Responses request of one console round, from the client's body.

    Raises:
        HTTPException: 400: ``input`` is not a non-empty list of items, a tool is not an official one, ``model`` or
            ``instructions`` is not a string, or the request holds a value plain JSON in UTF-8 cannot carry.
    """
    items, tools = payload.get("input"), payload.get("tools") or []
    if not isinstance(items, list) or not items or not all(isinstance(item, dict) for item in items):
        raise HTTPException(400, "input must be a non-empty list of items")
    try:
        tools = BackendConfig(model="client", tools=tools).tools
    except ValidationError as exc:  # the answer the console's parity cases record
        raise HTTPException(400, "tools must be function or web_search tools") from exc
    request = {"input": items, "tools": tools, "tool_choice": "auto", "stream": True, "store": False}
    for key in ("model", "instructions"):
        value = payload.get(key)
        if value is not None and not isinstance(value, str):
            raise HTTPException(400, f"{key} must be a string")
        if value:
            request[key] = value
    try:  # the backend's model client sends plain JSON in UTF-8: a value it cannot carry is the client's error
        json.dumps(request, allow_nan=False, ensure_ascii=False).encode("utf-8")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    return request


async def read_round(stream) -> dict:
    """Read a console round's stream to its end; return the completed response.

    Raises:
        RuntimeError: The round ended ``failed`` or ``incomplete``, or with no terminal.
    """
    completed = None
    try:
        async for event in stream:
            kind = event.get("type")
            if kind == "response.completed":
                completed = event["response"]
            elif kind in {"response.failed", "response.incomplete"}:
                raise RuntimeError(f"The delegation core ended the round with {kind}")
    finally:
        await stream.aclose()  # a round cut short ends its one-shot work in the core
    if completed is None:
        raise RuntimeError("The delegation core ended the round without a terminal event")
    return completed


async def run_round(gateway: LiveGateway, request: dict) -> dict:
    """Check the round's model with the core, then send the round and read it to its end, within the backstop.

    Raises:
        HTTPException: 503: the server has no delegation core, or it could not be built; 400: the core does not serve
            the model, or refuses the request; 502: any other fault, the round's failure or its backstop included.
    """
    try:
        core = gateway.core()
        async with asyncio.timeout(gateway.settings.console_round_seconds):
            if "model" in request:
                try:
                    await core.retrieve_model(request["model"])
                except ValueError as unknown:  # GET /v1/models/{model}: the core does not serve it
                    raise HTTPException(400, f"Unknown model {request['model']!r}") from unknown
            try:
                stream = await core.create_response(request)
            except ValueError as refused:  # the core refused the request, before any event
                raise HTTPException(400, str(refused)) from refused
            return await read_round(stream)
    except HTTPException:
        raise
    except ProtocolError as exc:  # no core, or one that could not be built: the server's fault, said without detail
        raise HTTPException(503, "The delegation core is not available") from exc
    except Exception as exc:
        logger.exception(f"A console round failed: {exc}")
        raise HTTPException(502, "The backend could not complete the round") from exc


def add_extras(app: FastAPI, gateway: LiveGateway, describe: Describe | None = None) -> None:
    """Add ``GET /v1/live/info`` and ``POST /v1/live/delegate`` to ``app``, behind ``gateway``'s security.

    ``describe`` returns plain JSON for ``/info``. ``/delegate`` runs on the gateway's one delegation core client
    (``LiveGateway.core``), which it builds on first use, within the console route's limits: at most
    ``max_console_rounds`` rounds at once (429 over it), a body of at most ``console_body_bytes``, a model the core
    serves (400 otherwise), and, with no token, a ``Host`` that is an IP literal, ``localhost`` or an allowed name.
    """
    running = 0  # console rounds in flight

    @app.get("/v1/live/info")
    async def info(request: Request) -> dict:
        """Describe the server for clients: the protocol's voices, plus the example's models, prompts and guards."""
        gateway.authorize(request)
        described = await describe() if describe is not None else {}
        return {"voices": sorted(LIVE_VOICES), "max_sessions": gateway.settings.max_sessions, **described}

    @app.post("/v1/live/delegate")
    async def delegate(request: Request) -> dict:
        """Run one backend round for a client that owns its delegation.

        The client sends the whole conversation as ``input`` (and optionally ``tools``, ``model``, ``instructions``),
        runs the function calls in the reply, and sends the next round with their outputs. A bad body is a 400 before
        any round runs; a model the core does not serve and the core's refusal of the request are a 400 too; a server
        fault is a 502 or a 503 that names no detail of the server.
        """
        nonlocal running
        security.check_host(gateway.settings, request)
        data = await gateway.body(request, gateway.settings.console_body_bytes)
        round_request = console_request(data)
        if running >= gateway.settings.max_console_rounds:
            raise HTTPException(429, f"Too many console rounds: at most {gateway.settings.max_console_rounds} at once")
        running += 1
        try:
            completed = await run_round(gateway, round_request)
        finally:
            running -= 1
        return {
            "response": {"id": completed["id"], "output": completed["output"]},
            "sent_request": {
                "model": completed.get("model"),
                "tools": [tool.get("name") or tool.get("type") for tool in round_request["tools"]],
            },
        }
