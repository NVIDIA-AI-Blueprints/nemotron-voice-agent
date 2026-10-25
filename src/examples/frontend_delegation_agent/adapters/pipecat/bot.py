# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``bot()``: the example's entry point on the repository's shared server (``src/server.py``).

* ``protocol: "realtime"``: the shared ``/v1/realtime`` route authenticated the socket, resolved
  ``?model=nvidia/nemotron-realtime-frontend-delegation`` and handed the socket over before
  sending any event (``frontend-delegation-agent`` is a session-owned pipeline in
  ``src/realtime/gateway.py``). The session then runs exactly as on FDA's own server before: the
  same session, Realtime wire code and pipeline, always with the ``eval`` profile, and the same
  admission (``server_busy``, close code 1013).
* ``protocol: "rtvi"``: a browser UI session (WebRTC or the UI WebSocket) with the ``default``
  profile (``ui_session.py``). With ``FDA_PROFILE=eval`` browser sessions are refused.
"""

from __future__ import annotations

import json
from typing import Any

from loguru import logger
from pipecat.runner.types import RunnerArguments

from examples.frontend_delegation_agent.adapters.pipecat.profiles import REALTIME_PROFILE, ProfileError, profile_name
from examples.frontend_delegation_agent.core.server_common import show_silent_filler
from examples.frontend_delegation_agent.core.wire import server_events as ev

EXAMPLE_KEY = "frontend-delegation-agent"
REALTIME_MODEL = "nvidia/nemotron-realtime-frontend-delegation"


async def bot(runner_args: RunnerArguments) -> None:
    """Serve one session: OpenAI Realtime (always ``eval``) or the browser UI (``default``)."""
    body = getattr(runner_args, "body", None) or {}
    if body.get("protocol") == "realtime":
        await serve_realtime(runner_args)
        return
    if profile_name() == REALTIME_PROFILE:
        raise RuntimeError(
            f"{EXAMPLE_KEY} runs with FDA_PROFILE=eval, which serves the OpenAI Realtime WebSocket only "
            f"(model {REALTIME_MODEL} on /v1/realtime); deploy with FDA_PROFILE=default (or unset) for the browser UI"
        )
    from examples.frontend_delegation_agent.adapters.pipecat.ui_session import serve_ui  # noqa: PLC0415

    await serve_ui(runner_args)


async def serve_realtime(runner_args: RunnerArguments, *, runtime: Any = None) -> None:
    """Serve one OpenAI Realtime connection (``runtime`` replaces the ``eval`` runtime in tests)."""
    from examples.frontend_delegation_agent.adapters.pipecat.runtime import realtime_runtime  # noqa: PLC0415
    from examples.frontend_delegation_agent.adapters.pipecat.session import PipecatSession  # noqa: PLC0415

    body = getattr(runner_args, "body", None) or {}
    websocket = getattr(runner_args, "websocket", None)
    if body.get("protocol") != "realtime" or websocket is None:
        raise RuntimeError(f"{EXAMPLE_KEY}: serve_realtime needs the OpenAI Realtime route's WebSocket")
    query = websocket.query_params
    if runtime is None:
        try:
            runtime = realtime_runtime()
        except ProfileError as exc:
            logger.error(f"{EXAMPLE_KEY}: {exc}")
            await websocket.send_text(json.dumps(ev.error(str(exc), code="server_error", error_type="server_error")))
            await websocket.close(code=1011, reason="profile error")
            return
    if not await runtime.admit(websocket):
        return
    session = None
    try:
        factory = runtime.speech_factory
        session = PipecatSession(
            config=runtime.config.voice,
            settings=runtime.settings,
            websocket=websocket,
            routing_sink=runtime.routing_sink,
            filler_log=runtime.filler_log,
            turn_manager_factory=runtime.delegation.turn_manager,
            model=query.get("model", ""),
            show_silent_filler=show_silent_filler(query),
            speech=factory(None) if factory is not None else None,
        )
        await session.run()
    except Exception:  # noqa: BLE001 - log and drop this connection only, as FDA's own server did
        logger.exception(f"[{session.session_id if session else '-'}] session crashed")
    finally:
        runtime.release()
