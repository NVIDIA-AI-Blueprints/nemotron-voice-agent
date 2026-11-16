# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Turn plain functions into LiveKit tools that report their calls to the web UI."""

from __future__ import annotations

import functools
import inspect
import json
import time
from collections.abc import Callable

from livekit.agents import function_tool


def _ms(started: float) -> int:
    return round((time.monotonic() - started) * 1000)


def traced_tool(fn: Callable, emit: Callable[..., None]):
    """Wrap `fn` as a LiveKit function tool. `emit(kind, **data)` receives tool_call and tool_result events.

    `functools.wraps` keeps the name, docstring, and type hints, which LiveKit reads to build the tool schema.
    Errors return to the LLM as text instead of raising, so the agent can explain them.
    """
    signature = inspect.signature(fn)

    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        name = fn.__name__
        # LiveKit may pass arguments positionally, so bind them to parameter names for the UI.
        try:
            shown = dict(signature.bind(*args, **kwargs).arguments)
        except TypeError:
            shown = dict(kwargs)
        emit("tool_call", name=name, args=shown)
        started = time.monotonic()
        try:
            result = fn(*args, **kwargs)
            if inspect.isawaitable(result):
                result = await result
        except Exception as exc:
            emit("tool_result", name=name, ok=False, result=str(exc), ms=_ms(started))
            return f"Error: {exc}"
        emit("tool_result", name=name, ok=True, result=result, ms=_ms(started))
        return json.dumps(result)

    return function_tool(wrapper)
