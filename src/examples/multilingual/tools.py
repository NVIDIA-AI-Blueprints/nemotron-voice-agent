# SPDX-FileCopyrightText: Copyright (c) 2024–2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Client-declared tool schemas for the multilingual assistant pipeline.

Every tool executed here runs on the *client*, not the server — the browser
is the only party in a position to do so safely or accurately (it alone
knows the user's own wall-clock time, and arithmetic is evaluated inside a
sandboxed Web Worker with every network-capable API disabled, so the server
never has to run arbitrary LLM-authored code).

Rather than a fixed schema baked into the server, the client declares what
it can do at connect time: ``requestData.tools`` on the WebRTC offer (or the
``client_tools`` query parameter on the WebSocket URL) carries a list of
``{name, description, parameters}`` entries, one per
``client.registerFunctionCallHandler`` the browser has wired up (see
``client/src/lib/clientTools.ts``). An entry can also carry ``perceivable: true``
when running the tool is something the user sees or hears by itself (a robot
moving its head, a sound playing): recorded sessions then count the moment the
call is sent as the first response of the turn. The flag is metadata for the
recorder only and is never part of the schema the LLM receives. ``build_client_tools`` below turns that
untrusted, client-supplied JSON into a ``ToolsSchema`` the LLM context can
use, forwarding each declared name to the client via RTVI (see
``tool_handlers.py``) exactly like a server-defined tool would be forwarded.

Since the payload arrives over the wire from the browser, it is validated as
a system boundary: malformed or excessive entries are dropped with a
warning rather than allowed to reach the LLM or crash the session.
"""

from __future__ import annotations

import re

from loguru import logger
from pipecat.adapters.schemas.tools_schema import AdapterType, ToolsSchema

_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_MAX_DESCRIPTION_LEN = 2000
_MAX_TOOLS = 20


def _is_perceivable(raw: dict, name: str) -> bool:
    """Return the tool's optional ``perceivable`` flag (absent or malformed means false)."""
    flag = raw.get("perceivable", False)
    if isinstance(flag, bool):
        return flag
    logger.warning(f"Ignoring non-boolean 'perceivable' on client tool '{name}': {flag!r}")
    return False


def perceivable_client_tools(raw_tools: object, names: tuple[str, ...]) -> tuple[str, ...]:
    """Return the accepted tool ``names`` the client declared with ``perceivable: true``.

    ``names`` is the second value returned by ``build_client_tools`` for the same
    payload, so a dropped or duplicate declaration never contributes a flag: the
    first declaration of a name is the one that counts, as in ``build_client_tools``.
    """
    if not isinstance(raw_tools, list):
        return ()
    flags: dict[str, bool] = {}
    for raw in raw_tools[:_MAX_TOOLS]:
        name = raw.get("name") if isinstance(raw, dict) else None
        if isinstance(name, str) and name in names and name not in flags and _validate_tool(raw, quiet=True):
            flags[name] = _is_perceivable(raw, name)
    return tuple(name for name in names if flags.get(name))


def _validate_tool(raw: object, *, quiet: bool = False) -> dict | None:
    """Return a well-formed OpenAI-style function schema for one declared tool, or None.

    Only ``name``, ``description`` and ``parameters`` are copied: any other key of
    the declaration (``perceivable`` included) stays out of the LLM schema.
    """
    warn = (lambda message: None) if quiet else logger.warning
    if not isinstance(raw, dict):
        warn(f"Ignoring client tool declaration that is not an object: {raw!r}")
        return None

    name = raw.get("name")
    if not isinstance(name, str) or not _NAME_RE.match(name):
        warn(f"Ignoring client tool declaration with invalid name: {name!r}")
        return None

    description = raw.get("description", "")
    if not isinstance(description, str) or len(description) > _MAX_DESCRIPTION_LEN:
        warn(f"Ignoring client tool '{name}': invalid or oversized description")
        return None

    parameters = raw.get("parameters", {"type": "object", "properties": {}})
    if not isinstance(parameters, dict) or parameters.get("type") != "object":
        warn(f"Ignoring client tool '{name}': parameters must be a JSON-schema object")
        return None

    return {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": parameters},
    }


def build_client_tools(raw_tools: object) -> tuple[ToolsSchema | None, tuple[str, ...]]:
    """Validate a client-declared tool list and build the schema the LLM context uses.

    Returns ``(None, ())`` when there is nothing usable to offer — callers
    should leave the context's ``tools`` as ``NOT_GIVEN`` in that case rather
    than passing an empty ``ToolsSchema``.
    """
    if not isinstance(raw_tools, list):
        if raw_tools:
            logger.warning(f"Ignoring client tool declarations: expected a list, got {type(raw_tools).__name__}")
        return None, ()

    schemas: list[dict] = []
    names: list[str] = []
    for raw in raw_tools[:_MAX_TOOLS]:
        schema = _validate_tool(raw)
        if schema is None:
            continue
        name = schema["function"]["name"]
        if name in names:
            logger.warning(f"Ignoring duplicate client tool declaration: {name}")
            continue
        schemas.append(schema)
        names.append(name)

    if len(raw_tools) > _MAX_TOOLS:
        logger.warning(f"Client declared {len(raw_tools)} tools, only the first {_MAX_TOOLS} were considered")

    if not schemas:
        return None, ()

    return ToolsSchema(standard_tools=[], custom_tools={AdapterType.OPENAI: schemas}), tuple(names)
