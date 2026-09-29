# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Schema-validated client-tool contracts for Generic Realtime planning."""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from jsonschema.protocols import Validator

from examples.frontend_backend_agent.src.protocol import tool_result
from realtime.tool_schema import compile_tool_arguments_validator, tool_argument_validation_failure

ClientToolRoundExecutor = Callable[
    [tuple[tuple[str, dict[str, Any]], ...], float],
    Awaitable[list[str | dict[str, Any]]],
]

_SPACE_RE = re.compile(r"\s+")
_MAX_CLIENT_RESULT_SPEECH_CHARS = 450


@dataclass(frozen=True, slots=True)
class ClientToolSpec:
    """One client-owned function admitted to the hidden planner."""

    name: str
    description: str
    parameters: dict[str, Any]
    validator: Validator


def build_client_tool_specs(raw_tools: Sequence[Mapping[str, Any]]) -> dict[str, ClientToolSpec]:
    """Compile canonical Realtime schemas once at session setup."""
    specs: dict[str, ClientToolSpec] = {}
    for raw in raw_tools:
        name = raw.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("Client tool names must be non-empty strings")
        if name in specs:
            raise ValueError(f"Duplicate client tool name {name!r}")
        parameters = raw.get("parameters", {})
        if not isinstance(parameters, Mapping):
            raise ValueError(f"Client tool {name!r} parameters must be an object")
        description = raw.get("description", "")
        if not isinstance(description, str):
            raise ValueError(f"Client tool {name!r} description must be text")
        schema = dict(parameters)
        specs[name] = ClientToolSpec(
            name=name,
            description=description,
            parameters=schema,
            validator=compile_tool_arguments_validator(schema),
        )
    return specs


def validate_client_arguments(spec: ClientToolSpec, arguments: object) -> str | None:
    """Return a bounded public-safe validation message, if invalid."""
    failure = tool_argument_validation_failure(spec.validator, arguments)
    return failure.message if failure is not None else None


def client_call_fingerprint(name: str, arguments: Mapping[str, Any]) -> str:
    """Return the stable duplicate-suppression identity for one call."""
    encoded_arguments = json.dumps(
        arguments,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"{name}\0{encoded_arguments}"


def format_client_result(name: str, arguments: dict[str, Any], output: str | dict[str, Any]) -> dict[str, Any]:
    """Turn an opaque client output into one grounded, bounded agent payload."""
    if isinstance(output, dict):
        error = output.get("error")
        if isinstance(error, Mapping):
            status = "unavailable"
            message = str(error.get("message") or "The client tool did not return a usable result.")
        else:
            status = "success" if output.get("ok", True) is not False else "error"
            message = json.dumps(output, ensure_ascii=False, allow_nan=False, sort_keys=True)
        raw_result: Any = output
    else:
        stripped = output.strip()
        status = "error" if stripped.casefold().startswith("error:") else "success"
        message = stripped
        try:
            raw_result = json.loads(stripped)
        except json.JSONDecodeError:
            raw_result = stripped
    speech = _SPACE_RE.sub(" ", message).strip()
    if len(speech) > _MAX_CLIENT_RESULT_SPEECH_CHARS:
        speech = speech[: _MAX_CLIENT_RESULT_SPEECH_CHARS - 1].rsplit(" ", 1)[0].rstrip(" ,;:-.") + "."
    if not speech:
        speech = "The client tool returned an empty result."
        status = "error"
    return tool_result(
        tool=name,
        status=status,
        data={"arguments": arguments, "result": raw_result, "owner": "client"},
        response_text=speech,
        context=name,
    )
