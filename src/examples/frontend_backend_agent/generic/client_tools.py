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


_SPOKEN_DIGITS = {
    "zero": "0",
    "oh": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
}
_LOWER_SNAKE_ID_RE = re.compile(r"^[a-z]+(?:_[a-z]+)*_\d+$")
_UPPER_ALNUM_ID_RE = re.compile(r"^[A-Z0-9]{4,}$")
_EXAMPLE_RE = re.compile(r"such as ['\"]([^'\"]+)['\"]")


def _spoken_digits_to_numerals(value: str) -> str:
    """Collapse a dictated run of digit words into the digits it names."""
    parts = value.split("_")
    out: list[str] = []
    run: list[str] = []
    for part in parts:
        digit = _SPOKEN_DIGITS.get(part.casefold())
        if digit is not None:
            run.append(digit)
            continue
        if run:
            out.append("".join(run))
            run = []
        out.append(part)
    if run:
        out.append("".join(run))
    return "_".join(out)


def _identifier_example(schema: object) -> str | None:
    """Return the literal the caller offered as this field's shape, if any."""
    if not isinstance(schema, Mapping):
        return None
    description = schema.get("description")
    if not isinstance(description, str):
        return None
    found = _EXAMPLE_RE.search(description)
    return found.group(1) if found else None


def normalize_client_arguments(spec: ClientToolSpec, arguments: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Repair identifiers a caller spelled out loud back into the caller's own shape.

    A user reads an account number down the phone, so the transcript carries
    "Omar underscore Davis underscore three eight one seven" and the model
    relays what it heard. The store wants ``omar_davis_3817``, and an exact-key
    lookup simply misses -- which fails authentication, and every policy that
    depends on it. Only fields whose own description names an example are
    touched, and only when the repair lands on that example's shape, so a
    passenger's name or date of birth is never rewritten.
    """
    properties = spec.parameters.get("properties")
    if not isinstance(properties, Mapping):
        return dict(arguments), []
    repaired: dict[str, Any] = dict(arguments)
    changed: list[str] = []
    for name, value in arguments.items():
        if not isinstance(value, str) or not value:
            continue
        example = _identifier_example(properties.get(name))
        if example is None:
            continue
        if _LOWER_SNAKE_ID_RE.match(example):
            candidate = _spoken_digits_to_numerals(value).casefold()
            matches = _LOWER_SNAKE_ID_RE.match(candidate)
        elif _UPPER_ALNUM_ID_RE.match(example):
            candidate = _spoken_digits_to_numerals(value).replace("_", "").replace(" ", "").upper()
            matches = _UPPER_ALNUM_ID_RE.match(candidate)
        else:
            continue
        if matches and candidate != value:
            repaired[name] = candidate
            changed.append(name)
    return repaired, changed


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


def _classify_mapping_result(output: Mapping[str, Any]) -> tuple[str, str | None]:
    """Classify one structured client-tool envelope into a status and speech message.

    A message comes back only when the envelope carries a human-readable one.
    A structured payload yields ``None``: its fields are the tool's data, not a
    sentence. Serializing them into speech is exactly how a client tool's raw
    records reach the speaker verbatim, personal details included. The data
    stays in the payload for the Talker to compose from instead.
    """
    error = output.get("error")
    if isinstance(error, Mapping):
        return "unavailable", str(error.get("message") or "The client tool did not return a usable result.")
    if isinstance(error, str) and error.strip():
        return "unavailable", error.strip()
    if output.get("ok", True) is False:
        return "error", None
    return "success", None


def format_client_result(name: str, arguments: dict[str, Any], output: str | dict[str, Any]) -> dict[str, Any]:
    """Turn an opaque client output into one grounded, bounded agent payload."""
    raw_result: Any
    message: str | None
    if isinstance(output, dict):
        raw_result = output
        status, message = _classify_mapping_result(output)
    else:
        stripped = output.strip()
        try:
            raw_result = json.loads(stripped)
        except json.JSONDecodeError:
            raw_result = stripped
        if isinstance(raw_result, Mapping):
            # Client transports often hand back a JSON-encoded envelope. Classify it
            # by its parsed contents so an error payload is never read as success.
            status, message = _classify_mapping_result(raw_result)
        elif isinstance(raw_result, (list, tuple)):
            # A bare array is data for the same reason a mapping is.
            status, message = "success", None
        else:
            status = "error" if stripped.casefold().startswith("error:") else "success"
            message = stripped
    if message is None:
        # Structured data carries no sentence. Say only that the result
        # arrived; the Talker speaks the facts from the payload's data.
        speech = f"I have the {name} result." if status == "success" else f"The {name} result came back unusable."
    else:
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
