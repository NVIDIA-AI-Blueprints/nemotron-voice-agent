# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The session choices that the UI sends with the join token, validated. No LiveKit imports.

The UI server turns the choices into participant attributes of the join token, and the agent reads them when the user
joins. A client cannot change the attributes afterwards, and anything that the server did not offer is dropped here.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from nemotron_voice_agent.core.service_status import parse_endpoint

from .config import CUSTOM_FIELDS, custom_entry
from .scenarios import get_scenario
from .service_state import find_entry, voices_of

MAX_FIELD_CHARS = 512
MAX_PROMPT_CHARS = 8000
MAX_SETTING_CHARS = 64


def _expect(value: object, kind: type | tuple[type, ...], name: str) -> None:
    """Reject a field of the wrong JSON type, so a client bug is reported instead of silently dropping the choice.

    Raises:
        ValueError: `value` is present but not of `kind`.
    """
    if value is not None and not isinstance(value, kind):
        raise ValueError(f"{name} has the wrong type")


def _setting_values(raw: object, schema: Mapping[str, Any]) -> dict[str, Any]:
    """Keep the values of settings that the entry offers. The agent checks each value against its schema."""
    if not isinstance(raw, Mapping):
        return {}
    return {
        name: value
        for name, value in raw.items()
        if name in schema
        and isinstance(value, bool | int | float | str)
        and not (isinstance(value, str) and len(value) > MAX_SETTING_CHARS)
    }


def _custom_fields(slot: str, raw: Mapping[str, Any]) -> dict[str, str]:
    """Return a custom endpoint's fields.

    Raises:
        ValueError: A field is not a short string, a required one is empty, or the address is not usable.
    """
    fields = {name: raw.get(name) or "" for name in CUSTOM_FIELDS[slot]}
    if not all(isinstance(value, str) and len(value) <= MAX_FIELD_CHARS for value in fields.values()):
        raise ValueError(f"custom {slot} fields must be text of at most {MAX_FIELD_CHARS} characters")
    entry = custom_entry(slot, fields)
    address = entry["base_url"] if slot == "llm" else entry["server"]
    if not _usable_address(slot, address):
        raise ValueError(f"custom {slot} address {address!r} is not usable")
    return {name: value for name, value in entry.items() if value}


def _usable_address(slot: str, address: str) -> bool:
    """An LLM needs an http(s) URL. A speech service needs a gRPC `host:port`. Neither may carry a user or password."""
    endpoint = parse_endpoint(address)
    if endpoint is None or endpoint[1] == 0 or "@" in address or any(char.isspace() for char in address):
        return False
    if slot == "llm":
        return address.startswith(("http://", "https://"))
    return "://" not in address and address.rpartition(":")[2].isdigit()


def _choice(slot: str, raw: object, state: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return one slot's validated choice: an offered entry with its settings, or a custom endpoint."""
    _expect(raw, Mapping, f"services.{slot}")
    if raw is None:
        return None
    _expect(raw.get("custom"), Mapping, f"services.{slot}.custom")
    _expect(raw.get("key"), str, f"services.{slot}.key")
    _expect(raw.get("settings"), Mapping, f"services.{slot}.settings")
    if raw.get("custom") is not None:
        return {"custom": _custom_fields(slot, raw["custom"])}
    key = raw.get("key")
    entry = find_entry(state, slot, key) if isinstance(key, str) else None
    if entry is None:
        return None
    settings = _setting_values(raw.get("settings"), entry.get("settings") or {})
    return {"key": key, **({"settings": settings} if settings else {})}


def _default_key(state: Mapping[str, Any], slot: str) -> str:
    entries = state["services"].get(slot) or []
    return next((entry["id"] for entry in entries if entry.get("default")), entries[0]["id"] if entries else "")


def session_attributes(body: Mapping[str, Any], state: Mapping[str, Any]) -> dict[str, str]:
    """Return the participant attributes for a join token from the UI's session choices.

    Args:
        body: The JSON that the UI posts: `scenario`, an optional custom `prompt`, `voice`, and per slot under
            `services` either `{"key": ..., "settings": {...}}` or `{"custom": {...}}`.
        state: The services that the worker published, from `service_state.read_service_state`.

    The use case falls back to the default when it is missing or unknown. A service, a setting, or a voice that the
    worker did not offer is dropped, so a client cannot make the agent use an arbitrary catalog key or voice name.

    Raises:
        ValueError: A field has the wrong type, the custom prompt is too long, or a custom endpoint is not usable.
            The message says why, for the UI.
    """
    for name, kind in (("services", Mapping), ("scenario", str), ("prompt", str), ("voice", str)):
        _expect(body.get(name), kind, name)
    services = body.get("services") or {}
    choices = {slot: choice for slot in state["services"] if (choice := _choice(slot, services.get(slot), state))}
    attributes = {"scenario": get_scenario(body.get("scenario")).id}
    prompt = (body.get("prompt") or "").strip()
    if len(prompt) > MAX_PROMPT_CHARS:
        raise ValueError(f"the custom prompt is longer than {MAX_PROMPT_CHARS} characters")
    if prompt:
        attributes["prompt"] = prompt
    tts = choices.get("tts", {})
    if "custom" not in tts:
        voice = body.get("voice")
        listed = {v["id"] for v in voices_of(state, tts.get("key") or _default_key(state, "tts"))["voices"]}
        if isinstance(voice, str) and voice in listed:
            attributes["voice"] = voice
    if choices:
        attributes["services"] = json.dumps(choices, separators=(",", ":"))
    return attributes


def read_choices(raw: object) -> dict[str, dict[str, Any]]:
    """Return the per-slot choices from the `services` attribute, or none when it is missing or malformed."""
    try:
        choices = json.loads(raw) if isinstance(raw, str) else {}
    except ValueError:
        return {}
    if not isinstance(choices, dict):
        return {}
    return {slot: choice for slot, choice in choices.items() if slot in CUSTOM_FIELDS and isinstance(choice, dict)}
