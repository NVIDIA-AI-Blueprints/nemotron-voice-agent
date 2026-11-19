# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The services that the agent worker publishes for the UI server: what each slot offers, and the voices of each TTS.

The worker has the recipe, the API key, and the network that reach the models, and the UI server has none of them. So
the worker writes this file into the state volume, and the UI server only reads it. Standard library only.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from nemotron_voice_agent.core.service_status import ServiceStatus, probe_http, probe_tcp

from .state import read_json, write_json_atomic

SERVICES_FILE = "services.json"
EMPTY: dict[str, Any] = {"services": {}, "voices": {}}

PROBE_TIMEOUT_S = 2
SELF_HOSTED = "self-hosted"

StatusProbe = Callable[[str, Mapping[str, Any]], ServiceStatus | None]


def probe_entry(slot: str, entry: Mapping[str, Any]) -> ServiceStatus | None:
    """Return whether a self-hosted entry runs: its LLM health URL, else its gRPC port. None for NVIDIA Cloud."""
    if entry.get("source") != SELF_HOSTED:
        return None
    base_url = str(entry.get("base_url") or "")
    if base_url and entry.get("health_path"):
        parts = urlsplit(base_url)
        return probe_http(f"{parts.scheme}://{parts.netloc}{entry['health_path']}", False, PROBE_TIMEOUT_S)
    return probe_tcp(base_url or str(entry.get("server") or ""), PROBE_TIMEOUT_S)


def with_status(options: Mapping[str, list[dict[str, Any]]], probe: StatusProbe) -> dict[str, list[dict[str, Any]]]:
    """Add each entry's `status` from `probe`, and hide the entries that are not running.

    A slot whose entries are all offline keeps them, so the UI still shows what the deployment would use. The first
    entry left in each slot is the default.
    """
    result = {}
    for slot, entries in options.items():
        annotated = [
            {**entry, "status": status} if (status := probe(slot, entry)) else dict(entry) for entry in entries
        ]
        visible = [entry for entry in annotated if entry.get("status") != "offline"] or annotated
        result[slot] = [{**entry, "default": index == 0} for index, entry in enumerate(visible)]
    return result


def write_service_state(
    path: Path, services: Mapping[str, list[dict[str, Any]]], voices: Mapping[str, Mapping[str, Any]]
) -> None:
    """Write the offered services and the voices of each TTS entry, as `{"voices": [...], "default": id}` by id."""
    write_json_atomic(path, {"services": services, "voices": voices})


def read_service_state(path: Path) -> dict[str, Any]:
    """Return `{"services": {slot: [entry]}, "voices": {tts id: {...}}}`, or empty maps when the file is unusable."""
    data = read_json(path)
    if not isinstance(data, dict) or not isinstance(data.get("services"), dict):
        return dict(EMPTY)
    services = {
        slot: [entry for entry in entries if isinstance(entry, dict) and isinstance(entry.get("id"), str)]
        for slot, entries in data["services"].items()
        if isinstance(entries, list)
    }
    voices = data.get("voices") if isinstance(data.get("voices"), dict) else {}
    return {"services": services, "voices": voices}


def find_entry(state: Mapping[str, Any], slot: str, key: str) -> dict[str, Any] | None:
    """Return the offered entry `key` of `slot`, or None."""
    return next((entry for entry in state["services"].get(slot, []) if entry["id"] == key), None)


def voices_of(state: Mapping[str, Any], tts_key: str) -> dict[str, Any]:
    """Return `{"voices": [{"id", "name"}], "default": id}` for an offered TTS entry. Empty when it was not listed."""
    found = state["voices"].get(tts_key)
    if not isinstance(found, dict):
        return {"voices": [], "default": ""}
    voices = [v for v in found.get("voices") or [] if isinstance(v, dict) and "id" in v and "name" in v]
    return {"voices": voices, "default": str(found.get("default") or "")}
