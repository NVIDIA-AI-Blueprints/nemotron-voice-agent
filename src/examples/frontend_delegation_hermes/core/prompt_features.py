# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prompt variants: named ``{% if features.<name> %}`` blocks in the prompt catalogs.

Each catalog gets a ``features`` mapping from config (``prompt_features`` in
``delegation_agent.yaml`` for the frontend catalog, in ``gateway.yaml`` for the backend
one). A feature that is off, or not set, renders nothing, so with every feature off the
prompts are byte-identical to the baseline (tau3-failure-fixes-plan.md section 1).
Stdlib only: the gateway imports it too.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

#: Frontend (``prompts.yaml``) variants and their defaults.
FRONTEND_FEATURES: dict[str, bool] = {"replay_intent": False}
#: Backend (``prompts.backend.yaml``) variants and their defaults.
BACKEND_FEATURES: dict[str, bool] = {
    "spelling_v2": False,
    "spoken_output": False,
    "write_consent": False,
    "domain_notes": False,  # <domain_notes>: the telecom note (on in gateway.yaml)
    "phone_format": False,
    # tau3-geval-failure-fixes-plan.md section 4 (each off by default; gateway.eval_<fix>.yaml turns one on).
    "format_correction": False,  # A2: a separator or case change is a format correction, not a variant
    "write_consent_v2": False,  # B1: the policy's confirmation, to the exact change, before a write
    "policy_check": False,  # B2: <before_change> policy check before every write
    "request_scope": False,  # B3: <before_change> cover every record and option the request names
    "transfer_check": False,  # C1: <before_transfer> required transfers at once, else remaining steps first
    "capability_check": False,  # C2: <before_transfer> check the tools before saying "I can't"
    # fix-plan-f1-f2-f4-f8.md (each off by default).
    "situational_facts": False,  # F4: <situational_facts> ask early where the user is, if the policy depends on it
    "identity_from_identifier": False,  # F8: a name inside a spoken identifier may be used once confirmed
}


class PromptFeatures(dict[str, bool]):
    """A feature map where an unknown name reads as off (``features.x`` in Jinja)."""

    def __missing__(self, key: str) -> bool:
        """An unset feature is off."""
        return False


def features(values: Mapping[str, object] | None, known: Mapping[str, bool]) -> PromptFeatures:
    """``known`` defaults overridden by ``values``; an unknown name is an error."""
    merged = dict(known)
    for name, value in (values or {}).items():
        if name not in known:
            raise ValueError(f"unknown prompt feature {name!r}; known: {sorted(known)}")
        merged[name] = _bool(value, name)
    return PromptFeatures(merged)


def render_backend_system(templates: Any, instructions: str, domain_note: str = "") -> str:
    """The rendered ``backend_system``: the one render used by the gateway and the G2 replay.

    fix-plan-f1-f2-f4-f8.md section 3.2: both call this, so an arm's replayed prompt is exactly what the
    gateway would send.
    """
    if domain_note:
        return templates.render("backend_system", instructions=instructions, domain_note=domain_note)
    return templates.render("backend_system", instructions=instructions)


def tools_canonical_sha256(tools: Any) -> str:
    """One hash of a tool list in either wire shape (Realtime flat or Chat Completions ``function``-wrapped).

    sha256 of the name-sorted ``{name, description, parameters}`` entries as canonical JSON (first 16 hex
    digits). The voice server (``backend_configured``) and the backend trace (``session``) both log it, so a
    trace joins its session by the same value (fix-plan-f1-f2-f4-f8.md section 1).
    """
    entries = []
    for tool in tools or ():
        function = tool.get("function") if isinstance(tool.get("function"), Mapping) else tool
        entries.append(
            {
                "name": str(function.get("name") or ""),
                "description": str(function.get("description") or ""),
                "parameters": function.get("parameters") or {},
            }
        )
    entries.sort(key=lambda entry: entry["name"])
    blob = json.dumps(entries, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def sha256_text(text: str) -> str:
    """The fingerprint of one rendered prompt or catalog (first 16 hex digits)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _bool(value: object, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in ("true", "false", "1", "0", "yes", "no", "on", "off"):
        return value.lower() in ("true", "1", "yes", "on")
    raise ValueError(f"prompt feature {name!r}: {value!r} is not a boolean")
