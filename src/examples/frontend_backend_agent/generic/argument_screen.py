# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Session-scoped spoken-identifier screen and lookup-recovery hints.

Everything here is derived from the session's own tool schemas
(:func:`build_schema_rules`) and is limited to actions that are harmless if a
write were misclassified as a read:

* a read's identifier argument is put into its schema example's shape, and
  only when the value does not already have that shape;
* a clearly unfinished identifier is not sent; the caller is asked for the rest;
* hints are added to the Thinker's copy of a result, never to the client's.

Code never retries a call on its own; any second attempt is a Thinker plan.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

from examples.frontend_backend_agent.src.normalization import (
    EMPTY_SCHEMA_RULES,
    SchemaRules,
    ToolCall,
    build_schema_rules,
    other_phone_form,
    screen_call,
    tools_sha256,
)
from examples.frontend_backend_agent.src.protocol import response_hint
from utils import load_prompt_catalog

#: A not-found result that names a person-like entity (§6.1 of the parity plan).
PERSON_NOT_FOUND_RE = re.compile(
    r"(?i)\b(users?|customers?|members?|clients?|profiles?)\b[^.]*\b(not found|no such|does not exist)\b"
    r"|\bno (records|results|matches) found\b[^.]*\b(users?|customers?|members?|clients?|profiles?)\b"
)
#: Any not-found wording, for lookups by phone number.
NOT_FOUND_RE = re.compile(r"(?i)\b(not found|no such|does not exist|no (records?|results?|matches?)( found)?)\b")
#: Recovery hints stop after this many misses until a lookup succeeds.
MAX_RECOVERY_HINTS = 3


@cache
def voice_hint_texts() -> dict[str, str]:
    """Return the fixed hint texts from this example's prompt catalog."""
    entry = load_prompt_catalog(str(Path(__file__).resolve().parent.parent / "pipeline.py")).get("generic_voice_hints")
    messages = entry.get("messages") if isinstance(entry, dict) else None
    if not isinstance(messages, dict):
        raise KeyError("Prompt 'generic_voice_hints' needs a messages mapping")
    return {str(key): " ".join(str(value).split()) for key, value in messages.items()}


@dataclass(slots=True)
class ArgumentScreen:
    """One session's schema rules plus the counters its hints need."""

    rules: SchemaRules = EMPTY_SCHEMA_RULES
    tools_sha256: str = ""
    enabled: bool = True
    phone_hints: bool = True
    _incomplete_asks: dict[tuple[str, str], int] = field(default_factory=dict)
    _lookup_misses: int = 0

    @classmethod
    def for_tools(cls, tools: Sequence[Mapping[str, Any]], *, enabled: bool, phone_hints: bool) -> ArgumentScreen:
        """Build the screen for one set of client tool schemas."""
        listed = list(tools)
        return cls(
            rules=build_schema_rules(listed) if enabled or phone_hints else EMPTY_SCHEMA_RULES,
            tools_sha256=tools_sha256(listed),
            enabled=enabled,
            phone_hints=phone_hints,
        )

    def screen(self, name: str, arguments: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """Return the arguments to send, or a question for the caller when an identifier is unfinished.

        Writes come back as an exact copy.
        """
        if not self.enabled:
            return dict(arguments), None
        result = screen_call(self.rules, ToolCall(name=name, arguments=arguments))
        if not result.incomplete:
            return result.arguments, None
        argument = result.incomplete[0]
        key = (name, argument)
        asked = self._incomplete_asks.get(key, 0)
        self._incomplete_asks[key] = asked + 1
        label = self.rules.labels.get(key) or argument.replace("_", " ")
        texts = voice_hint_texts()
        if asked == 0:
            text = texts["incomplete_read_back"].format(label=label, value=_spoken(result.arguments.get(argument)))
        else:
            text = texts["incomplete_spell_all"].format(label=label)
        payload = response_hint(
            reason="params_invalid",
            action="req_params",
            params_needed=[argument],
            response_text=text,
            context=name,
        )
        payload["answered_locally"] = {"tool": name, "arguments": dict(result.arguments)}
        return result.arguments, payload

    def add_result_hints(self, payload: dict[str, Any], *, is_read: bool) -> None:
        """Add recovery hints to the Thinker's copy of one client result; the client's result is unchanged."""
        if not is_read or not (self.enabled or self.phone_hints):
            return
        data = payload.get("data")
        if not isinstance(data, Mapping) or data.get("owner") != "client":
            return
        status = str(payload.get("status") or "")
        if status == "success" and not _says_not_found(payload, NOT_FOUND_RE):
            self._lookup_misses = 0
            return
        tool = str(payload.get("tool") or "")
        texts = voice_hint_texts()
        arguments = data.get("arguments") if isinstance(data.get("arguments"), Mapping) else {}
        phone_args = self.rules.phone_arguments.get(tool, ())
        if self.phone_hints and phone_args and _says_not_found(payload, NOT_FOUND_RE):
            other = other_phone_form(str(arguments.get(phone_args[0]) or ""))
            if other is not None:
                payload["thinker_hint"] = texts["phone_other_form"].format(other=other)
                return
        if not self.enabled:
            return
        if _says_not_found(payload, PERSON_NOT_FOUND_RE) and self._lookup_misses < MAX_RECOVERY_HINTS:
            self._lookup_misses += 1
            key = "lookup_not_found_first" if self._lookup_misses == 1 else "lookup_not_found_later"
            payload["thinker_hint"] = texts[key]

    @staticmethod
    def repeated_call_hint() -> str:
        """Return the read-back hint attached to a suppressed repeat of a failed call."""
        return voice_hint_texts()["repeated_call"]


def _says_not_found(payload: Mapping[str, Any], pattern: re.Pattern[str]) -> bool:
    data = payload.get("data")
    result = data.get("result") if isinstance(data, Mapping) else None
    text = f"{payload.get('response_text') or ''} {json.dumps(result, ensure_ascii=False, default=str)}"
    return pattern.search(text) is not None


_SYMBOL_WORDS = {"_": "underscore", "-": "dash", "#": "hash", ".": "dot"}


def _spoken(value: object) -> str:
    """Spell an identifier for speech, one character at a time."""
    return " ".join(_SYMBOL_WORDS.get(char, char) for char in str(value or "") if not char.isspace())
