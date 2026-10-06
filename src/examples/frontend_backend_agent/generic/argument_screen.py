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
    IdentifierRule,
    SchemaRules,
    ToolCall,
    build_schema_rules,
    is_incomplete,
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
#: A not-found value is answered with a re-spell question at most this many
#: times per tool, field and value. Its own counter: no other read resets it.
MAX_RESPELL_QUESTIONS = 2


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
    #: tool -> the arguments of its most recent call, when that call found nothing.
    _not_found: dict[str, dict[str, Any]] = field(default_factory=dict)
    _respell_asks: dict[tuple[str, str, str], int] = field(default_factory=dict)

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
        tool = str(payload.get("tool") or "")
        arguments = data.get("arguments") if isinstance(data.get("arguments"), Mapping) else {}
        self._remember_not_found(tool, status, arguments, payload)
        if status == "success" and not _says_not_found(payload, NOT_FOUND_RE):
            self._lookup_misses = 0
            return
        texts = voice_hint_texts()
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

    def _remember_not_found(
        self, tool: str, status: str, arguments: Mapping[str, Any], payload: Mapping[str, Any]
    ) -> None:
        """Keep each tool's latest call when it found nothing; any other outcome of that tool replaces it."""
        if not tool:
            return
        transient = status in {"unavailable", "timeout"}
        pattern = PERSON_NOT_FOUND_RE if PERSON_NOT_FOUND_RE.search(_result_text(payload)) else NOT_FOUND_RE
        if not transient and _says_not_found(payload, pattern):
            self._not_found[tool] = dict(arguments)
        elif not transient:
            self._not_found.pop(tool, None)

    def respell_question(self, hint: Mapping[str, Any], query: str) -> dict[str, Any] | None:
        """Return a re-spell question in place of a schema re-ask after a not-found lookup, or None.

        All must hold, so an unrelated "not found" never becomes a spelling
        request: the hint is about the tool whose latest call found nothing;
        it asks for an argument of that call which the schema marks as an
        identifier; the delegated request does not carry a different value
        for it; and this tool, field and value were asked about fewer than
        :data:`MAX_RESPELL_QUESTIONS` times. The words are code-written.
        """
        if not self.enabled or hint.get("type") != "response_hint":
            return None
        if hint.get("reason") not in {"params_missing", "params_invalid"}:
            return None
        tool = str(hint.get("context") or "")
        failed = self._not_found.get(tool)
        requested = hint.get("params_needed")
        if failed is None or not isinstance(requested, list):
            return None
        for argument in (str(name) for name in requested):
            value = failed.get(argument)
            if not isinstance(value, str) or not value.strip() or not self._is_identifier(tool, argument):
                continue
            if _carries_other_value(query, value, self.rules.rule_for(tool, argument)):
                continue
            key = (tool, argument, _folded(value))
            asked = self._respell_asks.get(key, 0)
            if asked >= MAX_RESPELL_QUESTIONS:
                continue
            self._respell_asks[key] = asked + 1
            label = self.rules.labels.get((tool, argument)) or argument.replace("_", " ")
            texts = voice_hint_texts()
            prefix = texts["not_found_prefix"].format(label=label)
            text = f"{prefix} {texts['incomplete_spell_all'].format(label=label)}"
            return response_hint(
                reason="params_invalid",
                action="req_params",
                params_needed=[argument],
                response_text=text,
                context=tool,
                respell=True,
            )
        return None

    def _is_identifier(self, tool: str, argument: str) -> bool:
        lowered = argument.casefold()
        return (
            self.rules.rule_for(tool, argument) is not None
            or lowered == "id"
            or lowered.endswith("_id")
            or lowered.endswith("_ids")
        )

    @staticmethod
    def repeated_call_hint() -> str:
        """Return the read-back hint attached to a suppressed repeat of a failed call."""
        return voice_hint_texts()["repeated_call"]


def _says_not_found(payload: Mapping[str, Any], pattern: re.Pattern[str]) -> bool:
    return pattern.search(_result_text(payload)) is not None


def _result_text(payload: Mapping[str, Any]) -> str:
    data = payload.get("data")
    result = data.get("result") if isinstance(data, Mapping) else None
    return f"{payload.get('response_text') or ''} {json.dumps(result, ensure_ascii=False, default=str)}"


_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-]*")


def _folded(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").casefold())


def _carries_other_value(query: str, failed: str, rule: IdentifierRule | None) -> bool:
    """Return whether the request names a different value of the identifier's shape than the one that failed.

    Without a schema rule only the failed value itself can be recognized, so
    a request that does not contain it is not taken as a new value.
    """
    if rule is None or _folded(failed) in _folded(query):
        return False
    return any(_shaped_like(rule, token) for token in _TOKEN_RE.findall(query))


def _shaped_like(rule: IdentifierRule, token: str) -> bool:
    """Return whether one written token has the shape of a finished value for ``rule``."""
    if rule.separator:
        return token.count(rule.separator) == rule.segments - 1 and not is_incomplete(rule, token)
    alnum = re.sub(r"[^A-Za-z0-9]", "", token)
    upper = rule.max_length or 64
    return any(char.isdigit() for char in alnum) and rule.min_length <= len(alnum) <= upper


_SYMBOL_WORDS = {"_": "underscore", "-": "dash", "#": "hash", ".": "dot"}


def _spoken(value: object) -> str:
    """Spell an identifier for speech, one character at a time."""
    return " ".join(_SYMBOL_WORDS.get(char, char) for char in str(value or "") if not char.isspace())
