# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A1 / A1-lite: a not-found phone lookup, tried once more with the same digits in the other format.

``tau3-geval-failure-fixes-plan.md`` section 4. :meth:`FormatRetry.candidate` decides, for one call
whose output was a permanent failure, whether a retry applies. All of these must hold:

1. the tool is a read tool with a **genuine phone field** (argument name or schema description,
   :func:`~examples.frontend_delegation_hermes.core.normalization.schema_rules.is_phone_argument`);
   a value that merely has ten digits never qualifies;
2. the value has exactly ten digits: digits only becomes ``XXX-XXX-XXXX``, a value with ``-``,
   ``.``, spaces or parentheses becomes digits only;
3. the argument has no argument rule, the other form is not a known failure, and this value was
   not tried before in the session.

The candidate's ``call`` says who looks it up: the relay itself (``mode: call`` and a tool whose
schema **declares** read-only, idempotent semantics), or the backend after a hint (A1-lite).
Pure: the relay owns the wire, the failed set and the messages.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

from examples.frontend_delegation_hermes.core.normalization.arguments import FailureKey, FormatRetrySettings
from examples.frontend_delegation_hermes.core.text.messages import canonical_json

_SEPARATED = re.compile(r"[\d\-. ()]+")


def other_format(value: str) -> str | None:
    """The same ten digits in the other common format, or ``None`` (any other digit count or character)."""
    if not _SEPARATED.fullmatch(value):
        return None
    digits = re.sub(r"\D", "", value)
    if len(digits) != 10:
        return None
    if digits == value:
        return f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"
    return digits


@dataclass(frozen=True, slots=True)
class Candidate:
    """One format retry: the argument, its two forms and the call in the other form."""

    tool: str
    argument: str
    before: str
    after: str
    arguments: str  # the call's JSON arguments with ``after``
    key: FailureKey  # the retry guard's key of that call
    call: bool  # True: the relay looks it up itself (A1); False: a hint for the backend (A1-lite)


class FormatRetry:
    """Per-session state: which tool arguments qualify, which tools are declared read-only, what was tried."""

    def __init__(
        self,
        settings: FormatRetrySettings,
        *,
        fields: Mapping[str, Iterable[str]],
        read_only_tools: Iterable[str] = (),
    ) -> None:
        """``fields``: ``tool -> phone arguments`` of the session's read tools (from the schemas)."""
        self._settings = settings
        self._fields = {tool: tuple(arguments) for tool, arguments in fields.items()}
        self._read_only = frozenset(read_only_tools)
        self._tried: set[tuple[str, str, str]] = set()

    @property
    def settings(self) -> FormatRetrySettings:
        """The settings this instance was built from."""
        return self._settings

    def candidate(
        self,
        tool: str,
        arguments_json: str,
        *,
        has_rule: Callable[[str, str], bool],
        failed: frozenset[FailureKey],
    ) -> Candidate | None:
        """The retry for a call of ``tool`` that failed permanently, or ``None``; marks the value tried."""
        names = self._fields.get(tool)
        if not names:
            return None
        try:
            arguments = json.loads(arguments_json or "{}")
        except json.JSONDecodeError:
            return None
        if not isinstance(arguments, dict):
            return None
        for name in names:
            value = arguments.get(name)
            if not isinstance(value, str) or has_rule(tool, name):
                continue
            after = other_format(value)
            if after is None:
                continue
            tried = (tool, name, re.sub(r"\D", "", value))
            if tried in self._tried:
                continue
            changed = {**arguments, name: after}
            key = (tool, canonical_json(changed))
            if key in failed:
                continue
            self._tried.add(tried)
            return Candidate(
                tool=tool,
                argument=name,
                before=value,
                after=after,
                arguments=json.dumps(changed, ensure_ascii=False),
                key=key,
                call=self._settings.mode == "call" and tool in self._read_only,
            )
        return None
