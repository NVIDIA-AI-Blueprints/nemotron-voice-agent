# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A recovery note appended to a failed user or customer lookup's result (I3, I4).

One instance per session. It counts misses of the watched lookups (``result_hints.tools``,
or with ``tools: auto`` every read tool of the session, from the tool schemas): an output
matching ``failure_pattern`` (with ``auto``, a not-found text naming a user or customer),
and, with ``count_local_invalid``, a local ``invalid`` answer (with ``auto``, only for an
argument naming a person, such as ``user_id``). The first external miss gets
``message_key`` (I3), later ones ``escalate_message_key`` (I4; "" = the first message
again). At most ``max_hints`` notes until a watched lookup succeeds, which resets the
count. Local answers, transient errors, successes and other tools' outputs are never changed.
With ``field_provenance`` (A3) the relay adds one factual line to the 2nd and later note
(:mod:`.field_provenance`). With ``ledger`` (F8) the relay appends the identification ledger's note
(:mod:`.identity_ledger`), and the count resets only when an identification lookup returns a person
(the relay passes ``reset``), never on an unrelated successful read.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from examples.frontend_delegation_hermes.core.normalization.arguments import ResultHintSettings, render_message
from examples.frontend_delegation_hermes.core.normalization.schema_rules import is_person_argument
from examples.frontend_delegation_hermes.core.tools.identity_ledger import IdentityLedger


@dataclass(frozen=True, slots=True)
class Hint:
    """One appended note."""

    output: str
    message_key: str
    miss: int
    note: str = ""  # the appended text alone


class ResultHints:
    """The per-session miss counter and the note it appends."""

    def __init__(
        self,
        settings: ResultHintSettings,
        *,
        message: str,
        escalate_message: str = "",
        read_tools: Iterable[str] = (),
        ledger: IdentityLedger | None = None,
    ) -> None:
        """``message`` and ``escalate_message`` are the catalog texts of the two keys.

        ``read_tools`` (the session's read tools, from the schemas) are watched when ``tools: auto``.
        """
        self._settings = settings
        self._tools = frozenset(read_tools) if settings.auto_tools else frozenset(settings.tools)
        self._failure = re.compile(settings.failure_pattern)
        self._message = render_message(message)
        self._escalate = render_message(escalate_message) if escalate_message else ""
        self._misses = 0
        self._hints = 0
        #: F8: the session's identification ledger (None when off).
        self.ledger = ledger

    @property
    def field_provenance(self) -> bool:
        """A3: the 2nd and later note also states which values were spelled or only heard."""
        return self._settings.field_provenance

    def watches(self, tool: str) -> bool:
        """Whether ``tool`` is one of the watched lookups."""
        return tool in self._tools

    def on_local_invalid(self, tool: str, argument: str = "") -> None:
        """A local ``invalid`` answer for ``tool`` (never annotated; may count as a miss)."""
        if not (self._settings.count_local_invalid and self.watches(tool)):
            return
        if self._settings.auto_tools and not is_person_argument(argument):
            return
        self._misses += 1

    def on_output(self, tool: str, output: str, *, reset: bool | None = None) -> Hint | None:
        """The client's output of one call; a :class:`Hint` when a note is appended to it.

        ``reset`` (F8): whether a non-failure output resets the count; None keeps today's rule (any
        output of a watched tool that is not an error).
        """
        if not self.watches(tool):
            return None
        if self._failure.search(output):
            self._misses += 1
            if self._hints >= self._settings.max_hints:
                return None
            self._hints += 1
            first = self._misses == 1 or not self._escalate
            text, key = (
                (self._message, self._settings.message_key)
                if first
                else (self._escalate, self._settings.escalate_message_key)
            )
            return Hint(output=f"{output.rstrip()}\n\n{text}", message_key=key, miss=self._misses, note=text)
        if (not output.lstrip().startswith("Error")) if reset is None else reset:
            self._misses = 0
            self._hints = 0
        return None
