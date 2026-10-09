# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The delegations of one call: what was asked, what state it is in, and what the frontend may say about it.

The frontend reads this to know what work is still open. It lets the talker see that earlier requests are
in progress, answers "any update?" without starting more work, and lets an answer be phrased knowing a newer
request changes it.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field

OPEN = ("queued", "running")
MAX_KEPT_CLOSED = 32
REQUEST_CHARS = 140

# Short turns that only ask how the work is going. Used only while work is open.
_STATUS_REQUEST = re.compile(
    r"""\b(
        any\s+(update|news|luck|progress|word)
        | (are|is)\s+(you|it|that|this)\s+(still\s+)?(there|working|done|ready|going)
        | you\s+still\s+(there|on\s+it|working)
        | still\s+(there|waiting|working|on\s+it)
        | how\s+(much\s+)?longer
        | how\s+long
        | hello
        | what'?s\s+(taking|the\s+status)
        | (did|have)\s+you\s+(find|get|got|found)\s+(it|anything|that)
        | status
    )\b""",
    re.I | re.X,
)
MAX_STATUS_WORDS = 8

STATUS_LINES = (
    "I'm still working on that, thanks for your patience.",
    "Still on it, one more moment.",
    "Still checking, this is taking a little longer than usual.",
)


def asks_for_status(text: str) -> bool:
    """Whether a short turn only asks how the work is going ("any update?", "hello?", "are you still there?")."""
    words = text.split()
    return 0 < len(words) <= MAX_STATUS_WORDS and bool(_STATUS_REQUEST.search(text))


def humanize(tool_name: str) -> str:
    """A tool name as words: ``search_direct_flight`` -> ``search direct flight``."""
    return re.sub(r"[_\-]+", " ", tool_name).strip()


@dataclass
class Delegation:
    """One delegation: the caller's request and where it stands."""

    id: str
    request: str
    created_at: float
    state: str = "queued"  # queued | running | answered | cancelled
    started_at: float | None = None
    step: str = ""
    progress_lines: int = 0  # status lines spoken while this was the oldest open delegation


@dataclass
class DelegationLedger:
    """All delegations of a call, in the order they were made."""

    clock: Callable[[], float] = field(default_factory=lambda: time.monotonic)
    entries: dict[str, Delegation] = field(default_factory=dict)

    def add(self, delegation_id: str, request: str) -> None:
        """Record a new queued delegation."""
        self.entries[delegation_id] = Delegation(delegation_id, request, self.clock())

    def merge(self, delegation_id: str, request: str) -> None:
        """Add a request that joined a delegation that has not started."""
        entry = self.entries.get(delegation_id)
        if entry:
            entry.request = f"{entry.request} / {request}"

    def start(self, delegation_id: str) -> None:
        """The backend began the delegation."""
        if entry := self.entries.get(delegation_id):
            entry.state, entry.started_at = "running", self.clock()

    def step(self, delegation_id: str, tool_names: list[str]) -> None:
        """Record what a running delegation is doing now."""
        if (entry := self.entries.get(delegation_id)) and tool_names:
            entry.step = ", ".join(humanize(name) for name in tool_names)

    def answered(self, delegation_id: str) -> None:
        """The delegation produced its answer."""
        self._close(delegation_id, "answered")

    def cancelled(self, delegation_id: str) -> None:
        """The delegation will never answer (replaced, failed or abandoned)."""
        self._close(delegation_id, "cancelled")

    def _close(self, delegation_id: str, state: str) -> None:
        if (entry := self.entries.get(delegation_id)) and entry.state in OPEN:
            entry.state = state
        closed = [key for key, e in self.entries.items() if e.state not in OPEN]
        for key in closed[:-MAX_KEPT_CLOSED]:
            del self.entries[key]

    # ------------------------------------------------------------------------------ views
    def open(self) -> list[Delegation]:
        """The delegations still waiting or running, oldest first."""
        return [e for e in self.entries.values() if e.state in OPEN]

    def newest_open(self) -> Delegation | None:
        """The most recent delegation that is still open."""
        open_entries = self.open()
        return open_entries[-1] if open_entries else None

    def open_after(self, delegation_id: str) -> list[Delegation]:
        """The delegations still open that were made after ``delegation_id``."""
        entries = list(self.entries.values())
        ids = [e.id for e in entries]
        if delegation_id not in ids:
            return []
        return [e for e in entries[ids.index(delegation_id) + 1 :] if e.state in OPEN]

    def oldest_open(self) -> Delegation | None:
        """The delegation the caller has waited on longest."""
        open_entries = self.open()
        return open_entries[0] if open_entries else None

    def oldest_open_since(self) -> float | None:
        """When the oldest open delegation was made (the caller has waited since then)."""
        open_entries = self.open()
        return min(e.created_at for e in open_entries) if open_entries else None

    def describe(self) -> str:
        """The open work as a note for the talker, or an empty string when nothing is open."""
        open_entries = self.open()
        if not open_entries:
            return ""
        now = self.clock()
        lines = []
        for number, entry in enumerate(open_entries, 1):
            if entry.state == "running":
                state = f"running for {round(now - (entry.started_at or now))} s"
                if entry.step:
                    state += f", now using {entry.step}"
            else:
                state = "waiting to start"
            request = entry.request if len(entry.request) <= REQUEST_CHARS else entry.request[:REQUEST_CHARS] + "..."
            lines.append(f'{number}. {state}: "{request}"')
        return WORK_IN_PROGRESS.format(items="\n".join(lines))


WORK_IN_PROGRESS = """Work in progress (requests you delegated earlier; each answer is spoken automatically when it
arrives):
{items}
Do not delegate again to ask for progress or to repeat one of these requests. If the caller asks how it is going,
speak a short status."""
