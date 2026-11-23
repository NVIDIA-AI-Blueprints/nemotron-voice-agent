# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One session's state at the airline desk, besides its accounts and bookings (``store.DeskStore``)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# The one-time disclosure that the first result of a search or a status carries as ``data_notice``.
DATA_NOTICE = (
    "Demo Air is a made-up airline that flies between twelve U.S. cities; "
    "nothing here is a real flight, and nothing is booked or charged"
)


@dataclass(slots=True)
class PendingChange:
    """A change that was read back to the caller and now waits for their yes.

    ``key`` identifies the exact change: a confirmed call runs only when it resolves to the same key.
    """

    tool: str
    key: tuple[str, ...]


@dataclass(slots=True)
class FlightSearch:
    """The latest flight search: route, day, cabin and up to five flights (the desk's flight objects).

    ``filters`` is what the results were narrowed to, in words, or "" for an unfiltered search.
    """

    origin: dict[str, str]
    destination: dict[str, str]
    date: str
    cabin: str
    sort: str
    results: list[dict[str, Any]]
    filters: str = ""


@dataclass(slots=True)
class DeskState:
    """Mutable state of one session."""

    account_id: str | None = None
    search: FlightSearch | None = None
    pending: PendingChange | None = None
    last_change: dict[str, str] | None = None
    notice_given: bool = False

    def clear_pending(self) -> bool:
        """Drop a change that waits for confirmation; return whether there was one."""
        had = self.pending is not None
        self.pending = None
        return had
