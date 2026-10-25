# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JSONL event log of the gateway (``gateway.log``; plan section 14)."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any


class GatewayEventLog:
    """Appends ``{"ts", "event", "session_id", ...}`` records (in memory without a path)."""

    def __init__(self, path: str | Path | None) -> None:
        """Open lazily on first write."""
        self._path = Path(path) if path else None
        self._lock = threading.Lock()
        self.records: list[dict[str, Any]] = []
        self.keep_in_memory = self._path is None

    def write(self, event: str, session_id: str | None, data: dict[str, Any]) -> None:
        """Append one record."""
        record = {"ts": round(time.time(), 3), "event": event, "session_id": session_id, **data}
        if self.keep_in_memory:
            self.records.append(record)
            return
        assert self._path is not None  # noqa: S101
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
