# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The backend decision trace: one ``<session_id>.trace.jsonl`` per worker (P0.3 of tau3-geval-failure-fixes-plan.md).

Every backend writes the same records, so traces of different backends can be compared
request by request: a ``session`` record (tool schemas as the model sees them, model,
ephemeral system prompt), then a ``request`` / ``response`` pair per model call. Hermes
feeds it through its lifecycle hooks (``hermes/host.py``); ``LLMAgent`` calls it directly.
Logging only: nothing it does reaches the model or the call. Stdlib only.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

from examples.frontend_delegation_hermes.core.prompt_features import tools_canonical_sha256

#: Bumped when the trace record layout changes.
TRACE_VERSION = 1

logger = logging.getLogger(__name__)


class BackendTrace:
    """One worker's decision trace: a ``session`` record, then a ``request`` / ``response`` pair per model call.

    ``request`` carries the messages exactly as sent (system prompt included) and the hash of the
    tool schemas of the ``session`` record; ``response`` the assistant message (content and tool calls),
    the finish reason and the usage. Records are joined by ``api_request_id``. A write error is logged
    once and the trace stops; it never fails a model call.
    """

    def __init__(self, path: Path, *, session_id: str) -> None:
        """Append to ``path`` (respawned workers of a session append to the same file)."""
        self.path = path
        self._session_id = session_id
        self._lock = threading.Lock()
        self._broken = False
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, kind: str, **fields: Any) -> None:
        """Append one record."""
        if self._broken:
            return
        record = {"v": TRACE_VERSION, "ts": round(time.time(), 3), "kind": kind, "session_id": self._session_id}
        record.update(fields)
        try:
            line = json.dumps(record, ensure_ascii=False, default=_jsonable)
            with self._lock, self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except Exception:  # noqa: BLE001 - tracing must never break a run
            self._broken = True
            logger.exception("backend trace disabled after a write error (%s)", self.path)

    def session(self, *, tools: Any, model: str, instructions: str, request_overrides: Any) -> None:
        """The session's tool schemas (as the model sees them), model and ephemeral system prompt."""
        self.write(
            "session",
            model=model,
            tools=tools,
            tools_sha256=_sha256_json(tools),
            tools_canonical_sha256=tools_canonical_sha256(tools),
            ephemeral_system_prompt=instructions,
            request_overrides=request_overrides,
        )

    def on_request(self, **kwargs: Any) -> None:
        """Hermes ``pre_api_request`` hook."""
        if str(kwargs.get("session_id") or self._session_id) != self._session_id:
            return
        self.write(
            "request",
            api_request_id=kwargs.get("api_request_id"),
            task_id=kwargs.get("task_id"),
            turn_id=kwargs.get("turn_id"),
            api_call_count=kwargs.get("api_call_count"),
            retry_count=kwargs.get("retry_count"),
            model=kwargs.get("model"),
            messages=kwargs.get("request_messages"),
            system_prompt=kwargs.get("system_prompt"),
            tool_count=kwargs.get("tool_count"),
        )

    def on_response(self, **kwargs: Any) -> None:
        """Hermes ``post_api_request`` hook."""
        if str(kwargs.get("session_id") or self._session_id) != self._session_id:
            return
        response = kwargs.get("response") or {}
        self.write(
            "response",
            api_request_id=kwargs.get("api_request_id"),
            task_id=kwargs.get("task_id"),
            turn_id=kwargs.get("turn_id"),
            api_call_count=kwargs.get("api_call_count"),
            finish_reason=kwargs.get("finish_reason"),
            assistant_message=response.get("assistant_message") if isinstance(response, dict) else None,
            usage=kwargs.get("usage"),
            api_duration=kwargs.get("api_duration"),
        )


def _jsonable(value: Any) -> Any:
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump()
    if hasattr(value, "__dict__"):
        return {k: v for k, v in vars(value).items() if not k.startswith("_")}
    return str(value)


def _sha256_json(value: Any) -> str:
    blob = json.dumps(value, sort_keys=True, ensure_ascii=False, default=_jsonable)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
