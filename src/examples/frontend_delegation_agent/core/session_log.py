# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Session-level event-log records shared by every head.

The benchmark report tooling joins a run to the server's event log by ``session_start.model``
(the ``pine-`` tag) and reads ``speech_stopped``/``session_end``, so every head writes
these records the same way. Each record carries the head's current audio time.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from examples.frontend_delegation_agent.core.engine.sinks import EventLog
from examples.frontend_delegation_agent.core.voice_config import VoiceConfig
from examples.frontend_delegation_agent.core.wire.session_view import SessionSettings


class SessionLog:
    """Writes one session's records to the shared event log."""

    def __init__(self, event_log: EventLog, session_id: str, audio_now: Callable[[], float]) -> None:
        """Bind the log, the session id and the head's audio clock."""
        self._event_log = event_log
        self._session_id = session_id
        self._audio_now = audio_now

    def write(self, kind: str, **data: Any) -> None:
        """One record, stamped with the current audio time."""
        self._event_log.write(kind, self._session_id, {"audio_ms": int(self._audio_now()), **data})

    def session_start(self, *, model: str, config: VoiceConfig, **extra: Any) -> None:
        """The record the benchmark report joins on (``model`` is the client's ``?model=`` tag)."""
        history = config.agent.backend.conversation_history
        self.write(
            "session_start",
            model=model,
            backend_history=history.effective_include,
            backend_history_guidance=history.resolved_guidance_key,
            normalization=config.normalization.summary(),
            **extra,
        )

    def session_updated(self, settings: SessionSettings) -> None:
        """An accepted ``session.update``."""
        self.write(
            "session_updated",
            input_format=settings.input_format.to_wire(),
            output_format=settings.output_format.to_wire(),
            turn_detection=settings.turn_detection.effective() | {"mode": settings.turn_detection.mode},
            tools=[tool.get("name") for tool in settings.tools],
            instructions_chars=len(settings.instructions),
        )

    def session_end(self) -> None:
        """The session closed."""
        self.write("session_end")
