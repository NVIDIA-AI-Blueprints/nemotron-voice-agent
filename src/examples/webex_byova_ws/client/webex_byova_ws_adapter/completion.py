"""Encapsulated assistant-output completion policy."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class BaselineCompletionPolicy:
    """Use Generic's bot-stop event while guaranteeing one FINAL per reply.

    Generic has no authoritative output-drained event. This compatibility
    policy is deliberately isolated so it can be replaced without changing
    Cisco protocol handling.
    """

    reply_open: bool = False
    audio_seen: bool = False
    final_sent: bool = False

    def begin_reply(self) -> None:
        """Open a new logical reply if none is currently active."""
        if not self.reply_open:
            self.reply_open = True
            self.audio_seen = False
            self.final_sent = False

    def note_audio(self) -> None:
        """Record that this reply produced assistant audio."""
        self.begin_reply()
        self.audio_seen = True

    def should_finalize(self, event_type: str) -> bool:
        """Return true once for the accepted baseline completion edge."""
        if event_type != "bot-stopped-speaking" or not self.reply_open or self.final_sent:
            return False
        if not self.audio_seen:
            return False
        self.final_sent = True
        self.reply_open = False
        return True

    def force_finalize(self) -> bool:
        """Finalize an active reply once on timeout or event-only completion."""
        if not self.reply_open or self.final_sent:
            return False
        self.final_sent = True
        self.reply_open = False
        return True
