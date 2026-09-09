"""Exactly-once lifecycle for one outbound Cisco reply."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ReplyLifecycle:
    """Track one logical reply without inferring transport completion."""

    active: bool = False
    label: str = ""
    ordinal: int = 0
    audio_seen: bool = False
    final_sent: bool = False

    def begin(self, label: str) -> None:
        """Open a new logical reply."""
        if self.active:
            raise RuntimeError(f"reply {self.ordinal} is already active")
        self.active = True
        self.label = label
        self.ordinal += 1
        self.audio_seen = False
        self.final_sent = False

    def note_audio(self) -> None:
        """Record serialized audio for the active reply."""
        if not self.active:
            raise RuntimeError("audio arrived without an active reply")
        self.audio_seen = True

    def finish(self) -> bool:
        """Mark the active reply final exactly once."""
        if not self.active or self.final_sent:
            return False
        self.final_sent = True
        self.active = False
        return True

    def reset(self) -> None:
        """Clear reply state when the session terminates."""
        self.active = False
        self.label = ""
        self.audio_seen = False
        self.final_sent = False
