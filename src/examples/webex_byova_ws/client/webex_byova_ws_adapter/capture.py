"""Optional on-disk capture of outbound prompt audio.

Enabled by `BYOVA_CAPTURE_DIR`, this records what a live conversation actually
put on the wire so the exact bytes can be inspected offline or shared with
Cisco when diagnosing playback. It is off by default and never affects the
audio that is sent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from webex_byova_ws_adapter.audio import PROMPT_AUDIO_SUFFIX, WEBEX_RATE


@dataclass(slots=True)
class PromptAudioRecorder:
    """Persist each outbound prompt chunk of one conversation."""

    directory: Path
    profile: str
    _index: int = field(default=0, init=False)

    @classmethod
    def for_conversation(cls, root: str, conversation_id: str, profile: str) -> PromptAudioRecorder | None:
        """Build a recorder under `root`, or None when capture is disabled."""
        if not root:
            return None
        directory = Path(root) / conversation_id
        directory.mkdir(parents=True, exist_ok=True)
        return cls(directory, profile)

    def record(self, pcm: bytes, payload: bytes) -> None:
        """Write one chunk as source PCM16, as sent, and as a manifest entry."""
        self._index += 1
        stem = f"chunk-{self._index:04d}"
        (self.directory / f"{stem}.pcm").write_bytes(pcm)
        (self.directory / f"{stem}.{PROMPT_AUDIO_SUFFIX[self.profile]}").write_bytes(payload)
        entry = {
            "chunk": self._index,
            "pcm_bytes": len(pcm),
            "payload_bytes": len(payload),
            "sample_rate_hertz": WEBEX_RATE,
            "profile": self.profile,
        }
        with (self.directory / "manifest.jsonl").open("a", encoding="utf-8") as manifest:
            manifest.write(json.dumps(entry) + "\n")
