# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Speaker context state helpers for the transport agent."""

from __future__ import annotations

from pipecat.processors.aggregators.llm_context import LLMContext

from examples.omni_assistant_subagents.subagents.media_analyzer import SPEAKER_STATE_PREFIXES as MEDIA_STATE_PREFIXES
from examples.omni_assistant_subagents.subagents.webcam import SPEAKER_STATE_PREFIXES as WEBCAM_STATE_PREFIXES
from examples.shared.subagents import SPEAKER_CAPABILITIES_PREFIX

SPEAKER_STATE_PREFIXES: tuple[str, ...] = (
    SPEAKER_CAPABILITIES_PREFIX,
    *WEBCAM_STATE_PREFIXES,
    *MEDIA_STATE_PREFIXES,
)


class SpeakerContextManager:
    """Own compact state messages injected into Speaker Omni's context."""

    def __init__(self, *, context: LLMContext) -> None:
        """Initialize the context manager for one voice session."""
        self._context = context

    def set_pinned_state(self, prefix: str, content: str = "") -> None:
        """Pin the latest state message for ``prefix`` after the base prompt (empty ``content`` clears it)."""
        messages = self._context.get_messages()
        base = list(messages[:1])
        pinned: dict[str, dict] = {}
        conversation: list[dict] = []
        for message in messages[1:]:
            text = str(message.get("content") or "") if isinstance(message, dict) else ""
            slot = next((candidate for candidate in SPEAKER_STATE_PREFIXES if text.startswith(candidate)), "")
            if slot:
                pinned[slot] = message
            else:
                conversation.append(message)
        body = content.strip()
        if body:
            pinned[prefix] = {"role": "system", "content": body}
        else:
            pinned.pop(prefix, None)
        ordered = [pinned[candidate] for candidate in SPEAKER_STATE_PREFIXES if candidate in pinned]
        self._context.set_messages([*base, *ordered, *conversation])
