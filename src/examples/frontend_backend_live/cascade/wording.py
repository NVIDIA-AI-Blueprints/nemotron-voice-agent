# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The result wordings, by the name the ``result_wording`` setting gives: ``as_written`` and ``writer``."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from examples.frontend_backend_live.cascade.memory import TalkerMemory, split_messages
from examples.frontend_backend_live.cascade.ports import ReadyResult, ResultWording, WordingNotes
from examples.frontend_backend_live.cascade.writer.wording import WriterWording, as_written

if TYPE_CHECKING:
    from examples.frontend_backend_live.cascade.agent import LiveAgent

# Builds a wording from the talker's agent and memory, and the session's trace hook.
WordingFactory = Callable[["LiveAgent", TalkerMemory, Callable[..., None]], ResultWording]


class AsWritten:
    """``as_written``: every ready result's speak part cleaned for speech, in order; instructions say their fallback."""

    async def words(self, results: Sequence[ReadyResult], notes: WordingNotes) -> str:
        """The speak parts as plain spoken sentences (no list markers, headings or emphasis), joined into one line."""
        return as_written(results)

    async def progress(self, update: str, told: str | None, latest: str | None) -> str:
        """No progress line: nothing words one."""
        return ""


def writer_wording(agent: LiveAgent, memory: TalkerMemory, observe: Callable[..., None]) -> WriterWording:
    """``writer``: one writer call for every ready result, reading the talker's memory at the moment it words.

    The results enter the memory only when their line is handed to speech, so the memory read here never holds them.
    """
    return WriterWording(
        agent.writer,
        lambda: split_messages(memory.messages()),
        timeout_seconds=agent.frontend.reliability.talker_attempt_timeout_seconds,
        observe=observe,
    )


# The result wordings by the name the settings give (``result_wording``).
RESULT_WORDINGS: dict[str, WordingFactory] = {
    "writer": writer_wording,
    "as_written": lambda agent, memory, observe: AsWritten(),
}
