# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ``writer`` result wording: one writer call words every ready result into one line.

A call that fails, misses its deadline or says nothing leaves the results said as written (:func:`as_written`), so no
fact is dropped.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence

from loguru import logger

from examples.frontend_backend_live.cascade.ports import ReadyResult, WordingNotes
from examples.frontend_backend_live.cascade.writer.guards import plain_speech
from examples.frontend_backend_live.cascade.writer.writer import CommentaryWriter

# The talker's instructions and voice history as the writer reads them, at the moment it words.
Context = Callable[[], tuple[str, list[dict]]]


def as_written(results: Sequence[ReadyResult]) -> str:
    """The results as the backend wrote them, each as plain spoken sentences, in the order given."""
    return " ".join(line for line in (plain_speech(result.speak) for result in results) if line)


class WriterWording:
    """Words every ready result into one line with one writer call; see the module docstring."""

    def __init__(
        self,
        writer: CommentaryWriter,
        context: Context,
        *,
        timeout_seconds: float,
        observe: Callable[..., None],
    ):
        """Bind the writer and the talker's memory it reads.

        Args:
            writer: The writer (its model, prompt set, notes and specs).
            context: Called once per wording: the talker's instructions and voice history at that moment. The results
                being worded are not in it yet; they enter the talker's memory when their line is handed to speech.
            timeout_seconds: The deadline of the one attempt; past it, the results are said as written.
            observe: ``observe(kind, **fields)`` hook for traces.
        """
        self.writer, self.context = writer, context
        self.timeout_seconds, self.observe = timeout_seconds, observe

    async def words(self, results: Sequence[ReadyResult], notes: WordingNotes) -> str:
        """The one line to say for ``results``, in the order given; empty when there are none."""
        if not results:
            return ""
        delegations = dict.fromkeys(result.delegation_id for result in results)
        told = " ".join(notes.told[delegation] for delegation in delegations if notes.told.get(delegation))
        note = self.writer.note(told or None, newer_requests=notes.newer_requests)
        instructions, history = self.context()
        content = "\n\n".join(result.speak for result in results)
        try:
            async with asyncio.timeout(self.timeout_seconds):
                decision = await self.writer.render(instructions, history, content, answer=True, note=note, attempts=1)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(f"Wording failed ({type(exc).__name__}: {exc}); saying the results as written")
            self.observe("commentary.recovered", error_type=type(exc).__name__, results=len(results))
            return as_written(results)
        if decision.action == "speak" and decision.speech.strip():
            return decision.speech
        # The caller waits on these results: they are said, even only as written.
        self.observe("commentary.answer_forced", talker_action=decision.action, results=len(results))
        return as_written(results)
