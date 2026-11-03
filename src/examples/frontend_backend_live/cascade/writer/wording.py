# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ``writer`` result wording: one writer call words every ready result into one line.

A call that fails, misses its deadline, answers anything but ``speak`` (unusable output parses as ``delegate``), says
nothing, gives the update back unchanged or repeats a line the voice already said leaves the results said as written
(:func:`as_written`), so no fact is dropped; instructions (:class:`Instruction`) are never said, their fallback line
is. When a progress line cannot be worded, nothing is said.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from loguru import logger

from examples.frontend_backend_live.cascade.ports import ReadyResult, WordingNotes
from examples.frontend_backend_live.cascade.writer.plain_speech import plain_speech
from examples.frontend_backend_live.cascade.writer.writer import CommentaryWriter

# The talker's instructions and voice history as the writer reads them, at the moment it words.
Context = Callable[[], tuple[str, list[dict]]]


@dataclass(frozen=True)
class Instruction(ReadyResult):
    """Instructions worded into a line, such as the greeting's; when they cannot be, ``fallback`` is said instead."""

    fallback: str  # the line said in their place: the instructions themselves are never said

    @property
    def unworded(self) -> str:
        """The fallback line."""
        return self.fallback


def as_written(results: Sequence[ReadyResult]) -> str:
    """The results as the backend wrote them, each as plain spoken sentences, in the order given."""
    return " ".join(line for line in (plain_speech(result.unworded) for result in results) if line)


def _settled(text: str) -> str:
    """``text`` cleaned for speech until cleaning changes nothing.

    A cleaning can reveal a list marker (one that emphasis hid, or a line that is only a number), which the next one
    drops. After the first cleaning, each further one only drops characters, so the loop ends.
    """
    while (cleaned := plain_speech(text)) != text:
        text = cleaned
    return text


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
                decision = await self.writer.render(instructions, history, content, note=note, attempts=1)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(f"Wording failed ({type(exc).__name__}: {exc}); saying the results as written")
            self.observe("commentary.recovered", error_type=type(exc).__name__, results=len(results))
            return as_written(results)
        line = decision.speech
        said = {_settled(entry["content"]) for entry in history if entry["role"] == "assistant"}
        # Not worded: any action but speak, no words, the update given back as written or cleaned for speech, or a
        # line the voice already said in its place.
        if (
            decision.action != "speak"
            or not line.strip()
            or _settled(line) == _settled(content)
            or _settled(line) in said
        ):
            # The caller waits on these results: they are said, even only as written.
            self.observe("commentary.answer_forced", talker_action=decision.action, results=len(results))
            return as_written(results)
        return line

    async def progress(self, update: str, told: str | None, latest: str | None) -> str:
        """A progress line for ``update``, the work note's line about one delegation, or none.

        A call that fails, misses its deadline, answers anything but ``speak``, carries a ``recovery`` or says the
        backend's newest progress line ``latest`` as written says nothing: a progress line holds no fact to keep.
        """
        instructions, history = self.context()
        try:
            async with asyncio.timeout(self.timeout_seconds):
                decision = await self.writer.progress(instructions, history, update, note=self.writer.note(told))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(f"Progress wording failed ({type(exc).__name__}: {exc}); saying nothing")
            self.observe("progress.recovered", error_type=type(exc).__name__)
            return ""
        line = decision.speech.strip()
        if (
            decision.action != "speak"
            or decision.diagnostics.get("recovery")
            or (line and latest and _settled(line) == _settled(latest))
        ):
            self.observe("progress.recovered", talker_action=decision.action)
            return ""
        return line
