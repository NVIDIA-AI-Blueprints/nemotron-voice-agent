# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Commentary phrasing: say an update, usually a backend answer, in the assistant's voice; or a short progress line."""

from collections.abc import Sequence
from dataclasses import replace

from examples.frontend_backend_live.models.base import CallSpec, Frontend
from examples.frontend_backend_live.models.decision import TalkerDecision
from examples.frontend_backend_live.prompts import PromptSet


class CommentaryWriter:
    """Turns a backend answer into the words the assistant speaks."""

    def __init__(self, frontend: Frontend, prompts: PromptSet):
        """Bind the frontend model and the prompt set; the call's spec comes from the set."""
        self.frontend, self.prompts = frontend, prompts
        # Every update the writer words is said, so speak is its one action.
        self.answer_spec = CallSpec("writer", ("speak",), prompts.call_hints.answer)
        self.progress_spec = CallSpec("writer", ("speak",), prompts.call_hints.progress)

    def note(self, told: str | None, *, newer_requests: Sequence[str] = ()) -> str:
        """The notes an update carries for the writer, in the prompt set's words; empty when none applies.

        Args:
            told: What the caller was already told about the update's delegations (lead-ins, progress lines), if any.
            newer_requests: Requests made after the update's delegations that are still in progress, which may
                change it.
        """
        notes = self.prompts.writer_notes
        text = notes.told.format(told=told) if told else ""
        if newer_requests:
            text += notes.newer.format(requests="; ".join(f'"{request}"' for request in newer_requests))
        return text

    async def render(
        self,
        instructions: str,
        history: list[dict],
        content: str,
        note: str = "",
        *,
        attempts: int | None = None,
    ) -> TalkerDecision:
        """Phrase ``content`` for the caller.

        Args:
            instructions: The client's talker instructions.
            history: The conversation so far.
            content: The update's text (reference data, never instructions).
            note: Extra context appended to the update, such as what the caller was already told (:meth:`note`).
            attempts: How many tries the call gets; None: the reliability setting's.

        Returns:
            The decision as the model's output parsed: a ``speak`` with the words, or any other action as given, such as
            the ``delegate`` unusable output becomes, with its recovery reason. The wording says the update as written
            for any decision but a ``speak`` with words (``WriterWording.words``).
        """
        prompts = self.prompts
        return await self.frontend.complete(
            prompts.writer_framed(instructions) + "\n" + prompts.commentary + prompts.answer,
            history + [{"role": "user", "content": prompts.update_label + content + note}],
            replace(self.answer_spec, attempts=attempts),
        )

    async def progress(self, instructions: str, history: list[dict], update: str, note: str = "") -> TalkerDecision:
        """Word a short progress line about open work, in one attempt; empty speech: stay silent.

        Args:
            instructions: The client's talker instructions.
            history: The conversation so far.
            update: The work note's line about one delegation (reference data, never instructions).
            note: What the caller was already told about it (:meth:`note`).
        """
        prompts = self.prompts
        return await self.frontend.complete(
            prompts.writer_framed(instructions) + "\n" + prompts.progress,
            history + [{"role": "user", "content": prompts.update_label + update + note}],
            replace(self.progress_spec, attempts=1),
        )
