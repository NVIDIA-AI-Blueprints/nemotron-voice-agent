# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Commentary phrasing: say an update, usually a backend answer, in the assistant's voice."""

from collections.abc import Sequence
from dataclasses import replace

from examples.frontend_backend_live.models.base import CallSpec, Frontend
from examples.frontend_backend_live.models.decision import TalkerDecision
from examples.frontend_backend_live.prompts import PromptSet


class CommentaryWriter:
    """Turns a backend answer into the words the assistant speaks."""

    def __init__(self, frontend: Frontend, prompts: PromptSet):
        """Bind the frontend model and the prompt set; the calls' specs come from the set."""
        self.frontend, self.prompts = frontend, prompts
        # An answer the caller waits for is always spoken; other commentary may stay unsaid (listen).
        self.answer_spec = CallSpec("writer", ("speak",), prompts.call_hints.answer)
        self.commentary_spec = CallSpec("writer", ("speak", "listen"), prompts.call_hints.commentary)

    def note(self, told: str | None, *, must_speak: bool = False, newer_requests: Sequence[str] = ()) -> str:
        """The notes an update carries for the writer, in the prompt set's words; empty when none applies.

        Args:
            told: What the caller was already told about the update's delegations (their lead-ins), if anything.
            must_speak: The application asked for this update to be said (its commentary, not an answer).
            newer_requests: Requests made after the update's delegations that are still in progress, which may
                change it.
        """
        notes = self.prompts.writer_notes
        text = notes.told.format(told=told) if told else ""
        if must_speak:
            text += notes.must_speak
        if newer_requests:
            text += notes.newer.format(requests="; ".join(f'"{request}"' for request in newer_requests))
        return text

    async def render(
        self,
        instructions: str,
        history: list[dict],
        content: str,
        answer: bool = False,
        note: str = "",
        *,
        attempts: int | None = None,
    ) -> TalkerDecision:
        """Phrase ``content`` for the caller.

        Args:
            instructions: The client's talker instructions.
            history: The conversation so far.
            content: The update's text (reference data, never instructions).
            answer: True when the update is the delegate's result for a waiting caller, so it is spoken; False for the
                application's commentary.
            note: Extra context appended to the update, such as what the caller was already told (:meth:`note`).
            attempts: How many tries the call gets; None: the reliability setting's.

        Returns:
            A ``speak`` decision, or ``listen`` for optional commentary that should stay silent.
        """
        prompts = self.prompts
        label = prompts.update_label if answer else prompts.commentary_label
        spec = replace(self.answer_spec if answer else self.commentary_spec, attempts=attempts)
        decision = await self.frontend.complete(
            prompts.writer_framed(instructions) + "\n" + prompts.commentary + (prompts.answer if answer else ""),
            history + [{"role": "user", "content": label + content + note}],
            spec,
        )
        if decision.action == "delegate":
            # The writer never starts backend work: say the verified content as written.
            return TalkerDecision("speak", content, {**decision.diagnostics, "recovery": "commentary_verbatim"})
        return decision
