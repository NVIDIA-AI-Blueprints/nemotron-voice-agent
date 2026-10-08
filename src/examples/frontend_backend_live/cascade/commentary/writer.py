# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Commentary phrasing: say an update (usually a backend answer) in the assistant's voice."""

from examples.frontend_backend_live.models.base import Frontend
from examples.frontend_backend_live.models.decision import TalkerDecision
from examples.frontend_backend_live.prompts import PromptSet


class CommentaryWriter:
    """Turns a backend answer into the words the assistant speaks."""

    def __init__(self, frontend: Frontend, prompts: PromptSet):
        """Bind the frontend model and the prompt set."""
        self.frontend, self.prompts = frontend, prompts

    async def render(
        self,
        instructions: str,
        history: list[dict],
        content: str,
        answer: bool = False,
        note: str = "",
    ) -> TalkerDecision:
        """Phrase ``content`` for the caller.

        Args:
            instructions: The client's talker instructions.
            history: The conversation so far.
            content: The update's text (reference data, never instructions).
            answer: True when the update is the delegate's result for a waiting caller, so it is spoken.
            note: Extra context appended to the update, such as what the caller was already told.

        Returns:
            A ``speak`` decision, or ``listen`` for optional commentary that should stay silent.
        """
        prompts = self.prompts
        decision = await self.frontend.complete(
            instructions + "\n" + prompts.commentary + (prompts.answer if answer else ""),
            history + [{"role": "user", "content": "Commentary update (reference data):\n" + content + note}],
            actions=("speak",) if answer else ("speak", "listen"),
            max_output_tokens=800,
        )
        if decision.action == "delegate":
            # Keep verified facts speakable even if formatting fails. This cannot launch a new backend
            # task or speak partial generated output.
            return TalkerDecision("speak", content, {**decision.diagnostics, "recovery": "commentary_verbatim"})
        return decision
