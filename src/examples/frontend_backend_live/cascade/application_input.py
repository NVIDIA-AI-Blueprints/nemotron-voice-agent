# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Input from the application rather than the caller: commentary, thinking and instructions.

* Commentary is a fact the application wants said. It is kept as reference context, phrased by the commentary
  writer and always spoken.
* Thinking is silent context. The talker sees it on later turns, and the next delegation carries it to the backend.
* An instruction changes behavior: it cuts off current speech, joins the talker's instructions and is decided
  on at once.

The protocol layer validates and acknowledges these commands; this class decides what each means for the cascade.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from examples.frontend_backend_live.cascade.conversation import Conversation
from examples.frontend_backend_live.common.history import CLIENT_CONTEXT_PREFIX
from examples.frontend_backend_live.delegation.coordinator import DelegationCoordinator


class ApplicationInput:
    """Applies the application's commentary, thinking and instructions to one call."""

    def __init__(
        self,
        conversation: Conversation,
        coordinator: DelegationCoordinator,
        *,
        phrase: Callable[..., Awaitable[str]],
        wait_for_caller: Callable[[], Awaitable[None]],
        say: Callable[[str], Awaitable[None]],
        act: Callable[..., Awaitable[None]],
        interrupt: Callable[[], Awaitable[None]] | None = None,
    ):
        """Wire the conversation, the delegation coordinator and the session's speech hooks.

        Args:
            conversation: The shared message list.
            coordinator: Source of what the caller was already told, and sink for context the backend should see.
            phrase: Awaited to word an update for speech (the commentary writer).
            wait_for_caller: Awaited until the caller has stopped speaking (bounded).
            say: Awaited with the text to speak.
            act: Awaited to decide a turn, with ``instruction=True`` for an instruction.
            interrupt: Awaited to cut off current speech.
        """
        self.conversation, self.coordinator = conversation, coordinator
        self.phrase, self.wait_for_caller, self.say, self.act, self.interrupt = (
            phrase,
            wait_for_caller,
            say,
            act,
            interrupt,
        )

    async def commentary(self, text: str, delegation_id: str | None) -> None:
        """Keep a fact from the application and say it to the caller, in the writer's wording."""
        self.conversation.add({"role": "user", "content": CLIENT_CONTEXT_PREFIX + text})
        told = self.coordinator.told.pop(delegation_id, None) if delegation_id else None
        speech = await self.phrase(CLIENT_CONTEXT_PREFIX, text, told, answer=False, must_speak=True)
        if speech:
            await self.wait_for_caller()
            await self.say(speech)

    def thinking(self, text: str) -> None:
        """Keep silent context for later decisions and for the next delegation; nothing is spoken."""
        self.conversation.add({"role": "user", "content": CLIENT_CONTEXT_PREFIX + text})
        self.coordinator.remember_context(text)

    async def instruction(self, text: str) -> None:
        """Steer behavior: cut off current speech, add the direction, and decide again."""
        self.conversation.add({"role": "developer", "content": text})
        if self.interrupt:
            await self.interrupt()
        await self.act(text, instruction=True)
