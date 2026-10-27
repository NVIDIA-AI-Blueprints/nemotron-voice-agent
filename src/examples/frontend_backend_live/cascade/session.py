# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One call's cascade: its Pipecat-free parts, assembled, and the entry points its host calls.

The host hands it a turn buffer (Pipecat's context), a ``say`` coroutine and the call's delegation part, so the same
parts run in tests, in the live adapter and in the UI bot. A delegation part that reports events (the server
delegation) hands back each delegation's official events by plain calls that only record or queue, so delegated work
never waits for the voice. One that does not (Live client mode) only announces each delegation to the client app,
which answers with commentary.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from loguru import logger

from examples.frontend_backend_live.cascade.agent import LiveAgent
from examples.frontend_backend_live.cascade.appends import ApplicationInput
from examples.frontend_backend_live.cascade.delegations import DelegationReader
from examples.frontend_backend_live.cascade.display import NullDisplay
from examples.frontend_backend_live.cascade.floor import Floor
from examples.frontend_backend_live.cascade.heard_log import HeardLog
from examples.frontend_backend_live.cascade.lines import FixedLines, catalog_lines
from examples.frontend_backend_live.cascade.memory import TalkerMemory, TurnBuffer
from examples.frontend_backend_live.cascade.ports import DisplaySink
from examples.frontend_backend_live.cascade.presenter import Presenter
from examples.frontend_backend_live.cascade.settings import CascadeSettings
from examples.frontend_backend_live.cascade.turns import Acknowledgements, CallerTurns
from examples.frontend_backend_live.cascade.utterances import Playout, Say, Utterances
from examples.frontend_backend_live.cascade.worknote import WorkNote
from examples.frontend_backend_live.config_manager.schema import LiveConfig

if TYPE_CHECKING:
    from live.ports import DelegationPort

__all__ = ["CascadeSession", "log_observation"]

Observe = Callable[..., None]


def log_observation(kind: str, **fields) -> None:
    """Default observation hook: one debug log line per event."""
    logger.debug(f"live.{kind} {fields}")


class CascadeSession:
    """Runs the cascade frontend for one call: its parts, assembled, and the entry points its host calls."""

    def __init__(
        self,
        agent: LiveAgent,
        config: LiveConfig,
        conversation: TurnBuffer,
        *,
        delegation: DelegationPort,
        say: Say,
        interrupt: Callable[[], Awaitable[None]] | None = None,
        observe: Observe = log_observation,
        display: DisplaySink | None = None,
        settings: CascadeSettings | None = None,
        lines: FixedLines | None = None,
        reports_playout: bool = False,
        reports_sentences: bool = False,
        clock: Callable[[], float] = time.monotonic,
    ):
        """Assemble the parts of one call.

        Args:
            agent: The talker and the commentary writer.
            config: The validated example configuration.
            conversation: The host's turn buffer: the seed the talker's memory starts from, then each committed caller
                turn. The cascade only reads it.
            delegation: The call's delegation part.
            say: Awaited with the text to speak.
            interrupt: Awaited to cut off the line in progress (an instruction append).
            observe: Optional ``observe(kind, **fields)`` hook for traces.
            display: Where the display part of each delegation goes; nothing is shown by default.
            settings: The cascade's settings; by default read from ``config``.
            lines: The voice's fixed lines; by default the example's ``prompts.yaml`` lines.
            reports_playout: Whether the host reports each line's playout to :attr:`playout`.
            reports_sentences: Whether that playout reports each sentence's words (per-sentence speech).
            clock: The call's one clock.
        """
        self.agent, self.config, self.observe, self.delegation = agent, config, observe, delegation
        self.settings = settings or CascadeSettings.from_live_config(config)
        self.lines = lines or catalog_lines()
        self.display = display or NullDisplay()
        self.buffer = conversation
        settings = self.settings
        self.memory = TalkerMemory.seeded(
            conversation.messages(), context_tokens=settings.context_tokens, compact_at=settings.compact_at
        )
        self.conversation = self.memory  # what the adapter reads for /history and the context meter
        self.floor = Floor(clock)
        self.heard_log = HeardLog()
        self.utterances = Utterances(
            say,
            self.heard_log,
            self.memory,
            self.floor,
            observe=observe,
            reports_playout=reports_playout,
            reports_sentences=reports_sentences,
            start_timeout_seconds=settings.start_timeout_seconds,
            retry_silent_lines=settings.retry_silent_lines,
        )
        self.worknote = WorkNote(agent.prompts, tracks=delegation.reports_events, clock=clock, observe=observe)
        self.acknowledgements = Acknowledgements(self.utterances, self.lines, observe=observe)
        self.presenter = Presenter.from_settings(
            settings,
            agent=agent,
            memory=self.memory,
            utterances=self.utterances,
            worknote=self.worknote,
            display=self.display,
            lines=self.lines,
            observe=observe,
        )
        self.delegations = DelegationReader(
            self.worknote,
            self.presenter,
            self.display,
            self.lines,
            display_label=settings.display_label,
            app_wait_seconds=settings.app_wait_seconds,
            observe=observe,
            outcome_arrived=self.acknowledgements.drop,
        )
        self.turns = CallerTurns(
            agent,
            config,
            self.memory,
            conversation,
            heard_log=self.heard_log,
            utterances=self.utterances,
            worknote=self.worknote,
            delegation=delegation,
            acknowledgements=self.acknowledgements,
            lines=self.lines,
            display=self.display,
            resume_after_noise=settings.resume_after_noise,
            observe=observe,
        )
        self.application = ApplicationInput(
            self.memory,
            self.heard_log,
            self.presenter,
            self.turns,
            self.acknowledgements,
            self.utterances,
            interrupt=interrupt,
        )
        delegation.listen(self.delegations)

    @property
    def playout(self) -> Playout:
        """What the host's playout tracker reports each line's facts to, in playout order."""
        return self.utterances

    def start(self) -> None:
        """Start the presenter's tasks (delivery and the quiet spell). The host that built the cascade starts it."""
        self.presenter.start()

    async def close(self) -> None:
        """Ignore later delegation events, drop what waits to be said, stop every timer, release the talker's model.

        The host that started the cascade closes it, once.
        """
        self.delegations.close()
        self.acknowledgements.drop_all()
        await self.presenter.close()
        self.utterances.close()
        await self.agent.close()

    def caller_started(self) -> None:
        """The caller started speaking: the voice says nothing on its own meanwhile, and pending acknowledgements go."""
        self.floor.set_caller_speaking(True)
        self.acknowledgements.drop_all()

    def caller_stopped(self) -> None:
        """The caller stopped speaking: their committed turns are heard now, and wait for a decision."""
        self.turns.hear()
        self.floor.set_caller_speaking(False)

    def interrupted(self) -> None:
        """An interruption passed the binding: the lines handed before it are the ones its cut may settle."""
        self.utterances.interrupted()

    async def on_turn(self) -> None:
        """Hear the caller's new turns and decide the ones not decided yet (see :class:`CallerTurns`)."""
        await self.turns.on_turn()

    # ----------------------------------------------------------- client commands (appends)
    async def on_commentary(self, text: str, delegation_id: str | None = None) -> None:
        """Queue a client-supplied fact to be said (see :class:`ApplicationInput`)."""
        self.application.commentary(text, delegation_id)

    def on_thinking(self, text: str, delegation_id: str | None = None) -> None:
        """Retain silent context (see :class:`ApplicationInput`)."""
        self.application.thinking(text)

    async def on_instruction(self, text: str) -> None:
        """Steer behavior (see :class:`ApplicationInput`)."""
        await self.application.instruction(text)

    def on_typed(self, text: str) -> None:
        """The application typed a question itself: the talker's memory holds it; the heard log does not."""
        self.memory.typed(text)

    def greet(self, text: str) -> None:
        """Greet the caller before they speak: ``text`` is pinned as an instruction and worded and said, at idle."""
        self.memory.instruction(text)
        self.presenter.commentary(text)
