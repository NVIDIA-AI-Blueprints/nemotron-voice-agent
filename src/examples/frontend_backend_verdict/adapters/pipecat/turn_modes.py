# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in Realtime turn modes: manual ``input_audio_buffer.commit`` and true ``semantic_vad``.

Plan ``misc/frontend-backend-verdict/fba-removal-design-5.8-5.9.md`` (5.8, approved). Built only
when ``FBV_REALTIME_TURN_MODES`` allows ``manual`` or ``semantic_vad`` and the container is not in
eval mode (``switches.realtime_turn_modes``); otherwise the Realtime pipeline is built exactly as
before, with one ``SpeechTimeoutUserTurnStopStrategy``.

:class:`ModeSwitchStopStrategy` is the one stop strategy the user-turn processor sees. It starts
in ``server_vad`` (today's strategy) and a ``session.update`` switches it:

* ``server_vad``: today's speech-timeout strategy ends turns;
* ``semantic_vad``: the shared Smart Turn strategy ends turns (the UI path's model);
* ``manual``: nothing ends a turn except :meth:`commit` (the client's ``input_audio_buffer.commit``);
  the turn-stop watchdog is suspended meanwhile.

Each mode's strategies keep receiving the turn-boundary callbacks, so switching never leaves one
with a stale turn; only the active mode's strategies see frames.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pipecat.frames.frames import Frame
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.turns.types import ProcessFrameResult
from pipecat.turns.user_stop.base_user_turn_stop_strategy import BaseUserTurnStopStrategy

SERVER_VAD = "server_vad"
SEMANTIC_VAD = "semantic_vad"
MANUAL = "manual"
#: Every mode name ``FBV_REALTIME_TURN_MODES`` accepts.
MODES = (SERVER_VAD, MANUAL, SEMANTIC_VAD)
#: The VAD stop (seconds) Smart Turn needs to see a pause early; the browser UI uses the same value.
SMART_TURN_VAD_STOP_SECS = 0.2
#: The turn-stop watchdog while in manual mode: only ``commit`` ends a turn.
MANUAL_STOP_TIMEOUT_SECS = 24 * 3600.0


class ModeSwitchStopStrategy(BaseUserTurnStopStrategy):
    """One stop strategy that delegates to the strategies of the session's current turn mode."""

    def __init__(self, strategies: Mapping[str, Sequence[BaseUserTurnStopStrategy]]) -> None:
        """``strategies`` maps each allowed mode to its stop strategies (``manual`` maps to none)."""
        super().__init__()
        if SERVER_VAD not in strategies:
            raise ValueError("the server_vad strategies are required")
        self._strategies = {mode: list(items) for mode, items in strategies.items()}
        self.mode = SERVER_VAD
        self._user_turn = False
        self._controller: Any = None
        self._default_stop_timeout: float | None = None

    @property
    def allowed(self) -> frozenset[str]:
        """The modes this session may select."""
        return frozenset(self._strategies)

    def _all(self) -> list[BaseUserTurnStopStrategy]:
        return [strategy for items in self._strategies.values() for strategy in items]

    @property
    def resolves_proposed_turn_stop_frames(self) -> bool:
        """Whether the active mode's strategies resolve proposed turn stops."""
        return any(s.resolves_proposed_turn_stop_frames for s in self._strategies[self.mode])

    def add_event_handler(self, event_name: str, handler: Any) -> None:
        """Subscribe to this wrapper (for :meth:`commit`) and to every wrapped strategy."""
        super().add_event_handler(event_name, handler)
        for strategy in self._all():
            strategy.add_event_handler(event_name, handler)

    def bind_controller(self, controller: Any) -> None:
        """The user-turn controller, whose stop watchdog manual mode suspends."""
        self._controller = controller
        self._default_stop_timeout = getattr(controller, "_user_turn_stop_timeout", None)

    def select(self, mode: str) -> str:
        """Switch to ``mode`` if allowed (else ``server_vad``); return the mode now active."""
        mode = mode if mode in self._strategies else SERVER_VAD
        self.mode = mode
        if self._controller is not None and self._default_stop_timeout is not None:
            timeout = MANUAL_STOP_TIMEOUT_SECS if mode == MANUAL else self._default_stop_timeout
            # Pipecat 1.11 reads the timeout on each wait; setting the event restarts the wait.
            self._controller._user_turn_stop_timeout = timeout
            self._controller._user_turn_stop_timeout_event.set()
        return mode

    async def commit(self) -> bool:
        """End the current user turn (manual mode); ``False`` when no turn is in progress."""
        if self.mode != MANUAL or not self._user_turn:
            return False
        await self.trigger_user_turn_stopped()
        return True

    async def setup(self, setup: FrameProcessorSetup) -> None:
        """Set up every wrapped strategy."""
        await super().setup(setup)
        for strategy in self._all():
            await strategy.setup(setup)

    async def cleanup(self) -> None:
        """Clean up every wrapped strategy."""
        await super().cleanup()
        for strategy in self._all():
            await strategy.cleanup()

    async def handle_user_turn_started(self) -> None:
        """A turn started: every wrapped strategy arms for it."""
        self._user_turn = True
        for strategy in self._all():
            await strategy.handle_user_turn_started()

    async def handle_user_turn_stopped(self) -> None:
        """A turn ended (whichever strategy ended it): every wrapped strategy resets."""
        self._user_turn = False
        for strategy in self._all():
            await strategy.handle_user_turn_stopped()

    async def process_frame(self, frame: Frame) -> ProcessFrameResult | None:
        """Only the active mode's strategies see frames."""
        for strategy in self._strategies[self.mode]:
            if await strategy.process_frame(frame) == ProcessFrameResult.STOP:
                return ProcessFrameResult.STOP
        return ProcessFrameResult.CONTINUE
