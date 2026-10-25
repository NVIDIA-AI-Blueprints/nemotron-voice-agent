# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Pipecat's native user turns, turned into Realtime input events and turn-manager calls.

Pipecat decides when a user turn starts and stops (VAD plus user-turn strategies, Smart
Turn disabled) and when to interrupt. The bridge only follows those decisions:

* ``UserStartedSpeakingFrame``: urgent ``input_audio_buffer.speech_started`` with the VAD
  onset on the input-audio clock; the frame is forwarded unchanged.
* the ``InterruptionFrame`` that follows it is forwarded first; then, in a separate task
  that never blocks the pipeline, the bridge waits (bounded) for the output tagger to
  confirm the agent's audio was cut and calls ``on_speech_started``. A turn that stops (or
  is cleared) before that task runs settles it first, so the turn manager always sees
  ``on_speech_started`` once, before the turn's ``on_user_input``, never after it.
* final ``TranscriptionFrame``\\ s are collected; ``UserStoppedSpeakingFrame`` emits
  ``speech_stopped``, commits the user item, emits the transcription and calls
  ``on_user_input`` exactly as the τ³ head does after its ASR final.

The bridge never starts an interruption of its own for user speech and never holds a
turn open (FDA's spelling hold is off in the Pipecat profiles).
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from dataclasses import dataclass, field

from loguru import logger
from pipecat.frames.frames import (
    Frame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from examples.frontend_delegation_agent.adapters.pipecat.speech_output import OutputTagger
from examples.frontend_delegation_agent.adapters.pipecat.timing import InputCursor
from examples.frontend_delegation_agent.core.clock import WallClock
from examples.frontend_delegation_agent.core.engine.filler import Stamp
from examples.frontend_delegation_agent.core.head_api import Emit, TurnManagerLike, UserInput
from examples.frontend_delegation_agent.core.session_log import SessionLog
from examples.frontend_delegation_agent.core.voice_config import VoiceConfig
from examples.frontend_delegation_agent.core.wire import server_events as ev
from examples.frontend_delegation_agent.core.wire.ids import new_item_id
from examples.frontend_delegation_agent.core.wire.response_writer import ConversationOrder

# How long the bridge waits for the output to be cut before telling the turn manager anyway.
INTERRUPTION_ACK_TIMEOUT_S = 0.5


@dataclass(slots=True)
class _UserTurn:
    item_id: str
    turn_start: Stamp
    texts: list[str] = field(default_factory=list)
    interim: str = ""
    told_turn_manager: bool = False  # on_speech_started was called for this turn
    ack_task: asyncio.Task[None] | None = None  # pending interruption acknowledgement


class FdaTurnBridge(FrameProcessor):
    """Between the user-turn processor and the TTS service: owns the session's user turns."""

    def __init__(
        self,
        *,
        config: VoiceConfig,
        emit_urgent: Emit,
        conversation: ConversationOrder,
        cursor: InputCursor,
        tagger: OutputTagger,
        log: SessionLog,
        clock: WallClock,
        session_id: str,
        interruptions: bool = True,
        **kwargs: object,
    ) -> None:
        """Bind the session's collaborators; :meth:`bind` adds the turn manager."""
        super().__init__(**kwargs)
        self._config = config
        self._emit = emit_urgent
        self._conversation = conversation
        self._cursor = cursor
        self._tagger = tagger
        self._log = log
        self._clock = clock
        self._session_id = session_id
        self._interruptions = interruptions
        self._turns: TurnManagerLike | None = None
        self._turn: _UserTurn | None = None
        self._tasks: set[asyncio.Task[None]] = set()
        self.on_cleared: Callable[[], None] | None = None

    def bind(self, turns: TurnManagerLike) -> None:
        """Attach the session's turn manager."""
        self._turns = turns

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Follow Pipecat's user-turn decisions; forward every frame."""
        await super().process_frame(frame, direction)
        if isinstance(frame, UserStartedSpeakingFrame):
            turn = self._start_turn()
            await self.push_frame(frame, direction)
            if not self._interruptions:
                await self._tell_speech_started(turn)
        elif isinstance(frame, InterruptionFrame):
            await self.push_frame(frame, direction)
            turn = self._turn
            if turn is not None and not turn.told_turn_manager and turn.ack_task is None:
                turn.ack_task = self._spawn(self._after_interruption(turn))
        elif isinstance(frame, TranscriptionFrame):
            if self._turn is not None and frame.text.strip():
                self._turn.texts.append(frame.text.strip())
            await self.push_frame(frame, direction)
        elif isinstance(frame, InterimTranscriptionFrame):
            self._interim(frame.text)
            await self.push_frame(frame, direction)
        elif isinstance(frame, UserStoppedSpeakingFrame):
            await self.push_frame(frame, direction)
            await self._stop_turn()
        else:
            await self.push_frame(frame, direction)

    # -- turn handling ------------------------------------------------------------------

    def _start_turn(self) -> _UserTurn:
        self._tagger.interrupted.clear()  # wait for this turn's interruption, not an earlier one
        onset_ms, source = self._cursor.take_onset()
        item_id = new_item_id()
        turn = self._turn = _UserTurn(item_id=item_id, turn_start=Stamp.now(self._clock, onset_ms))
        self._emit(ev.speech_started(int(onset_ms), item_id))
        self._log.write("speech_started", item_id=item_id, audio_start_ms=int(onset_ms), onset_source=source)
        return turn

    async def _after_interruption(self, turn: _UserTurn) -> None:
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._tagger.interrupted.wait(), timeout=INTERRUPTION_ACK_TIMEOUT_S)
        await self._tell_speech_started(turn)

    async def _tell_speech_started(self, turn: _UserTurn) -> None:
        """Call ``on_speech_started`` once per turn (the turn manager counts open utterances)."""
        if turn.told_turn_manager:
            return
        turn.told_turn_manager = True
        if self._turns is not None:
            await self._turns.on_speech_started()

    @staticmethod
    def _cancel_ack(turn: _UserTurn) -> None:
        """Drop a pending acknowledgement that has not told the turn manager yet."""
        if turn.ack_task is not None and not turn.told_turn_manager:
            turn.ack_task.cancel()

    def _interim(self, text: str) -> None:
        turn = self._turn
        if turn is None or not self._config.asr.interim_results:
            return
        if text.startswith(turn.interim) and len(text) > len(turn.interim):
            self._emit(ev.input_transcription_delta(turn.item_id, text[len(turn.interim) :]))
            turn.interim = text

    async def _stop_turn(self) -> None:
        turn, self._turn = self._turn, None
        if turn is None:
            return
        end_ms = self._cursor.take_offset()
        turn_end = Stamp.now(self._clock, end_ms)
        self._emit(ev.speech_stopped(int(end_ms), turn.item_id))
        previous = self._conversation.append(turn.item_id)
        self._emit(ev.audio_committed(turn.item_id, previous))
        item = ev.user_audio_item(turn.item_id)
        self._emit(ev.item_added(item, previous))
        self._emit(ev.item_done(item, previous))
        self._log.write("speech_stopped", item_id=turn.item_id, audio_end_ms=int(end_ms))
        transcript = " ".join(turn.texts).strip()
        asr_final = Stamp.now(self._clock, self._cursor.now_ms)
        self._log.write(
            "asr_final",
            item_id=turn.item_id,
            transcript=transcript,
            latency_ms=int(round((asr_final.mono - turn_end.mono) * 1000)),
        )
        if self._config.protocol.emit_input_transcription:
            self._emit(ev.input_transcription_completed(turn.item_id, transcript))
        if self._turns is None:
            return
        # A turn Pipecat ended without an interruption frame, or before the interruption was
        # acknowledged: tell the turn manager now, so the acknowledgement never comes after
        # on_user_input and leaves an utterance open (which would hold all output).
        self._cancel_ack(turn)
        await self._tell_speech_started(turn)
        self._turns.on_user_input(
            UserInput(
                text=transcript,
                turn_start=turn.turn_start,
                turn_end=turn_end,
                asr_final=asr_final,
                item_id=turn.item_id,
            ),
            from_audio=True,
        )

    def clear(self) -> str | None:
        """``input_audio_buffer.clear``: drop the open user turn; its item id, if any."""
        turn, self._turn = self._turn, None
        if turn is None:
            return None
        self._cancel_ack(turn)
        return turn.item_id

    def _spawn(self, coroutine: object) -> asyncio.Task[None]:
        task = asyncio.get_running_loop().create_task(coroutine)  # type: ignore[arg-type]
        self._tasks.add(task)
        task.add_done_callback(self._done)
        return task

    def _done(self, task: asyncio.Task[None]) -> None:
        self._tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            logger.error(f"[{self._session_id}] turn bridge task failed: {task.exception()}")

    async def aclose(self) -> None:
        """Cancel pending tasks."""
        for task in list(self._tasks):
            task.cancel()
        for task in list(self._tasks):
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
