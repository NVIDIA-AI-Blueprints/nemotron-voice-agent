# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Pipecat's user turns, turned into Realtime input events and turn-manager calls.

Ported from FDH ``adapters/pipecat/turn_bridge.py`` (``a0ef50c``, including fix ``1a9f37b``)
and re-pointed at FBA's ``TurnManager``. Pipecat decides when a user turn starts and stops
(VAD plus user-turn strategies, Smart Turn off). The bridge follows those decisions, in v1's
order (plan §4):

* ``UserStartedSpeakingFrame`` (S2): urgent ``input_audio_buffer.speech_started`` with the
  VAD onset on the input-audio clock, then ``on_speech_started`` **at once** (no
  acknowledgement wait), on a task that never blocks the pipeline. Pipecat's own user-turn
  start strategies broadcast the interruption that cuts the agent's audio (B9), gated per
  turn in ``pipeline.py``; the turn manager then decides what happens to the response.
* interim ``InterimTranscriptionFrame``\\ s (S3): a ``.delta`` only when the text extends the
  previous interim, as v1 does.
* final ``TranscriptionFrame``\\ s are collected; ``UserStoppedSpeakingFrame`` (S4) emits
  ``speech_stopped`` and commits the user item at once, as v1 does at the end of speech. The
  stop is a system frame and can overtake the turn's final transcript, so the bridge queues a
  :class:`UserTurnFinalFrame` on itself behind the transcripts already pushed; on that marker
  it logs ``asr_final``, emits the transcription and calls ``on_user_input``, as v1 does after
  its ASR final.
* an ASR error during a turn (S5) ends that turn with ``input_audio_transcription.failed``
  and ``on_utterance_dropped("asr_failed")`` instead of a transcript, as v1 does.
* a TTS error (an upstream ``ErrorFrame`` from the TTS service) fails the item being spoken,
  so the turn manager reports ``response_failed``.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

from loguru import logger
from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from examples.frontend_backend_agent.adapters.pipecat.frames import UserTurnFinalFrame
from examples.frontend_backend_agent.adapters.pipecat.session_log import SessionLog
from examples.frontend_backend_agent.adapters.pipecat.speech_output import OutputTagger
from examples.frontend_backend_agent.adapters.pipecat.timing import InputCursor
from examples.frontend_backend_agent.core.clock import WallClock
from examples.frontend_backend_agent.core.config.voice import VoiceConfig
from examples.frontend_backend_agent.core.engine import server_events as ev
from examples.frontend_backend_agent.core.engine.filler import Stamp
from examples.frontend_backend_agent.core.engine.response_writer import ConversationOrder
from examples.frontend_backend_agent.core.engine.server_events import new_item_id
from examples.frontend_backend_agent.core.engine.turn_api import Emit, TurnManagerLike
from examples.frontend_backend_agent.core.engine.turn_manager import UserInput


@dataclass(slots=True)
class _UserTurn:
    item_id: str
    turn_start: Stamp
    texts: list[str] = field(default_factory=list)
    interim: str = ""
    told_turn_manager: bool = False  # on_speech_started was called for this turn
    asr_error: str = ""  # set when the ASR service failed during this turn (S5)
    started_task: asyncio.Task[None] | None = None
    turn_end: Stamp | None = None  # set when the turn stopped


class FbaTurnBridge(FrameProcessor):
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
        **kwargs: object,
    ) -> None:
        """Bind the session's collaborators; :meth:`bind` adds the turn manager and the TTS service."""
        super().__init__(**kwargs)
        self._config = config
        self._emit = emit_urgent
        self._conversation = conversation
        self._cursor = cursor
        self._tagger = tagger
        self._log = log
        self._clock = clock
        self._session_id = session_id
        self._turns: TurnManagerLike | None = None
        self._tts: FrameProcessor | None = None
        self._turn: _UserTurn | None = None
        self._closing: deque[_UserTurn] = deque()  # stopped turns waiting for their UserTurnFinalFrame
        self._tasks: set[asyncio.Task[None]] = set()

    def bind(self, turns: TurnManagerLike, *, tts: FrameProcessor | None = None) -> None:
        """Attach the session's turn manager and the TTS service whose errors fail the spoken item."""
        self._turns = turns
        self._tts = tts

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Follow Pipecat's user-turn decisions; forward every frame."""
        await super().process_frame(frame, direction)
        if isinstance(frame, ErrorFrame) and direction == FrameDirection.UPSTREAM:
            if self._tts is not None and frame.processor is self._tts:
                logger.warning(f"[{self._session_id}] TTS error: {frame.error}")
                self._tagger.fail_current(str(frame.error))
            await self.push_frame(frame, direction)
            return
        if isinstance(frame, UserStartedSpeakingFrame):
            turn = self._start_turn()
            await self.push_frame(frame, direction)
            turn.started_task = self._spawn(self._tell_speech_started(turn))
        elif isinstance(frame, InterruptionFrame):
            await self.push_frame(frame, direction)
        elif isinstance(frame, UserTurnFinalFrame):
            await self._finalize_turn()
        elif isinstance(frame, TranscriptionFrame):
            turn = self._collecting()
            if turn is not None and frame.text.strip():
                turn.texts.append(frame.text.strip())
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
        self._tagger.interrupted.clear()
        onset_ms, source = self._cursor.take_onset()
        item_id = new_item_id()
        turn = self._turn = _UserTurn(item_id=item_id, turn_start=Stamp.now(self._clock, onset_ms))
        self._emit(ev.speech_started(int(onset_ms), item_id))
        self._log.write("speech_started", item_id=item_id, audio_start_ms=int(onset_ms), onset_source=source)
        return turn

    async def _tell_speech_started(self, turn: _UserTurn) -> None:
        """Call ``on_speech_started`` once per turn (the turn manager counts open utterances)."""
        if turn.told_turn_manager:
            return
        turn.told_turn_manager = True
        if self._turns is not None:
            await self._turns.on_speech_started()

    def _collecting(self) -> _UserTurn | None:
        """The turn a transcript belongs to: the oldest stopped turn not yet final, else the open one."""
        return self._closing[0] if self._closing else self._turn

    def _interim(self, text: str) -> None:
        turn = self._turn
        if turn is None or not self._config.asr.interim_results:
            return
        if text.startswith(turn.interim) and len(text) > len(turn.interim):
            self._emit(ev.input_transcription_delta(turn.item_id, text[len(turn.interim) :]))
            turn.interim = text

    def on_asr_error(self, error: str) -> None:
        """The ASR service failed: the open turn ends without a transcript (S5)."""
        turn = self._collecting()
        if turn is not None:
            turn.asr_error = error or "ASR failed"
        logger.error(f"[{self._session_id}] ASR error: {error}")

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
        turn.turn_end = turn_end
        self._closing.append(turn)
        await self.queue_frame(UserTurnFinalFrame(item_id=turn.item_id))

    async def _finalize_turn(self) -> None:
        """The stopped turn's transcripts are in: hand the turn to the turn manager (v1's ASR final)."""
        if not self._closing:
            return
        turn = self._closing.popleft()
        turn_end = turn.turn_end or Stamp.now(self._clock, self._cursor.now_ms)
        # The turn manager must see on_speech_started before this turn's input or drop.
        if turn.started_task is not None and not turn.started_task.done():
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await turn.started_task
        await self._tell_speech_started(turn)
        if turn.asr_error:
            self._emit(ev.input_transcription_failed(turn.item_id, turn.asr_error))
            if self._turns is not None:
                self._turns.on_utterance_dropped(turn.item_id, reason="asr_failed")
            return
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
        """``input_audio_buffer.clear``: drop the open user turn.

        Returns its item id when the turn manager counted it (``on_speech_started`` ran), so the
        caller reports it dropped. A start not delivered yet is cancelled instead: delivered
        after the drop, it would leave an utterance open and hold every reply.
        """
        turn, self._turn = self._turn, None
        if turn is None:
            return None
        if not turn.told_turn_manager:
            if turn.started_task is not None:
                turn.started_task.cancel()
            return None
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


class AsrErrorTap(FrameProcessor):
    """Just before the STT service: forwards the STT's upstream ``ErrorFrame`` to the bridge (S5)."""

    def __init__(self, on_error: Callable[[str], None], **kwargs: object) -> None:
        """Bind the bridge's ASR error handler; :attr:`stt` is set when the pipeline is built."""
        super().__init__(**kwargs)
        self._on_error = on_error
        self.stt: FrameProcessor | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Report errors that came from the STT service; forward everything."""
        await super().process_frame(frame, direction)
        if (
            isinstance(frame, ErrorFrame)
            and direction == FrameDirection.UPSTREAM
            and self.stt is not None
            and frame.processor is self.stt
        ):
            self._on_error(str(frame.error))
        await self.push_frame(frame, direction)
