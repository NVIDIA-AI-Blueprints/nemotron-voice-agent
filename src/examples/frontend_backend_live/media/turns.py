# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How the caller's turn ends, and when the speech-to-text finalizes what it heard (``speech.turn_end``).

NVIDIA's streaming speech-to-text closes a stretch of speech on its own after about 400 ms of silence
(``stop_history``), or at once when the pipeline sends it ``force_eou``. A final forced early in a pause cuts its
decoder off, so a final forced between the letters of a spelled id loses letters. Each turn end in :data:`TURN_ENDS`
therefore builds the user aggregator's settings (the VAD and the turn strategies) together with a speech-to-text that
finalizes when that turn end needs it:

* :class:`SmartTurn` (the default): the VAD reports a pause after ``vad_stop_seconds``, and Smart Turn judges whether
  the turn is over. The speech-to-text finalizes once Smart Turn judges it over, or once the pause reaches
  ``turn_silence_seconds``; a pause judged unfinished is left to the speech-to-text's own endpointing.
* :class:`Silence`: no Smart Turn. A pause of ``turn_silence_seconds`` ends the turn, and the speech-to-text finalizes
  there.
* :class:`Shared`: the shared pipeline (``examples/shared/pipeline_utils.py``, with its environment switches): Smart
  Turn judges each 0.2 s pause, and the speech-to-text finalizes at every pause.

With ``SmartTurn`` and ``Silence``, a turn whose speech brought no words ends once the speech-to-text has had its budget
for the final, not at the user aggregator's 5 s ``user_turn_stop_timeout``.

``SmartTurn`` and ``Silence`` build their own Silero VAD: ``vad_start_seconds`` of speech mark the caller as speaking.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from loguru import logger
from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VAD_STOP_SECS, VADParams
from pipecat.frames.frames import (
    Frame,
    InterimTranscriptionFrame,
    SystemFrame,
    TranscriptionFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.aggregators.llm_response_universal import LLMUserAggregatorParams
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.stt_latency import DEFAULT_TTFS_P99, NVIDIA_TTFS_P99
from pipecat.turns.types import ProcessFrameResult
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy, TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from examples.shared.nvidia_force_eou_stt import NvidiaForceEouSTTService
from examples.shared.pipeline_utils import build_user_aggregator_params, build_user_mute_strategies

if TYPE_CHECKING:
    from examples.frontend_backend_live.config_manager.schema import SpeechConfig

# The time a final gets after the VAD stop: Pipecat's NVIDIA ttfs_p99_latency less the 0.2 s VAD stop it assumes.
# Pipecat's NVIDIA value is its 1.0 s placeholder for a speech-to-text that runs locally, not a measured p99.
FINAL_BUDGET_SECONDS = NVIDIA_TTFS_P99 - VAD_STOP_SECS


def ttfs_after(stop_secs: float) -> float:
    """The ``ttfs_p99_latency`` that gives the final the same budget after a VAD stop of ``stop_secs``."""
    return stop_secs + FINAL_BUDGET_SECONDS


@dataclass
class STTFinalizeFrame(SystemFrame):
    """Asks the speech-to-text to finalize, now, the caller's speech it has so far.

    A system frame, so it overtakes queued frames: the end of the caller's turn waits for the final it asks for.
    """


class FinalizeOnRequestSTTService(NvidiaForceEouSTTService):
    """The shared force-EOU speech-to-text, finalizing at an :class:`STTFinalizeFrame` instead of at every VAD stop."""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Handle a VAD stop as Pipecat's NVIDIA service does, without the final; finalize when asked."""
        if isinstance(frame, VADUserStoppedSpeakingFrame):
            # The shared service would force a final here; the speech-to-text's own handling of the stop stays.
            await super(NvidiaForceEouSTTService, self).process_frame(frame, direction)
            return
        await super().process_frame(frame, direction)
        if isinstance(frame, STTFinalizeFrame):
            await self.request_force_eou()


class FinalizeAtVerdictUserTurnStopStrategy(TurnAnalyzerUserTurnStopStrategy):
    """Pipecat's Smart Turn stop strategy, which also says when the speech-to-text finalizes.

    Pipecat ends the turn once Smart Turn judges it complete and a final transcript is in. Here the final is asked for
    only then. At a COMPLETE verdict at a VAD stop, the strategy broadcasts an :class:`STTFinalizeFrame`: the
    speech-to-text's partial results trail the audio, so only a final asked for now covers the speech that just ended.
    At Smart Turn's silence fallback it asks only when a partial result has not been finalized since. The turn ends at
    the next final, or, when none comes within the speech-to-text's budget for it (Pipecat's ``ttfs_p99_latency`` less
    the VAD's ``stop_secs``), without one: a turn whose words the speech-to-text missed ends then, not at the user
    aggregator's 5 s ``user_turn_stop_timeout``. The next final ends the turn even when it closes an earlier stretch of
    the caller's speech; a final that comes after it, or after the budget, starts the next turn with its words.
    """

    def __init__(self, **kwargs: Any):
        """Start with no partial result unfinalized and no final asked for."""
        super().__init__(**kwargs)
        self._unfinalized = False  # a partial result came after the last final
        self._judging_stop = False  # the trigger check runs on the verdict at a VAD stop
        self._asked = False  # a final was asked for, and neither it nor the end of its wait has come
        self._final_wait_task: asyncio.Task | None = None

    async def process_frame(self, frame: Frame) -> ProcessFrameResult:
        """Note whether the speech-to-text holds text it has not finalized, then run Pipecat's strategy."""
        if isinstance(frame, InterimTranscriptionFrame):
            self._unfinalized = True
        elif isinstance(frame, TranscriptionFrame) and frame.finalized:
            self._unfinalized = False
            await self._final_arrived()
        return await super().process_frame(frame)

    async def cleanup(self):
        """Stop waiting for a final."""
        await super().cleanup()
        await self._cancel_final_wait()

    async def _discard_pending_end_of_turn(self):
        """The caller spoke again, or the turn ended: no final is awaited for this turn any more."""
        await super()._discard_pending_end_of_turn()
        self._asked = False
        await self._cancel_final_wait()

    async def _handle_vad_user_stopped_speaking(self, frame: VADUserStoppedSpeakingFrame):
        """Judge the pause (Pipecat); a COMPLETE verdict here always asks for a final."""
        self._judging_stop = True
        try:
            await super()._handle_vad_user_stopped_speaking(frame)
        finally:
            self._judging_stop = False

    async def _maybe_trigger_user_turn_stopped(self):
        """End the turn once it is complete and its speech is final; ask for the final when it is not."""
        if not self._turn_complete:
            return
        if self._asked:
            return  # the final, or the end of its wait, ends the turn
        if self._judging_stop or self._unfinalized:
            await self._ask_for_final("COMPLETE at a pause" if self._judging_stop else "a partial is unfinalized")
            return
        if self._timeout_task:
            await self.task_manager.cancel_task(self._timeout_task)
            self._timeout_task = None
        await self.trigger_user_turn_stopped()

    async def _ask_for_final(self, why: str):
        self._asked = True
        budget = max(0.0, (self._stt_timeout or DEFAULT_TTFS_P99) - self._stop_secs)
        logger.debug(f"{self}: asks the speech-to-text for its final ({why}); waits {budget:.2f} s")
        await self.broadcast_frame(STTFinalizeFrame)
        self._final_wait_task = self.task_manager.create_task(self._final_wait(budget), f"{self}::_final_wait")

    async def _final_arrived(self):
        if self._asked:
            self._asked = False
            await self._cancel_final_wait()

    async def _final_wait(self, budget: float):
        try:
            await asyncio.sleep(budget)
        except asyncio.CancelledError:
            return
        finally:
            self._final_wait_task = None  # ending the turn below discards the wait; this task must not cancel itself
        self._asked = False
        self._unfinalized = False  # nothing more is coming for it
        await self._maybe_trigger_user_turn_stopped()

    async def _cancel_final_wait(self):
        if self._final_wait_task:
            await self.task_manager.cancel_task(self._final_wait_task)
            self._final_wait_task = None


# ------------------------------------------------------------------------------------------- the turn ends
class TurnEnd(Protocol):
    """How the caller's turn ends, and when the speech-to-text finalizes (``speech.turn_end``).

    It holds only its seconds: each session gets its own speech-to-text and user-aggregator settings from it.
    """

    def stt(self, **kwargs: Any) -> NvidiaForceEouSTTService:
        """A new speech-to-text from ``build_stt``'s arguments, finalizing as this turn end needs."""

    def user_params(self, welcome_enabled: bool) -> LLMUserAggregatorParams:
        """New user-aggregator settings: the VAD and the turn strategies.

        With ``welcome_enabled``, the shared mute also keeps the caller's audio out until the welcome has played.
        """

    def describe(self) -> dict:
        """The seconds it runs with, as plain JSON for ``GET /v1/live/info``."""


def _vad_start(speech: SpeechConfig, stop: float, stop_key: str) -> float:
    """``speech.vad_start_seconds``, the speech that marks the caller as speaking, paired with the VAD's ``stop``.

    Raises:
        ValueError: The start is longer than that stop (``speech.<stop_key>``).
    """
    if speech.vad_start_seconds > stop:
        raise ValueError(
            f"speech.vad_start_seconds must not be above speech.{stop_key}, the VAD stop it is paired with"
        )
    return speech.vad_start_seconds


class SmartTurn:
    """Smart Turn judges each pause, and the speech-to-text finalizes once it judges the turn over."""

    def __init__(self, speech: SpeechConfig):
        """Read the VAD's start, the pause Smart Turn judges, and the silence that ends a turn whatever it judged.

        Raises:
            ValueError: The judged pause is not shorter than that silence, or the VAD's start is longer than the pause.
        """
        if speech.vad_stop_seconds >= speech.turn_silence_seconds:
            raise ValueError(
                "speech.vad_stop_seconds must be below speech.turn_silence_seconds: Smart Turn judges a pause before "
                "the silence that ends the turn anyway"
            )
        self.vad_start = _vad_start(speech, speech.vad_stop_seconds, "vad_stop_seconds")
        self.vad_stop = speech.vad_stop_seconds
        self.fallback = round(speech.turn_silence_seconds - speech.vad_stop_seconds, 3)  # Smart Turn's, after the stop

    def stt(self, **kwargs: Any) -> NvidiaForceEouSTTService:
        """A speech-to-text that finalizes when asked, its final's budget measured from this stop."""
        return FinalizeOnRequestSTTService(**kwargs, ttfs_p99_latency=ttfs_after(self.vad_stop))

    def user_params(self, welcome_enabled: bool) -> LLMUserAggregatorParams:
        """Silero VAD stopping after the judged pause, and Smart Turn deciding when to ask for the final."""
        analyzer = LocalSmartTurnAnalyzerV3(params=SmartTurnParams(stop_secs=self.fallback))
        return LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(params=VADParams(start_secs=self.vad_start, stop_secs=self.vad_stop)),
            user_mute_strategies=build_user_mute_strategies(welcome_enabled),
            user_turn_strategies=UserTurnStrategies(
                stop=[FinalizeAtVerdictUserTurnStopStrategy(turn_analyzer=analyzer)]
            ),
        )

    def describe(self) -> dict:
        """The VAD's start, the pause Smart Turn judges, and the silence that ends any turn."""
        return {
            "vad_start_seconds": self.vad_start,
            "vad_stop_seconds": self.vad_stop,
            "turn_silence_seconds": round(self.vad_stop + self.fallback, 3),
        }


class Silence:
    """A pause of ``turn_silence_seconds`` ends the turn, and the speech-to-text finalizes there; no Smart Turn.

    Pipecat's own strategy: the VAD's stop is the turn's end. The turn does not wait for words: it ends at the final,
    or once the final has had its budget after the stop, so a turn whose words the speech-to-text missed does not wait
    for the 5 s ``user_turn_stop_timeout``.
    """

    def __init__(self, speech: SpeechConfig):
        """Read the VAD's start and the pause that ends a turn.

        Raises:
            ValueError: The VAD's start is longer than that pause.
        """
        self.stop = speech.turn_silence_seconds
        self.start = _vad_start(speech, self.stop, "turn_silence_seconds")

    def stt(self, **kwargs: Any) -> NvidiaForceEouSTTService:
        """The shared force-EOU speech-to-text (a final at each VAD stop), its budget measured from this stop."""
        return NvidiaForceEouSTTService(**kwargs, ttfs_p99_latency=ttfs_after(self.stop))

    def user_params(self, welcome_enabled: bool) -> LLMUserAggregatorParams:
        """Silero VAD stopping after the pause, and the speech-timeout stop strategy with no wait of its own."""
        return LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(params=VADParams(start_secs=self.start, stop_secs=self.stop)),
            user_mute_strategies=build_user_mute_strategies(welcome_enabled),
            user_turn_strategies=UserTurnStrategies(
                stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0.0, wait_for_transcript=False)]
            ),
        )

    def describe(self) -> dict:
        """The VAD's start, and the pause that ends a turn, which is the VAD's stop."""
        return {"vad_start_seconds": self.start, "vad_stop_seconds": self.stop, "turn_silence_seconds": self.stop}


class Shared:
    """The shared pipeline: Smart Turn at a 0.2 s pause and a final at every pause, with its environment switches."""

    def __init__(self, speech: SpeechConfig):
        """Take none of ``speech``'s seconds: the shared pipeline's environment switches set them."""

    def stt(self, **kwargs: Any) -> NvidiaForceEouSTTService:
        """The shared force-EOU speech-to-text."""
        return NvidiaForceEouSTTService(**kwargs)

    def user_params(self, welcome_enabled: bool) -> LLMUserAggregatorParams:
        """The shared user-aggregator settings."""
        return build_user_aggregator_params(welcome_enabled)

    def describe(self) -> dict:
        """Nothing: the seconds are the shared pipeline's, set by its environment switches."""
        return {}


TURN_ENDS: dict[str, type[TurnEnd]] = {"smart_turn": SmartTurn, "silence": Silence, "shared": Shared}
