# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One Pipecat-adapter Realtime session: routing, event classes, pipeline and turn manager.

The session builds the shared pieces every adapter uses (the Realtime view, the
transactional routing, the session log, a ``TurnContext`` and the delegation turn
manager) around a Pipecat pipeline. It is the routing's ``HeadInput``: appended audio
becomes pipeline input frames; clear and commit act on the pipeline's user turn.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import TYPE_CHECKING, Any

from loguru import logger
from pipecat.frames.frames import Frame
from pipecat.workers.runner import WorkerRunner

from examples.frontend_delegation_hermes.adapters.pipecat.frames import (
    ResponseEventFrame,
    StreamEventFrame,
    UrgentEventFrame,
    event_type,
)
from examples.frontend_delegation_hermes.adapters.pipecat.settings import PipecatHeadSettings
from examples.frontend_delegation_hermes.adapters.pipecat.speech_output import PipecatSpeechOutput
from examples.frontend_delegation_hermes.adapters.pipecat.timing import PlayoutEstimator
from examples.frontend_delegation_hermes.core.audio.formats import AudioFormat
from examples.frontend_delegation_hermes.core.audio.resample import StreamResampler
from examples.frontend_delegation_hermes.core.clock import SystemClock, WallClock
from examples.frontend_delegation_hermes.core.engine.filler import FillerLog, FillerTap, Stamp
from examples.frontend_delegation_hermes.core.engine.sinks import SessionRoutingSink
from examples.frontend_delegation_hermes.core.errors import WireProtocolError
from examples.frontend_delegation_hermes.core.head_api import TurnContext, TurnManagerFactory, TurnManagerLike
from examples.frontend_delegation_hermes.core.session_log import SessionLog
from examples.frontend_delegation_hermes.core.voice_config import VoiceConfig
from examples.frontend_delegation_hermes.core.wire import server_events as ev
from examples.frontend_delegation_hermes.core.wire.ids import new_session_id
from examples.frontend_delegation_hermes.core.wire.response_writer import ConversationOrder
from examples.frontend_delegation_hermes.core.wire.session_protocol import RealtimeProtocol
from examples.frontend_delegation_hermes.core.wire.session_view import (
    RealtimeSessionView,
    SessionDefaults,
    SessionSettings,
)

if TYPE_CHECKING:
    from fastapi import WebSocket

    from examples.frontend_delegation_hermes.adapters.pipecat.pipeline import SessionPipeline, SpeechStack

# Events of the turn manager that must not wait behind agent audio.
_URGENT_TURN_EVENTS = frozenset({"error", "conversation.item.truncated", "conversation.item.deleted"})
# Deltas that belong to audio being played: in-band at the tagger, dropped by an interruption.
_STREAM_EVENTS = frozenset({"response.output_audio_transcript.delta", "response.output_text.delta"})


class PipecatSession:
    """Serves one WebSocket connection with a Pipecat pipeline (implements ``HeadInput``)."""

    def __init__(
        self,
        *,
        config: VoiceConfig,
        settings: PipecatHeadSettings,
        websocket: WebSocket,
        routing_sink: SessionRoutingSink,
        filler_log: FillerLog,
        turn_manager_factory: TurnManagerFactory,
        clock: WallClock | None = None,
        model: str = "",
        session_id: str | None = None,
        show_silent_filler: bool = False,
        speech: SpeechStack | None = None,
    ) -> None:
        """Wire the shared pieces and the pipeline; nothing runs until :meth:`run`.

        ``speech`` replaces Silero and the NVIDIA services (tests use the no-GPU stubs).
        """
        from examples.frontend_delegation_hermes.adapters.pipecat.pipeline import real_speech  # noqa: PLC0415

        self.config = config
        self.settings = settings
        self.session_id = session_id or new_session_id()
        self.clock = clock or SystemClock()
        self._websocket = websocket
        self._routing_sink = routing_sink
        self._event_log = routing_sink.event_log
        self.input_rate = settings.audio.in_sample_rate
        defaults = SessionDefaults(
            input_format=config.audio.default_input_format,
            output_format=config.audio.default_output_format,
            threshold=config.turn_detection.threshold,
            prefix_padding_ms=config.turn_detection.prefix_padding_ms,
            silence_duration_ms=config.turn_detection.silence_duration_ms,
            honor_client_values=config.turn_detection.honor_client_values,
        )
        self.view = RealtimeSessionView(session_id=self.session_id, model=model, defaults=defaults)
        self.conversation = ConversationOrder()
        self.log = SessionLog(self._event_log, self.session_id, lambda: self.pipeline.cursor.now_ms)
        self._queue: asyncio.Queue[Frame | None] = asyncio.Queue()
        self._pending_audio: bytes | None = None
        self._warned_manual = False
        self.pipeline = self._build_pipeline(websocket, speech or real_speech(self))
        self.protocol = RealtimeProtocol(
            config=config,
            emit=self.emit_urgent,
            view=self.view,
            conversation=self.conversation,
            head=self,
            log=self.log,
            session_id=self.session_id,
        )
        self._pump: asyncio.Task[None] | None = None
        self._input_resampler: StreamResampler | None = None
        self._input_rates: tuple[int, int] = (0, 0)
        self._session_start = Stamp.now(self.clock, 0.0)
        output = PipecatSpeechOutput(
            push=self.pipeline.push_to_tts,
            interrupt=self.pipeline.interrupt_output,
            tagger=self.pipeline.tagger,
            sentence_split=config.tts.sentence_split,
        )
        self.turns: TurnManagerLike = turn_manager_factory(
            TurnContext(
                output=output,
                emit=self.emit_turn,
                conversation=self.conversation,
                settings=lambda: self.view.settings,
                voice=self.protocol.resolve_voice,
                config=config,
                clock=self.clock,
                audio_now=lambda: self.pipeline.cursor.now_ms,
                session_id=self.session_id,
                session_start=self._session_start,
                filler_log=filler_log,
                event_log=self._event_log,
                model=model,
                show_silent_filler=show_silent_filler,
            )
        )
        self.protocol.bind(self.turns)
        self.pipeline.bridge.bind(self.turns)
        self.playout = PlayoutEstimator(self.turns.on_audio_advance)

    def _build_pipeline(self, websocket: WebSocket, speech: SpeechStack) -> SessionPipeline:
        """The OpenAI Realtime pipeline on ``websocket`` (a browser UI session builds its own)."""
        from examples.frontend_delegation_hermes.adapters.pipecat.pipeline import SessionPipeline  # noqa: PLC0415

        return SessionPipeline(self, websocket, speech)

    # -- events -------------------------------------------------------------------------

    def emit_urgent(self, event: dict[str, Any]) -> None:
        """Session, input-buffer, user-item and error events: never behind agent audio."""
        self._queue.put_nowait(UrgentEventFrame(message=event))

    def emit_turn(self, event: dict[str, Any]) -> None:
        """The turn manager's events, classified (see ``frames.py``)."""
        kind = event_type(event)
        if kind in _URGENT_TURN_EVENTS:
            self._queue.put_nowait(UrgentEventFrame(message=event))
        elif kind in _STREAM_EVENTS:
            capture = self.pipeline.tagger.capture
            if capture is not None:
                capture.append(StreamEventFrame(message=event))
            else:
                self._queue.put_nowait(StreamEventFrame(message=event))
        else:
            self._queue.put_nowait(ResponseEventFrame(message=event))

    async def _pump_events(self) -> None:
        """Hand events to the output tagger in emission order (it sits right before the transport)."""
        while True:
            frame = await self._queue.get()
            if frame is None:
                return
            try:
                await self.pipeline.tagger.queue_frame(frame)
            except Exception as exc:  # noqa: BLE001 - one lost event must not end the session
                logger.error(f"[{self.session_id}] event dropped: {exc}")

    # -- HeadInput ----------------------------------------------------------------------

    async def on_client_message(self, data: str | bytes) -> bytes | None:
        """Route one client frame; the decoded, resampled audio for an append."""
        self._pending_audio = None
        await self.protocol.handle_message(data)
        audio, self._pending_audio = self._pending_audio, None
        return audio

    async def append(self, audio: bytes) -> None:
        """Decode the client's format and resample to the pipeline input rate."""
        in_format = self.view.settings.input_format
        pcm = in_format.decode(audio)
        rates = (in_format.rate, self.input_rate)
        if rates[0] != rates[1]:
            if self._input_resampler is None or self._input_rates != rates:
                self._input_resampler = StreamResampler(*rates)
                self._input_rates = rates
            pcm = self._input_resampler.process(pcm)
        self._pending_audio = pcm or None

    async def commit(self) -> None:
        """Turns are Pipecat's (server VAD); a manual commit is not supported by this adapter."""
        raise WireProtocolError(
            "input_audio_buffer.commit is not supported: the Pipecat adapter ends turns with its own VAD",
            code="unsupported_event",
            param=None,
        )

    async def clear(self) -> str | None:
        """Drop the open user turn; its item id, if any."""
        return self.pipeline.bridge.clear()

    def configure(self, settings: SessionSettings) -> None:
        """Turn detection stays Pipecat's; the input format is read per append from the view.

        A client asking for manual turns (``turn_detection: null``) is accepted with a warning:
        text items work as usual, audio turns are still ended by Pipecat, and ``commit`` is an error.
        """
        if settings.turn_detection.mode == "manual" and not self._warned_manual:
            self._warned_manual = True
            logger.warning(f"[{self.session_id}] manual turn detection requested; Pipecat's VAD still ends audio turns")
        self._input_resampler = None

    def output_format(self) -> AudioFormat:
        """The client's current output format."""
        return self.view.settings.output_format

    # -- lifecycle ----------------------------------------------------------------------

    async def run(self) -> None:
        """Serve until the client disconnects."""
        loop = asyncio.get_running_loop()
        self._routing_sink.register(self.session_id, FillerTap(loop, self.turns.on_filler, clock=self.clock))
        await self._websocket.send_text(json.dumps(ev.session_created(self.view.public())))
        self.log.session_start(
            model=self.view.model, config=self.config, adapter="pipecat", pipecat=self.settings.summary()
        )
        logger.info(f"[{self.session_id}] session started (model={self.view.model or '-'}, adapter=pipecat)")
        await self.turns.start()
        self._pump = loop.create_task(self._pump_events(), name=f"fdh-events-{self.session_id}")
        runner = WorkerRunner(handle_sigint=False, handle_sigterm=False)
        try:
            await runner.add_workers(self.pipeline.worker)
            await runner.run()
        finally:
            await self.close()

    async def close(self) -> None:
        """Release everything the session started."""
        self._routing_sink.unregister(self.session_id)
        with contextlib.suppress(Exception):
            await self.turns.close()
        await self.pipeline.bridge.aclose()
        await self.playout.aclose()
        if self._pump is not None:
            self._queue.put_nowait(None)
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await asyncio.wait_for(self._pump, timeout=2.0)
        self.log.session_end()
        logger.info(f"[{self.session_id}] session closed")
