# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One τ³-head Realtime session: wire + input audio + ASR + the delegation turn manager.

The reader handles client events strictly in arrival order (the shared
:class:`~examples.frontend_delegation_agent.core.wire.session_protocol.RealtimeProtocol`).
Appends are decoded and segmented inline on the audio clock (cheap); each utterance
gets its own recognition stream, finalized on a task chained behind the previous
utterance's, so transcripts are delivered in order while the reader keeps consuming audio.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from loguru import logger

from _fda_test_head.input_path import InputPath
from _fda_test_head.output_path import OutputPath
from _fda_test_head.ports import RecognizerStream, SpeechServices
from _fda_test_head.segmenter import SpeechAudio, SpeechEnd, SpeechStart
from _fda_test_head.writer import WireTransport, WireWriter
from examples.frontend_delegation_agent.core.clock import SystemClock, WallClock
from examples.frontend_delegation_agent.core.engine.filler import FillerLog, FillerTap, Stamp
from examples.frontend_delegation_agent.core.engine.sinks import SessionRoutingSink
from examples.frontend_delegation_agent.core.errors import WireProtocolError
from examples.frontend_delegation_agent.core.head_api import TurnContext, TurnManagerFactory, UserInput
from examples.frontend_delegation_agent.core.session_log import SessionLog
from examples.frontend_delegation_agent.core.voice_config import VoiceConfig
from examples.frontend_delegation_agent.core.wire import server_events as ev
from examples.frontend_delegation_agent.core.wire.ids import new_item_id, new_session_id
from examples.frontend_delegation_agent.core.wire.response_writer import ConversationOrder
from examples.frontend_delegation_agent.core.wire.session_protocol import RealtimeProtocol
from examples.frontend_delegation_agent.core.wire.session_view import (
    RealtimeSessionView,
    SessionDefaults,
    SessionSettings,
)

Receive = Callable[[], Awaitable[str | None]]


@dataclass(slots=True)
class _Utterance:
    item_id: str
    stream: RecognizerStream
    turn_start: Stamp
    interim: str = ""


class RealtimeSession:
    """Serves one WebSocket connection (implements the routing's ``HeadInput``)."""

    def __init__(
        self,
        *,
        config: VoiceConfig,
        transport: WireTransport,
        services: SpeechServices,
        routing_sink: SessionRoutingSink,
        filler_log: FillerLog,
        turn_manager_factory: TurnManagerFactory,
        clock: WallClock | None = None,
        model: str = "",
        session_id: str | None = None,
        show_silent_filler: bool = False,
    ) -> None:
        """Wire the session; nothing is sent until :meth:`run`."""
        self.config = config
        self.session_id = session_id or new_session_id()
        self._clock = clock or SystemClock()
        self._services = services
        self._routing_sink = routing_sink
        self._event_log = routing_sink.event_log
        self.writer = WireWriter(transport, log_wire=config.logging.log_wire, session_id=self.session_id)
        defaults = SessionDefaults(
            input_format=config.audio.default_input_format,
            output_format=config.audio.default_output_format,
            threshold=config.turn_detection.threshold,
            prefix_padding_ms=config.turn_detection.prefix_padding_ms,
            silence_duration_ms=config.turn_detection.silence_duration_ms,
            honor_client_values=config.turn_detection.honor_client_values,
        )
        self.view = RealtimeSessionView(session_id=self.session_id, model=model, defaults=defaults)
        settings = self.view.settings
        self.input = InputPath(
            engine_rate=config.audio.engine_rate,
            vad=services.vad_factory(config.audio.engine_rate),
            input_format=settings.input_format,
            turn_detection=settings.turn_detection,
            min_speech_ms=config.turn_detection.min_speech_ms,
        )
        self.conversation = ConversationOrder()
        self.log = SessionLog(self._event_log, self.session_id, lambda: self.input.clock.now_ms)
        self.protocol = RealtimeProtocol(
            config=config,
            emit=self.writer.emit,
            view=self.view,
            conversation=self.conversation,
            head=self,
            log=self.log,
            session_id=self.session_id,
        )
        self._session_start = Stamp.now(self._clock, 0.0)
        output = OutputPath(
            synthesizer=services.synthesizer,
            normalizer=services.normalizer,
            sentence_split=config.tts.sentence_split,
            chunk_ms=config.audio.output_chunk_ms,
            pace_output=config.audio.pace_output,
            pace_lead_ms=config.audio.pace_lead_ms,
            monotonic=self._clock.monotonic,
        )
        self.turns = turn_manager_factory(
            TurnContext(
                output=output,
                emit=self.writer.emit,
                conversation=self.conversation,
                settings=lambda: self.view.settings,
                voice=self.protocol.resolve_voice,
                config=config,
                clock=self._clock,
                audio_now=lambda: self.input.clock.now_ms,
                session_id=self.session_id,
                session_start=self._session_start,
                filler_log=filler_log,
                event_log=self._event_log,
                model=model,
                show_silent_filler=show_silent_filler,
            )
        )
        self.protocol.bind(self.turns)
        self._utterance: _Utterance | None = None
        self._asr_chain: asyncio.Task[None] | None = None

    # -- lifecycle ------------------------------------------------------------------

    async def run(self, receive: Receive) -> None:
        """Serve until the client disconnects (``receive`` returns ``None``)."""
        loop = asyncio.get_running_loop()
        self._routing_sink.register(self.session_id, FillerTap(loop, self.turns.on_filler, clock=self._clock))
        self.writer.start()
        self.writer.emit(ev.session_created(self.view.public()))
        self.log.session_start(model=self.view.model, config=self.config)
        logger.info(f"[{self.session_id}] session started (model={self.view.model or '-'})")
        await self.turns.start()
        try:
            await self.protocol.serve(receive)
        finally:
            await self.close()

    async def close(self) -> None:
        """Release everything the session started."""
        self._routing_sink.unregister(self.session_id)
        await self.turns.close()
        if self._utterance is not None:
            with contextlib.suppress(Exception):
                await self._utterance.stream.cancel()
            self._utterance = None
        if self._asr_chain is not None and not self._asr_chain.done():
            self._asr_chain.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._asr_chain
        await self.writer.close()
        self.log.session_end()
        logger.info(f"[{self.session_id}] session closed")

    async def handle_message(self, message: str | bytes) -> None:
        """Handle one client frame (see :meth:`RealtimeProtocol.handle_message`)."""
        await self.protocol.handle_message(message)

    # -- HeadInput ----------------------------------------------------------------------

    async def append(self, audio: bytes) -> None:
        """Decode, segment on the audio clock, and advance the playout cursor."""
        result = self.input.feed(audio)
        self.turns.on_audio_advance(result.step_ms)
        for event in result.events:
            if isinstance(event, SpeechStart):
                await self._on_speech_start(event)
            elif isinstance(event, SpeechAudio):
                if self._utterance is not None:
                    await self._utterance.stream.push(event.audio)
            elif isinstance(event, SpeechEnd):
                self._on_speech_end(event.end_ms)

    async def commit(self) -> None:
        """Manual mode: the buffered audio becomes one utterance; server VAD: close the open one."""
        committed = self.input.commit()
        if isinstance(committed, bytes):
            if not committed:
                raise WireProtocolError(
                    "input audio buffer is empty", code="input_audio_buffer_commit_empty", param=None
                )
            item_id = new_item_id()
            stamp = Stamp.now(self._clock, self.input.clock.now_ms)
            stream = self._services.recognizer.open(sample_rate=self.config.audio.engine_rate)
            await stream.push(committed)
            self._commit_item(item_id)
            utterance = _Utterance(item_id=item_id, stream=stream, turn_start=stamp)
            previous = self._asr_chain
            self._asr_chain = asyncio.create_task(self._finalize(utterance, stamp, previous))
            return
        for event in committed:
            if isinstance(event, SpeechEnd):
                self._on_speech_end(event.end_ms)

    async def clear(self) -> str | None:
        """Cancel the open utterance and drop buffered audio; the dropped item id, if any."""
        cleared = await self._cancel_utterance()
        self.input.clear()
        return cleared

    def configure(self, settings: SessionSettings) -> None:
        """Apply the accepted input format and turn detection."""
        self.input.configure(settings.input_format, settings.turn_detection)

    # -- input audio ------------------------------------------------------------------

    async def _on_speech_start(self, event: SpeechStart) -> None:
        item_id = new_item_id()
        stamp = Stamp.now(self._clock, event.start_ms)
        self.writer.emit(ev.speech_started(int(event.start_ms), item_id))
        self.log.write("speech_started", item_id=item_id, audio_start_ms=int(event.start_ms))
        stream = self._services.recognizer.open(
            sample_rate=self.config.audio.engine_rate,
            on_interim=(lambda text, iid=item_id: self._on_interim(iid, text))
            if self.config.asr.interim_results
            else None,
        )
        self._utterance = _Utterance(item_id=item_id, stream=stream, turn_start=stamp)
        await stream.push(event.audio)
        await self.turns.on_speech_started()

    def _on_interim(self, item_id: str, text: str) -> None:
        utterance = self._utterance
        previous = utterance.interim if utterance is not None and utterance.item_id == item_id else ""
        if text.startswith(previous) and len(text) > len(previous):
            self.writer.emit(ev.input_transcription_delta(item_id, text[len(previous) :]))
            if utterance is not None and utterance.item_id == item_id:
                utterance.interim = text

    def _on_speech_end(self, end_ms: float) -> None:
        utterance = self._utterance
        self._utterance = None
        if utterance is None:
            return
        turn_end = Stamp.now(self._clock, end_ms)
        self.writer.emit(ev.speech_stopped(int(end_ms), utterance.item_id))
        self._commit_item(utterance.item_id)
        self.log.write("speech_stopped", item_id=utterance.item_id, audio_end_ms=int(end_ms))
        previous = self._asr_chain
        self._asr_chain = asyncio.create_task(
            self._finalize(utterance, turn_end, previous), name=f"asr-final-{utterance.item_id}"
        )

    def _commit_item(self, item_id: str) -> None:
        previous = self.conversation.append(item_id)
        self.writer.emit(ev.audio_committed(item_id, previous))
        item = ev.user_audio_item(item_id)
        self.writer.emit(ev.item_added(item, previous))
        self.writer.emit(ev.item_done(item, previous))

    async def _finalize(self, utterance: _Utterance, turn_end: Stamp, previous: asyncio.Task[None] | None) -> None:
        if previous is not None and not previous.done():
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await previous
        try:
            transcript = (await utterance.stream.finish()).strip()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - one failed utterance must not end the session
            logger.error(f"[{self.session_id}] ASR failed for {utterance.item_id}: {exc}")
            self.writer.emit(ev.input_transcription_failed(utterance.item_id, str(exc)))
            self.turns.on_utterance_dropped(utterance.item_id, reason="asr_failed")
            return
        asr_final = Stamp.now(self._clock, self.input.clock.now_ms)
        self.log.write(
            "asr_final",
            item_id=utterance.item_id,
            transcript=transcript,
            latency_ms=int(round((asr_final.mono - turn_end.mono) * 1000)),
        )
        if self.config.protocol.emit_input_transcription:
            self.writer.emit(ev.input_transcription_completed(utterance.item_id, transcript))
        self.turns.on_user_input(
            UserInput(
                text=transcript,
                turn_start=utterance.turn_start,
                turn_end=turn_end,
                asr_final=asr_final,
                item_id=utterance.item_id,
            ),
            from_audio=True,
        )

    async def _cancel_utterance(self) -> str | None:
        """Cancel the open utterance; its ``item_id``, or ``None`` when there was none."""
        utterance = self._utterance
        if utterance is None:
            return None
        with contextlib.suppress(Exception):
            await utterance.stream.cancel()
        self._utterance = None
        return utterance.item_id
