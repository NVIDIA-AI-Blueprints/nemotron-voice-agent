# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One Realtime session: wire + input audio + ASR + turn manager + agent.

The reader handles client events strictly in arrival order. Appends are decoded
and segmented inline (cheap); each utterance gets its own recognition stream,
finalized on a task chained behind the previous utterance's, so transcripts are
delivered in order while the reader keeps consuming audio.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from loguru import logger

from examples.frontend_backend_verdict.voice.agent.filler import FillerLog, FillerTap, Stamp
from examples.frontend_backend_verdict.voice.agent.port import AgentPort
from examples.frontend_backend_verdict.voice.agent.sinks import SessionRoutingSink
from examples.frontend_backend_verdict.voice.clock import SystemClock, WallClock
from examples.frontend_backend_verdict.voice.config import VoiceConfig
from examples.frontend_backend_verdict.voice.engine.input_path import InputPath
from examples.frontend_backend_verdict.voice.engine.output_path import OutputPath
from examples.frontend_backend_verdict.voice.engine.segmenter import SpeechAudio, SpeechEnd, SpeechStart
from examples.frontend_backend_verdict.voice.engine.turn_api import SessionChange, TurnContext, TurnManagerFactory
from examples.frontend_backend_verdict.voice.engine.turn_manager import TurnManager, UserInput
from examples.frontend_backend_verdict.voice.errors import WireProtocolError
from examples.frontend_backend_verdict.voice.speech.ports import RecognizerStream, SpeechServices
from examples.frontend_backend_verdict.voice.wire import client_events as ce
from examples.frontend_backend_verdict.voice.wire import server_events as ev
from examples.frontend_backend_verdict.voice.wire.ids import new_item_id, new_session_id
from examples.frontend_backend_verdict.voice.wire.response_writer import ConversationOrder
from examples.frontend_backend_verdict.voice.wire.session_view import RealtimeSessionView, SessionDefaults
from examples.frontend_backend_verdict.voice.wire.writer import WireTransport, WireWriter

AgentFactory = Callable[[str], AgentPort]
Receive = Callable[[], Awaitable[str | None]]


@dataclass(slots=True)
class _Utterance:
    item_id: str
    stream: RecognizerStream
    turn_start: Stamp
    interim: str = ""


class RealtimeSession:
    """Serves one WebSocket connection."""

    def __init__(
        self,
        *,
        config: VoiceConfig,
        transport: WireTransport,
        services: SpeechServices,
        agent_factory: AgentFactory | None = None,
        routing_sink: SessionRoutingSink,
        filler_log: FillerLog,
        clock: WallClock | None = None,
        model: str = "",
        session_id: str | None = None,
        show_silent_filler: bool = False,
        turn_manager_factory: TurnManagerFactory | None = None,
    ) -> None:
        """Wire the session; nothing is sent until :meth:`run`.

        With ``turn_manager_factory`` the session builds no agent (``agent_factory`` is
        never called) and ``session.update`` is transactional (plan 4.3 of the
        frontend-delegation-hermes prototype).
        """
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
        self._custom_turns = turn_manager_factory is not None
        if agent_factory is None and turn_manager_factory is None:
            raise ValueError("RealtimeSession needs an agent_factory or a turn_manager_factory")
        self.agent = None if turn_manager_factory is not None else agent_factory(self.session_id)  # type: ignore[misc]
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
        if turn_manager_factory is not None:
            self.turns = turn_manager_factory(
                TurnContext(
                    output=output,
                    emit=self.writer.emit,
                    conversation=self.conversation,
                    settings=lambda: self.view.settings,
                    voice=self._resolve_voice,
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
        else:
            self.turns = self._default_turns(output, config, filler_log, show_silent_filler)
        self._utterance: _Utterance | None = None
        self._asr_chain: asyncio.Task[None] | None = None
        self._configured = False
        self._warned: set[str] = set()

    def _default_turns(
        self, output: OutputPath, config: VoiceConfig, filler_log: FillerLog, show_silent_filler: bool
    ) -> TurnManager:
        return TurnManager(
            agent=self.agent,
            output=output,
            emit=self.writer.emit,
            conversation=self.conversation,
            settings=lambda: self.view.settings,
            voice=self._resolve_voice,
            config=config,
            clock=self._clock,
            audio_now=lambda: self.input.clock.now_ms,
            session_id=self.session_id,
            session_start=self._session_start,
            filler_log=filler_log,
            event_log=self._event_log,
            show_silent_filler=show_silent_filler,
        )

    # -- lifecycle ------------------------------------------------------------------

    async def run(self, receive: Receive) -> None:
        """Serve until the client disconnects (``receive`` returns ``None``)."""
        loop = asyncio.get_running_loop()
        self._routing_sink.register(self.session_id, FillerTap(loop, self.turns.on_filler, clock=self._clock))
        self.writer.start()
        self.writer.emit(ev.session_created(self.view.public()))
        history = self.config.agent.backend.conversation_history
        self._log(
            "session_start",
            model=self.view.model,
            backend_history=history.effective_include,
            backend_history_guidance=history.resolved_guidance_key,
            normalization=self.config.normalization.summary(),
        )
        logger.info(f"[{self.session_id}] session started (model={self.view.model or '-'})")
        if self._custom_turns:
            await self.turns.start()
        timeout = self.config.server.session_update_timeout_s
        try:
            while True:
                try:
                    if not self._configured:
                        message = await asyncio.wait_for(receive(), timeout=timeout)
                    else:
                        message = await receive()
                except TimeoutError:
                    logger.warning(f"[{self.session_id}] no session.update within {timeout}s; using defaults")
                    self._configured = True
                    continue
                if message is None:
                    break
                await self.handle_message(message)
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
        self._log("session_end")
        logger.info(f"[{self.session_id}] session closed")

    async def handle_message(self, message: str | bytes) -> None:
        """Handle one client frame; protocol errors become non-fatal ``error`` events."""
        event_id: str | None = None
        try:
            data = ce.decode_message(message)
            event_id = data.get("event_id") if isinstance(data.get("event_id"), str) else None
            command = ce.parse_client_event(data)
            await self._dispatch(command)
        except WireProtocolError as exc:
            self.writer.emit(ev.error(str(exc), code=exc.code, param=exc.param, client_event_id=event_id))

    # -- dispatch ---------------------------------------------------------------------

    async def _dispatch(self, command: ce.ClientCommand) -> None:
        if isinstance(command, ce.AudioAppend):
            await self._on_append(command.audio)
        elif isinstance(command, ce.SessionUpdate):
            await self._on_session_update(command.session)
        elif isinstance(command, ce.ItemCreate):
            self._on_item_create(command)
        elif isinstance(command, ce.ResponseCreate):
            overrides = {key for key in ("instructions", "tools", "tool_choice") if key in command.overrides}
            if overrides:
                raise WireProtocolError(
                    f"per-response overrides {sorted(overrides)} are not supported",
                    code="unsupported_response_override",
                    param="response",
                )
            self.turns.on_response_create()
        elif isinstance(command, ce.AudioCommit):
            await self._on_commit()
        elif isinstance(command, ce.AudioClear):
            cleared = await self._cancel_utterance()
            self.input.clear()
            if cleared is not None:
                self.turns.on_utterance_dropped(cleared, reason="audio_cleared")
            self.writer.emit(ev.audio_cleared())
        elif isinstance(command, ce.ItemTruncate):
            self.turns.on_truncate(command.item_id, command.content_index, command.audio_end_ms)
        elif isinstance(command, ce.ItemDelete):
            if not self.turns.delete_pending_input(command.item_id):
                raise WireProtocolError(
                    f"item {command.item_id!r} cannot be deleted (only unconsumed user text items can)",
                    code="invalid_value",
                    param="item_id",
                )
            self.writer.emit(ev.item_deleted(command.item_id))
        elif isinstance(command, ce.ResponseCancel):
            await self.turns.on_response_cancel()
        elif isinstance(command, ce.OutputAudioBufferClear):
            await self.turns.on_output_audio_clear()

    async def _on_session_update(self, patch: dict[str, Any]) -> None:
        if self._custom_turns:
            await self._on_session_update_transactional(patch)
            return
        result = self.view.apply(patch)
        settings = result.settings
        for warning in result.warnings:
            if warning not in self._warned:
                self._warned.add(warning)
                logger.warning(f"[{self.session_id}] {warning}")
        if self.config.tools.source == "config" and patch.get("tools"):
            self.writer.emit(
                ev.error(
                    "client tools are ignored: this server runs with tools.source: config (internal tools)",
                    code="client_tools_ignored",
                    param="session.tools",
                )
            )
        first = not self._configured
        if first or result.tools_changed or result.instructions_changed:
            try:
                self.agent.configure(tools=settings.tools, instructions=settings.instructions)
            except Exception as exc:  # noqa: BLE001 - a bad tool schema must not kill the session
                raise WireProtocolError(f"cannot configure the agent: {exc}", param="session") from exc
        self.input.configure(settings.input_format, settings.turn_detection)
        self._configured = True
        self._log(
            "session_updated",
            input_format=settings.input_format.to_wire(),
            output_format=settings.output_format.to_wire(),
            turn_detection=settings.turn_detection.effective() | {"mode": settings.turn_detection.mode},
            tools=[tool.get("name") for tool in settings.tools],
            instructions_chars=len(settings.instructions),
        )
        self.writer.emit(ev.session_updated(self.view.public()))
        if first and self.config.protocol.greeting_enabled:
            self.turns.speak_greeting()

    async def _on_session_update_transactional(self, patch: dict[str, Any]) -> None:
        """Validate on a copy, let the turn manager accept it, then commit (nothing changes on rejection)."""
        preview = self.view.preview(patch)
        result = preview.result
        settings = result.settings
        first = not self._configured
        change = SessionChange(
            first=first, tools_changed=result.tools_changed, instructions_changed=result.instructions_changed
        )
        try:
            await self.turns.on_session_update(settings, change)
        except WireProtocolError:
            raise
        except Exception as exc:  # noqa: BLE001 - a rejected update must not kill the session
            raise WireProtocolError(f"session.update rejected: {exc}", param="session") from exc
        preview.commit()
        for warning in result.warnings:
            if warning not in self._warned:
                self._warned.add(warning)
                logger.warning(f"[{self.session_id}] {warning}")
        self.input.configure(settings.input_format, settings.turn_detection)
        self._configured = True
        self._log(
            "session_updated",
            input_format=settings.input_format.to_wire(),
            output_format=settings.output_format.to_wire(),
            turn_detection=settings.turn_detection.effective() | {"mode": settings.turn_detection.mode},
            tools=[tool.get("name") for tool in settings.tools],
            instructions_chars=len(settings.instructions),
        )
        self.writer.emit(ev.session_updated(self.view.public()))
        if first and self.config.protocol.greeting_enabled:
            self.turns.speak_greeting()

    def _on_item_create(self, command: ce.ItemCreate) -> None:
        item = command.item
        if isinstance(item, ce.FunctionCallOutput):
            self.turns.on_function_output(item.call_id, item.output)
            item_id = item.item_id or new_item_id()
            wire_item = ev.function_call_output_item(item_id, call_id=item.call_id, output=item.output)
        else:
            item_id = item.item_id or new_item_id()
            wire_item = ev.user_text_item(item_id, item.text)
        previous = self.conversation.append(item_id)
        self.writer.emit(ev.item_added(wire_item, previous))
        self.writer.emit(ev.item_done(wire_item, previous))
        if isinstance(item, ce.UserText):
            self.turns.on_user_input(UserInput(text=item.text, item_id=item_id), from_audio=False)

    # -- input audio ------------------------------------------------------------------

    async def _on_append(self, payload: bytes) -> None:
        result = self.input.feed(payload)
        self.turns.on_audio_advance(result.step_ms)
        for event in result.events:
            if isinstance(event, SpeechStart):
                await self._on_speech_start(event)
            elif isinstance(event, SpeechAudio):
                if self._utterance is not None:
                    await self._utterance.stream.push(event.audio)
            elif isinstance(event, SpeechEnd):
                self._on_speech_end(event.end_ms)

    async def _on_speech_start(self, event: SpeechStart) -> None:
        item_id = new_item_id()
        stamp = Stamp.now(self._clock, event.start_ms)
        self.writer.emit(ev.speech_started(int(event.start_ms), item_id))
        self._log("speech_started", item_id=item_id, audio_start_ms=int(event.start_ms))
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
        self._log("speech_stopped", item_id=utterance.item_id, audio_end_ms=int(end_ms))
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
        self._log(
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

    async def _on_commit(self) -> None:
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

    async def _cancel_utterance(self) -> str | None:
        """Cancel the open utterance; its ``item_id``, or ``None`` when there was none."""
        utterance = self._utterance
        if utterance is None:
            return None
        with contextlib.suppress(Exception):
            await utterance.stream.cancel()
        self._utterance = None
        return utterance.item_id

    # -- helpers --------------------------------------------------------------------------

    def _resolve_voice(self) -> str | None:
        requested = self.view.settings.voice
        voice_map = self.config.tts.voice_map
        if requested in voice_map:
            return voice_map[requested] or None
        if requested and requested not in self._warned:
            self._warned.add(requested)
            logger.warning(f"[{self.session_id}] voice {requested!r} not in tts.voice_map; using the catalog default")
        return None

    def _log(self, kind: str, **data: Any) -> None:
        self._event_log.write(kind, self.session_id, {"audio_ms": int(self.input.clock.now_ms), **data})
