# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One Realtime session on the Pipecat adapter: client-message routing, event classes and the turn manager.

The routing is v1's ``RealtimeSession`` (``voice/engine/session.py`` at tag ``fbv-v1-baseline``) without its own
input path and ASR, which Pipecat replaces; the Pipecat wiring is FDH's
``adapters/pipecat/session.py`` (``a0ef50c``). Every rule of v1's session is kept (plan §4):

* S7 per-response overrides rejected; S8 ``client_tools_ignored``; S9 non-transactional
  ``session.update`` (``agent.configure`` on the first update and when tools or instructions
  change); S10 the 30 s first-update timeout logs and continues; S11 the greeting after the
  first update; S13 the filler tap; S14 item create; S16 delete; S17 cancel, clear and
  truncate; S18 protocol errors become non-fatal ``error`` events; S19 the voice map; S21 v1's
  event-log records; S22 v1's close order.
* Input audio is decoded and resampled exactly as v1's ``InputPath.feed`` (B6), and each
  append's audio-clock step travels with its frame (S1).
* Manual ``input_audio_buffer.commit`` is not supported (S15, B7): Pipecat ends audio turns.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from loguru import logger
from pipecat.frames.frames import Frame
from pipecat.workers.runner import WorkerRunner

from examples.frontend_backend_verdict.adapters.pipecat import client_events as ce
from examples.frontend_backend_verdict.adapters.pipecat import switches, turn_modes
from examples.frontend_backend_verdict.adapters.pipecat.frames import (
    ResponseEventFrame,
    StreamEventFrame,
    UrgentEventFrame,
    event_type,
)
from examples.frontend_backend_verdict.adapters.pipecat.pipeline import SessionPipeline, real_speech
from examples.frontend_backend_verdict.adapters.pipecat.session_log import SessionLog
from examples.frontend_backend_verdict.adapters.pipecat.speech_output import PipecatSpeechOutput
from examples.frontend_backend_verdict.core.audio.formats import AudioFormat
from examples.frontend_backend_verdict.core.audio.resample import StreamResampler
from examples.frontend_backend_verdict.core.clock import SystemClock, WallClock
from examples.frontend_backend_verdict.core.engine import server_events as ev
from examples.frontend_backend_verdict.core.engine.filler import FillerTap, Stamp
from examples.frontend_backend_verdict.core.engine.response_writer import ConversationOrder
from examples.frontend_backend_verdict.core.engine.server_events import new_item_id, new_session_id
from examples.frontend_backend_verdict.core.engine.session_settings import RealtimeSessionView, SessionDefaults
from examples.frontend_backend_verdict.core.engine.turn_manager import TurnManager, UserInput
from examples.frontend_backend_verdict.core.errors import WireProtocolError

if TYPE_CHECKING:
    from fastapi import WebSocket

    from examples.frontend_backend_verdict.adapters.pipecat.pipeline import SpeechStack
    from examples.frontend_backend_verdict.adapters.pipecat.runtime import PipecatRuntime, SpeechFactory

# Turn-manager events that must not wait behind agent audio.
_URGENT_TURN_EVENTS = frozenset({"error", "conversation.item.truncated", "conversation.item.deleted"})
# Deltas that belong to audio being played: in-band at the tagger, dropped by an interruption.
_STREAM_EVENTS = frozenset({"response.output_audio_transcript.delta", "response.output_text.delta"})


class PipecatSession:
    """Serves one WebSocket connection with the Pipecat pipeline and FBV's turn manager."""

    def __init__(
        self,
        *,
        runtime: PipecatRuntime,
        websocket: WebSocket,
        model: str = "",
        show_silent_filler: bool = False,
        speech: SpeechFactory | None = None,
        clock: WallClock | None = None,
        session_id: str | None = None,
    ) -> None:
        """Wire the session; nothing runs until :meth:`run`."""
        config = runtime.config
        self.config = config
        self.settings = runtime.settings
        self._runtime_prompts = runtime.prompts
        self.session_id = session_id or new_session_id()
        self._mcp: Any = None  # opt-in MCP tools (mcp_tools.py), set once the turn manager exists
        self.clock = clock or SystemClock()
        self._websocket = websocket
        self._routing_sink = runtime.state.routing_sink
        self._event_log = self._routing_sink.event_log
        self.input_rate = self.settings.audio.in_sample_rate
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
        self.agent = runtime.agent_factory(self.session_id)
        self._queue: asyncio.Queue[Frame | None] = asyncio.Queue()
        self.pipeline = self._build_pipeline(websocket, (speech or real_speech)(self))
        self._session_start = Stamp.now(self.clock, 0.0)
        output = PipecatSpeechOutput(
            push=self.pipeline.push_to_tts,
            interrupt=self.pipeline.interrupt_output,
            tagger=self.pipeline.tagger,
            sentence_split=config.tts.sentence_split,
        )
        # Built exactly as v1's RealtimeSession._default_turns builds it (plan §2).
        self.turns = TurnManager(
            agent=self.agent,
            output=output,
            emit=self.emit_turn,
            conversation=self.conversation,
            settings=lambda: self.view.settings,
            voice=self._resolve_voice,
            config=config,
            clock=self.clock,
            audio_now=lambda: self.pipeline.cursor.now_ms,
            session_id=self.session_id,
            session_start=self._session_start,
            filler_log=runtime.state.filler_log,
            event_log=self._event_log,
            show_silent_filler=show_silent_filler,
        )
        self._mcp = self._create_mcp()
        self.pipeline.bridge.bind(self.turns, tts=self.pipeline.tts)
        self.pipeline.cursor.on_advance = self.turns.on_audio_advance
        self.pipeline.apply_turn_detection(self.view.settings.turn_detection)
        self._format = self.view.settings.input_format
        self._resampler = StreamResampler(self._format.rate, self.input_rate)
        self._partial = b""
        self._pending_step = 0.0
        self._pending_audio: tuple[bytes, float] | None = None
        self._configured = False
        self._warned: set[str] = set()
        self._pump: asyncio.Task[None] | None = None
        self._update_timer: asyncio.Task[None] | None = None

    def _build_pipeline(self, websocket: WebSocket, speech: SpeechStack) -> SessionPipeline:
        """The Realtime pipeline (the browser UI session builds its own, see ``ui_session.py``)."""
        return SessionPipeline(self, websocket, speech)

    # -- events -------------------------------------------------------------------------

    def interruptions_allowed(self) -> bool:
        """Whether a user turn interrupts the agent's audio: the turn manager's own rule (B9)."""
        return self.config.barge_in.enabled and self.view.settings.turn_detection.interrupt_response

    def emit_urgent(self, event: dict[str, Any]) -> None:
        """Session, input-buffer, user-item and error events: never behind agent audio (B10)."""
        if self._mcp is not None and self._mcp.running and event.get("type") == "input_audio_buffer.committed":
            self._mcp.cancel_all("the caller spoke")  # barge-in: stop running MCP calls
        self._queue.put_nowait(UrgentEventFrame(message=event))

    def emit_turn(self, event: dict[str, Any]) -> None:
        """The turn manager's events, classified (see ``frames.py``)."""
        if self._mcp is not None:
            # Opt-in MCP: a hidden call's events never reach the client (mcp_tools.py).
            for visible in self._mcp.outbound(event):
                self._emit_turn_event(visible)
            return
        self._emit_turn_event(event)

    def _emit_turn_event(self, event: dict[str, Any]) -> None:
        self.pipeline.tagger.note_event(event)
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

    # -- client messages (v1's routing) --------------------------------------------------

    async def on_client_message(self, data: str | bytes) -> tuple[bytes, float] | None:
        """Route one client message; the decoded audio and its audio-clock step for an append."""
        self._pending_audio = None
        await self.handle_message(data)
        audio, self._pending_audio = self._pending_audio, None
        return audio

    async def handle_message(self, message: str | bytes) -> None:
        """Handle one client message; protocol errors become non-fatal ``error`` events (S18)."""
        if not self._configured:
            self._restart_update_timer()
        event_id: str | None = None
        try:
            data = ce.decode_message(message)
            event_id = data.get("event_id") if isinstance(data.get("event_id"), str) else None
            command = ce.parse_client_event(data)
            await self._dispatch(command)
        except WireProtocolError as exc:
            self.emit_urgent(ev.error(str(exc), code=exc.code, param=exc.param, client_event_id=event_id))

    async def _dispatch(self, command: ce.ClientCommand) -> None:
        if isinstance(command, ce.AudioAppend):
            self._on_append(command.audio)
        elif isinstance(command, ce.SessionUpdate):
            session = command.session
            if self._mcp is not None:
                session = await self._mcp.prepare_update(session)
            self._on_session_update(session)
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
            modes = self.pipeline.turn_modes
            if modes is None or modes.mode != turn_modes.MANUAL:
                raise WireProtocolError(
                    "input_audio_buffer.commit is not supported: the Pipecat adapter ends turns with its own VAD",
                    code="unsupported_event",
                    param=None,
                )
            if not await modes.commit():
                raise WireProtocolError(
                    "cannot commit: the input audio buffer holds no speech",
                    code="input_audio_buffer_commit_empty",
                    param=None,
                )
        elif isinstance(command, ce.AudioClear):
            cleared = self.pipeline.bridge.clear()
            self._clear_input()
            if cleared is not None:
                self.turns.on_utterance_dropped(cleared, reason="audio_cleared")
            self.emit_urgent(ev.audio_cleared())
        elif isinstance(command, ce.ItemTruncate):
            self.turns.on_truncate(command.item_id, command.content_index, command.audio_end_ms)
        elif isinstance(command, ce.ItemDelete):
            if not self.turns.delete_pending_input(command.item_id):
                raise WireProtocolError(
                    f"item {command.item_id!r} cannot be deleted (only unconsumed user text items can)",
                    code="invalid_value",
                    param="item_id",
                )
            self.emit_urgent(ev.item_deleted(command.item_id))
        elif isinstance(command, ce.ResponseCancel):
            if self._mcp is not None and self._mcp.running:
                self._mcp.cancel_all("response.cancel")
            await self._stopping_output(self.turns.on_response_cancel)
        elif isinstance(command, ce.OutputAudioBufferClear):
            await self._stopping_output(self.turns.on_output_audio_clear)

    async def _stopping_output(self, handler: Callable[[], Awaitable[None]]) -> None:
        """Run a client stop (S17); flush Pipecat's output if the turn manager dropped unplayed audio.

        v1 had written all synthesized audio by the time a response was done, so dropping the
        unplayed rest (``ResponseProgress.discard_unplayed``) needed no output action. Here that
        audio can still be queued in Pipecat's transport, and it must not reach the client.
        """
        before = self.turns.progress.sent_ms
        await handler()
        if self.turns.progress.sent_ms < before:
            await self.pipeline.interrupt_output()

    def _on_session_update(self, patch: dict[str, Any]) -> None:
        """v1's non-transactional ``session.update`` (S8, S9, S11)."""
        result = self.view.apply(patch)
        settings = result.settings
        for warning in result.warnings:
            if warning not in self._warned:
                self._warned.add(warning)
                logger.warning(f"[{self.session_id}] {warning}")
        if self.config.tools.source == "config" and patch.get("tools"):
            self.emit_urgent(
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
        self._configure_input(settings.input_format)
        active = self._select_turn_mode(settings.turn_detection)
        stop_secs = turn_modes.SMART_TURN_VAD_STOP_SECS if active == turn_modes.SEMANTIC_VAD else None
        self.pipeline.apply_turn_detection(settings.turn_detection, stop_secs=stop_secs)
        if settings.turn_detection.mode == "manual" and active != turn_modes.MANUAL and "manual" not in self._warned:
            self._warned.add("manual")
            logger.warning(f"[{self.session_id}] manual turn detection requested; Pipecat's VAD still ends audio turns")
        self._configured = True
        self.log.write(
            "session_updated",
            input_format=settings.input_format.to_wire(),
            output_format=settings.output_format.to_wire(),
            turn_detection=settings.turn_detection.effective() | {"mode": settings.turn_detection.mode},
            tools=[tool.get("name") for tool in settings.tools],
            instructions_chars=len(settings.instructions),
        )
        self.emit_urgent(ev.session_updated(self._public_session(active)))
        if first and self.config.protocol.greeting_enabled:
            self.turns.speak_greeting()

    def _on_item_create(self, command: ce.ItemCreate) -> None:
        """v1's item create (S14)."""
        item = command.item
        if isinstance(item, ce.FunctionCallOutput):
            self.turns.on_function_output(item.call_id, item.output)
            item_id = item.item_id or new_item_id()
            wire_item = ev.function_call_output_item(item_id, call_id=item.call_id, output=item.output)
        else:
            item_id = item.item_id or new_item_id()
            wire_item = ev.user_text_item(item_id, item.text)
        previous = self.conversation.append(item_id)
        self.emit_urgent(ev.item_added(wire_item, previous))
        self.emit_urgent(ev.item_done(wire_item, previous))
        if isinstance(item, ce.UserText):
            self.turns.on_user_input(UserInput(text=item.text, item_id=item_id), from_audio=False)

    # -- input audio (v1's InputPath decode, B6; S1 step) --------------------------------

    def _configure_input(self, input_format: AudioFormat) -> None:
        """A format change replaces the resampler and drops a held byte (as v1's ``InputPath.configure``)."""
        if input_format != self._format:
            self._format = input_format
            self._resampler = StreamResampler(input_format.rate, self.input_rate)
            self._partial = b""

    def _clear_input(self) -> None:
        """``input_audio_buffer.clear``: drop unfinished input (as v1's ``InputPath.clear``)."""
        self._partial = b""

    def _on_append(self, payload: bytes) -> None:
        payload = self._partial + payload
        whole = len(payload) - len(payload) % self._format.bytes_per_sample
        payload, self._partial = payload[:whole], payload[whole:]
        samples = self._format.samples_in(payload)
        if samples > 0:
            self._pending_step += samples * 1000.0 / self._format.rate
        pcm = self._resampler.process(self._format.decode(payload))
        self.pipeline.apply_pending_vad()
        if pcm:
            self._pending_audio = (pcm, self._pending_step)
            self._pending_step = 0.0

    def output_format(self) -> AudioFormat:
        """The client's current output format."""
        return self.view.settings.output_format

    def mcp_allowed(self) -> bool:
        """Whether this session's client may declare MCP tools (``FBV_REALTIME_MCP``, never in eval mode)."""
        return switches.realtime_switch(switches.REALTIME_MCP_ENV)

    def _create_mcp(self) -> Any:
        """The session's MCP tools (``mcp_tools.py``), or ``None`` when MCP is off."""
        if not self.mcp_allowed():
            return None
        from examples.frontend_backend_verdict.adapters.pipecat import mcp_tools  # noqa: PLC0415

        if not mcp_tools.configured():
            logger.warning(f"{switches.REALTIME_MCP_ENV}=true needs REALTIME_MCP_ALLOWED_SERVER_URLS; MCP stays off")
            return None
        return mcp_tools.McpTools(
            feed_output=self.turns.on_function_output,
            resume=lambda: asyncio.get_running_loop().call_soon(self._resume_after_mcp),
            log=self.log.write,
        )

    def _resume_after_mcp(self) -> None:
        """A response held only MCP calls: ask for the continuation the client would have asked for."""
        try:
            self.turns.on_response_create()
        except WireProtocolError as exc:
            logger.info(f"[{self.session_id}] no continuation after MCP calls: {exc}")

    def allowed_turn_modes(self) -> frozenset[str]:
        """The turn modes this session's client may select (``switches.realtime_turn_modes``)."""
        return switches.realtime_turn_modes()

    def _select_turn_mode(self, turn: Any) -> str | None:
        """Switch the opt-in turn modes to the session's request; ``None`` when they are off."""
        modes = self.pipeline.turn_modes
        if modes is None:
            return None
        if turn.mode == "manual":
            requested = turn_modes.MANUAL
        elif turn.requested_type == "semantic_vad":
            requested = turn_modes.SEMANTIC_VAD
        else:
            requested = turn_modes.SERVER_VAD
        return modes.select(requested)

    def _public_session(self, active: str | None) -> dict[str, Any]:
        """The ``session.updated`` payload; the opt-in turn modes and MCP tools show as the client set them."""
        public = self.view.public()
        if self._mcp is not None and isinstance(public.get("tools"), list):
            public["tools"] = self._mcp.public_tools(public["tools"])
        if active is None:
            return public
        detection = public.get("audio", {}).get("input", {}).get("turn_detection")
        if isinstance(detection, dict):
            detection["x_nvidia_turn_strategy"] = {
                turn_modes.SEMANTIC_VAD: "smart_turn",
                turn_modes.MANUAL: "manual",
            }.get(active, "server_vad")
        return public

    def _resolve_voice(self) -> str | None:
        """v1's voice map (S19); the Pipecat TTS keeps its catalog voice (B8)."""
        requested = self.view.settings.voice
        voice_map = self.config.tts.voice_map
        if requested in voice_map:
            return voice_map[requested] or None
        if requested and requested not in self._warned:
            self._warned.add(requested)
            logger.warning(f"[{self.session_id}] voice {requested!r} not in tts.voice_map; using the catalog default")
        return None

    # -- lifecycle ----------------------------------------------------------------------

    def _restart_update_timer(self) -> None:
        """S10: v1 waited for each message with the timeout while unconfigured, so any message restarts it."""
        if self._update_timer is None:
            return  # not running yet
        if not self._update_timer.done():
            self._update_timer.cancel()
        self._update_timer = asyncio.get_running_loop().create_task(
            self._first_update_timeout(), name=f"fbv-update-{self.session_id}"
        )

    async def _first_update_timeout(self) -> None:
        """S10: no client message within the timeout before a ``session.update``: log and continue with defaults."""
        timeout = self.config.server.session_update_timeout_s
        await asyncio.sleep(timeout)
        if not self._configured:
            logger.warning(f"[{self.session_id}] no session.update within {timeout}s; using defaults")
            self._configured = True

    async def run(self) -> None:
        """Serve until the client disconnects."""
        loop = asyncio.get_running_loop()
        self._routing_sink.register(self.session_id, FillerTap(loop, self.turns.on_filler, clock=self.clock))
        await self._websocket.send_text(json.dumps(ev.session_created(self.view.public())))
        history = self.config.agent.backend.conversation_history
        self.log.write(
            "session_start",
            model=self.view.model,
            backend_history=history.effective_include,
            backend_history_guidance=history.resolved_guidance_key,
            normalization=self.config.normalization.summary(),
            adapter="pipecat",
            pipecat=self.settings.summary(),
            prompts=self._runtime_prompts,
        )
        logger.info(f"[{self.session_id}] session started (model={self.view.model or '-'}, adapter=pipecat)")
        self._pump = loop.create_task(self._pump_events(), name=f"fbv-events-{self.session_id}")
        self._update_timer = loop.create_task(self._first_update_timeout(), name=f"fbv-update-{self.session_id}")
        runner = WorkerRunner(handle_sigint=False, handle_sigterm=False)
        try:
            await runner.add_workers(self.pipeline.worker)
            await runner.run()
        finally:
            await self.close()

    async def close(self) -> None:
        """Release everything the session started, in v1's order (S22)."""
        self._routing_sink.unregister(self.session_id)
        if self._mcp is not None:
            self._mcp.shutdown()
        with contextlib.suppress(Exception):
            await self.turns.close()
        await self.pipeline.bridge.aclose()
        if self._update_timer is not None and not self._update_timer.done():
            self._update_timer.cancel()
        if self._pump is not None:
            self._queue.put_nowait(None)
            # Not CancelledError: a session cancelled from outside must stay cancelled.
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._pump, timeout=2.0)
        self.log.write("session_end")
        logger.info(f"[{self.session_id}] session closed")
