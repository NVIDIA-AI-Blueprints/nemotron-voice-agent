# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Cisco BYOVA gRPC service backed by a persistent Nemotron session."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import time
from collections.abc import AsyncIterator

import grpc
from google.protobuf.struct_pb2 import Struct

from cisco_webex_byova_adapter.audio import normalize_caller_audio, to_byova_audio
from cisco_webex_byova_adapter.config import AdapterConfig
from cisco_webex_byova_adapter.generated import (
    byova_common_pb2,
    health_pb2,
    health_pb2_grpc,
    voicevirtualagent_pb2,
    voicevirtualagent_pb2_grpc,
)
from cisco_webex_byova_adapter.nemotron_bridge import NemotronSession

logger = logging.getLogger(__name__)


_TRACE_PATH = os.environ.get("NEMOTRON_BYOVA_ADAPTER_TRACE_FILE", "")


def _trace(message: str) -> None:
    if not _TRACE_PATH:
        return
    with open(_TRACE_PATH, "a", encoding="utf-8") as trace_file:
        trace_file.write(message + "\n")


def _new_response(response_type: int) -> voicevirtualagent_pb2.VoiceVAResponse:
    return voicevirtualagent_pb2.VoiceVAResponse(
        input_mode=voicevirtualagent_pb2.INPUT_VOICE,
        response_type=response_type,
    )


def _chunk_response_with_audio(
    audio_16k: bytes,
    text: str = "",
    barge_in: bool = True,
) -> voicevirtualagent_pb2.VoiceVAResponse:
    """Build a CHUNK response carrying bot TTS audio for Webex.

    The input is 16 kHz mono int16 PCM (Nemotron's native rate); we downsample
    to 8 kHz here so Webex Universal Harness plays it at the right pitch.
    """
    response = _new_response(voicevirtualagent_pb2.VoiceVAResponse.CHUNK)
    prompt = voicevirtualagent_pb2.Prompt(
        audio_content=to_byova_audio(audio_16k),
        is_barge_in_enabled=barge_in,
    )
    if text:
        prompt.text = text
    response.prompts.append(prompt)
    return response


def _final_turn_response() -> voicevirtualagent_pb2.VoiceVAResponse:
    return _new_response(voicevirtualagent_pb2.VoiceVAResponse.FINAL)


def _start_of_input_response() -> voicevirtualagent_pb2.VoiceVAResponse:
    """Build a START_OF_INPUT response with only an output event.

    This intentionally matches Cisco's event-only response shape for input
    boundary notifications instead of using the normal prompt/final helpers.
    """
    return voicevirtualagent_pb2.VoiceVAResponse(
        output_events=[
            byova_common_pb2.OutputEvent(
                event_type=byova_common_pb2.OutputEvent.START_OF_INPUT,
            )
        ]
    )


def _end_of_input_response() -> voicevirtualagent_pb2.VoiceVAResponse:
    """Build an END_OF_INPUT response with only an output event.

    This intentionally matches Cisco's event-only response shape for input
    boundary notifications instead of using the normal prompt/final helpers.
    """
    return voicevirtualagent_pb2.VoiceVAResponse(
        output_events=[
            byova_common_pb2.OutputEvent(
                event_type=byova_common_pb2.OutputEvent.END_OF_INPUT,
            )
        ]
    )


def _streaming_chunk_response(
    audio_mulaw: bytes,
) -> voicevirtualagent_pb2.VoiceVAResponse:
    """Build a CHUNK response carrying bot audio."""
    return voicevirtualagent_pb2.VoiceVAResponse(
        prompts=[
            voicevirtualagent_pb2.Prompt(
                audio_content=audio_mulaw,
                is_barge_in_enabled=True,
            )
        ],
        input_mode=voicevirtualagent_pb2.INPUT_VOICE,
        response_type=voicevirtualagent_pb2.VoiceVAResponse.CHUNK,
    )


def _empty_final_response() -> voicevirtualagent_pb2.VoiceVAResponse:
    """Build the terminal FINAL response with an empty audio prompt."""
    return voicevirtualagent_pb2.VoiceVAResponse(
        prompts=[
            voicevirtualagent_pb2.Prompt(
                audio_content=b"",
                is_barge_in_enabled=True,
            )
        ],
        input_mode=voicevirtualagent_pb2.INPUT_VOICE,
        response_type=voicevirtualagent_pb2.VoiceVAResponse.FINAL,
    )


class _SessionEntry:
    """Per-conversation state held in the servicer registry."""

    __slots__ = (
        "session",
        "lock",
        "active_drainer_id",
        "last_touched",
        "greeted",
    )

    def __init__(self, session: NemotronSession) -> None:
        self.session = session
        # Serialize drain helpers that read from session.outbound_queue.
        self.lock = asyncio.Lock()
        # Monotonic counter used for log correlation only. Output ownership is
        # serialized by lock so a newer Cisco RPC does not preempt the prior
        # RPC while that prior RPC is still streaming TTS chunks.
        self.active_drainer_id: int = 0
        self.last_touched: float = time.monotonic()
        self.greeted: bool = False


class VoiceVirtualAgentServicer(voicevirtualagent_pb2_grpc.VoiceVirtualAgentServicer):
    """Cisco BYOVA gRPC servicer backed by a persistent Nemotron session."""

    def __init__(self, config: AdapterConfig) -> None:
        """Create a servicer with shared session state for active calls."""
        self._config = config
        self._sessions: dict[str, _SessionEntry] = {}
        self._sessions_lock = asyncio.Lock()
        self._session_reaper_task: asyncio.Task | None = None

    async def ListVirtualAgents(self, request, context):
        """Return the single adapter-backed virtual agent advertised to Cisco."""
        del request, context
        return byova_common_pb2.ListVAResponse(
            virtual_agents=[
                byova_common_pb2.VirtualAgentInfo(
                    virtual_agent_id=self._config.virtual_agent_id,
                    virtual_agent_name=self._config.virtual_agent_name,
                    is_default=True,
                    attributes={
                        "provider": "nemotron",
                        "pipeline_mode": self._config.pipeline_mode,
                    },
                )
            ]
        )

    async def _get_session(self, conversation_id: str) -> _SessionEntry | None:
        async with self._sessions_lock:
            return self._sessions.get(conversation_id)

    async def _get_or_create_session(
        self,
        conversation_id: str,
        vendor_specific_config: str,
    ) -> _SessionEntry:
        async with self._sessions_lock:
            entry = self._sessions.get(conversation_id)
            if entry is not None:
                return entry
            session = NemotronSession(
                config=self._config,
                conversation_id=conversation_id,
                vendor_specific_config=vendor_specific_config,
            )
            await session.start()
            entry = _SessionEntry(session)
            self._sessions[conversation_id] = entry
            logger.info("Created NemotronSession conversation_id=%s", conversation_id)
            _trace(f"created session conversation_id={conversation_id}")
            return entry

    async def _remove_session(self, conversation_id: str) -> _SessionEntry | None:
        async with self._sessions_lock:
            return self._sessions.pop(conversation_id, None)

    async def _ensure_session_reaper_started(self) -> None:
        """Start one background task that closes idle Nemotron sessions."""
        if self._session_reaper_task and not self._session_reaper_task.done():
            return
        self._session_reaper_task = asyncio.create_task(
            self._idle_session_reaper_loop(),
            name="adapter-idle-session-reaper",
        )

    async def _idle_session_reaper_loop(self) -> None:
        """Gracefully close hung/idle calls from adapter side.

        When Cisco leaves a conversation hanging (no new RPC/audio/events),
        this reaper removes stale sessions and closes Nemotron WebSocket so
        upstream pipeline tasks terminate cleanly.
        """
        interval = max(5.0, min(float(self._config.idle_session_timeout_secs) / 4.0, 30.0))
        while True:
            await asyncio.sleep(interval)
            now = time.monotonic()
            stale: list[tuple[str, _SessionEntry, float]] = []
            async with self._sessions_lock:
                for cid, entry in list(self._sessions.items()):
                    idle_for = now - entry.last_touched
                    if idle_for >= float(self._config.idle_session_timeout_secs):
                        stale.append((cid, entry, idle_for))
                        self._sessions.pop(cid, None)
            for cid, entry, idle_for in stale:
                logger.info(
                    "Reaping idle session conversation_id=%s idle_for=%.1fs timeout=%ss",
                    cid,
                    idle_for,
                    self._config.idle_session_timeout_secs,
                )
                with contextlib.suppress(Exception):
                    await entry.session.close()

    async def ProcessCallerInput(
        self,
        request_iterator: AsyncIterator[voicevirtualagent_pb2.VoiceVARequest],
        context: grpc.aio.ServicerContext,
    ):
        """Handle one Cisco turn stream and bridge it to Nemotron."""
        await self._ensure_session_reaper_started()
        logger.info("ProcessCallerInput opened")
        _trace("ProcessCallerInput opened")

        conversation_id: str | None = None
        entry: _SessionEntry | None = None
        terminal_yielded = False

        try:
            async for request in request_iterator:
                if conversation_id is None:
                    conversation_id = request.conversation_id
                    entry = await self._get_session(conversation_id)
                    if entry is not None:
                        logger.info("Resumed existing session conversation_id=%s", conversation_id)
                        _trace(f"resumed session conversation_id={conversation_id}")

                event_type = request.event_input.event_type

                if event_type == byova_common_pb2.EventInput.SESSION_START:
                    if entry is None:
                        entry = await self._get_or_create_session(conversation_id, request.vendor_specific_config)
                    entry.last_touched = time.monotonic()
                    if not entry.greeted:
                        # The first SESSION_START triggers the backend's intro turn
                        # before any caller audio is forwarded.
                        logger.info(
                            "Streaming Nemotron intro for conversation_id=%s",
                            conversation_id,
                        )
                        _trace(f"stream nemotron intro conversation_id={conversation_id}")
                        yielded_intro_audio = False
                        async for chunk in self._drain_bot_turn(entry):
                            if chunk.response_type == voicevirtualagent_pb2.VoiceVAResponse.FINAL:
                                terminal_yielded = True
                                removed = await self._remove_session(conversation_id)
                                if removed is not None:
                                    with contextlib.suppress(Exception):
                                        await removed.session.close()
                                yield chunk
                                return
                            yielded_intro_audio = yielded_intro_audio or (
                                chunk.response_type == voicevirtualagent_pb2.VoiceVAResponse.CHUNK
                            )
                            yield chunk
                        if not yielded_intro_audio:
                            logger.warning(
                                "Nemotron intro yielded no audio for conversation_id=%s",
                                conversation_id,
                            )
                        yield self._build_turn_final(entry, include_empty_audio=False)
                        entry.greeted = True
                        terminal_yielded = True
                    else:
                        yield _final_turn_response()
                        terminal_yielded = True
                    continue

                if event_type == byova_common_pb2.EventInput.SESSION_END:
                    logger.info("Received SESSION_END conversation_id=%s", conversation_id)
                    _trace(f"SESSION_END {conversation_id}")
                    removed = await self._remove_session(conversation_id)
                    if removed is not None:
                        try:
                            await removed.session.close()
                        except Exception:
                            logger.exception("Error closing Nemotron session on SESSION_END")
                    yield self._session_end_response()
                    terminal_yielded = True
                    return

                if event_type == byova_common_pb2.EventInput.NO_INPUT:
                    # Caller paused longer than no_input_timeout. We do NOT
                    # emit a NO_INPUT output_event back — that would surface
                    # as a terminal "VA gave up" signal in Webex's flow and
                    # bounce the call into the next activity (Queue/Music).
                    # Yielding a plain FINAL turn instead keeps the VA active
                    # and lets the caller try speaking again.
                    logger.info(
                        "Received NO_INPUT from Webex conversation_id=%s; keeping VA active",
                        conversation_id,
                    )
                    _trace(f"NO_INPUT conversation_id={conversation_id}")
                    if entry is not None:
                        entry.last_touched = time.monotonic()
                    yield _final_turn_response()
                    terminal_yielded = True
                    continue

                if request.audio_input.caller_audio:
                    if entry is None:
                        await context.abort(
                            grpc.StatusCode.FAILED_PRECONDITION,
                            "SESSION_START is required first",
                        )
                    cid = conversation_id
                    entry.last_touched = time.monotonic()
                    end_of_input_sent = False
                    end_of_input_pending = False
                    start_of_input_sent = False
                    rpc_start_t = time.monotonic()
                    rpc_recv_frame_n = 1
                    rpc_recv_bytes = len(request.audio_input.caller_audio)
                    rpc_last_recv_t = rpc_start_t

                    first_audio_pcm: bytes = b""
                    try:
                        first_audio_pcm, entry.session.caller_resample_state = normalize_caller_audio(
                            request.audio_input.caller_audio,
                            request.audio_input.encoding,
                            request.audio_input.sample_rate_hertz,
                            entry.session.caller_resample_state,
                        )
                    except ValueError as exc:
                        logger.warning("Dropping unsupported caller audio chunk: %s", exc)

                    entry.active_drainer_id += 1
                    my_id = entry.active_drainer_id
                    logger.info(
                        "[LAT] T1 first_user_audio conversation_id=%s t=%.3f rpc_id=%d",
                        cid,
                        time.monotonic(),
                        my_id,
                    )
                    logger.info(
                        (
                            "[IN] cisco_audio_recv conversation_id=%s rpc_id=%d frame_n=%d "
                            "raw_bytes=%d pcm_bytes=%d sample_rate=%d encoding=%d "
                            "gap_ms=0 elapsed_ms=0"
                        ),
                        cid,
                        my_id,
                        rpc_recv_frame_n,
                        len(request.audio_input.caller_audio),
                        len(first_audio_pcm),
                        request.audio_input.sample_rate_hertz,
                        request.audio_input.encoding,
                    )

                    if first_audio_pcm:
                        send_start_t = time.monotonic()
                        await entry.session.send_audio(first_audio_pcm)
                        logger.info(
                            (
                                "[IN] cisco_audio_forwarded conversation_id=%s rpc_id=%d "
                                "frame_n=%d pcm_bytes=%d send_await_ms=%.1f"
                            ),
                            cid,
                            my_id,
                            rpc_recv_frame_n,
                            len(first_audio_pcm),
                            (time.monotonic() - send_start_t) * 1000.0,
                        )

                    reader_done = asyncio.Event()
                    session_end_received = asyncio.Event()

                    session = entry.session

                    async def _reader_loop(
                        cid: str = cid,
                        my_id: int = my_id,
                        session: NemotronSession = session,
                        rpc_start_t: float = rpc_start_t,
                        reader_done: asyncio.Event = reader_done,
                        session_end_received: asyncio.Event = session_end_received,
                    ):
                        # Keep reading caller audio while the outer coroutine
                        # owns output draining for this turn.
                        nonlocal rpc_recv_frame_n, rpc_recv_bytes, rpc_last_recv_t
                        try:
                            async for req in request_iterator:
                                if req.event_input.event_type == byova_common_pb2.EventInput.SESSION_END:
                                    _trace(f"mid-turn SESSION_END {cid}")
                                    session_end_received.set()
                                    return
                                if not req.audio_input.caller_audio:
                                    continue
                                now = time.monotonic()
                                gap_ms = (now - rpc_last_recv_t) * 1000.0
                                rpc_last_recv_t = now
                                rpc_recv_frame_n += 1
                                rpc_recv_bytes += len(req.audio_input.caller_audio)
                                try:
                                    a, session.caller_resample_state = normalize_caller_audio(
                                        req.audio_input.caller_audio,
                                        req.audio_input.encoding,
                                        req.audio_input.sample_rate_hertz,
                                        session.caller_resample_state,
                                    )
                                except ValueError:
                                    continue
                                if rpc_recv_frame_n <= 20 or rpc_recv_frame_n % 50 == 0 or gap_ms > 150.0:
                                    log = logger.warning if gap_ms > 500.0 else logger.info
                                    log(
                                        (
                                            "[IN] cisco_audio_recv conversation_id=%s rpc_id=%d "
                                            "frame_n=%d raw_bytes=%d pcm_bytes=%d "
                                            "rpc_total_raw_bytes=%d sample_rate=%d encoding=%d "
                                            "gap_ms=%.0f elapsed_ms=%.0f"
                                        ),
                                        cid,
                                        my_id,
                                        rpc_recv_frame_n,
                                        len(req.audio_input.caller_audio),
                                        len(a),
                                        rpc_recv_bytes,
                                        req.audio_input.sample_rate_hertz,
                                        req.audio_input.encoding,
                                        gap_ms,
                                        (now - rpc_start_t) * 1000.0,
                                    )
                                send_start_t = time.monotonic()
                                await session.send_audio(a)
                                if rpc_recv_frame_n <= 20 or rpc_recv_frame_n % 50 == 0 or gap_ms > 150.0:
                                    logger.info(
                                        (
                                            "[IN] cisco_audio_forwarded conversation_id=%s rpc_id=%d "
                                            "frame_n=%d pcm_bytes=%d send_await_ms=%.1f"
                                        ),
                                        cid,
                                        my_id,
                                        rpc_recv_frame_n,
                                        len(a),
                                        (time.monotonic() - send_start_t) * 1000.0,
                                    )
                        except asyncio.CancelledError:
                            raise
                        except Exception:
                            logger.exception("reader loop error conversation_id=%s", cid)
                        finally:
                            logger.info(
                                (
                                    "[IN] cisco_reader_done conversation_id=%s rpc_id=%d "
                                    "frames=%d raw_bytes=%d elapsed_ms=%.0f"
                                ),
                                cid,
                                my_id,
                                rpc_recv_frame_n,
                                rpc_recv_bytes,
                                (time.monotonic() - rpc_start_t) * 1000.0,
                            )
                            reader_done.set()

                    reader_task = asyncio.create_task(_reader_loop())

                    yielded_first_audio = False
                    bot_response_done = False
                    bridge_error_response: voicevirtualagent_pb2.VoiceVAResponse | None = None
                    idle_timeout = max(self._config.response_idle_timeout_secs, 0.5)
                    no_final_idle_secs = max(idle_timeout, 2.0)

                    try:
                        async with entry.lock:
                            # Only one gRPC turn for a conversation can drain
                            # the shared Nemotron outbound queue at a time.
                            logger.info(
                                "[LAT] output_drain_acquired conversation_id=%s rpc_id=%d",
                                cid,
                                my_id,
                            )
                            last_activity_t = time.monotonic()
                            turn_start_t = last_activity_t
                            while True:
                                if session_end_received.is_set():
                                    break
                                try:
                                    item = await asyncio.wait_for(
                                        entry.session.outbound_queue.get(),
                                        timeout=0.10,
                                    )
                                except TimeoutError:
                                    # Nemotron emits an explicit final marker
                                    # after output goes idle, but we also stop
                                    # once the caller stream is closed and no
                                    # more bot activity arrives.
                                    if bot_response_done and (time.monotonic() - last_activity_t) > 0.3:
                                        break
                                    if reader_done.is_set():
                                        idle_for = time.monotonic() - last_activity_t
                                        turn_age = time.monotonic() - turn_start_t
                                        if yielded_first_audio and idle_for > no_final_idle_secs:
                                            logger.warning(
                                                (
                                                    "No explicit final marker; closing turn after "
                                                    "post-audio idle period conversation_id=%s "
                                                    "idle=%.2fs"
                                                ),
                                                cid,
                                                idle_for,
                                            )
                                            break
                                        if not yielded_first_audio and turn_age > self._config.first_audio_timeout_secs:
                                            logger.warning(
                                                (
                                                    "No bot audio within first-audio timeout; "
                                                    "closing turn conversation_id=%s turn_age=%.2fs"
                                                ),
                                                cid,
                                                turn_age,
                                            )
                                            break
                                    continue

                                last_activity_t = time.monotonic()
                                kind = item.get("kind") if isinstance(item, dict) else None

                                if kind == "user_started_speaking":
                                    if not start_of_input_sent:
                                        # Webex expects a single START_OF_INPUT
                                        # notification per caller turn.
                                        start_of_input_sent = True
                                        logger.info("[LAT] START_OF_INPUT (from pipecat VAD) conversation_id=%s", cid)
                                        yield _start_of_input_response()
                                elif kind == "user_stopped_speaking":
                                    if start_of_input_sent and not end_of_input_sent:
                                        # Defer END_OF_INPUT until bot output
                                        # begins so Webex keeps forwarding caller
                                        # audio during barge-in/follow-up speech.
                                        end_of_input_pending = True
                                        logger.info(
                                            "[LAT] END_OF_INPUT deferred until bot starts conversation_id=%s",
                                            cid,
                                        )
                                elif kind == "bot_started_speaking":
                                    if end_of_input_pending and not end_of_input_sent:
                                        end_of_input_sent = True
                                        end_of_input_pending = False
                                        logger.info(
                                            "[LAT] END_OF_INPUT (deferred until bot start) conversation_id=%s",
                                            cid,
                                        )
                                        yield _end_of_input_response()
                                    logger.info("[LAT] BOT_STARTED_SPEAKING conversation_id=%s", cid)
                                elif kind == "bot_stopped_speaking":
                                    logger.info("[LAT] BOT_STOPPED_SPEAKING conversation_id=%s", cid)
                                elif kind == "audio":
                                    if end_of_input_pending and not end_of_input_sent:
                                        end_of_input_sent = True
                                        end_of_input_pending = False
                                        logger.info(
                                            "[LAT] END_OF_INPUT (fallback before first bot audio) conversation_id=%s",
                                            cid,
                                        )
                                        yield _end_of_input_response()
                                    if not yielded_first_audio:
                                        yielded_first_audio = True
                                        logger.info(
                                            "[LAT] T3 first_chunk_yielded conversation_id=%s t=%.3f gap_since_T1=%.3fs",
                                            cid,
                                            time.monotonic(),
                                            time.monotonic() - entry.last_touched,
                                        )
                                    mulaw = to_byova_audio(item["audio"])
                                    if mulaw and len(mulaw) >= 100:
                                        yield _streaming_chunk_response(mulaw)
                                elif kind == "final":
                                    bot_response_done = True
                                elif kind == "error":
                                    bridge_error_response = self._convert_bridge_item(item)
                                    break
                                # message / unknown: ignore
                    finally:
                        if not reader_task.done():
                            reader_task.cancel()
                            with contextlib.suppress(asyncio.CancelledError, Exception):
                                await reader_task

                    if session_end_received.is_set():
                        removed = await self._remove_session(conversation_id)
                        if removed is not None:
                            with contextlib.suppress(Exception):
                                await removed.session.close()
                        yield self._session_end_response()
                        terminal_yielded = True
                        return

                    if bridge_error_response is not None:
                        terminal_yielded = True
                        removed = await self._remove_session(conversation_id)
                        if removed is not None:
                            with contextlib.suppress(Exception):
                                await removed.session.close()
                        yield bridge_error_response
                        return

                    logger.info(
                        "[LAT] T5 FINAL conversation_id=%s t=%.3f yielded_audio=%s start_of_input=%s end_of_input=%s",
                        cid,
                        time.monotonic(),
                        yielded_first_audio,
                        start_of_input_sent,
                        end_of_input_sent,
                    )
                    response = self._build_turn_final(entry, include_empty_audio=True)
                    yield response
                    terminal_yielded = True
                    if self._response_is_terminal(response):
                        removed = await self._remove_session(conversation_id)
                        if removed is not None:
                            with contextlib.suppress(Exception):
                                await removed.session.close()
                        return
                    break  # turn done

            if not terminal_yielded:
                yield _final_turn_response()

        except grpc.aio.AioRpcError:
            raise
        except Exception:
            logger.exception("ProcessCallerInput failed")
            if not terminal_yielded:
                with contextlib.suppress(Exception):
                    yield _final_turn_response()

    async def _drain_bot_turn(self, entry: _SessionEntry):
        """Yield bot audio until the current turn completes."""
        async with entry.lock:
            session = entry.session
            loop = asyncio.get_running_loop()
            first_audio_deadline = loop.time() + self._config.first_audio_timeout_secs
            idle_timeout = max(self._config.response_idle_timeout_secs, 0.5)
            got_audio = False

            while True:
                if got_audio:
                    timeout = idle_timeout
                else:
                    timeout = first_audio_deadline - loop.time()
                    if timeout <= 0:
                        logger.warning(
                            "Nemotron produced no audio within %.1fs",
                            self._config.first_audio_timeout_secs,
                        )
                        _trace(f"drain timeout first_audio={self._config.first_audio_timeout_secs}")
                        return

                try:
                    item = await asyncio.wait_for(session.outbound_queue.get(), timeout=timeout)
                except TimeoutError:
                    if got_audio:
                        return
                    continue

                kind = item.get("kind") if isinstance(item, dict) else None
                if kind == "audio":
                    got_audio = True
                    response = self._convert_bridge_item(item)
                    if response is not None:
                        yield response
                elif kind == "final":
                    # Ignore stray finals until at least one audio chunk for
                    # this turn has been observed.
                    if got_audio:
                        return
                elif kind == "error":
                    response = self._convert_bridge_item(item)
                    if response is not None:
                        yield response
                    return

    def _session_end_response(self) -> voicevirtualagent_pb2.VoiceVAResponse:
        response = _final_turn_response()
        response.output_events.append(
            byova_common_pb2.OutputEvent(
                event_type=byova_common_pb2.OutputEvent.SESSION_END,
                name="session_ended_by_client",
            )
        )
        return response

    def _build_turn_final(
        self,
        entry: _SessionEntry,
        *,
        include_empty_audio: bool,
    ) -> voicevirtualagent_pb2.VoiceVAResponse:
        response = _empty_final_response() if include_empty_audio else _final_turn_response()
        session = entry.session
        if session.transfer_requested:
            response.output_events.append(self._transfer_output_event(session.transfer_metadata()))
        elif session.end_session_requested:
            response.output_events.append(
                byova_common_pb2.OutputEvent(
                    event_type=byova_common_pb2.OutputEvent.SESSION_END,
                    name="session_ended",
                )
            )
        return response

    def _convert_bridge_item(self, bridge_item: object) -> voicevirtualagent_pb2.VoiceVAResponse | None:
        if not isinstance(bridge_item, dict):
            return None
        kind = bridge_item.get("kind")
        if kind == "audio":
            return _chunk_response_with_audio(
                bridge_item["audio"],
                text=bridge_item.get("text", ""),
                barge_in=True,
            )
        if kind in ("final", "message"):
            return None
        if kind == "error":
            response = _final_turn_response()
            metadata = Struct()
            metadata.update({"detail": str(bridge_item.get("error", ""))})
            response.output_events.append(
                byova_common_pb2.OutputEvent(
                    event_type=byova_common_pb2.OutputEvent.CUSTOM_EVENT,
                    name="nemotron-error",
                    metadata=metadata,
                )
            )
            return response
        return None

    def _transfer_output_event(self, metadata_values: dict) -> byova_common_pb2.OutputEvent:
        metadata = Struct()
        normalized = json.loads(json.dumps(metadata_values)) if metadata_values else {}
        metadata.update(normalized)
        return byova_common_pb2.OutputEvent(
            event_type=byova_common_pb2.OutputEvent.TRANSFER_TO_AGENT,
            name="transfer_to_agent",
            metadata=metadata,
        )

    @staticmethod
    def _response_is_terminal(response: voicevirtualagent_pb2.VoiceVAResponse) -> bool:
        terminal_types = {
            byova_common_pb2.OutputEvent.TRANSFER_TO_AGENT,
            byova_common_pb2.OutputEvent.SESSION_END,
        }
        return any(event.event_type in terminal_types for event in response.output_events)


class HealthServicer(health_pb2_grpc.HealthServicer):
    """Simple gRPC health servicer for the adapter process."""

    async def Check(self, request, context):
        """Return a SERVING health result."""
        del request, context
        return health_pb2.HealthCheckResponse(status=health_pb2.HealthCheckResponse.SERVING)

    async def Watch(self, request, context):
        """Stream a SERVING health result."""
        del request, context
        yield health_pb2.HealthCheckResponse(status=health_pb2.HealthCheckResponse.SERVING)
