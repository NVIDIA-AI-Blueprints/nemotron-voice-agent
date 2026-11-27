# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NVIDIA streaming ASR that keeps its blocking gRPC readers off the default executor.

``NvidiaSTTService`` reads each session's gRPC response stream on a thread from
``asyncio.to_thread``. That thread blocks for the whole session, and it comes from the
event loop's default executor, which is capped at 32 threads and is shared with every
other blocking call in the worker, including the service readiness probes run before a
WebSocket is accepted. Once a worker holds 32 sessions, a new session's readiness probe
waits for a free thread until its timeout expires, even though every model is healthy.

This module provides:

* ``IsolatedNvidiaSTTService``: reads the stream on a dedicated ``nva-asr-stream`` pool,
  so the default executor stays free for readiness probes and other short blocking work.
  It can also batch input PCM into larger gRPC messages (``NVA_ASR_INPUT_BATCH_MS``).
* ``AsrStreamSlots``: a per-worker-process count of sessions that may hold an ASR stream
  (``NVA_ASR_MAX_STREAMS_PER_WORKER``). The server reserves a slot before it accepts a
  WebSocket, so an overloaded worker answers HTTP 503 instead of accepting a session whose
  ASR reader can never start.
"""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

from loguru import logger
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.services.nvidia.stt import AudioChunkIterator, NvidiaSTTService

from nemotron_voice_agent.pipecat.runtime.utils import parse_env_int

ASR_MAX_STREAMS_ENV = "NVA_ASR_MAX_STREAMS_PER_WORKER"
ASR_INPUT_BATCH_MS_ENV = "NVA_ASR_INPUT_BATCH_MS"
DEFAULT_MAX_STREAMS_PER_WORKER = 64

# A reconnect starts the replacement reader before the cancelled one has unwound, so the
# pool is slightly larger than the admission cap.
_RECONNECT_HEADROOM = 8
_BYTES_PER_SAMPLE = 2  # The pipelines feed 16-bit mono PCM to the ASR service.


def max_streams_per_worker() -> int:
    """Return the configured number of concurrent ASR streams one worker process may hold."""
    return parse_env_int(ASR_MAX_STREAMS_ENV, DEFAULT_MAX_STREAMS_PER_WORKER, min_value=1)


def input_batch_ms() -> int:
    """Return the configured ASR input batch length in milliseconds; ``0`` sends PCM as it arrives."""
    return parse_env_int(ASR_INPUT_BATCH_MS_ENV, 0, min_value=0)


class AsrStreamSlots:
    """Count sessions that hold an ASR stream in this process.

    The count is touched only from the event loop, so it needs no lock.
    """

    def __init__(self, capacity: int) -> None:
        """Create the counter.

        Args:
            capacity: Most sessions that may hold a slot at once.
        """
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self.capacity = capacity
        self.active = 0

    def try_acquire(self) -> bool:
        """Reserve a slot, returning ``False`` when the worker is at capacity."""
        if self.active >= self.capacity:
            return False
        self.active += 1
        return True

    def release(self) -> None:
        """Return a slot reserved by ``try_acquire``."""
        if self.active > 0:
            self.active -= 1


_slots: AsrStreamSlots | None = None


def get_asr_stream_slots() -> AsrStreamSlots:
    """Return this process's slot counter, creating it from the environment on first use."""
    global _slots
    if _slots is None:
        _slots = AsrStreamSlots(max_streams_per_worker())
    return _slots


_executor: ThreadPoolExecutor | None = None
_executor_lock = threading.Lock()


def _asr_stream_executor() -> ThreadPoolExecutor:
    global _executor
    with _executor_lock:
        if _executor is None:
            _executor = ThreadPoolExecutor(
                max_workers=max_streams_per_worker() + _RECONNECT_HEADROOM,
                thread_name_prefix="nva-asr-stream",
            )
        return _executor


class _BatchableAudioChunkIterator(AudioChunkIterator):
    """``AudioChunkIterator`` that can also be fed from a synchronous event-loop callback."""

    def put_nowait(self, audio: bytes) -> None:
        """Enqueue audio without awaiting; call only on the event loop."""
        if not self.closed:
            self._queue.put_nowait(audio)


class IsolatedNvidiaSTTService(NvidiaSTTService):
    """``NvidiaSTTService`` with a dedicated stream-reader pool and optional input batching."""

    def __init__(self, *, input_batch_ms_override: int | None = None, **kwargs) -> None:
        """Create the service.

        Args:
            input_batch_ms_override: PCM batch length in milliseconds. Defaults to
                ``NVA_ASR_INPUT_BATCH_MS``. ``0`` forwards each input frame unchanged.
            **kwargs: Passed to ``NvidiaSTTService``.
        """
        super().__init__(**kwargs)
        self._batch_ms = input_batch_ms() if input_batch_ms_override is None else input_batch_ms_override
        if self._batch_ms < 0:
            raise ValueError("input_batch_ms_override must not be negative")
        self._pending_pcm = bytearray()
        self._tail_timer: asyncio.TimerHandle | None = None

    async def setup(self, setup: FrameProcessorSetup) -> None:
        """Set up the service and swap in an iterator that supports batching."""
        await super().setup(setup)
        if self._batch_ms:
            await self._audio_iterator.close()
            self._audio_iterator = _BatchableAudioChunkIterator(self.get_event_loop())

    async def _thread_task_handler(self) -> None:
        iterator = self._audio_iterator
        if iterator is None:
            return
        config = self._config
        await self.get_event_loop().run_in_executor(_asr_stream_executor(), self._response_handler, iterator, config)

    async def run_stt(self, audio: bytes):
        """Forward ``audio`` to the stream, batched into ``input_batch_ms`` messages when enabled."""
        if not self._batch_ms:
            async for frame in super().run_stt(audio):
                yield frame
            return

        iterator = self._audio_iterator
        if iterator is not None and not iterator.closed:
            target = self.sample_rate * _BYTES_PER_SAMPLE * self._batch_ms // 1000
            self._pending_pcm.extend(audio)
            while len(self._pending_pcm) >= target:
                iterator.put_nowait(bytes(self._pending_pcm[:target]))
                del self._pending_pcm[:target]
            if self._pending_pcm:
                self._schedule_tail_flush()
            else:
                self._cancel_tail_flush()
        yield None

    def _schedule_tail_flush(self) -> None:
        """Send a short remainder after one batch period so trailing speech is never held back."""
        if self._tail_timer is None:
            self._tail_timer = self.get_event_loop().call_later(self._batch_ms / 1000, self._flush_tail)

    def _cancel_tail_flush(self) -> None:
        if self._tail_timer is not None:
            self._tail_timer.cancel()
            self._tail_timer = None

    def _flush_tail(self) -> None:
        self._tail_timer = None
        if not self._pending_pcm:
            return
        audio = bytes(self._pending_pcm)
        self._pending_pcm.clear()
        iterator = self._audio_iterator
        if iterator is not None and not iterator.closed:
            iterator.put_nowait(audio)

    async def _stop_tasks(self, close_iterator: bool = True) -> None:
        if close_iterator:
            self._cancel_tail_flush()
            self._pending_pcm.clear()
        await super()._stop_tasks(close_iterator=close_iterator)


async def reject_at_capacity(websocket, slots: AsrStreamSlots) -> None:
    """Refuse a WebSocket before accepting it because this worker has no free ASR stream slot.

    Sends HTTP 503 when the server supports WebSocket denial responses, else closes with 1013.
    """
    from starlette.responses import JSONResponse

    logger.warning(f"Rejecting WebSocket: worker ASR stream capacity reached ({slots.active}/{slots.capacity})")
    message = "Voice Agent worker ASR stream capacity reached"
    if "websocket.http.response" in websocket.scope.get("extensions", {}):
        await websocket.send_denial_response(JSONResponse({"info": message}, status_code=503))
    else:
        await websocket.close(code=1013, reason=message)
