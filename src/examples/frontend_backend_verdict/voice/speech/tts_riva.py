# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Riva streaming TTS (nemo-speech Magpie or NVCF), one RPC per sentence.

Same concurrency model as the ASR adapter: one shared channel, one
``SynthesizeOnline`` RPC per sentence, created and consumed on one worker thread
of a dedicated bounded executor. Closing the async iterator (barge-in) cancels
the gRPC call, which frees the thread and the server-side stream promptly.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import riva.client
import riva.client.proto.riva_tts_pb2 as rtts
from loguru import logger

from examples.frontend_backend_verdict.voice.errors import SpeechServiceError
from examples.frontend_backend_verdict.voice.speech.catalog import SpeechEndpoint

_BACKPRESSURE_LOG_MS = 50.0


class RivaSynthesizer:
    """Shared Riva TTS channel plus a bounded executor for sentence RPCs."""

    def __init__(self, endpoint: SpeechEndpoint, *, sample_rate: int, max_streams: int) -> None:
        """Create the channel (no RPC yet; :meth:`warmup` probes the endpoint)."""
        self.endpoint = endpoint
        self._rate = sample_rate
        self._auth = riva.client.Auth(None, endpoint.use_ssl, endpoint.server, endpoint.metadata())
        self._service = riva.client.SpeechSynthesisService(self._auth)
        self._executor = ThreadPoolExecutor(max_workers=max(1, max_streams), thread_name_prefix="riva-tts")
        self.worker_threads: list[str] = []
        logger.info(f"TTS channel ready: {endpoint.describe()} sample_rate={sample_rate} max_streams={max_streams}")

    @property
    def sample_rate(self) -> int:
        """PCM16 rate of synthesized audio."""
        return self._rate

    def _request(self, text: str, voice: str | None) -> Any:
        request = rtts.SynthesizeSpeechRequest(
            text=text,
            language_code=self.endpoint.language_code,
            sample_rate_hz=self._rate,
            encoding=riva.client.AudioEncoding.LINEAR_PCM,
        )
        voice_name = voice or self.endpoint.voice_id
        if voice_name:
            request.voice_name = voice_name
        return request

    async def synthesize(self, text: str, *, voice: str | None = None) -> AsyncIterator[bytes]:
        """Yield PCM16 chunks for one sentence; closing the iterator cancels the RPC."""
        loop = asyncio.get_running_loop()
        chunks: asyncio.Queue[bytes | BaseException | None] = asyncio.Queue()
        stop = threading.Event()
        state: dict[str, Any] = {"call": None}
        queued_at = time.monotonic()
        request = self._request(text, voice)

        def deliver(item: bytes | BaseException | None) -> None:
            with contextlib.suppress(RuntimeError):  # event loop closed: the session is gone
                loop.call_soon_threadsafe(chunks.put_nowait, item)

        def worker() -> None:
            self.worker_threads.append(threading.current_thread().name)
            waited_ms = (time.monotonic() - queued_at) * 1000.0
            if waited_ms > _BACKPRESSURE_LOG_MS:
                logger.info(f"speech_backpressure tts waited_ms={waited_ms:.0f}")
            try:
                if stop.is_set():
                    return
                call = self._service.stub.SynthesizeOnline(
                    iter([request]), metadata=self._service.auth.get_auth_metadata()
                )
                state["call"] = call
                for response in call:
                    if stop.is_set():
                        break
                    if response.audio:
                        deliver(bytes(response.audio))
            except Exception as exc:  # noqa: BLE001 - surfaced to the async side
                if not stop.is_set():
                    deliver(exc)
            finally:
                state["call"] = None
                deliver(None)

        loop.run_in_executor(self._executor, worker)
        try:
            while True:
                item = await chunks.get()
                if item is None:
                    return
                if isinstance(item, BaseException):
                    raise SpeechServiceError(f"TTS failed: {item}") from item
                yield item
        finally:
            stop.set()
            call = state["call"]
            if call is not None:
                try:
                    call.cancel()
                except Exception as exc:  # noqa: BLE001 - best effort
                    logger.debug(f"TTS call cancel failed: {exc}")

    async def warmup(self) -> None:
        """Synthesize a short phrase so the first real turn does not pay channel setup."""
        try:
            received = 0
            async for chunk in self.synthesize("Hello."):
                received += len(chunk)
            if not received:
                raise SpeechServiceError("TTS returned no audio")
        except SpeechServiceError as exc:
            raise SpeechServiceError(f"TTS warm-up failed against {self.endpoint.describe()}: {exc}") from exc

    async def aclose(self) -> None:
        """Stop accepting work."""
        self._executor.shutdown(wait=False, cancel_futures=True)
