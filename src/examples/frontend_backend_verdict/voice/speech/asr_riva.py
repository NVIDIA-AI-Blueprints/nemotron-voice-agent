# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Riva streaming ASR (nemo-speech or NVCF), one gRPC stream per utterance.

Concurrency model (plan section 13.5):

* one ``riva.client.Auth`` channel and one ``ASRService`` per process, shared
  by all sessions (gRPC channels are thread-safe and multiplex RPCs);
* every utterance gets its own ``StreamingRecognize`` RPC, created, consumed and
  closed on **one** worker thread of a dedicated bounded executor; results cross
  to the event loop through ``call_soon_threadsafe``;
* cancelling an utterance closes its request iterator, which ends the RPC and
  frees the thread.

Utterance-scoped streams make a transcript depend only on the utterance audio,
never on seconds of silence or on wall-clock gaps between client appends.
"""

from __future__ import annotations

import asyncio
import contextlib
import queue
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import riva.client
from loguru import logger

from examples.frontend_backend_verdict.voice.errors import SpeechServiceError
from examples.frontend_backend_verdict.voice.speech.catalog import SpeechEndpoint

_FINISH_TIMEOUT_S = 15.0
_BACKPRESSURE_LOG_MS = 50.0


class RivaRecognizerStream:
    """One utterance's recognition RPC."""

    def __init__(
        self,
        *,
        service: Any,
        streaming_config: Any,
        executor: ThreadPoolExecutor,
        on_interim: Callable[[str], None] | None,
    ) -> None:
        """Start the worker immediately; audio is queued until the RPC consumes it."""
        self._service = service
        self._config = streaming_config
        self._loop = asyncio.get_running_loop()
        self._on_interim = on_interim
        self._audio: queue.Queue[bytes | None] = queue.Queue()
        self._cancelled = threading.Event()
        self._closed = False
        self._queued_at = time.monotonic()
        self.worker_thread: str | None = None
        self._future = self._loop.run_in_executor(executor, self._worker)

    def _chunks(self) -> Iterator[bytes]:
        while True:
            chunk = self._audio.get()
            if chunk is None or self._cancelled.is_set():
                return
            yield chunk

    def _notify(self, text: str) -> None:
        if self._on_interim is None:
            return
        with contextlib.suppress(RuntimeError):  # event loop closed: the session is gone
            self._loop.call_soon_threadsafe(self._on_interim, text)

    def _worker(self) -> str:
        self.worker_thread = threading.current_thread().name
        waited_ms = (time.monotonic() - self._queued_at) * 1000.0
        if waited_ms > _BACKPRESSURE_LOG_MS:
            logger.info(f"speech_backpressure asr waited_ms={waited_ms:.0f}")
        finals: list[str] = []
        responses = self._service.streaming_response_generator(
            audio_chunks=self._chunks(), streaming_config=self._config
        )
        try:
            for response in responses:
                if self._cancelled.is_set():
                    break
                for result in response.results:
                    if not result.alternatives:
                        continue
                    text = result.alternatives[0].transcript.strip()
                    if result.is_final:
                        if text:
                            finals.append(text)
                    elif text:
                        self._notify(" ".join([*finals, text]))
        finally:
            close = getattr(responses, "close", None)
            if close is not None:
                close()
        return " ".join(finals)

    async def push(self, pcm: bytes) -> None:
        """Queue audio for the RPC."""
        if not self._closed and pcm:
            self._audio.put_nowait(pcm)

    async def finish(self) -> str:
        """End the audio stream and wait for the final transcript."""
        if not self._closed:
            self._closed = True
            self._audio.put_nowait(None)
        try:
            return await asyncio.wait_for(asyncio.shield(self._future), timeout=_FINISH_TIMEOUT_S)
        except TimeoutError as exc:
            self._cancelled.set()
            raise SpeechServiceError(f"ASR did not finalize within {_FINISH_TIMEOUT_S}s") from exc
        except Exception as exc:
            raise SpeechServiceError(f"ASR stream failed: {exc}") from exc

    async def cancel(self) -> None:
        """Abandon the utterance; the worker thread exits once the RPC ends."""
        self._cancelled.set()
        if not self._closed:
            self._closed = True
            self._audio.put_nowait(None)


class RivaStreamingRecognizer:
    """Shared Riva ASR channel plus a bounded executor for utterance streams."""

    def __init__(self, endpoint: SpeechEndpoint, *, max_streams: int, interim_results: bool = True) -> None:
        """Create the channel (no RPC yet; :meth:`warmup` probes the endpoint)."""
        self.endpoint = endpoint
        self._interim = interim_results
        self._auth = riva.client.Auth(None, endpoint.use_ssl, endpoint.server, endpoint.metadata())
        self._service = riva.client.ASRService(self._auth)
        self._executor = ThreadPoolExecutor(max_workers=max(1, max_streams), thread_name_prefix="riva-asr")
        logger.info(f"ASR channel ready: {endpoint.describe()} max_streams={max_streams}")

    def _streaming_config(self, sample_rate: int) -> Any:
        return riva.client.StreamingRecognitionConfig(
            config=riva.client.RecognitionConfig(
                encoding=riva.client.AudioEncoding.LINEAR_PCM,
                language_code=self.endpoint.language_code,
                model="",
                max_alternatives=1,
                enable_automatic_punctuation=True,
                verbatim_transcripts=False,
                sample_rate_hertz=sample_rate,
                audio_channel_count=1,
            ),
            interim_results=self._interim,
        )

    def open(self, *, sample_rate: int, on_interim: Callable[[str], None] | None = None) -> RivaRecognizerStream:
        """Start one utterance's RPC."""
        return RivaRecognizerStream(
            service=self._service,
            streaming_config=self._streaming_config(sample_rate),
            executor=self._executor,
            on_interim=on_interim,
        )

    async def warmup(self) -> None:
        """Run a short silent utterance so the first real turn does not pay channel setup."""
        stream = self.open(sample_rate=16000)
        await stream.push(b"\x00\x00" * 8000)
        try:
            await stream.finish()
        except SpeechServiceError as exc:
            raise SpeechServiceError(f"ASR warm-up failed against {self.endpoint.describe()}: {exc}") from exc

    async def aclose(self) -> None:
        """Stop accepting work (running streams finish on their own)."""
        self._executor.shutdown(wait=False, cancel_futures=True)
