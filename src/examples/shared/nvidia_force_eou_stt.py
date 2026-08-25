# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""NvidiaSTTService subclass that can force end-of-utterance on the gRPC stream."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterable
from concurrent.futures import CancelledError as FuturesCancelledError
from dataclasses import dataclass
from typing import cast

from loguru import logger
from pipecat.frames.frames import Frame, StartFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.nvidia.stt import AudioChunkIterator, NvidiaSTTService
from pipecat.services.stt_service import STTService

from examples.shared.stt_finalize_frame import STTFinalizeFrame

try:
    import grpc
    import riva.client.proto.riva_asr_pb2 as rasr
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    raise ImportError(f"Missing module: {e}") from e

# 80 ms of silence matches Nemotron ASR's endpointing frame size.
FORCE_EOU_SILENCE_SECS = 0.08
# NVIDIA recommends a high stop_history when the client drives EOU.
NEMOTRON_FORCE_EOU_STOP_HISTORY_MS = 4500
DEFAULT_STOP_HISTORY_MS = 400


def stop_history_for_asr_model(asr_model: str | None = "") -> int:
    """Return ASR ``stop_history`` (ms) for the active model.

    Nemotron ASR Streaming honors ``force_eou``, so server-side silence
    endpointing is raised to 4500 ms and Smart Turn becomes the EOU driver.
    Parakeet ignores ``force_eou`` and keeps the 400 ms default.
    """
    model = (asr_model or "").lower()
    if "parakeet" in model:
        return DEFAULT_STOP_HISTORY_MS
    return NEMOTRON_FORCE_EOU_STOP_HISTORY_MS


def build_nvidia_stt_service(*, asr_kwargs: dict, asr_model: str = "") -> NvidiaForceEouSTTService:
    """Construct the cascaded-pipeline STT service with force-EOU support."""
    return NvidiaForceEouSTTService(
        **asr_kwargs,
        stop_history=stop_history_for_asr_model(asr_model),
    )


@dataclass(frozen=True)
class AudioChunkWithRuntimeConfig:
    """Audio bytes plus optional per-chunk Riva ``runtime_config``."""

    audio: bytes
    runtime_config: dict[str, str]


def force_eou_streaming_request_generator(
    audio_chunks: Iterable[bytes | AudioChunkWithRuntimeConfig],
    streaming_config: rasr.StreamingRecognitionConfig,
):
    """Yield ``StreamingRecognizeRequest`` messages, attaching ``runtime_config`` when present."""
    yield rasr.StreamingRecognizeRequest(streaming_config=streaming_config)
    for chunk in audio_chunks:
        if isinstance(chunk, AudioChunkWithRuntimeConfig):
            yield rasr.StreamingRecognizeRequest(
                audio_content=chunk.audio,
                runtime_config=chunk.runtime_config,
            )
        else:
            yield rasr.StreamingRecognizeRequest(audio_content=chunk)


class ForceEouAudioChunkIterator(AudioChunkIterator):
    """Audio iterator that can tag a chunk with per-request ``runtime_config``."""

    async def put(self, audio: bytes, runtime_config: dict[str, str] | None = None) -> None:
        """Enqueue audio, optionally with a one-shot runtime config map."""
        if self._closed:
            return
        if runtime_config:
            await self._queue.put(AudioChunkWithRuntimeConfig(audio=audio, runtime_config=runtime_config))
        else:
            await self._queue.put(audio)

    def __next__(self) -> bytes | AudioChunkWithRuntimeConfig:
        """Get the next audio chunk or tagged chunk for the active stream."""
        if self._closed:
            raise StopIteration

        try:
            future = asyncio.run_coroutine_threadsafe(self._queue.get(), self._event_loop)
            item = future.result()
        except FuturesCancelledError:
            raise StopIteration

        if item is self._QUEUE_SENTINEL:
            self._closed = True
            raise StopIteration

        return cast(bytes | AudioChunkWithRuntimeConfig, item)


class NvidiaForceEouSTTService(NvidiaSTTService):
    """NvidiaSTTService that can flush the current utterance via ``force_eou``.

    Nemotron ASR Streaming honors a per-chunk ``runtime_config`` flag
    ``force_eou=true``. On :class:`STTFinalizeFrame` (or a direct
    :meth:`request_force_eou` call) this service injects a short silence
    chunk with that flag so the server emits a final transcript without
    closing the stream.
    """

    async def start(self, frame: StartFrame):
        """Start streaming with an iterator that can carry ``runtime_config``."""
        await STTService.start(self, frame)
        self._initialize_client()
        self._config = self._create_recognition_config()
        self._audio_iterator = ForceEouAudioChunkIterator(self.get_event_loop())

        if not self._thread_task:
            self._thread_task = self.create_task(self._thread_task_handler())

        self._create_keepalive_task()

        logger.debug(f"Initialized NvidiaForceEouSTTService with model: {self._settings.model}")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Handle upstream finalize requests, then defer to the parent STT path."""
        if isinstance(frame, STTFinalizeFrame):
            await self.request_force_eou()
            return
        await super().process_frame(frame, direction)

    async def request_force_eou(self) -> None:
        """Send ``force_eou`` on a short silence chunk so ASR finalizes immediately."""
        iterator = self._audio_iterator
        if not isinstance(iterator, ForceEouAudioChunkIterator) or iterator.closed:
            logger.debug(f"{self} force_eou skipped: no active stream")
            return

        sample_rate = self.sample_rate or 16000
        num_samples = max(1, int(sample_rate * FORCE_EOU_SILENCE_SECS))
        silence = b"\x00" * (num_samples * 2)
        await iterator.put(silence, runtime_config={"force_eou": "true"})
        self._last_audio_time = time.monotonic()
        logger.debug(f"{self} sent force_eou")

    def _response_handler(self, iterator: AudioChunkIterator):
        drop_reason = None
        try:
            asr_service = self._asr_service
            assert asr_service is not None, "ASR service not initialized"
            responses = asr_service.stub.StreamingRecognize(
                force_eou_streaming_request_generator(iterator, self._config),
                metadata=asr_service.auth.get_auth_metadata(),
            )
            for response in responses:
                if not response.results:
                    continue
                asyncio.run_coroutine_threadsafe(
                    self._handle_response(response), self.get_event_loop()
                )
            drop_reason = "server closed the gRPC stream"
        except grpc.RpcError as e:
            status = e.code().name if hasattr(e, "code") else "UNKNOWN"
            details = e.details() if hasattr(e, "details") else str(e)
            drop_reason = f"gRPC {status}: {details}"
        except Exception as e:
            drop_reason = str(e)
            logger.error(f"{self} unexpected streaming error: {e}")

        if drop_reason:
            asyncio.run_coroutine_threadsafe(
                self._handle_stream_drop(iterator, drop_reason),
                self.get_event_loop(),
            )
