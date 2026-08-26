# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""NvidiaSTTService subclass that can force end-of-utterance on the gRPC stream."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Iterable

from loguru import logger
from pipecat.frames.frames import Frame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.nvidia.stt import AudioChunkIterator, NvidiaSTTService

from examples.shared.stt_finalize_frame import STTFinalizeFrame
from utils import parse_env_bool

try:
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

    Smart Turn sends ``force_eou`` on COMPLETE, so Nemotron silence
    endpointing is raised to 4500 ms. Pure Silero VAD turn detection
    (``USE_SILERO_VAD_TURN_DETECTION=true``, including
    ``generic-assistant/server-perf``) never sends ``force_eou``, so it
    keeps the 400 ms window. Cloud Parakeet CTC/RNNT ignore ``force_eou``
    and also keep 400 ms. The local Nemotron NIM reports as
    ``cache-aware-parakeet-rnnt-*`` and still honors ``force_eou``.
    """
    if parse_env_bool("USE_SILERO_VAD_TURN_DETECTION", default=False):
        return DEFAULT_STOP_HISTORY_MS
    model = (asr_model or "").lower()
    if "parakeet" in model and "cache-aware" not in model:
        return DEFAULT_STOP_HISTORY_MS
    return NEMOTRON_FORCE_EOU_STOP_HISTORY_MS


def build_nvidia_stt_service(*, asr_kwargs: dict, asr_model: str = "") -> NvidiaForceEouSTTService:
    """Construct the cascaded-pipeline STT service with force-EOU support."""
    stop_history = stop_history_for_asr_model(asr_model)
    vad_turns = parse_env_bool("USE_SILERO_VAD_TURN_DETECTION", default=False)
    logger.info(f"ASR stop_history={stop_history}ms (model={asr_model or 'default'}, silero_vad_turns={vad_turns})")
    return NvidiaForceEouSTTService(
        **asr_kwargs,
        stop_history=stop_history,
    )


class NvidiaForceEouSTTService(NvidiaSTTService):
    """NvidiaSTTService that flushes the current utterance via ``force_eou``.

    Pipecat has no STT-finalize frame. Deepgram / Soniox / Speechmatics flush
    on ``VADUserStoppedSpeakingFrame``, which is too early for Smart Turn
    (VAD stop can be an incomplete pause). Smart Turn instead pushes
    :class:`STTFinalizeFrame` upstream on COMPLETE.

    ``STTService.request_finalize()`` only marks TTFB metrics; NVIDIA already
    sets ``TranscriptionFrame.finalized`` from ``is_final``. This subclass
    queues trailing silence and attaches ``runtime_config.force_eou`` to that
    chunk so already-buffered speech is sent first.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the silence-chunk force-EOU queue."""
        super().__init__(*args, **kwargs)
        self._force_eou_silences: deque[bytes] = deque()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Consume ``STTFinalizeFrame``; otherwise use the parent STT path."""
        if isinstance(frame, STTFinalizeFrame):
            await self.request_force_eou()
            return
        await super().process_frame(frame, direction)

    def _force_eou_silence(self) -> bytes:
        """Return an 80 ms PCM silence chunk for the current sample rate."""
        sample_rate = self.sample_rate or 16000
        num_samples = max(1, int(sample_rate * FORCE_EOU_SILENCE_SECS))
        return b"\x00" * (num_samples * 2)

    async def request_force_eou(self) -> None:
        """Append silence and tag that same chunk with ``force_eou`` when sent."""
        iterator = self._audio_iterator
        if iterator is None or iterator.closed:
            logger.debug(f"{self} force_eou skipped: no active stream")
            return

        silence = self._force_eou_silence()
        self._force_eou_silences.append(silence)
        await self._send_keepalive(silence)
        self._last_audio_time = time.monotonic()
        logger.debug(f"{self} queued force_eou")

    def _iter_streaming_requests(
        self,
        audio_chunks: Iterable[bytes],
        streaming_config: rasr.StreamingRecognitionConfig,
    ):
        """Yield gRPC requests, attaching ``force_eou`` only to tagged silence."""
        yield rasr.StreamingRecognizeRequest(streaming_config=streaming_config)
        for chunk in audio_chunks:
            runtime_config = {}
            if self._force_eou_silences and chunk is self._force_eou_silences[0]:
                self._force_eou_silences.popleft()
                runtime_config = {"force_eou": "true"}
                logger.info(f"{self} sending force_eou on {len(chunk)}-byte chunk")
            yield rasr.StreamingRecognizeRequest(
                audio_content=chunk,
                runtime_config=runtime_config,
            )

    def _response_handler(self, iterator: AudioChunkIterator):
        """Reuse the parent stream loop, injecting force_eou on gRPC requests."""
        asr_service = self._asr_service
        original = asr_service.streaming_response_generator

        def streaming_response_generator(audio_chunks, streaming_config):
            yield from asr_service.stub.StreamingRecognize(
                self._iter_streaming_requests(audio_chunks, streaming_config),
                metadata=asr_service.auth.get_auth_metadata(),
            )

        asr_service.streaming_response_generator = streaming_response_generator
        try:
            super()._response_handler(iterator)
        finally:
            asr_service.streaming_response_generator = original
