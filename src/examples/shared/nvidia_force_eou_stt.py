# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""NvidiaSTTService subclass that can force end-of-utterance on the gRPC stream."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Iterable

from loguru import logger
from pipecat.frames.frames import Frame, VADUserStoppedSpeakingFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.nvidia.stt import AudioChunkIterator, NvidiaSTTService

try:
    import riva.client.proto.riva_asr_pb2 as rasr
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    raise ImportError(f"Missing module: {e}") from e

# 80 ms of silence matches Nemotron ASR's endpointing frame size.
FORCE_EOU_SILENCE_SECS = 0.08


class NvidiaForceEouSTTService(NvidiaSTTService):
    """NvidiaSTTService that flushes the current utterance via ``force_eou``.

    Local VAD requests finalization at each speech stop. The resulting
    ``TranscriptionFrame(finalized=True)`` informs Pipecat's turn stop strategy
    that all audio submitted for that speech segment has been transcribed;
    Smart Turn remains solely responsible for closing the semantic user turn.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the silence-chunk force-EOU queue."""
        super().__init__(*args, **kwargs)
        self._force_eou_silences: deque[bytes] = deque()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Use the parent STT path, then finalize submitted audio on VAD stop."""
        await super().process_frame(frame, direction)
        if isinstance(frame, VADUserStoppedSpeakingFrame):
            await self.request_force_eou()

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
            self._force_eou_silences.clear()
