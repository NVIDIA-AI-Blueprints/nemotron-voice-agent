# SPDX-License-Identifier: Apache-2.0

"""Whole-turn Transcribe ASR with an optional official transcription template."""

import asyncio
import base64
from collections.abc import AsyncGenerator

import httpx
from loguru import logger
from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame
from pipecat.services.settings import STTSettings
from pipecat.services.stt_latency import OPENAI_TTFS_P99
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.utils.time import time_now_iso8601


class TranscribeChatSTTService(SegmentedSTTService):
    """Send Pipecat's WAV segments and an official template to chat completions."""

    def __init__(self, *, base_url: str, model: str, transcription_prompt: str, language: str | None = None, **kwargs):
        """Configure the transcription client for this session."""
        super().__init__(
            settings=STTSettings(model=model, language=language),
            trailing_silence_secs=0,
            ttfs_p99_latency=OPENAI_TTFS_P99,
            **kwargs,
        )
        self._endpoint = f"{base_url.rstrip('/')}/chat/completions"
        self._transcription_prompt = transcription_prompt
        self._client = httpx.AsyncClient(timeout=30)

    def can_generate_metrics(self) -> bool:
        """Report processing latency using the standard STT metrics."""
        return True

    async def cleanup(self):
        """Close this session's HTTP client."""
        try:
            await super().cleanup()
        finally:
            await self._client.aclose()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Transcribe a WAV segment and retry temporary server errors."""
        request = {
            "model": self._settings.model,
            **({"language": self._settings.language} if self._settings.language else {}),
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {"data": base64.b64encode(audio).decode(), "format": "wav"},
                        },
                        {"type": "text", "text": self._transcription_prompt},
                    ],
                }
            ],
        }
        try:
            await self.start_processing_metrics()
            for attempt in range(3):
                response = await self._client.post(self._endpoint, json=request)
                if response.status_code in {429, 503} and attempt < 2:
                    logger.warning(f"Transcribe ASR returned {response.status_code}; retrying segment")
                    await asyncio.sleep(0.1 * 2**attempt)
                    continue
                response.raise_for_status()
                break
            result = response.json()
            text = (result["choices"][0]["message"]["content"] or "").strip()
            if text:
                logger.debug(f"Transcription: [{text}]")
                yield TranscriptionFrame(text, self._user_id, time_now_iso8601(), result=result, finalized=True)
        except Exception as error:
            logger.error(f"Transcribe ASR failed: {error!r}")
            yield ErrorFrame(error=f"Transcribe ASR failed: {error!r}")
        finally:
            await self.stop_processing_metrics()
