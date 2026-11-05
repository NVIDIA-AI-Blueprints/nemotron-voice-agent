# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: D100, D101, D102

import base64
import json
import unittest
from unittest.mock import AsyncMock, patch

import httpx
from pipecat.frames.frames import ErrorFrame, TranscriptionFrame
from pipecat.services.stt_service import SegmentedSTTService

from examples.shared.transcribe_chat_stt import TranscribeChatSTTService

PROMPT = "Produce a verbatim transcript of the audio."


class TranscribeChatTests(unittest.IsolatedAsyncioTestCase):
    def make_service(self, handler, language="en"):
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        with patch("examples.shared.transcribe_chat_stt.httpx.AsyncClient", return_value=client):
            service = TranscribeChatSTTService(
                base_url="http://speech.invalid/v1/", model="transcribe", transcription_prompt=PROMPT, language=language
            )
        service.start_processing_metrics = AsyncMock()
        service.stop_processing_metrics = AsyncMock()
        self.addAsyncCleanup(client.aclose)
        return service

    async def test_prompt_audio_and_language_reach_the_chat_endpoint(self):
        for language in ("en", "es", None):
            with self.subTest(language=language):
                requests = []

                def respond(request, requests=requests):
                    requests.append(request)
                    return httpx.Response(200, json={"choices": [{"message": {"content": " No. "}}]})

                service = self.make_service(respond, language)
                frames = [frame async for frame in service.run_stt(b"wav")]
                self.assertIsInstance(frames[0], TranscriptionFrame)
                self.assertEqual(frames[0].text, "No.")
                self.assertTrue(frames[0].finalized)
                self.assertEqual(str(requests[0].url), "http://speech.invalid/v1/chat/completions")
                payload = json.loads(requests[0].content)
                message = payload.pop("messages")[0]
                expected = {"model": "transcribe", "stream": False}
                if language:
                    expected["language"] = language
                self.assertEqual(payload, expected)
                self.assertEqual(message["role"], "user")
                audio, prompt = message["content"]
                self.assertEqual(
                    audio,
                    {
                        "type": "input_audio",
                        "input_audio": {"data": base64.b64encode(b"wav").decode(), "format": "wav"},
                    },
                )
                self.assertEqual(prompt, {"type": "text", "text": PROMPT})

    async def test_empty_transcripts_are_dropped(self):
        for content in ("", " ", None):
            with self.subTest(content=content):
                service = self.make_service(
                    lambda request, content=content: httpx.Response(
                        200, json={"choices": [{"message": {"content": content}}]}
                    )
                )
                self.assertEqual([frame async for frame in service.run_stt(b"wav")], [])

    async def test_transient_errors_retry_but_permanent_errors_do_not(self):
        for status, succeeds in ((429, True), (503, False), (400, False)):
            with self.subTest(status=status):
                requests = []

                def respond(request, requests=requests, succeeds=succeeds, status=status):
                    requests.append(request)
                    if succeeds and len(requests) == 3:
                        return httpx.Response(200, json={"choices": [{"message": {"content": "Hello."}}]})
                    return httpx.Response(status)

                service = self.make_service(respond)
                with patch("examples.shared.transcribe_chat_stt.asyncio.sleep", new_callable=AsyncMock) as sleep:
                    frames = [frame async for frame in service.run_stt(b"wav")]
                self.assertEqual(len(requests), 1 if status == 400 else 3)
                self.assertEqual(sleep.await_count, 0 if status == 400 else 2)
                self.assertEqual(len({request.content for request in requests}), 1)
                self.assertIsInstance(frames[0], TranscriptionFrame if succeeds else ErrorFrame)
                service.stop_processing_metrics.assert_awaited_once()

    async def test_empty_timeout_message_still_identifies_the_cause(self):
        def timeout(request):
            raise httpx.ReadTimeout("", request=request)

        service = self.make_service(timeout)
        with patch("examples.shared.transcribe_chat_stt.logger.error") as log:
            frames = [frame async for frame in service.run_stt(b"wav")]
        self.assertEqual(frames[0].error, "Transcribe ASR failed: ReadTimeout('')")
        log.assert_called_once_with(frames[0].error)
        service.stop_processing_metrics.assert_awaited_once()

    async def test_cleanup_closes_the_session_client(self):
        service = self.make_service(lambda request: httpx.Response(200))
        with patch.object(SegmentedSTTService, "cleanup", new_callable=AsyncMock):
            await service.cleanup()
        self.assertTrue(service._client.is_closed)
