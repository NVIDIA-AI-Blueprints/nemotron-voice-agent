# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NVIDIA LLM service that prefills the user's words while they are still speaking.

Drop-in replacements for ``NvidiaLLMService`` and ``LLMUserAggregator`` when
the LLM is a vLLM server exposing the ``/v1/streaming-session`` websocket
(see ``examples.shared.vllm_streaming_input.server``). The user aggregator forwards every
ASR update to the LLM, which appends it to a vLLM StreamingInput session, so
the prompt is already in the KV cache when the user turn ends. Everything
after that (text, reasoning, tool calls, usage metrics, interruptions) is the
standard ``NvidiaLLMService`` path, fed from the websocket instead of HTTP.

Session events, shaped after vLLM's ``/v1/realtime`` API with text input::

    client -> server  session.update            {request}
                      input_text_buffer.append  {text}
                      input_text_buffer.clear
                      input_text_buffer.commit  {response_id}
                      response.cancel           {response_id}
    server -> client  session.created
                      response.delta            {response_id, chunk}
                      response.done             {response_id}
                      error                     {error, code}
"""

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from urllib.parse import urlparse, urlunparse
from uuid import uuid4

from loguru import logger
from openai import NotGiven as OpenAINotGiven
from openai.types.chat import ChatCompletionChunk
from pipecat.frames.frames import (
    CancelFrame,
    DataFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMUserAggregator
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.nvidia.llm import NvidiaLLMService
from pipecat.utils.string import TextPartForConcatenation, concatenate_aggregated_text
from pipecat.utils.types import NotGiven, assert_given
from websockets.asyncio.client import ClientConnection, connect

STREAMING_SESSION_PATH = "/streaming-session"


@dataclass
class StreamingLLMInputFrame(DataFrame):
    """The user turn as heard so far, sent to the LLM before the turn ends.

    Parameters:
        context: The conversation the user turn will be added to.
        text: The user turn text so far.
    """

    context: LLMContext
    text: str


class StreamingLLMUserAggregator(LLMUserAggregator):
    """``LLMUserAggregator`` that also streams the live user turn to the LLM."""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Forward each transcript as a ``StreamingLLMInputFrame``, then aggregate it."""
        if (
            isinstance(frame, (InterimTranscriptionFrame, TranscriptionFrame))
            and frame.text.strip()
            and not self._user_is_muted
        ):
            part = TextPartForConcatenation(frame.text, includes_inter_part_spaces=frame.includes_inter_frame_spaces)
            text = concatenate_aggregated_text([*self._aggregation, part])
            await self.push_frame(StreamingLLMInputFrame(context=self._context, text=text))
        await super().process_frame(frame, direction)


def _http_base_url(url: str) -> str:
    """Map ``ws(s)://host/v1/streaming-session`` to the server's ``http(s)://host/v1``."""
    parsed = urlparse(url)
    scheme = {"ws": "http", "wss": "https"}.get(parsed.scheme, parsed.scheme)
    return urlunparse(parsed._replace(scheme=scheme, path=parsed.path.removesuffix(STREAMING_SESSION_PATH)))


class NvidiaStreamingLLMService(NvidiaLLMService):
    """``NvidiaLLMService`` that streams the user turn into a vLLM StreamingInput session.

    Pair it with ``StreamingLLMUserAggregator``. ``base_url`` is the
    session websocket. The same server's chat completions route, derived from
    it, serves out-of-band requests such as ``run_inference``.
    """

    def __init__(self, *, base_url: str, **kwargs):
        """Initialize the service.

        Args:
            base_url: The ``ws://`` or ``wss://`` URL of the streaming session.
            **kwargs: Passed to ``NvidiaLLMService``.
        """
        super().__init__(base_url=_http_base_url(base_url), **kwargs)
        self._session_url = base_url
        self._websocket: ClientConnection | None = None
        self._session_request: dict | None = None
        self._session_text = ""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Prefill the session from ``StreamingLLMInputFrame`` updates."""
        await super().process_frame(frame, direction)
        if not isinstance(frame, StreamingLLMInputFrame):
            return
        try:
            await self._sync_session(self._session_request_for(frame.context), frame.text)
        except Exception as e:
            logger.warning(f"{self}: streaming prefill failed, so the turn is sent at commit instead: {e}")
            await self._disconnect()

    async def get_chat_completions(self, context: LLMContext) -> AsyncIterator[ChatCompletionChunk]:
        """Commit the user turn on the session and stream back its chat completion chunks."""
        request = self._session_request_for(context)
        text = ""
        last = request["messages"][-1] if request["messages"] else {}
        if last.get("role") == "user" and isinstance(last.get("content"), str):
            request["messages"], text = request["messages"][:-1], last["content"]
        response_id = str(uuid4())
        try:
            await self._sync_session(request, text)
            await self._send("input_text_buffer.commit", response_id=response_id)
        except Exception:
            await self._disconnect()
            raise
        self._session_text = ""
        return self._handle_reasoning_content(self._response_chunks(response_id))

    async def stop(self, frame: EndFrame):
        """Close the session and stop the service."""
        await self._disconnect()
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        """Close the session and cancel the service."""
        await self._disconnect()
        await super().cancel(frame)

    def _session_request_for(self, context: LLMContext) -> dict:
        """Build the chat completion request ``NvidiaLLMService`` would send for ``context``."""
        params = self.build_chat_completion_params(
            self.get_llm_adapter().get_llm_invocation_params(
                context,
                system_instruction=assert_given(self._settings.system_instruction),
                convert_developer_to_user=not self.supports_developer_role,
            )
        )
        params.update(params.pop("extra_body", None) or {})
        return {key: value for key, value in params.items() if not isinstance(value, (NotGiven, OpenAINotGiven))}

    async def _sync_session(self, request: dict, text: str) -> None:
        """Bring the server's conversation and user text buffer up to date."""
        if self._websocket is None:
            await self._connect()
        if request != self._session_request:
            await self._send("session.update", request=request)
            self._session_request = request
        if not text.startswith(self._session_text):
            await self._send("input_text_buffer.clear")
            self._session_text = ""
        if text != self._session_text:
            await self._send("input_text_buffer.append", text=text[len(self._session_text) :])
            self._session_text = text

    async def _connect(self) -> None:
        """Open the session websocket and wait for ``session.created``."""
        self._websocket = await connect(self._session_url, max_size=None)
        await self._websocket.recv()
        self._session_request = None
        self._session_text = ""

    async def _response_chunks(self, response_id: str) -> AsyncIterator[ChatCompletionChunk]:
        done = False
        try:
            while not done:
                event = json.loads(await self._websocket.recv())
                if event["type"] == "error":
                    raise RuntimeError(f"Streaming session error: {event.get('error')}")
                if event.get("response_id") != response_id:
                    continue
                if event["type"] == "response.delta":
                    yield ChatCompletionChunk.model_validate(event["chunk"])
                done = event["type"] == "response.done"
        finally:
            if not done and self._websocket is not None:
                try:
                    await self._send("response.cancel", response_id=response_id)
                except Exception:
                    await self._disconnect()

    async def _send(self, event_type: str, **data) -> None:
        await self._websocket.send(json.dumps({"type": event_type, **data}))

    async def _disconnect(self) -> None:
        websocket, self._websocket = self._websocket, None
        if websocket is not None:
            await websocket.close()
