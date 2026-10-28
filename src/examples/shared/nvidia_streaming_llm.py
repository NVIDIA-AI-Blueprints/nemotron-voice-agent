# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NVIDIA LLM service that prefills the user's words while they are still speaking.

Drop-in replacements for ``NvidiaLLMService`` and ``LLMUserAggregator`` when
the LLM accepts the live user turn over a websocket. The user aggregator
forwards every ASR update to the LLM, which appends it to a vLLM
StreamingInput session, so the prompt is already in the KV cache when the user
turn ends. Everything after that (text, reasoning, tool calls, usage metrics,
interruptions) is the standard ``NvidiaLLMService`` path, fed from the
websocket instead of HTTP.

The websocket URL path selects one of two servers.

``/v1/streaming-session``: the vLLM wrapper in
``examples.shared.vllm_streaming_input.server``. Each session carries the full
chat completion request, so sampling settings and ``chat_template_kwargs``
apply::

    client -> server  session.update            {request}
                      input_text_buffer.append  {text}
                      input_text_buffer.clear
                      input_text_buffer.commit  {response_id}
                      response.cancel           {response_id}
    server -> client  session.created
                      response.delta            {response_id, chunk}
                      response.done             {response_id}
                      error                     {error, code}

``/v1/realtime``: NVIDIA Dynamo's OpenAI Realtime text sessions with the
``input_text.*`` prefill extension. The conversation is mirrored as realtime
items. The model, tools, tool choice, and output token limit map to session
fields, and ``chat_template_kwargs.enable_thinking`` maps to
``reasoning.effort`` (``medium`` for on, ``minimal`` for off). Realtime
sessions have no sampling fields, so the worker's defaults apply and other
request options are dropped::

    client -> server  session.update            {session}
                      conversation.item.create  {item}
                      conversation.item.delete  {item_id}
                      input_text.append         {text}
                      input_text.clear
                      input_text.commit
                      response.create
                      response.cancel           {response_id}
    server -> client  session.created
                      conversation.item.done    {item}
                      response.created          {response}
                      response.output_text.delta
                      response.output_item.added
                      response.function_call_arguments.delta
                      response.done             {response}
                      error                     {error}
"""

import asyncio
import json
from collections.abc import AsyncIterator
from copy import deepcopy
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
REALTIME_PATH = "/realtime"
# Errors a late ``response.cancel`` can draw after its response already ended.
STALE_CANCEL_ERRORS = {"no_active_response", "response_not_found"}
CANCEL_DRAIN_TIMEOUT_SECS = 5.0
REALTIME_REQUEST_KEYS = {
    "model",
    "messages",
    "stream",
    "stream_options",
    "tools",
    "tool_choice",
    "max_tokens",
    "max_completion_tokens",
}


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
            # Downstream frames are queued. Snapshot history before final
            # aggregation appends this user turn to the shared context.
            context = LLMContext(
                deepcopy(self._context.get_messages()),
                tools=self._context.tools,
                tool_choice=self._context.tool_choice,
            )
            await self.push_frame(StreamingLLMInputFrame(context=context, text=text))
        await super().process_frame(frame, direction)


def is_realtime_url(url: str) -> bool:
    """Return True when ``url`` is a Dynamo ``/v1/realtime`` websocket."""
    return urlparse(url).path.rstrip("/").endswith(REALTIME_PATH)


def _http_base_url(url: str) -> str:
    """Map ``ws(s)://host/v1/streaming-session`` or ``/v1/realtime`` to the server's ``http(s)://host/v1``."""
    parsed = urlparse(url)
    scheme = {"ws": "http", "wss": "https"}.get(parsed.scheme, parsed.scheme)
    path = parsed.path.rstrip("/").removesuffix(STREAMING_SESSION_PATH).removesuffix(REALTIME_PATH)
    return urlunparse(parsed._replace(scheme=scheme, path=path))


def _merge_tool_call_turns(messages: list[dict]) -> list[dict]:
    """Join the spoken text and tool calls of one answer, which the context keeps as two assistant messages.

    The server prefills the follow-up turn with the answer rendered as one
    message, so only the joined history extends what it holds.
    """
    merged: list[dict] = []
    for message in messages:
        previous = merged[-1] if merged else {}
        if (
            previous.get("role") == message.get("role") == "assistant"
            and isinstance(previous.get("content"), str)
            and not previous.get("tool_calls")
            and message.get("tool_calls")
            and not message.get("content")
        ):
            merged[-1] = {**previous, "tool_calls": message["tool_calls"]}
        else:
            merged.append(message)
    return merged


def _message_text(message: dict) -> str:
    """Return the text of a chat message whose content is a string or a list of text parts."""
    content = message.get("content")
    if content is None or isinstance(content, str):
        return content or ""
    parts = [part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text"]
    if len(parts) != len(content):
        raise ValueError(f"{message.get('role')} message content must be text only")
    return "".join(parts)


def _realtime_items(message: dict) -> list[dict]:
    """Map one chat completion message to the realtime conversation items that render it."""
    role = message.get("role")
    if role == "tool":
        return [
            {
                "type": "function_call_output",
                "status": "completed",
                "call_id": message["tool_call_id"],
                "output": _message_text(message),
            }
        ]
    if role not in ("system", "user", "assistant"):
        raise ValueError(f"unsupported message role {role!r}")
    text = _message_text(message)
    items = []
    if text or role != "assistant" or message.get("tool_calls"):
        content_type = "output_text" if role == "assistant" else "input_text"
        items.append({"type": "message", "role": role, "content": [{"type": content_type, "text": text}]})
    for call in message.get("tool_calls") or []:
        function = call["function"]
        items.append(
            {
                "type": "function_call",
                "status": "completed",
                "call_id": call["id"],
                "name": function["name"],
                "arguments": function.get("arguments") or "",
            }
        )
    return items


def _item_key(item: dict) -> tuple:
    """Return what renders a realtime item, ignoring ids and status."""
    if item["type"] == "message":
        return ("message", item["role"], "".join(part.get("text", "") for part in item.get("content") or []))
    if item["type"] == "function_call":
        return ("function_call", item["call_id"], item["name"], item.get("arguments", ""))
    return (item["type"], item.get("call_id"), item.get("output"))


def _realtime_tools(tools: list[dict]) -> list[dict]:
    """Map chat completion function tools to realtime session tools."""
    return [
        {
            "type": "function",
            "name": tool["function"]["name"],
            "description": tool["function"].get("description", ""),
            "parameters": tool["function"].get("parameters") or {"type": "object", "properties": {}},
        }
        for tool in tools
        if tool.get("type") == "function"
    ]


def _realtime_tool_choice(tool_choice) -> str | dict:
    """Map a chat completion ``tool_choice`` to its realtime form."""
    if isinstance(tool_choice, dict):
        return {"type": "function", "name": tool_choice["function"]["name"]}
    return tool_choice


def _chunk(
    response_id: str, model: str, *, delta: dict | None = None, usage: dict | None = None
) -> ChatCompletionChunk:
    """Build the chat completion chunk ``NvidiaLLMService`` consumes for one realtime event."""
    return ChatCompletionChunk.model_validate(
        {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": 0,
            "model": model,
            "choices": [] if delta is None else [{"index": 0, "delta": delta, "finish_reason": None}],
            "usage": usage,
        }
    )


class _StreamingInputSession:
    """The vLLM ``/v1/streaming-session`` wire protocol."""

    def __init__(self, url: str):
        self._url = url
        self._websocket: ClientConnection | None = None
        self._request: dict | None = None
        self._text = ""

    async def sync(self, request: dict, text: str) -> None:
        """Bring the server's conversation and user text buffer up to date."""
        if self._websocket is None:
            await self._connect()
        if request != self._request:
            await self._send("session.update", request=request)
            self._request = request
        if not text.startswith(self._text):
            await self._send("input_text_buffer.clear")
            self._text = ""
        if text != self._text:
            await self._send("input_text_buffer.append", text=text[len(self._text) :])
            self._text = text

    async def respond(self) -> AsyncIterator[ChatCompletionChunk]:
        """Commit the user text buffer and return the answer's chunks."""
        response_id = str(uuid4())
        await self._send("input_text_buffer.commit", response_id=response_id)
        self._text = ""
        return self._chunks(response_id)

    async def close(self) -> None:
        websocket, self._websocket = self._websocket, None
        if websocket is not None:
            await websocket.close()

    async def _connect(self) -> None:
        self._websocket = await connect(self._url, max_size=None)
        await self._websocket.recv()
        self._request = None
        self._text = ""

    async def _chunks(self, response_id: str) -> AsyncIterator[ChatCompletionChunk]:
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
                    await self.close()

    async def _send(self, event_type: str, **data) -> None:
        await self._websocket.send(json.dumps({"type": event_type, **data}))


class _RealtimeSession:
    """The Dynamo ``/v1/realtime`` text wire protocol.

    The server adds each committed user turn and each answer to its
    conversation. The context records answers as they were spoken, so before
    each update the session keeps the items both agree on, deletes the rest,
    and creates the context's remaining messages as items.
    """

    def __init__(self, url: str, model: str):
        self._url = url
        self._model = model
        self._websocket: ClientConnection | None = None
        self._session: dict | None = None
        self._items: list[tuple[tuple, str]] = []
        self._text = ""
        self._open_response: str | None = None
        self._cancel_sent = False
        self._dropped_keys: set[str] = set()

    async def sync(self, request: dict, text: str) -> None:
        """Bring the server's session, conversation, and user text buffer up to date."""
        if self._websocket is None:
            await self._connect()
        if self._open_response is not None:
            await self._finish_open_response()
        session = self._session_for(request)
        items = [item for message in request["messages"] for item in _realtime_items(message)]
        keys = [_item_key(item) for item in items]
        kept = 0
        while kept < min(len(keys), len(self._items)) and keys[kept] == self._items[kept][0]:
            kept += 1
        changes = session != self._session or kept < len(self._items) or kept < len(items)
        if self._text and (changes or not text.startswith(self._text)):
            await self._send({"type": "input_text.clear"})
            self._text = ""
        if session != self._session:
            await self._send({"type": "session.update", "session": session})
            self._session = session
        for _, item_id in reversed(self._items[kept:]):
            await self._send({"type": "conversation.item.delete", "item_id": item_id})
        del self._items[kept:]
        for item in items[kept:]:
            item_id = f"item_{uuid4().hex}"
            await self._send({"type": "conversation.item.create", "item": {"id": item_id, **item}})
            self._items.append((_item_key(item), item_id))
        if text != self._text:
            await self._send({"type": "input_text.append", "text": text[len(self._text) :]})
            self._text = text

    async def respond(self) -> AsyncIterator[ChatCompletionChunk]:
        """Commit the user text buffer, if any, and return the answer's chunks."""
        if self._text:
            await self._send({"type": "input_text.commit"})
            self._text = ""
        await self._send({"type": "response.create"})
        self._open_response = ""
        self._cancel_sent = False
        return self._chunks()

    async def close(self) -> None:
        websocket, self._websocket = self._websocket, None
        if websocket is not None:
            await websocket.close()

    def _session_for(self, request: dict) -> dict:
        if dropped := set(request) - REALTIME_REQUEST_KEYS - self._dropped_keys:
            logger.warning(f"Dynamo realtime sessions ignore these request options: {sorted(dropped)}")
            self._dropped_keys |= dropped
        session = {
            "type": "realtime",
            "model": self._model,
            "output_modalities": ["text"],
            "tools": _realtime_tools(request.get("tools") or []),
            "tool_choice": _realtime_tool_choice(request.get("tool_choice") or "auto"),
        }
        max_tokens = request.get("max_completion_tokens", request.get("max_tokens"))
        if max_tokens is not None:
            session["max_output_tokens"] = max_tokens
        thinking = (request.get("chat_template_kwargs") or {}).get("enable_thinking")
        if thinking is not None:
            # Realtime has no "none" effort. The worker reads "minimal" as thinking off.
            session["reasoning"] = {"effort": "medium" if thinking else "minimal"}
        return session

    async def _connect(self) -> None:
        self._websocket = await connect(self._url, max_size=None)
        event = json.loads(await self._websocket.recv())
        if event.get("type") != "session.created":
            raise RuntimeError(f"Expected realtime session.created, got {event.get('type')!r}")
        self._session = None
        self._items = []
        self._text = ""
        self._open_response = None
        self._cancel_sent = False

    def _record_item(self, event: dict) -> None:
        """Track a conversation item the server added, such as a committed user turn or an answer."""
        item = event["item"]
        if all(item["id"] != item_id for _, item_id in self._items):
            self._items.append((_item_key(item), item["id"]))

    async def _finish_open_response(self) -> None:
        """Cancel an answer that was not read to the end and read its remaining events, so its items are known."""
        try:
            if not self._cancel_sent:
                await self._send_cancel(self._open_response)
            while self._open_response is not None:
                event = json.loads(await asyncio.wait_for(self._websocket.recv(), CANCEL_DRAIN_TIMEOUT_SECS))
                if event.get("type") == "conversation.item.done":
                    self._record_item(event)
                elif event.get("type") == "response.done" and self._open_response in ("", event["response"]["id"]):
                    self._open_response = None
        except Exception:
            await self.close()
            await self._connect()

    async def _chunks(self) -> AsyncIterator[ChatCompletionChunk]:
        response_id = None
        call_indexes: dict[str, int] = {}
        done = False
        try:
            while not done:
                event = json.loads(await self._websocket.recv())
                event_type = event.get("type")
                if event_type == "error":
                    if response_id is None and (event.get("error") or {}).get("code") in STALE_CANCEL_ERRORS:
                        continue
                    raise RuntimeError(f"Realtime session error: {event.get('error')}")
                if event_type == "conversation.item.done":
                    self._record_item(event)
                    continue
                if event_type == "response.created" and response_id is None:
                    response_id = self._open_response = event["response"]["id"]
                    continue
                if response_id is None or response_id != event.get("response_id", event.get("response", {}).get("id")):
                    continue
                if event_type == "response.output_text.delta" and event.get("delta"):
                    yield _chunk(response_id, self._model, delta={"content": event["delta"]})
                elif event_type == "response.output_item.added" and event["item"].get("type") == "function_call":
                    item = event["item"]
                    call_indexes[item["call_id"]] = len(call_indexes)
                    call = {
                        "index": call_indexes[item["call_id"]],
                        "id": item["call_id"],
                        "type": "function",
                        "function": {"name": item["name"], "arguments": ""},
                    }
                    yield _chunk(response_id, self._model, delta={"tool_calls": [call]})
                elif event_type == "response.function_call_arguments.delta" and event.get("delta"):
                    call = {"index": call_indexes[event["call_id"]], "function": {"arguments": event["delta"]}}
                    yield _chunk(response_id, self._model, delta={"tool_calls": [call]})
                elif event_type == "response.done":
                    done = True
                    self._open_response = None
                    response = event["response"]
                    if response.get("status") == "failed":
                        raise RuntimeError(f"Realtime response failed: {response.get('status_details')}")
                    if usage := response.get("usage"):
                        yield _chunk(
                            response_id,
                            self._model,
                            usage={
                                "prompt_tokens": usage.get("input_tokens", 0),
                                "completion_tokens": usage.get("output_tokens", 0),
                                "total_tokens": usage.get("total_tokens", 0),
                            },
                        )
        except Exception:
            await self.close()
            raise
        finally:
            if not done and self._websocket is not None:
                try:
                    await self._send_cancel(response_id)
                except Exception:
                    await self.close()

    async def _send_cancel(self, response_id: str | None) -> None:
        self._cancel_sent = True
        await self._send({"type": "response.cancel", **({"response_id": response_id} if response_id else {})})

    async def _send(self, event: dict) -> None:
        await self._websocket.send(json.dumps(event))


class NvidiaStreamingLLMService(NvidiaLLMService):
    """``NvidiaLLMService`` that streams the user turn into a StreamingInput session.

    Pair it with ``StreamingLLMUserAggregator``. ``base_url`` is the session
    websocket, either the vLLM wrapper's ``/v1/streaming-session`` or Dynamo's
    ``/v1/realtime``. The same server's chat completions route, derived from
    it, serves out-of-band requests such as ``run_inference``. A Dynamo
    realtime worker serves chat completions only with ``--realtime-chat``.
    """

    def __init__(self, *, base_url: str, **kwargs):
        """Initialize the service.

        Args:
            base_url: The ``ws://`` or ``wss://`` URL of the streaming session.
            **kwargs: Passed to ``NvidiaLLMService``.
        """
        super().__init__(base_url=_http_base_url(base_url), **kwargs)
        if is_realtime_url(base_url):
            self._session = _RealtimeSession(base_url, assert_given(self._settings.model))
        else:
            self._session = _StreamingInputSession(base_url)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Prefill the session from ``StreamingLLMInputFrame`` updates."""
        await super().process_frame(frame, direction)
        if not isinstance(frame, StreamingLLMInputFrame):
            return
        try:
            await self._session.sync(self._session_request_for(frame.context), frame.text)
        except Exception as e:
            logger.warning(f"{self}: streaming prefill failed, so the turn is sent at commit instead: {e}")
            await self._session.close()

    async def get_chat_completions(self, context: LLMContext) -> AsyncIterator[ChatCompletionChunk]:
        """Commit the user turn on the session and stream back its chat completion chunks."""
        request = self._session_request_for(context)
        text = ""
        last = request["messages"][-1] if request["messages"] else {}
        if last.get("role") == "user" and isinstance(last.get("content"), str):
            request["messages"], text = request["messages"][:-1], last["content"]
        try:
            await self._session.sync(request, text)
            chunks = await self._session.respond()
        except Exception:
            await self._session.close()
            raise
        return self._handle_reasoning_content(chunks)

    async def stop(self, frame: EndFrame):
        """Close the session and stop the service."""
        await self._session.close()
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        """Close the session and cancel the service."""
        await self._session.close()
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
        params["messages"] = _merge_tool_call_turns(list(params.get("messages") or []))
        return {key: value for key, value in params.items() if not isinstance(value, (NotGiven, OpenAINotGiven))}
