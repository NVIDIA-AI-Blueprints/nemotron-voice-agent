# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D105, D107

"""A scripted OpenAI-compatible model endpoint, so a real ``hermes acp`` runs real turns in tests at no model cost.

* :class:`Say`: an assistant message, streamed in chunks (reasoning first, as a reasoning model streams);
* :class:`Call`: tool calls, by the end of the tool's name (``"__lookup"`` finds ``mcp__w1__lookup``);
* :class:`Hold`: wait until the test releases it, then play the next step of the same reply;
* :class:`Fail`: an HTTP error status (a provider error).

Every request body is kept in ``requests``. ``HARNESS_MODEL_BIND`` and ``HARNESS_MODEL_HOST`` let an agent on another
host or in another container reach it.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

MODEL = "scripted/model"
# Where the endpoint listens, and the host its base URL names (another container reaches it by that name).
BIND = os.environ.get("HARNESS_MODEL_BIND", "127.0.0.1")
HOST = os.environ.get("HARNESS_MODEL_HOST") or BIND


@dataclass
class Say:
    text: str
    reasoning: str = "Thinking it through."
    chunks: int = 3


@dataclass
class Call:
    calls: Sequence[tuple[str, dict]]  # (the end of the tool's name, its arguments)
    text: str = ""


@dataclass
class Hold:
    event: asyncio.Event = field(default_factory=asyncio.Event)
    then: Say | Call | None = None


@dataclass
class Fail:
    status: int = 403
    message: str = "Forbidden: the key was refused"


Reply = Say | Call | Hold | Fail


def tool_names(body: dict) -> list[str]:
    """The names of the tools a model request offers."""
    return [tool.get("function", {}).get("name", "") for tool in body.get("tools") or []]


def offers(body: dict, suffix: str) -> bool:
    """Whether the request offers a tool whose name ends with ``suffix`` (``"__start_work"``: a conductor's)."""
    return any(name.endswith(suffix) for name in tool_names(body))


def after_tool(body: dict) -> bool:
    """Whether the request's newest message is a tool result (the model answers it next)."""
    messages = body.get("messages") or []
    return bool(messages) and messages[-1].get("role") == "tool"


def user_text(body: dict) -> str:
    """Every user message's text of the request, joined."""
    texts = []
    for message in body.get("messages") or []:
        if message.get("role") == "user":
            content = message.get("content")
            texts.append(content if isinstance(content, str) else " ".join(p.get("text", "") for p in content or []))
    return "\n".join(texts)


def _split(text: str, parts: int) -> list[str]:
    size = max(1, -(-len(text) // max(parts, 1)))
    return [text[index : index + size] for index in range(0, len(text), size)] or [""]


class ScriptedModel:
    """An HTTP server (on loopback by default) that plays scripted chat replies, one per model request, in order."""

    def __init__(self, replies: Sequence[Reply] = (), *, default: Reply | None = None, policy=None):
        self.replies: list[Reply] = list(replies)
        self.default = default
        self.policy = policy  # policy(body) -> Reply | None: a reply by what the request asks, before the queue
        self.requests: list[dict] = []
        self.waiting: list[Hold] = []  # the holds a request waits on now
        self.server: asyncio.base_events.Server | None = None
        self.port = 0
        self.prompt_tokens = 0

    @property
    def base_url(self) -> str:
        return f"http://{HOST}:{self.port}/v1"

    def add(self, *replies: Reply) -> None:
        self.replies.extend(replies)

    async def start(self) -> ScriptedModel:
        self.server = await asyncio.start_server(self._serve, BIND, 0)
        self.port = self.server.sockets[0].getsockname()[1]
        return self

    async def stop(self) -> None:
        for hold in list(self.waiting):  # a request still held is released: the server's close waits for it
            hold.event.set()
        if self.server is not None:
            self.server.close()
            await self.server.wait_closed()

    def system_prompts(self) -> list[str]:
        return [
            str(message.get("content"))
            for body in self.requests
            for message in body.get("messages", [])[:1]
            if message.get("role") == "system"
        ]

    # ------------------------------------------------------------------------------------------------ HTTP
    async def _serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while True:
                head = await reader.readuntil(b"\r\n\r\n")
                lines = head.decode("latin-1").split("\r\n")
                method, path, _ = lines[0].split(" ", 2)
                headers = {
                    k.strip().lower(): v.strip() for k, _, v in (line.partition(":") for line in lines[1:] if line)
                }
                body = await reader.readexactly(int(headers.get("content-length", "0") or 0))
                if method == "GET" and path.rstrip("/").endswith("/models"):
                    await self._json(writer, 200, {"object": "list", "data": [{"id": MODEL, "object": "model"}]})
                elif method == "POST" and path.endswith("/chat/completions"):
                    await self._chat(writer, json.loads(body or b"{}"))
                else:
                    await self._json(writer, 404, {"error": {"message": f"no route {path}"}})
                if headers.get("connection", "").lower() == "close":
                    break
        except (asyncio.IncompleteReadError, ConnectionError, asyncio.LimitOverrunError):
            pass
        finally:
            writer.close()

    async def _json(self, writer: asyncio.StreamWriter, status: int, payload: dict) -> None:
        data = json.dumps(payload).encode()
        writer.write(
            f"HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {len(data)}\r\n\r\n".encode()
            + data
        )
        await writer.drain()

    async def _chat(self, writer: asyncio.StreamWriter, body: dict) -> None:
        self.requests.append(body)
        reply = self.policy(body) if self.policy is not None else None
        if reply is None:
            reply = self.replies.pop(0) if self.replies else self.default
        while isinstance(reply, Hold):
            self.waiting.append(reply)
            try:
                await reply.event.wait()
            finally:
                self.waiting.remove(reply)
            reply = reply.then or (self.replies.pop(0) if self.replies else self.default)
        if reply is None:
            reply = Say("(no scripted reply left)")
        if isinstance(reply, Fail):
            await self._json(writer, reply.status, {"error": {"message": reply.message, "type": "forbidden"}})
            return
        self.prompt_tokens += 100
        tools = [tool.get("function", {}).get("name", "") for tool in body.get("tools") or []]
        if body.get("stream"):
            await self._stream(writer, reply, tools)
        else:
            await self._json(writer, 200, self._whole(reply, tools))

    def _calls(self, reply: Call, tools: list[str]) -> list[dict]:
        out = []
        for index, (suffix, arguments) in enumerate(reply.calls):
            name = next((tool for tool in tools if tool.endswith(suffix)), suffix)
            out.append(
                {
                    "index": index,
                    "id": f"call_{int(time.time() * 1000)}_{index}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                }
            )
        return out

    def _usage(self) -> dict:
        return {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}

    def _whole(self, reply: Say | Call, tools: list[str]) -> dict:
        message: dict = {"role": "assistant", "content": reply.text or None}
        finish = "stop"
        if isinstance(reply, Call):
            message["tool_calls"] = [
                {k: v for k, v in call.items() if k != "index"} for call in self._calls(reply, tools)
            ]
            finish = "tool_calls"
        return {
            "id": "chatcmpl-scripted",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": MODEL,
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": self._usage(),
        }

    async def _stream(self, writer: asyncio.StreamWriter, reply: Say | Call, tools: list[str]) -> None:
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n")
        writer.write(b"Transfer-Encoding: chunked\r\n\r\n")

        async def send(payload: dict | str) -> None:
            data = ("data: " + (payload if isinstance(payload, str) else json.dumps(payload)) + "\n\n").encode()
            writer.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
            await writer.drain()
            await asyncio.sleep(0.01)

        def chunk(delta: dict, finish: str | None = None) -> dict:
            return {
                "id": "chatcmpl-scripted",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": MODEL,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }

        await send(chunk({"role": "assistant", "content": ""}))
        reasoning = getattr(reply, "reasoning", "")
        if reasoning:
            for part in _split(reasoning, 2):
                await send(chunk({"reasoning_content": part}))
        if reply.text:
            for part in _split(reply.text, getattr(reply, "chunks", 2)):
                await send(chunk({"content": part}))
        finish = "stop"
        if isinstance(reply, Call):
            for call in self._calls(reply, tools):
                await send(chunk({"tool_calls": [call]}))
            finish = "tool_calls"
        await send(chunk({}, finish))
        await send({**chunk({}), "choices": [], "usage": self._usage()})
        await send("[DONE]")
        writer.write(b"0\r\n\r\n")
        await writer.drain()
