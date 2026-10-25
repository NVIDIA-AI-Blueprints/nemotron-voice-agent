# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: D100, D101, D102, D103, D107

"""A scripted OpenAI-compatible streaming server for the LLM backend tests (stdlib only).

Tests queue one :class:`Reply` per expected request; an empty queue answers ``"fallback"``.
Every request body is recorded. A reply can wait before its headers (``hold_s``: a request in
prefill), wait between chunks (``chunk_s``), cut the stream after its first chunk (``cut``) or
fail with a status code (``status``, ``retry_after``).
"""

from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


@dataclass
class Reply:
    content: str = ""
    reasoning: str = ""
    tool_calls: list[tuple[str, str]] = field(default_factory=list)  # (name, arguments json)
    finish: str | None = None
    hold_s: float = 0.0
    chunk_s: float = 0.0
    cut: bool = False
    status: int = 200
    retry_after: str | None = None


def tool(name: str, **args: Any) -> tuple[str, str]:
    return name, json.dumps(args)


class ScriptedOpenAI:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.replies: queue.Queue[Reply] = queue.Queue()
        self.started = threading.Event()  # set when a request arrives
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_: Any) -> None:
                return

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                outer.requests.append(body)
                outer.started.set()
                try:
                    reply = outer.replies.get_nowait()
                except queue.Empty:
                    reply = Reply(content="fallback")
                time.sleep(reply.hold_s)
                if reply.status != 200:
                    payload = json.dumps({"error": {"message": "scripted", "type": "x"}}).encode()
                    self.send_response(reply.status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    if reply.retry_after:
                        self.send_header("Retry-After", reply.retry_after)
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                try:
                    for index, chunk in enumerate(outer._chunks(reply, str(body.get("model")))):
                        if index and reply.chunk_s:
                            time.sleep(reply.chunk_s)
                        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                        self.wfile.flush()
                        if reply.cut and index == 0:
                            self.connection.shutdown(2)
                            return
                    self.wfile.write(b"data: [DONE]\n\n")
                except OSError:
                    return

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def _chunks(self, reply: Reply, model: str) -> list[dict[str, Any]]:
        def chunk(delta: dict[str, Any], finish: str | None = None, usage: bool = False) -> dict[str, Any]:
            data: dict[str, Any] = {
                "id": "c",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            if usage:
                data["usage"] = {
                    "prompt_tokens": 11,
                    "completion_tokens": 7,
                    "total_tokens": 18,
                    "completion_tokens_details": {"reasoning_tokens": 3},
                }
            return data

        chunks = [chunk({"role": "assistant"})]
        if reply.reasoning:
            chunks.append(chunk({"reasoning_content": reply.reasoning}))
        for piece in (reply.content[i : i + 8] for i in range(0, len(reply.content), 8)):
            chunks.append(chunk({"content": piece}))
        for index, (name, arguments) in enumerate(reply.tool_calls):
            call = {"index": index, "id": f"call_{index}_{time.time_ns()}", "type": "function"}
            chunks.append(chunk({"tool_calls": [{**call, "function": {"name": name, "arguments": arguments}}]}))
        finish = reply.finish or ("tool_calls" if reply.tool_calls else "stop")
        chunks.append(chunk({}, finish, usage=True))
        return chunks

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def queue(self, *replies: Reply) -> None:
        for reply in replies:
            self.replies.put(reply)

    def start(self) -> ScriptedOpenAI:
        self.thread.start()
        return self

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()
