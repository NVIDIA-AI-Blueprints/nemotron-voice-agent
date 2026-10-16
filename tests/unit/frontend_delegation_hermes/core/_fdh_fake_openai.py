# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: D100, D101, D102, D103, D107

"""A tiny OpenAI-compatible chat server for the Hermes tests (stdlib only).

Stateless: a user turn gets one tool call to the first offered tool (its first
parameter set to ``"1"``); a turn ending in a tool result gets a final answer that
echoes the tool output and the parameter names of the tool schema it was offered, so
tests can prove which schema each session saw. Streaming requests get a single-chunk
SSE stream.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


def _completion(model: str, message: dict[str, Any], finish: str) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-{time.time_ns()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    }


def reply_for(body: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """The assistant message and finish reason for one request."""
    messages = body.get("messages") or []
    tools = body.get("tools") or []
    last = messages[-1] if messages else {}
    first = (tools[0].get("function") or {}) if tools else {}
    params = sorted(((first.get("parameters") or {}).get("properties") or {}).keys())
    if last.get("role") == "tool":
        return {"role": "assistant", "content": f"done: {last.get('content')} | schema: {params}"}, "stop"
    if not tools or "no tool" in str(last.get("content")):
        return {"role": "assistant", "content": "plain answer"}, "stop"
    args = {params[0]: "1"} if params else {}
    call = {
        "id": f"call_{time.time_ns()}",
        "type": "function",
        "function": {"name": first.get("name"), "arguments": json.dumps(args)},
    }
    return {"role": "assistant", "content": None, "tool_calls": [call]}, "tool_calls"


class FakeOpenAI:
    """Serve on 127.0.0.1:<random port> in a daemon thread; records request bodies."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_: Any) -> None:
                return

            def _json(self, data: dict[str, Any], status: int = 200) -> None:
                payload = json.dumps(data).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self) -> None:  # noqa: N802
                if self.path.rstrip("/").endswith("/models"):
                    self._json({"object": "list", "data": [{"id": "fake-model", "object": "model"}]})
                else:
                    self._json({"error": "not found"}, 404)

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                outer.requests.append(body)
                if not self.path.rstrip("/").endswith("/chat/completions"):
                    self._json({"error": "not found"}, 404)
                    return
                message, finish = reply_for(body)
                model = str(body.get("model") or "fake-model")
                if not body.get("stream"):
                    self._json(_completion(model, message, finish))
                    return
                delta: dict[str, Any] = {"role": "assistant"}
                if message.get("content"):
                    delta["content"] = message["content"]
                if message.get("tool_calls"):
                    delta["tool_calls"] = [{"index": 0, **call} for call in message["tool_calls"]]
                chunks = [
                    {
                        "id": "c",
                        "object": "chat.completion.chunk",
                        "created": 0,
                        "model": model,
                        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                    },
                    {
                        "id": "c",
                        "object": "chat.completion.chunk",
                        "created": 0,
                        "model": model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
                    },
                ]
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for chunk in chunks:
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        """``http://127.0.0.1:<port>/v1``."""
        return f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def start(self) -> FakeOpenAI:
        """Start serving."""
        self.thread.start()
        return self

    def stop(self) -> None:
        """Stop serving."""
        self.server.shutdown()
        self.server.server_close()
